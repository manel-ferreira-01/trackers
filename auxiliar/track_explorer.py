"""
Interactive feature track explorer with completion quality overlay.

Usage:
    from track_explorer import build_track_explorer

    build_track_explorer(
        video_tensor,
        obs_mat,
        vf=rec['vf'],
        vp=rec['vp'],
        final_W=rec['final_W'],
        mask_f=rec['mask_f'],
        K=K,
        resize_factor=0.25,
        output_path='tracks.html',
    )
"""

import base64, io, json
import numpy as np
import torch
from pathlib import Path


def _frame_to_b64(frame_np):
    from PIL import Image
    img = Image.fromarray((frame_np * 255).clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _to_json_list(arr):
    out = []
    for row in arr:
        out.append([None if np.isnan(v) else round(float(v), 2) for v in row])
    return out


def _backproject(final_W, K, F_surv, P, H_orig=None, W_orig=None):
    while K.ndim > 2 and K.shape[0] == 1:
        K = K.squeeze(0)
    z    = final_W[2::3, :P].float()
    x_n  = final_W[0::3, :P].float() / z.clamp(min=1e-6)
    y_n  = final_W[1::3, :P].float() / z.clamp(min=1e-6)
    if K.ndim == 2:
        fx, fy, cx, cy = K[0,0].item(), K[1,1].item(), K[0,2].item(), K[1,2].item()
        xp = (x_n * fx + cx).cpu().numpy().astype("float32")
        yp = (y_n * fy + cy).cpu().numpy().astype("float32")
    else:
        K_ = K[:F_surv]
        xp = (x_n * K_[:,0,0].unsqueeze(1) + K_[:,0,2].unsqueeze(1)).cpu().numpy().astype("float32")
        yp = (y_n * K_[:,1,1].unsqueeze(1) + K_[:,1,2].unsqueeze(1)).cpu().numpy().astype("float32")
    bad = z.cpu().numpy() <= 0
    if H_orig is not None and W_orig is not None:
        bad |= (xp < 0) | (xp >= W_orig) | (yp < 0) | (yp >= H_orig)
    xp[bad] = float("nan"); yp[bad] = float("nan")
    return xp, yp


def build_track_explorer(
    video_tensor,
    obs_mat: torch.Tensor,
    vf=None,
    vp=None,
    final_W=None,
    mask_f=None,
    K=None,
    resize_factor: float = 0.25,
    output_path: str = 'tracks.html',
    display_in_notebook: bool = True,
    max_features: int = 10_000,
):
    """
    Parameters
    ----------
    video_tensor  (1,F,H,W,3)|(F,H,W,3)   values in [-1,1] or [0,1]
    obs_mat       (2F_orig, P_orig)         pixel coords, NaN where missing
    vf            (F_orig,) bool            surviving frames from run_projective_reconstruction
    vp            (P_orig,) bool            surviving points
    final_W       (3F_surv, P_surv)         completed normalised rays [x_n, y_n, 1]
    mask_f        (3F_surv, P_surv) float   1=originally observed  0=ALS-filled
    K             (3,3) or (F_surv,3,3)    intrinsics for back-projecting final_W to pixels
    """

    frames = video_tensor[0] if video_tensor.ndim == 5 else video_tensor
    frames = frames.float()
    if frames.min() < -0.5:
        frames = (frames + 1) / 2
    frames = frames.clamp(0, 1)

    F_orig_video = frames.shape[0]
    H_orig, W_orig = frames.shape[1], frames.shape[2]
    H_new = int(H_orig * resize_factor)
    W_new = int(W_orig * resize_factor)

    # ── obs_mat tells us how many frames there really are ──
    # obs_mat shape is (2*F_obs, P_obs) where F_obs may be smaller than F_orig_video
    F_obs = obs_mat.shape[0] // 2
    P_obs = obs_mat.shape[1]

    # vf: (F_obs,) bool — surviving frames out of the F_obs that have tracks
    # vp: (P_obs,) bool — surviving points
    vf_bool = vf.bool().cpu() if vf is not None else torch.ones(F_obs, dtype=torch.bool)
    vp_bool = vp.bool().cpu() if vp is not None else torch.ones(P_obs, dtype=torch.bool)

    # clamp to actual obs_mat dimensions in case masks are larger
    if vf_bool.shape[0] > F_obs:
        vf_bool = vf_bool[:F_obs]
    if vp_bool.shape[0] > P_obs:
        vp_bool = vp_bool[:P_obs]

    frame_indices_obs = vf_bool.nonzero(as_tuple=True)[0]   # indices into obs_mat frames
    F_surv = len(frame_indices_obs)
    P_surv = int(vp_bool.sum().item())
    P      = min(P_surv, max_features)

    print(f"obs_mat frames: {F_obs}  surviving: {F_surv}")
    print(f"obs_mat points: {P_obs}  surviving: {P_surv}  (showing {P})")

    # map obs frame indices → video frame indices
    # if video has more frames than obs_mat, assume 1-to-1 from the start
    frame_indices_video = frame_indices_obs.clamp(max=F_orig_video - 1)
    frames_sel   = frames[frame_indices_video]
    frames_small = torch.nn.functional.interpolate(
        frames_sel.permute(0,3,1,2), size=(H_new, W_new),
        mode='bilinear', align_corners=False,
    ).permute(0,2,3,1).cpu().numpy()

    print(f"Encoding {F_surv} frames …")
    frame_b64 = [_frame_to_b64(frames_small[i]) for i in range(F_surv)]

    # observed coords for surviving frames + points
    # vf_2x indexes into (2*F_obs,) rows of obs_mat
    vf_2x    = vf_bool.repeat_interleave(2)             # (2*F_obs,)
    obs_filt = obs_mat[vf_2x][:, vp_bool][:, :P]        # (2*F_surv, P)
    xs_obs = (obs_filt[0::2].float() * resize_factor).cpu().numpy()
    ys_obs = (obs_filt[1::2].float() * resize_factor).cpu().numpy()

    # back-projected completed coords
    has_comp = (final_W is not None) and (K is not None)
    xs_comp  = np.full((F_surv, P), np.nan, dtype=np.float32)
    ys_comp  = np.full((F_surv, P), np.nan, dtype=np.float32)
    obs_mask_np  = np.ones( (F_surv, P), dtype=np.float32)
    per_frame_resid = [0.0] * F_surv
    fill_frac       = [0.0] * F_surv

    if has_comp:
        K_t = torch.as_tensor(K, dtype=torch.float32)
        xp_all, yp_all = _backproject(final_W, K_t, F_surv, P, H_orig, W_orig)  # (F_surv,P) pixels

        if mask_f is not None:
            obs_mask_np = mask_f[0::3, :P].float().cpu().numpy()   # 1=observed

        observed = obs_mask_np > 0.5

        # completed-only positions (NaN where observed)
        xs_comp = np.where(~observed, xp_all * resize_factor, np.nan).astype(np.float32)
        ys_comp = np.where(~observed, yp_all * resize_factor, np.nan).astype(np.float32)

        # residual on observed entries (back-projection vs original obs_mat)
        x_orig = obs_filt[0::2].float().cpu().numpy()   # (F_surv,P) pixels
        y_orig = obs_filt[1::2].float().cpu().numpy()
        resid  = np.sqrt(
            np.where(observed, (xp_all - x_orig)**2 + (yp_all - y_orig)**2, np.nan)
        )
        per_frame_resid = [
            round(float(np.nanmean(resid[f])), 2) if np.any(observed[f]) else 0.0
            for f in range(F_surv)
        ]
        fill_frac = [
            round(float(1.0 - obs_mask_np[f].mean()), 3)
            for f in range(F_surv)
        ]

    longevity = (~torch.isnan(obs_filt[0::2])).sum(dim=0).cpu().numpy().tolist()

    print("Serialising …")
    html = _build_html(
        F_surv, P, H_new, W_new,
        json.dumps(frame_b64),
        json.dumps(_to_json_list(xs_obs)),
        json.dumps(_to_json_list(ys_obs)),
        json.dumps(_to_json_list(xs_comp)),
        json.dumps(_to_json_list(ys_comp)),
        json.dumps(longevity),
        json.dumps(per_frame_resid),
        json.dumps(fill_frac),
        json.dumps(_to_json_list(obs_mask_np)),
        has_comp,
    )

    Path(output_path).write_text(html, encoding='utf-8')
    print(f"Saved → {output_path}")

    if display_in_notebook:
        try:
            from IPython.display import IFrame, display
            display(IFrame(output_path, width='100%', height=860))
        except Exception:
            pass


def _build_html(F, P, H, W, b64_j,
                xs_obs_j, ys_obs_j, xs_comp_j, ys_comp_j,
                lon_j, resid_j, fill_j, obs_mask_j, has_comp):

    comp_controls = ("""
  <span style="color:#333">|</span>
  <label style="display:flex;align-items:center;gap:4px;cursor:pointer">
    <input type="checkbox" id="show-obs"  checked onchange="redrawAll()">
    <span>observed</span>
  </label>
  <label style="display:flex;align-items:center;gap:4px;cursor:pointer">
    <input type="checkbox" id="show-comp" checked onchange="redrawAll()">
    <span style="color:#f83">filled</span>
  </label>
  <label style="display:flex;align-items:center;gap:4px;cursor:pointer">
    <input type="checkbox" id="show-resid" onchange="redrawAll()">
    <span>resid overlay</span>
  </label>""" if has_comp else
    '<span id="show-obs-dummy" style="display:none"></span>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Track Explorer</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:monospace;background:#0d0d0d;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}}
#toolbar{{display:flex;align-items:center;gap:9px;padding:6px 12px;background:#161616;
          border-bottom:1px solid #2a2a2a;flex-shrink:0;flex-wrap:wrap}}
#toolbar label{{font-size:11px;color:#888}}
#toolbar input[type=text]{{background:#1e1e1e;border:1px solid #333;color:#ddd;
  padding:3px 7px;border-radius:3px;font-family:monospace;font-size:11px;width:220px}}
#toolbar input[type=range]{{width:75px;accent-color:#4af}}
#toolbar input[type=checkbox]{{accent-color:#f83;width:12px;height:12px}}
#toolbar select{{background:#1e1e1e;border:1px solid #333;color:#ddd;
  padding:3px 5px;border-radius:3px;font-size:11px}}
#toolbar button{{background:#112233;border:1px solid #4af;color:#4af;
  padding:3px 8px;border-radius:3px;font-size:11px;cursor:pointer}}
#toolbar button:hover{{background:#4af;color:#000}}
#info-bar{{padding:3px 12px;font-size:11px;color:#4af;background:#111;
           flex-shrink:0;min-height:17px}}
#grid-wrap{{flex:1;overflow:auto;padding:8px}}
#grid{{display:flex;flex-wrap:wrap;gap:5px}}
.cell{{position:relative;flex-shrink:0;border:1px solid #222;
       border-radius:2px;overflow:hidden;cursor:crosshair}}
.cell canvas{{position:absolute;top:0;left:0}}
.cell .fnum{{position:absolute;top:2px;left:4px;font-size:9px;
             color:#fff;text-shadow:0 0 3px #000;pointer-events:none;z-index:5}}
.cell .meta{{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.5);
             padding:1px 4px;font-size:9px;color:#999;pointer-events:none;
             display:flex;justify-content:space-between;z-index:5}}
</style>
</head>
<body>
<div id="toolbar">
  <label>feature idx</label>
  <input type="text" id="feat-input" placeholder="all  |  3,7,42  |  0-99">
  <button onclick="applySelection()">apply</button>
  <button onclick="clearSelection()">clear</button>
  <span style="color:#333">|</span>
  <label>dot</label>
  <input type="range" id="dot-size" min="1" max="8" value="3" step="0.5" oninput="redrawAll()">
  <label>alpha</label>
  <input type="range" id="opacity" min="0.1" max="1" value="0.8" step="0.05" oninput="redrawAll()">
  <label>colour</label>
  <select id="color-mode" onchange="redrawAll()">
    <option value="longevity">longevity</option>
    <option value="index">index</option>
    <option value="uniform">uniform</option>
  </select>
  {comp_controls}
  <span id="stat-lbl" style="font-size:10px;color:#444;margin-left:4px"></span>
</div>
<div id="info-bar">hover a dot for info</div>
<div id="grid-wrap"><div id="grid"></div></div>

<script>
const F={F},P={P},H_px={H},W_px={W};
const HAS_COMP={str(has_comp).lower()};
const FRAMES={b64_j};
const XO={xs_obs_j},YO={ys_obs_j};
const XC={xs_comp_j},YC={ys_comp_j};
const LON={lon_j},RESID={resid_j},FILL={fill_j},OMASK={obs_mask_j};

const maxLon=Math.max(...LON);
const maxRes=Math.max(...RESID,0.001);
let selectedFeatures=null, hoveredFeat=-1, hoveredLayer=null;

function lonCol(l,a){{
  const t=l/maxLon;
  return `rgba(${{Math.round(255*(t<.5?2*t:1))}},${{Math.round(255*(t<.5?0:2*(t-.5)))}},${{Math.round(255*(1-t))}},${{a}})`;
}}
function idxCol(p,a){{return `hsla(${{(p*137.508)%360}},70%,60%,${{a}})`;}}
function fCol(p,a){{
  const m=document.getElementById('color-mode').value;
  return m==='longevity'?lonCol(LON[p],a):m==='index'?idxCol(p,a):`rgba(80,180,255,${{a}})`;
}}

const CVS=[],IMGS=[];

function buildGrid(){{
  const g=document.getElementById('grid');
  g.innerHTML='';CVS.length=0;IMGS.length=0;
  for(let f=0;f<F;f++){{
    const cell=document.createElement('div');
    cell.className='cell';
    cell.style.cssText=`width:${{W_px}}px;height:${{H_px}}px`;

    const img=document.createElement('img');
    img.width=W_px;img.height=H_px;
    img.src='data:image/jpeg;base64,'+FRAMES[f];
    img.style.display='block';

    const cv=document.createElement('canvas');
    cv.width=W_px;cv.height=H_px;cv.dataset.frame=f;

    const lbl=document.createElement('div');
    lbl.className='fnum';lbl.textContent='f'+f;

    const meta=document.createElement('div');
    meta.className='meta';meta.id='m'+f;
    if(HAS_COMP)
      meta.innerHTML=`<span>fill ${{(FILL[f]*100).toFixed(1)}}%</span><span style="color:${{RESID[f]>2?'#f84':'#999'}}">Δ${{RESID[f].toFixed(2)}}px</span>`;

    cell.appendChild(img);cell.appendChild(cv);
    cell.appendChild(lbl);if(HAS_COMP)cell.appendChild(meta);
    g.appendChild(cell);
    CVS.push(cv);IMGS.push(img);

    cv.addEventListener('mousemove',onHov);
    cv.addEventListener('mouseleave',onLeave);
    cv.addEventListener('click',onClick);
  }}
}}

function drawFrame(f){{
  const ctx=CVS[f].getContext('2d');
  ctx.clearRect(0,0,W_px,H_px);
  const r=parseFloat(document.getElementById('dot-size').value);
  const a=parseFloat(document.getElementById('opacity').value);
  const showO=!HAS_COMP||document.getElementById('show-obs')?.checked!==false;
  const showC=HAS_COMP&&document.getElementById('show-comp')?.checked;
  const showR=HAS_COMP&&document.getElementById('show-resid')?.checked;

  if(showR&&RESID[f]>0){{
    const t=Math.min(RESID[f]/maxRes,1);
    ctx.fillStyle=`rgba(255,60,20,${{0.08+0.15*t}})`;
    ctx.fillRect(0,0,W_px,H_px);
    ctx.fillStyle=`rgba(255,${{Math.round(140*(1-t))}},${{Math.round(20*(1-t))}},${{0.7+0.3*t}})`;
    ctx.fillRect(0,H_px-3,Math.round(W_px*t),3);
  }}

  const feats=selectedFeatures!==null?selectedFeatures:[...Array(P).keys()];
  for(const p of feats){{
    const hov=(p===hoveredFeat);
    const rr=hov?r*2.2:r;

    if(showO){{
      const x=XO[f][p],y=YO[f][p];
      if(x!==null&&y!==null){{
        const hovThis=hov&&(hoveredLayer==='obs'||hoveredLayer===null);
        const rrr=hovThis?r*2.2:r;
        ctx.beginPath();ctx.arc(x,y,rrr,0,Math.PI*2);
        ctx.fillStyle=hovThis?'rgba(255,255,80,1)':fCol(p,a);
        ctx.fill();
        if(hovThis){{ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke();}}
      }}
    }}

    if(showC){{
      const xc=XC[f][p],yc=YC[f][p];
      if(xc!==null&&yc!==null){{
        const hovThis=hov&&hoveredLayer==='comp';
        const rrr=hovThis?r*2.2:r;
        ctx.beginPath();ctx.arc(xc,yc,rrr+1.5,0,Math.PI*2);
        ctx.strokeStyle=hovThis?'rgba(255,220,0,1)':'rgba(255,120,40,.75)';
        ctx.lineWidth=hovThis?2:1.5;ctx.stroke();
        ctx.beginPath();ctx.arc(xc,yc,rrr*.5,0,Math.PI*2);
        ctx.fillStyle=hovThis?'rgba(255,220,0,.9)':'rgba(255,120,40,.6)';ctx.fill();
      }}
    }}
  }}
}}

function redrawAll(){{
  for(let f=0;f<F;f++)drawFrame(f);
  document.getElementById('stat-lbl').textContent=
    `${{selectedFeatures===null?P:selectedFeatures.length}} features`;
}}

function parseInput(raw){{
  raw=raw.trim();if(!raw)return null;
  const s=new Set();
  for(const t of raw.split(',').map(v=>v.trim()).filter(Boolean)){{
    if(t.includes('-')){{const[a,b]=t.split('-').map(Number);
      for(let i=Math.max(0,a);i<=Math.min(P-1,b);i++)s.add(i);}}
    else{{const n=parseInt(t);if(!isNaN(n)&&n>=0&&n<P)s.add(n);}}
  }}
  return s.size?[...s]:null;
}}
function applySelection(){{selectedFeatures=parseInput(document.getElementById('feat-input').value);redrawAll();}}
function clearSelection(){{selectedFeatures=null;document.getElementById('feat-input').value='';redrawAll();}}

// Returns {{feat, layer}} where layer is 'obs'|'comp'|null
function nearest(f,mx,my){{
  const feats=selectedFeatures!==null?selectedFeatures:[...Array(P).keys()];
  let bestFeat=-1,bestLayer=null,bd=Infinity;
  const showO=!HAS_COMP||document.getElementById('show-obs')?.checked!==false;
  const showC=HAS_COMP&&document.getElementById('show-comp')?.checked;
  for(const p of feats){{
    if(showO){{
      const x=XO[f][p],y=YO[f][p];
      if(x!==null&&y!==null){{
        const d=(x-mx)**2+(y-my)**2;
        if(d<bd){{bd=d;bestFeat=p;bestLayer='obs';}}
      }}
    }}
    if(showC){{
      const x=XC[f][p],y=YC[f][p];
      if(x!==null&&y!==null){{
        const d=(x-mx)**2+(y-my)**2;
        if(d<bd){{bd=d;bestFeat=p;bestLayer='comp';}}
      }}
    }}
  }}
  return bd<400?{{feat:bestFeat,layer:bestLayer}}:{{feat:-1,layer:null}};
}}

function onHov(e){{
  const f=parseInt(e.target.dataset.frame);
  const rc=e.target.getBoundingClientRect();
  const {{feat:p,layer}}=nearest(f,e.clientX-rc.left,e.clientY-rc.top);
  if(p!==hoveredFeat){{
    hoveredFeat=p;hoveredLayer=layer;redrawAll();
    if(p>=0){{
      const nobs=OMASK.filter(row=>row[p]!==null&&row[p]>0.5).length;
      const nfil=OMASK.filter(row=>row[p]!==null&&row[p]<=0.5).length;
      const layerStr=layer==='comp'?' [ALS-filled dot]':' [observed dot]';
      document.getElementById('info-bar').textContent=
        `feat ${{p}}${{layerStr}}  ·  observed ${{nobs}}/${{F}} frames  ·  ALS-filled ${{nfil}} frames  ·  longevity ${{(100*LON[p]/F).toFixed(1)}}%`;
    }}
  }}
}}
function onLeave(){{hoveredFeat=-1;hoveredLayer=null;redrawAll();document.getElementById('info-bar').textContent='hover a dot for info';}}
function onClick(e){{
  const f=parseInt(e.target.dataset.frame);
  const rc=e.target.getBoundingClientRect();
  const {{feat:p}}=nearest(f,e.clientX-rc.left,e.clientY-rc.top);
  if(p<0)return;
  const inp=document.getElementById('feat-input');
  inp.value=(inp.value.trim()?inp.value.trim()+',':'')+p;
  selectedFeatures=parseInput(inp.value);redrawAll();
}}
document.getElementById('feat-input').addEventListener('keydown',e=>{{if(e.key==='Enter')applySelection();}});

buildGrid();
let loaded=0;
IMGS.forEach((img,f)=>{{
  if(img.complete){{loaded++;if(loaded===F)redrawAll();}}
  else img.onload=()=>{{loaded++;if(loaded===F)redrawAll();}};
}});
</script>
</body>
</html>
"""