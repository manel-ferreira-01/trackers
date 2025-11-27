import torch
from tqdm import tqdm
from src.ortho_factorization import marques_factorization


def fit_scales_offsets_spectral_tail(W, lam_hat, r=4, iters=1000,
                                     lr=1e-2, alpha=1e-2, beta=1e-3, device=None):
    """
    W:       [2F,P]
    lam_hat: [F,P]
    Returns s,t and the corrected Y(s,t).
    """
    if device is None:
        device = W.device
    F, P = lam_hat.shape
    assert W.shape[0] == 2*F and W.shape[1] == P

    # parameters with safe reparameterization
    s_raw = torch.zeros(F, device=device, requires_grad=True)  # s = 1 + softplus(s_raw)
    t_raw = torch.zeros(F, device=device, requires_grad=True)  # t =     softplus/tanh choice

    opt = torch.optim.Adam([s_raw, t_raw], lr=lr)

    s_history = []
    t_history = []
    loss_history = []

    def build_Y():

        s_history.append(s_raw.detach().cpu().clone())
        t_history.append(t_raw.detach().cpu().clone())

        s = 1.0 + s_raw          # keep s >= 1
        t = t_raw                            # keep t in ~[-0.1,0.1] (adjust)
        #s = s_raw
        #t = t_raw
        lam_corr = (s[:,None]*lam_hat + t[:,None]).clamp_min(1e-6)
        Y = lam_corr.repeat_interleave(2, dim=0) * W

        #project the Y into the motion manifold
        Y = marques_factorization(Y)[0]

        return Y, s, t
    
    first_W = lam_hat.repeat_interleave(2, dim=0) * W

    best = None
    for _ in tqdm(range(iters)):
        opt.zero_grad()
        Y, s, t = build_Y()

        # singular values
        if 0:
            S = torch.linalg.svdvals(Y)
            tail = S[r:]                      # σ_{r+1..}
            loss_tail = (tail**2).sum()
            loss_reg = alpha * ((s-1)**2).sum() + beta * (t**2).sum()
            loss = loss_tail + loss_reg
        else:
            U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
            Sr = torch.diag(S[:r])
            S_tail = S[r:]
            Y_r = U[:, :r] @ Sr @ Vh[:r, :]
            loss = torch.norm(Y - Y_r, p='fro')**2

        loss_history.append(loss.item())
        loss.backward()
        opt.step()

        # keep best by tail energy
        with torch.no_grad():
            if best is None or loss.item() < best[0]:
                best = (loss.item(), s.detach().clone(), t.detach().clone(), Y.detach().clone())

    _, s_best, t_best, Y_best = best

    return torch.stack(s_history), torch.stack(t_history), Y_best, first_W.cpu().detach(), loss_history
