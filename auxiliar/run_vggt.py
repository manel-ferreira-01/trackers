import sys
import cv2
import io
import grpc
import json
sys.path.append("../boxes/vggt/protos")
import pipeline_pb2, pipeline_pb2_grpc, aux
import numpy as np

def load_image_bytes(paths):
    images_bytes = []
    for p in paths:
        with open(p, "rb") as f:
            images_bytes.append(f.read())
    return images_bytes


def frames_to_image_bytes(frames):
    """Convert a list of (H, W, 3) numpy arrays or torch tensors (RGB, uint8) to JPEG bytes."""
    images_bytes = []
    for frame in frames:
        if hasattr(frame, "numpy"):
            frame = frame.detach().cpu().numpy()
        if frame.dtype != np.uint8:
            frame = ((frame + 1) * 127.5).clip(0, 255)
        img = frame.astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img_bgr)
        if not ok:
            raise RuntimeError("Failed to encode frame")
        images_bytes.append(buf.tobytes())
    return images_bytes


def make_request(image_paths):
    config = {
            "command": "3d_infer",
            "parameters": {"conf_threshold": 25, "device": "cuda:0"}
        }
    images = load_image_bytes(image_paths)
    return pipeline_pb2.Envelope(
        config_json=json.dumps(config),
        data={"images": aux.wrap_value(images)}
    )


def run_vggt_frames(frames):
    """Run VGGT from a list of (H, W, 3) RGB tensors/arrays instead of file paths."""
    max_message_length = 512 * 1024 * 1024

    channel = grpc.insecure_channel(
        "localhost:8061",
        options=[
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
        ],
    )
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)
    config = {
            "command": "3d_infer",
            "parameters": {"conf_threshold": 25, "device": "cuda:0"}
        }
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps(config),
        data={"images": aux.wrap_value(frames_to_image_bytes(frames))}
    ))

    extrinsic = np.load(io.BytesIO(aux.unwrap_value(response.data["extrinsic"])))
    return extrinsic, response


def run_vggt(image_paths):
    # Increase gRPC max message size limits (e.g. 512 MB)
    max_message_length = 512 * 1024 * 1024

    channel = grpc.insecure_channel(
        "localhost:8061",
        options=[
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
        ],
    )
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)
    response = stub.Process(
        make_request(image_paths)
    )
    extrinsic = np.load(io.BytesIO(aux.unwrap_value(response.data["extrinsic"])))
    return extrinsic