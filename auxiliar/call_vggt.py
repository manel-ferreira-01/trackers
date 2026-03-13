import grpc
import os
import sys
import json

sys.path.append('../boxes/vggt/protos')
import pipeline_pb2 as vggt_pb2
import pipeline_pb2_grpc as vggt_pb2_grpc
from aux import wrap_value, unwrap_value
from PIL import Image
import io
def call_vggt(video_tensor,device="cuda:=0", path=None):

    image_byte_list = []
    # List all files in the directory
    if 0:
        
        files = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]

        for file in sorted(files):
            # Open the file in binary read mode ('rb') and read its entire content
            with open(file, 'rb') as f:
                image_bytes = f.read()
                image_byte_list.append(image_bytes)
                print(f"Read {file}: {len(image_bytes) / (1024 * 1024):.2f} MB")

    else:

        # Convert video frames (tensor) to a list of byte arrays
        for frame in video_tensor[0]:  # Assuming video_tensor_original is [1, time, H, W, C]
            # Convert tensor to numpy array and scale to [0, 255]
            frame_np = (frame.numpy() * 255).astype('uint8')
            # Create an Image object
            img = Image.fromarray(frame_np)
            # Save the image to a bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            # Append the bytes to the list
            image_byte_list.append(buffer.getvalue())

    config_json = {
            "command": "3d_infer",
            "parameters": {
                "device": device, # TODO: implement this
                "conf_vis": 50.0
            }
    }

    request = vggt_pb2.Envelope(data={"images":wrap_value(image_byte_list)},
                                config_json = json.dumps(config_json))
    channel_opt = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    channel=grpc.insecure_channel("localhost:8061",options=channel_opt)
    estimator_stub = vggt_pb2_grpc.PipelineServiceStub(channel)
    response = estimator_stub.Process(request)
    channel.close()

    #write the glb as a file
    glb_file = unwrap_value(response.data["glb_file"])
    with open("output.glb", "wb") as f:
        f.write(glb_file)

    return response
        