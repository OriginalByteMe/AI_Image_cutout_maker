from modal import Image, Mount, App

s3_handler_image = Image.debian_slim().pip_install("boto3", "botocore")

cutout_generator_image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "segment-anything", "opencv-python", "botocore", "boto3", "fastapi", "starlette"
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y git wget libgl1-mesa-glx libglib2.0-0",
        "git clone https://github.com/IDEA-Research/GroundingDINO.git",
        "pip install -q -e GroundingDINO/",
        "mkdir -p /weights",
        "mkdir -p /data",
        "pip uninstall -y supervision",
        "pip uninstall -y opencv-python",
        "pip install opencv-python==4.8.0.74",
        "pip install -q supervision==0.6.0",
        "wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P weights/",
        "wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights/",
        "wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P weights/",
        "wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P weights/",
    )
)

local_packages = Mount.from_local_python_packages(
    "app.cutout_handler.dino",
    "app.cutout_handler.segment",
    "app.cutout_handler.s3_handler",
    "app.cutout_handler.grounded_cutouts",
)

cutout_handler_app = App(image=cutout_generator_image, name="cutout_generator")
s3_handler_cutouts_app = App(image=s3_handler_image, name="s3_handler_cutouts")
s3_handler_app = App(image=s3_handler_image, name="s3_handler")
