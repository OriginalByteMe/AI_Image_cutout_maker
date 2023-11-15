import os
from modal import asgi_app, Secret, Stub, Mount, Image
from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()

stub = Stub(name="cutout_generator")

origins = [
    "http://localhost:3000",  # localdevelopment
    "https://cutouts.noahrijkaard.com",  # main website
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

local_packages = Mount.from_local_python_packages(
    "cutout", "dino", "segment", "s3_handler"
)
cutout_generator_image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install("segment-anything", "opencv-python","botocore", "boto3", "fastapi", "starlette")
    .run_commands(
        "apt-get update",
        "apt-get install -y git wget libgl1-mesa-glx libglib2.0-0",
        "echo $CUDA_HOME",
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
        "wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg -P images/",
        "ls -F",
        "ls -F GroundingDINO/groundingdino/config",
        "ls -F GroundingDINO/groundingdino/models/GroundingDINO/",
    )
)

HOME = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
GROUNDING_DINO_CONFIG_PATH = os.path.join(
    HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
    HOME, "weights", "groundingdino_swint_ogc.pth"
)
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")


@app.post("/create-cutouts/{image_name}")
async def create_cutouts(image_name: str, classes: List[str] = Body(...)):
    """Create cutouts from an image and upload them to S3.

    Args:
        image_name (str): Name of image to create cutouts from.
        classes (List[str], optional): A list of classes for the AI to detect for. Defaults to Body(...).

    Returns:
        _type_: _description_
    """
    from cutout import CutoutCreator
    from s3_handler import Boto3Client

    s3 = Boto3Client()
    cutout = CutoutCreator(
        classes=classes,
        grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        grounding_dino_config_path=GROUNDING_DINO_CONFIG_PATH,
    )
    cutout.create_cutouts(image_name, SAM_CHECKPOINT_PATH)
    return s3.generate_presigned_urls(f"cutouts/{image_name}")

@app.post("/create-cutouts")
async def create_all_cutouts(image_names: List[str] = Body(...), classes: List[str] = Body(...)):
    """Create cutouts from multiple images and upload them to S3.

    Args:
        image_names (List[str]): List of image names to create cutouts from.
        classes (List[str], optional): A list of classes for the AI to detect for. Defaults to Body(...).

    Returns:
        Dict[str, List[str]]: A dictionary where the keys are the image names and the values are the lists of presigned URLs for the cutouts.
    """
    from cutout import CutoutCreator
    from s3_handler import Boto3Client

    s3 = Boto3Client()
    cutout = CutoutCreator(
        classes=classes,
        grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        grounding_dino_config_path=GROUNDING_DINO_CONFIG_PATH,
    )

    result = {}
    for image_name in image_names:
        cutout.create_cutouts(image_name, SAM_CHECKPOINT_PATH)
        result[image_name] = s3.generate_presigned_urls(f"cutouts/{image_name}")

    return result

@stub.function(
    image=cutout_generator_image,
    gpu="T4",
    mounts=[local_packages],
    secret=Secret.from_name("my-aws-secret"),
)
@asgi_app()
def cutout_app():
    """Create a FastAPI app for creating cutouts.

    Returns:
        FastAPI: FastAPI app for creating cutouts.
    """
    return app
