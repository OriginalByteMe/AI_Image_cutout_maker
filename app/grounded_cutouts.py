import os
from modal import asgi_app, Secret, Stub, Mount, Image
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()

stub = Stub(name="cutout_generator")

local_packages = Mount.from_local_python_packages("cutout", "dino", "segment", "s3_handler", "fastapi", "starlette")
cutout_generator_image = Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3").pip_install( "segment-anything", "opencv-python", "botocore", "boto3").run_commands(
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
  "ls -F GroundingDINO/groundingdino/models/GroundingDINO/"
)

HOME = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

@stub.function(mounts=[Mount.from_local_python_packages("s3_handler")], secret=Secret.from_name("my-aws-secret"))
@app.post("/upload-image")
async def upload_image_to_s3(image: UploadFile = File(...)):
    from s3_handler import Boto3Client
    s3_client = Boto3Client()
    s3_client.upload_to_s3(image.file,"images", image.filename)
    return {"message": "Image uploaded successfully"}

@stub.function(mounts=[Mount.from_local_python_packages("s3_handler")], secret=Secret.from_name("my-aws-secret"))
@app.get("/create-presigned-cutout/{image_name}")
async def create_presigned_cutout(image_name: str):
    from s3_handler import Boto3Client
    s3_client = Boto3Client()
    presigned_url = s3_client.generate_presigned_url(image_name)
    return {"presigned_url": presigned_url}

@stub.function(image=cutout_generator_image, mounts=[local_packages], gpu="T4", secret=Secret.from_name("my-aws-secret"))
@app.get("/create-cutouts/{image_name}")
async def create_cutouts(image_name: str):
    from cutout import CutoutCreator
    from s3_handler import Boto3Client
    s3 = Boto3Client()
    cutout = CutoutCreator(grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, grounding_dino_config_path=GROUNDING_DINO_CONFIG_PATH)
    cutout.create_cutouts(image_name, SAM_CHECKPOINT_PATH )
    return s3.generate_presigned_urls(f"cutouts/{image_name}")

@stub.function(image=cutout_generator_image,gpu="T4", mounts=[local_packages], secret=Secret.from_name("my-aws-secret"))
@asgi_app()
def cutout_app():
  return app
