import os
import modal
import cv2

stub = modal.Stub(name="cutout_generator")

img_volume = modal.NetworkFileSystem.persisted("image-storage-vol")
cutout_volume = modal.NetworkFileSystem.persisted("cutout-storage-vol")
local_packages = modal.Mount.from_local_python_packages("cutout", "dino", "segment", "s3_handler")
cutout_generator_image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3").pip_install( "segment-anything", "opencv-python", "botocore", "boto3").run_commands(
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
  "wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg -P data/",
  "ls -F",
  "ls -F GroundingDINO/groundingdino/config",
  "ls -F GroundingDINO/groundingdino/models/GroundingDINO/"
)
# Setup paths
HOME = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

@stub.function(image=cutout_generator_image, mounts=[local_packages], gpu="T4", secret=modal.Secret.from_name("my-aws-secret"), network_file_systems={"/images": img_volume, "/cutouts": cutout_volume})
@modal.web_endpoint()
def main(image_name):
  # Import relevant classes
  from dino import Dino
  from segment import Segmenter
  from cutout import CutoutCreator
  from s3_handler import Boto3Client
  SOURCE_IMAGE_PATH = os.path.join(HOME, "data", image_name)
  print(f"SOURCE_IMAGE_PATH: {SOURCE_IMAGE_PATH}")
  SAVE_IMAGE_PATH = os.path.join(HOME, "data")
  OUTPUT_CUTOUT_PATH = os.path.join(HOME, "cutouts")
  dino = Dino(classes=['person', 'nose', 'chair', 'shoe', 'ear', 'hat'],
      box_threshold=0.35,
      text_threshold=0.25,
      model_config_path=GROUNDING_DINO_CONFIG_PATH,
      model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
  
  segment = Segmenter(sam_encoder_version="vit_h", sam_checkpoint_path=SAM_CHECKPOINT_PATH)
  
  cutout = CutoutCreator(image_folder=SAVE_IMAGE_PATH)
  
  s3 = Boto3Client()
  
  s3.download_from_s3(SAVE_IMAGE_PATH, "cutout-image-store", f"images/{image_name}")
  
  image = cv2.imread(SOURCE_IMAGE_PATH)
  
  # Run the DINO model on the image
  detections = dino.predict(image)
  
  detections.mask = segment.segment(image, detections.xyxy)
  
  cutout.create_cutouts(image_name, detections.mask, OUTPUT_CUTOUT_PATH, "cutout-image-store", s3)
  
  return "Success"  
  
