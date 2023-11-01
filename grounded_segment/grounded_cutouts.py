import os
import modal
import cv2

# Import relevant classes
from dino import Dino
from segment import Segmenter
from cutout import CutoutCreator
from s3_handler import Boto3Client

stub = modal.Stub(name="cutout_generator")

cutout_generator_image = modal.Image.debian_slim(python_version="3.10").pip_install("torch", "segment-anything", "cv2", "botocore").run_commands(
  "apt-get update",
  "apt-get install -y git wget",
  "git clone https://github.com/IDEA-Research/GroundingDINO.git",
  "mkdir -p /weights",
  "mkdir -p /data",
  "git -C GroundingDINO/ checkout -q 57535c5a79791cb76e36fdb64975271354f10251",
  "pip install -q -e GroundingDINO/",
  "pip install 'git+https://github.com/facebookresearch/segment-anything.git'",
  "pip uninstall -y supervision",
  "pip install -q supervision==0.6.0",
  "wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P weights/",
  "wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights/",
  "wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg -P data/",
)
# Setup paths
HOME = os.getcwd()
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

@stub.function(image=cutout_generator_image)
def main(image_name):
  SOURCE_IMAGE_PATH = os.path.join(HOME, "data", image_name)
  
  dino = Dino(classes=['person', 'nose', 'chair', 'shoe', 'ear', 'hat'],
      box_threshold=0.35,
      text_threshold=0.25,
      model_config_path=GROUNDING_DINO_CONFIG_PATH,
      model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
  
  segment = Segmenter(sam_encoder_version="vit_h", sam_checkpoint_path=SAM_CHECKPOINT_PATH)
  
  cutout = CutoutCreator(image_folder=f"${HOME}/data")
  
  s3 = Boto3Client()
  
  s3.download_from_s3(SOURCE_IMAGE_PATH, "cutoutimagestore", image_name)
  
  image = cv2.imread(SOURCE_IMAGE_PATH)
  
  # Run the DINO model on the image
  detections = dino.predict(image)
  
  detections.mask = segment.segment(image, detections.xyxy)
  
  cutout.create_cutouts(image_name, detections.mask)
  