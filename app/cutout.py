import cv2
import numpy as np
import os
from s3_handler import Boto3Client
from dino import Dino
from segment import Segmenter

HOME = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

class CutoutCreator:
  def __init__(self, grounding_dino_config_path, grounding_dino_checkpoint_path):
    self.dino = Dino(classes=['person', 'nose', 'chair', 'shoe', 'ear', 'hat'],
        box_threshold=0.35,
        text_threshold=0.25,
        model_config_path=grounding_dino_config_path,
        model_checkpoint_path=grounding_dino_checkpoint_path)
    self.s3 = Boto3Client()

  def create_cutouts(self, image_name, sam_checkpoint_path):
    
    # Download image from s3
    image_path = self.s3.download_from_s3(os.path.join(HOME, "data"), image_name)
    if not os.path.exists(os.path.join(HOME, "cutouts")):
      os.mkdir(os.path.join(HOME, "cutouts"))
    image = cv2.imread(image_path)
    segment = Segmenter(sam_encoder_version="vit_h", sam_checkpoint_path=sam_checkpoint_path)
    detections = self.dino.predict(image)
    
    masks = segment.segment(image, detections.xyxy)
    # Load the image
    # image_path = os.path.join(self.image_folder, image_name)
    # for item in os.listdir(self.image_folder):
    #   print("Item: ",item)
    if not os.path.exists(image_path):
      print(f"Image {image_name} not found in folder {image_path}")
      return

    image = cv2.imread(image_path)

    # Apply each mask to the image
    for i, mask in enumerate(masks):
      # Ensure the mask is a boolean array
      mask = mask.astype(bool)

      # Apply the mask to create a cutout
      cutout = np.zeros_like(image)
      cutout[mask] = image[mask]

      # Save the cutout
      cutout_name = f"{image_name}_cutout_{i}.png"
      cutout_path = os.path.join(HOME, "cutouts", cutout_name)
      cv2.imwrite(cutout_path, cutout)

      # Upload the cutout to S3
      with open(cutout_path, "rb") as f:
        self.s3.upload_to_s3(f.read(), "cutouts",f"{image_name}/{cutout_name}")

