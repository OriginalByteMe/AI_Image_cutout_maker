import cv2
import numpy as np
import os


class CutoutCreator:
  def __init__(self, image_folder):
    self.image_folder = image_folder

  def create_cutouts(self, image_name, masks, output_folder,bucket_name, s3):
    # Load the image
    image_path = os.path.join(self.image_folder, image_name)
    for item in os.listdir(self.image_folder):
      print("Item: ",item)
    if not os.path.exists(image_path):
      print(f"Image {image_name} not found in folder {self.image_folder}")
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
      cutout_path = os.path.join(output_folder, cutout_name)
      cv2.imwrite(cutout_path, cutout)

      # Upload the cutout to S3
      with open(cutout_path, "rb") as f:
        s3.upload_to_s3(bucket_name, f.read(), f"cutouts/{image_name}/{cutout_name}")

