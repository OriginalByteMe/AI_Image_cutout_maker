import cv2
import numpy as np
import os


class CutoutCreator:
  def __init__(self, image_folder):
    self.image_folder = image_folder

  def create_cutouts(self, image_name, masks):
    # Load the image
    image_path = os.path.join(self.image_folder, image_name)
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
      cv2.imwrite(cutout_name, cutout)

