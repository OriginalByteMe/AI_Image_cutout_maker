import logging
import os

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils


class CutoutHandler:
    # Define the logging format
    log_format = "%(asctime)s [%(levelname)s] %(message)s"

    # Define the color codes for each log level
    color_codes = {
        "DEBUG": "\033[32m",  # Green
        "INFO": "\033[34m",  # Blue
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }

    # Define a custom logging formatter that adds color to the log messages
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            levelname = record.levelname
            if levelname in CutoutHandler.color_codes:
                levelname_color = (
                    f"{CutoutHandler.color_codes[levelname]}{levelname}\033[0m"
                )
                record.levelname = levelname_color
            return super().format(record)

    def __init__(self, model, predictor):
        self.model = model
        self.predictor = predictor

        # Create a logger object
        self.logger = logging.getLogger(__name__)

        # Set the logging level to DEBUG
        self.logger.setLevel(logging.DEBUG)

        # Create a console handler and set its formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            CutoutHandler.ColoredFormatter(CutoutHandler.log_format)
        )

        # Add the console handler to the logger
        self.logger.addHandler(console_handler)

        self.cutout_folder = "generated_images"
        if not os.path.exists(self.cutout_folder):
            os.makedirs(self.cutout_folder)

    def process_image(self, image: Image, name: str, prompt: str) -> list:
        self.logger.debug(f"Processing image '{name}' with prompt '{prompt}'")

        # Convert image to rgb color space
        image = image.convert("RGB")
        # Convert the image to a numpy array
        image_np = np.array(image)
        print("array_shape: ", image_np.shape)
        # Convert the image to RGB format
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Run the model on the image
        results = self.model(image_np)
        # print(results)
        # Set the image for the SegmentAnything predictor
        self.predictor.set_image(image_np)

        masks_list = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                if result.names[int(box.cls[0])] == prompt:
                    self.logger.debug(
                        f"Found bounding box for prompt '{prompt}' in image '{name}'"
                    )
                    # Get the mask from the SegmentAnything predictor
                    masks, scores, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box.xyxy[0].astype(int),
                        multimask_output=False,
                    )
                    if scores[0] < 0.6:
                        self.logger.debug(
                            f"Skipping mask with score {scores[0]} for prompt '{prompt}' in image '{name}'"
                        )
                        continue
                    for i, mask in enumerate(masks):
                        self.logger.debug(
                            f"Processing mask {i+1} of {len(masks)} for prompt '{prompt}' in image '{name}'"
                        )
                        # Convert the mask numpy array to a binary mask
                        mask_binary = np.zeros(
                            (mask.shape[0], mask.shape[1]), dtype=np.uint8
                        )
                        mask_binary[mask > 0] = 1

                        # Convert the binary mask to a COCO RLE format
                        mask_rle = maskUtils.encode(np.asfortranarray(mask_binary))

                        # Extract the counts key from the mask RLE dictionary
                        counts = mask_rle["counts"]

                        # Create a new dictionary with the required keys for COCO RLE format
                        mask_coco_rle = {
                            "size": [mask.shape[0], mask.shape[1]],
                            "counts": counts.decode("utf-8"),
                        }

                        # Add the mask to the list of masks
                        masks_list.append(mask_coco_rle)

        self.logger.debug(
            f"Processed {len(masks_list)} masks for prompt '{prompt}' in image '{name}'"
        )

        # Once finished, create cutouts from the masks
        return self.create_cutout(image, name, masks_list)

    def create_cutout(self, image: np.ndarray, name: str, masks_list: list) -> list:
        cutouts = []
        # Convert the image to a numpy array
        image_np = np.array(image)
        for i, mask in enumerate(masks_list):
            self.logger.debug(
                f"Generating cutout {i+1} of {len(masks_list)} for image '{name}'"
            )

            size = mask["size"]
            counts = mask["counts"]
            mask_decoded = maskUtils.decode({"size": size, "counts": counts})
            mask_binary = np.zeros((size[0], size[1]), dtype=np.uint8)
            mask_binary[mask_decoded > 0] = 1

            # Resize the mask to match the shape of the image
            mask_resized = cv2.resize(mask_decoded, (image_np.shape[1], image_np.shape[0]))

            # Extract the cutout from the image using the mask
            cutout = image * mask_resized[..., np.newaxis]

            # Create an alpha channel for the cutout image
            alpha = np.zeros(cutout.shape[:2], dtype=np.uint8)
            alpha[mask_resized > 0] = 255
            cutout = cv2.merge((cutout, alpha))

            # Crop the cutout image to the bounding rectangle
            x, y, w, h = cv2.boundingRect(mask_resized)
            cutout = cutout[y : y + h, x : x + w]

            # Create a PIL Image from the cutout numpy array
            cutout_pil = Image.fromarray(cutout)

            # Save the cutout to a file
            cutout_filename = f"{name}_{i+1}.png"
            cutout_path = os.path.join(self.cutout_folder, cutout_filename)
            cutout_pil.save(cutout_path)

            self.logger.debug(
                f"Saved cutout {i+1} of {len(masks_list)} to file '{cutout_path}' for image '{name}'"
            )

            # Add the cutout to the list of cutouts
            cutouts.append(cutout_path)
            self.logger.debug(
                f"Added cutout {i+1} of {len(masks_list)} to the list of cutouts for image '{name}'"
            )

        self.logger.info(f"Generated {len(cutouts)} cutouts for image '{name}'")

        return cutouts
