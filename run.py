import os
import json
import numpy as np
from pycocotools import mask as maskUtils
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from io import BytesIO
import base64
import cv2
from PIL import Image
from s3FileHandler import Boto3Client

client = Boto3Client()

def generate_cutout(**inputs) -> None:
    prompt = inputs["prompt"]
    image = inputs["image"]

    # Get the current working directory
    cwd = os.getcwd()

    images_path = os.path.join("uploaded_images/Images")
    masks_path = os.path.join("masks")
    cutouts_path = os.path.join("cutouts")
    svg_path = os.path.join("svg")
    video_images_path = os.path.join("video_images")

    #  LOAD SEGMENT ANYTHING MODEL
    sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    # LOAD YOLO MODEL
    model = YOLO("./models/yolov8x.pt")

    # Process the image and append the cutouts to the list
    result = process_image(image, "output.jpg", prompt, model, predictor)
    
    return result



def process_image(image: Image, name: str, prompt: str, model, predictor) -> list:
    # 1. Run YOLO to get bounding boxes
    # 2. Run SAM to get masks
    # 3. Return the masks
    # Convert image to rgb color space
    image = image.convert("RGB")
    # Convert the image to a numpy array
    image_np = np.array(image)

    # Convert the image to RGB format
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Run the model on the image
    results = model(image_np)

    # Set the image for the SegmentAnything predictor
    predictor.set_image(image_np)

    masks_list = []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            if result.names[int(box.cls[0])] == prompt:
                # Get the mask from the SegmentAnything predictor
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box.xyxy[0].astype(int),
                    multimask_output=False,
                )
                if scores[0] < 0.6:
                    print("Score is too low, skipping, score is: ", scores[0])
                    continue
                for i, mask in enumerate(masks):
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

    # Once finished, create cutouts from the masks
    create_cutout(image, name, masks_list)


def create_cutout(image: Image, name, masks: list) -> None:
    # Convert the image to a numpy array
    image_np = np.array(image)
    
    # Remove extension from name
    name = os.path.splitext(name)[0]
    preview_urls = []
    # Loop through all the masks
    for i, mask in enumerate(masks):
        # Decode the mask
        size = mask["size"]
        counts = mask["counts"]
        mask_decoded = maskUtils.decode({"size": size, "counts": counts})
        mask_binary = np.zeros((size[0], size[1]), dtype=np.uint8)
        mask_binary[mask_decoded > 0] = 1

        # Resize the mask to match the shape of the image
        mask_resized = cv2.resize(mask_decoded, (image_np.shape[1], image_np.shape[0]))

        # Extract the cutout from the image using the mask
        cutout = image_np * mask_resized[..., np.newaxis]

        # Create an alpha channel for the cutout image
        alpha = np.zeros(cutout.shape[:2], dtype=np.uint8)
        alpha[mask_resized > 0] = 255
        cutout = cv2.merge((cutout, alpha))

        # Crop the cutout image to the bounding rectangle
        x, y, w, h = cv2.boundingRect(mask_resized)
        cutout = cutout[y : y + h, x : x + w]

        # Create a PIL Image from the cutout numpy array
        cutout_pil = Image.fromarray(cutout)

        # Save the cutout to the cutouts folder
        cutout_key = f"{name}_cutout_{i}.png"
        # Upload the cutout to the S3 bucket
        bucket_name = "cutoutimagestore"
        file_body = BytesIO()
        cutout_pil.save(file_body, format="png")
        file_body.seek(0)
        client.upload_to_s3(bucket_name, file_body, cutout_key)
        url = client.generate_presigned_url(bucket_name, cutout_key)
        preview_urls.append(url)
    return url

  


def list_directories(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            print(os.path.join(root, dir))

def list_files(directory):
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            print(filename)