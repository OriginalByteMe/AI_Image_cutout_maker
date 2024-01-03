from app.common import cpu_cutout_handler_stub
from modal import web_endpoint

@cpu_cutout_handler_stub.function()
@web_endpoint(method="POST")
async def create_cutouts(image_name: str, request: Request):
    """
    Create cutouts from an image and upload them to S3.

    Args:
        image_name (str): Name of image to create cutouts from.
        classes (List[str], optional): A list of classes for the AI to detect for. Defaults to Body(...).

    Returns:
        _type_: _description_
    """
    from app.aws.s3_handler import Boto3Client

    try:
        # Log the start of the process
        logger.info("Creating cutouts for image %s ", image_name)

        # Parse the request body as JSON
        data = await request.json()

        # Get the classes and accuracy level from the JSON data
        classes = data.get("classes", [])
        accuracy_level = data.get("accuracy_level", "low")
        logger.info("Classes: %s", classes)
        logger.info("Accuracy level: %s", accuracy_level)

        # Select the SAM checkpoint path based on the accuracy level
        accuracy_encoder_versions = {
            "high": "vit_h",
            "mid": "vit_l",
            "low": "vit_b",
        }
        encoder_version = accuracy_encoder_versions.get(accuracy_level, "vit_b")

        # Initialize the S3 client and the CutoutCreator
        s3 = Boto3Client()
        cutout = CutoutCreator(
            classes=classes,
            grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            grounding_dino_config_path=GROUNDING_DINO_CONFIG_PATH,
            encoder_version=encoder_version,
        )

        # Create the cutouts
        logger.info(f"CREATING CUTOUTS FOR IMAGE {image_name}")
        cutout.create_cutouts(image_name)
        logger.info("Cutouts created for image %s", image_name)

        # Generate presigned URLs for the cutouts
        urls = s3.generate_presigned_urls(f"cutouts/{image_name}")
        logger.info("Presigned URLs generated for cutouts of image %s", image_name)

        # Return the URLs
        return urls
    except Exception as e:
        # Log any errors that occur
        logger.error(
            "An error occurred while creating cutouts for image %s: %s", image_name, e
        )
        raise
