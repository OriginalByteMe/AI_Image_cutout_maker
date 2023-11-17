import torch
from groundingdino.util.inference import Model
from typing import List


class Dino:
    """ A class for object detection using GroundingDINO.
    """
    def __init__(
        self,
        classes,
        box_threshold,
        text_threshold,
        model_config_path,
        model_checkpoint_path,
    ):
        self.classes = classes
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grounding_dino_model = Model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
        )

    def enhance_class_name(self, class_names: List[str]) -> List[str]:
        """Enhance class names for GroundingDINO.

        Args:
            class_names (List[str]): List of class names.

        Returns:
            List[str]: List of class names with "all" prepended.
        """
        return [f"all {class_name}s" for class_name in class_names]

    def predict(self, image):
        """Predict objects in an image.

        Args:
            image (File): Image to be used for object detection.

        Returns:
            Dict[str, list]: Dictionary of objects detected in the image.
        """
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=self.enhance_class_name(class_names=self.classes),
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        detections = detections[detections.class_id != None]
        return detections


# Example usage
# dino = Dino(classes=['person', 'nose', 'chair', 'shoe', 'ear', 'hat'],
#       box_threshold=0.35,
#       text_threshold=0.25,
#       model_config_path=GROUNDING_DINO_CONFIG_PATH,
#       model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
