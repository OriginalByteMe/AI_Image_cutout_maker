import torch
from groundingdino.util.inference import Model

class Dino:
  def __init__(self, classes, box_threshold, text_threshold, model_config_path, model_checkpoint_path):
    self.classes = classes
    self.box_threshold = box_threshold
    self.text_threshold = text_threshold
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.grounding_dino_model = Model(model_config_path=model_config_path, model_checkpoint_path=model_checkpoint_path)

  def predict(self, image):
    detections = self.grounding_dino_model.predict(image, self.classes, self.box_threshold, self.text_threshold)
    detections = detections[detections.class_id != None]
    return detections
# Example usage
# dino = Dino(classes=['person', 'nose', 'chair', 'shoe', 'ear', 'hat'],
#       box_threshold=0.35,
#       text_threshold=0.25,
#       model_config_path=GROUNDING_DINO_CONFIG_PATH,
#       model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
