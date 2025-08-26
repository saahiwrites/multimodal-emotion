import torch
import cv2
import numpy as np
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
from typing import Dict, List

class ImageEmotionDetector:
    def __init__(self, config: Dict):
        self.model_name = config.get("model_name", "trpakov/vit-face-expression")
        self.image_size = config.get("image_size", 224)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)
        self.model = ViTForImageClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_face(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]
            return face
        else:
            return image

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face = self.detect_face(image)
        face_pil = Image.fromarray(face)

        inputs = self.feature_extractor(
            images=face_pil,
            return_tensors="pt"
        )

        return inputs.pixel_values.to(self.device)

    def predict(self, image_path: str) -> Dict[str, float]:
        inputs = self.preprocess_image(image_path)

        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]

        return dict(zip(self.emotions, probabilities.tolist()))

    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, float]]:
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        return results
