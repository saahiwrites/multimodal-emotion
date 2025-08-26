import torch
import numpy as np
from typing import Dict, List, Optional, Union
from .text_detector import TextEmotionDetector
from .audio_detector import AudioEmotionDetector
from .image_detector import ImageEmotionDetector
from .utils import load_config, combine_predictions

class MultiModalEmotionDetector:
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = load_config(config_path)
        self.emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

        self.text_detector = TextEmotionDetector(self.config.get("text", {}))
        self.audio_detector = AudioEmotionDetector(self.config.get("audio", {}))
        self.image_detector = ImageEmotionDetector(self.config.get("image", {}))

        self.fusion_weights = self.config.get("fusion_weights", {
            "text": 0.4,
            "audio": 0.3, 
            "image": 0.3
        })

    def predict_text(self, text: str) -> Dict[str, float]:
        return self.text_detector.predict(text)

    def predict_audio(self, audio_path: str) -> Dict[str, float]:
        return self.audio_detector.predict(audio_path)

    def predict_image(self, image_path: str) -> Dict[str, float]:
        return self.image_detector.predict(image_path)

    def predict_multimodal(
        self,
        text: Optional[str] = None,
        audio: Optional[str] = None,
        image: Optional[str] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        predictions = {}
        weights = custom_weights or self.fusion_weights

        if text:
            predictions["text"] = self.predict_text(text)

        if audio:
            predictions["audio"] = self.predict_audio(audio)

        if image:
            predictions["image"] = self.predict_image(image)

        if not predictions:
            raise ValueError("At least one modality input must be provided")

        return combine_predictions(predictions, weights, self.emotions)

    def predict_batch(self, inputs: List[Dict]) -> List[Dict[str, float]]:
        results = []
        for input_item in inputs:
            result = self.predict_multimodal(**input_item)
            results.append(result)
        return results
