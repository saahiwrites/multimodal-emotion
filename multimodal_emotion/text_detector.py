import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List

class TextEmotionDetector:
    def __init__(self, config: Dict):
        self.model_name = config.get("model_name", "j-hartmann/emotion-english-distilroberta-base")
        self.max_length = config.get("max_length", 512)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

    def preprocess(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self, text: str) -> Dict[str, float]:
        inputs = self.preprocess(text)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]

        return dict(zip(self.emotions, probabilities.tolist()))

    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
