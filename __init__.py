
from .core import MultiModalEmotionDetector
from .text_detector import TextEmotionDetector
from .audio_detector import AudioEmotionDetector
from .image_detector import ImageEmotionDetector

__version__ = "0.1.0"
__all__ = [
    "MultiModalEmotionDetector",
    "TextEmotionDetector", 
    "AudioEmotionDetector",
    "ImageEmotionDetector"
]
