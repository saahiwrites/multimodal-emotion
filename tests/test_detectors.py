import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_emotion import (
    MultiModalEmotionDetector,
    TextEmotionDetector
)

class TestEmotionDetectors(unittest.TestCase):

    def setUp(self):
        self.text_detector = TextEmotionDetector({})
        self.emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

    def test_text_detection(self):
        result = self.text_detector.predict("I'm so happy today!")

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(self.emotions))

        for emotion in self.emotions:
            self.assertIn(emotion, result)
            self.assertIsInstance(result[emotion], float)
            self.assertGreaterEqual(result[emotion], 0.0)
            self.assertLessEqual(result[emotion], 1.0)

        total_prob = sum(result.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)

    def test_text_only_multimodal(self):
        detector = MultiModalEmotionDetector()
        result = detector.predict_multimodal(text="I'm happy!")

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(self.emotions))

if __name__ == "__main__":
    unittest.main()
