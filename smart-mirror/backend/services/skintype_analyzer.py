import cv2
import numpy as np


class SkinTypeAnalyzer:

    def analyze(self, image):
        """
        Basic heuristic skin type detection.
        image: numpy array (BGR image from OpenCV)
        returns: dict
        """

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)
        texture_variation = np.std(gray)

        if brightness > 170 and texture_variation < 30:
            skin_type = "dry"
        elif brightness < 100 and texture_variation > 50:
            skin_type = "oily"
        elif texture_variation > 70:
            skin_type = "acne_prone"
        elif 100 <= brightness <= 170:
            skin_type = "normal"
        else:
            skin_type = "combination"

        return {
            "skin_type": skin_type,
            "brightness": float(brightness),
            "texture_score": float(texture_variation)
        }