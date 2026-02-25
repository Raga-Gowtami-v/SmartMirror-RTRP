import cv2
import numpy as np


class SkinTypeAnalyzer:

    def analyze(self, image):
        """
        Analyze skin type using HSV + texture heuristics.
        image: BGR OpenCV image
        """

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        mean_s = np.mean(s)
        mean_v = np.mean(v)

        # Texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Heuristic logic
        if mean_s > 120 and mean_v > 140:
            skin_type = "oily"
        elif mean_s < 60 and mean_v < 120:
            skin_type = "dry"
        elif texture > 150:
            skin_type = "acne_prone"
        elif 60 <= mean_s <= 120:
            skin_type = "normal"
        else:
            skin_type = "combination"

        return {
            "skin_type": skin_type,
            "mean_saturation": float(mean_s),
            "mean_brightness": float(mean_v),
            "texture_score": float(texture)
        }