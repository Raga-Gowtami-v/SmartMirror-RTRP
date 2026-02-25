import cv2
import numpy as np

# Load Haar cascade once
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


class SkinTypeAnalyzer:

    def analyze(self, image):
        """
        Analyze skin type using HSV + texture heuristics.
        Returns dictionary with result.
        """

        # --- Face Detection ---
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_full,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) == 0:
            return {
                "skin_type": "No face detected"
            }

        # Take first detected face
        x, y, w, h = faces[0]
        face_region = image[y:y+h, x:x+w]

        # --- HSV Analysis on FACE ONLY ---
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        mean_s = np.mean(s_channel)
        mean_v = np.mean(v_channel)

        # --- Texture Analysis on FACE ONLY ---
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_face, (5, 5), 0)
        texture = cv2.Laplacian(blur, cv2.CV_64F).var()

        # --- Improved Heuristic Logic ---
        # # Detect oiliness
        if mean_v > 155 and mean_s > 90:
            base_type = "oily"
        # Detect dryness
        elif mean_v < 120 and mean_s < 70:
            base_type = "dry"
        # Otherwise normal
        else:
            base_type = "normal"
        # Detect acne separately using texture
        if texture > 280:
            if base_type == "normal":
                skin_type = "acne_prone"
            else:
                skin_type = base_type + " + acne_prone"
        else:
            skin_type = base_type

        return {
            "skin_type": skin_type,
            "mean_saturation": float(mean_s),
            "mean_brightness": float(mean_v),
            "texture_score": float(texture)
        }