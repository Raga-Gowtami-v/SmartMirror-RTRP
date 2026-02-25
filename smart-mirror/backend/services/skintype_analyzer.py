import cv2
import numpy as np

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


class SkinTypeAnalyzer:

    def analyze(self, image):

        # -------------------- FACE DETECTION --------------------
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_full,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        if len(faces) == 0:
            return {"skin_type": "No face detected"}

        # Take largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        face_region = image[y:y+h, x:x+w]

        # -------------------- CHEEK CROPPING --------------------
        fh, fw = face_region.shape[:2]

        cheek_region = face_region[
            int(fh * 0.35):int(fh * 0.8),
            int(fw * 0.25):int(fw * 0.75)
        ]

        if cheek_region.size == 0:
            cheek_region = face_region

        # -------------------- HSV ANALYSIS --------------------
        hsv = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2HSV)
        _, s_channel, v_channel = cv2.split(hsv)

        mean_s = np.mean(s_channel)
        mean_v = np.mean(v_channel)

        # -------------------- TEXTURE ANALYSIS --------------------
        gray_face = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2GRAY)

        # Slight contrast boost (to fight Mac smoothing)
        gray_face = cv2.convertScaleAbs(gray_face, alpha=1.6, beta=0)

        # Light blur to remove noise
        blur = cv2.GaussianBlur(gray_face, (3, 3), 0)

        texture = cv2.Laplacian(blur, cv2.CV_64F).var()

        # -------------------- BASE SKIN TYPE --------------------

        # OILY: strong shine + high saturation
        if mean_v > 150 and mean_s > 95:
            base_type = "oily"

        # DRY: dull + low saturation
        elif mean_v < 110 and mean_s < 60:
            base_type = "dry"

        # NORMAL: balanced brightness + balanced saturation
        elif 120 <= mean_v <= 145 and 70 <= mean_s <= 90:
            base_type = "normal"

        # Everything else → combination
        else:
            base_type = "combination"
            print("Brightness:", mean_v, "Saturation:", mean_s)

        # -------------------- ACNE DETECTION --------------------

        # Acne only when:
        # 1) Texture at upper edge of your real range
        # 2) Skin already oily or combination
        # 3) Saturation high (avoid dry acne)
        if texture > 16.5 and base_type in ["oily", "combination"] and mean_s > 88:
            skin_type = base_type + " + acne_prone"
        else:
            skin_type = base_type

        return {
            "skin_type": skin_type,
            "mean_saturation": float(mean_s),
            "mean_brightness": float(mean_v),
            "texture_score": float(texture)
        }