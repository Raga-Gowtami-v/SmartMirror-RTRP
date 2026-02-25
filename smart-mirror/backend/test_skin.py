import cv2
from services.skintype_analyzer import SkinTypeAnalyzer

# Load test image
img = cv2.imread("test.jpg")

if img is None:
    print("Image not loaded. Check file name.")
else:
    analyzer = SkinTypeAnalyzer()
    result = analyzer.analyze(img)
    print("Skin Type Result:")
    print(result)