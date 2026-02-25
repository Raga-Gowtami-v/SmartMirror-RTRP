import cv2
from services.skintype_analyzer import SkinTypeAnalyzer

img = cv2.imread("test.jpg")  # Put any face image here
analyzer = SkinTypeAnalyzer()

result = analyzer.analyze(img)
print(result)