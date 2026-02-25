import cv2
from services.skintype_analyzer import SkinTypeAnalyzer

# Initialize analyzer
analyzer = SkinTypeAnalyzer()

# Start webcam
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Analyze skin type
    result = analyzer.analyze(frame)
    skin_type = result["skin_type"]

    # Display result on frame
    cv2.putText(
        frame,
        f"Skin Type: {skin_type}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Smart Mirror - Skin Type Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()