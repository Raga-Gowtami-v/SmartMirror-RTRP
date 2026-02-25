import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cheek_x1 = x + int(w * 0.6)
        cheek_y1 = y + int(h * 0.5)
        cheek_x2 = x + int(w * 0.9)
        cheek_y2 = y + int(h * 0.8)

        cheek_roi = frame[cheek_y1:cheek_y2, cheek_x1:cheek_x2]

        if cheek_roi.size != 0:

            lab = cv2.cvtColor(cheek_roi, cv2.COLOR_BGR2LAB)

            # Split LAB channels
            L_channel, A_channel, B_channel = cv2.split(lab)

            # Apply Histogram Equalization to reduce lighting impact
            L_channel = cv2.equalizeHist(L_channel)

            # Merge back
            lab_normalized = cv2.merge((L_channel, A_channel, B_channel))

            L = np.mean(L_channel)
            A = np.mean(A_channel)
            B = np.mean(B_channel)

            # Skin Tone Classification (more stable)
            if L > 160:
                tone = "Fair"
            elif L > 110:
                tone = "Medium"
            else:
                tone = "Dark"

            # Undertone Classification
            if abs(A - B) < 5:
                undertone = "Neutral"
            elif B > A:
                undertone = "Warm"
            else:
                undertone = "Cool"

            cv2.putText(frame, f"L:{int(L)} A:{int(A)} B:{int(B)}",
                        (x, y-35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1)

            cv2.putText(frame, f"Tone: {tone}, Undertone: {undertone}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            cv2.rectangle(frame, (cheek_x1, cheek_y1),
                          (cheek_x2, cheek_y2), (255, 0, 0), 2)

    cv2.imshow("Skin Tone & Undertone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()