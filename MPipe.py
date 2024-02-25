import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import winsound

MINIMUM_EAR = 0.2
MAXIMUM_FRAME_COUNT = 10
EYE_CLOSED_COUNTER = 0

mp_face_mesh = mp.solutions.face_mesh

webcamFeed = cv2.VideoCapture(0)

left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145]
right_eye_indices = [362, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380]


def eye_aspect_ratio(eye_landmarks):
    p2_minus_p6 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    p3_minus_p5 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    p1_minus_p4 = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear


def sound_alert():
    winsound.Beep(1000, 500)


try:
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2) as face_mesh:
        while True:
            (status, image) = webcamFeed.read()
            if not status:
                print("Error reading frame from webcam.")
                break

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    left_eye_landmarks = [
                        (int(landmarks.landmark[i].x * image.shape[1]), int(landmarks.landmark[i].y * image.shape[0]))
                        for i in left_eye_indices]
                    right_eye_landmarks = [
                        (int(landmarks.landmark[i].x * image.shape[1]), int(landmarks.landmark[i].y * image.shape[0]))
                        for i in right_eye_indices]

                    left_ear = eye_aspect_ratio(left_eye_landmarks)
                    right_ear = eye_aspect_ratio(right_eye_landmarks)
                    avg_ear = (left_ear + right_ear) / 2.0

                    if avg_ear < MINIMUM_EAR:
                        EYE_CLOSED_COUNTER += 1
                        cv2.putText(image, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(image, f"Counter: {EYE_CLOSED_COUNTER}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                    else:
                        EYE_CLOSED_COUNTER = 0
                        cv2.putText(image, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    for point in left_eye_landmarks + right_eye_landmarks:
                        cv2.circle(image, point, 2, (0, 255, 0), -1)

            if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
                sound_alert()

            cv2.imshow("Frame", image)
            cv2.waitKey(1)

except KeyboardInterrupt:
    pass

webcamFeed.release()
cv2.destroyAllWindows()
