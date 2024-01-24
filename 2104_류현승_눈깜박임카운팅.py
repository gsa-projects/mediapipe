import cv2
import mediapipe as mp
import numpy as np

# MediaPipe의 얼굴 메쉬 모듈 초기화
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe의 드로잉 유틸리티를 위한 초기화
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def calculate_eye_aspect_ratio(eye):
    # 눈의 수직 거리 계산
    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])

    # 눈의 수평 거리 계산
    horizontal = np.linalg.norm(eye[0] - eye[3])

    # EAR (Eye Aspect Ratio) 계산
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# 눈을 감았다고 판단하는 EAR 임계값 설정
EAR_THRESHOLD = 1.12
# 연속적으로 눈을 감은 프레임 수 임계값
CONSEC_FRAMES = 2

# 카메라 초기화
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    # 깜박임 카운트 변수 초기화
    blink_count = 0
    frame_count = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # 이미지를 BGR에서 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 얼굴 메쉬 감지 수행
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 랜드마크를 이미지에 그림
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                # 눈 위치 좌표 추출
                landmarks = face_landmarks.landmark
                left_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in range(362, 374)])
                right_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in range(384, 398)])

                # EAR 계산
                left_ear = calculate_eye_aspect_ratio(left_eye)
                right_ear = calculate_eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                # 눈 깜박임 감지
                if ear >= EAR_THRESHOLD:
                    frame_count += 1
                else:
                    if frame_count >= CONSEC_FRAMES:
                        blink_count += 1
                    frame_count = 0

        # 깜박임 카운트 표시
        cv2.putText(image, "Blinks: {}".format(blink_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Eye Blink Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
