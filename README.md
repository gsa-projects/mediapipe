# mediapipe-study

`mediapipe` 로 눈 깜박임 카운팅하기.

1. facemesh를 그리고, 눈 위치를 추출함.
```py
 # 눈 위치 좌표 추출
landmarks = face_landmarks.landmark
left_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in range(362, 374)])
right_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in range(384, 398)])
```

2. EAR(Eye Aspect Ratio)을 구하는 함수를 작성함.
```py
def calculate_eye_aspect_ratio(eye):
    # 눈의 수직 거리 계산
    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])

    # 눈의 수평 거리 계산
    horizontal = np.linalg.norm(eye[0] - eye[3])

    # EAR (Eye Aspect Ratio) 계산
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear
```

3. 미리 정해놓은 threshold보다 크면 눈을 깜박였다고 가정함.
```py
# 눈을 감았다고 판단하는 EAR 임계값 설정
EAR_THRESHOLD = 1.12
# 연속적으로 눈을 감은 프레임 수 임계값
CONSEC_FRAMES = 2
# 눈 깜박임 감지
if ear >= EAR_THRESHOLD:
    frame_count += 1
else:
    if frame_count >= CONSEC_FRAMES:
        blink_count += 1
    frame_count = 0
```
