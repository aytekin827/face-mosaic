from ultralytics import YOLO
import cv2
import moviepy.editor as mp

# YOLO 설정 (ultralytics에서 제공)
model = YOLO('yolov8n-pose.pt')  # YOLOv8n 포즈 모델 사용 (작고 빠름). 필요에 따라 변경 가능

# 동영상 파일 경로 설정
video_path = 'input_video.mp4'
output_path = 'output_video.mp4'  # 출력 동영상 경로
pose_output_path = 'pose_output_video.mp4'  # 포즈 시각화 출력 동영상 경로
mosaic_output_path = 'mosaic_output_video.mp4'  # 포즈 시각화 출력 동영상 경로

# 동영상 파일 열기
cap = cv2.VideoCapture(video_path)

# 프레임당 속성을 가져오기 위한 초기 설정
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# VideoWriter 객체 초기화
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
pose_out = cv2.VideoWriter(pose_output_path, fourcc, fps, (frame_width, frame_height))

# 모자이크 박스크기 조절 변수 (기본값)
mosaic_box_scale = 2  # 모자이크 박스 크기 조절을 위한 스케일, 필요에 따라 수정 가능
blur_strength = 19  # 블러 처리의 강도를 조절하는 변수

while cap.isOpened():
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break

    # YOLO 추론
    results = model.predict(source=frame, save=False, conf=0.5)  # confidence threshold 설정
    detections = results[0].boxes  # 검출 결과
    keypoints = results[0].keypoints  # 포즈 결과 (사람의 관절 정보)

    # 포즈 시각화
    pose_frame = frame.copy()
    results[0].plot(show=False, save=False, source=pose_frame)  # YOLO 포즈 결과를 플로팅하여 시각화

    pose_out.write(results[0].plot(show=False, save=False, source=pose_frame))  # 포즈 시각화된 프레임을 출력 파일에 기록

    for i, box in enumerate(detections):
        # 박스 정보 가져오기
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표 변환
        confidence = box.conf[0].item()
        cls = int(box.cls[0].item())

        # 클래스가 사람(person)인 경우에만 얼굴 검출 수행 (YOLOv8에서 클래스 0은 일반적으로 사람)
        if cls == 0 and keypoints is not None:  # 사람이 검출된 경우
            person_keypoints = keypoints[i]  # 해당 사람의 관절 정보

            # 필요한 keypoints: 왼쪽 눈(1), 오른쪽 눈(2), 코(0), 왼쪽 귀(3), 오른쪽 귀(4), 입술(7)
            keypoints_indices = [0, 1, 2, 3, 4]
            selected_keypoints = [person_keypoints.data.squeeze()[idx] for idx in keypoints_indices]

            # 유효한 keypoints만 사용
            valid_keypoints = [kp for kp in selected_keypoints if kp[2] > 0.5]  # 신뢰도가 0.5 이상인 것만
            if len(valid_keypoints) >= 2:
                # x, y 좌표 분리
                x_coords = [int(kp[0]) for kp in valid_keypoints]
                y_coords = [int(kp[1]) for kp in valid_keypoints]

                # 얼굴 영역을 포함하는 최소/최대 좌표 계산
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # 모자이크 박스 크기 계산 (스케일 적용)
                box_width = int((x_max - x_min) * mosaic_box_scale)
                box_height = int((y_max - y_min) * mosaic_box_scale)

                # 중앙 좌표를 기준으로 박스 설정
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                x1_face = max(0, center_x - box_width // 3)
                y1_face = max(0, center_y - (box_height*3) // 2)
                x2_face = min(frame_width, center_x + box_width // 3)
                y2_face = min(frame_height, center_y + (box_height*3) // 2)

                # 얼굴 영역 추출
                face_img = frame[y1_face:y2_face, x1_face:x2_face]

                # 모자이크 처리: 축소 후 확대
                if box_width < 10 or box_height < 10:
                    continue
                # 얼굴 영역 추출
                face_img = frame[y1_face:y2_face, x1_face:x2_face]
                # 블러처리
                face_img = cv2.GaussianBlur(face_img, (blur_strength | 1, blur_strength | 1), 0)

                # 모자이크 처리된 얼굴 영역을 원본 프레임에 삽입
                frame[y1_face:y2_face, x1_face:x2_face] = face_img

    out.write(frame)  # 모자이크 처리된 프레임을 출력 파일에 기록

    # 모자이크 처리 과정을 화면에 출력
    cv2.imshow('Mosaic', frame)
    cv2.imshow('Mosaic Process', results[0].plot(show=False, save=False, source=pose_frame))
    cv2.imshow('Pose Visualization', pose_frame)

    # ESC 키를 누르면 중지
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 모든 자원 해제
cap.release()
out.release()
pose_out.release()
cv2.destroyAllWindows()

# 원본 오디오를 출력 영상에 합치기
input_video = mp.VideoFileClip(video_path)
output_video = mp.VideoFileClip(output_path)
output_with_audio = output_video.set_audio(input_video.audio)
output_with_audio.write_videofile('output_video_with_audio.mp4', codec='libx264', audio_codec='aac', fps=fps)