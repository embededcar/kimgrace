import cv2
import numpy as np
import torch

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 이미지 파일 경로
image_path = 'color_img/green3.jpg'

# 이미지 로드
img = cv2.imread(image_path)

# 이미지에서 신호등 탐지
results = model(image_path)
detections = results.pandas().xyxy[0]  # x1, y1, x2, y2, confidence, class

# 신호등 색상 판별 함수
def detect_traffic_light_color(cropped_img):
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    # 색상 범위 설정
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    lower_yellow = np.array([15, 70, 50])
    upper_yellow = np.array([35, 255, 255])

    lower_green = np.array([36, 70, 50])
    upper_green = np.array([89, 255, 255])

    red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

    red_ratio = np.sum(red_mask) / (cropped_img.size / 3)
    yellow_ratio = np.sum(yellow_mask) / (cropped_img.size / 3)
    green_ratio = np.sum(green_mask) / (cropped_img.size / 3)

    # 색상 비율 기준 조정
    if red_ratio > 0.05:
        return 'Red'
    elif yellow_ratio > 0.05:
        return 'Yellow'
    elif green_ratio > 0.05:
        return 'Green'
    else:
        return 'Unknown'

def preprocess_image(img):
    # Gaussian Blur 적용
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def visualize_masks(cropped_img):
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    lower_yellow = np.array([15, 70, 50])
    upper_yellow = np.array([35, 255, 255])

    lower_green = np.array([36, 70, 50])
    upper_green = np.array([89, 255, 255])

    red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

    cv2.imshow('Red Mask', red_mask)
    cv2.imshow('Yellow Mask', yellow_mask)
    cv2.imshow('Green Mask', green_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 검출 결과를 원본 이미지에 표시
for index, row in detections.iterrows():
    if row['class'] == 9:  # YOLOv5에서 신호등 클래스 ID가 9인 경우
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cropped_img = img[y1:y2, x1:x2]

        # 전처리 및 시각화
        cropped_img = preprocess_image(cropped_img)
        visualize_masks(cropped_img)

        # 신호등 색상 판별
        color = detect_traffic_light_color(cropped_img)
        print(f'Traffic light detected at ({x1}, {y1}, {x2}, {y2}) with color: {color}')

        # 결과 이미지에 박스와 색상 레이블 표시
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, color, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 결과 이미지 표시
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
