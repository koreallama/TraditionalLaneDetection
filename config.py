import numpy as np


# 로그 레벨 설정
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" 중 선택
DEBUGING_VIDEOS = True # 중간 과정 영상 출력

# 영상 관련 설정
CAMERA_MODE = False  # True이면 웹캠 사용, False이면 비디오 파일 사용
VIDEO_PATH = "videotest1.mp4"  # 비디오 파일 경로
CAMERA_INDEX = 0  # 사용할 카메라 번호

# 카메라 해상도 설정
CAMERA_WIDTH = 1280  # 가로 해상도
CAMERA_HEIGHT = 720  # 세로 해상도

# OpenCV 윈도우 설정
WINDOW_NAME = "Lane Detection"  # 윈도우 이름
FRAME_WAIT_TIME = 1  # 프레임 속도 조절, 카메라 모드일 때 자동 조정됨
EXIT_KEY = 'q' # 프로그램 종료 키

# 카메라 연결 재시도 설정
MAX_RETRIES = 5  # 최대 재시도 횟수
RETRY_DELAY = 1  # 재시도 간격 (초)

# 차선(Line) 관련 설정
LANE_COLOR = "green"  # 차선을 그릴 색상 (BGR)
LANE_THICKNESS = 5  # 차선을 그릴 두께

# 차선 검출 필터 설정
GAUSSIAN_BLUR_KERNEL = (5, 5)  # 블러 필터 크기
CANNY_LOW_THRESHOLD = 50       # Canny 엣지 검출 저역 임계값
CANNY_HIGH_THRESHOLD = 150     # Canny 엣지 검출 고역 임계값

# 허프 변환 파라미터 설정
HOUGH_RHO = 1  # 거리 해상도 (픽셀)
HOUGH_THETA = np.pi / 180  # 각도 해상도 (라디안)
HOUGH_THRESHOLD = 50  # 최소 교차점 개수 (라인 검출 최소 임계값)
HOUGH_MIN_LINE_LENGTH = 50  # 최소 라인 길이
HOUGH_MAX_LINE_GAP = 150  # 라인 간 최대 허용 간격
HOUGH_LINE_COLOR = "green" # 허프 변환으로 검출된 직선의 색상
HOUGH_LINE_THICKNESS = 2 # 직선의 두께께

# 차선 검출 기울기 범위
LEFT_LANE_SLOPE_RANGE = (-0.9, -0.3)
RIGHT_LANE_SLOPE_RANGE = (0.3, 0.9)

# 차선 높이 설정
LANE_Y2_RATIO = 0.6

# 차선 영역 시각화 설정
LANE_AREA_COLOR = "green"  # 초록색
LANE_AREA_ALPHA = 0.3  # 투명도

# 텍스트 색상 설정
TEXT_COLOR_WARNING = "green"  # 감지된 차선 상태
TEXT_COLOR_STEERING = "red"  # 스티어링 오프셋 표시
TEXT_COLOR_INSTRUCTION = "blue"  # 제어 신호 표시
TEXT_FONT_SCALE = 1  # 글자 크기
TEXT_THICKNESS = 2  # 글자 두께

# 텍스트 위치 설정
TEXT_POSITION_WARNING = (50, 50)
TEXT_POSITION_STEERING = (50, 100)
TEXT_POSITION_INSTRUCTION = (50, 150)

# 교차점 및 기준선 색상 설정
INTERSECTION_LINE_COLOR = "yellow"  # 기준선 색상
INTERSECTION_POINT_COLOR = "red"  # 교차점 색상
CAR_CENTER_COLOR = "blue"  # 차량 중심점 색상

# 선 및 점의 크기 설정
INTERSECTION_LINE_THICKNESS = 2
INTERSECTION_POINT_RADIUS = 8
INTERSECTION_POINT_THICKNESS = -1  # 채우기

# 중간점 투명도 설정
MIDDLE_POINT_ALPHA = 0.3  # 반투명 점의 투명도 (0: 완전 투명, 1: 불투명)

# 스티어링 제어 설정
STEERING_KP = 0.02  # P 제어 계수 (비례 제어)
HORIZONTAL_Y_RATIO = 0.8  # 차선 교차점의 수평선 위치 (프레임 높이 비율)

# 스티어링 방향 설정 (편차 기준)
STEERING_THRESHOLD_STRAIGHT = 10  # 중앙과의 편차가 10 이하면 직진
STEERING_THRESHOLD_SLIGHT = 20  # 중앙과의 편차가 20 이상이면 약한 회전
STEERING_THRESHOLD_SHARP = 50  # 중앙과의 편차가 50 이상이면 급회전