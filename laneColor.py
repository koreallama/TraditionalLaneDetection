from config import (LANE_COLOR, LANE_AREA_COLOR , TEXT_COLOR_WARNING,
                    TEXT_COLOR_STEERING, TEXT_COLOR_INSTRUCTION,
                    INTERSECTION_LINE_COLOR, INTERSECTION_POINT_COLOR,
                    CAR_CENTER_COLOR, HOUGH_LINE_COLOR)


# 색상 변환 딕셔너리 (텍스트 → BGR)
COLOR_MAP = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "orange": (0, 165, 255),
    "purple": (128, 0, 128),
    "pink": (203, 192, 255),
}

def get_bgr_color(color_name):
    """텍스트 색상 이름을 BGR 값으로 변환"""
    return COLOR_MAP.get(color_name.lower(), (0, 255, 0))  # 기본값: 초록색

lane_color = get_bgr_color(LANE_COLOR)
lane_area_color = get_bgr_color(LANE_AREA_COLOR)
text_color_warning = get_bgr_color(TEXT_COLOR_WARNING)
text_color_steering = get_bgr_color(TEXT_COLOR_STEERING)
text_color_instruction = get_bgr_color(TEXT_COLOR_INSTRUCTION)
intersection_line_color = get_bgr_color(INTERSECTION_LINE_COLOR)
intersection_point_color = get_bgr_color(INTERSECTION_POINT_COLOR)
car_center_color = get_bgr_color(CAR_CENTER_COLOR)
hough_color = get_bgr_color(HOUGH_LINE_COLOR)