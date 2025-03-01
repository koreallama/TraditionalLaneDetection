import cv2
import numpy as np
from lane import Lane
from config import (GAUSSIAN_BLUR_KERNEL, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD,
                    HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH,
                    HOUGH_MAX_LINE_GAP, LEFT_LANE_SLOPE_RANGE, RIGHT_LANE_SLOPE_RANGE,
                    LANE_Y2_RATIO, LANE_AREA_ALPHA, TEXT_FONT_SCALE, TEXT_THICKNESS,
                    INTERSECTION_LINE_THICKNESS, INTERSECTION_POINT_THICKNESS,
                    INTERSECTION_POINT_RADIUS, MIDDLE_POINT_ALPHA, STEERING_KP,
                    HORIZONTAL_Y_RATIO, STEERING_THRESHOLD_SHARP, STEERING_THRESHOLD_SLIGHT,
                    STEERING_THRESHOLD_STRAIGHT, TEXT_POSITION_WARNING, TEXT_POSITION_STEERING,
                    TEXT_POSITION_INSTRUCTION, HOUGH_LINE_THICKNESS)
from laneColor import (lane_area_color, text_color_warning, text_color_steering,
                       text_color_instruction, intersection_line_color, intersection_point_color,
                       car_center_color, hough_color)


class LaneDetector:
    def __init__(self):
        self.left_lane = Lane()
        self.right_lane = Lane()
        self.steering_offset = 0

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
        edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        return gray, blurred, edges

    def region_of_interest(self, img):
        height, width = img.shape
        mask = np.zeros_like(img, dtype=np.uint8)
        polygon = np.array([[
            (-width // 4, height),
            (width + width // 4, height),
            (width // 2, height // 2)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(img, mask)

    def classify_lane_line(self, line):
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
        intercept = y1 - (slope * x1)

        if LEFT_LANE_SLOPE_RANGE[0] < slope < LEFT_LANE_SLOPE_RANGE[1]:
            return 'left', (slope, intercept)
        elif RIGHT_LANE_SLOPE_RANGE[0] < slope < RIGHT_LANE_SLOPE_RANGE[1]:
            return 'right', (slope, intercept)
        return None, None

    def hough_transform(self, edges):
        lines = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                                minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)
        left_lines, right_lines = [], []
        hough_result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                category, fit = self.classify_lane_line(line[0])

                if category == 'left':
                    left_lines.append(fit)
                elif category == 'right':
                    right_lines.append(fit)
                
                slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')

                if (LEFT_LANE_SLOPE_RANGE[0] < slope < LEFT_LANE_SLOPE_RANGE[1]) or (RIGHT_LANE_SLOPE_RANGE[0] < slope < RIGHT_LANE_SLOPE_RANGE[1]):
                    cv2.line(hough_result, (x1, y1), (x2, y2), hough_color, HOUGH_LINE_THICKNESS)

        left_fit = np.mean(left_lines, axis=0) if left_lines else None
        right_fit = np.mean(right_lines, axis=0) if right_lines else None

        return hough_result, left_fit, right_fit

    def update_lane(self, lane, fit, y1, y2):
        if fit is not None:
            lane.update(fit[0], fit[1], y1, y2)

    def update_lanes(self, left_fit, right_fit, frame):
        height, width, _ = frame.shape
        y1 = height
        y2 = int(height * LANE_Y2_RATIO)

        self.update_lane(self.left_lane, left_fit, y1, y2)
        self.update_lane(self.right_lane, right_fit, y1, y2)

    def apply_overlay(self, frame, overlay, alpha):
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_overlay(self, frame, polygon, color, alpha):
        overlay = frame.copy()
        cv2.fillPoly(overlay, polygon, color)
        self.apply_overlay(frame, overlay, alpha)

    def draw_lane_area(self, frame):
        if self.left_lane.is_valid() and self.right_lane.is_valid():
            polygon = np.array([[
                (self.left_lane.x1, self.left_lane.y1),
                (self.left_lane.x2, self.left_lane.y2),
                (self.right_lane.x2, self.right_lane.y2),
                (self.right_lane.x1, self.right_lane.y1)
            ]], np.int32)

            self.draw_overlay(frame, polygon, lane_area_color, LANE_AREA_ALPHA)

    def draw_text_on_frame(self, frame, text, position, color):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, color, TEXT_THICKNESS)

    def draw_multiple_texts(self, frame, texts):
        for text, position, color in texts:
            self.draw_text_on_frame(frame, text, position, color)

    def detect_lanes(self, frame):
        _, _, edges = self.preprocess_frame(frame)
        roi_edges = self.region_of_interest(edges)
        _, left_fit, right_fit = self.hough_transform(roi_edges)

        if left_fit is not None or right_fit is not None:
            self.update_lanes(left_fit, right_fit, frame)
            warning = "Lanes Detected"
        else:
            warning = "Warning, No Lanes Detected"

        self.draw_lane_area(frame)
        self.left_lane.draw(frame)
        self.right_lane.draw(frame)
        self.draw_intersection_points(frame)

        frame_height, frame_width, _ = frame.shape
        control_signal, steering_angle = self.compute_steering_signal(frame_width, frame_height)

        texts = [
            (warning, TEXT_POSITION_WARNING, text_color_warning),
            (f"Steering Offset: {steering_angle}", TEXT_POSITION_STEERING, text_color_steering),
            (f"Instruction: {control_signal}", TEXT_POSITION_INSTRUCTION, text_color_instruction),
        ]
        self.draw_multiple_texts(frame, texts)

        return frame

    def compute_lane_intersections(self, frame_height):
        if not (self.left_lane.is_valid() and self.right_lane.is_valid()):
            return None, None, None

        horizontal_y = int(frame_height * HORIZONTAL_Y_RATIO)
        left_x_intersect = int((horizontal_y - self.left_lane.intercept) / self.left_lane.slope)
        right_x_intersect = int((horizontal_y - self.right_lane.intercept) / self.right_lane.slope)
        lane_center_x = (left_x_intersect + right_x_intersect) // 2

        return left_x_intersect, right_x_intersect, lane_center_x

    def draw_intersection_points(self, frame):
        left_x_intersect, right_x_intersect, center_x = self.compute_lane_intersections(frame.shape[0])

        if left_x_intersect is None or right_x_intersect is None:
            return

        height, width, _ = frame.shape
        car_center_x = width // 2
        horizontal_y = int(height * HORIZONTAL_Y_RATIO)

        cv2.line(frame, (0, horizontal_y), (width, horizontal_y), intersection_line_color, INTERSECTION_LINE_THICKNESS)

        cv2.circle(frame, (left_x_intersect, horizontal_y), INTERSECTION_POINT_RADIUS, intersection_point_color, INTERSECTION_POINT_THICKNESS)  # 왼쪽 차선 교차점
        cv2.circle(frame, (right_x_intersect, horizontal_y), INTERSECTION_POINT_RADIUS, intersection_point_color, INTERSECTION_POINT_THICKNESS)  # 오른쪽 차선 교차점

        cv2.circle(frame, (car_center_x, horizontal_y), INTERSECTION_POINT_RADIUS, car_center_color, INTERSECTION_POINT_THICKNESS)

        overlay = frame.copy()
        cv2.circle(overlay, (center_x, horizontal_y), INTERSECTION_POINT_RADIUS,
                   intersection_point_color, INTERSECTION_POINT_THICKNESS)
        self.apply_overlay(frame, overlay, MIDDLE_POINT_ALPHA)

    def compute_steering_signal(self, frame_width, frame_height):
        _, _, lane_center_x = self.compute_lane_intersections(frame_height)

        if lane_center_x is None:
            return "No lane detected", 0

        car_center_x = frame_width // 2
        deviation = lane_center_x - car_center_x
        steering_angle = STEERING_KP * deviation

        # 스티어링 방향 결정
        if abs(deviation) < STEERING_THRESHOLD_STRAIGHT:
            control_signal = "Go Straight"
        elif deviation > STEERING_THRESHOLD_SHARP:
            control_signal = "Turn Right (Sharp)"
        elif deviation > STEERING_THRESHOLD_SLIGHT:
            control_signal = "Turn Right (Slight)"
        elif deviation < -STEERING_THRESHOLD_SHARP:
            control_signal = "Turn Left (Sharp)"
        elif deviation < -STEERING_THRESHOLD_SLIGHT:
            control_signal = "Turn Left (Slight)"
        else:
            control_signal = "Go Straight"

        return control_signal, steering_angle