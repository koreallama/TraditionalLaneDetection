import cv2
from config import LANE_THICKNESS
from laneColor import lane_color


class Lane:
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.x1, self.y1 = None, None
        self.x2, self.y2 = None, None

    def update(self, slope, intercept, y1, y2):
        self.slope = slope
        self.intercept = intercept
        self.y1, self.y2 = y1, y2
        self.x1 = int((y1 - intercept) / slope)
        self.x2 = int((y2 - intercept) / slope)

    def is_valid(self):
        return self.slope is not None and self.intercept is not None

    def draw(self, frame):
        if self.is_valid() and self.x1 >= 0 and self.y1 >= 0 and self.x2 >= 0 and self.y2 >= 0:
            cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2), lane_color, LANE_THICKNESS)