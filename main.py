import cv2
import time
from laneDetector import LaneDetector
from config import (VIDEO_PATH, WINDOW_NAME, FRAME_WAIT_TIME, CAMERA_MODE, CAMERA_INDEX,
                    MAX_RETRIES, RETRY_DELAY, EXIT_KEY, LOG_LEVEL,
                    CAMERA_WIDTH, CAMERA_HEIGHT, DEBUGING_VIDEOS)


# 로그 메시지를 출력하는 함수
# 설정된 LOG_LEVEL에 따라 지정된 수준 이상의 로그만 출력
def log(level, message):
    # 로그 레벨 리스트
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    # 현재 설정된 LOG_LEVEL보다 높은 레벨의 메시지만 출력
    if levels.index(level) >= levels.index(LOG_LEVEL):
        print(f"[{level}] {message}")


def main():
    # 카메라 모드인지 비디오 모드인지 설정 확인
    if CAMERA_MODE:
        # 지정된 카메라 번호로 웹캠 실행
        cap = cv2.VideoCapture(CAMERA_INDEX)
        log("INFO", f"카메라 모드로 실행됩니다. (카메라 인덱스: {CAMERA_INDEX})")

        # 카메라 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # 실제 설정된 해상도 확인
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log("INFO", f"설정된 카메라 해상도: {actual_width}x{actual_height}")
    else:
        # 지정된 비디오 파일 실행
        cap = cv2.VideoCapture(VIDEO_PATH)
        log("INFO", f"영상 파일 '{VIDEO_PATH}'을(를) 재생합니다.")

    # 카메라나 비디오 파일이 열릴 때까지 최대 MAX_RETRIES번 재시도
    for attempt in range(MAX_RETRIES):
        # 카메라 또는 비디오가 정상적으로 열렸는지 확인
        if cap.isOpened():
            break
        log("WARNING", f"카메라 연결 실패. {RETRY_DELAY}초 후 재시도 ({attempt + 1}/{MAX_RETRIES})...")

        # 지정된 시간만큼 대기 후 다시 시도
        time.sleep(RETRY_DELAY)

    # 카메라 또는 비디오 파일이 열리지 않으면 프로그램 종료
    if not cap.isOpened():
        log("ERROR", "영상을 열 수 없습니다. 파일 경로 또는 카메라 연결을 확인하세요.")
        return

    # 차선 감지 모델(LaneDetector) 초기화
    try:
        detector = LaneDetector()
    except Exception as e:
        log("CRITICAL", f"LaneDetector 초기화 실패: {e}")
        return

    # 처리한 프레임 개수를 저장하는 변수
    frame_count = 0

    # OpenCV 윈도우 생성
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # 카메라 모드일 경우 프레임 속도를 최소 1ms로 설정하여 실시간성 유지
    frame_wait_time = max(1, FRAME_WAIT_TIME) if CAMERA_MODE else FRAME_WAIT_TIME

    # 사용자 정의 종료 키를 ASCII 코드로 변환
    exit_key_code = ord(EXIT_KEY)

    # 영상 프레임을 읽어와 처리하는 루프
    while cap.isOpened():
        # 프레임 한 장을 읽어옴
        ret, frame = cap.read()

        # 프레임을 읽지 못한 경우 (영상이 끝났거나 손상됨)
        if not ret or frame is None:
            log("WARNING", "프레임을 읽지 못했습니다. 영상이 끝났거나 손상되었습니다.")
            break

        # 프레임 개수 증가
        frame_count += 1

        # 프레임 크기 정보 출력
        log("DEBUG", f"{frame_count}번째 프레임 크기: {frame.shape}")

        # 디버그 모드일 때 중간 과정 영상 출력
        if DEBUGING_VIDEOS:
            # 변환된 프레임 3개(gray, blurred, edges) 반환
            gray, blurred, edges = detector.preprocess_frame(frame)
            # 관심 영역 적용
            roi_applied = detector.region_of_interest(edges)
            # 허프 변환 적용
            hough_result, _, _, = detector.hough_transform(roi_applied)

            # 그레이스케일 영상
            cv2.imshow("Gray Scale", gray)
            # 블러 영상
            cv2.imshow("Blurred", blurred)
            # Canny 엣지 검출 영상
            cv2.imshow("Edge Detection", edges)
            # 관심 영역 적용된 영상
            cv2.imshow("ROI Applied", roi_applied)
            # 허프 변환 적용된 영상
            cv2.imshow("Hough Transform", hough_result)

        # 차선 감지 모델을 통해 프레임 처리
        processed_frame = detector.detect_lanes(frame)

        # 처리된 프레임을 화면에 표시
        cv2.imshow(WINDOW_NAME, processed_frame)

        # 사용자가 종료 키를 누르면 프로그램 종료
        if cv2.waitKey(frame_wait_time) & 0xFF == exit_key_code:
            log("INFO", f"'{EXIT_KEY}' 키가 눌려 프로그램을 종료합니다.")
            break

    # 카메라 또는 비디오 파일 닫기
    cap.release()

    # OpenCV 창 닫기
    cv2.destroyAllWindows()


# 프로그램 실행 시작
if __name__ == "__main__":
    main()