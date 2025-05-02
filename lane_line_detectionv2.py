import cv2
import numpy as np
import logging
import argparse
import os

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

FRAME_SKIP = 2  # Process every 2nd frame

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return [], []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = make_coordinates(image, np.average(left_fit, axis=0)) if left_fit else None
    right_line = make_coordinates(image, np.average(right_fit, axis=0)) if right_fit else None

    return left_line, right_line

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(image, left_line, right_line):
    line_image = np.zeros_like(image)
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)  # Red (left)
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)  # Green (right)
    return line_image

def process_frame(frame):
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=5)
    left_line, right_line = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, left_line, right_line)
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def run_on_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logging.info(f"Saving output to {output_path}")
    else:
        out = None

    frame_count = 0
    logging.info("Processing video... Press 'q' to exit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video stream or read error.")
            break

        # Skip every N frames for performance
        if frame_count % FRAME_SKIP == 0:
            processed = process_frame(frame)
        else:
            processed = frame

        if out:
            out.write(processed)
        cv2.imshow("Lane Line Detection", processed)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    logging.info("Processing complete.")

# def main():
#     parser = argparse.ArgumentParser(description="Enhanced Lane Line Detection on Video.")
#     parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
#     parser.add_argument("--output", type=str, help="Optional: Path to save the output video.")
#     args = parser.parse_args()

#     run_on_video(args.video, args.output)

# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    input_path = "videos/test_video.mp4"
    output_path = None  # set
    run_on_video(input_path, output_path)
