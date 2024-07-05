import cv2
import numpy as np

def detect_pupil_and_calculate_magnification(video_path):
    cap = cv2.VideoCapture(video_path)
    baseline_radius = None
    magnification_levels = []
    last_center = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        _, threshold = cv2.threshold(enhanced_gray, 30, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                if radius > 5:
                    if baseline_radius is None:
                        baseline_radius = radius

                    magnification = radius / baseline_radius if baseline_radius else 1
                    magnification_levels.append(magnification)
                    print(f"Current Magnification: {magnification:.2f}x") 

                    if last_center:
                        dx = center[0] - last_center[0]
                        dy = center[1] - last_center[1]
                        print(f"Movement - X: {dx}, Y: {dy}")

                    cv2.circle(frame, center, radius, (255, 0, 0), 2)
                    cv2.circle(frame, center, 2, (0, 255, 0), -1)
                    cv2.putText(frame, f"Magnification: {magnification:.2f}x", (center[0] - 100, center[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.line(frame, (center[0], 0), (center[0], frame.shape[0]), (0, 255, 0), 1)
                    cv2.line(frame, (0, center[1]), (frame.shape[1], center[1]), (0, 255, 0), 1)
                    last_center = center
                    break

        draw_magnification_graph(frame, magnification_levels)
        cv2.imshow('Pupil Detection and Magnification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def calculate_magnification(pupil_size):
    return 10.0 / pupil_size if pupil_size > 0 else 1.0

def draw_magnification_graph(frame, magnification_levels):
    graph_height = 100
    graph_width = frame.shape[1]
    graph = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
    max_level = max(magnification_levels) if magnification_levels else 1

    for i, level in enumerate(magnification_levels[-graph_width:]):
        x = i
        y = int(graph_height - (level / max_level) * graph_height)
        cv2.line(graph, (x, graph_height), (x, y), (0, 255, 0), 1)

    frame[-graph_height:, :] = graph

if __name__ == "__main__":
    video_path = 'v2.mp4'
    detect_pupil_and_calculate_magnification(video_path)
