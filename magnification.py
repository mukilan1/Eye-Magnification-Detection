
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Global variable for storing magnification levels
magnification_levels = []

def detect_pupil_and_visualize_magnification(video_path):
    global magnification_levels
    
    mp_face_mesh = mp.solutions.face_mesh

    # Initialize MediaPipe Face Mesh.
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0  # Initialize frame counter for saving screenshots

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect face mesh landmarks.
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            # Assume only one face is detected, so take the first face's landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract landmarks for the left eye only (assuming one eye in the video)
            left_eye_landmarks = [face_landmarks.landmark[i] for i in range(33, 42)]

            # Extract landmarks for the left eye
            left_eye_points = [(landmark.x, landmark.y) for landmark in left_eye_landmarks]

            # Calculate distance between eye landmarks (as an approximation of pupil size)
            if len(left_eye_points) == 9:
                left_eye_size = calculate_pupil_size(left_eye_points)

                # Calculate magnification rate based on pupil size
                left_eye_magnification = calculate_magnification(left_eye_size)

                # Store magnification level for visualization
                magnification_levels.append(left_eye_magnification)

                # Print and save magnification rate
                print(f"Left eye magnification rate: {left_eye_magnification:.2f}")

                # Save frame with magnification value
                save_screenshot(frame, left_eye_magnification, frame_count)
                frame_count += 1
            else:
                print("Not enough landmarks for left eye detected.")

        # Display the frame with magnification value.
        cv2.imshow('Pupil Magnification Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot the magnification levels graph
    plot_magnification_levels()

def calculate_pupil_size(points):
    # Calculate the distance between center and outer points of the pupil
    x_center, y_center = points[4]
    x_outer, y_outer = points[0]
    distance = ((x_outer - x_center) ** 2 + (y_outer - y_center) ** 2) ** 0.5
    return distance

def calculate_magnification(pupil_size):
    # Example function to calculate magnification rate based on pupil size
    if pupil_size <= 0:
        return 1.0  # Default magnification rate if size is invalid or zero
    else:
        return 10.0 / pupil_size  # Adjust based on calibration and testing

def save_screenshot(frame, magnification, frame_count):
    # Save frame as screenshot with magnification values
    filename = f"screenshot_{frame_count}.png"
    cv2.putText(frame, f"Magnification: {magnification:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(filename, frame)

def plot_magnification_levels():
    # Plot the magnification levels over time
    plt.figure(figsize=(10, 5))
    plt.plot(magnification_levels, label='Magnification Levels')
    plt.xlabel('Frame')
    plt.ylabel('Magnification Level')
    plt.title('Magnification Levels Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    video_path = 'mag.mp4'  # Replace with your video file path
    detect_pupil_and_visualize_magnification(video_path)


