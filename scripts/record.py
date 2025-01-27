import cv2
import time



def record(duration_seconds = 7):
# Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Define the codec and create a VideoWriter object
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("recording.avi", fourcc, 20.0, (frame_width, frame_height))

    # Countdown timer
    print("Recording will start in:")
    for i in range(5, 0, -1):
        print(i)
        time.sleep(1)

    print("Recording...")

    # Record for duration_seconds
    start_time = time.time()
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Write the frame to the output file
        out.write(frame)

        # Display the frame
        cv2.imshow('Recording', frame)

        # Break the loop after 30 seconds or if 'q' is pressed
        if time.time() - start_time > duration_seconds or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Recording stopped and saved as 'recording.avi'")

if __name__ == "__main__":
    record()