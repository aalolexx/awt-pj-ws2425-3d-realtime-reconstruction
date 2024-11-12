import cv2
import time

from DepthEstimator import DepthEstimator


class PointCloudGenerator:
    def __init__(self):
        """Initialize the PointCloudGenerator with webcam capture."""
        
        self.depth_estimator = DepthEstimator()
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise Exception("Could not open webcam")


    def generate(self):
        """Continuously capture frames from webcam and process them."""
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
                
            self.process_frame(frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                            
        # Release the capture when done
        self.cap.release()
        cv2.destroyAllWindows()
        

    def process_frame(self, frame):
        """Process a single frame from the webcam.
        This method can be overridden to implement custom frame processing.
        """

        depth = self.depth_estimator.predict(frame)

        # Display the frame
        cv2.imshow('Webcam', depth)



def test_component():
    """Main function to be executed from the terminal."""
    generator = PointCloudGenerator()
    generator.generate()

if __name__ == "__main__":
    test_component()
