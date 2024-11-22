import cv2
import time
import numpy as np
import open3d as o3d

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
            #start_time = time.time()

            # Capture frame-by-frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
                
            self.process_frame(frame)
            #end_time = time.time()
            #print(f"Depth prediction took: {end_time - start_time} seconds")
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                            
        # Release the capture when done
        self.cap.release()
        cv2.destroyAllWindows()


    def create_point_cloud(self, image_rgb, image_depth):
        width = np.shape(image_depth)[1]
        height = np.shape(image_depth)[0]

        # Convert depth to 3D points without perspective scaling
        depth = np.asarray(image_depth)

        # Generate a 3D point grid
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to 3D coordinates
        x = u 
        y = v
        z = depth * 1000 # Maintain straight-line scaling without perspective adjustment

        # Stack and filter valid points
        points = np.dstack((x, y, z)).reshape(-1, 3)
        valid_points = points[depth.reshape(-1) > 0]  # Remove zero-depth points

        # Create an Open3D point cloud from the resulting 3D points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)

        # Set the colors from the RGB image
        rgb = np.asarray(image_rgb).reshape(-1, 3) / 255.0  # Normalize to [0, 1]
        pcd.colors = o3d.utility.Vector3dVector(rgb[depth.reshape(-1) > 0])

        # Flip the point cloud (optional, depending on the coordinate system)
        #pcd.transform([[-1, 0, 0, 0],
        #            [0, -1, 0, 0],
        #            [0, 0, 1, 0],
        #            [0, 0, 0, 1]])
        
        # Visualize the point cloud
        #o3d.visualization.draw_geometries([pcd])
        

    def process_frame(self, frame):
        """Process a single frame from the webcam.
        This method can be overridden to implement custom frame processing.
        """

        depth = self.depth_estimator.predict(frame)
        self.create_point_cloud(frame, depth)
        print("frame")


        # Display the frame
        cv2.imshow('Webcam', depth)



def test_component():
    """Main function to be executed from the terminal."""
    generator = PointCloudGenerator()
    generator.generate()

if __name__ == "__main__":
    test_component()
