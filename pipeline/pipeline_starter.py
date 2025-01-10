"""
Initialize Pipeline Components#
"""
import cv2
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import box
import time

from components.pc_generator.DepthEstimator import DepthEstimator
from components.pc_generator.PointCloudGenerator import PointCloudGenerator

depth_estimator = DepthEstimator(visualize=False)
pointcloud_generator = PointCloudGenerator(visualize=True)

# Prepare Webcam
cap = cv2.VideoCapture(0)

"""
 CONSOLE PRINTING FUNCTIONS
"""
def create_console_pipeline(data):
    """Create a table representing the pipeline with 3 boxes."""
    table = Table(title="Pipeline", box=box.ROUNDED)
    table.add_column("Stage", justify="left")
    table.add_column("Value (MS)", justify="left")
    table.add_row("[bold cyan]Frame Captured[/bold cyan]", f"[bold green]{data[0]:.4f}[/bold green]")
    table.add_row("[bold cyan]Depth Estimator[/bold cyan]", f"[bold green]{data[1]:.4f}[/bold green]")
    table.add_row("[bold cyan]PCD Generator[/bold cyan]", f"[bold green]{data[2]:.4f}[/bold green]")
    return table

"""
Run Pipeline
"""

def run_pipeline():
    """Continuously capture frames from webcam and process them."""

    # Initialize the console
    console = Console()

    time_per_module = [0, 0, 0]
    with Live(create_console_pipeline(time_per_module), refresh_per_second=4, console=console) as live: # for console updates
        while True:
            # ------------------------------
            # Run actual Pipeline

            # Capture frame-by-frame
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[0] = elapsed_time * 1000

            # Depth Estimation
            start_time = time.perf_counter()
            depth_image = depth_estimator.run_step(frame)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[1] = elapsed_time * 1000

            # Incomplete PCD Estimation
            start_time = time.perf_counter()
            pcd = pointcloud_generator.run_step(depth_image)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[2] = elapsed_time * 1000

            # Update Console Interface
            live.update(create_console_pipeline(time_per_module))

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture when done
        cap.release()


if __name__ == '__main__':
    run_pipeline()