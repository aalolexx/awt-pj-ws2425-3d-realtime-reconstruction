"""
Initialize Pipeline Components#
"""
import cv2
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import box
import time
import open3d as o3d

from components.pc_generator.DepthEstimator import DepthEstimator
from components.pc_generator.PointCloudGenerator import PointCloudGenerator
from components.pc_generator.ForegroundExtractor import ForegroundExtractor
from components.pc_reconstructor.PointCloudReconstructor import PointCloudReconstructor

# Initialize the Pipeline Modules
depth_estimator = DepthEstimator(visualize=False)
foreground_extractor = ForegroundExtractor(visualize=False)
pointcloud_generator = PointCloudGenerator(visualize=False)
pointcloud_reconstructor = PointCloudReconstructor(
                            model_name="VoxelAutoEncoder",
                            checkpoint_name="voxel_weights_6.pth",
                            visualize=True
                           )

# Prepare Webcam
cap = cv2.VideoCapture(0)


###
# CONSOLE PRINTING FUNCTIONS
###
def create_console_pipeline(data, active_module_idx):
    """Create a table representing the pipeline with 3 boxes."""
    table = Table(title="Pipeline", box=box.ROUNDED)
    table.add_column("Module", justify="left")
    table.add_column("Time (MS)", justify="left")
    table.add_row("[bold cyan]Frame Captured[/bold cyan]",
                  f"[bold green]{data[0]:.4f}[/bold green]",
                  f"{'🤖' if active_module_idx == 0 else ''}"
                  )
    table.add_row("[bold cyan]Depth Estimator[/bold cyan]",
                  f"[bold green]{data[1]:.4f}[/bold green]",
                  f"{'🤖' if active_module_idx == 1 else ''}"
                  )
    table.add_row("[bold cyan]Foreground Extractor[/bold cyan]",
                  f"[bold green]{data[2]:.4f}[/bold green]",
                  f"{'🤖' if active_module_idx == 2 else ''}"
                  )
    table.add_row("[bold cyan]PCD Generator[/bold cyan]",
                  f"[bold green]{data[3]:.4f}[/bold green]",
                  f"{'🤖' if active_module_idx == 3 else ''}"
                  )
    table.add_row("[bold cyan]ML Reconstruction[/bold cyan]",
                  f"[bold green]{data[4]:.4f}[/bold green]",
                  f"{'🤖' if active_module_idx == 4 else ''}"
                  )
    return table


###
# Run Pipeline
###

def run_pipeline():
    """Continuously capture frames from webcam and process them."""

    # Initialize the console
    console = Console()

    time_per_module = [0, 0, 0, 0, 0]
    count = 0

    with Live(create_console_pipeline(time_per_module, 0), refresh_per_second=4, console=console) as live: # for console updates
        while True:
            count += 1
            # ------------------------------
            # Run actual Pipeline

            live.update(create_console_pipeline(time_per_module, 0))

            # Capture frame-by-frame
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[0] = elapsed_time * 1000

            live.update(create_console_pipeline(time_per_module, 1))

            # Depth Estimation
            start_time = time.perf_counter()
            depth_image = depth_estimator.run_step(frame)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[1] = elapsed_time * 1000

            live.update(create_console_pipeline(time_per_module, 2))

            # Foreground Extraction Model
            start_time = time.perf_counter()
            foreground_mask = foreground_extractor.run_step(frame)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[2] = elapsed_time * 1000

            live.update(create_console_pipeline(time_per_module, 3))

            # Incomplete PCD Estimation
            start_time = time.perf_counter()
            incomplete_pcd = pointcloud_generator.run_step(foreground_mask, depth_image)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[3] = elapsed_time * 1000

            live.update(create_console_pipeline(time_per_module, 4))

            # Incomplete PCD Estimation
            start_time = time.perf_counter()
            reconstructed_pcd = pointcloud_reconstructor.run_step(incomplete_pcd)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[4] = elapsed_time * 1000

            #if count % 3 == 0 and reconstructed_pcd is not None:
            #    o3d.io.write_point_cloud(f"PCD_ESTIMATED_FRAME_{count}.ply", reconstructed_pcd)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture when done
        cap.release()


if __name__ == '__main__':
    run_pipeline()