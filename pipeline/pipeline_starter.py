"""
Initialize Pipeline Components#
"""
import cv2
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import box
import time
import numpy as np
import open3d as o3d

from components.pc_generator.DepthEstimator import DepthEstimator
from components.pc_generator.PointCloudGenerator import PointCloudGenerator
from components.pc_generator.ForegroundExtractor_RMBG import ForegroundExtractor
from components.pc_generator.DepthThresholder import DepthThresholder
from components.pc_generator.DepthSegmenter import DepthSegmenter
from components.pc_reconstructor.PointCloudReconstructor import PointCloudReconstructor
from components.mesh_generator.MeshGenerator import MeshGenerator
from components.streaming.MeshStreamer import MeshStreamer
from components.streaming.PcdStreamer import PcdStreamer


###
# PIPELINE RUN MODES
###
is_highperformance_mode = True  # Use RMBG Model or Fast Segmentation Thresholding
is_live_stream_mode = False  # Use live stream from webcam source or recording5.mp4


###
# Initialize the Pipeline Modules
###
depth_estimator = DepthEstimator(visualize=False)
if is_highperformance_mode:
    foreground_masker = DepthThresholder(visualize=True)
    #  depth_segmenter = DepthSegmenter(visualize=True)  contour based approach
else:
    foreground_masker = ForegroundExtractor(visualize=True)

pointcloud_generator = PointCloudGenerator(visualize=True)
pointcloud_reconstructor = PointCloudReconstructor(
                            model_name="UnetVoxelAutoEncoder",
                            checkpoint_name="unet_auto_encoder.pth",
                            visualize=False
                           )
mesh_generator = MeshGenerator(visualize=True, approach='marching')
mesh_streamer = MeshStreamer(visualize=False)
pcd_streamer = PcdStreamer(visualize=False)

# Prepare Video Stream
if is_live_stream_mode:
    cap = cv2.VideoCapture(0)
else:
    video_path = "../demo-material/recordings/recording5.mp4"
    cap = cv2.VideoCapture(video_path)

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
    table.add_row("[bold cyan]Mesh Generation[/bold cyan]",
                  f"[bold green]{data[5]:.4f}[/bold green]",
                  f"{'🤖' if active_module_idx == 5 else ''}"
                  )
    table.add_row("[bold cyan]PCD Streaming[/bold cyan]",
                  f"[bold green]{data[6]:.4f}[/bold green]",
                  f"{'🤖' if active_module_idx == 6 else ''}"
                  )
    table.add_row("[bold cyan]Mesh Streaming[/bold cyan]",
                  f"[bold green]{data[7]:.4f}[/bold green]",
                  f"{'🤖' if active_module_idx == 7 else ''}"
                  )
    table.add_row("[bold cyan]Total Time[/bold cyan]",
                  f"[bold green]{data[8]:.4f}[/bold green]",
                  f"{'🤖' if active_module_idx == 8 else ''}"
                  )
    return table


def is_foreground_ok(foreground_mask):
    white_pixels = np.sum(foreground_mask == 255)
    total_pixels = foreground_mask.size  # Total number of pixels in the mask
    white_percentage = (white_pixels / total_pixels) * 100
    return white_percentage > 5  # 5 percent


def is_depth_ok(depth_image):
    return not np.all(depth_image == 0)

###
# Run Pipeline
###
def run_pipeline():
    """Continuously capture frames from webcam and process them."""

    # Initialize the console
    console = Console()

    time_per_module = [0, 0, 0, 0, 0, 0, 0, 0, 0]
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
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.imshow('original', frame)

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
            if is_highperformance_mode:
                foreground_mask = foreground_masker.run_step(depth_image)
            else:
                foreground_mask = foreground_masker.run_step(frame)

            elapsed_time = time.perf_counter() - start_time
            time_per_module[2] = elapsed_time * 1000

            live.update(create_console_pipeline(time_per_module, 3))

            # CHECKER -> If foreground mask is broken, don't update the 3d pcd
            if not is_foreground_ok(foreground_mask):
                time_per_module[3] = 0
                time_per_module[4] = 0
                continue

            # Incomplete PCD Estimation
            start_time = time.perf_counter()
            incomplete_pcd = pointcloud_generator.run_step(foreground_mask, depth_image)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[3] = elapsed_time * 1000

            live.update(create_console_pipeline(time_per_module, 4))

            # PCD Reconstruction
            start_time = time.perf_counter()
            reconstructed_pcd, scaling_factor = pointcloud_reconstructor.run_step(incomplete_pcd)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[4] = elapsed_time * 1000

            # Mesh Generation
            start_time = time.perf_counter()
            reconstructed_mesh = mesh_generator.run_step(reconstructed_pcd, scaling_factor)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[5] = elapsed_time * 1000

            live.update(create_console_pipeline(time_per_module, 5))

            # PCD Streaming
            start_time = time.perf_counter()
            rescaled_reconstructed_pcd = pointcloud_reconstructor.reverse_scale_of_point_cloud(
                reconstructed_pcd,
                scaling_factor
            )
            pcd_streamer.run_step(rescaled_reconstructed_pcd)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[6] = elapsed_time * 1000

            live.update(create_console_pipeline(time_per_module, 6))

            # Mesh Streaming
            start_time = time.perf_counter()
            mesh_streamer.run_step(reconstructed_mesh)
            elapsed_time = time.perf_counter() - start_time
            time_per_module[7] = elapsed_time * 1000

            live.update(create_console_pipeline(time_per_module, 7))

            # calculate total time for a frame 
            time_per_module[8] = 0
            total_time = sum(time_per_module)
            time_per_module[8] = total_time
            live.update(create_console_pipeline(time_per_module, 8))
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture when done
        cap.release()


if __name__ == '__main__':
    run_pipeline()