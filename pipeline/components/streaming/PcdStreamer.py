import open3d as o3d
import numpy as np
import socket
import json
import selectors
import types
import threading

from util.base_module import BaseModule


class PcdStreamer(BaseModule):
    def __init__(self, visualize=False, host='127.0.0.1', port=8764):
        """Initialize the non-blocking PCD socket server."""
        self.host = host
        self.port = port

        # Create selector for non-blocking I/O
        self._selector = selectors.DefaultSelector()

        # Create server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1)  # Allow only one client

        # Set server socket to non-blocking
        self._server_socket.setblocking(False)

        # Register server socket with selector
        self._selector.register(self._server_socket, selectors.EVENT_READ, self._accept_connection)

        # Track client socket
        self._client_socket = None

        print(f"PCD Streamer waiting for connection on {self.host}:{self.port}")

    def _accept_connection(self, sock):
        """Handle new incoming connection."""
        # If already have a client, reject new connections
        if self._client_socket is not None:
            try:
                conn, addr = sock.accept()
                conn.close()
            except:
                pass
            return

        # Accept new connection
        conn, addr = sock.accept()
        conn.setblocking(False)
        print(f"Connection from {addr}")
        self._client_socket = conn


    def run_step(self, pcd):
        """Non-blocking socket communication for point cloud data."""
        # Check for any socket events
        events = self._selector.select(timeout=0)

        for key, _ in events:
            callback = key.data
            callback(key.fileobj)

        # Prepare point cloud data
        points = []
        pcd_points = np.asarray(pcd.points)
        compression_factor = 5

        for i in range(0, len(pcd_points), compression_factor):
            p = pcd_points[i]
            points.append({
                "x": round(float(p[0]), 4),
                "y": round(float(p[1]), 4),
                "z": round(float(p[2]), 4)
            })

        # Convert to binary for efficient transmission
        message = json.dumps({"points": points}).encode('UTF8')

        # Send to client if connected
        # Send to client if connected in a separate thread
        if self._client_socket:
            thread = threading.Thread(target=self._send_data, args=(message,))
            thread.start()


    def _send_data(self, data):
        """Send data in a separate thread."""
        if self._client_socket:
            try:
                data_length = len(data).to_bytes(4, byteorder='big')
                self._client_socket.sendall(data_length + data)
            except (ConnectionResetError, BrokenPipeError):
                # Handle client disconnection
                print("Client disconnected")
                self._client_socket.close()
                self._client_socket = None