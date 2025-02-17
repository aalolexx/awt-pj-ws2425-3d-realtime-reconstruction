import open3d as o3d
import numpy as np
import socket
import json
import selectors
import threading

from util.base_module import BaseModule


"""
Streams the provided mesh as a web socket server
"""


class MeshStreamer(BaseModule):
    def __init__(self, visualize=False, host='127.0.0.1', port=8763):
        """Initialize the non-blocking PCD socket server."""
        self.HOST = host
        self.PORT = port

        # Create selector for non-blocking I/O
        self._selector = selectors.DefaultSelector()

        # Create server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.bind((self.HOST, self.PORT))
        self._server_socket.listen()

        # Set server socket to non-blocking
        self._server_socket.setblocking(False)

        # Register server socket with selector
        self._selector.register(self._server_socket, selectors.EVENT_READ, self._accept_connection)

        self._client_socket = None
        print(f"MeshStreamer waiting for connection on {self.HOST}:{self.PORT}")

    #
    # Run Step
    #
    def run_step(self, points, faces):
        # Check for any socket events
        events = self._selector.select(timeout=0)

        for key, _ in events:
            callback = key.data
            callback(key.fileobj)

        # If we have a client socket and a mesh, send data
        if self._client_socket:
            thread = threading.Thread(target=self._send_data, args=(points, faces,))
            thread.start()


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

    def _send_data(self, points, faces):
        """Send data in a separate thread."""
        data = self.mesh_to_obj_string(points, faces)

        if self._client_socket:
            try:
                data_length = len(data).to_bytes(4, byteorder='big')
                self._client_socket.sendall(data_length + data.encode("UTF8"))
            except (ConnectionResetError, BrokenPipeError):
                # Handle client disconnection
                print("Client disconnected")
                self._client_socket.close()
                self._client_socket = None


    def mesh_to_obj_string(self, points, faces):
        obj_str = "# Created Mesh Reconstruction Pipeline\n"
        obj_str += "# object name: pipeline_mesh\n"
        obj_str += f"# number of vertices: {len(points)}\n"
        obj_str += f"# number of triangles: {len(faces)}\n"

        if (len(points) <= 0):
            print("ATTENTION: Streamed mesh has no vertices")

        for i, vertex in enumerate(points):
            obj_str += f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"  # Write vertices

        #for i, normal in enumerate(mesh.vertex_normals):
        #    obj_str += f"vn {normal[0]} {normal[1]} {normal[2]}\n"  # Write normals

        for triangle in faces:
            obj_str += f"f {triangle[0] + 1}//{triangle[0] + 1} {triangle[1] + 1}//{triangle[1] + 1} {triangle[2] + 1}//{triangle[2] + 1}\n"
            # Correct "f v1//vn1 v2//vn2 v3//vn3" format

        #print(obj_str)
#
        #print("-------")
        #print(len(mesh.vertices))
        #print(len(mesh.vertex_normals))

        return obj_str
