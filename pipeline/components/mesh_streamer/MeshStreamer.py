import open3d as o3d
import numpy as np
import socket
import json

from util.base_module import BaseModule


"""
Streams the provided mesh as a web socket server
"""


class MeshStreamer(BaseModule):
    def __init__(self, visualize=False):
        """Initialize the PointCloudReconstructor."""
        self._visualize = visualize

        HOST = '127.0.0.1'
        PORT = 65432
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind((HOST, PORT))

        print("INIT MESH STREAMER: Waiting for a socket to connect")

        self._socket.listen(1)
        self._conn, addr = self._socket.accept()

        print("CONNECTED TO SOCKET")

    #
    # Run Step
    #
    def run_step(self, mesh):
        # Extract vertices and faces
        mesh_data = self.mesh_to_obj_string(mesh)
        self._conn.sendall(mesh_data.encode("UTF8"))


    def mesh_to_obj_string(self, mesh):
        obj_str = "# Created Mesh Reconstruction Pipeline\n"
        obj_str += "# object name: pipeline_mesh\n"
        obj_str += f"# number of vertices: {len(mesh.vertices)}\n"
        obj_str += f"# number of triangles: {len(mesh.triangles)}\n"

        if (len(mesh.vertices) <= 0):
            print("ATTENTION: Streamed mesh has no vertices")

        for i, vertex in enumerate(mesh.vertices):
            obj_str += f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"  # Write vertices

        for i, normal in enumerate(mesh.vertex_normals):
            obj_str += f"vn {normal[0]} {normal[1]} {normal[2]}\n"  # Write normals

        for triangle in mesh.triangles:
            obj_str += f"f {triangle[0] + 1}//{triangle[0] + 1} {triangle[1] + 1}//{triangle[1] + 1} {triangle[2] + 1}//{triangle[2] + 1}\n"
            # Correct "f v1//vn1 v2//vn2 v3//vn3" format

        #print(obj_str)
#
        #print("-------")
        #print(len(mesh.vertices))
        #print(len(mesh.vertex_normals))

        return obj_str
