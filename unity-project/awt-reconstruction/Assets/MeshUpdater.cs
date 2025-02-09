using System;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Linq;

public class MeshUpdater : MonoBehaviour
{
    private MeshFilter meshFilter;
    private MeshRenderer meshRenderer;
    
    TcpClient client;
    NetworkStream stream;
    const int MAX_MESH_SIZE = 10 * 1024 * 1024;  // 10MB limit
    byte[] buffer = new byte[1024 * 1024];

    void Start()
    {
        meshFilter = GetComponent<MeshFilter>();
        meshRenderer = GetComponent<MeshRenderer>();

        // Add a default material if no material is assigned
        if (meshRenderer == null)
        {
            meshRenderer = gameObject.AddComponent<MeshRenderer>();
            meshRenderer.material = new Material(Shader.Find("Standard"));
        }

        client = new TcpClient("127.0.0.1", 8763);
        stream = client.GetStream();
        StartCoroutine(ReceiveMeshData());
    }

    IEnumerator ReceiveMeshData()
    {
        while (true)
        {
            if (stream.DataAvailable)
            {
                byte[] lengthBuffer = new byte[4];  // Read the 4-byte length prefix
                int lengthBytesRead = stream.Read(lengthBuffer, 0, 4);
                
                if (lengthBytesRead < 4)
                {
                    Debug.LogError("Failed to read full length prefix.");
                    yield return null;
                    continue;
                }

                int messageLength = BitConverter.ToInt32(lengthBuffer.Reverse().ToArray(), 0); // Convert from Big-endian
                
                // Validate message size
                if (messageLength <= 0 || messageLength > MAX_MESH_SIZE)
                {
                    Debug.LogError($"Invalid message length: {messageLength}");
                    stream.Flush();  // Clear buffer to avoid sync issues
                    continue;
                }

                // Read full message
                int bytesRead = 0;
                byte[] dataBuffer = new byte[messageLength];

                while (bytesRead < messageLength)
                {
                    int read = stream.Read(dataBuffer, bytesRead, messageLength - bytesRead);
                    if (read == 0)
                    {
                        Debug.LogError("Connection closed while reading data.");
                        client.Close();
                        yield break;
                    }
                    bytesRead += read;
                }

                string objDataString = Encoding.UTF8.GetString(dataBuffer);
                Mesh mesh = ObjToMesh(objDataString);
                if (mesh != null)
                {
                    meshFilter.mesh = mesh;
                }
            }
            yield return null;
        }
    }

    Mesh ObjToMesh(string objData)
    {
        Mesh mesh = new Mesh();
        var vertices = new List<Vector3>();
        var normals = new List<Vector3>();
        var triangles = new List<int>();

        string[] lines = objData.Split(new char[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string line in lines)
        {
            if (line.StartsWith("#")) continue;

            if (line.StartsWith("v "))
            {
                string[] vertex = line.Split(' ');
                if (vertex.Length == 4)
                {
                    vertices.Add(new Vector3(float.Parse(vertex[1]), float.Parse(vertex[2]), float.Parse(vertex[3])));
                }
            }
            else if (line.StartsWith("vn "))
            {
                string[] normal = line.Split(' ');
                if (normal.Length == 4)
                {
                    normals.Add(new Vector3(float.Parse(normal[1]), float.Parse(normal[2]), float.Parse(normal[3])));
                }
            }
            else if (line.StartsWith("f "))
            {
                string[] face = line.Split(' ');
                if (face.Length == 4)
                {
                    int v1 = int.Parse(face[1].Split("//")[0]) - 1;
                    int v2 = int.Parse(face[2].Split("//")[0]) - 1;
                    int v3 = int.Parse(face[3].Split("//")[0]) - 1;

                    triangles.Add(v1);
                    triangles.Add(v2);
                    triangles.Add(v3);
                }
            }
        }

        if (vertices.Count == 0 || triangles.Count == 0)
        {
            Debug.LogError("Mesh has no vertices or faces.");
            return null;
        }

        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles.ToArray();
        mesh.RecalculateNormals();

        return mesh;
    }
}
