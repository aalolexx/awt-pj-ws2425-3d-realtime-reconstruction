using System;
using UnityEngine;
using System.Collections;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class MeshUpdater : MonoBehaviour
{
    private MeshFilter meshFilter;
    private MeshRenderer meshRenderer;
    
    TcpClient client;
    NetworkStream stream;
    byte[] buffer = new byte[2048 * 2048];
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

        client = new TcpClient("127.0.0.1", 65432);
        stream = client.GetStream();
        StartCoroutine(ReceiveMeshData());
    }

    IEnumerator ReceiveMeshData()
    {
        while (true)
        {
            if (stream.DataAvailable)
            {
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                string objDataString = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                Debug.Log(objDataString.Length);
                Mesh mesh = ObjToMesh(objDataString);
                if (mesh != null)
                {
                    meshFilter.mesh = mesh;  // Update the mesh in Unity
                    Debug.Log("Mesh updated.");
                }
            }
            yield return null;
        }
    }

    // Convert OBJ file content to a Unity Mesh
    // Convert OBJ file content to a Unity Mesh
    Mesh ObjToMesh(string objData)
    {
        Mesh mesh = new Mesh();
        var vertices = new System.Collections.Generic.List<Vector3>();
        var normals = new System.Collections.Generic.List<Vector3>();
        var triangles = new System.Collections.Generic.List<int>();

        // Split the objData into lines
        string[] lines = objData.Split(new char[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);

        foreach (string line in lines)
        {
            // Ignore comments (lines starting with '#')
            if (line.StartsWith("#"))
                continue;

            // Parse vertices (lines starting with 'v')
            if (line.StartsWith("v "))
            {
                try
                {
                    string[] vertex = line.Split(' ');
                    if (vertex.Length == 4)  // Ensure it's "v x y z"
                    {
                        float x = float.Parse(vertex[1]);
                        float y = float.Parse(vertex[2]);
                        float z = float.Parse(vertex[3]);
                        vertices.Add(new Vector3(x, y, z));
                    }
                }
                catch (FormatException e)
                {
                    Debug.LogError($"Error parsing vertex line: {line} | {e.Message}");
                }
            }
            // Parse vertex normals (lines starting with 'vn')
            else if (line.StartsWith("vn "))
            {
                try
                {
                    string[] normal = line.Split(' ');
                    if (normal.Length == 4)  // Ensure it's "vn x y z"
                    {
                        float nx = float.Parse(normal[1]);
                        float ny = float.Parse(normal[2]);
                        float nz = float.Parse(normal[3]);
                        normals.Add(new Vector3(nx, ny, nz));
                    }
                }
                catch (FormatException e)
                {
                    Debug.LogError($"Error parsing normal line: {line} | {e.Message}");
                }
            }
            // Parse faces (lines starting with 'f')
            else if (line.StartsWith("f "))
            {
                try
                {
                    string[] face = line.Split(' ');
                    if (face.Length == 4)  // Ensure it's "f v1//vn1 v2//vn2 v3//vn3"
                    {
                        // Extract vertex and normal indices separately
                        string[] v1Data = face[1].Split("//");
                        string[] v2Data = face[2].Split("//");
                        string[] v3Data = face[3].Split("//");

                        int v1 = int.Parse(v1Data[0]) - 1;  // Vertex index
                        int v2 = int.Parse(v2Data[0]) - 1;
                        int v3 = int.Parse(v3Data[0]) - 1;

                        int n1 = int.Parse(v1Data[1]) - 1;  // Normal index
                        int n2 = int.Parse(v2Data[1]) - 1;
                        int n3 = int.Parse(v3Data[1]) - 1;

                        triangles.Add(v1);
                        triangles.Add(v2);
                        triangles.Add(v3);

                        // Ensure normals are assigned per vertex
                        if (normals.Count > n1 && normals.Count > n2 && normals.Count > n3)
                        {
                            mesh.normals[v1] = normals[n1];
                            mesh.normals[v2] = normals[n2];
                            mesh.normals[v3] = normals[n3];
                        }
                    }
                }
                catch (FormatException e)
                {
                    Debug.LogError($"Error parsing face line: {line} | {e.Message}");
                }
            }
        }

        // Ensure that we have vertices and faces
        if (vertices.Count == 0 || triangles.Count == 0)
        {
            Debug.LogError("Mesh has no vertices or faces.");
            return null;  // If mesh data is incomplete, return null
        }

        // If no normals are provided, calculate them
        if (normals.Count == 0)
        {
            mesh.normals = new Vector3[vertices.Count];
            mesh.RecalculateNormals();  // Recalculate normals if they are missing
        }
        else if (normals.Count != vertices.Count)
        {
            Debug.LogWarning("Normals count does not match vertices count. Recalculating normals...");
            mesh.normals = new Vector3[vertices.Count];
            mesh.RecalculateNormals();  // Recalculate normals if the count doesn't match
        }
        else
        {
            mesh.normals = normals.ToArray();  // If normals match, assign them directly
        }

        // Assign vertices and triangles
        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles.ToArray();

        return mesh;
    }
}