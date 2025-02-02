using UnityEngine;
using System.IO;
using System.Text;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class BakeMeshExporter : MonoBehaviour
{
  public SkinnedMeshRenderer skinnedMeshRenderer;

  void Update()
  {
    if (Application.isPlaying && Input.GetKeyDown(KeyCode.B)) // Works in Play Mode
    {
      BakeMesh();
    }
  }

#if UNITY_EDITOR
  [ContextMenu("Bake and Export Mesh")]  // Right-click option in Editor
  void BakeMeshEditor()
  {
    if (skinnedMeshRenderer == null)
    {
      Debug.LogError("SkinnedMeshRenderer not assigned!");
      return;
    }

    BakeMesh();
  }
#endif

  void BakeMesh()
  {
    if (skinnedMeshRenderer == null)
    {
      Debug.LogError("No SkinnedMeshRenderer found!");
      return;
    }

    Mesh bakedMesh = new Mesh();
    skinnedMeshRenderer.BakeMesh(bakedMesh);
    Debug.Log("Mesh baked successfully!");

    string exportPath = GetUniqueFilePath();
    ExportMeshToOBJ(bakedMesh, exportPath);
  }

  string GetUniqueFilePath()
  {
    string folderPath = Application.dataPath + "/Meshes";

    // Ensure the folder exists
    if (!Directory.Exists(folderPath))
    {
      Directory.CreateDirectory(folderPath);
    }

    int index = 0;
    string filePath;

    // Find the first available file name
    do
    {
      filePath = Path.Combine(folderPath, index.ToString() + ".obj");
      index++;
    } while (File.Exists(filePath));

    return filePath;
  }

  void ExportMeshToOBJ(Mesh mesh, string filePath)
  {
    if (mesh == null)
    {
      Debug.LogError("No mesh to export.");
      return;
    }

    StringBuilder sb = new StringBuilder();
    sb.Append("g ").Append(mesh.name).Append("\n");

    Vector3[] vertices = mesh.vertices;
    Vector3[] normals = mesh.normals;
    Vector2[] uvs = mesh.uv;
    int[] triangles = mesh.triangles;

    // Write vertices
    for (int i = 0; i < vertices.Length; i++)
    {
      sb.AppendFormat("v {0} {1} {2}\n", vertices[i].x, vertices[i].y, vertices[i].z);
    }
    sb.Append("\n");

    // Write normals (if available)
    if (normals.Length > 0)
    {
      for (int i = 0; i < normals.Length; i++)
      {
        sb.AppendFormat("vn {0} {1} {2}\n", normals[i].x, normals[i].y, normals[i].z);
      }
      sb.Append("\n");
    }

    // Write UVs (if available)
    if (uvs.Length > 0)
    {
      for (int i = 0; i < uvs.Length; i++)
      {
        sb.AppendFormat("vt {0} {1}\n", uvs[i].x, uvs[i].y);
      }
      sb.Append("\n");
    }

    // Write faces (fixing 1-based OBJ format)
    for (int i = 0; i < triangles.Length; i += 3)
    {
      sb.AppendFormat("f {0} {1} {2}\n",
          triangles[i] + 1, triangles[i + 1] + 1, triangles[i + 2] + 1); // Convert to 1-based index
    }

    // Write everything to file at once
    File.WriteAllText(filePath, sb.ToString());

    Debug.Log($"Mesh exported to {filePath}");
  }
}