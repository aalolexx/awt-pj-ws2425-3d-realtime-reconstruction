using System;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Text;

public class PcdUpdaasdter : MonoBehaviour
{
    public ComputeShader computeShader;  // Optional for processing
    public GameObject pointPrefab;       // A simple sphere or cube for point rendering
    private List<GameObject> spawnedPoints = new List<GameObject>();  // Store references to instantiated points
    
    TcpClient client;
    NetworkStream stream;
    const int MAX_MESH_SIZE = 10 * 1024 * 1024;  // 10MB limit
    byte[] buffer = new byte[1024 * 1024];
    
    [Serializable]
    public class PointData
    {
        public float x, y, z;
    }

    [Serializable]
    public class PointCloudData
    {
        public List<PointData> points;
    }
    
    void Start()
    {
        client = new TcpClient("127.0.0.1", 8764);
        stream = client.GetStream();
        StartCoroutine(ReceivePcdData());
    }

    IEnumerator ReceivePcdData()
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

                string dataString = Encoding.UTF8.GetString(dataBuffer);
                
                try
                {
                    PointCloudData pointCloud = JsonUtility.FromJson<PointCloudData>(dataString);

                    //Debug.Log(pointCloud);

                    if (pointCloud != null && pointCloud.points.Count > 0)
                    {
                        // Convert PointData[] to Vector3[]
                        Vector3[] positions = new Vector3[pointCloud.points.Count];
                        for (int i = 0; i < pointCloud.points.Count; i++)
                        {
                            positions[i] = new Vector3(pointCloud.points[i].x, pointCloud.points[i].y,
                                pointCloud.points[i].z);
                        }

                        RenderPoints(positions);
                    }
                } catch (System.Exception e)
                {
                    Debug.LogError(e);
                }
            }
            yield return null;
        }
    }

    void RenderPoints(Vector3[] points)
    {
        // **Clear previous points before rendering new ones**
        ClearPreviousPoints();

        foreach (Vector3 pos in points)
        {
            Vector3 worldPos = new Vector3(pos.x, pos.y, pos.z);
            Vector3 localPos = transform.TransformPoint(worldPos); // Convert to local coordinates
            GameObject point = Instantiate(pointPrefab, localPos, Quaternion.identity);
            spawnedPoints.Add(point);
        }
    }

    void ClearPreviousPoints()
    {
        foreach (GameObject point in spawnedPoints)
        {
            Destroy(point);
        }
        spawnedPoints.Clear();
    }
}