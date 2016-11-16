// Copyright (c) 2016, Long Qian
// All rights reserved.

using UnityEngine;
using System.Runtime.InteropServices;
using System;

public class arucoDisplay : MonoBehaviour {

    // Original Video parameters
    public int deviceNumber;
    private WebCamTexture _webcamTexture;
    private const int imWidth = 1280;
    private const int imHeight = 720;


    // Processed Video parameters
    public MeshRenderer ProcessedTextureRenderer;
    private Texture2D processedTexture;
    byte[] processedImageData;


    public TextMesh DebugTextMesh;



    private int detectCount = 0;
    private int frameCount = 0;



    // ARUCO native functions
    [DllImport("aruco_core")]
    public static extern void initArucoController();
    [DllImport("aruco_core")]
    public static extern void destroyArucoController();
    [DllImport("aruco_core")]
    public static extern void newImage(IntPtr imageData);
    [DllImport("aruco_core")]
    public static extern void setImageSize(int row, int col);
    [DllImport("aruco_core")]
    public static extern void detect();
    [DllImport("aruco_core")]
    public static extern IntPtr getProcessedImage();
    [DllImport("aruco_core")]
    public static extern int getNumMarkers();
    [DllImport("aruco_core")]
    public static extern int getSize();
    [DllImport("aruco_core")]
    public static extern int getRows();
    [DllImport("aruco_core")]
    public static extern int getCols();

    // getInt is for debugging use
    [DllImport("aruco_core")]
    public static extern int getInt();



    void Start() {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length > 0) {

            _webcamTexture = new WebCamTexture(devices[deviceNumber].name, 1280, 720, 30);
            // Play the video source
            _webcamTexture.Play();
            
            processedTexture = new Texture2D(imWidth, imHeight, TextureFormat.RGBA32, false, true);
            ProcessedTextureRenderer.material.mainTexture = processedTexture;

            processedImageData = new byte[imHeight * imWidth * 4];

            initArucoController();
            setImageSize(imHeight, imWidth);
            Debug.Log("Image size: " + getRows() + ", " + getCols());

            DebugTextMesh.text = getRows() + ", " + getCols();
        }
    }



    void Update() {
        if (_webcamTexture.isPlaying) {
            if (_webcamTexture.didUpdateThisFrame) {

                // Send video capture to ARUCO controller
                Color32[] c = _webcamTexture.GetPixels32();
                IntPtr imageHandle = getImageHandle(c);
                newImage(imageHandle);

                // Marker detect
                detect();

                //// Fetch the processed image and render
                imageHandle = getProcessedImage();
                Marshal.Copy(imageHandle, processedImageData, 0, imWidth * imHeight * 4);
                processedTexture.LoadRawTextureData(processedImageData);
                processedTexture.Apply();

                DebugTextMesh.text = "       Markers: " + getNumMarkers();

                // Frame rate notification
                detectCount++;
                if (detectCount % 30 == 0) {
                    Debug.Log("Number of markers: " + getNumMarkers());
                    //DebugTextMesh.text = "Number of markers: " + getNumMarkers();
                }
            }
        }
        else {
            Debug.Log("Can't find camera!");
        }
    }

    private static IntPtr getImageHandle(Color32[] colors) {
        IntPtr ptr;
        GCHandle handle = default(GCHandle);
        try {
            handle = GCHandle.Alloc(colors, GCHandleType.Pinned);
            ptr = handle.AddrOfPinnedObject();
        }
        finally {
            if (handle != default(GCHandle))
                handle.Free();
        }
        return ptr;
    }


    public void OnDestroy() {
        destroyArucoController();
    }


}