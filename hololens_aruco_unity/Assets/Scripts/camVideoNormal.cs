using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class camVideoNormal : MonoBehaviour {

    public MeshRenderer UseWebcamTexture;
    public int deviceNumber;
    private WebCamTexture webcamTexture;

    void Start() {
        if (Application.platform == RuntimePlatform.WindowsEditor)
            deviceNumber = 1;

        WebCamDevice[] devices = WebCamTexture.devices;

        if (deviceNumber < devices.Length ) {
            webcamTexture = new WebCamTexture(devices[deviceNumber].name, 1280, 720, 30 );
        }
        else {
            Debug.Log("No device with specified deviceNumber found");
        }
        Debug.Log("Device name: " + webcamTexture.deviceName);
        
        UseWebcamTexture.material.mainTexture = webcamTexture;
        webcamTexture.Play();
    }

    void OnGUI() {
        if (webcamTexture.isPlaying) {
            if (GUILayout.Button("Pause")) {
                webcamTexture.Pause();
            }
            if (GUILayout.Button("Stop")) {
                webcamTexture.Stop();
            }
        }
        else {
            if (GUILayout.Button("Play")) {
                webcamTexture.Play();
            }
        }
    }

    void Update() {
        ;
    }
}
