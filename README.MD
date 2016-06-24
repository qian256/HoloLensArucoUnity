HoloLens with Aruco
===
Author: Long Qian

Date: 2016-06-10

## Installation Guide

### CMake for Windows 10
Follow [this](https://github.com/Microsoft/CMake/tree/feature/Win10) to build CMake tool with support for Windows 10.

### OpenCV for UWP
* Follow [this](https://blogs.msdn.microsoft.com/lucian/2015/11/27/opencv-building-uwp-binaries/) to build OpenCV binaries for Windows UWP.
* The "Long way" is recommended. ```opencv_uwp``` folder contains the opencv binary I built for Windows UWP (x86 only, since HoloLens run on x86).
* Set environment ```OpenCV_DIR``` to the your folder of ```opencv_uwp```.

### Aruco Wrapper
[Aruco](http://www.uco.es/investiga/grupos/ava/node/26) is an open source marker tracking application based on OpenCV. ```aruco_core/aruco_core.cpp``` is a modification of ```aruco_simple``` example.

* Use the CMake tool for UWP to build the VS solution for the wrapper, navigate to the build folder, and execute: ```Path\To\CMake\bin\Release\cmake.exe -G "Visual Studio 14 2015" -DCMAKE_SYSTEM_NAME:String=WindowsStore -DCMAKE_SYSTEM_VERSION:String="10.0" -DCMAKE_VS_EFFECTIVE_PLATFORMS:String=x86 ..```
* Open the VS solution and configure the project so that "/ZW" option is activated (Consume Windows Runtime Extension).
* Build the dll.

### Unity3D for HoloLens
* Put the ```aruco_core.dll``` and necessary OpenCV dlls to the ```Plugins/WSA/x86/``` folder.
* Export to HoleLens and run!

## Issues
* Frame rate is lowered since C# is transfering raw image data to aruco_core.dll. This could be improved by compressing the image on both sides (Unity script and aruco_core).
