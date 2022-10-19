# Stereo Generation
This repository contains code to generate stereo image from single view image captured by camera .
### Requirment
* [libtorch 10.2 (Release)](https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-1.8.1%2Bcu102.zip)
* [libtorch 10.2 (Debug)](https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-debug-1.8.1%2Bcu102.zip)
* [opencv 4.5.0](https://sourceforge.net/projects/opencvlibrary/files/4.5.0/opencv-4.5.0-vc14_vc15.exe/download)

### Architecture
`External`: Place requirment (libtorch & opencv 4.5.0)

`model`: 
```
wget https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small-traced.pt -O weights/model-small-traced.pt
```
