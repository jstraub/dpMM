
### Dependencies

This code depends on the following other libraries and was tested under Ubuntu
14.04. 
- Eigen3 (3.0.5) 
- Boost (1.52)

### Library
*libcudaPcl.so* collects all the cuda code into one shared library. The rest
of the code is in the form of header files.

### Executables
- *openniSmoothNormals*: grab RGB-D frames from an openni device, smooth the
  depth image using a fast GPU enhanced guided filter and extract surface normals.
```
  Allowed options:
    -h [ --help ]         produce help message
    -f [ --f_d ] arg      focal length of depth camera
    -e [ --eps ] arg      sqrt of the epsilon parameter of the guided filter
    -b [ --B ] arg        guided filter windows size (size will be (2B+1)x(2B+1))
    -c [ --compress ]     compress the computed normals
```
