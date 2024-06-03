# 3D Reconstruction from Stereo Images

This project demonstrates a complete pipeline for 3D reconstruction from a pair of stereo images.  

The main steps include:
- computing the fundamental matrix,
- camera pose estimation,
- triangulation,
- pose disambiguation,
- image rectification, and
- generating a disparity map.

  
The workflow is implemented in Python with the help of OpenCV, NumPy, and Matplotlib.

## Results

### Input Image Pair

<table>
  <tr>
    <th>LEFT</th>
    <th>RIGHT</th>
  </tr>
  <tr>
    <td><img src="left.bmp" width="400"/></td>
    <td><img src="right.bmp" width="400"/></td>
  </tr>
</table>

### SIFT Feature Matches

<img src = "https://github.com/mohitydv09/stereo-reconstruction-using-SIFT-features/assets/101336175/07afbda3-3d0a-4547-98f0-d1ec12a46ea1" width=850 />

### Epipolar Lines

<img src = "https://github.com/mohitydv09/stereo-reconstruction-using-SIFT-features/assets/101336175/d21a6852-0db3-43f5-a10f-2bf1bd3ad6c5" width=850 />

### Camera Pose Disambuigy Correction

<table>
  <tr>
    <th>Camera Poses</th>
    <th>3D Points</th>
  </tr>
  <tr>
    <td><img src="https://github.com/mohitydv09/stereo-reconstruction-using-SIFT-features/assets/101336175/0a92e7eb-6b0f-473f-b293-860a066d5ef4" width="400"/></td>
    <td><img src="https://github.com/mohitydv09/stereo-reconstruction-using-SIFT-features/assets/101336175/81eebbe9-d827-4b3b-9ce2-587ec2bf10ce" width="400"/></td>
  </tr>
</table>


### Rectified Images

<img src = "https://github.com/mohitydv09/stereo-reconstruction-using-SIFT-features/assets/101336175/10cc974d-6946-45ca-aab4-ec7326c9a5c3" width=850 />

### Reconstructed Depth Image

Blue represents points closer to the camera, while red represents farther away points.
<img src = "https://github.com/mohitydv09/stereo-reconstruction-using-SIFT-features/assets/101336175/9d753781-08e4-4752-a4ab-05177c366cb1" width=850 />

Note :  I used precomputed SIFT features in this code.

