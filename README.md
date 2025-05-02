This project implements three different approaches to lane line detection in road-driving videos:
1.Classical (Edge + Hough)
2.Enhanced Classical (Color-coded lanes + Optimization + Output saving)
3.ML-Based (Using pre-trained segmentation model)

1️.Classical Lane Detection
File: legacy
Uses Canny edge detection + Hough Transform
Basic region-of-interest masking
Works directly on video
Both lanes are drawn in the same color

2.  Enhanced Lane Detection
File: lane_line_detectionv2.py
Adds:
Color-coded lanes (Left = Red, Right = Green)
Frame skipping for performance
Optional output video saving

3.ML-Based Lane Detection
File: lane_line_detectionv3.py
Uses a pre-trained deep learning model (e.g., Ultra-Fast Lane Detection)
Produces a segmentation mask of lanes
Overlays colored heatmap on original video
Requires PyTorch and a pre-trained model checkpoint
Requirements : pip install torch torchvision opencv-python

Future Improvements
Add real-time webcam support
Train on custom lane datasets
Export results as lane coordinates for control tasks
