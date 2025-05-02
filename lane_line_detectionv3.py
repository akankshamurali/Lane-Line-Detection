import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Load pre-trained model from GitHub or TorchHub (simplified)
model = torch.hub.load('qfgaohao/Ultra-Fast-Lane-Detection', 'ultra_fast_lane_detection_culane', pretrained=True)
model.eval()

# Transform for model input
transform = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def run_on_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(input_img).unsqueeze(0)

        with torch.no_grad():
            out = model(input_tensor)[0].cpu().numpy()

        # Post-process: threshold + overlay on original frame
        lane_mask = (out > 0.3).astype(np.uint8)[0] * 255
        lane_mask = cv2.resize(lane_mask, (frame.shape[1], frame.shape[0]))
        colored = cv2.applyColorMap(lane_mask, cv2.COLORMAP_JET)
        combined = cv2.addWeighted(frame, 0.8, colored, 0.6, 0)

        cv2.imshow('Lane Detection (ML)', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

run_on_video("test_video.mp4")
