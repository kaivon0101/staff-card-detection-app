# Staff Card Detection in Video
This project automatically detects a staff name card in video frames using template matching and edge detection, even when the staff card is slanted, scaled, or partially blurred. Detected frames are saved as images, and a final annotated video is generated.

## Features
- Detect staff name card using a reference image.
- Robust to different scales and small rotations.
- Uses motion detection to limit search area for efficiency.
- Outputs:
    - Annotated frames saved as images (detected_frames/)
    - Annotated video (staff_final_results.mp4)
- Grayscale + edge-based detection for better contrast and robustness.

## Files
- staff_card.png → Reference image of the staff card.
- sample.mp4 → Input video with staff walking along the corridor.
- staff_final_results.mp4 → Output video with staff card detection annotated.
- detected_frames/ → Folder containing frames where the staff card was detected.
- detect_staff.py → Main Python script for detection.

## Requirements
- Python 3.8+
- OpenCV
- NumPy

## How It Works
1. Load Reference Image
- Convert the staff card reference image to grayscale.
- Perform edge detection emphasizing vertical edges (matches the piano-key design).

2. Video Processing
- Open video file and read frame by frame.
- Convert frames to grayscale and extract edges.
- Use background subtraction to identify moving objects (likely the staff).

3. Template Matching
- For each moving object, perform multi-scale and multi-angle template matching with the reference edges.
- If similarity exceeds a threshold, mark the region as the staff card.

4. Output
- Draw green rectangle around detected staff card.
- Save frame images and write annotated video.

## Usage
- Run the detection script:
    - python detect_staff.py
- Video will be processed frame by frame.
- Annotated video will be saved as staff_final_results.mp4.
- Detected frames are saved in detected_frames/.
- Press q to stop the live verification window early.