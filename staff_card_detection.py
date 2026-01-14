import cv2
import numpy as np
import os

# --- PARAMETERS ---
reference_path = "staff_card.png"
video_path = "sample.mp4"
output_video_path = "staff_final_results.mp4"
output_folder = "detected_frames"

# Create folder for images if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Detection Settings
VERTICAL_WEIGHT = 0.85 
HORIZONTAL_WEIGHT = 0.15
EDGE_THRESHOLD = 30
SCALES = np.linspace(0.1, 0.4, 15) 
ANGLES = range(-10, 11, 2)
CONFIDENCE_THRESHOLD = 0.66 

# --- PREP REFERENCE ---
ref_img = cv2.imread(reference_path)
ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
ref_gray = cv2.equalizeHist(ref_gray)

def get_edges(img_gray):
    sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = VERTICAL_WEIGHT * np.abs(sx) + HORIZONTAL_WEIGHT * np.abs(sy)
    return cv2.convertScaleAbs(edges)

ref_edges = get_edges(ref_gray)
h_ref, w_ref = ref_edges.shape

# --- VIDEO SETUP ---
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Save as MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w_vid, h_vid))

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

print(f"Processing... Images will be saved in /{output_folder}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    target_edges = get_edges(gray)

    # Motion Detection
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_found_this_frame = False

    for cnt in contours:
        if cv2.contourArea(cnt) < 1500: continue
        
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        roi_edges = target_edges[y:y+h_box, x:x+w_box]

        best_val = -1
        best_match = None

        for scale in SCALES:
            nw, nh = int(w_ref * scale), int(h_ref * scale)
            if nh > h_box or nw > w_box or nh < 10: continue
            scaled_ref = cv2.resize(ref_edges, (nw, nh))

            for angle in ANGLES:
                M = cv2.getRotationMatrix2D((nw/2, nh/2), angle, 1)
                rotated_ref = cv2.warpAffine(scaled_ref, M, (nw, nh))
                res = cv2.matchTemplate(roi_edges, rotated_ref, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > best_val:
                    best_val = max_val
                    best_match = (max_loc, nw, nh)

        if best_val >= CONFIDENCE_THRESHOLD:
            card_found_this_frame = True
            mloc, mw, mh = best_match
            gx, gy = mloc[0] + x, mloc[1] + y
            
            # Draw rectangle and metadata
            cv2.rectangle(frame, (gx, gy), (gx + mw, gy + mh), (0, 255, 0), 2)
            cv2.putText(frame, f"F:{frame_id} Score:{best_val:.2f}", (gx, gy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 1. SAVE FRAME AS IMAGE if card was found
    if card_found_this_frame:
        img_name = os.path.join(output_folder, f"frame_{frame_id}.jpg")
        cv2.imwrite(img_name, frame)

    # 2. SAVE VIDEO
    out.write(frame)
    
    # Optional: Display frame number in corner for verification
    cv2.putText(frame, f"Frame: {frame_id}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Verification Mode", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done! Video saved as {output_video_path}")