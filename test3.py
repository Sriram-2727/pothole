from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO("C:\\Users\\91988\\Desktop\\yolov8-roadpothole-detection-main\\latest.pt")
class_names = model.names
cap = cv2.VideoCapture(0)
snap_count = 0  # Counter for saved images

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    img_resized = cv2.resize(img, (1020, 500))
    
    cv2.imshow('Live Feed', img_resized)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit if 'q' is pressed
        break
    elif key == ord('s'):  # Save snapshot if 's' is pressed
        snap_count += 1
        snap_filename = f"snapshot_{snap_count}.png"
        cv2.imwrite(snap_filename, img_resized)
        print(f"Snapshot saved as {snap_filename}")
        
        # Load the saved image for processing
        saved_img = cv2.imread(snap_filename)
        h, w, _ = saved_img.shape
        
        # Perform YOLO prediction on the saved image
        results = model.predict(saved_img,conf=0.5)

        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            
            if masks is not None and len(masks) > 0:
                masks = masks.data.cpu().numpy()
                for seg, box in zip(masks, boxes):
                    seg = cv2.resize(seg, (w, h))
                    contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        d = int(box.cls)
                        c = class_names[d]
                        x, y, bw, bh = cv2.boundingRect(contour)
                        cv2.polylines(saved_img, [contour], True, color=(0, 0, 255), thickness=2)
                        cv2.putText(saved_img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Show the processed image
        cv2.imshow('Processed Image', saved_img)

cap.release()
cv2.destroyAllWindows()
