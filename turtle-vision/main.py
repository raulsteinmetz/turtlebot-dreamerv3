import cv2
import numpy as np
import yaml

def detect_color_centers(im_path, lower_color, upper_color):
    image = cv2.imread(im_path)
    if image is None:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_color), np.array(upper_color))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
    return centers

def merge_close_centers(centers, threshold=20):
    merged = []
    for center in centers:
        found = False
        for i, m in enumerate(merged):
            if np.linalg.norm(np.array(center) - np.array(m)) < threshold:
                merged[i] = ((m[0] + center[0]) // 2, (m[1] + center[1]) // 2)
                found = True
                break
        if not found:
            merged.append(center)
    return merged

def average_centers(centers):
    if not centers:
        return None
    arr = np.array(centers)
    avg = np.mean(arr, axis=0)
    return (int(avg[0]), int(avg[1]))

def compute_scale(yellow_centers):
    if len(yellow_centers) < 2:
        return None
    distances = []
    for i, center in enumerate(yellow_centers):
        dists = [np.linalg.norm(np.array(center) - np.array(other))
                 for j, other in enumerate(yellow_centers) if j != i]
        if dists:
            distances.append(min(dists))
    if distances:
        return np.mean(distances)
    return None

def detect_all_colors(im_path, thresholds, merge_threshold=20):
    results = {}
    for color, thresh in thresholds.items():
        centers = detect_color_centers(im_path, thresh['lower'], thresh['upper'])
        merged_centers = merge_close_centers(centers, threshold=merge_threshold)
        if color == "dark_blue":
            # Average all dark blue detections into a single center
            results[color] = average_centers(merged_centers)
        else:
            results[color] = merged_centers
    return results

def main():
    im_path = 'images/WIN_20250129_14_11_55_Pro.jpg'
    with open('thresholds.yaml', 'r') as f:
        thresholds = yaml.safe_load(f)
    results = detect_all_colors(im_path, thresholds, merge_threshold=100)
    
    print("Detected centers:")
    for color, centers in results.items():
        print(f"{color.capitalize()}: {centers}")
    
    yellow_centers = results.get("yellow", [])
    scale_pixels = compute_scale(yellow_centers)
    if scale_pixels is not None:
        print(f"Scale: 1 meter = {scale_pixels:.2f} pixels")
    else:
        print("Scale: Not enough yellow centers to compute scale")
    
    # Draw all centers on the image in black
    image = cv2.imread(im_path)
    if image is None:
        print("Image not found:", im_path)
        return
    for color, centers in results.items():
        if color == "dark_blue":
            if centers is not None:
                cv2.circle(image, centers, 5, (0, 0, 0), -1)
        else:
            for center in centers:
                cv2.circle(image, center, 5, (0, 0, 0), -1)
    
    cv2.imshow("Detected Centers", image)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
