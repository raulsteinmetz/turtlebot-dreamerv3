import cv2
import numpy as np
import glob
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

def detect_all_colors(im_path, thresholds, merge_threshold=100):
    results = {}
    for color, thresh in thresholds.items():
        centers = detect_color_centers(im_path, thresh['lower'], thresh['upper'])
        merged_centers = merge_close_centers(centers, threshold=merge_threshold)
        results[color] = merged_centers
    return results

def main():
    image_files = glob.glob('./images/*.*')
    with open('thresholds.yaml', 'r') as f:
        thresholds = yaml.safe_load(f)
    
    expected_counts = {
        'pink': 4,
        'yellow': 3,
        'dark_blue': 1,
        'light_blue': 1,
    }
    
    for im_path in sorted(image_files):
        results = detect_all_colors(im_path, thresholds, merge_threshold=100)
        failures = []
        for color, expected in expected_counts.items():
            count = len(results.get(color, []))
            if count != expected:
                failures.append(f"{color}: expected {expected}, got {count}")
        if failures:
            print("Image:", im_path)
            for f in failures:
                print("  -", f)
            print()

if __name__ == '__main__':
    main()
