import cv2
import numpy as np
import yaml
import math
import glob

def detect_color_centers_from_image(image, lower_color, upper_color):
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

def compute_scale(yellow_centers):
    if len(yellow_centers) < 2:
        return None
    distances = []
    for i in range(len(yellow_centers)):
        for j in range(i+1, len(yellow_centers)):
            distance = np.linalg.norm(np.array(yellow_centers[i]) - np.array(yellow_centers[j]))
            distances.append(distance)
    if distances:
        return np.mean(distances)
    return None

def detect_all_colors_from_image(image, thresholds, merge_threshold=20):
    results = {}
    for color, thresh in thresholds.items():
        centers = detect_color_centers_from_image(image, thresh['lower'], thresh['upper'])
        merged_centers = merge_close_centers(centers, threshold=merge_threshold)
        results[color] = merged_centers
    return results

def middle_point(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def signed_angle_between(v1, v2):
    angle1 = math.atan2(v1[1], v1[0])
    angle2 = math.atan2(v2[1], v2[0])
    angle = angle2 - angle1
    return math.degrees(math.atan2(math.sin(angle), math.cos(angle)))

def compute_robot_target_info(image, thresholds, merge_threshold=20):
    results = detect_all_colors_from_image(image, thresholds, merge_threshold)
    yellow_centers = results.get("yellow", [])
    scale_pixels = compute_scale(yellow_centers)
    if scale_pixels is None or not results.get("dark_blue") or not results.get("light_blue") or not results.get("pink"):
        return None, None
    dark_blue_center = results["dark_blue"][0]
    light_blue_center = results["light_blue"][0]
    pink_center = results["pink"][0]
    robot_mid = middle_point(dark_blue_center, light_blue_center)
    d = (light_blue_center[0] - dark_blue_center[0],
         light_blue_center[1] - dark_blue_center[1])
    if dark_blue_center[1] < light_blue_center[1]:
        orientation = (-d[1], d[0])
    else:
        orientation = (d[1], -d[0])
    v_to_pink = (pink_center[0] - robot_mid[0],
                 pink_center[1] - robot_mid[1])
    angle = signed_angle_between(orientation, v_to_pink)
    pixel_distance = np.linalg.norm(v_to_pink)
    real_distance = pixel_distance / scale_pixels
    return real_distance, angle

def draw_robot_target_info(image, thresholds, merge_threshold=20):
    results = detect_all_colors_from_image(image, thresholds, merge_threshold)
    yellow_centers = results.get("yellow", [])
    scale_pixels = compute_scale(yellow_centers)
    if scale_pixels is None or not results.get("dark_blue") or not results.get("light_blue") or not results.get("pink"):
        return image
    dark_blue_center = results["dark_blue"][0]
    light_blue_center = results["light_blue"][0]
    pink_center = results["pink"][0]
    robot_mid = middle_point(dark_blue_center, light_blue_center)
    d = (light_blue_center[0] - dark_blue_center[0],
         light_blue_center[1] - dark_blue_center[1])
    if dark_blue_center[1] < light_blue_center[1]:
        orientation = (-d[1], d[0])
    else:
        orientation = (d[1], -d[0])
    v_to_pink = (pink_center[0] - robot_mid[0],
                 pink_center[1] - robot_mid[1])
    angle = signed_angle_between(orientation, v_to_pink)
    pixel_distance = np.linalg.norm(v_to_pink)
    real_distance = pixel_distance / scale_pixels

    # Draw all circle centers in black.
    for centers in results.values():
        for center in centers:
            cv2.circle(image, center, 5, (0, 0, 0), -1)
    cv2.circle(image, robot_mid, 5, (0, 0, 0), -1)
    orientation_norm = np.linalg.norm(orientation)
    if orientation_norm != 0:
        orientation_unit = (orientation[0] / orientation_norm, orientation[1] / orientation_norm)
    else:
        orientation_unit = (0, 0)
    facing_line_end = (int(robot_mid[0] + orientation_unit[0] * 100),
                       int(robot_mid[1] + orientation_unit[1] * 100))
    cv2.line(image, robot_mid, facing_line_end, (255, 0, 0), 2)
    cv2.line(image, robot_mid, pink_center, (0, 255, 0), 2)
    text = f"Distance: {real_distance:.2f} m, Angle: {angle:.2f} deg"
    cv2.putText(image, text, (10, image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image

def main():
    thresholds = {
        "pink": {"lower": [140, 50, 50], "upper": [170, 255, 255]},
        "yellow": {"lower": [20, 200, 200], "upper": [30, 255, 255]},
        "dark_blue": {"lower": [110, 140, 190], "upper": [130, 255, 255]},
        "light_blue": {"lower": [84, 230, 230], "upper": [104, 255, 255]}
    }
    image_files = glob.glob('./images/*.*')
    for im_path in sorted(image_files):
        image = cv2.imread(im_path)
        if image is None:
            print("Image not found:", im_path)
            continue
        real_distance, angle = compute_robot_target_info(image, thresholds, merge_threshold=20)
        print(f"{im_path}: Real distance = {real_distance}, Angle = {angle}")
        drawn_image = draw_robot_target_info(image, thresholds, merge_threshold=20)
        cv2.imshow("Robot Target Info", drawn_image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
