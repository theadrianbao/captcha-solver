import cv2
import numpy as np


def process_captcha(image_path):
    original_image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((7, 7), np.uint8)
    denoised_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

    _, binary_image = cv2.threshold(denoised_image, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    white_background = np.ones_like(original_image, dtype=np.uint8) * 255

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < 50:
            continue

        cv2.drawContours(white_background, [contour], 0, (0, 102, 255), thickness=cv2.FILLED)

    characters = cv2.bitwise_and(original_image, white_background)

    return characters
