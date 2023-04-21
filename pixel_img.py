import numpy as np
from numba import njit
import cv2

def convert_image(image, palette, pixel_size = 8):
    image = cv2.transpose(image)
    width, height = image.shape[0], image.shape[1]
    palette, color_coef = palette
    converted_image = np.zeros((height, width, 3), np.uint8)
    side = pixel_size - 1
    array_of_values = accelerate_conversion(image, width, height, color_coef, pixel_size)
    for color_key, (x, y) in array_of_values:
        color = [int(x) for x in palette[color_key]]
        pt1 = (x, y)
        pt2 = (x + side , y + side)
        cv2.rectangle(converted_image, pt1, pt2, color, cv2.FILLED)
    return converted_image
    
def create_palette(color_lvl):
    colors, color_coeff = np.linspace(0, 255, num=color_lvl, dtype=int, retstep=True)
    color_palette = [np.array([b, g, r]) for b in colors for g in colors for r in colors]
    palette = {}
    color_coeff = int(color_coeff)
    for color in color_palette:
        color_key = tuple(color // color_coeff)
        palette[color_key] = color
    return palette, color_coeff

#@njit(fastmath=True)
def accelerate_conversion(image, width, height, color_coeff, step):
    array_of_values = []
    side = step - 1
    color_indices = image // color_coeff
    for x in range(0, width, step):
        for y in range(0, height, step):
            b, g, r = get_average_color(color_indices, x, y, side, width, height)
            if b + g + r:
                array_of_values.append(((b, g, r), (x, y)))
    return array_of_values   

#@njit(fastmath=True)
def get_average_color(image, x, y, side, width, height):
    x_border = min(x + side, width)
    y_border = min(y + side, height)
    #color_sum = np.zeros(3)
    # for i in range(x, x_border):
    #     for j in range(y, y_border):
    #         color_sum += image[i, j]
    # replacing_pixels = image[x : x_border, y : y_border]
    #print(np.mean(replacing_pixels, axis=(0, 1)))
    # return color_sum // side**2
    replacing_pixels = image[x : x_border, y : y_border]
    return np.round(cv2.mean(replacing_pixels)[:3])
    
if __name__ == '__main__':
    
    image = convert_image(cv2.imread('resources/t55a.jpg'), create_palette(16), 4)
    cv2.imshow('img', image)
    key = cv2.waitKey()
    if key == ord("s"):
        print('pixel-art saved')
        cv2.imwrite('output/pixel-art.jpg', image)
    cv2.destroyAllWindows()