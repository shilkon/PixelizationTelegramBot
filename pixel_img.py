import numpy as np
from numba import njit
import cv2
import time

def convert_image(image, palette, pixel_size = 8):
    image = cv2.transpose(image)
    width, height = image.shape[0], image.shape[1]
    palette, color_coef = palette
    converted_image = np.zeros((height, width, 3), np.uint8)
    side = pixel_size - 1
    color_indices = image // color_coef    
    for x in range(0, width, pixel_size):
        for y in range(0, height, pixel_size):
            color_key = get_average_color(color_indices, x, y, side, width, height)
            if sum(color_key):
                color = palette[color_key]
                pt1 = (x, y)
                pt2 = (x + side , y + side)
                cv2.rectangle(converted_image, pt1, pt2, color, cv2.FILLED)
    return converted_image

def get_average_color(image, x, y, side, width, height):
    x_border = min(x + side, width)
    y_border = min(y + side, height)
    return tuple(np.round(cv2.mean(image[x : x_border, y : y_border])[:3]))
    
def create_palette(color_depth):
    colors, color_coeff = np.linspace(0, 255, num=color_depth, dtype=int, retstep=True)
    color_palette = [np.array([b, g, r]) for b in colors for g in colors for r in colors]
    palette = {}
    color_coeff = int(color_coeff)
    for color in color_palette:
        color_key = tuple(color // color_coeff)
        palette[color_key] = tuple(int(x) for x in color)
    return palette, color_coeff

def convert_image_numba(image, palette, pixel_size = 8):
    image = cv2.transpose(image)
    width, height = image.shape[0], image.shape[1]
    palette, color_coef = palette
    converted_image = np.zeros((height, width, 3), np.uint8)
    side = pixel_size - 1
    array_of_values = accelerate_conversion(image, width, height, color_coef, pixel_size)
    for color_key, (x, y) in array_of_values:
        color = palette[color_key]
        pt1 = (x, y)
        pt2 = (x + side , y + side)
        cv2.rectangle(converted_image, pt1, pt2, color, cv2.FILLED)
    return converted_image

@njit(fastmath=True, cache=True)
def accelerate_conversion(image, width, height, color_coeff, step):
    array_of_values = []
    side = step - 1
    color_indices = image // color_coeff
    for x in range(0, width, step):
        for y in range(0, height, step):
            b, g, r = get_average_color_numba(color_indices, x, y, side, width, height)
            if b + g + r:
                array_of_values.append(((b, g, r), (x, y)))
    return array_of_values   

@njit(fastmath=True, cache=True)
def get_average_color_numba(image, x, y, side, width, height):
    x_border = min(x + side, width)
    y_border = min(y + side, height)
    color_sum = np.sum(np.sum(image[x : x_border, y : y_border], 0), 0)
    return np.round(color_sum / side**2, 0, color_sum)
    
if __name__ == '__main__':
    def single_process(convert_func, image, color_depth, pixel_size):
        print('Processing image single process...')
        start_time = time.time()
        image = convert_func(image, create_palette(color_depth), pixel_size)
        end_time = time.time()
        print('Time: {}\n'.format(end_time - start_time))
        # cv2.imshow('img', image)
        # key = cv2.waitKey()
        # if key == ord("s"):
        #     print('pixel-art saved')
        #     cv2.imwrite('output/pixel-art.jpg', image)
    
    image = cv2.imread('resources/mountain.jpg')
    
    print('cv2.mean')
    single_process(convert_image, image, 16, 4)
        
    print('Numba')    
    single_process(convert_image_numba, image, 16, 4)
    # cv2.destroyAllWindows()