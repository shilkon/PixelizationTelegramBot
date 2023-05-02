import numpy as np
from numba import njit
import cv2
import time
import pixel_exception as pe

AVAILABLE_COLOR_LEVELS = (4, 8, 16, 32, 64)

def create_palette(color_lvl):
    if color_lvl not in AVAILABLE_COLOR_LEVELS:
        raise pe.InvalidColorLvl
    
    colors, color_coeff = np.linspace(0, 255, num=color_lvl, dtype=int, retstep=True)
    color_palette = [np.array([b, g, r]) for b in colors for g in colors for r in colors]
    palette = {}
    color_coeff = int(color_coeff)
    
    for color in color_palette:
        color_key = tuple(color // color_coeff)
        palette[color_key] = tuple(int(x) for x in color)
        
    return palette, color_coeff

def process_image(image, palette, pixel_size):
    (height, width) = image.shape[:2]
    
    if pixel_size < 2 or pixel_size > width or pixel_size > height:
        raise pe.InvalidPixelSize
    
    palette, color_coef = palette
    
    side = pixel_size - 1
    color_indices = image // color_coef  
      
    for y in range(0, height, pixel_size):
        for x in range(0, width, pixel_size):
            color = palette[get_average_color(color_indices, y, x, side, height, width)]
            cv2.rectangle(image, (x, y), (x + side , y + side), color, cv2.FILLED)
                
    return image

def get_average_color(image, y, x, side, height, width):
    y_border = min(y + side, height)
    x_border = min(x + side, width)
    
    return tuple(np.round(cv2.mean(image[y : y_border, x : x_border])[:3]))

def pixelize_image(image, pixel_size):
    (height, width) = image.shape[:2]
    
    if pixel_size < 2 or pixel_size > width or pixel_size > height:
        raise pe.InvalidPixelSize
    
    side = pixel_size - 1
    for y in range(0, height, pixel_size):
        for x in range(0, width, pixel_size):
            color = get_average_color(image, y, x, side, height, width)
            cv2.rectangle(image, (x, y), (x + side , y + side), color, cv2.FILLED)
    
    return image

def pixelize_image_acc_for_video(image, pixel_size):
    (height, width) = image.shape[:2]
    
    if pixel_size < 2 or pixel_size > width or pixel_size > height:
        raise pe.InvalidPixelSize
    
    side = pixel_size - 1
    pixels = accelerate_pixelization(image, height, width, pixel_size, side)
    
    for color, (x, y) in pixels:
        cv2.rectangle(image, (x, y), (x + side , y + side), color, cv2.FILLED)
    
    return image

@njit(fastmath=True)
def accelerate_pixelization(image, height, width, step, side):
    pixels = []
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            b, g, r = get_average_color_acc_for_video(image, y, x, side, height, width)
            pixels.append(((b, g, r), (x, y)))
                
    return pixels  

@njit(fastmath=True)
def get_average_color_acc_for_video(image, y, x, side, height, width):
    y_border = min(y + side, height)
    x_border = min(x + side, width)
    color_sum = np.sum(np.sum(image[y : y_border, x : x_border], 0), 0)
    
    return np.round(color_sum / side**2, 0, color_sum) 
    
if __name__ == '__main__':
    def single_process(convert_func, *args):
        start_time = time.time()
        image = convert_func(*args)
        end_time = time.time()
        
        print('Time: {}\n'.format(end_time - start_time))
        
        # cv2.imshow('img', image)
        # if cv2.waitKey() == ord("s"):
        #     print('pixel-art saved')
        #     cv2.imwrite('output/pixel-art.jpg', image)
    
    image = cv2.imread('resources/mountain.jpg')
    
    print('Processing image')
    single_process(process_image, image, create_palette(32), 8)
    
    print('Pixelize image')    
    single_process(pixelize_image, image, 8)
    
    print('Pixelize image accelerated for video')    
    single_process(pixelize_image_acc_for_video, image, 8)
    
    cv2.destroyAllWindows()