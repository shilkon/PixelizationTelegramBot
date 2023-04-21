import pygame as pg
import pygame.gfxdraw
from numba import njit
import numpy as np
import cv2

@njit(fastmath=True)
def accelerate_conversion(image, width, height, color_coeff, step):
    array_of_values = []
    for x in range(0, width, step):
        for y in range(0, height, step):
            r, g, b = image[x, y] // color_coeff
            if r + g + b:
                array_of_values.append(((r, g, b), (x, y)))
    return array_of_values

class ArtConverter:
    def __init__(self, path = 'resources/bebek.mp4', pixel_size = 7, color_lvl = 8):
        pg.init()
        self.path = path
        self.capture = cv2.VideoCapture(path)
        self.PIXEL_SIZE = pixel_size
        self.COLOR_LVL = color_lvl
        self.image = self.get_image()
        self.RES = self.WIDTH, self.HEIGHT = self.image.shape[0], self.image.shape[1]
        self.surface = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()
        self.PALETTE, self.COLOR_COEFF = self.create_palette()
        
    def create_palette(self):
        colors, color_coeff = np.linspace(0, 255, num=self.COLOR_LVL, dtype=int, retstep=True)
        color_palette = [np.array([r, g, b]) for r in colors for g in colors for b in colors]
        palette = {}
        color_coeff = int(color_coeff)
        for color in color_palette:
            color_key = tuple(color // color_coeff)
            palette[color_key] = color
        return palette, color_coeff
            
    def get_image(self):
        ret, self.cv2_image = self.capture.read()
        if not ret:
            exit()
        transposed_image = cv2.transpose(self.cv2_image)
        image = cv2.cvtColor(transposed_image, cv2.COLOR_RGB2BGR)
        return image
    
    def draw_converted_image(self):
        self.image = self.get_image()
        array_of_values = accelerate_conversion(self.image, self.WIDTH, self.HEIGHT, self.COLOR_COEFF, self.PIXEL_SIZE)
        for color_key, (x, y) in array_of_values:
            color = self.PALETTE[color_key]
            pygame.gfxdraw.box(self.surface, (x, y, self.PIXEL_SIZE, self.PIXEL_SIZE), color)
                    
    def save_image(self):
        pygame_image = pg.surfarray.array3d(self.surface)
        cv2_img = cv2.transpose(pygame_image)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('output/pixel_art/pixel-art_video.jpg', cv2_img)
    
    def draw(self):
        self.surface.fill('black')
        self.draw_converted_image()
        cv2.imshow('img', self.cv2_image)
    
    def run(self):
        while True:
            for i in pg.event.get():
                if i.type == pg.QUIT:
                    exit()
                elif i.type == pg.KEYDOWN:
                    if i.key == pg.K_s:
                        self.save_image()
                        print('PIXEL-art saved')
            self.draw()
            pg.display.set_caption(str(self.clock.get_fps()))
            pg.display.flip()
            self.clock.tick()
    
if __name__ == '__main__':
    app = ArtConverter()
    app.run()