import pygame as pg
import numpy as np
import cv2

class ArtConverter:
    def __init__(self, path = 'resources/chika.jpeg', font_size = 12, color_lvl = 8):
        pg.init()
        self.path = path
        self.COLOR_LVL = color_lvl
        self.image, self.grey_image = self.get_image()
        self.RES = self.WIDTH, self.HEIGHT = self.image.shape[0], self.image.shape[1]
        self.surface = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()
        
        self.ASCII_CHARS = ' ixzao*#MW&8%B@$'
        self.ASCII_COEF = 255 // (len(self.ASCII_CHARS) - 1)
        
        self.font = pg.font.SysFont('Courier', font_size, bold = True)
        self.CHAR_STEP = int(font_size * 0.6)
        self.PALETTE, self.COLOR_COEFF = self.create_palette()
        
    def create_palette(self):
        colors, color_coeff = np.linspace(0, 255, num=self.COLOR_LVL, dtype=int, retstep=True)
        color_palette = [np.array([r, g, b]) for r in colors for g in colors for b in colors]
        palette = dict.fromkeys(self.ASCII_CHARS, None)
        color_coeff = int(color_coeff)
        for char in palette:
            char_palette = {}
            for color in color_palette:
                color_key = tuple(color // color_coeff)
                char_palette[color_key] = self.font.render(char, False, tuple(color))
            palette[char] = char_palette
        return palette, color_coeff
            
    def get_image(self):
        self.cv2_image = cv2.imread(self.path)
        transposed_image = cv2.transpose(self.cv2_image)
        image = cv2.cvtColor(transposed_image, cv2.COLOR_RGB2BGR)
        grey_image = cv2.cvtColor(transposed_image, cv2.COLOR_BGR2GRAY)
        return image, grey_image
    
    def draw_converted_image(self):
        char_indices = self.grey_image // self.ASCII_COEF
        color_indices = self.image // self.COLOR_COEFF
        for x in range(0, self.WIDTH, self.CHAR_STEP):
            for y in range(0, self.HEIGHT, self.CHAR_STEP):
                char_index = char_indices[x ,y]
                if char_index:
                    char = self.ASCII_CHARS[char_index]
                    color = tuple(color_indices[x, y])
                    self.surface.blit(self.PALETTE[char][color], (x, y))
                    
    def save_image(self):
        pygame_image = pg.surfarray.array3d(self.surface)
        cv2_img = cv2.transpose(pygame_image)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('output/ascii_image_color.jpg', cv2_img)
    
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
                        print('ASCII-Art-colors saved')
            self.draw()
            pg.display.set_caption(str(self.clock.get_fps()))
            pg.display.flip()
            self.clock.tick()
    
if __name__ == '__main__':
    app = ArtConverter()
    app.run()