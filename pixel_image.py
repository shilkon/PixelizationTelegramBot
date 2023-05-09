import cv2
import face_recognition
import numpy as np
from numba import njit

import pixel_exception as pe


class ImageHandler:
    AVAILABLE_COLOR_LEVELS = (4, 8, 16, 32, 64)

    def __init__(self, image: np.ndarray) -> None:
        self.__image = image
        self.__height, self.__width = image.shape[:2]

    def process(self, color_level: int, pixel_size: int) -> None:
        self.__create_palette(color_level)

        self.__check_pixel_size(pixel_size)

        for y in range(0, self.__height, pixel_size):
            for x in range(0, self.__width, pixel_size):
                color = self.__get_average_color(y, x, self.__height, self.__width)
                cv2.rectangle(
                    image,
                    (x, y),
                    (x + self.__side, y + self.__side),
                    self.__palette[tuple(c // self.__color_coeff for c in color)],
                    cv2.FILLED,
                )

    def pixelize(self, pixel_size: int) -> None:
        self.__check_pixel_size(pixel_size)

        for y in range(0, self.__height, pixel_size):
            for x in range(0, self.__width, pixel_size):
                color = self.__get_average_color(y, x, self.__height, self.__width)
                cv2.rectangle(
                    self.__image,
                    (x, y),
                    (x + self.__side, y + self.__side),
                    color,
                    cv2.FILLED,
                )

    def pixelize_for_video(self, pixel_size: int) -> None:
        self.__check_pixel_size(pixel_size)

        pixels = accelerate_pixelization(
            self.__image, self.__height, self.__width, self.__side
        )

        for color, (x, y) in pixels:
            cv2.rectangle(
                self.__image,
                (x, y),
                (x + self.__side, y + self.__side),
                color,
                cv2.FILLED,
            )

    def pixelize_faces(self) -> bool:
        faces = face_recognition.face_locations(self.__image)

        for top, right, bottom, left in faces:
            self.__pixelize_face(left, top, right - left, bottom - top)

        return len(faces) > 0

    def __create_palette(self, color_level: int) -> None:
        if color_level not in self.AVAILABLE_COLOR_LEVELS:
            raise pe.InvalidColorLvl

        colors, color_coeff = np.linspace(0, 255, color_level, dtype=int, retstep=True)
        color_palette = [
            np.array([b, g, r]) for b in colors for g in colors for r in colors
        ]
        self.__palette = {}
        self.__color_coeff = int(color_coeff)

        for color in color_palette:
            color_key = tuple(color // self.__color_coeff)
            self.__palette[color_key] = tuple(int(x) for x in color)

    def __check_pixel_size(self, pixel_size: int) -> None:
        if pixel_size < 2 or pixel_size > self.__width or pixel_size > self.__height:
            raise pe.InvalidPixelSize
        self.__side = pixel_size - 1

    def __get_average_color(self, y: int, x: int, height: int, width: int):
        y_border = min(y + self.__side, height)
        x_border = min(x + self.__side, width)

        return tuple(np.round(cv2.mean(self.__image[y:y_border, x:x_border])[:3]))

    def __pixelize_face(self, x: int, y: int, width: int, height: int) -> None:
        pixel_size = height // 8 if height > 16 else 2

        self.__check_pixel_size(pixel_size)

        for y_face in range(y, y + height, pixel_size):
            for x_face in range(x, x + width, pixel_size):
                color = self.__get_average_color(y_face, x_face, y + height, x + width)
                cv2.rectangle(
                    self.__image,
                    (x_face, y_face),
                    (x_face + self.__side, y_face + self.__side),
                    color,
                    cv2.FILLED,
                )


@njit(fastmath=True)
def accelerate_pixelization(image, height, width, side):
    pixels = []

    for y in range(0, height, side + 1):
        for x in range(0, width, side + 1):
            b, g, r = get_average_color(image, y, x, height, width, side)
            pixels.append(((b, g, r), (x, y)))

    return pixels


@njit(fastmath=True)
def get_average_color(image, y, x, height, width, side):
    y_border = min(y + side, height)
    x_border = min(x + side, width)
    color_sum = np.sum(np.sum(image[y:y_border, x:x_border], 0), 0)

    return np.round(color_sum / side**2, 0, color_sum)


if __name__ == "__main__":
    image_file = open("resources/t55a.jpg", "rb")

    image = face_recognition.load_image_file(image_file)

    ImageHandler(image).process(16, 4)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("img", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
