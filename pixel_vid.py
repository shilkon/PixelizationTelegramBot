import cv2
import pixel_img

def convert_video(path, pixel_size = 8, color_lvl = 16):
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    print(fourcc)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # print(fourcc)
    
    palette = pixel_img.create_palette(color_lvl)
    out = cv2.VideoWriter('output/pixel-art_video.avi', fourcc, fps, (width,  height))
    while True:
        ret, frame = capture.read()
        if not ret:
            print('Video ended')
            break
        converted_frame = pixel_img.convert_image(frame, palette, pixel_size)
        #cv2.imshow('frame', converted_frame)
        out.write(converted_frame)
        if cv2.waitKey(int(fps)) == ord('q'):
            break
        
        #print(capture.get(cv2.CAP_PROP_POS_AVI_RATIO))
    out.release()

if __name__ == '__main__':
    convert_video('resources/bebek.mp4', 10, 16)
    