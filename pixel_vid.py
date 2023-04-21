import cv2
import pixel_img
import time

def convert_video(path, output_name,  palette, pixel_size = 8):
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    out = cv2.VideoWriter(output_name, fourcc, fps, (width,  height))
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        converted_frame = pixel_img.convert_image(frame, palette, pixel_size)
        #cv2.imshow('frame', converted_frame)
        out.write(converted_frame)
        if cv2.waitKey(int(fps)) == ord('q'):
            break
        
        #print(capture.get(cv2.CAP_PROP_POS_AVI_RATIO))
    out.release()
    
def convert_video_numba(path, output_name, palette, pixel_size = 8):
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    out = cv2.VideoWriter(output_name, fourcc, fps, (width,  height))
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        converted_frame = pixel_img.convert_image_numba(frame, palette, pixel_size)
        #cv2.imshow('frame', converted_frame)
        out.write(converted_frame)
        if cv2.waitKey(int(fps)) == ord('q'):
            break
        
        #print(capture.get(cv2.CAP_PROP_POS_AVI_RATIO))
    out.release()

if __name__ == '__main__':
    def single_process(convert_func, output_name, path, color_lvl, pixel_size):
        print('Processing video single process...')
        palette = pixel_img.create_palette(color_lvl)
        start_time = time.time()
        convert_func(path, output_name, palette, pixel_size)
        end_time = time.time()
        print('Time: {}\n'.format(end_time - start_time))
        
    path = 'resources/bebek.mp4'
    
    print('cv2.mean')
    single_process(convert_video, 'output/mean.avi', path, 16, 8)
        
    print('Numba')    
    single_process(convert_video_numba, 'output/numba.avi', path, 16, 8)
    