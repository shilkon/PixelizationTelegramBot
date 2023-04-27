import cv2
import pixel_img
import time
import subprocess as sp
import multiprocess as mp
from os import remove

def process_video_sp(process_func, path, output_name,  palette, pixel_size = 8):
    cap = cv2.VideoCapture(path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    out = cv2.VideoWriter(output_name, fourcc, fps, (width,  height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        converted_frame = process_func(frame, palette, pixel_size)
        out.write(converted_frame)

    cap.release()
    out.release()
    
def process_video_mp(process_img_func, path, output_name, palette, pixel_size = 8):
    num_processes = 4
    threads_count = mp.cpu_count()
    if num_processes > threads_count:
        num_processes = threads_count
    
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_shift = int(frame_count // num_processes)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap.release()
    
    def process_video_part(part_number):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_shift * part_number)
        
        out = cv2.VideoWriter("output/temp/output_{}.mp4".format(part_number), fourcc, fps, (width,  height))
        part_end = frame_shift
        if part_number == num_processes - 1:
            part_end = frame_count - frame_shift * part_number
        for proc_frames in range(part_end):
            ret, frame = cap.read()
            if not ret:
                break
            converted_frame = process_img_func(frame, palette, pixel_size)
            out.write(converted_frame)
            
        cap.release()
        out.release()
    
    p = mp.Pool(num_processes)
    p.map(process_video_part, range(num_processes))

    combine_output_files(num_processes, output_name)
    
def combine_output_files(num_processes, output_file_name):
    # Create a list of output files and store the file names in a txt file
    list_of_output_files = ["output/temp/output_{}.mp4".format(i) for i in range(num_processes)]
    with open("list_of_output_files.txt", "w") as f:
        for t in list_of_output_files:
            f.write("file {} \n".format(t))

    # use ffmpeg to combine the video output files
    ffmpeg_cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i list_of_output_files.txt -vcodec copy " + output_file_name
    sp.Popen(ffmpeg_cmd, shell=True).wait()

    # Remove the temperory output files
    for f in list_of_output_files:
        remove(f)
    remove("list_of_output_files.txt")

if __name__ == '__main__':
    def single_process(process_video_func, process_img_func, path, output_name, color_depth, pixel_size):
        print('Processing video single process...')
        
        palette = pixel_img.create_palette(color_depth)
        
        start_time = time.time()
        process_video_func(process_img_func, path, output_name, palette, pixel_size)
        end_time = time.time()
        
        print('Time: {}\n'.format(end_time - start_time))
        
    def multi_process(process_video_func, process_img_func, path, output_name, color_depth, pixel_size):
        print('Processing video multiple processes...')
        
        palette = pixel_img.create_palette(color_depth)
        
        start_time = time.time()
        process_video_func(process_img_func, path, output_name, palette, pixel_size)
        end_time = time.time()
        
        print('Time: {}\n'.format(end_time - start_time))
        
    path = 'resources/bebek.mp4'
    
    # print('cv2.mean SP')
    # single_process(process_video_sp, pixel_img.convert_image, path, 'output/meanSP.avi', 32, 8)
        
    # print('Numba SP')    
    # single_process(process_video_sp, pixel_img.convert_image_numba, path, 'output/numbaSP.avi', 32, 8)
    
    # print('cv2.mean MP')
    # multi_process(process_video_mp, pixel_img.convert_image, path, 'output/mp/meanMP.avi', 32, 8)
    
    print('Numba MP')
    multi_process(process_video_mp, pixel_img.process_image_numba, path, 'output/numbaMP.avi', 32, 8)