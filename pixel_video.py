import cv2
import pixel_image
import time
import subprocess as sp
import multiprocess as mp
from os import (
    remove,
    mkdir,
    rmdir
)
    
def pixelize_video(path: str, pixel_size: int) -> str:
    file = path.split('/')[-1]
    file_name = file.split('.')[0]
    dir_name = f"temp/{file_name}"
    mkdir(dir_name)
    
    audio_file = f"{dir_name}/audio.wav"
    extract_audio(path, audio_file)
    
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
        
        out = cv2.VideoWriter(f"{dir_name}/part_{part_number}.mp4", fourcc, fps, (width,  height))
        part_end = frame_shift
        if part_number == num_processes - 1:
            part_end = frame_count - frame_shift * part_number
        for proc_frames in range(part_end):
            ret, frame = cap.read()
            if not ret:
                break
            converted_frame = pixel_image.pixelize_image_acc_for_video(frame, pixel_size)
            out.write(converted_frame)
            
        cap.release()
        out.release()
    
    p = mp.Pool(num_processes)
    p.map(process_video_part, range(num_processes))
    
    video_without_audio_file = f"{dir_name}/video.mp4"
    combine_video_parts(num_processes, video_without_audio_file, dir_name)
    result_video = f"temp/{file}"
    add_audio_to_video(video_without_audio_file, audio_file, result_video)
    rmdir(dir_name)
    
    return result_video
    
def extract_audio(video_source: str, audio_file: str):
    ffmpeg_cmd = f"ffmpeg -y -loglevel error -i {video_source} {audio_file}"
    sp.Popen(ffmpeg_cmd, shell=True).wait()
    
def combine_video_parts(num_processes: int, output_file_name: str, dir_name: str):
    video_parts = ["part_{}.mp4".format(i) for i in range(num_processes)]
    
    video_parts_file = f"{dir_name}/video_part_files.txt"
    with open(video_parts_file, "w") as video_part:
        for t in video_parts:
            video_part.write("file {} \n".format(t))

    ffmpeg_cmd = f"ffmpeg -y -loglevel error -f concat -safe 0 -i {video_parts_file} -vcodec copy {output_file_name}"
    sp.Popen(ffmpeg_cmd, shell=True).wait()

    for f in video_parts:
        remove(f"{dir_name}/{f}")
    remove(video_parts_file)
    
def add_audio_to_video(video_source: str, audio_source: str, output_name: str):
    ffmpeg_cmd = f"ffmpeg -y -loglevel error -i {video_source} -i {audio_source} {output_name}"
    sp.Popen(ffmpeg_cmd, shell=True).wait()
    
    remove(video_source)
    remove(audio_source)

if __name__ == '__main__':
    def test(path, pixel_size):
        start_time = time.time()
        pixelize_video(path, pixel_size)
        end_time = time.time()
        
        print('Time: {}\n'.format(end_time - start_time))
        
    path = 'resources/can.mp4'
    
    print('Pixelize video')
    pixel_video = test(path, 8)