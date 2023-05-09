import subprocess as sp
import time
from os import mkdir, remove, rmdir

import cv2
import multiprocess as mp

from pixel_image import ImageHandler


class VideoHandler:
    PIXELIZE, ANONYMIZE = range(2)

    def __init__(self, path: str) -> None:
        self.__path = path
        self.__file = path.split("/")[-1]
        self.__file_name = self.__file.split(".")[0]
        self.__dir_name = f"temp/{self.__file_name}"

        self.__init_num_processes()

        cap = cv2.VideoCapture(path)
        self.__frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__frame_shift = int(self.__frame_count // self.__num_processes)
        self.__fps = cap.get(cv2.CAP_PROP_FPS)
        self.__width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap.release()

    def __init_num_processes(self) -> None:
        self.__num_processes = 4
        threads_count = mp.cpu_count()
        if self.__num_processes > threads_count:
            self.__num_processes = threads_count

    def pixelize(self, pixel_size: int) -> str:
        self.__pixel_size = pixel_size
        return self.__process(self.PIXELIZE)

    def anonymize(self) -> str:
        return self.__process(self.ANONYMIZE)

    def faces_not_found(self) -> bool:
        return not self.__are_faces_found.value

    def __extract_audio(self) -> None:
        ffmpeg_cmd = f"ffmpeg -y -loglevel error -i {self.__path} {self.__audio_file}"
        sp.Popen(ffmpeg_cmd, shell=True).wait()

    def __pixelize_video_part(self, part_number: int) -> None:
        cap = cv2.VideoCapture(self.__path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.__frame_shift * part_number)

        out = cv2.VideoWriter(
            f"{self.__dir_name}/part_{part_number}.mp4",
            self.__fourcc,
            self.__fps,
            (self.__width, self.__height),
        )
        part_end = self.__frame_shift
        if part_number == self.__num_processes - 1:
            part_end = self.__frame_count - self.__frame_shift * part_number
        for _ in range(part_end):
            ret, frame = cap.read()
            if not ret:
                break

            ImageHandler(frame).pixelize_for_video(self.__pixel_size)

            out.write(frame)

        cap.release()
        out.release()

    def __anonymize_video_part(self, part_number: int) -> None:
        cap = cv2.VideoCapture(self.__path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.__frame_shift * part_number)

        out = cv2.VideoWriter(
            f"{self.__dir_name}/part_{part_number}.mp4",
            self.__fourcc,
            self.__fps,
            (self.__width, self.__height),
        )
        part_end = self.__frame_shift
        if part_number == self.__num_processes - 1:
            part_end = self.__frame_count - self.__frame_shift * part_number

        are_faces_found = False
        for _ in range(part_end):
            ret, frame = cap.read()
            if not ret:
                break

            are_faces_found_in_frame = ImageHandler(frame).pixelize_faces()
            if not are_faces_found and are_faces_found_in_frame:
                are_faces_found = True

            out.write(frame)

        if not self.__are_faces_found.value and are_faces_found:
            self.__are_faces_found.value = True

        cap.release()
        out.release()

    def __combine_video_parts(self) -> None:
        self.__video_parts = [f"part_{i}.mp4" for i in range(self.__num_processes)]

        video_parts_file = f"{self.__dir_name}/video_part_files.txt"
        with open(video_parts_file, "w") as video_parts:
            for video_part in self.__video_parts:
                video_parts.write(f"file {video_part} \n")

        self.__video_without_audio_file = f"{self.__dir_name}/video.mp4"
        ffmpeg_cmd = (
            "ffmpeg -y -loglevel error -f concat -safe 0 -i "
            f"{video_parts_file} -vcodec copy {self.__video_without_audio_file}"
        )
        sp.Popen(ffmpeg_cmd, shell=True).wait()

        remove(video_parts_file)

    def __add_audio_to_video(self) -> None:
        self.__result_video = f"temp/{self.__file}"
        ffmpeg_cmd = (
            f"ffmpeg -y -loglevel error -i {self.__video_without_audio_file} "
            f"-i {self.__audio_file} {self.__result_video}"
        )
        sp.Popen(ffmpeg_cmd, shell=True).wait()

    def __process(self, mode) -> str:
        mkdir(self.__dir_name)
        self.__audio_file = f"{self.__dir_name}/audio.wav"
        self.__extract_audio()

        pool = mp.Pool(self.__num_processes)
        match mode:
            case self.PIXELIZE:
                pool.map(self.__pixelize_video_part, range(self.__num_processes))
            case self.ANONYMIZE:
                self.__are_faces_found = mp.Manager().Value(bool, False)
                pool.map(self.__anonymize_video_part, range(self.__num_processes))
        pool.close()
        pool.join()

        self.__combine_video_parts()
        for f in self.__video_parts:
            remove(f"{self.__dir_name}/{f}")

        self.__add_audio_to_video()
        remove(self.__video_without_audio_file)
        remove(self.__audio_file)
        rmdir(self.__dir_name)

        return self.__result_video


if __name__ == "__main__":
    path = "resources/can.mp4"

    print("Pixelize video")
    start_time = time.time()
    handler = VideoHandler(path)
    video = handler.anonymize()
    print(handler.faces_not_found())
    end_time = time.time()

    remove(video)

    print(f"Time: {end_time - start_time}\n")
