from re import L
import cv2 as cv
from pathlib import Path

def main(**kwargs):
    video_path = kwargs['video_path']
    video_name = kwargs['video_name']
    output_path = kwargs['output_path']
    vidcap = cv.VideoCapture(str(video_path/video_name))
    success,image = vidcap.read()
    count = 0
    while success:
        cv.imwrite(f"{output_path}/frame{count}.png", image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


if __name__ == "__main__":
    video_path = Path("./datasets/BDD/validation/camera_videos")
    video_name = "1811.mp4"
    output_path = Path("./datasets/BDD/scene/scene_06")
    output_path.mkdir(exist_ok=True, parents=True)
    main(video_path=video_path, video_name=video_name, output_path=output_path)