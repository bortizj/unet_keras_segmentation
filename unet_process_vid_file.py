from pathlib import Path
import tqdm

import torch
from unet import UNetModel

from video_reader import VideoReader
from video_writer import VideoWriter

import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATH_MODEL = Path(r"D:\gitProjects\segmentation_unet\data_set\data")

if __name__ == "__main__":
    path_vid_file = Path(r"F:\CaseRecording_1 CAM 1.mp4")
    # path_vid_file = Path(r"F:\CaseRecording_2 CAM 1.mp4")
    # path_vid_file = Path(r"F:\CaseRecording_4 CAM 1.mp4")
    # path_vid_file = Path(r"F:\CaseRecording_23 CAM 1.mp4")
    out_vid_file = path_vid_file.parent.joinpath(path_vid_file.name + "_processed.mp4")

    vr = VideoReader(str(path_vid_file))
    vw = VideoWriter(str(out_vid_file))

    unet_model = UNetModel(
        path_model=str(PATH_MODEL.joinpath("unet_checkpoint.tar")),
        in_channels=3,
        out_channels=4,
        sampling_factor=8,
        device=DEVICE,
    )

    count = 0

    with tqdm.tqdm(total=100000) as pbar:
        while True:
            frame = vr.read_frame()

            if frame is not None:
                out_img = unet_model.evaluate_image(frame)

                vw.write_frame(out_img)
            else:
                break

            count += 1

            pbar.update(1)
