import cv2
import numpy as np
import logging
from typing import List


def frames_to_video(
    frames: list,
    filename: str,
    path: str = "./",
    codecs: list = ["mp4v"],
    formats: list = ["mp4"],
    framerate: int = 60,  # Hz
    resolution: List[int] = [2560, 1440],
    save_last_frame: bool = True,
    frame_path: str = "../videos/last_frame/",
):
    for codec, format_ in zip(codecs, formats):
        logging.info("Writing {} {}".format(codec, format_))
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video = cv2.VideoWriter(
            path + filename + ".{}".format(format_),
            fourcc,
            framerate,
            resolution,
        )
        for frame in frames:
            video.write(np.flip(frame, axis=2))
        if save_last_frame:
            cv2.imwrite(frame_path + filename + ".jpg", np.flip(frame, axis=2))
        video.release()
    logging.info("Video and frames are saved.")
    return video
