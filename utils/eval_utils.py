import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def resize_image(im, new_width=None, new_height=None):
    """Resizes an image while maintaining aspect ratio"""
    h = im.shape[0]
    w = im.shape[1]

    if new_width is None:
        ratio = new_height/h
        new_shape = (int(ratio * w), new_height)
    else:
        ratio = new_width/w
        new_shape = (new_width, int(ratio * h))

    return cv2.resize(im, new_shape, cv2.INTER_AREA)


def play(frames, resize=1080, fps=30, start_frame=None, end_frame=None, start_paused=True):
        """Play video with controls"""
        delay = round(1000/fps) if fps != 0 else 0
        first_index = 0 if start_frame is None else start_frame
        final_index = len(frames) - 1 if end_frame is None else end_frame
        is_paused = start_paused
        temp_delay = delay

        i = first_index
        while True:
            frame = frames[i]
            frame = resize_image(frame, new_width=resize) if frame.shape[1] > frame.shape[0] else resize_image(frame, new_height=resize)
            cv2.imshow("Video", frame)
            key = cv2.waitKey(temp_delay if not is_paused else 0)

            if key == ord('q'): #close the video
                break
            elif key == ord(" "): #pause
                is_paused = not is_paused
            elif key == ord('a'): #previous frame
                i = max(first_index, i - 1)
            elif key == ord('d'): #next frame
                i = min(i + 1, final_index)
            elif key == ord('w'): #restart
                i = first_index
            elif key == ord('s'): #end
                i = final_index
            elif key == ord('z'): #reduce fps
                temp_delay *= 2
            elif key == ord('x'): #reset fps
                temp_delay = delay
            elif key == ord('c'): #increase fps
                temp_delay = math.ceil(temp_delay/2)
            else:
                i = min(i + 1, final_index)
            
        cv2.destroyAllWindows()


