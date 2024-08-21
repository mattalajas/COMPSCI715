import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import csv
import ast
import sys


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


def plt_to_cv2(fig):
    """Converts a matplotlib figure to a cv2 image"""
    fig.canvas.draw()
    im = np.array(fig.canvas.renderer.buffer_rgba())
    im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
    
    return im


def stitch_images(im1, im2, left=True):
    """Attachs im1 to the left or right of im (im1 is resized to match im2)"""
    im1 = resize_image(im1, new_height=im2.shape[0])
    
    to_concat = [im2, im1] if left else [im1, im2]
    return cv2.hconcat(to_concat)    


if __name__ == "__main__":
    #code for testing the funcs above
    session_folder = r"c:\Users\kelha\Documents\Uni\CS715\vr_dataset_sample\files\5_2_Earth_Gym"
    csv_file = open(os.path.join(session_folder, "data_file.csv"))
    im_folder = os.path.join(session_folder, "video")
    
    images = []
    plots = []
    values = []
    
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)
        
    reader = csv.DictReader(csv_file)
    start_frame = 2600
    max_frames = 300
    for i, row in enumerate(reader):
        if int(row["frame"]) < start_frame: continue
        
        frame = cv2.imread(os.path.join(im_folder, f"{row['frame']}.jpg"))
        images.append(frame)
        
        thumbstick = ast.literal_eval(row["Thumbstick"])[:2]
        values.append(thumbstick)
        
        
        fig = plt.figure()
        plt.plot([p[0] for p in values], [p[1] for p in values])
        plt.title("Thumbstick left controls")
        plots.append(plt_to_cv2(fig))
        plt.close(fig)
        
        if i + 1 >= max_frames: break
        
        
    frames = []
    for im, pl in zip(images, plots):
        frames.append(stitch_images(im, pl, left=False))
        
    play(frames)
        
    
    