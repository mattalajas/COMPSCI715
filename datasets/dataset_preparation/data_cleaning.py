import zipfile
import os
import pandas as pd

# Unzip video files
def extract_video_files(zip_file_path, output_dir, file_extension):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith(file_extension):
                zip_ref.extract(file_info.filename, output_dir)

folder_path = '/data/ysun209/VR.net/game_sessions'
for file_name in os.listdir(folder_path):
    if file_name.endswith('.zip'):
        print(folder_path+"/"+file_name)
        zip_file_path = folder_path+"/"+file_name
        output_dir = '/data/ysun209/VR.net/videos'
        file_extension = '.jpg'
        extract_video_files(zip_file_path, output_dir, file_extension)
        