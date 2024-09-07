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
        
# Store csv files to parquet files
class ZipFolderLister:
    def __init__(self, zip_file_path):
        self.zip_file_path = zip_file_path

    def list_top_level_directories(self):
        """Returns a list of top-level directories in the ZIP file."""
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            # List all files and folders in the ZIP file
            all_files = zip_ref.namelist()
            
            # Use a set to track top-level directories
            top_level_dirs = set()
            
            for file in all_files:
                # Check if the entry is a directory and is at the top level
                if file.endswith('/'):
                    # Split the path and ensure it does not contain additional slashes
                    if file.count('/') == 1:
                        top_level_dirs.add(file)
                        
            return sorted(top_level_dirs)  # Return sorted list of top-level directories


    def find_csv_files_in_top_dirs(self):
        """Returns a list of `data_file.csv` files located under top-level directories in the ZIP file."""
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            top_level_dirs = self.list_top_level_directories()
            
            # List to store paths of data_file.csv
            csv_files = []
            
            # Check each top-level directory for data_file.csv
            for top_dir in top_level_dirs:
                for file in all_files:
                    if file.startswith(top_dir) and file.endswith('data_file.csv'):
                        csv_files.append(file)
            
            return sorted(csv_files)  # Return sorted list of CSV file paths

    def csv_file_size_after_unzip(self):
        """Store dataframe into parquet format"""
        part_size = 0
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            csv_files = self.find_csv_files_in_top_dirs()

            part_size = 0
            for csv_file in csv_files:
                file_info = zip_ref.getinfo(csv_file).file_size
                part_size = part_size + file_info
        return part_size

    def store_dataframe_into_parquet(self):
        """Store dataframe into parquet format"""
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            csv_files = self.find_csv_files_in_top_dirs()

            for csv_file in csv_files:
                print(csv_file)
                try:
                    with zip_ref.open(csv_file) as file:
                        # Read the content of the file and decode it
                        # content = csv_file.read().decode('utf-8')
                        # reader = csv.reader(io.StringIO(content))
                        df = pd.read_csv(file, usecols=['frame', 'timestamp', 'ConnectedControllerTypes', 'Buttons', 'Touches', 
                           'NearTouches', 'IndexTrigger', 'HandTrigger', 'Thumbstick', 'video', 
                           'head_dir', 'head_pos', 'head_vel', 'head_angvel', 'left_eye_dir', 
                           'left_eye_pos', 'left_eye_vel', 'left_eye_angvel', 'right_eye_dir', 
                           'right_eye_pos', 'right_eye_vel', 'right_eye_angvel'])
                        df['game_name'] = os.path.splitext(os.path.basename(self.zip_file_path))[0]
                        df['game_session'] = os.path.basename(os.path.dirname(csv_file))
                        df.to_parquet("/data/ysun209/VR.net/parquet/"+csv_file.split('/')[0] + ".parquet")
                except:
                    pass

def save_zip_to_parquet(zip_file):
    print(zip_file)
    lister = ZipFolderLister(zip_file)
    lister.store_dataframe_into_parquet()


folder_path = '/data/ysun209/VR.net/game_sessions'
for file_name in os.listdir(folder_path):
    if file_name.endswith('.zip'):
        save_zip_to_parquet(folder_path+"/"+file_name)
