import os
import numpy as np
import pandas as pd
from fastparquet import ParquetFile
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch

class DataUtils:
    @staticmethod
    def load_data_by_name(gamename=''):
        """
        gamename: str, the name of one game to load data for, if empty, load all data.
        return: pd.DataFrame, the data loaded from the parquet files.
        gamename = '3D_Play_House' or '98_Escapes' or 'Airplane_Obstacle' or 'American_Idol' or 'Arena_Clash' or 'Army_Men' or 'Barbie' or 'Barnyard' or 'Bobber_Bay_Fishing' or 'Bonnie_Revenge' or 'Born_With_Power' or 'Breakneck_Canyon' or 'Canyon_Runners' or 'Cartoon_Wars' or 'Circle_Kawaii' or 'Citadel' or 'City_Parkour' or 'Creature_Feature' or 'Delivery_Dash' or 'Earth_Gym' or 'Escape_Puzzle_Mansion' or 'Fight_the_Night' or 'Flight_Squad' or 'Frisbee_Golf' or 'Fun_House' or 'Geometry_Gunners' or 'Giant_Paddle_Golf' or 'Halloween_Wars' or 'Horizon_Boxing' or 'HoverTag' or 'Jail_Simulator' or 'Junior_Chef' or 'Kawaii_Daycare' or 'Kawaii_Fire_Station' or 'Kawaii_House' or 'Kawaii_Playroom' or 'Kawaii_Police_Station' or 'Kowloon' or 'Land_Beyond' or 'Live_Sandbox' or 'Man_of_Moon_Mountain' or 'Mars_Miners' or 'MB_Deja_Vu' or 'Mech_Playground' or 'Mega_Tasty_BBQ' or 'Meta_Pizza_Hut_Classic' or 'Metablocks_Adventure' or 'Metdonalds' or 'NBA_Arena' or 'New_Olympus' or 'Octopus_Bash' or 'Out_Of_Control' or 'Pirate_Life' or 'Puddles_Theme_Park' or 'Puddles_Water_Park' or 'Red_Dead' or 'Retro_Zombie' or 'Roommate' or 'Scifi_Sandbox' or 'Sky_High_Trampoline_Park' or 'Slash' or 'Slash_RPG' or 'Spy_School' or 'Super_Rumble' or 'Superhero_Arena' or 'The_aquarium' or 'Titanic_Simulation' or 'UFO_crash_site_venue' or 'Venues' or 'VR_Bank' or 'VR_Classroom' or 'Waffle_Restaurant' or 'Wake_the_Robot' or 'Walking_Dead' or 'Water_Battling' or 'Western_Skies_RPG' or 'Wild_Quest' or 'Wizard_Sandbox' or 'Wood_Warehouse' or 'Zombie' or 'Zoo_Chef_Challenge'
        """
        folder_path = '/data/ysun209/VR.net/parquet/'
        file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        file_paths = [os.path.join(folder_path, f) for f in file_names]
        df_list = []
        for file_path in file_paths:
            if gamename in file_path:
                # df = ParquetFile(file_path).to_pandas()
                # df = ParquetFile(file_path).to_pandas(columns=['game_name','game_session','frame', 'timestamp',
                #                                                'ConnectedControllerTypes', 'Buttons', 'Touches', 
                #                                                'NearTouches', 'IndexTrigger', 'HandTrigger', 'Thumbstick', 'video', 
                #                                                'head_dir', 'head_pos', 'head_vel', 'head_angvel', 'left_eye_dir', 
                #                                                'left_eye_pos', 'left_eye_vel', 'left_eye_angvel', 'right_eye_dir', 
                #                                                'right_eye_pos', 'right_eye_vel', 'right_eye_angvel'])
                df = ParquetFile(file_path).to_pandas()#columns=['game_name','game_session','frame', 'timestamp', 'video', 'Thumbstick'])

                df_list.append(df)
        return pd.concat(df_list, ignore_index=True)
    
    @staticmethod
    def split_series(s):
        """
        Splits a series (df column) that contains a multivalued attribute (e.g. head_dir)
        s: pandas series to be split
        """
        return s.str.strip("()").str.split(", ", expand=True).astype(float)
    
    @staticmethod
    def append_split_columns(df, name, new_names):
        """
        Splits a multivalued column and adds it to dataframe
        df: dataframe to add new columns to
        name: string name of the column to be split
        new_names: list of strings corresponding to the names of the new split columns
        """
        df[new_names] = DataUtils.split_series(df[name])
        df.drop(name, axis=1, inplace = True)

    
    @staticmethod
    def format_dataset(df, include_buttons=False):
        """
        Formats the dataset to remove unwanted attributes and split multivalued attributes
        df: pandas dataframe to format
        include_buttons: whether to include the Buttons, Touches, NearTouches attributes
        """
        
        cols_to_keep = ["game_name","game_session", "frame", "video",
                        "head_dir", "head_pos", "head_vel", "head_angvel",
                        "IndexTrigger","HandTrigger","Thumbstick"]
                        #"left_controller_dir","left_controller_pos","left_controller_vel","left_controller_angvel",
                        #"right_controller_dir","right_controller_pos","right_controller_vel","right_controller_angvel"]
        
        if include_buttons: cols_to_keep += ["Buttons","Touches","NearTouches"]
        
        #remove unwanted columns
        df = df.drop(columns = [c for c in df.columns if not c in cols_to_keep], axis=1)
        
        #remove rows with no images
        df = df.dropna(subset=["video"])
        
        #helper functions for creating new column names
        two_col_names = lambda name: [name + "_" + c for c in ["left", "right"]]
        three_col_names = lambda name: [name + "_" + c for c in ["x", "y", "z"]]
        four_col_names = lambda name: [name + "_" + c for c in ["a", "b", "c", "d"]]
        four_col_names_2 = lambda name: [name + "_" + c for c in ["left_x", "left_y", "right_x", "right_y"]]
    
        #split head multivalued columns
        DataUtils.append_split_columns(df, "head_pos", three_col_names("head_pos"))
        DataUtils.append_split_columns(df, "head_vel", three_col_names("head_vel"))
        DataUtils.append_split_columns(df, "head_dir", four_col_names("head_dir"))
        DataUtils.append_split_columns(df, "head_angvel", three_col_names("head_angvel"))
        
        #split controller multivalued columns
        DataUtils.append_split_columns(df, "IndexTrigger", two_col_names("index_trigger"))
        DataUtils.append_split_columns(df, "HandTrigger", two_col_names("hand_trigger"))
        DataUtils.append_split_columns(df, "Thumbstick", four_col_names_2("thumbstick"))
        
        #reorder the columns of the dataframe
        col_order = cols_to_keep[:4] +\
                    [c for c in df.columns if c not in cols_to_keep] +\
                    (cols_to_keep[-3:] if include_buttons else [])
                    
        df = df[col_order]          
        return df
    
    
class SingleGameDataset(Dataset):
    def __init__(self, game_name, frame_count = 1, cols_to_predict=None, transform=None, target_transform=None):
        """
        Pytorch dataset for a single game
        game_name: name of the game to create the dataset around e.g. 'Barbie'
        frame_count: number of frames to return with each item (1 will return the current frame, > 1 will return the current and previous frames)
        cols_to_predict: list of column names (from formated dataset) that are treated as labels/prediction targets
        transform: optional transformation applied to (torch 2 or 3D tensor) images 
        target_transform: optional transformation to be applied to target 1D tensor
        """
        #Default cols to predict
        if cols_to_predict is None: cols_to_predict = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y"]

        #set up df so each row has the current frame number and columns for previous frame numbers
        self.df = DataUtils.load_data_by_name(game_name)
        self.df = DataUtils.format_dataset(self.df)
        self.df = self.df[["game_session", "frame"] + cols_to_predict]
        for i in range(1, frame_count):
            self.df[f"frame_{i}"] = self.df.groupby("game_session")["frame"].shift(i)
            
        #reorder cols and remove rows with not enough previous frames
        self.df = self.df[["game_session", "frame"] + [f"frame_{i}" for i in range(1, frame_count)] + cols_to_predict]      
        self.df = self.df.dropna()
        
        #change name of frame col to match previous frame col name
        if frame_count > 1: self.df = self.df.rename(columns = {"frame" : "frame_0"})

        #helper function to build image dirs
        self.get_im_dir = lambda session_name, frame: f"/data/ysun209/VR.net/videos/{session_name}/video/{frame}.jpg"        
        
        self.frame_count = frame_count
        self.cols_to_predict = cols_to_predict
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        data_row = self.df.iloc[index]
        session = data_row["game_session"]
        
        if self.frame_count == 1:
            #Read single image
            image = read_image(self.get_im_dir(session, data_row["frame"]))
        else:
            #read multiple images
            image = []
            for i in range(0, self.frame_count):
                image.append(read_image(self.get_im_dir(session, int(data_row[f"frame_{i}"]))))
            image = torch.from_numpy(np.array(image))
        
        #read target
        target = data_row[self.cols_to_predict]
        target = torch.from_numpy(target.to_numpy().astype(float))
        
        #apply transformations
        if self.transform: image = self.transform(image)
        if self.target_transform: target = self.target_transform(target)
        
        return image, target