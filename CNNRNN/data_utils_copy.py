import os
import numpy as np
import pandas as pd
from fastparquet import ParquetFile
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch
import math

class DataUtils:
    @staticmethod
    def load_data_by_name(gamename='', folder_path='/data/ysun209/VR.net/parquet/'):
        """
        gamename: str, the name of one game to load data for, if empty, load all data.
        return: pd.DataFrame, the data loaded from the parquet files.
        gamename = '3D_Play_House' or '98_Escapes' or 'Airplane_Obstacle' or 'American_Idol' or 
                'Arena_Clash' or 'Army_Men' or 'Barbie' or 'Barnyard' or 'Bobber_Bay_Fishing' or 'Bonnie_Revenge' 
                or 'Born_With_Power' or 'Breakneck_Canyon' or 'Canyon_Runners' or 'Cartoon_Wars' or 'Circle_Kawaii' 
                or 'Citadel' or 'City_Parkour' or 'Creature_Feature' or 'Delivery_Dash' or 'Earth_Gym' or
                'Escape_Puzzle_Mansion' or 'Fight_the_Night' or 'Flight_Squad' or 'Frisbee_Golf' or 'Fun_House' or
                'Geometry_Gunners' or 'Giant_Paddle_Golf' or 'Halloween_Wars' or 'Horizon_Boxing' or 'HoverTag' or
                'Jail_Simulator' or 'Junior_Chef' or 'Kawaii_Daycare' or 'Kawaii_Fire_Station' or 'Kawaii_House' or
                'Kawaii_Playroom' or 'Kawaii_Police_Station' or 'Kowloon' or 'Land_Beyond' or 'Live_Sandbox' or
                'Man_of_Moon_Mountain' or 'Mars_Miners' or 'MB_Deja_Vu' or 'Mech_Playground' or 'Mega_Tasty_BBQ' or
                'Meta_Pizza_Hut_Classic' or 'Metablocks_Adventure' or 'Metdonalds' or 'NBA_Arena' or 'New_Olympus' or
                'Octopus_Bash' or 'Out_Of_Control' or 'Pirate_Life' or 'Puddles_Theme_Park' or 'Puddles_Water_Park' or
                'Red_Dead' or 'Retro_Zombie' or 'Roommate' or 'Scifi_Sandbox' or 'Sky_High_Trampoline_Park' or 'Slash' or
                'Slash_RPG' or 'Spy_School' or 'Super_Rumble' or 'Superhero_Arena' or 'The_aquarium' or 'Titanic_Simulation'
                or 'UFO_crash_site_venue' or 'Venues' or 'VR_Bank' or 'VR_Classroom' or 'Waffle_Restaurant' or 'Wake_the_Robot'
                or 'Walking_Dead' or 'Water_Battling' or 'Western_Skies_RPG' or 'Wild_Quest' or 'Wizard_Sandbox' or
                'Wood_Warehouse' or 'Zombie' or 'Zoo_Chef_Challenge'
        """
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
    
    @staticmethod
    def ml_format(df, frame_count, cols_to_predict, cols_to_keep):
        """
        Formats a (formatted) data frame for training, validating and testing models
        df: dataframe to be formatted
        frame_count: number of frames to return with each item (1 will return the current frame, > 1 will return the current and previous frames)
        cols_to_predict: list of column names (from formated dataset) that are treated as labels/prediction targets
        """
        #Remove uneeded columns
        df = df[["game_session", "frame"] + cols_to_keep + cols_to_predict]
        
        #Add column for each additional previous frame
        for i in range(1, frame_count):
            df[f"frame_{i}"] = df.groupby("game_session")["frame"].shift(i)
            
        #reorder cols and remove rows with not enough previous frames
        df = df[["game_session", "frame"] + [f"frame_{i}" for i in range(1, frame_count)] + cols_to_keep + cols_to_predict]      
        df = df.dropna()
        
        #change name of frame col to match previous frame col name
        if frame_count > 1:df = df.rename(columns = {"frame" : "frame_0"})
        
        return df
    
    
    @staticmethod
    def write_to_txt(file_name, rows):
        with open(file_name, 'w') as f:
            f.writelines([row + "\n" for row in rows])
            
    @staticmethod
    def read_txt(file_name):
        with open(file_name, "r") as f:
            lines = [line.strip("\n") for line in f.readlines()]
        
        return lines
    
    @staticmethod
    def create_session_sets(df, train_size=0.7, val_size=0.15):
        """
        Creates txt files recording the game sessions in train, val and test set
        Only includeds sessions in the supplied df
        test_size is assumes to be 1 - train_size - val_size
        """
        all_sessions = df["game_session"].unique()
        np.random.shuffle(all_sessions)
        
        train_stop_index = math.floor(len(all_sessions) * train_size)
        val_stop_index = train_stop_index + math.ceil(len(all_sessions) * val_size)
        
        train_set = all_sessions[:train_stop_index]
        val_set = all_sessions[train_stop_index:val_stop_index]
        test_set = all_sessions[val_stop_index:]
        
        DataUtils.write_to_txt("train.txt", train_set)
        DataUtils.write_to_txt("val.txt", val_set)
        DataUtils.write_to_txt("test.txt", test_set)
    
class DatasetTemplate(Dataset):
    def __init__(self, frame_count=1, cols_to_predict=None, cols_to_keep=None, transform=None, target_transform=None):
        """
        Dataset template to be used for making spesific datasets, NOT TO IMPLEMENTED, ONLY INHERITED
        frame_count: number of frames to return with each item (1 will return the current frame, > 1 will return the current and previous frames)
        cols_to_predict: list of column names (from formated dataset) that are treated as labels/prediction targets
        cols_to_keep: list of column names to include with images as a models input
        transform: optional transformation applied to (torch 2 or 3D tensor) images 
        target_transform: optional transformation to be applied to target 1D tensor
        """
        #Default value for cols
        if cols_to_predict is None: cols_to_predict = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y"]
        if cols_to_keep is None: cols_to_keep = []
     
        #Helper function used to find location of stored images
        self.get_im_path = lambda session_name, frame: f"/data/ysun209/VR.net/videos/{session_name}/video/{frame}.jpg"  
        
        self.frame_count = frame_count
        self.cols_to_predict = cols_to_predict
        self.cols_to_keep = cols_to_keep
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        """
        Returns a single (x, y) dataset item with the given index
        If len(cols_to_keep) > 0, then x is a tuple of the form (image, control_vector), otherwise its just the image
        If frame_count > 1, then image is a 3D tensor, otherwise its a 2D tensor
        Control vector and y are both 1D tensors
        """
        data_row = self.df.iloc[index]
        session = data_row["game_session"]
        
        if self.frame_count == 1:
            #Read single image
            x = read_image(self.get_im_path(session, data_row["frame"]))
        else:
            #read multiple images
            x = []
            for i in range(0, self.frame_count):
                x.append(read_image(self.get_im_path(session, int(data_row[f"frame_{i}"]))))
            x = torch.from_numpy(np.array(x))
            
        if len(self.cols_to_keep):
            control_vector = data_row[self.cols_to_keep]
            control_vector = torch.from_numpy(control_vector.to_numpy().astype(float))
            x = (x, control_vector)
        
        #read target
        y = data_row[self.cols_to_predict]
        y = torch.from_numpy(y.to_numpy().astype(float))
        
        #apply transformations
        if self.transform: x = self.transform(x)
        if self.target_transform: y = self.target_transform(y)
        
        return x, y
    

class SingleSessionDataset(DatasetTemplate):
     def __init__(self, session_name, frame_count = 1, cols_to_predict=None, cols_to_keep=None, transform=None, target_transform=None):
        """
        Pytorch dataset for a single session of a game game
        game_name: name of the game to create the dataset around e.g. 'Barbie'
        frame_count: number of frames to return with each item (1 will return the current frame, > 1 will return the current and previous frames)
        cols_to_predict: list of column names (from formated dataset) that are treated as labels/prediction targets
        transform: optional transformation applied to (torch 2 or 3D tensor) images 
        target_transform: optional transformation to be applied to target 1D tensor
        """
        super().__init__(frame_count, cols_to_predict, cols_to_keep, transform, target_transform)
        
        #Create and format the dataframe
        self.df = DataUtils.load_data_by_name(session_name.split("_")[2])
        self.df = DataUtils.format_dataset(self.df)
        self.df = self.df[self.df["game_session"] == session_name]
        self.df = DataUtils.ml_format(self.df, self.frame_count, self.cols_to_predict, self.cols_to_keep)
        

class SingleGameDataset(DatasetTemplate):
    def __init__(self, game_name, session_set=None, frame_count = 1, cols_to_predict=None, cols_to_keep=None, transform=None, target_transform=None):
        """
        Pytorch dataset for a single game
        game_name: name of the game to create the dataset around e.g. 'Barbie'
        session_set: list of game session names to include
        frame_count: number of frames to return with each item (1 will return the current frame, > 1 will return the current and previous frames)
        cols_to_predict: list of column names (from formated dataset) that are treated as labels/prediction targets
        transform: optional transformation applied to (torch 2 or 3D tensor) images 
        target_transform: optional transformation to be applied to target 1D tensor
        """
        super().__init__(frame_count, cols_to_predict, cols_to_keep, transform, target_transform)
        
        #Create and format the dataframe
        self.df = DataUtils.load_data_by_name(game_name)
        self.df = DataUtils.format_dataset(self.df)
        
        #if no session set provided, use all sessions
        if session_set is None: session_set = self.df["game_session"].unique()
        self.df = self.df[self.df["game_session"].isin(session_set)]
        
        self.df = DataUtils.ml_format(self.df, self.frame_count, self.cols_to_predict, self.cols_to_keep)

class MultiGameDataset(DatasetTemplate):
    def __init__(self, game_names, session_set=None, frame_count = 1, cols_to_predict=None, cols_to_keep=None, transform=None, target_transform=None):
        """
        Pytorch dataset for a single game
        game_names: list of game names to create the dataset around e.g. '[Barbie]'
        session_set: list of game session names to include
        frame_count: number of frames to return with each item (1 will return the current frame, > 1 will return the current and previous frames)
        cols_to_predict: list of column names (from formated dataset) that are treated as labels/prediction targets
        transform: optional transformation applied to (torch 2 or 3D tensor) images 
        target_transform: optional transformation to be applied to target 1D tensor
        """
        super().__init__(frame_count, cols_to_predict, cols_to_keep, transform, target_transform)
        
        dfs = []
        #Create and format the dataframe
        for game in game_names:
            single_df = DataUtils.load_data_by_name(game)
            single_df = DataUtils.format_dataset(single_df)

            dfs.append(single_df)
        
        self.df = pd.concat(dfs, ignore_index=True)
        
        #if no session set provided, use all sessions
        if session_set is None: session_set = self.df["game_session"].unique()
        self.df = self.df[self.df["game_session"].isin(session_set)]
        
        self.df = DataUtils.ml_format(self.df, self.frame_count, self.cols_to_predict, self.cols_to_keep)


if __name__ == "__main__":
    #Demo for creating train, val and test sets for a game
    
    train_sessions = DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/train.txt")
    val_sessions = DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/val.txt")
    test_sessions = DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/test.txt")
    
    barbie_train_set = SingleGameDataset("Barbie", train_sessions)
    barbie_val_set = SingleGameDataset("Barbie", val_sessions)
    barbie_test_set = SingleGameDataset("Barbie", test_sessions)
    
    print(f"Items in train set: {len(barbie_train_set)}")
    print(f"Items in val set: {len(barbie_val_set)}")
    print(f"Items in test set: {len(barbie_val_set)}")