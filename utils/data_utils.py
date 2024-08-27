import os
from ast import literal_eval
import numpy as np
import pandas as pd
from fastparquet import ParquetFile

from tqdm import tqdm

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
    

#all the fields on intrest in the formatted dataset
all_fields = ["player_id",
          "order",
          "game_name",
          "frame_index",
          
          "head_pos_x",
          "head_pos_y",
          "head_pos_z",
          "left_controller_pos_x",
          "left_controller_pos_y",
          "left_controller_pos_z",
          "right_controller_pos_x",
          "right_controller_pos_y",
          "right_controller_pos_z",
          "head_vel_x",
          "head_vel_y",
          "head_vel_z",
          "left_controller_vel_x",
          "left_controller_vel_y",
          "left_controller_vel_z",
          "right_controller_vel_x",
          "right_controller_vel_y",
          "right_controller_vel_z",
          
          "head_dir_a",
          "head_dir_b",
          "head_dir_c",
          "head_dir_d",
          "left_controller_dir_a",
          "left_controller_dir_b",
          "left_controller_dir_c",
          "left_controller_dir_d",
          "right_controller_dir_a",
          "right_controller_dir_b",
          "right_controller_dir_c",
          "right_controller_dir_d",
          "head_angvel_x",
          "head_angvel_y",
          "head_angvel_z",
          "left_controller_angvel_x",
          "left_controller_angvel_y",
          "left_controller_angvel_z",
          "right_controller_angvel_x",
          "right_controller_angvel_y",
          "right_controller_angvel_z",
          
          "Index_trigger_left",
          "Index_trigger_right",
          "Hand_trigger_left",
          "Hand_trigger_right",
          "Thumbstick_left_x",
          "Thumbstick_left_y",
          "Thumbstick_right_x",
          "Thumbstick_right_y",
          
          "Buttons",
          "Touches",
          "NearTouches"
          ]

#the fields that the model should receive as input and output
input_control_fields = all_fields[4:13] + all_fields[22:34]
output_control_fields = all_fields[13:22] + all_fields[34:-3]

def dataset_stats(rows):
    """Computes the min, max and average of each field in the dataset."""

    mins = np.zeros((50,), dtype=float)
    avgs = np.zeros((50,), dtype=float)
    maxs = np.zeros((50,), dtype=float)
    count = 0

    for row in tqdm(rows, total=len(rows)):
        row_values = list(map(float, row[4:]))
        avgs += np.array(row_values)
        maxs = np.maximum(maxs, np.array(row_values))
        mins = np.minimum(mins, np.array(row_values))
        
        count += 1

    for i, name in enumerate(all_fields[4:]):
        print(f"{name}:")
        print(f"min: {mins[i]}")
        print(f"avg: {avgs[i]/count}")
        print(f"max: {maxs[i]}\n")