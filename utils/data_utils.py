import multi_csv
import numpy as np
from tqdm import tqdm

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