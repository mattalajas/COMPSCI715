import csv
import os
import ast
from tqdm import tqdm
import sys

dataset_dir = os.path.abspath("../vr_dataset_sample/files")
output_dir = "dataset"

if not os.path.isdir(output_dir): os.mkdir(output_dir)

#create and/or empty frames.txt
frame_txt = open(f"{output_dir}/frames.txt", "w")
frame_txt.close()
frame_txt = open(f"{output_dir}/frames.txt", "a")

index_txt = open(f"{output_dir}/indices.txt", "w")
index_txt.close()
index_txt = open(f"{output_dir}/indices.txt", "a")

#create all the csvs
num_of_csvs = 10
hash_func = lambda x: hash(x) % num_of_csvs

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
        

fields = ["player_id",
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
          "head_angvel_a",
          "head_angvel_b",
          "head_angvel_c",
          "head_angvel_d",
          "left_controller_angvel_a",
          "left_controller_angvel_b",
          "left_controller_angvel_c",
          "left_controller_angvel_d",
          "right_controller_angvel_a",
          "right_controller_angvel_b",
          "right_controller_angvel_c",
          "right_controller_angvel_d",
          
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

new_csvs = [open(f"{output_dir}/dataset_{i}.csv", "w", newline="") for i in range(num_of_csvs)]
for new_csv in new_csvs: 
    writer = csv.DictWriter(new_csv, fields)
    writer.writeheader()
    new_csv.close()
    
row_counts = [0] * len(new_csvs)

#loop over all game sessions
for session_file in os.listdir(dataset_dir):
    player_id, order, *game_name = session_file.split("_")
    player_id = int(player_id)
    order = int(order)
    game_name = "_".join(game_name)
    
    print(f"Session: {player_id}_{order}_{game_name}")
    
    csv_file = open(os.path.join(dataset_dir, session_file, "data_file.csv"))
    image_dir = os.path.join(dataset_dir, session_file, "video")
    
    #loop over all rows in csv
    reader = csv.DictReader(csv_file)
    for row in reader:
        #if the row has no corrosponding images, skip it
        if not os.path.isfile(os.path.join(image_dir, row["frame"]+".jpg")):
            print(f"    skipping {player_id}_{order}_{game_name}_{row['frame']} - no image")
            continue
        
        hash_value = hash_func(f"{player_id}_{order}_{game_name}_{row['frame']}")
        
        try:
            new_row_values = [player_id,
                            order, 
                            game_name, 
                            row["frame"], 
                            
                            *ast.literal_eval(row["head_pos"]),
                            *ast.literal_eval(row["left_controller_pos"]),
                            *ast.literal_eval(row["right_controller_pos"]),

                            *ast.literal_eval(row["head_vel"]),
                            *ast.literal_eval(row["left_controller_vel"]),
                            *ast.literal_eval(row["right_controller_vel"]),

                            *ast.literal_eval(row["head_dir"]),
                            *ast.literal_eval(row["left_controller_dir"]),
                            *ast.literal_eval(row["right_controller_dir"]),

                            *ast.literal_eval(row["head_angvel"]),
                            *ast.literal_eval(row["left_controller_angvel"]),
                            *ast.literal_eval(row["right_controller_angvel"]),
                        
                            *ast.literal_eval(row["IndexTrigger"]),
                            *ast.literal_eval(row["HandTrigger"]),
                            *ast.literal_eval(row["Thumbstick"]),
                            
                            row["Buttons"],
                            row["Touches"],
                            row["NearTouches"]]
        except:
            print(f"    skipping {player_id}_{order}_{game_name}_{row['frame']} - missing values")
            continue
        
        new_row = {k:v for k,v in zip(fields, new_row_values)}
        
        with open(f"{output_dir}/dataset_{hash_value}.csv", "a", newline="") as new_csv:
            writer = csv.DictWriter(new_csv, fields)
            writer.writerow(new_row)
            row_counts[hash_value] += 1
        
        frame_txt.write(f"{player_id}_{order}_{game_name}_{row['frame']}\n")
        index_txt.write(f"{row_counts[hash_value]}\n")
      
    print()
  
#close all files
frame_txt.close()
index_txt.close()

print("\nFinished")