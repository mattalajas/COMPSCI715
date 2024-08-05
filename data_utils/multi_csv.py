import csv
import os
import pandas as pd

class MultiCSV:
    """Used to abstract reading from multiple csvs, this can be treated as a single large csv that can be indexed"""
    def __init__(self, csv_dir):
        #setup of csv vars
        self.csv_dir = csv_dir
        self.csv_paths = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
        self.num_of_csvs = len(self.csv_paths)
        self.hash_func = lambda x: hash(x) % self.num_of_csvs
        
        #reading name and indices txt
        with open(f"{csv_dir}/frames.txt", "r") as frame_txt:
            self.frames = frame_txt.read().splitlines()
            
        with open(f"{csv_dir}/indices.txt", "r") as indices_txt:
            self.indices = indices_txt.read().splitlines()
            self.indices = list(map(int, self.indices))
        
        self.num_of_items = len(self.frames)
        
        
    def __getitem__(self, index):
        """Returns the row at the given (global) index"""
        name = self.frames[index]
        index_in_csv = self.indices[index]
        hash_value = self.hash_func(name)
        
        csv_file = pd.read_csv(f"{self.csv_dir}/{self.csv_paths[hash_value]}")
        row = csv_file.iloc[index_in_csv]
        return row
    
    def __len__(self):
        return self.num_of_items
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= self.num_of_items: raise StopIteration
        row = self[self.i]
        self.i += 1
        return row

        