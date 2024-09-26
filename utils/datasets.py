import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from utils.data_utils import DataUtils


class DatasetTemplate(Dataset):
    def __init__(self, frame_count=1, cols_to_predict=None, cols_to_keep=None, transform=None, target_transform=None,
                 parquet_folder_path='/data/ysun209/VR.net/parquet/', video_folder_path='/data/ysun209/VR.net/videos/'):
        """
        Dataset template to be used for making spesific datasets, NOT TO IMPLEMENTED, ONLY INHERITED
        frame_count: number of frames to return with each item (1 will return the current frame, > 1 will return the current and previous frames)
        cols_to_predict: list of column names (from formated dataset) that are treated as labels/prediction targets
        cols_to_keep: list of column names to include with images as a models input
        transform: optional transformation applied to (torch 2 or 3D tensor) images
        target_transform: optional transformation to be applied to target 1D tensor
        parquet_folder_path: path to the parquet folder
        video_folder_path: path to the video folder
        """
        # Default value for cols
        if cols_to_predict is None: cols_to_predict = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x",
                                                       "thumbstick_right_y"]
        if cols_to_keep is None: cols_to_keep = []

        self.parquet_folder_path = parquet_folder_path
        self.video_folder_path = video_folder_path

        # Helper function used to find location of stored images
        self.get_im_path = lambda session_name, frame: f"{video_folder_path}/{session_name}/video/{frame}.jpg"

        self.frame_count = frame_count
        self.cols_to_predict = cols_to_predict
        self.cols_to_keep = cols_to_keep
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    @property
    def num_pred_features(self):
        return len(self.cols_to_predict)

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
            # Read single image
            x = read_image(self.get_im_path(session, data_row["frame"]))
        else:
            # read multiple images
            x = []
            for i in range(0, self.frame_count):
                x.append(read_image(self.get_im_path(session, int(data_row[f"frame_{i}"]))))
            x = torch.from_numpy(np.array(x))

        if len(self.cols_to_keep):
            control_vector = data_row[self.cols_to_keep]
            control_vector = torch.from_numpy(control_vector.to_numpy().astype(float))
            x = (x, control_vector)

        # read target
        y = data_row[self.cols_to_predict]
        y = torch.from_numpy(y.to_numpy().astype(float))

        # apply transformations
        if self.transform: x = self.transform(x)
        if self.target_transform: y = self.target_transform(y)

        return x, y


class SingleSessionDataset(DatasetTemplate):
    def __init__(self, session_name, frame_count=1, cols_to_predict=None, cols_to_keep=None, transform=None,
                 target_transform=None, parquet_folder_path='/data/ysun209/VR.net/parquet/',
                 video_folder_path='/data/ysun209/VR.net/videos/'):
        """
        Pytorch dataset for a single session of a game game
        game_name: name of the game to create the dataset around e.g. 'Barbie'
        frame_count: number of frames to return with each item (1 will return the current frame, > 1 will return the current and previous frames)
        cols_to_predict: list of column names (from formated dataset) that are treated as labels/prediction targets
        transform: optional transformation applied to (torch 2 or 3D tensor) images
        target_transform: optional transformation to be applied to target 1D tensor
        """
        super().__init__(frame_count, cols_to_predict, cols_to_keep, transform, target_transform, parquet_folder_path,
                         video_folder_path)

        # Create and format the dataframe
        self.df = DataUtils.load_data_by_name(session_name.split("_")[2], parquet_folder_path)
        self.df = DataUtils.format_dataset(self.df)
        self.df = self.df[self.df["game_session"] == session_name]
        self.df = DataUtils.ml_format(self.df, self.frame_count, self.cols_to_predict, self.cols_to_keep)


class SingleGameDataset(DatasetTemplate):
    def __init__(self, game_name, session_set=None, frame_count=1, cols_to_predict=None, cols_to_keep=None,
                 transform=None, target_transform=None, parquet_folder_path='/data/ysun209/VR.net/parquet/',
                 video_folder_path='/data/ysun209/VR.net/videos/'):
        """
        Pytorch dataset for a single game
        game_name: name of the game to create the dataset around e.g. 'Barbie'
        session_set: list of game session names to include
        frame_count: number of frames to return with each item (1 will return the current frame, > 1 will return the current and previous frames)
        cols_to_predict: list of column names (from formated dataset) that are treated as labels/prediction targets
        transform: optional transformation applied to (torch 2 or 3D tensor) images
        target_transform: optional transformation to be applied to target 1D tensor
        """
        super().__init__(frame_count, cols_to_predict, cols_to_keep, transform, target_transform, parquet_folder_path,
                         video_folder_path)

        # Create and format the dataframe
        self.df = DataUtils.load_data_by_name(game_name, parquet_folder_path)
        self.df = DataUtils.format_dataset(self.df)

        # if no session set provided, use all sessions
        if session_set is None: session_set = self.df["game_session"].unique()
        self.df = self.df[self.df["game_session"].isin(session_set)]

        self.df = DataUtils.ml_format(self.df, self.frame_count, self.cols_to_predict, self.cols_to_keep)


class MultiGameDataset(DatasetTemplate):
    def __init__(self, game_names, session_set=None, frame_count=1, cols_to_predict=None, cols_to_keep=None,
                 transform=None, target_transform=None, parquet_folder_path='/data/ysun209/VR.net/parquet/',
                 video_folder_path='/data/ysun209/VR.net/videos/'):
        """
        Pytorch dataset for a single game
        game_names: list of game names to create the dataset around e.g. '[Barbie]'
        session_set: list of game session names to include
        frame_count: number of frames to return with each item (1 will return the current frame, > 1 will return the current and previous frames)
        cols_to_predict: list of column names (from formated dataset) that are treated as labels/prediction targets
        transform: optional transformation applied to (torch 2 or 3D tensor) images
        target_transform: optional transformation to be applied to target 1D tensor
        parquet_folder_path: path to the parquet folder
        video_folder_path: path to the video folder
        """
        super().__init__(frame_count, cols_to_predict, cols_to_keep, transform, target_transform, parquet_folder_path,
                         video_folder_path)

        dfs = []
        # Create and format the dataframe
        for game in game_names:
            single_df = DataUtils.load_data_by_name(game, parquet_folder_path)
            single_df = DataUtils.format_dataset(single_df)

            dfs.append(single_df)

        self.df = pd.concat(dfs, ignore_index=True)

        # if no session set provided, use all sessions
        if session_set is None: session_set = self.df["game_session"].unique()
        self.df = self.df[self.df["game_session"].isin(session_set)]

        self.df = DataUtils.ml_format(self.df, self.frame_count, self.cols_to_predict, self.cols_to_keep)


class SingleGameControlDataset(Dataset):

    def __init__(self, game_name, session_set=None, frame_count=1, cols_to_keep=None,
                 transform=None, target_transform=None, parquet_folder_path='/data/ysun209/VR.net/parquet/'):
        """
        Pytorch dataset for a single game, only including control data - no images

        Intended for the C-RNN-GAN model

        game_name: name of the game to create the dataset around e.g. 'Barbie'
        session_set: list of game session names to include`
        frame_count: number of frames to return with each item (1 will return the current frame, > 1 will return the current and previous frames)
        cols_to_predict: list of column names (from formated dataset) that are treated as labels/prediction targets
        transform: optional transformation applied to (torch 2 or 3D tensor) images
        target_transform: optional transformation to be applied to target 1D tensor
        """
        assert cols_to_keep is not None and len(cols_to_keep) > 0, "cols_to_keep must be provided and not empty"

        self.cols_to_keep = cols_to_keep
        self.frame_count = frame_count

        self.transform = transform
        self.target_transform = target_transform

        # Create and format the dataframe
        self.df = DataUtils.load_data_by_name(game_name, parquet_folder_path)
        self.df = DataUtils.format_dataset(self.df)

        # if no session set provided, use all sessions
        if session_set is None: session_set = self.df["game_session"].unique()
        self.df = self.df[self.df["game_session"].isin(session_set)]

        self.df = DataUtils.ml_format(self.df, frame_count=1, cols_to_predict=[], cols_to_keep=self.cols_to_keep)

    def __getitem__(self, index: int):
        """
        Returns a single (control, _) dataset item with the given index,

        There is no label to return, so the second item is a dummy tensor

        @param index: index of the item to return
        @return: (control, _)
        """
        data_frame_slice = self.df.iloc[index: index + self.frame_count]

        x = data_frame_slice[self.cols_to_keep]
        x = torch.from_numpy(x.to_numpy().astype(np.float32))

        # apply transformations
        if self.transform: x = self.transform(x)

        assert x.shape[
                   0] == self.frame_count, f"Expected x to have shape ({self.frame_count}, num_features), got {x.shape}"

        return x, torch.tensor([0])  # x has shape (seq_len, num_features), y is a dummy tensor

    @property
    def num_features(self):
        return len(self.cols_to_keep)

    def __len__(self):
        return len(self.df) - self.frame_count + 1


if __name__ == "__main__":
    # Demo for creating train, val and test sets for a game

    train_sessions = DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/train.txt")
    val_sessions = DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/val.txt")
    test_sessions = DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/test.txt")

    barbie_train_set = SingleGameDataset("Barbie", train_sessions)
    barbie_val_set = SingleGameDataset("Barbie", val_sessions)
    barbie_test_set = SingleGameDataset("Barbie", test_sessions)

    print(f"Items in train set: {len(barbie_train_set)}")
    print(f"Items in val set: {len(barbie_val_set)}")
    print(f"Items in test set: {len(barbie_val_set)}")
