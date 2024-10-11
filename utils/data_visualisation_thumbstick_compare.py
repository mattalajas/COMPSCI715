import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image
import os
import ast

df = pd.read_csv('data_file.csv')

frame = df['frame']
thumbstick = df['Thumbstick']

df['Thumbstick'] = df['Thumbstick'].fillna('(0.0, 0.0, 0.0, 0.0)').apply(lambda x: ast.literal_eval(x))

df[['thumbstick_left_x', 'thumbstick_left_y', 'thumbstick_right_x', 'thumbstick_right_y']] = pd.DataFrame(df['Thumbstick'].tolist(), index=df.index)

real_df = df[['frame', 'thumbstick_left_x', 'thumbstick_left_y', 'thumbstick_right_x', 'thumbstick_right_y']]
# real_df = pd.read_csv('full_true.csv')

ModelName = 'CNN_RNN'
trained_df = pd.read_csv('full_cnn_rnn.csv')

# ModelName = 'Video Transformer'
# trained_df = pd.read_csv('full_vision_transformer.csv')

# ModelName = 'ResNet_LSTM'
# trained_df = pd.read_csv('full_resnet_lstm.csv')

# ModelName = 'ResNet'
# trained_df = pd.read_csv('full_resnet.csv')

# ModelName = 'GAIL_RNN'
# trained_df = pd.read_csv('full_gail_rnn.csv')

# ModelName = 'GAIL'
# trained_df = pd.read_csv('full_gail_simple.csv')

ModelName = 'GAIL-ResNet'
trained_df = pd.read_csv('full_gail_transfer.csv')




image_folder = 'video'

real_x_data = []
real_y_data = []
real_x_data2 = []
real_y_data2 = []

trained_x_data = []
trained_y_data = []
trained_x_data2 = []
trained_y_data2 = []

# fig, (ax_image, ax_trace1, ax_trace2 ) = plt.subplots(1, 3, figsize=(15, 5))

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
ax_image = axes[0, 0]    # First row, first column
ax_trace1 = axes[0, 1]   # First row, second column
ax_trace2 = axes[0, 2]   # First row, third column

ax_image1 = axes[1, 0]
ax_trace3 = axes[1, 1] 
ax_trace4 = axes[1, 2]


image_plot = ax_image.imshow(np.zeros((256, 256)), cmap='gray')
ax_image.axis('off')
ax_image1.axis('off')


line, = ax_trace1.plot([], [], 'b-')
current_point, = ax_trace1.plot([], [], 'ro', markersize=8)
ax_trace1.set_xlim(-1, 1)
ax_trace1.set_ylim(-1, 1)
ax_trace1.set_title('Original: Left Thumbstick')
ax_trace1.set_xlabel('thumbstick_left_x (X)')
ax_trace1.set_ylabel('thumbstick_left_y (Y)')

line2, = ax_trace2.plot([], [], 'g-')
current_point2, = ax_trace2.plot([], [], 'ro', markersize=8)
ax_trace2.set_xlim(-1, 1)
ax_trace2.set_ylim(-1, 1)
ax_trace2.set_title('Original: Right Thumbstick')
ax_trace2.set_xlabel('thumbstick_right_x (X)')
ax_trace2.set_ylabel('thumbstick_right_y (Y)')

line3, = ax_trace3.plot([], [], 'b-')
current_point3, = ax_trace3.plot([], [], 'ro', markersize=8)
ax_trace3.set_xlim(-1, 1)
ax_trace3.set_ylim(-1, 1)
ax_trace3.set_title(f'{ModelName}: Left Thumbstick')
ax_trace3.set_xlabel('thumbstick_left_x (X)')
ax_trace3.set_ylabel('thumbstick_left_y (Y)')

line4, = ax_trace4.plot([], [], 'g-')
current_point4, = ax_trace4.plot([], [], 'ro', markersize=8)
ax_trace4.set_xlim(-1, 1)
ax_trace4.set_ylim(-1, 1)
ax_trace4.set_title(f'{ModelName}: Right Thumbstick')
ax_trace4.set_xlabel('thumbstick_right_x (X)')
ax_trace4.set_ylabel('thumbstick_right_y (Y)')

index = 340
trained_thumbstick_left_x_prev = 0
trained_thumbstick_left_y_prev = 0
trained_thumbstick_right_x_prev = 0
trained_thumbstick_right_y_prev = 0

def update(frame):
    global index
    global trained_thumbstick_left_x_prev
    global trained_thumbstick_left_y_prev
    global trained_thumbstick_right_x_prev
    global trained_thumbstick_right_y_prev
    # if index < len(real_df):
    if index < 510:
        video_frame = real_df['frame'].iloc[index]
        frame_number = real_df['frame'].iloc[index]
        image_path = os.path.join(image_folder, f'{frame_number}.jpg')
        if os.path.exists(image_path):
            img = Image.open(image_path)
            image_plot.set_data(img)
        # Thumbstick 0 and Thumbstick 1
        real_thumbstick_left_x = real_df['thumbstick_left_x'].iloc[index]
        real_thumbstick_left_y = real_df['thumbstick_left_y'].iloc[index]
        real_x_data.append(real_thumbstick_left_x)
        real_y_data.append(real_thumbstick_left_y)
        # Keep only the last 100 points
        real_x_data[:] = real_x_data[-50:]
        real_y_data[:] = real_y_data[-50:]
        line.set_data(real_x_data, real_y_data)
        current_point.set_data([real_thumbstick_left_x], [real_thumbstick_left_y])
        # Thumbstick 2 and Thumbstick 3
        real_thumbstick_right_x = real_df['thumbstick_right_x'].iloc[index]
        real_thumbstick_right_y = real_df['thumbstick_right_y'].iloc[index]
        real_x_data2.append(real_thumbstick_right_x)
        real_y_data2.append(real_thumbstick_right_y)
        # Keep only the last 100 points
        real_x_data2[:] = real_x_data2[-50:]
        real_y_data2[:] = real_y_data2[-50:]
        line2.set_data(real_x_data2, real_y_data2)
        current_point2.set_data([real_thumbstick_right_x], [real_thumbstick_right_y])
        # Thumbstick 4 and Thumbstick 5
        try:
            trained_thumbstick_left_x =  trained_df.loc[(trained_df['frame'] == video_frame) & (trained_df['game_session'] == "125_2_Kawaii_Daycare")]['thumbstick_left_x'].iloc[0]
            trained_thumbstick_left_x_prev = trained_thumbstick_left_x
            trained_thumbstick_left_y =  trained_df.loc[(trained_df['frame'] == video_frame) & (trained_df['game_session'] == "125_2_Kawaii_Daycare")]['thumbstick_left_y'].iloc[0]
            trained_thumbstick_left_y_prev = trained_thumbstick_left_y
        except:
            trained_thumbstick_left_x = trained_thumbstick_left_x_prev
            trained_thumbstick_left_y = trained_thumbstick_left_y_prev
        if (trained_thumbstick_left_x**2 + trained_thumbstick_left_y**2) <= 0.1:
            trained_thumbstick_left_x = 0
            trained_thumbstick_left_y = 0
        else:
            trained_thumbstick_left_x = np.clip(trained_thumbstick_left_x, -0.99, 0.99)
            trained_thumbstick_left_y = np.clip(trained_thumbstick_left_y, -0.99, 0.99)
        trained_x_data.append(trained_thumbstick_left_x)
        trained_y_data.append(trained_thumbstick_left_y)
        # Keep only the last 100 points
        trained_x_data[:] = trained_x_data[-50:]
        trained_y_data[:] = trained_y_data[-50:]
        line3.set_data(trained_x_data, trained_y_data)
        current_point3.set_data([trained_thumbstick_left_x], [trained_thumbstick_left_y])
        # Thumbstick 6 and Thumbstick 7
        try:
            trained_thumbstick_right_x =  trained_df.loc[(trained_df['frame'] == video_frame) & (trained_df['game_session'] == "125_2_Kawaii_Daycare")]['thumbstick_right_x'].iloc[0]
            trained_thumbstick_right_x_prev = trained_thumbstick_right_x
            trained_thumbstick_right_y =  trained_df.loc[(trained_df['frame'] == video_frame) & (trained_df['game_session'] == "125_2_Kawaii_Daycare")]['thumbstick_right_y'].iloc[0]
            trained_thumbstick_right_y_prev = trained_thumbstick_right_y
        except:
            trained_thumbstick_right_x = trained_thumbstick_right_x_prev
            trained_thumbstick_right_y = trained_thumbstick_right_y_prev
        if (trained_thumbstick_right_x**2 + trained_thumbstick_right_y**2) <= 0.1:
            trained_thumbstick_right_x = 0
            trained_thumbstick_right_y = 0
        else:
            trained_thumbstick_right_x = np.clip(trained_thumbstick_right_x, -0.99, 0.99)
            trained_thumbstick_right_y = np.clip(trained_thumbstick_right_y, -0.99, 0.99)
        trained_x_data2.append(trained_thumbstick_right_x)
        trained_y_data2.append(trained_thumbstick_right_y)
        # Keep only the last 100 points
        trained_x_data2[:] = trained_x_data2[-50:]
        trained_y_data2[:] = trained_y_data2[-50:]
        line4.set_data(trained_x_data2, trained_y_data2)
        current_point4.set_data([trained_thumbstick_right_x], [trained_thumbstick_right_y])
        index += 1
    return image_plot, line, current_point, line2, current_point2, line3, current_point3, line4, current_point4


ani = FuncAnimation(fig, update, frames=np.arange(0, len(trained_df)), interval=100)

plt.tight_layout()
plt.show()
