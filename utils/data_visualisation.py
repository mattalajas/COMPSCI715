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

df[['Thumbstick_0', 'Thumbstick_1', 'Thumbstick_2', 'Thumbstick_3']] = pd.DataFrame(df['Thumbstick'].tolist(), index=df.index)

image_folder = 'video'

x_data = []
y_data = []
x_data2 = []
y_data2 = []

fig, (ax_image, ax_trace1, ax_trace2) = plt.subplots(1, 3, figsize=(15, 5))

image_plot = ax_image.imshow(np.zeros((256, 256)), cmap='gray')
ax_image.axis('off')

line, = ax_trace1.plot([], [], 'b-')
current_point, = ax_trace1.plot([], [], 'ro', markersize=8)
ax_trace1.set_xlim(-1, 1)
ax_trace1.set_ylim(-1, 1)
ax_trace1.set_title('XY Trace: Thumbstick_0 & Thumbstick_1')
ax_trace1.set_xlabel('Thumbstick_0 (X)')
ax_trace1.set_ylabel('Thumbstick_1 (Y)')

line2, = ax_trace2.plot([], [], 'g-')
current_point2, = ax_trace2.plot([], [], 'ro', markersize=8)
ax_trace2.set_xlim(-1, 1)
ax_trace2.set_ylim(-1, 1)
ax_trace2.set_title('XY Trace: Thumbstick_2 & Thumbstick_3')
ax_trace2.set_xlabel('Thumbstick_2 (X)')
ax_trace2.set_ylabel('Thumbstick_3 (Y)')

index = 0

def update(frame):
    global index
    if index < len(df):
        frame_number = df['frame'].iloc[index]
        image_path = os.path.join(image_folder, f'{frame_number}.jpg')
        if os.path.exists(image_path):
            img = Image.open(image_path)
            image_plot.set_data(img)
        thumbstick_0 = df['Thumbstick_0'].iloc[index]
        thumbstick_1 = df['Thumbstick_1'].iloc[index]
        x_data.append(thumbstick_0)
        y_data.append(thumbstick_1)
        line.set_data(x_data, y_data)
        current_point.set_data([thumbstick_0], [thumbstick_1])
        thumbstick_2 = df['Thumbstick_2'].iloc[index]
        thumbstick_3 = df['Thumbstick_3'].iloc[index]
        x_data2.append(thumbstick_2)
        y_data2.append(thumbstick_3)
        line2.set_data(x_data2, y_data2)
        current_point2.set_data([thumbstick_2], [thumbstick_3])
        index += 1
    return image_plot, line, current_point, line2, current_point2

ani = FuncAnimation(fig, update, frames=np.arange(0, len(df)), interval=100)

plt.tight_layout()
plt.show()
