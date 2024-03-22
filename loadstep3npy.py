import cv2, os, sys, subprocess, platform, torch
from PIL import Image
import numpy as np

imgs=np.load('temp/wholebody_60s.mp4_img_batchs.npy',allow_pickle=True)
# 获取视频.shape的宽度、高度和帧率
print(imgs[0].shape)
frame_height,frame_width =imgs[0][0].shape[:2]
print(frame_height,frame_width)
fps = 30

# 定义VideoWriter对象来保存视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编码方式
output_path = 'result/step3.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


for group in imgs:
    for img in group:
        # frame=Image.fromarray(img)
        frame=(img[:,:,3:]*255.).astype(np.uint8)
        # 写入帧
        frame=cv2.resize(frame,(frame_width, frame_height))
        out.write(frame)

# 释放对象
out.release()
cv2.destroyAllWindows()
command = 'ffmpeg -loglevel error -y -i {} -i {} -strict -2 -q:v 1 {}'.format('examples/audio/tts_hailan_030701_silence_30s.wav', 'result/step3.mp4', 'result/step3addaudio.mp4')
subprocess.call(command, shell=platform.system() != 'Windows')
print('done')