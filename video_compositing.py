import os
import cv2
import pdb

filelist = []

path = './docs/results'
filelist = os.listdir(path)
filelist.sort()
filelist.sort(key = lambda x: int(x[11:-4]))
# pdb.set_trace()

fps = 30 # 每秒显示的帧数
size = (1920, 1080)
file_path = "./docs/results/birds_test_30fps.avi"

fourcc = cv2.VideoWriter_fourcc('I','4','2','0')

video = cv2.VideoWriter( file_path, fourcc, fps, size )

for item in filelist:
    if item.endswith('.jpg'):
        item = path + '/' + item
        img = cv2.imread(item)
        video.write(img)

video.release() #释放
print("finish")