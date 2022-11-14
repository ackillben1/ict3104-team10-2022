import cv2
import os
filename = "P02T02C06.mp4"
vidcap = cv2.VideoCapture('./Data_Folder/Videos/'+filename)
success, image = vidcap.read()
count = 0
if not os.path.exists('./Data_Folder/Frames'):
    os.mkdir('./Data_Folder/Frames')
    os.mkdir('./Data_Folder/Frames/'+filename[:-4])
while success:
    # save frame as JPEG file
    cv2.imwrite("./Data_Folder/Frames/" + \
                filename[:-4]+"/frame%04d.jpg" % count, image)
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
