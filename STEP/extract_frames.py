import cv2
vidcap = cv2.VideoCapture('P02T02C06.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("./P02T02C06/frame%04d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1