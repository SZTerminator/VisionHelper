import cv2
vidcap = cv2.VideoCapture(r'D:\Home\SZTerminator\Desctop\VisionHelper\video_cutter\to_crop_list\a.mp4')
success, image = vidcap.read()
count = 1
while success:
  cv2.imwrite("video_data/image_%d.jpg" % count, image)    
  success, image = vidcap.read()
  print('Saved image ', count)
  count += 1