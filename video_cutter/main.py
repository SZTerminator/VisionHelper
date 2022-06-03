import cv2
import os


cut_folder = "to_crop_list"
output_folder = "cutout_images"
save_format = "jpg"
valid_format = "mp4"
message_for_logs = "saved: "
skip_massage = "Skip: "
move_files = "completed"

cl = True

class Logger():
    def __init__(self, print) -> None:
        self.print = print
    def log(self, str):
        print(str)

logger = Logger(cl)

videos_to_cut = os.listdir(cut_folder)

for video_name in videos_to_cut:
    logger.log(f"Start {video_name}")
    video_file_name = str(video_name)

    if(not video_file_name.endswith(f".{valid_format}")):
        continue
    video = cv2.VideoCapture(f"{cut_folder}/{video_file_name}")
    video_name = video_file_name.split('.')[0]
    
    counter = 0
    while video.isOpened():
        code, image = video.read()
        
        if not code:
            break

        del code

        saved_image_name = f"{output_folder}/{video_name}{counter}.{save_format}"

        if not os.path.isfile(saved_image_name):
            cv2.imwrite(saved_image_name,image)
            skipped = False
        else:
            skipped = True

        logger.log(f"{skip_massage if skipped else message_for_logs}{video_name}{counter} ")
        
        del skipped

        counter += 1
    video.release()
    os.rename(f"{cut_folder}/{video_file_name}",f"{move_files}/{video_file_name}")
    logger.log(f"End {video_name}")
