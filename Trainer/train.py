#!python

import argparse
import json
import os
import time
import json
from prettytable import PrettyTable
import requests
from torch import miopen_convolution_transpose
from PIL import Image

script_path = os.path.realpath(__file__)
dir_path = os.path.dirname(script_path)
del script_path

parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
parser.add_argument("--config", default=None, type=int, help="port number")
opt = parser.parse_args()
config_filename = opt.config if opt.config is not None else "config.json"
del parser, opt

with open(f"{dir_path}/{config_filename}","r",encoding="utf-8") as config_file:
    config = json.load(config_file)
del config_filename

path = "D:\Home\SZTerminator\Desctop\VisionHelper"


images_path = f"{path}/video_cutter/cutout_images"
classes_path = f"{path}\dataset\Labels\classes.txt"
images_new = f"{path}/video_cutter/cutout_images"
confidences_path = f"{path}/dataset/confidence"
labels_path = f"{path}/dataset/labels"
cuts = os.listdir(images_path)

def read_classes(filename):
    with open(filename,"r") as file:
        contents = file.read()
    cl = contents.split("\n")
    ma = dict()
    counter = 0
    for i in cl:
        try:
            ma.update({i:counter})
        except: pass
        counter +=1
    return ma, counter


class Log:
    def __init__(self) -> None:
        self.print = print
        self.timestamps = []
        self.contents = ""
    def clear(self):
        self.contents = ""
        self.timestamps = []
        self.table = ""
    def add(self, line):
        self.contents += line + "\n"
    def show(self):
        self.print(self.contents)
        self.print(self.table)
        self.print(f"finished in {self.get_time(0,1)} ms")
    def pop(self):
        self.show()
    def time(self):
        self.timestamps.append(time.time()*100)
    def get_time(self, first, last):
        return round(self.timestamps[last] - self.timestamps[first])
    def add_table(self,response,image_width,image_heigh):
        th = ["class","width","height","confidence"]
        table = PrettyTable(th)
        for item in response:
            data =[]

            class_name = item["name"]
            confidence = f'{round(item["confidence"]*100)}%'
            width_on_image = f'{round((item["xmax"]-item["xmin"])/image_width*100,2)}%'
            height_on_image = f'{round((item["ymax"]-item["ymin"])/image_heigh*100,2)}%'
            size_on_image = f'{width_on_image} {height_on_image}'

            data.append(class_name)
            data.append(width_on_image)
            data.append(height_on_image)
            data.append(confidence)

            table.add_row(data)
        self.table = table
    def clean(self):
        os.system("cls")
log = Log()
detected_during_this_session = 1

cl, counter = read_classes(classes_path)
size = len(cuts)
for image_file_name in cuts:
    log.clear()
    log.time()
    image_name = image_file_name.split(".")[0]
    image_file_path = f"{images_path}/{image_file_name}"
    image_file_new_location = f"{images_new}/{image_file_name}"
    object_detection_confidences = f"{confidences_path}/{image_name}.json"
    object_yolo5_label = f"{labels_path}/{image_name}.txt"

    with open(image_file_path,"rb") as image_file:
        contents = image_file.read()
    response = requests.post(config["request_url"],contents)
    if response.status_code == 200:
        response_object = response.json()
        json_response = json.dumps(response_object, indent=4, sort_keys=True)
        
        with open(object_detection_confidences,"w") as file:
            file.write(json_response)
        with open(object_yolo5_label,"w") as file:
            contents = ""
            
            for item in response_object:
                height = (item["ymax"] - item["ymin"] )/ 1920
                width  = (item["xmax"] - item["xmin"] )/ 1080
                x_center = (item["xmax"] + item["xmin"]) / 2 / 1080
                y_center = (item["ymax"] + item["ymin"]) / 2 / 1920

                class_name = item["name"]
                try:
                    cl[class_name]
                except KeyError:
                    cl.update({class_name:counter})
                    with open(classes_path, "a")as classes_file:
                        classes_file.write(f"{class_name}\n")
                    counter += 1
                contents += f'{cl[class_name]} {x_center} {y_center} {width} {height}\n'


            file.write(contents)
        
        # os.rename(image_file_path,image_file_new_location)
        log.time()
        
        log.add(f'this session: {detected_during_this_session} ({round((detected_during_this_session / size)*100,4)}%) | {image_name} : has {len(response_object)}')
        detected_during_this_session += 1
        log.add_table(response_object,1080,1920)
        
        log.clean()
        log.show()
