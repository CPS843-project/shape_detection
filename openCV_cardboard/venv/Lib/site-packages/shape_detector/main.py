import subprocess
import os
import time



def init_detector():

  bashCommand = "git clone https://github.com/ultralytics/yolov5 "
  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
  time.sleep(10)

  os.chdir('yolov5')
  time.sleep(5)

  bashCommand = "pip install ChexpertClassifier==0.0.1"
  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
  time.sleep(15)

  print("yolov5 has been initialized")





def detect_ellipse(weights, img, source):

  bashCommand = "python /content/yolov5/detect.py --save-txt --weights {} --img {} --conf 0.25 --source  {}".format(weights, img, source)
  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
  process.communicate()
  time.sleep(5)
  
  print("Check out /yolov5/runs directory for the results")

