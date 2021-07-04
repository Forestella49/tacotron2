import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import re

def img2char(img):
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  _,img_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
  chars = pytesseract.image_to_string(img, lang='kor', config='--psm 11 --oem 3')
  chars = re.sub('[^가-힣|\n| |0-9]+','',chars)
  x =[]
    for i in chars.split('\n'):
      if len(i)>2:
        x.append(i)
  return x