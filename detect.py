import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import re

def img2char(img_ori):
  gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
  height, width,channel = img_ori.shape
  img_thresh = cv2.adaptiveThreshold(
      gray, 
      maxValue=255.0, 
      adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
      thresholdType=cv2.THRESH_BINARY, 
      blockSize=19, 
      C=9
  )


  contours, _ = cv2.findContours(
      img_thresh, 
      mode=cv2.RETR_LIST, 
      method=cv2.CHAIN_APPROX_SIMPLE
  )

  temp_result = np.zeros((height, width, channel), dtype=np.uint8)

  contours_dict = []

  for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
      
      # insert to dict
      contours_dict.append({
          'contour': contour,
          'x': x,
          'y': y,
          'w': w,
          'h': h,
          'cx': x + (w / 2),
          'cy': y + (h / 2)
      })

  MIN_AREA = 30
  MAX_AREA =1000
  possible_contours = []
  cnt = 0
  for d in contours_dict:
      area = d['w'] * d['h']
      if MAX_AREA>area>MIN_AREA:
          d['idx'] = cnt
          cnt += 1
          possible_contours.append(d)
          
  # visualize possible contours
  temp_result = np.zeros((height, width, channel), dtype=np.uint8)

  for d in possible_contours:
  #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
      cv2.rectangle(temp_result, pt1=(d['x']-5, d['y']-5), pt2=(d['x']+d['w']+5, d['y']+d['h']+5), color=(255, 255, 255), thickness=-1)

  temp_result_inv = cv2.bitwise_not(temp_result)

  text_only = cv2.bitwise_and(img_ori, img_ori, mask=temp_result[:,:,0])
  text_only = temp_result_inv+text_only

  gray = cv2.cvtColor(text_only, cv2.COLOR_RGB2GRAY)
  gray = cv2.GaussianBlur(gray, ksize=(1, 1), sigmaX=0)
  img_thresh = cv2.adaptiveThreshold(
      gray, 
      maxValue=255.0, 
      adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
      thresholdType=cv2.THRESH_BINARY, 
      blockSize=19, 
      C=11
  )

  chars = pytesseract.image_to_string(img_thresh, lang='kor', config='--psm 4 --oem 3')
  chars = re.sub('[^가-힣|\n| |0-9]+','',chars)
  x =[]
    for i in chars.split('\n'):
      if len(i)>2:
        x.append(i)
  return x