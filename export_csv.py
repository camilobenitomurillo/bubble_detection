import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from skimage import img_as_ubyte
from skimage.feature import canny
from matplotlib_scalebar.scalebar import ScaleBar
from ransac import image_processing, ransac, toDF

try:
  os.mkdir('./csv')
except:
  pass

index = np.arange(2,81)

for i, ind in enumerate(index):
  print(f'Image {i + 1} of {len(index)}')
  
  path = f'./images/m_X{ind}.tif'
  img = cv.imread(path, cv.IMREAD_GRAYSCALE)
  
  contours = image_processing(img)
  
  xC_list = []
  yC_list = []
  a_list = []
  b_list = []
  angle_list = []
  
  for ic, contour in enumerate(contours):
    print(f'contour {ic + 1} of {len(contours)}')
    
    ellipse = ransac(contour, img)
    
    if ellipse is not None:
      center, axes, angle = ellipse
      
      '''
      cv.ellipse only takes interger center coordinates and axes length.
      ransac will return the length of the sides of the rotated rectangle
      (same as cv.fitEllipse), so it has to be halved to get the semi-major
      and semi-minor axis.
      '''
      center = tuple([int(e) for e in center])
      axes = tuple([int(e/2) for e in axes])
      
      xC, yC = center
      a, b = axes
      if a < b:
        a, b = b, a
        
      xC_list.append(xC)
      yC_list.append(yC)
      a_list.append(a)
      b_list.append(b)
      angle_list.append(angle)

  df = toDF(a_list, b_list, xC_list, yC_list, angle_list)
  
  outpath = f'./csv/m_X{ind}.csv'
  df.to_csv(outpath)