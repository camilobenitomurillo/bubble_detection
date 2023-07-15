import cv2 as cv
import numpy as np
import pandas as pd
import os

from ransac import ransac, image_processing, toDF

d_range = np.arange(start = 1, stop = 20, step = 3)
sigmaColor_range = np.arange(start = 10, stop = 75, step = 5)

try:
  os.mkdir('./auto')
except:
  pass

for i in range(2,81):
  path = f'./experimental_data/s1/m_X{i}.tif'
  img = cv.imread(path, cv.IMREAD_GRAYSCALE)
  
  for d in d_range:
    for sigmaColor in sigmaColor_range:
      xC_list = []
      yC_list = []
      a_list = []
      b_list = []
      angle_list = []
      
      contours = image_processing(img, d, sigmaColor)
      
      for contour in contours:
        ellipse = ransac(contour, img)
        
        if ellipse is not None:
          center, axes, angle = ellipse
          center = tuple([int(e) for e in center])
          axes = tuple([int(e/2) for e in axes])
          
          xC, yC = center
          a, b = axes
          if a < b:
            a, b = b, a
          
          a_list.append(a)
          b_list.append(b)  
          xC_list.append(xC)
          yC_list.append(yC)
          angle_list.append(angle)
          
          df = toDF(a_list, b_list, xC_list, yC_list, angle_list)
          
          outpath = f'./auto/m_X{i}_d_{d}_sigmaColor_{sigmaColor}.csv'
          df.to_csv(outpath)