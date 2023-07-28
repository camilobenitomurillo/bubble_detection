import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

from skimage import img_as_ubyte
from skimage.feature import canny
from matplotlib_scalebar.scalebar import ScaleBar
from ransac import image_processing, ransac

try:
  os.mkdir('./figures')
except:
  pass

dx = 1./723.  #723 px per mm
scale = ScaleBar(dx, 'mm')

index = [2] # comment this line if processing images in batches

# uncomment this line if processing images in batches
# index = np.arange(start = 2, stop = 81)

for i, ind in enumerate(index):
  print(f'Image {i + 1} of {len(index)}')
  
  path = f'./images/m_X{ind}.tif'
  img = cv.imread(path, cv.IMREAD_GRAYSCALE)
  out = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
  
  contours = image_processing(img)
  
  for ic, contour in enumerate(contours):
    print(f'contour {ic + 1} in {len(contours)}')
    
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
      
      ellipse_parameters = [0, 360, (255,0,0), 3]
      
      cv.ellipse(out, center, axes, angle, *ellipse_parameters)
    
  plt.imshow(out)
  
  plt.gca().add_artist(scale) # comment this line if processing images in batches
  
  outpath = f'./figures/m_X{ind}.png'
  plt.savefig(outpath)
  plt.close()