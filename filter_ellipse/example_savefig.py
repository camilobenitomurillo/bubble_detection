import cv2 as cv
import matplotlib.pyplot as plt
import os

from ransac import image_processing, ransac

for i in range(21,81):
  print(f'Image #{i}')
  
  path = f'./experimental_data/s1/m_X{i}.tif'
  img = cv.imread(path, cv.IMREAD_GRAYSCALE)
  out = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

  contours = image_processing(img)

  for ic, contour in enumerate(contours):
    ellipse = ransac(contour, img)
    
    if ellipse is not None:
      center, axes, angle = ellipse
      center = tuple([int(e) for e in center])
      axes = tuple([int(e/2) for e in axes])
      
      ellipse_params = [0, 360, (255, 0, 0), 2]
      
      cv.ellipse(out, center, axes, angle, *ellipse_params)

  try:
    os.mkdir('out_filtered')
  except:
    pass

  outpath = f'./out_filtered/m_X{i}.png'
  plt.imshow(out)
  plt.savefig(outpath, bbox_inches = 'tight', dpi = 200)
  plt.close()