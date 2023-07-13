import numpy as np
import random
import cv2 as cv
import pandas as pd

from math import sqrt,cos,sin,pi,log
from statistics import mean

def random_points(contour, npoints = 6):
  '''
  Selects a given number of points from a contour.
  -----
  Parameters:
  contour : np.ndarray-like
            Array containing the coordinates of the points in the contour.
            Element of the array returned by the function cv.findContours.
  npoints : int
            Number of random points to be selected.
  ------
  Returns:
  points : np.ndarray-like
           Array containing the coordinates of the randomly selected points.
           Same array style as contour.
  '''
  points = np.empty(shape = (npoints,1,2), dtype = int)
  contour_len = len(contour)
  
  indexes = []
  while len(indexes) < 6:
    index = random.randint(0,contour_len-1)
    
    if index not in indexes:
      points[len(indexes)] = contour[index]
      indexes.append(index)
      
  return points

def change_frame_of_reference(point, angle, rad=False):
  '''
  Changes the frame of reference to one where the ellipse is not rotated.
  ------
  Parameters:
  point : tuple of int
          x and y coordinates of the point in the orginial frame of reference.
  angle : float
          Rotation angle of the ellipse.
  rad : bool, optional
        True if the angle is given in radians. Default is False.
  ------
  Returns:
  xp : float
       x coordinate in the rotated frame of reference
  yp : float
       y coordinate in the new frame of refernce
  '''
  x, y = point
  
  if rad is False:
    angle = pi*angle/180
    
  xp = x*cos(angle) + y*sin(angle)
  yp = y*cos(angle) - x*sin(angle)
  
  return xp,yp

def distance_to_ellipse(axes, center, point, angle):
  '''
  Approximation of the minimal distance between a point and the ellipse. Works 
  best for lower excentricities.
  -----
  Parameters:
  axes : tuple of float
         Contains the semi-major axis and semi-minor axis
  center : tuple of float
           Contains the x and y coordinates of the center of the ellipse.
  point : tuple of float
          Contains the x and y coordinates of the point.
  angle : float
          Rotation angle of the ellipse.
  -----
  Returns:
  distance : float
             Minimal distance between the point and the ellipse
  '''
  a, b = axes
  xC, yC = change_frame_of_reference(center, angle)
  xM, yM = change_frame_of_reference(point, angle)
  
  if xM == xC:
    r = abs(abs(yM - yC) - a) #Vertical distance
  else:
    m = (yM-yC)/(xM-xC)
    p = yC - m*xC
    
    #x is one of the two solutions. Given that the solutions are symmetrical and
    #we will only use the norm, there's no need to calculate both solutions.
    x = xC + (a*b)/sqrt(b**2 + (a*m)**2)
    y = m*x + p
  
    r = sqrt((x-xC)**2 + (y-yC)**2)
    
  CM = sqrt((xM-xC)**2 + (yM-yC)**2)
    
  distance = abs(CM - r)
    
  return distance

def distance_2_points(point1, point2):
  '''
  Calculates the distance between two points.
  ----
  Parameters:
  point1 : tuple of int
           x and y coordinates of the first point.
  point2 : tuple of int
           x and y coordinates of the second point.
  ----
  Returns:
  d : int
      Distance between the two points
  '''
  x1,y1 = point1
  x2,y2 = point2
  
  d = sqrt((x1-x2)**2 + (y1-y2)**2)
  
  return d

def general_ellipse(center, axes, angle, point, rad = False):
  '''
  The equation describing a center unrotated ellipse is : x²/a² + y²/b² = 1.
  If the left term is exactly one, the point (x,y) lies exactly on the ellipse.
  If it's greater than 1, the point lies outside of the ellipse and if it's 
  less than 1, the point lies inside of the ellipse.
  The function calculates that value.
  --------
  Parameters:
  axes : tuple of float
         Contains the semi-major axis and semi-minor axis
  center : tuple of float
           Contains the x and y coordinates of the center of the ellipse.
  point : tuple of float
          Contains the x and y coordinates of the point.
  angle : float
          Rotation angle of the ellipse.
  point : tuple of int
          Coordinates of the point to evaluate.
  rad : bool
        If false, angle is converted from degrees to radians. Default is False.
  '''
  xC,yC = center
  a,b = axes
  y,x = point # x and y axes are inverted
  x = x-xC
  y = -(y-yC) # y axis is flipped
  
  if not rad:
    angle = (np.pi*angle)/180
    
  xp = x*np.cos(angle) - y*np.sin(angle)
  yp = y*np.cos(angle) + x*sin(angle)
  
  return (xp/a)**2 + (yp/b)**2

def innerLuminosity(img, center, axes, angle):
  '''
  Calculates the average luminosity of an elliptic area of an image.
  ------
  Parameters:
  img : Mat, np.ndarray-like. Variable returned by opencv functions.
  center : tuple of float
           Coordinates of the center of the ellipse.
  axes : tuple of float
         Semi-major and semi-minor axis
  angle : float
          Rotation angle of the ellipse.
  ------
  Returns:
  luminosity : float
               Average luminosity of the elliptic area.
  '''
  yC,xC = center
  a,b = axes
  if a < b:
    a,b = b,a
  
  xmin = xC - a if xC - a > 0 else 0
  xmax = xC + a if xC + a < 1024 else 1023
  ymin = yC - a if yC - a > 0 else 0
  ymax = yC + a if yC + a < 1024 else 1023
  
  luminosity = 0
  npoints = 0
  
  for x in range(xmin, xmax + 1):
    for y in range(ymin, ymax + 1):
      if general_ellipse(center, axes, angle, (x,y)) < 1:
        luminosity += img[x][y]
        npoints += 1
  
  if npoints == 0:
    print('No points found in the ellipse.')
    return -1
  else:
    return luminosity/npoints

def ransac(contour, niter = 600, threshold = 5, limit = 0.2, min_id = 0.5):
  '''
  RANdom SAmple Consensus algorithm adapted to fit an ellipse.
  ------
  Parameters:
  contour : np.ndarray-like
            Array containing the coordinates of the points in the contour.
            Element of the array returned by the function cv.findContours.
  threshold : float
              Distance to the ellipse at which a point starts to be considered an
              outlier.
  niter : int
          Number of times the RANSAC algorithm will run before giving its final
          result. 
  limit : float
          Maximum value for the normalized axis difference. Between 0 and 1.
  min_id : float
           Minimum inlier density to consider the model. Between 0 and 1.
  ------
  Returns:
  best_model : tuple
               Return values from the best model calculated with cv.fitEllipseAMS.
  '''
  best_inlier_count = 0
  best_model = None
  
  for i in range(niter):
    inlier_count = 0
    points = random_points(contour)
    
    ellipse = cv.fitEllipseAMS(points)
    center, axes, angle = ellipse
    axes = tuple([a/2 for a in axes]) #cv.fitEllipseAMS returns the rotated
                                      #rectangle in which the ellipse is 
                                      #inscribed.
    a,b = axes 
    if a < b:
      a,b = b,a                                 
    
    try:
      e = abs(a-b)/(a+b)
    except:
      e = 1
    #Only considers ellipses with sufficiently low eccentricity
    if e < limit:
      for point in contour:
        point = point[0]    #point is a list with a single tuple inside it.
        
        if distance_to_ellipse(axes, center, point, angle) < threshold:
          inlier_count += 1
      
      inlier_density = inlier_count/len(contour)
      
      #Only considers ellipses with a sufficienty 'tight' fit
      if inlier_count > best_inlier_count and inlier_density > min_id:
        best_model = ellipse
        best_inlier_count = inlier_count
  
  return best_model