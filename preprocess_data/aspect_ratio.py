import cv2
import imutils


class ProperAspectRatio ():
  '''
  this class eliminate squeezing a rectangular image into 
  a square frame thereby allowing our model to train 
  on images of the proper aspect ratio.
  '''

  def __init__(self, height, width, inter):
    self.height = height
    self.width = width
    self.inter = inter

  def build (self, image):
    try:
      # grab image original height and width
      H, W = image.shape[:2]
      dH = 0
      dW = 0
      
      # if original height is smaller than original width then resize image along original height
      if H < W :
        image = imutils.resize(image, height=self.height, inter=self.inter)
        dW = int((image.shape[1] - self.width)/2.0)
      else:
        image = imutils.resize(image, width=self.width, inter=self.inter)
        dH = int((image.shape[0] - self.height)/2.0)
      # grab again image H and W
      H, W = image.shape[:2]
      # crop image using the new spatial dimensions
      image = image[dH:H-dH, dW:W-dW]
      

      # finally resize our image spatial dimentsions to the one espected by our model
      return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
      
    except Exception as e:
      raise e
