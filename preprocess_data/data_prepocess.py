# usage
# python data_preprocess.py
#--dataset dataset
#--width  this is optional
#--heigh  this is optional

import cv2
import os
import pickle
import numpy as np
import progressbar
import argparse
from aspect_ratio import ProperAspectRatio

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to dataset')
ap.add_argument('-w', '--width', default=224, type=int, help='Model input width')
ap.add_argument('-he', '--height', default=224, type=int, help='Model input width')
args = vars(ap.parse_args())

class DataPreparation ():
  def __init__(self, dataset, height, width, inter=cv2.INTER_AREA):
    self.dataset = dataset
    self.height = height # target model input height dimension
    self.width = width  # target model input width dimension
    self.inter = inter
  
  def run (self):
    try:
      # prepare a list of different data types folder name (train, val, test)
      dtypePaths = [dtype for dtype in os.listdir(self.dataset)]
      #loop over the list of folder names and then construct the full path of the folder
      # after that pass the full path to imagePathsList function, 
      # which returns all the path of images located different sub folder.
      #then pass list of images to preprocess each image and then serialize it and its correspoding label.
      for dtype in dtypePaths:
        if not os.path.isdir(os.path.join(self.dataset, dtype)):
            continue
        path = os.path.join(self.dataset, dtype)
        imagePaths = self.imagePathsList(path)
        self.preprocess(imagePaths, dtype)
    
    except Exception as e:
      raise e
  
  def imagePathsList (self, path):
    try:
      # initialize the imge paths list
      imagePaths=list()
      # prepare a list that contains full path of every sub folder (0, 1, 2 etc...)
      folderPaths = [os.path.join(path, folder) for folder in os.listdir(path)]
      # loop over folder path. for each sub folder grab the image path and then append image paths list.
      # finally return image paths list for a specific dtype (e.g. train).
      for folderPath in folderPaths:
        if not os.path.isdir(folderPath):
          continue
      
        for image in os.listdir(folderPath):
          imagePath = os.path.join(folderPath,image)
          imagePaths.append(imagePath)
      return imagePaths
    except Exception as e:
      raise e


  def preprocess(self, imagePaths, dtype):
    try:
      widgets = [f'Preprocessing {dtype} images:', progressbar.Percentage(), '', progressbar.Bar(), '', progressbar.ETA()]
      pBar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

      # instantiate the class ProperAspectRatio
      aspectRatio = ProperAspectRatio(self.height, self.width, self.inter)
      # initialize label and data lists.
      labels = list()
      data = list()
      # loop over image paths list 
      for i, imagePath in enumerate(imagePaths):
        # get image label from its path
        label = imagePath.split(os.path.sep)[-2]
        # read image prepocess it and then push it to data list.
        img = cv2.imread(imagePath)
        img = aspectRatio.build(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        data.append(img)
        labels.append(label)

        pBar.update(i)
      # stack data lis
      data = np.vstack(data)
      labels = np.array(labels)
      pBar.finish()

      #serialize data list
      with open(f'../dataset/X_{dtype}.pickle', 'wb') as f:
         f.write(pickle.dumps(data))
    
       # serialize label list
      with open(f'../dataset/y_{dtype}.pickle', 'wb') as f:
         f.write(pickle.dumps(labels))

      print(f'{dtype} dataset preprocessing is completed')
    except Exception as e:
      raise e


def main ():
  try:
    preparedata = DataPreparation(args['dataset'], args['height'], args['height'])
    preparedata.run()
  except Exception as e:
    raise e

if __name__ == '__main__':
  main()
