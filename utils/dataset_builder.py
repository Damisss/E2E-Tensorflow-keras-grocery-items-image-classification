import os
import pandas as pd
import config
import shutil
import random
import argparse
import pickle

ap = argparse.ArgumentParser()
#ap.add_argument('-b', '--base_path', required=True, help='Path to dataset')
args = vars(ap.parse_args())


# class DataLoader ():
#   def __init__ (self, basePath):
#     self.basePath = basePath
#     pass

#   def buildDataset (self):
#     try:
#       # args['base_path']
#       #basePath = '../dataset/GroceryStoreDataset/dataset'
#       folderNames = ['train.csv', 'test.csv', 'val.csv']
      
#       for folderName in folderNames:
#         df = pd.read_csv(os.path.join(self.basePath, folderName), names=['imagePath', 'classId', 'Coarse Class ID'])
#         for imagePath in df.iterrows:
#           print(imagePath)
#         #['imagePath'].items():
#           # className = imagePath.split(os.path.sep)[2]

#           # if className in config.CLASSES.keys():
#           #   dTypePah = os.path.join('../',config.BASE_PATH, folderName.replace('.txt', ''))
#           #   path = os.path.sep.join(['../', config.BASE_PATH, dTypePah, str(config.CLASSES[className])])

#           #   if not os.path.isdir(dTypePah):
#           #     os.makedirs(dTypePah)
              
#           #     if not os.path.isdir(path):
#           #       os.makedirs(path)
            
#           #   else:
#           #     if not os.path.isdir(path):
#           #       os.makedirs(path)

#           #   image = os.path.join(basePath, imagePath)
#           #   shutil.copy2(image, path)

#     except Exception as e:
#       raise e

# # As the amount or train data is almost the same as the amount of test.
# # Hence, we will transfer 70% of each class in test dataset to corresponding class in trainset
# # def transferData ():
# #   try:
# #     prefix = '../'
# #     testBasePath = os.path.join(prefix, config.BASE_PATH, 'test')
# #     testClasses = [name for name in os.listdir(testBasePath)]
# #     for className in testClasses:
# #       testClassPath = os.path.join(testBasePath, className)
# #       trainClassPath = os.path.sep.join([prefix, config.BASE_PATH, 'train', className])
# #       imagesList = [os.path.join(testClassPath, image) for image in os.listdir(testClassPath)]
# #       split = int(len(imagesList) * config.TRAIN_SPLIT)
# #       random.shuffle(imagesList)

# #       for i in range(split):
# #         image = imagesList[i]
# #         os.rename(image, image.replace('_', ''))
# #         shutil.move(image.replace('_', ''), trainClassPath)

# #   except Exception as e:
# #     raise e

# #transferData()


# def main ():
#   try:
#     loader = DataLoader('../dataset/GroceryStoreDataset/dataset')
#     loader.buildDataset()

#   except Exception as e:
#     raise e

# if __name__ == '__main__':
#   main()

X = pickle.loads(open('../dataset/X_val.pickle', 'rb').read())
y = pickle.loads(open('../dataset/y_val.pickle', 'rb').read())
print(X.shape)
print(y.shape)