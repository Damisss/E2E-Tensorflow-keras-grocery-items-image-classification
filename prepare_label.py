from utils import config
import pandas as pd
import os
# ap = argparse.ArgumentParser()
# ap.add_argument('-b', '--base_path', required=True, help='base path to image folders')
# args = vars(ap.parse_args())

def prepare_data(dType='train'):
  try:
    basePath = config.BASE_PATH
    labels = list()
    dTypeFolderPath = os.path.join(basePath, dType)
    classNames = [name for name in os.listdir(dTypeFolderPath)]

    for className in classNames:
      path = os.path.sep.join([basePath,dType, className])
      imagesPath = [images for images in os.listdir(path)]
      for _ in imagesPath:
        labels.append(className)

    df = pd.Series(labels, name='labels')
    return df.astype('int').sort_values()
  except Exception as e:
    raise e

train_df = prepare_data()
train_df.to_csv(f'{config.BASE_PATH}/train.csv', index=False)
val_df = prepare_data('val')
val_df.to_csv(f'{config.BASE_PATH}/val.csv', index=False)
test_df = prepare_data('test')
test_df.to_csv(f'{config.BASE_PATH}/test.csv', index=False)