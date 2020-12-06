#usage python fine_tune.py \
#--model model/grocery.h5

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from nn.fcHeadNet import FCHeadNet
# from tensorflow.keras import backend as K
# from tensorflow.python.framework import graph_io
# import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import numpy as np
import pickle
from utils import config, plot_training
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='Path to model')
args = vars(ap.parse_args())

class Training ():
  @staticmethod
  def run(): 
    try:
      # load train val and test from disk
      X_train = pickle.loads(open(config.X_TRAIN, 'rb').read())
      y_train = pickle.loads(open(config.Y_TRAIN, 'rb').read())
      X_val = pickle.loads(open(config.X_VAL, 'rb').read())
      y_val = pickle.loads(open(config.Y_VAL, 'rb').read())
      X_test = pickle.loads(open(config.X_TEST, 'rb').read())
      y_test = pickle.loads(open(config.Y_TEST, 'rb').read())

      # transform label into onehot
      le = LabelBinarizer()
      le.fit(y_train)
      y_train = le.transform(y_train)
      y_val = le.transform(y_val)
      y_test = le.transform(y_test)
      # compute weight which will be associate to every class (this help hadling unbalance dataset)
      totalClasses = y_train.sum(axis=1)
      classWeights = totalClasses.max()/totalClasses
      # apply data augmentations (this prevent model overfit)
      trainAug = ImageDataGenerator(
        zoom_range=.15, 
        width_shift_range=.2, 
        height_shift_range=.2, 
        shear_range=.15, 
        horizontal_flip=True, 
        rotation_range=30, 
        fill_mode='nearest'
      )
      # actually the don't need image augmentation for validation or test dataset
      # but we usiing here because we are going to apply imagenet mean substrations.
      valAug = ImageDataGenerator()

      # mean substraction
      trainAug.mean = config.MEAN
      valAug.mean = config.MEAN

      # generate training data generator
      trainGen = trainAug.flow(
        X_train,
        y_train,
        shuffle=True,
        batch_size=config.BATCH_SIZE
      )

      valGen = valAug.flow(
        X_val,
        y_val,
        shuffle=False,
        batch_size=config.BATCH_SIZE
      )

      testGen = valAug.flow(
        X_test,
        y_test,
        shuffle=False,
        batch_size=config.BATCH_SIZE
      )

      # define the model
      baseModel = VGG16(
        weights='imagenet', 
        include_top=False, 
        input_tensor=Input(shape=(224, 224, 3))
        )

      headModel = FCHeadNet.build(baseModel, 11)
      model = Model(inputs=baseModel.input, outputs=headModel)
    
      # compile the model
      opt = Adam(lr=1e-5)
      # compile model
      model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
      #freeze model layer
      for layer in baseModel.layers:
        layer.trainable = False
      # apply EarlyStopping callback to prevent model overfitting
      early_stop = EarlyStopping(monitor='val_loss',  patience=10)
        
      # train the model
      H = model.fit_generator(
        trainGen, 
        steps_per_epoch= X_train.shape[0]//config.BATCH_SIZE, 
        epochs=config.EPOCHS, 
        validation_data=valGen, 
        validation_steps=X_val.shape[0]//config.BATCH_SIZE,
        callbacks=[early_stop],
        class_weight=classWeights
        )

      testGen.reset()
      predInds = model.predict_generator(
        testGen, 
        steps=(X_test.shape[0]//config.BATCH_SIZE) + 1
      )
      predInds = np.argmax(predInds, axis=1)

      model.save(args['model'])
      
      with open(config.LB,'wb') as f:
        f.write(pickle.dumps(le))
    # sess = K.get_session()
    # frozen = tf.graph_util.convert_variables_to_constants(
    #   sess,
    #   sess.graph_def, [model.output.op.name])
    # graph_io.write_graph(
    #   frozen, config.FROZENGRAPH,
    #   "grocery_model.pb", as_text=False)

      plot_training.plot(H, config.PLOT_PATH)
      
      print(classification_report(
        np.argmax(y_test, axis=1), 
        predInds, 
        target_names=config.CLASSES.keys())
      )
      
      print(classification_report(
        np.argmax(y_test, axis=1), 
        predInds, 
        target_names=le.classes_)
      )

    except Exception as e:
      raise e


def main():
  try:
    Training.run()
  except Exception as e:
    raise e

if __name__ == '__main__':
  main()


