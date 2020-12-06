from tensorflow.keras.layers import Dense, Dropout, Flatten

class FCHeadNet ():
  @staticmethod
  def build (baseModel, numClasses, numNeurons=1024, dropProba=.5):
    try:
      
      X = baseModel.output
      X = Flatten(name='flatten')(X)
      X = Dense(numNeurons//2, activation='relu', kernel_initializer='he_normal')(X)
      X = Dropout(dropProba)(X)
      X = Dense(numNeurons, activation='relu', kernel_initializer='he_normal')(X)
      X = Dropout(dropProba)(X)
      X = Dense(numClasses, activation='softmax')(X)

      return X
    except Exception as e:
      raise e