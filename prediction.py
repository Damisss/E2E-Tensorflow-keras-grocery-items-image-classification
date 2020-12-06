#Usage 
# python prediction.py --inference_mode image or video --input path to input file
import cv2
import argparse
import pickle
import time
print(cv2.__version__)

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--inference_mode', default='video', help='inference mode.')
ap.add_argument('-i', '--input', help='Path to input image.')
ap.add_argument('-le', '--labels', default='output/le.pickle', help='Path fo labels')
args = vars(ap.parse_args())

class Inference ():
  def __init__(self):
    self.net =  cv2.dnn.readNetFromTensorflow('model/grocery.pb')
    self.le = pickle.loads(open(args['labels'], 'rb').read())
    self.labels = self.le.classes_

  def image(self):
    try:
      
      img = cv2.imread(args['input'])
      start = time.time()
      # preprocess image 
      blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (123.68, 116.779, 103.939), swapRB=True, crop=False)
      # pass blob to our network
      self.net.setInput(blob)
      # make prediction
      preds = self.net.forward()
       # sort prediction result 
      preds = [(i, proba) for (i, proba) in enumerate(preds[0])]
      results = sorted(preds, key=lambda x: x[1], reverse=True)[:5]
      
      for i, (ind, proba) in enumerate(results):
        if i == 0:
          end = time.time()
          # prediction probabilty
          proba = '%.2f'%(proba * 100)
          # prediction label and probability
          text = f'{self.labels[ind]}, {proba}%'
          # prediction time
          predTime = '%.2f'%(end-start)
          predTime = f'Prediction time: {str(predTime)} ms'
          font = cv2.FONT_HERSHEY_SIMPLEX
          cv2.putText(img, text, (5, 20), font, .8, (0, 0, 255), 2)
          cv2.putText(img, predTime, (5, 50), font, .85, (0, 0, 255), 2)
    
          cv2.imshow('Grocery', img)
  
      cv2.waitKey(0)
      cv2.destroyAllWindows()

    except Exception as e:
      raise e
 
  def realTime (self):
    try:

      if not args.get('input', False):
        cap = cv2.VideoCapture(0)
      else:
        cap = cv2.VideoCapture(args['input'])
    
      start = time.time()
      while cap.isOpened():
        ret, frame = cap.read()

        # break the loop if there is no frame
        if not ret:
          break
        #preprocess image  
        blob = cv2.dnn.blobFromImage(frame, 1.0, (240, 240), (123.68, 116.779, 103.939), swapRB=True, crop=False)
        # pass blob to our network
        self.net.setInput(blob)
        # make prediction
        results = self.net.forward()
        # sort prediction result 
        results = [(i, proba) for (i, proba) in enumerate(results[0])]
        results = sorted(results, key=lambda x: x[1], reverse=True)[:5]

        for i, (ind, proba) in enumerate(results):
          end = time.time()
          # prediction probabilty
          proba = '%.2f'%(proba)
          # prediction label and probability
          text = f'{self.labels[ind]}, {proba}%'
          # prediction time
          predTime = '%.2f'%(end-start)
          predTime = f'Prediction time: {str(predTime)}ms'
          font = cv2.FONT_HERSHEY_SIMPLEX
          copyFrame = frame.copy()
          copyFrame = cv2.putText(copyFrame, text, (5, 20), font, .8, (0, 0, 255), 2)
          copyFrame = cv2.putText(copyFrame, predTime, (5, 50), font, .85, (0, 0, 255), 2)
        #break the loop if q letter is press in keyboard
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

        # show frame
        cv2.imshow('Grocery', copyFrame)
      # release camera
      cap.release()
      # close all windows
      cv2.destroyAllWindows()
    except Exception as e:
      raise e


def main (mode):
  try:
    prediction =Inference()
    if mode == 'image':
      prediction.image()
    else:
      prediction.realTime()

  except Exception as e:
    raise e

if __name__ == '__main__':
  main(args['inference_mode'])