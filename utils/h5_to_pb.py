# Usage
# python h5_to_pb.py --model ../model/grocery.h5 --frozen_graph ../model/grocery.pb

import tensorflow.keras.backend as k
from tensorflow.keras.models import load_model
from tensorflow.graph_util import convert_variables_to_constants, remove_training_nodes
from tensorflow.python.framework import graph_io
from tensorflow.python.util import deprecation
import os
import argparse

# remove waring message
deprecation.__PRINT_DEPRECATION_WARNINGS = False
os.environ['TF__CPP_MIN_LOG_LEVEL'] = '3'

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='Path to keras model file.')
ap.add_argument('-o', '--frozen_graph', required=True, help='Path to frozen_graph.')
args = vars(ap.parse_args())

class Frozen ():
  '''
  The aim of this class is to freeze our ft_keras model  into frozen pb graph.
  '''
  @staticmethod
  def build (graph, session, outputNames):
    try:
      # start with default graph
      with graph.as_default():
        # remove nodes which are not needed for inference
        remove_nodes = remove_training_nodes(graph.as_graph_def())
        # freeze graph by converting all variables into constants.
        frozenGraph = convert_variables_to_constants(session, remove_nodes, outputNames)
        return frozenGraph
    except Exception as e:
      raise e

def main ():
  try:
    # set training phase to 0 (this tell to tf_keras that we going to use model for inferencing)
    k.set_learning_phase(0)
    # load model
    model = load_model(args['model'])
    # get the session
    sess = k.get_session()
    #get the list of model output name
    outputNames = [p.op.name for p in model.outputs]
    # freeze our tensorflow model
    frozenGraph = Frozen.build(sess.graph, sess, outputNames)
    # serialized the freeze graph
    graph_io.write_graph(frozenGraph, '', args['frozen_graph'], as_text=False)

  except Exception as e:
    raise e

if __name__ == '__main__':
  main()