from keras.models import load_model
import tensorflow as tf
import os 
from keras import backend as K
from tensorflow.python.framework import graph_util,graph_io
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

model_path = 'ssd_detect.h5'
outname = 'ssd_detect.pb'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})

out_nodes = []
for i in range(len(model.outputs)):
    out_nodes.append('output' + str(i + 1))
    tf.identity(model.output[i], 'output' + str(i + 1))

sess = K.get_session()
init_graph = sess.graph.as_graph_def()
main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
graph_io.write_graph(main_graph, '.',name = outname,as_text = False)

