#!/usr/bin/env python
# coding: utf-8

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
import os


def getfiles(path):
    filelist = []
    for root, dirs, files in os.walk(path):
        filelist.extend([ os.path.join(root, file) for file in files if file.endswith(('.jpg', '.png', '.JPG', '.PNG'))])
    return filelist


# Set the image size.
img_height = 300
img_width = 300

# ## 1. Load a trained SSD
# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = 'ssd_detect.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})

# ## 2. Load some images
outdir = './output1'
if not os.path.exists(outdir):
    os.makedirs(outdir)
imglists = getfiles('/data/vocGT/val_images/')
for img_path in imglists:

    orig_images = imread(img_path)
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img) 
    input_images = [img]
    input_images = np.array(input_images)

    # ## 3. Make predictions
    y_pred = model.predict(input_images)

    confidence_threshold = 0.5

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
    '''
    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[0])
    '''
    # ## 4. Visualize the predictions

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    classes = ['background', 'papers_z', 'papers_f', 'card_z', 'card_f', 'papers_hk_z', 'papers_hk_f']

    plt.figure()
    plt.imshow(orig_images)

    current_axis = plt.gca()

    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_images.shape[1] / img_width
        ymin = box[3] * orig_images.shape[0] / img_height
        xmax = box[4] * orig_images.shape[1] / img_width
        ymax = box[5] * orig_images.shape[0] / img_height
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

    plt.savefig(os.path.join(outdir, os.path.basename(img_path)))




