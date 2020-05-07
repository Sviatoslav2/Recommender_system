from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

vgg_model = vgg16.VGG16(weights='imagenet')
feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
feat_extractor.summary()

