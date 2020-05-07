from config import *
from PIL import Image
import numpy as np
from skimage.transform import resize
import model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

lst_of_paths = lst_of_paths_to_data


def read_img(path, height=height, weight=weight,channels=channels):
    img = Image.open(path)
    img = img.resize((224, 224))
    numpy_img = img_to_array(img)
    return np.expand_dims(numpy_img, axis=0).astype('float16')




def get_batch_iteration(lst_of_paths, size_of_batch):
    return model.feat_extractor.predict(np.vstack(list(map(read_img, lst_of_paths[0:size_of_batch]))))


def get_data(lst_of_paths, size_of_batch):

    return get_batch_iteration(lst_of_paths, size_of_batch)

data1 = get_data(lst_of_paths, 8700)

print(data1.shape)
with open('data.pkl','wb') as f:
    pickle.dump(data1, f, protocol=4)

#with open('data.pkl', 'rb') as pickle_file:
#    data = pickle.load(pickle_file)

#print(data.shape)