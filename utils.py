import config
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

data = config.data
lst_pf_paths = config.lst_of_paths_to_data
print(data.shape)
class Image:
    def __init__(self, path, value):
        self.path = path
        self.value = value


lst_of_image = [Image(lst_pf_paths[i], value) for i, value in enumerate(data)]


def get_sim_images(imag, lst_img, num_sim):
    lst_img.sort(key= lambda x: np.dot(imag.value, x.value) / (np.linalg.norm(imag.value) * np.linalg.norm(x.value)))
    return lst_img[:num_sim]


def get_sim_values(imag, lst_img, num_sim):
    return list(map(lambda x:x.value, get_sim_images(imag, lst_img, num_sim)))

def get_sim_paths(imag, lst_img, num_sim):
    return list(map(lambda x: x.path, get_sim_images(imag, lst_img, num_sim)))


def get_sim_one_img(imag, lst_img, num_sim):
    #print(num_sim)
    return get_sim_images(imag, lst_img, num_sim)[num_sim-1]
#print(get_sim_values(Image(lst_pf_paths[0],data[0]), lst_of_image, 7))

######################################################################
######################################################################
######################################################################

class User:
    def __init__(self, img_history, res_img):
        self.img_history = img_history
        self.res_img = res_img


        self.path_history = [x.path for x in self.img_history]#list(map(lambda x:x.path, self.img_history))
        self.history = np.array([x.value for x in self.img_history])#np.array(list(map(lambda x:x.value, self.img_history)))
        self.value = self.res_img.value
len_history=7
def create_sim_user(img, lst_img, number_of_user, len_history=len_history):
    lst_of_user = []
    lst = get_sim_images(img, lst_img, len_history)

    lst_hist = []
    for i in lst:
        lst_hist.append(get_sim_images(i, lst_img, number_of_user)[:-1])
    lst_hist = np.array(lst_hist).T

    print(lst_hist.shape)
    for i, row in enumerate(lst_hist):
        lst_of_user.append(User(list(row), get_sim_images(lst[-1],lst_img, number_of_user)[i]))

    return lst_of_user


 #+ [User([random.choice(lst_of_image) for i in range(len_history-1)], random.choice(lst_of_image)) for j in range(100)]










































































