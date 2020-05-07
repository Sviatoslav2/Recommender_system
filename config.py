import pickle
from os import listdir
from os.path import isfile, join
import os
height, weight = 224, 224
channels = 3
number_of_users = 100
number_of_history = 7

def get_list(mypath):
    return [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

def get_list_dir(mypath):
    return [os.path.join(mypath, o) for o in os.listdir(mypath) if os.path.isdir(os.path.join(mypath,o))]

data_path = "C:\\Users\\Fedoriv\\Desktop\\main_tasks\\Work\\Pro4\\Data"

lst_of_paths_to_data = get_list(data_path)

with open('data.pkl', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

print(data.shape)