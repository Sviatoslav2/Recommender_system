import utils
import config
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

lst_of_users = utils.lst_data

def pair_sim(X,Y):
    return 1 - cdist(X,Y,'cosine')

def metrica(X,Y):
    X = np.reshape(X, lst_of_users[0].history.shape)
    Y = np.reshape(Y, lst_of_users[0].history.shape)
    #return np.sum(pair_sim(X,Y))/ (lst_user[0].history.shape[0]**2)
    return pair_sim(X,Y).min()


def get_one_user_y(user1):

    y = [i[0] for i in pair_sim(user1.history, [user1.value])]
    return np.array(y)


def get_xy(lst_users):
    #print(pair_sim(lst_users[0].history,[lst_users[0].res]))


    for index, user in enumerate(lst_users):
        if index == 0:
            X = user.history
            Y = get_one_user_y(user)
        X = np.concatenate((X, user.history), axis=0)
        Y = np.concatenate((Y, get_one_user_y(user)), axis=0)

    return X, Y


X, Y = get_xy(lst_of_users)
reg = LinearRegression().fit(X, Y)

weigths = reg.coef_
print(weigths)