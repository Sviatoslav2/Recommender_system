from sklearn.neighbors import NearestNeighbors, KDTree
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from metric import *
#import weigth




lst_user = create_sim_user(lst_of_image[50],lst_of_image, 101)
USER = lst_user[0]

def metrica(X,Y,shape=USER.history.shape):
    X = np.reshape(X, shape)
    Y = np.reshape(Y, shape)
    return np.sum(pair_sim(X,Y))/ (lst_user[0].history.shape[0]**2)


lst_user_values1d = [ np.reshape(i.history, (np.product(i.history.shape))) for i in lst_user]
print("lst_user_values1d[0].shape == ",lst_user_values1d[0].shape)
neigh = NearestNeighbors(1000, 0.2,metric=metrica)
neigh.fit(np.array(lst_user_values1d))
indexes_train = neigh.kneighbors([np.reshape(USER.history, (np.product(USER.history.shape)))], 10, return_distance=False)
print("indexes == ", indexes_train)





for i in indexes_train[0]:
    print(lst_user[i].res_img.path)

print("max == ",get_max_pair(USER, [lst_user[i] for i in indexes_train[0]]))
print("mean == ",get_mean_sim(USER, [lst_user[i] for i in indexes_train[0]]))
print("min == ", get_min_sim(USER, [lst_user[i] for i in indexes_train[0]]))
lst = get_all_sim(USER, [lst_user[i] for i in indexes_train[0]])
lst.sort()
print("All == ",lst)


lst_res_path = [lst_user[i].res_img for i in indexes_train[0]]
lst_res_path.sort(key= lambda x: np.dot(USER.value, x.value) / (np.linalg.norm(USER.value) * np.linalg.norm(x.value)))
lst_res_path = [i.path for i in lst_res_path]


