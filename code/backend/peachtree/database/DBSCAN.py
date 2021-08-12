import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv('dataDB_other.csv')
idvars=['Unnamed: 0.1','Unnamed: 0','id_scholarship','scholarship_name','link','other']
unusedvars=['major','region','date_start','date_end','feature','feature_specified',
            'age_max','grade_max','last_grade_max','recommendation','year']
data_input = data.drop(idvars, axis = 1)
data_input = data_input.drop(unusedvars, axis = 1)

scaler = MinMaxScaler()

data_input_array = scaler.fit_transform(data_input)
data_input = pd.DataFrame(data_input_array, columns=data_input.columns)

data_input_1=data_input[['grade_min']]
X=np.array(data_input_1)

db_1=DBSCAN(eps = 0.03, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_1.labels_, dtype=bool)
core_samples_mask[db_1.core_sample_indices_] = True
labels_1 = db_1.labels_
data['labels_1 : grade_min'] = labels_1

n_clusters_1 = len(set(labels_1)) - (1 if -1 in labels_1 else 0)
n_noise_1 = list(labels_1).count(-1)

print('Variable : grade_min')
print('Estimated number of clusters: %d' % n_clusters_1)
print('Estimated number of noise points: %d' % n_noise_1)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_1))
print()

data_input_2=data_input[['last_grade_min']]
X=np.array(data_input_2)
db_2=DBSCAN(eps = 0.03, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_2.labels_, dtype=bool)
core_samples_mask[db_2.core_sample_indices_] = True
labels_2 = db_2.labels_
data['labels_2 : last_grade_min'] = labels_2

n_clusters_2 = len(set(labels_2)) - (1 if -1 in labels_2 else 0)
n_noise_2 = list(labels_2).count(-1)

print('Variable : last_grade_min')
print('Estimated number of clusters: %d' % n_clusters_2)
print('Estimated number of noise points: %d' % n_noise_2)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_2))
print()

data_input_3=data_input[['last_grade_min','grade_min']]
X=np.array(data_input_3)
db_3=DBSCAN(eps = 0.05, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_3.labels_, dtype=bool)
core_samples_mask[db_3.core_sample_indices_] = True
labels_3 = db_3.labels_
data['labels_3 : last_grade_min, grade_min'] = labels_3

n_clusters_3 = len(set(labels_3)) - (1 if -1 in labels_3 else 0)
n_noise_3 = list(labels_3).count(-1)

print('Variable : last_grade_min, grade_min')
print('Estimated number of clusters: %d' % n_clusters_3)
print('Estimated number of noise points: %d' % n_noise_3)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_3))
print()

unique_labels = set(labels_3)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_3 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.suptitle('other_similarity vs. scholarship_price')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_3, metrics.silhouette_score(X, labels_3)))
plt.show()

data_input_4=data_input[['paybyhour']]
X=np.array(data_input_4)
db_4=DBSCAN(eps = 0.05, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_4.labels_, dtype=bool)
core_samples_mask[db_4.core_sample_indices_] = True
labels_4 = db_4.labels_
data['labels_4 : paybyhour'] = labels_4

n_clusters_4 = len(set(labels_4)) - (1 if -1 in labels_4 else 0)
n_noise_4 = list(labels_4).count(-1)

print('Variable : paybyhour')
print('Estimated number of clusters: %d' % n_clusters_4)
print('Estimated number of noise points: %d' % n_noise_4)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_4))
print()

data_input_5=data_input[['characteristic','characteristic_money','scholarship_price']]
X=np.array(data_input_5)
db_5=DBSCAN(eps = 0.1, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_5.labels_, dtype=bool)
core_samples_mask[db_5.core_sample_indices_] = True
labels_5 = db_5.labels_
data['labels_5 : characteristic, characteristic_money, scholarship_price'] = labels_5

n_clusters_5 = len(set(labels_5)) - (1 if -1 in labels_5 else 0)
n_noise_5 = list(labels_5).count(-1)

print('Variable : characteristic, characteristic_money, scholarship_price')
print('Estimated number of clusters: %d' % n_clusters_5)
print('Estimated number of noise points: %d' % n_noise_5)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_5))
print()

unique_labels = set(labels_5)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_5 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')

x = data_input_5['characteristic']
y = data_input_5['characteristic_money']
z = data_input_5['scholarship_price']
ax.scatter(x,y,z,c=labels_5, s = 20, alpha = 0.5, cmap='rainbow')
plt.xlabel('characteristic')
plt.ylabel('characteristic_money')
ax.set_zlabel('scholarship_price')
plt.suptitle('characteristic(X) vs. characteristic_money(Y) vs. scholarship_price(Z)')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' %(n_clusters_5, metrics.silhouette_score(X, labels_5)))
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')

x = data_input_5['characteristic']
y = data_input_5['characteristic_money']
z = data_input_5['scholarship_price']
ax.scatter(x,y,z,c=labels_5, s = 20, alpha = 0.5, cmap='rainbow')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_5, metrics.silhouette_score(X, labels_5)))
plt.suptitle('characteristic(X) vs. characteristic_money(Y) vs. scholarship_price(Z)')
plt.show()

data_input_6=data_input[['income_max']]
X=np.array(data_input_6)
db_6=DBSCAN(eps = 0.1, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_6.labels_, dtype=bool)
core_samples_mask[db_6.core_sample_indices_] = True
labels_6 = db_6.labels_
data['labels_6 : income_max'] = labels_6

n_clusters_6 = len(set(labels_6)) - (1 if -1 in labels_6 else 0)
n_noise_6 = list(labels_6).count(-1)

print('Variable : income_max')
print('Estimated number of clusters: %d' % n_clusters_6)
print('Estimated number of noise points: %d' % n_noise_6)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_6))
print()

data_input_onehot_1=data_input[['other_similarity']]
X=np.array(data_input_onehot_1)
db_onehot_1=DBSCAN(eps = 0.015, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_1.labels_, dtype=bool)
core_samples_mask[db_onehot_1.core_sample_indices_] = True
labels_onehot_1 = db_onehot_1.labels_
data['labels_onehot_1 : other_similarity'] = labels_onehot_1

n_clusters_onehot_1 = len(set(labels_onehot_1)) - (1 if -1 in labels_onehot_1 else 0)
n_noise_onehot_1 = list(labels_onehot_1).count(-1)

print('Variable : other_similarity')
print('Estimated number of clusters: %d' % n_clusters_onehot_1)
print('Estimated number of noise points: %d' % n_noise_onehot_1)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_1))
print()

data_input_onehot_2=data_input[['other_similarity', 'scholarship_price']]
X = np.array(data_input_onehot_2)
db_onehot_2=DBSCAN(eps = 0.05, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_2.labels_, dtype=bool)
core_samples_mask[db_onehot_2.core_sample_indices_] = True
labels_onehot_2 = db_onehot_2.labels_
data['labels_onehot_2 : other_similarity'] = labels_onehot_2

n_clusters_onehot_2 = len(set(labels_onehot_2)) - (1 if -1 in labels_onehot_2 else 0)
n_noise_onehot_2 = list(labels_onehot_2).count(-1)

print('Variable : other_similarity, scholarship_price')
print('Estimated number of clusters: %d' % n_clusters_onehot_2)
print('Estimated number of noise points: %d' % n_noise_onehot_2)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_2))
print()

import matplotlib.pyplot as plt
unique_labels = set(labels_onehot_2)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_2 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('other_similarity vs. scholarship_price')
plt.suptitle('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_2, metrics.silhouette_score(X, labels_onehot_2)))
plt.show()

data_input_onehot_3=data_input[['other_similarity', 'grade_min']]
X = np.array(data_input_onehot_3)
db_onehot_3=DBSCAN(eps = 0.06, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_3.labels_, dtype=bool)
core_samples_mask[db_onehot_3.core_sample_indices_] = True
labels_onehot_3 = db_onehot_3.labels_
data['labels_onehot_3 : other_similarity, grade_min'] = labels_onehot_3

n_clusters_onehot_3 = len(set(labels_onehot_3)) - (1 if -1 in labels_onehot_3 else 0)
n_noise_onehot_3 = list(labels_onehot_3).count(-1)

print('Variable : other_similarity, grade_min')
print('Estimated number of clusters: %d' % n_clusters_onehot_3)
print('Estimated number of noise points: %d' % n_noise_onehot_3)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_3))
print()

unique_labels = set(labels_onehot_3)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_3 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.suptitle('other_similarity vs. grade_min')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_3, metrics.silhouette_score(X, labels_onehot_3)))
plt.show()

data_input_onehot_4=data_input[['other_similarity', 'characteristic_money']]
X = np.array(data_input_onehot_4)
db_onehot_4=DBSCAN(eps = 0.09, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_4.labels_, dtype=bool)
core_samples_mask[db_onehot_4.core_sample_indices_] = True
labels_onehot_4 = db_onehot_4.labels_
data['labels_onehot_4 : other_similarity, characteristic_money'] = labels_onehot_4

n_clusters_onehot_4 = len(set(labels_onehot_4)) - (1 if -1 in labels_onehot_4 else 0)
n_noise_onehot_4 = list(labels_onehot_4).count(-1)

print('Variable : other_similarity, characteristic_money')
print('Estimated number of clusters: %d' % n_clusters_onehot_4)
print('Estimated number of noise points: %d' % n_noise_onehot_4)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_4))
print()

unique_labels = set(labels_onehot_4)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_4 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.suptitle('other_similarity vs. characteristic_money')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_4, metrics.silhouette_score(X, labels_onehot_4)))
plt.show()

data_input_onehot_5=data_input[['other_similarity', 'characteristic']]
X = np.array(data_input_onehot_5)
db_onehot_5=DBSCAN(eps = 0.09, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_5.labels_, dtype=bool)
core_samples_mask[db_onehot_5.core_sample_indices_] = True
labels_onehot_5 = db_onehot_5.labels_
data['labels_onehot_5 : other_similarity, characteristic'] = labels_onehot_5

n_clusters_onehot_5 = len(set(labels_onehot_5)) - (1 if -1 in labels_onehot_5 else 0)
n_noise_onehot_5 = list(labels_onehot_5).count(-1)

print('Variable : other_similarity, characteristic')
print('Estimated number of clusters: %d' % n_clusters_onehot_5)
print('Estimated number of noise points: %d' % n_noise_onehot_5)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_5))
print()

unique_labels = set(labels_onehot_5)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_5 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.suptitle('other_similarity vs. characteristic')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_5, metrics.silhouette_score(X, labels_onehot_5)))
plt.show()

data_input_onehot_6=data_input[['other_similarity', 'feature_integer']]
X = np.array(data_input_onehot_6)
db_onehot_6=DBSCAN(eps = 0.09, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_6.labels_, dtype=bool)
core_samples_mask[db_onehot_6.core_sample_indices_] = True
labels_onehot_6 = db_onehot_6.labels_
data['labels_onehot_6 : other_similarity, feature_integer'] = labels_onehot_6

n_clusters_onehot_6 = len(set(labels_onehot_6)) - (1 if -1 in labels_onehot_6 else 0)
n_noise_onehot_6 = list(labels_onehot_6).count(-1)

print('Variable : other_similarity, feature_integer')
print('Estimated number of clusters: %d' % n_clusters_onehot_6)
print('Estimated number of noise points: %d' % n_noise_onehot_6)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_6))
print()

unique_labels = set(labels_onehot_6)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_6 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.suptitle('other_similarity vs. feature_integer')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_6, metrics.silhouette_score(X, labels_onehot_6)))
plt.show()

data_input_onehot_7=data_input[['other_similarity', 'characteristic_money', 'grade_min']]
X = np.array(data_input_onehot_7)
db_onehot_7=DBSCAN(eps = 0.09, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_7.labels_, dtype=bool)
core_samples_mask[db_onehot_7.core_sample_indices_] = True
labels_onehot_7 = db_onehot_7.labels_
data['labels_onehot_7 : other_similarity, characteristic_money, grade_min'] = labels_onehot_7

n_clusters_onehot_7 = len(set(labels_onehot_7)) - (1 if -1 in labels_onehot_7 else 0)
n_noise_onehot_7 = list(labels_onehot_7).count(-1)

print('Variable : other_similarity, characteristic_money, grade_min')
print('Estimated number of clusters: %d' % n_clusters_onehot_7)
print('Estimated number of noise points: %d' % n_noise_onehot_7)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_7))
print()

unique_labels = set(labels_onehot_7)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_7 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('other_similarity vs. characteristic_money vs. grade_min')
plt.suptitle('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_7, metrics.silhouette_score(X, labels_onehot_7)))
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')

x = X[:,2]
y = X[:,1]
z = X[:,0]
ax.scatter(x,y,z,c=labels_onehot_7, s = 20, alpha = 0.5, cmap='rainbow')
plt.xlabel('grade_min')
plt.ylabel('characteristic_money')
ax.set_zlabel('other_similarity')
plt.suptitle('grade_min(X) vs. characteristic_money(Y) vs. other_similarity(Z)')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' %(n_clusters_onehot_7, metrics.silhouette_score(X, labels_onehot_7)))
plt.show()

data_input_onehot_8=data_input[['other_similarity', 'characteristic_money', 'characteristic']]
X = np.array(data_input_onehot_8)
db_onehot_8=DBSCAN(eps = 0.12, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_8.labels_, dtype=bool)
core_samples_mask[db_onehot_8.core_sample_indices_] = True
labels_onehot_8 = db_onehot_8.labels_
data['labels_onehot_8 : other_similarity, characteristic_money, characteristic'] = labels_onehot_8

n_clusters_onehot_8 = len(set(labels_onehot_8)) - (1 if -1 in labels_onehot_8 else 0)
n_noise_onehot_8 = list(labels_onehot_8).count(-1)

print('Variable : other_similarity, characteristic_money, characteristic')
print('Estimated number of clusters: %d' % n_clusters_onehot_8)
print('Estimated number of noise points: %d' % n_noise_onehot_8)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_8))
print()

unique_labels = set(labels_onehot_8)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_8 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.suptitle('other_similarity vs. characteristic_money vs. characteristic')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_8, metrics.silhouette_score(X, labels_onehot_8)))
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')

x = data_input_onehot_8['characteristic']
y = data_input_onehot_8['characteristic_money']
z = data_input_onehot_8['other_similarity']
ax.scatter(x,y,z,c=labels_onehot_8, s = 20, alpha = 0.5, cmap='rainbow')
plt.xlabel('characteristic')
plt.ylabel('characteristic_money')
ax.set_zlabel('other_similarity')
plt.suptitle('characteristic(X) vs. characteristic_money(Y) vs. other_similarity(Z)')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' %(n_clusters_onehot_8, metrics.silhouette_score(X, labels_onehot_8)))
plt.show()

data.to_excel("data2.xlsx", encoding="utf-8")
