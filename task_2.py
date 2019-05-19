import numpy as np
import scipy as sp
from scipy import misc, stats
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.plotly as py
import seaborn as sns
from fitter import Fitter
from pylab import *
from itertools import zip_longest
from PIL import Image, ImageDraw
import random


DIRNAME = '/Users/alex/Downloads/mirflickr/'
COLOR = {'red': 0,
         'green': 1,
         'blue': 2}  # RGB


with open('/Users/alex/Downloads/mirflickr') as f:
    image_names = ['im'+ x.strip()+'.jpg' for x in f.readlines()]

#Zadanie 1





print (len(image_names))

MEAN_VECTOR_R = []
MEAN_VECTOR_G = []
MEAN_VECTOR_B = []

VAR_VECTOR_R = []
VAR_VECTOR_G = []
VAR_VECTOR_B = []

SKEW_VECTOR_R = []
SKEW_VECTOR_G = []
SKEW_VECTOR_B = []

KURT_VECTOR_R = []
KURT_VECTOR_G = []
KURT_VECTOR_B = []
data = {}
for name, num in COLOR.items():
    data[name] = pd.DataFrame()
    for image_name in image_names[:1]:
        image = np.array(Image.open(DIRNAME+image_name))
        a = image[:, :, num].ravel()
        d = {'name': image_name,
             'mean': np.mean(a),
             'var': np.var(a),
             'skewness': sp.stats.skew(a),
             'kurtosis': sp.stats.kurtosis(a)}
        data[name] = pd.concat([data[name], pd.DataFrame(pd.DataFrame(d, index=[0,]))], ignore_index=True)
        MEAN_VECTOR_R.append(round(d['mean'], 3))
        MEAN_VECTOR_G.append(round(d['mean'], 3))
        MEAN_VECTOR_B.append(round(d['mean'], 3))

        VAR_VECTOR_R.append(round(d['var'], 3))
        VAR_VECTOR_G.append(round(d['var'], 3))
        VAR_VECTOR_B.append(round(d['var'], 3))

        SKEW_VECTOR_R.append(round(d['skewness'], 3))
        SKEW_VECTOR_G.append(round(d['skewness'], 3))
        SKEW_VECTOR_B.append(round(d['skewness'], 3))

        KURT_VECTOR_R.append(round(d['kurtosis'], 3))
        KURT_VECTOR_G.append(round(d['kurtosis'], 3))
        KURT_VECTOR_B.append(round(d['kurtosis'], 3))

MATRICA_MEAN_ARRAY = np.array((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B))
MATRICA_VAR_ARRAY = np.array((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B))
MATRICA_SKEW_ARRAY = np.array((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B, SKEW_VECTOR_R, SKEW_VECTOR_G, SKEW_VECTOR_B))
MATRICA_KURT_ARRAY = np.array((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B, SKEW_VECTOR_R, SKEW_VECTOR_G, SKEW_VECTOR_B, KURT_VECTOR_R, KURT_VECTOR_G, KURT_VECTOR_B))

#################################
print('Матрица Мат.Ожидания')
print(MATRICA_MEAN_ARRAY)

MATRICA_MEAN_ARRAY_COV = np.cov(np.vstack((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B)))
print('Матрица ковариации Мат.Ожидания')
print(MATRICA_MEAN_ARRAY_COV)


print('Матрица Мат.Ожидания и дисперсии')
print(MATRICA_VAR_ARRAY)

MATRICA_MATRICA_VAR_COV = np.array(np.cov(np.vstack((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B))))
print('Матрица ковариации Мат.Ожидания и дисперсии')
print((MATRICA_MATRICA_VAR_COV))

#################################

print('Матрица Мат.Ожидания ,дисперсии и ексцесса')
print(MATRICA_SKEW_ARRAY)

MATRICA_SKEW_ARRAY_COV = np.array(np.cov(np.vstack((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B, SKEW_VECTOR_R, SKEW_VECTOR_G, SKEW_VECTOR_B))))
print('Матрица ковариации Мат.Ожидания ,дисперсии и ексцесса')
print(MATRICA_SKEW_ARRAY_COV)



for name, num in COLOR.items():
    data[name] = pd.DataFrame()

for image_name in image_names[:1]:
        image = np.array(Image.open(DIRNAME+image_name))
        print(len(image))
        print(len(image[0]))
        r = image[:, :, 0].ravel()
        g = image[:, :, 1].ravel()
        b = image[:, :, 2].ravel()


print(len(r))
print(len(g))
print(len(b))
print(r)
print(g)
print(b)


r_new = []
def chunks(lst, count):
    n = len(lst) // count
    return list(x for x in zip_longest(*[iter(lst)] * n))

r_new = chunks(r,375)
print((np.array(r_new)))

print(len(r_new))
print(len(r_new[1]))
U, s, V = np.linalg.svd(r_new, full_matrices=True)
print('Матрица U')
print (np.array(U))
print('Список s')
print (np.array(s))
print('Матрица V')
print (np.array(V))

print(len(s))

nulevaya_matriz = np.zeros((375, 500))
nulevaya_matriz[0][0] = s[0]
print (np.array(nulevaya_matriz))

Obshaya_matrica1 = np.dot(U, nulevaya_matriz)
Obshaya_matrica2 = np.dot(Obshaya_matrica1, V)
print ('Матрица U*s*V, где s - нулевая матрица кроме первого елемента')
print (len(Obshaya_matrica2[0]))
print (len(Obshaya_matrica2))


def listmerge1(lstlst):
    all=[]
    for lst in lstlst:
        for el in lst:
            all.append(el)
    return all


image = Image.open(DIRNAME+'im20171.jpg') #Открываем изображение.
draw = ImageDraw.Draw(image) #Создаем инструмент для рисования.
width = image.size[0] #Определяем ширину.
height = image.size[1] #Определяем высоту.
pix = image.load() #Выгружаем значения пикселей.

for i in range(500):
  for j in range(375):
        a = int(Obshaya_matrica2[j][i])
        b = pix[i, j][1]
        c = pix[i, j][2]
        draw.point((i, j), (a, b, c))
image.save("ans.jpg", "JPEG")
del draw


def transition_matrix(transitions):
    n = 1+ max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

stohastic_marix = np.array(transition_matrix(r))
print('Стохастическая матрица')
print(stohastic_marix)
print('Результат суммирования стохастической матрицы')
print((sum(stohastic_marix))/256)