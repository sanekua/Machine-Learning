import os
from collections import defaultdict

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
import sys
import threading
from datetime import datetime
from itertools import zip_longest
from PIL import Image, ImageDraw
import random







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
        MEAN_VECTOR_R.append(round(d['mean'],3))
        MEAN_VECTOR_G.append(round(d['mean'],3))
        MEAN_VECTOR_B.append(round(d['mean'],3))

        VAR_VECTOR_R.append(round(d['var'],3))
        VAR_VECTOR_G.append(round(d['var'],3))
        VAR_VECTOR_B.append(round(d['var'],3))

        SKEW_VECTOR_R.append(round(d['skewness'],3))
        SKEW_VECTOR_G.append(round(d['skewness'],3))
        SKEW_VECTOR_B.append(round(d['skewness'],3))

        KURT_VECTOR_R.append(round(d['kurtosis'],3))
        KURT_VECTOR_G.append(round(d['kurtosis'],3))
        KURT_VECTOR_B.append(round(d['kurtosis'],3))

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
