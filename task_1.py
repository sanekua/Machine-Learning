
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


DIRNAME = '/home/alex/Стільниця/test'

COLOR = {'red': 0,
         'green': 1,
         'blue': 2}  # RGB


with open('/home/alex/Стільниця/test') as f:
    image_names = ['im'+ x.strip()+'.jpg' for x in f.readlines()]



print (len(image_names))


data = {}
for name, num in COLOR.items():
    data[name] = pd.DataFrame()
    for image_name in image_names[:1]:
        image = np.array(Image.open(DIRNAME+image_name))
        a = image[:, :, num].ravel()
        d = {'name': image_name,
             'min': np.min(a),
             'max': np.max(a),
             'mean': np.mean(a),
             'var': np.var(a),
             'median': np.median(a),
             'interquartile': sp.stats.iqr(a),
             'skewness': sp.stats.skew(a),
             'kurtosis': sp.stats.kurtosis(a)}
        data[name] = pd.concat([data[name], pd.DataFrame(pd.DataFrame(d, index=[0,]))], ignore_index=True)


print('Значения RED')
print (data['red'])

print('Значения GREEN')
print (data['green'])


print('Значения BLUE')
print (data['blue'])


hist = {'red': defaultdict(int),
        'green': defaultdict(int),
        'blue': defaultdict(int),}

for name, num in COLOR.items():
    for image_name in image_names[0:1]:
        figure()
        image = np.array(Image.open(os.path.join(DIRNAME, image_name)))
        a = image[:, :, num].ravel()
        f = Fitter(a, distributions=['beta', 'gamma', 'uniform', 'norm'], bins=256)
        f.fit()
        f.summary()
        f.hist()
        hist[name][f.df_errors['sumsquare_error'].idxmin()] += 1
        show()