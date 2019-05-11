import numpy as np
import scipy as sp
from scipy import stats
from PIL import Image
import matplotlib.pyplot as plt
from fitter import Fitter


DIRNAME = '/home/alex/Стільниця/mirflickr/'
COLOR = {'red': 0,'green': 1,'blue': 2}
number_of_image = 1
image_names = []
for i in range(number_of_image):
    image_names.append('/home/alex/Стільниця/mirflickr/im'+str(i+1)+'.jpg')

f = open('extract.txt', 'w')
#image_names=['/home/alex/Стільниця/mirflickr/im2.jpg']
#image_names = ['/home/alex/Стільниця/mirflickr/im1.jpg', '/home/alex/Стільниця/mirflickr/im2.jpg']

for image_name in image_names:
    for name, number in COLOR.items():
        image = np.array(Image.open(image_name))
        a = image[:, :, number].ravel()
        d = {'name': image_name,
             'minimum': np.min(a),
             'maximum': np.max(a),
             'mean_value': np.mean(a),
             #dispers
             'variance': np.var(a),
             'mediana': np.median(a),
             'interquartile': sp.stats.iqr(a, rng=(25, 75)),
             'skewness': sp.stats.skew(a),
             'kurtosis': sp.stats.kurtosis(a),
             'summmm':sum(a)}
        print('name_channel for: {} channel and extracting data {}'.format(name,d))
        #f.write('name_channel for: {} channel and extracting data {}\n'.format(name,d))
        plt.figure(figsize=(10,5))
        f = Fitter(a, distributions=['beta', 'gamma', 'laplace', 'norm', 'uniform'],bins=256)
        f.fit()
        f.summary()
        f.hist()
        plt.show()
