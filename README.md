
Task 1

1. To form a test sample of images from https://press.liacs.nl/mirflickr/;
2. Calculate the following characteristics:
        1. Maximum / minimum value;
        2. Mathematical expectation and variance;
        3. Median values, interquartile range;
        4. asymmetry and kurtosis (normalized);
3. Build a histogram of pixel brightness values;
4. Determine the best fit;
5. Build a list of the types of probabilistic distributions used 
for which have a minimum value of the approximation error.


Task 2

1. To form a test sample of images from the source package;
2. Calculate the following characteristics:
    1. Expectation and variance;
    2. asymmetries and kurtosis are available (normalized);
3. To form the visible parameters of the images, consisting of:
    1. The mathematical expectation of brightness values ​​for each color channel;
    2. The mathematical expectation and dispersion of the brightness values ​​for each color channel;
    3. Mathematical expectations, variances and asymmetry coefficients of brightness values ​​for each color channel;
    4. Mathematical expectations, variances, asymmetry coefficients and kurtosis brightness values ​​for each color channel;
4. Build Gaussian image models using previously calculated parameters.
5. Decompose each channel.
    1. The introducing number of components allows for the reconstruction of individual channels of a 
    color image (gradually moving to a component with a minimum energy).
    2. Construct the dependence of the recovery errors (the standard deviation from the original 
    reconstructed image, MSE) on the number of components used.
6. To conduct a simulation of individual channels of color images using Markov chains:
    1. For each channel, a stochastic matrix of a Markov chain of the first and second orders 
    is calculated (processing pixels horizontally 
    from right to left and vice versa, and also vertically from top to bottom and vice versa). 
    As a result, you will get an explicit image of one of the channels of the color image;
    2. Checking the property of regularity, recurrence and irreversibility (irreducible) 
    for the obtained Markov models for 5 iterations.