# cudaWnnTimeSeriesForecasting
A Local Linear Wavelet Neural Network trained by Particle Swarm Optimization implemented in CUDA for times series forecasting.

Athanassios Kintsakis
athanassios.kintsakis@gmail.com
akintsakis@issel.ee.auth.gr

The implementation of the current system in cuda and the improvements in the PSO training algorithm can be found in
http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=7216611&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D7216611

Should you find the system useful, please cite the above work.

# Implementation Details
Please check the defined parameters at the top of the .cu file, they directly affect shared memory. Be sure to remain within the limits of your GPU architecture.

# Compile and Run
Compile with nvcc. You can experiment with the example input datasets. To run, you need to provide only one argument, the name of the input file that must be located within the same folder with the executable. For example, in windows to compile and run you can use

nvcc cudaWnnTimeSeriesForecasting.cu  
a.exe sample_input1_sinusoidal_plateau.txt  

you can replace the input file with any other valid input file, either from the included samples or your own.

# How to configure for maximum accuracy
The values of the inputs, size of training set and how many values forward to forecast depend on the actual input data.
Below are some values that seem to yield acceptable forecasts with the sample datasets (for all examples below we assume that threadnum is set to 256)

sample_input1_sinusoidal_plateau.txt  
define inputs 8  
define hidd_neurons 2  
define hours_each 4  
define off 5  

sample_input2_sinusoidal_plateau_random.txt  
define inputs 8  
define hidd_neurons 2  
define hours_each 4  
define off 5  

sample_input3_mckey_glass.txt  
define inputs 8  
define hidd_neurons 2  
define hours_each 4  
define off 5  

sample_input4_greek_energy_market_load.txt  
define inputs 24  
define hidd_neurons 2  
define hours_each 4  
define off 5  
