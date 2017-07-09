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

The following compile archs have been tested:
* nvcc -gencode arch=compute_35,code=sm_35 cudaWnnTimeSeriesForecasting.cu  
* nvcc -gencode arch=compute_61,code=sm_61 cudaWnnTimeSeriesForecasting.cu 

To run:
* a.out <num_of_pso_iterations> <input_file> 
Example
* ./a.out 3000 sample_input4_greek_energy_market_load.txt

you can replace the input file with any other valid input file, either from the included samples or your own. Please follow the format of the included samples.

# How to configure for maximum accuracy
The values of the inputs, size of training set and how many values forward to forecast depend on the actual input data.
Below are some values that seem to yield acceptable forecasts with the sample datasets (for all examples below we assume that threadnum is set to 256). The off value is how many values forward to predict, a value of 0 is for 1 value forward, a value of 23 is for 24 values forward etc. This parameter is to be set according to the user requirements. Generally, the higher the value, the less the expected accuracy.

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
