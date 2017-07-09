#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

/* WNN and PSO parameters that also affect CUDA shared memory size */
#define particle_num 1024 
#define inputs 24 //how many previous sequential inputs should be used to forecast
#define hidd_neurons 2
#define particle_dimension ((3*inputs+1)*hidd_neurons)
#define threadnum 256 //threadnum multiplied by hours_each equals size of training set
#define hours_each 4
#define off 8 //how many values forward to predict, a value of 0 means 1 value forward, a value of 2 means 2 values forward etc
#define training_time_hours (inputs+hours_each*threadnum+off)

/* */

/* Calculate Fitness Function for all particles Kernel */
__global__ void calculate_particle_fitnesses_kernel(float* particles, float* lbest, float* set, float* particle_fitness) {
    int start_set = inputs + threadIdx.x*hours_each;
    int end_set = start_set + hours_each;
    __shared__ float particle_data[particle_dimension];
    __shared__ float training_set[training_time_hours];
    __shared__ float thread_error[threadnum];
    __shared__ float total_thread_error;
    __shared__ int signal_value;

    thread_error[threadIdx.x] = 0;
    int i;

    if (particle_dimension < threadnum) {
        particle_data[threadIdx.x] = particles[blockIdx.x * particle_dimension + threadIdx.x];
    } else {
        int particle_dimension_each = particle_dimension / threadnum;
        int extras = particle_dimension % threadnum;
        for (i = 0; i < particle_dimension_each; i++) {
            particle_data[threadIdx.x * particle_dimension_each + i] = particles[blockIdx.x * particle_dimension + threadIdx.x * particle_dimension_each + i];
        }
        if (threadIdx.x == 0 && extras > 0) {
            for (i = 0; i < extras; i++) {
                particle_data[threadnum * particle_dimension_each + i] = particles[blockIdx.x * particle_dimension + threadnum * particle_dimension_each + i];
            }
        }
    }

    if (training_time_hours < threadnum) {
        training_set[threadIdx.x] = particles[threadIdx.x];
    } else {
        int training_time_hours_each = training_time_hours / threadnum;
        int training_time_hours_extras = training_time_hours % threadnum;
        for (i = 0; i < training_time_hours_each; i++) {
            training_set[i * threadnum + threadIdx.x] = set[i * threadnum + threadIdx.x];
        }
        if (threadIdx.x == 0 && training_time_hours_extras > 0) {
            for (i = 0; i < training_time_hours_extras; i++) {
                training_set[threadnum * training_time_hours_each + i] = set[threadnum * training_time_hours_each + i];
            }
        }

    }

    __syncthreads();
    int n_offset = 3*inputs+1;
    for (int training_set_position = start_set; training_set_position < end_set; training_set_position++) {
        int input_scale_back = inputs;
        float llwnn_output = 0;
        for (int j = 0; j < hidd_neurons; j++) {
            float linear_factor = 0;
            input_scale_back = inputs;
            for (int k = 0; k < inputs; k++) {
                linear_factor = linear_factor + particle_data[ j * n_offset + k + 1 ] * training_set[training_set_position + k - input_scale_back];
            }
            linear_factor = linear_factor + particle_data[ j * n_offset + 0 ];
			float total_wavelet_factor = 0;
			
			for(int k = 0; k < inputs; k++) {
				float a = (float)abs(particle_data[ j * n_offset + inputs + 1 ]);
				if (a == 0) {
                a = 0.00000000001f;
                }
				
				float b = particle_data[ j * n_offset + inputs + inputs + 1 ];
				float x = training_set[training_set_position + k - input_scale_back];
				float in = (x - b) / (a);
				total_wavelet_factor = total_wavelet_factor + (float) pow(a, -0.5f) *((-(in * in) / 2.0f) * (float) exp(-(in * in) / 2.0f));
			}
		
            llwnn_output = llwnn_output + linear_factor * total_wavelet_factor;
        }
        thread_error[threadIdx.x] = thread_error[threadIdx.x]+(training_set[training_set_position + off] - llwnn_output)*(training_set[training_set_position + off] - llwnn_output);
    }   

    atomicAdd(&total_thread_error, thread_error[threadIdx.x]);
	
    __syncthreads();
    if (threadIdx.x == 0) {
        signal_value = 0;
        float local_fitness = sqrt(total_thread_error / (float) (threadnum * hours_each));
        if (local_fitness < particle_fitness[blockIdx.x]) {
            particle_fitness[blockIdx.x] = local_fitness;
            signal_value = 1;

        }
    }
    __syncthreads();

    if (threadIdx.x < particle_dimension && signal_value == 1) {
        lbest[blockIdx.x * particle_dimension + threadIdx.x] = particle_data[threadIdx.x];
    } 
	
}

/* Find gbest particle Kernel */
__global__ void find_gbest_kernel(float *particle_best_positions, float *particle_fitnesses, float *gbest, float *current_gbest_fitness)
{
    __shared__ float particle_mins[particle_num];
    __shared__ int particle_mins_pos[particle_num];
    __shared__ int signal_value;

    int thr = particle_num / 2;
    particle_mins[threadIdx.x] = particle_fitnesses[threadIdx.x];
    particle_mins[thr + threadIdx.x] = particle_fitnesses[thr + threadIdx.x];
    particle_mins_pos[threadIdx.x] = threadIdx.x;
    particle_mins_pos[thr + threadIdx.x] = thr + threadIdx.x;

    while (thr >= 1) {
        if (threadIdx.x < particle_num) {
            if (particle_mins[threadIdx.x] > particle_mins[thr + threadIdx.x]) {
                particle_mins[threadIdx.x] = particle_mins[threadIdx.x + thr];
                particle_mins_pos[threadIdx.x] = particle_mins_pos[threadIdx.x + thr];
            }
        }
        thr = thr / 2;
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        signal_value = 0;
        if (particle_mins[0] < current_gbest_fitness[0]) {
            current_gbest_fitness[0] = particle_mins[0];
            signal_value = 1;
            //!!!! Disable the print for normal usage !!!! Use only for debug
            //printf("fitness: %f \n", current_gbest_fitness[0]);

        }
    }

    __syncthreads();
    if (threadIdx.x < particle_dimension && signal_value == 1) {
        gbest[threadIdx.x] = particle_best_positions[particle_mins_pos[0] * particle_dimension + threadIdx.x];
    }

}

/* Mark the bottom 25% particles, those with the highest fitness values Kernel */
__global__ void mark_worst_kernel(float *particle_fitnesses, float *particle_keep) {
	
    __shared__ float particle_mins[particle_num];
    __shared__ int particle_mins_pos[particle_num];    

    int thr = particle_num / 2;
	
    particle_mins[threadIdx.x] = particle_fitnesses[threadIdx.x];
    particle_mins[threadIdx.x + thr] = particle_fitnesses[thr + threadIdx.x];
    particle_mins_pos[threadIdx.x] = threadIdx.x;
    particle_mins_pos[threadIdx.x + thr] = thr + threadIdx.x;

    while (thr >= particle_num / 4) {
        if (threadIdx.x < particle_num) {
            if (particle_mins[threadIdx.x] < particle_mins[threadIdx.x + thr]) {
                particle_mins[threadIdx.x] = particle_mins[threadIdx.x + thr];
                particle_mins_pos[threadIdx.x] = particle_mins_pos[threadIdx.x + thr];
            }
        }
        thr = thr / 2;
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x < particle_num / 4) {
        particle_keep[particle_mins_pos[threadIdx.x]] = 1;
    }

}

/* Update positions of all particles Kernel */
__global__ void update_positions_kernel(float *particle_positions, float *particle_velocities, float *particle_lbest, float *particle_gbest, float *particle_keep, curandState* globalState, int k, int iterations_total, float b) //, curandState *my_curandstate)
{
    curandState localState = globalState[blockIdx.x * blockDim.x + threadIdx.x];
    int particle_dimension1 = particle_dimension;
    float Dmax1 = 1000000.0f;
    float Xmax1 = 1000000.0f;

    if (particle_keep[blockIdx.x] == 0) {
        float factor1 = (float) curand_uniform(&localState) * 2.0f;
        float factor2 = (float) curand_uniform(&localState) * 2.0f;
        globalState[blockIdx.x * blockDim.x + threadIdx.x] = localState;
        particle_velocities[blockIdx.x * particle_dimension1 + threadIdx.x] = 0.729f * particle_velocities[blockIdx.x * particle_dimension1 + threadIdx.x] + (particle_lbest[blockIdx.x * particle_dimension1 + threadIdx.x] - particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x]) * factor1 + (particle_gbest[threadIdx.x] - particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x]) * factor2;
        if (particle_velocities[blockIdx.x * particle_dimension1 + threadIdx.x] > Dmax1) {
            particle_velocities[blockIdx.x * particle_dimension1 + threadIdx.x] = -Dmax1;
        } else if (particle_velocities[blockIdx.x * particle_dimension1 + threadIdx.x] < -Dmax1) {
            particle_velocities[blockIdx.x * particle_dimension1 + threadIdx.x] = +Dmax1;
        }
        particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] = particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] + particle_velocities[blockIdx.x * particle_dimension1 + threadIdx.x];
        if (particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] > Xmax1) {
            particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] = Xmax1;
        } else if (particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] <= -Xmax1) {
            particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] = -Xmax1;
        }
		
    } else {
        float q = 0.5f;
        float temp2 = (float) curand_uniform(&localState);
        float temp1 = (float) curand_uniform(&localState);
        float temp3 = (float) curand_uniform(&localState);

        if (k > iterations_total / 3) {
            if (temp3 <= 0.5f) {
                particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] = (float) (particle_gbest[threadIdx.x] + (float) curand_uniform(&localState) * 500.0f * (float) (k / iterations_total));
                particle_lbest[blockIdx.x * particle_dimension1 + threadIdx.x] = particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x];
            } else {
                particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] = (float) (particle_gbest[threadIdx.x] - (float) curand_uniform(&localState) * 500.0f * (float) (k / iterations_total));
                particle_lbest[blockIdx.x * particle_dimension1 + threadIdx.x] = particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x];
            }
        }
		
        globalState[blockIdx.x * blockDim.x + threadIdx.x] = localState;

        if (temp2 <= 0.5f) {
            particle_velocities[blockIdx.x * particle_dimension1 + threadIdx.x] = (float) ((1.0f / b) * (float) log(temp1 / (1.0f - q)));
        } else {
            particle_velocities[blockIdx.x * particle_dimension1 + threadIdx.x] = (float) (-(1.0f / b) * (float) log((1.0f - temp1) / q));
        }
        particle_velocities[blockIdx.x * particle_dimension1 + threadIdx.x] = particle_velocities[blockIdx.x * particle_dimension1 + threadIdx.x] * 500.0f;
        particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] = particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] + particle_velocities[blockIdx.x * particle_dimension1 + threadIdx.x];
        if (particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] > Xmax1) {
            particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] = Xmax1;
        } else if (particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] < -Xmax1) {
            particle_positions[blockIdx.x * particle_dimension1 + threadIdx.x] = -Xmax1;
        }
    }

}

/* Init curand states Kernel */
__global__ void setup_kernel(curandState * state, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

int main(int argc, char **argv) {

	/* PSO Algorithm training parameters */
    int num_iterations = atoi(argv[1]);
	float Dmax1 = 1000000.0;
	float b = 0.4;
    float b0 = 0.05;
    clock_t t;
	/* */
	
    if (particle_dimension >= 1024) {
        printf("invalid particle_num size, maximum value is 1024 and your value was %d. Please reduce number of inputs or hidden neurons. \n", particle_dimension);
        return 1;
    }
	
	FILE *myFile;    
	myFile = fopen(argv[2], "r");
	if (myFile == NULL) {
		printf("Error opening input file \n");
		exit(1);
	}

    curandState* devStates;
    cudaMalloc(&devStates, particle_num * threadnum * sizeof ( curandState));
    //initializes the curand kernles
    setup_kernel << < particle_num, particle_dimension >>> (devStates, (unsigned) time(NULL));
    

    /* Read training set file into array */
    float *host_training_set = (float *) malloc((training_time_hours + (training_time_hours * 0.3)) * sizeof (float));  
	
	//test set size = 0.3 * training set size
	//float *host_test_set = (float *) malloc(training_time_hours + training_time_hours * 0.3 * sizeof (float));  
    if (myFile == NULL) {
        printf("Error Reading File\n");
        exit(0);
    }	
	
	int i;	
    for (i = 0; i < training_time_hours + (training_time_hours * 0.3); i++) {
        fscanf(myFile, "%f,", &host_training_set[i]);
    }
    fclose(myFile);
	/* */


	/* Initialiaze variables, cuda mallocs and cuda mem copies */
    float *host_gbest_fitness = (float *) malloc(1 * sizeof (float));
    host_gbest_fitness[0] = 100000000.0;
    float *dev_gbest_fitness;
    cudaMalloc(&dev_gbest_fitness, 1 * sizeof (float));
    cudaMemcpy(dev_gbest_fitness, host_gbest_fitness, 1 * sizeof (float), cudaMemcpyHostToDevice);

    float *host_particles_present_position = (float *) malloc(particle_num * particle_dimension * sizeof (float));
    float *host_particles_localbestpos = (float *) malloc(particle_num * particle_dimension * sizeof (float));
    float *host_particles_velocity = (float *) malloc(particle_num * particle_dimension * sizeof (float));

    //randomize initial weights
    for (i = 0; i < particle_num * particle_dimension; i++) {
        host_particles_present_position[i] = ((float) rand() / RAND_MAX) * Dmax1;
        if ((float) rand() / RAND_MAX < 0.5) {
            host_particles_present_position[i] = -host_particles_present_position[i];
        }
        host_particles_localbestpos[i] = host_particles_present_position[i];
        host_particles_velocity[i] = ((float) rand() / RAND_MAX) * Dmax1;
        if ((float) rand() / RAND_MAX < 0.5) {
            host_particles_velocity[i] = -host_particles_velocity[i];
        }
    }

    int *host_particle_keep = (int *) malloc(particle_num * sizeof (int));
    float *host_particles_fitness = (float *) malloc(particle_num * sizeof (float));
    float *host_globalbestpos = (float *) malloc(particle_dimension * sizeof (float));

    for (i = 0; i < particle_num; i++) {
        host_particles_fitness[i] = 100000000.0;
        host_particle_keep[i] = 0;
    }
    
    float *dev_particles_present_position, *dev_particles_localbestpos, *dev_particles_velocity;
    float *dev_training_set, *dev_particle_keep, *dev_particles_fitness, *dev_globalbestpos;

    cudaMalloc(&dev_particles_present_position, particle_num * particle_dimension * sizeof (float));
    cudaMalloc(&dev_particles_localbestpos, particle_num * particle_dimension * sizeof (float));
    cudaMalloc(&dev_particles_velocity, particle_num * particle_dimension * sizeof (float));
    cudaMalloc(&dev_particles_fitness, particle_num * sizeof (float));
    cudaMalloc(&dev_training_set, training_time_hours * sizeof (float));
    cudaMalloc(&dev_globalbestpos, particle_dimension * sizeof (float));

    cudaMalloc(&dev_particle_keep, particle_num * sizeof (int));

    cudaMemcpy(dev_particles_present_position, host_particles_present_position, particle_num * particle_dimension * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_particles_localbestpos, host_particles_localbestpos, particle_num * particle_dimension * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_particles_velocity, host_particles_velocity, particle_num * particle_dimension * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_particles_fitness, host_particles_fitness, particle_num * sizeof (float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_training_set, host_training_set, training_time_hours * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_particle_keep, host_particle_keep, particle_num * sizeof (int), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_globalbestpos, host_globalbestpos, particle_dimension * sizeof (float), cudaMemcpyHostToDevice);
	/* */

    //Training algorithm commences
    t = clock();
    for (int i = 0; i < num_iterations; i++) {
        calculate_particle_fitnesses_kernel << <particle_num, threadnum>>>(dev_particles_present_position, dev_particles_localbestpos, dev_training_set, dev_particles_fitness);
        if (i > 1) {
            b = (((b - b0) * (num_iterations - i)) / num_iterations) + b0;
        }
        find_gbest_kernel << <1, particle_num / 2 >> >(dev_particles_localbestpos, dev_particles_fitness, dev_globalbestpos, dev_gbest_fitness);
        mark_worst_kernel << <1, particle_num / 2 >> >(dev_particles_fitness, dev_particle_keep);
        update_positions_kernel << <particle_num, particle_dimension>>>(dev_particles_present_position, dev_particles_velocity, dev_particles_localbestpos, dev_globalbestpos, dev_particle_keep, devStates, i, num_iterations, b); //, curandState *my_curandstate)

    }
    cudaMemcpy(host_globalbestpos, dev_globalbestpos, particle_dimension * sizeof (float), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_gbest_fitness, dev_gbest_fitness, 1 * sizeof (float), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
	printf("CUDA RMSE on training set: %f \n", host_gbest_fitness[0]);
    t = clock() - t;
    printf("Training time took %d clicks (%f seconds).\n", t, ((float) t) / CLOCKS_PER_SEC);
	//Training algorithm completed    

    //freeing memory
    cudaFree(dev_particles_present_position);
    cudaFree(dev_particles_localbestpos);
    cudaFree(dev_particles_velocity);
    cudaFree(dev_particles_fitness);
    cudaFree(dev_training_set);

    /* Applying model on test data. By default testdata size is equal to 0.3 * training_data size and is located immediatelly after the training data,
       in the time series. Change it below with the Start and Stop indexes */
	/* Predicted and Actual values are written to file results.txt */
	FILE *f = fopen("results.txt", "w");
	if (f == NULL) {
		printf("Error opening file results.txt \n");
		exit(1);
	}

	
    int testSetStartPosition = training_time_hours;
    int testSetStopPosition = training_time_hours + training_time_hours * 0.3 - off;
    int n_offset = 3*inputs+1;

    float testSetRMSE = 0;    
    float s = 2.0f;
    for (int SetCurrentPosition = testSetStartPosition; SetCurrentPosition < testSetStopPosition; SetCurrentPosition++) {

	    float llwnn_output = 0;
        for (int j = 0; j < hidd_neurons; j++) {
			
			float linear_factor = 0;
            int input_scale_back = inputs;
            for (int k = 0; k < inputs; k++) {
                linear_factor = linear_factor + host_globalbestpos[ j * n_offset + k + 1 ] * host_training_set[SetCurrentPosition + k - input_scale_back];
            }
            linear_factor = linear_factor + host_globalbestpos[ j * n_offset + 0 ];
			float total_wavelet_factor = 0;
			
			for(int k = 0; k < inputs; k++) {
				float a = (float)abs(host_globalbestpos[ j * n_offset + inputs + 1 ]);
				if (a == 0) {
                a = 0.00000000001f;
                }
				
				float b = host_globalbestpos[ j * n_offset + inputs + inputs + 1 ];
				float x = host_training_set[SetCurrentPosition + k - input_scale_back];
				float in = (x - b) / (a);
				total_wavelet_factor = total_wavelet_factor + (float) pow(a, -0.5f) *((-(in * in) / 2.0f) * (float) exp(-(in * in) / 2.0f));
			}
			llwnn_output = llwnn_output + linear_factor*total_wavelet_factor;
        }

        testSetRMSE = testSetRMSE + (llwnn_output - host_training_set[SetCurrentPosition + off]) * (llwnn_output - host_training_set[SetCurrentPosition + off]);
		fprintf(f, "%f %f \n",host_training_set[SetCurrentPosition + off], llwnn_output);
    }
	fclose(f);
    testSetRMSE = sqrt(testSetRMSE / (float) (testSetStopPosition - testSetStartPosition));
    printf("Calculate RMSE on test set: %f \n", testSetRMSE);
    printf("\n");
    printf("\n");
    return 0;
}
