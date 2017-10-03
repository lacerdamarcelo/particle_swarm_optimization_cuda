/*This code algorithm could be better implemented if I had a device compatible
with dynamic threads (thread allocation from other threads in the device) and
thread groups, which I could use to sync threads from different blocks.*/

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define POS_MAX 100
#define INIT_W 0.9
#define FINAL_W 0.4
#define INIT_C1 2
#define FINAL_C1 2
#define INIT_C2 2
#define FINAL_C2 2
#define VEL_CLAMPING_FACTOR 0.8
#define MAX_ITERATIONS 5000
//For my GeForce 820m
#define MAX_THREAD_PER_BLOCK 1024

__device__ float sphere_function(float *vector, int individual_id, int dim){
	float sum = 0;
	for(int i = 0; i < dim; i++){
		sum += vector[individual_id * dim + i] * vector[individual_id * dim + i];
	}
	return -sum;
}

__global__ void get_global_best(float* positions_ptr, float* fitnesses_ptr,
								float *gbest, float *gbest_fitness, int dim,
								int pop_size){
	extern __shared__ int active_indexes[];
	int active_indexes_index = threadIdx.x;
	while(active_indexes_index < pop_size){
		active_indexes[active_indexes_index] = active_indexes_index;
		active_indexes_index += blockDim.x;
	}
	__syncthreads();	

	int active_indexes_counter = pop_size;
	while(active_indexes_counter != 1){
		int max_thread_index = active_indexes_counter / 2;
		int index_active_indexes1 = threadIdx.x;
		int index_active_indexes2 = threadIdx.x + max_thread_index;
	
		/*
		I found by accident that I needed to put these syncthreads in order
		to avoid some race condition, but I cannot see where there should be
		a race condition.
		*/
		while(index_active_indexes1 <= max_thread_index){
			if((index_active_indexes1 == max_thread_index) &&
				(index_active_indexes2 == active_indexes_counter)){
				__syncthreads();
				active_indexes[index_active_indexes1] = 
					active_indexes[index_active_indexes2 - 1];
			}else{
				float fitness1 =
					fitnesses_ptr[active_indexes[index_active_indexes1]];
				float fitness2 =
					fitnesses_ptr[active_indexes[index_active_indexes2]];
				int best_fitness_index = fitness1 > fitness2 ? 
					index_active_indexes1 :	index_active_indexes2;
				
				active_indexes[index_active_indexes1] =
					active_indexes[best_fitness_index];
			}
			index_active_indexes1 += blockDim.x;
			index_active_indexes2 += blockDim.x;
		}
		active_indexes_counter = active_indexes_counter % 2 == 0 ?
								 active_indexes_counter / 2 :
								 (active_indexes_counter / 2) + 1;
		__syncthreads();
	}
	
	if(fitnesses_ptr[active_indexes[0]] > *gbest_fitness){
		int index = threadIdx.x;
		while(index < dim){
			gbest[index] = positions_ptr[active_indexes[0] * dim + index];
			index += blockDim.x;
		}
	}

	__syncthreads();

	if(threadIdx.x == 0){
		if(fitnesses_ptr[active_indexes[0]] > *gbest_fitness){
			*gbest_fitness = fitnesses_ptr[active_indexes[0]];
		}
	}
}

__global__ void init_population(float* positions_ptr, float* velocities_ptr,
		float* fitnesses_ptr, float* personal_bests_ptr, float* personal_best_fitness_ptr,
		unsigned int nsec_time, float *gbest, float *gbest_fitness, int dim,
		int pop_size){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t state;
	curand_init(index + nsec_time, /* the seed controls the sequence of random values
					that are produced */
              0, /* the sequence number is only important with
              		multiple cores */
              0, /* the offset is how much extra we advance in the
              		sequence for each call, can be 0 */
              &state);
	int thread_index = threadIdx.x;
	while(thread_index < dim){
		unsigned int random_number = curand(&state);
		float value = ((float)(random_number % 100000)) / 100000;
		value = (((float)POS_MAX / 2) * value) + (POS_MAX / 2);
		positions_ptr[blockIdx.x * dim + thread_index] = value;
		personal_bests_ptr[blockIdx.x * dim + thread_index] = value;
		velocities_ptr[blockIdx.x * dim + thread_index] = 0;
		thread_index += blockDim.x;
	}
	
	__syncthreads();

	if(threadIdx.x == 0){
		fitnesses_ptr[blockIdx.x] = sphere_function(positions_ptr, blockIdx.x, dim);
		personal_best_fitness_ptr[blockIdx.x] = fitnesses_ptr[blockIdx.x];
	}

	if(blockIdx.x == 0){
		int thread_index = threadIdx.x;
		while(thread_index < dim){
			gbest[threadIdx.x] = positions_ptr[thread_index];
			thread_index += blockDim.x;
		}
		*gbest_fitness = fitnesses_ptr[0];
	}
}

__global__ void update_particle(float* positions_ptr, float* velocities_ptr,
			float* fitnesses_ptr, float* personal_bests_ptr, float* personal_best_fitness_ptr,
			unsigned int nsec_time,	float *gbest, float inertia_weight, float c1,
			float c2, int dim){
	curandState_t state;
	curand_init(blockIdx.x + nsec_time, /* the seed controls the sequence of random values
					that are produced */
              0, /* the sequence number is only important with
              		multiple cores */
              0, /* the offset is how much extra we advance in the
              		sequence for each call, can be 0 */
              &state);

	int index = threadIdx.x;

	while(index < dim){
		float r1 = (float)(curand(&state) % 100001) / 100000;
		float r2 = (float)(curand(&state) % 100001) / 100000;
		/*if(blockIdx.x == 0){
			printf("%d, %f, %f, %f, %f, %f, %f, %f\n", index, r1, r2, c1, c2, inertia_weight, population[blockIdx.x].personal_best_fitness);
		}*/
		velocities_ptr[blockIdx.x * dim + index] = inertia_weight *
			velocities_ptr[blockIdx.x * dim + index] +
			c1 * r1 * (personal_bests_ptr[blockIdx.x * dim + index] -
								  positions_ptr[blockIdx.x * dim + index]) +
			c2 * r2 * (gbest[index] -
								  positions_ptr[blockIdx.x * dim + index]);

		if(velocities_ptr[blockIdx.x * dim + index] > VEL_CLAMPING_FACTOR *
			POS_MAX){
			velocities_ptr[blockIdx.x * dim + index] = VEL_CLAMPING_FACTOR *
														   POS_MAX;
		}else if(velocities_ptr[blockIdx.x * dim + index] <
					-VEL_CLAMPING_FACTOR * POS_MAX){
			velocities_ptr[blockIdx.x * dim + index] = -VEL_CLAMPING_FACTOR *
															POS_MAX;
		}

		positions_ptr[blockIdx.x * dim + index] +=
			velocities_ptr[blockIdx.x * dim + index];

		if(positions_ptr[blockIdx.x * dim + index] > POS_MAX){
			positions_ptr[blockIdx.x * dim + index] = POS_MAX;
			velocities_ptr[blockIdx.x * dim + index] *= -1;
		}else if(positions_ptr[blockIdx.x * dim + index] < -POS_MAX){
			positions_ptr[blockIdx.x * dim + index] = -POS_MAX;
			velocities_ptr[blockIdx.x * dim + index] *= -1;
		}

		__syncthreads();

		//Calculate new fitness
		if(index == 0){
			fitnesses_ptr[blockIdx.x] =
				sphere_function(positions_ptr, blockIdx.x, dim);
		}

		__syncthreads();

		//Update personal best if necessary
		if(fitnesses_ptr[blockIdx.x] >
			personal_best_fitness_ptr[blockIdx.x]){
			personal_bests_ptr[blockIdx.x * dim + index] =
				positions_ptr[blockIdx.x * dim + index];
		}

		//TODO - PRECISO DISSO MESMO?
		__syncthreads();

		//Update new personal best fitness if necessary
		if(index == 0){
			if(fitnesses_ptr[blockIdx.x] >
				personal_best_fitness_ptr[blockIdx.x]){
					personal_best_fitness_ptr[blockIdx.x] =
						fitnesses_ptr[blockIdx.x];
			}
		}
		index += blockDim.x;
	}

}

int main(){
	int D_values_length = 4;
	int N_values_length = 4;
	int repetitions = 30;
	int D_values[] = {800, 1600, 3200, 6400};
	int N_values[] = {640, 1280, 2560, 5120};
	//int D_values[] = {100, 200, 400, 800, 1600, 3200, 6400};
	//int N_values[] = {10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120};
	//int D_values[] = {1, 2, 4, 8, 16, 32, 64};
	//int N_values[] = {10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120};
	printf("dim\tpop\ttime\tbest_fitness\tscore\n");

	for(int d_value = 0; d_value < D_values_length; d_value++){
		int D = D_values[d_value];
		for(int n_value = 0; n_value < N_values_length; n_value++){
			float avg_fitness_result = 0;
			double avg_time_result = 0;		
			double avg_general_score = 0;
			int N = N_values[n_value];
			for(int k = 0; k < repetitions; k++){
				clock_t begin = clock();
				struct timespec rawtime;

				float positions[N * D];
				float fitnesses[N];
				float gbest[D];
				float gbest_fitness;

				float *positions_ptr;
				cudaMalloc((void**) &positions_ptr, N * D * sizeof(float));

				float *velocities_ptr;
				cudaMalloc((void**) &velocities_ptr, N * D * sizeof(float));

				float *fitnesses_ptr;
				cudaMalloc((void**) &fitnesses_ptr, N * sizeof(float));

				float *personal_bests_ptr;
				cudaMalloc((void**) &personal_bests_ptr, N * D * sizeof(float));

				float *personal_best_fitness_ptr;
				cudaMalloc((void**) &personal_best_fitness_ptr, N * sizeof(float));
				
				float *gbest_dev_ptr;
				cudaMalloc((void**) &gbest_dev_ptr, D * sizeof(float));

				float *gbest_fitness_dev_ptr;
				cudaMalloc((void**) &gbest_fitness_dev_ptr, sizeof(float));	

				clock_gettime(CLOCK_MONOTONIC_RAW, &rawtime);
				unsigned int time_nsec = rawtime.tv_nsec;

				int allocated_threads_get_gbest = N < MAX_THREAD_PER_BLOCK ? N : MAX_THREAD_PER_BLOCK;
				int allocated_threads_dimensions = D < MAX_THREAD_PER_BLOCK ? D : MAX_THREAD_PER_BLOCK;
				init_population<<<N, allocated_threads_dimensions>>>(positions_ptr, velocities_ptr,
					fitnesses_ptr, personal_bests_ptr, personal_best_fitness_ptr, time_nsec,
					gbest_dev_ptr,gbest_fitness_dev_ptr, D, N);
				get_global_best<<<1, allocated_threads_get_gbest, N * sizeof(int)>>>(positions_ptr, fitnesses_ptr, gbest_dev_ptr, gbest_fitness_dev_ptr, D, N);

				float inertia_weight = INIT_W;
				float c1 = INIT_C1;
				float c2 = INIT_C2;
				for(int i = 0; i < MAX_ITERATIONS; i++){
					clock_gettime(CLOCK_MONOTONIC_RAW, &rawtime);
					time_nsec = rawtime.tv_nsec;
					update_particle<<<N, allocated_threads_dimensions>>>(positions_ptr, velocities_ptr,
						fitnesses_ptr, personal_bests_ptr, personal_best_fitness_ptr, time_nsec, gbest_dev_ptr,
						inertia_weight, c1, c2, D);
					get_global_best<<<1, allocated_threads_get_gbest, N * sizeof(int)>>>(positions_ptr, fitnesses_ptr,
						gbest_dev_ptr, gbest_fitness_dev_ptr, D, N);
					inertia_weight += (FINAL_W - INIT_W) / MAX_ITERATIONS;
					c1 += (FINAL_C1 - INIT_C1) / MAX_ITERATIONS;
					c2 += (FINAL_C2 - INIT_C2) / MAX_ITERATIONS;
				}

				cudaMemcpy(&positions, positions_ptr, N * D * sizeof(float),
						   cudaMemcpyDeviceToHost);
				cudaMemcpy(&fitnesses, fitnesses_ptr, N * sizeof(float),
					   	   cudaMemcpyDeviceToHost);
				cudaMemcpy(&gbest_fitness, gbest_fitness_dev_ptr, sizeof(float),
						   cudaMemcpyDeviceToHost);
				cudaMemcpy(&gbest, gbest_dev_ptr, D * sizeof(float),
					   	   cudaMemcpyDeviceToHost);

				/*for(int i = 0; i < N; i++){		
					for(int j = 0; j < D; j++){
						printf("%f ", positions[i * D + j]);
					}
					printf("= %f\n", fitnesses[i]);
				}
				printf("\n");*/

				/*for(int i = 0; i < D; i++){
					printf("%f ", gbest[i]);
				}
				printf("= %f\n", gbest_fitness);*/
					
				cudaFree(positions_ptr);
				cudaFree(velocities_ptr);
				cudaFree(fitnesses_ptr);
				cudaFree(personal_bests_ptr);
				cudaFree(personal_best_fitness_ptr);
				cudaFree(gbest_dev_ptr);
				cudaFree(gbest_fitness_dev_ptr);
				clock_t end = clock();
				double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
				avg_fitness_result += -gbest_fitness;
				avg_time_result += time_spent;
				avg_general_score += sqrt(pow(gbest_fitness, 2) + pow(time_spent, 2));

			}
			printf("%d\t%d\t%lf\t%f\t%lf\n", D, N, avg_time_result / repetitions,
					avg_fitness_result / repetitions, avg_general_score);
		}
		printf("\n");
	}	
}