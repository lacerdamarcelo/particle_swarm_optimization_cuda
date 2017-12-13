/*This code algorithm could be better implemented if I had a device compatible
with dynamic threads (thread allocation from other threads in the device) and
thread groups, which I could use to sync threads from different blocks.*/

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#include "pso_cuda.h"

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
								float *gbest, float *gbest_fitness, int* dim_ptr,
								int* pop_size_ptr, int* active_indexes_pos){
	int dim = *dim_ptr;
	int pop_size = *pop_size_ptr;
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
					fitnesses_ptr[active_indexes_pos[active_indexes[index_active_indexes1]]];
				float fitness2 =
					fitnesses_ptr[active_indexes_pos[active_indexes[index_active_indexes2]]];
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
	
	if(fitnesses_ptr[active_indexes_pos[active_indexes[0]]] > *gbest_fitness){
		int index = threadIdx.x;
		while(index < dim){
			gbest[index] = positions_ptr[active_indexes_pos[active_indexes[0]] * dim + index];
			index += blockDim.x;
		}
	}

	__syncthreads();

	if(threadIdx.x == 0){
		if(fitnesses_ptr[active_indexes_pos[active_indexes[0]]] > *gbest_fitness){
			*gbest_fitness = fitnesses_ptr[active_indexes_pos[active_indexes[0]]];
		}
	}
}

__global__ void init_population(float* positions_ptr, float* velocities_ptr,
		float* fitnesses_ptr, float* personal_bests_ptr, float* personal_best_fitness_ptr,
		unsigned int* nsec_time_ptr, float *gbest, float *gbest_fitness, int* dim_ptr,
		int* active_indexes){
	unsigned int nsec_time = *nsec_time_ptr;
	int dim = *dim_ptr;
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
		positions_ptr[active_indexes[blockIdx.x] * dim + thread_index] = value;
		personal_bests_ptr[active_indexes[blockIdx.x] * dim + thread_index] = value;
		velocities_ptr[active_indexes[blockIdx.x] * dim + thread_index] = 0;
		thread_index += blockDim.x;
	}
	
	__syncthreads();

	if(threadIdx.x == 0){
		fitnesses_ptr[active_indexes[blockIdx.x]] = sphere_function(positions_ptr,
			active_indexes[blockIdx.x], dim);
		personal_best_fitness_ptr[active_indexes[blockIdx.x]] = fitnesses_ptr[active_indexes[blockIdx.x]];
	}

	if(blockIdx.x == 0){
		int thread_index = threadIdx.x;
		while(thread_index < dim){
			gbest[threadIdx.x] = positions_ptr[active_indexes[0] + thread_index];
			thread_index += blockDim.x;
		}
		*gbest_fitness = fitnesses_ptr[active_indexes[0]];
	}
}

__global__ void update_particle(float* positions_ptr, float* velocities_ptr,
			float* fitnesses_ptr, float* personal_bests_ptr, float* personal_best_fitness_ptr,
			unsigned int* nsec_time_ptr, float *gbest, float* inertia_weight_ptr, float* c1_ptr,
			float* c2_ptr, int* dim_ptr, int* active_indexes){
	unsigned int nsec_time = *nsec_time_ptr;
	float inertia_weight = *inertia_weight_ptr;
	float c1 = *c1_ptr;
	float c2 = *c2_ptr;
	int dim = *dim_ptr;
	curandState_t state;
	curand_init(blockIdx.x + nsec_time, 0, 0, &state);

	int index = threadIdx.x;
	while(index < dim){
		float r1 = (float)(curand(&state) % 100001) / 100000;
		float r2 = (float)(curand(&state) % 100001) / 100000;
		/*printf("-0 - %f %f %f %f %f %f %f %f %f\n",
			personal_bests_ptr[active_indexes[blockIdx.x] * dim + index],
			gbest[index], positions_ptr[active_indexes[blockIdx.x] * dim + index],
			velocities_ptr[active_indexes[blockIdx.x] * dim + index], c1, c2, r1,
			r2, inertia_weight);*/
		velocities_ptr[active_indexes[blockIdx.x] * dim + index] = inertia_weight *
			velocities_ptr[active_indexes[blockIdx.x] * dim + index] +
			c1 * r1 * (personal_bests_ptr[active_indexes[blockIdx.x] * dim + index] -
								  positions_ptr[active_indexes[blockIdx.x] * dim + index]) +
			c2 * r2 * (gbest[index] -
								  positions_ptr[active_indexes[blockIdx.x] * dim + index]);

		if(velocities_ptr[active_indexes[blockIdx.x] * dim + index] > VEL_CLAMPING_FACTOR *
			POS_MAX){
			velocities_ptr[active_indexes[blockIdx.x] * dim + index] = VEL_CLAMPING_FACTOR *
														   POS_MAX;
		}else if(velocities_ptr[active_indexes[blockIdx.x] * dim + index] <
					-VEL_CLAMPING_FACTOR * POS_MAX){
			velocities_ptr[active_indexes[blockIdx.x] * dim + index] = -VEL_CLAMPING_FACTOR *
															POS_MAX;
		}
		/*printf("-1 - %f %f %f %f %f %f %f %f %f\n",
			personal_bests_ptr[active_indexes[blockIdx.x] * dim + index],
			gbest[index], positions_ptr[active_indexes[blockIdx.x] * dim + index],
			velocities_ptr[active_indexes[blockIdx.x] * dim + index], c1, c2, r1,
			r2, inertia_weight);*/
		positions_ptr[active_indexes[blockIdx.x] * dim + index] +=
			velocities_ptr[active_indexes[blockIdx.x] * dim + index];
		if(positions_ptr[active_indexes[blockIdx.x] * dim + index] > POS_MAX){
			positions_ptr[active_indexes[blockIdx.x] * dim + index] = POS_MAX;
			velocities_ptr[active_indexes[blockIdx.x] * dim + index] *= -1;
		}else if(positions_ptr[active_indexes[blockIdx.x] * dim + index] < -POS_MAX){
			positions_ptr[active_indexes[blockIdx.x] * dim + index] = -POS_MAX;
			velocities_ptr[active_indexes[blockIdx.x] * dim + index] *= -1;
		}

		/*printf("-2 - %f %f %f %f %f %f %f %f %f\n",
			personal_bests_ptr[active_indexes[blockIdx.x] * dim + index],
			gbest[index], positions_ptr[active_indexes[blockIdx.x] * dim + index],
			velocities_ptr[active_indexes[blockIdx.x] * dim + index], c1, c2, r1,
			r2, inertia_weight);*/

		__syncthreads();
		//Calculate new fitness
		if(index == 0){		
			fitnesses_ptr[active_indexes[blockIdx.x]] =
				sphere_function(positions_ptr, active_indexes[blockIdx.x], dim);
		}
		__syncthreads();

		//Update personal best if necessary
		if(fitnesses_ptr[active_indexes[blockIdx.x]] >
			personal_best_fitness_ptr[active_indexes[blockIdx.x]]){
			personal_bests_ptr[active_indexes[blockIdx.x] * dim + index] =
				positions_ptr[active_indexes[blockIdx.x] * dim + index];
		}

		//TODO - DO I REALLY NEED THIS?
		__syncthreads();

		//Update new personal best fitness if necessary
		if(index == 0){
			if(fitnesses_ptr[active_indexes[blockIdx.x]] >
				personal_best_fitness_ptr[active_indexes[blockIdx.x]]){
					personal_best_fitness_ptr[active_indexes[blockIdx.x]] =
						fitnesses_ptr[active_indexes[blockIdx.x]];
			}
		}
		
		index += blockDim.x;
	}
}

__global__ void calculate_new_fitnesses(int* D_ptr, float* positions, float* fitnesses,
								 		float* personal_best_fitness, int* active_indexes){
	int index = active_indexes[blockIdx.x];
	float fitness = sphere_function(positions, index, *D_ptr);
	fitnesses[index] = fitness;
	personal_best_fitness[index] = fitness;
}

__global__ void reset_velocity_new_individuals(int* D_ptr, float *velocities,
											   int* active_indexes){
	int index = threadIdx.x;
	while(index < *D_ptr){
		velocities[active_indexes[blockIdx.x] * *D_ptr + index] = 0;
		index += blockDim.x;
	}
}

float run(int N, int max_N, int D, float* positions, float* fitnesses,
		  float* gbest, float* gbest_fitness,
		  float* inertia_weight, float w_update_value, float* c1,
		  float c1_update_value, float* c2, float c2_update_value,
		  int n_iterations, int* active_indexes, int number_new_individuals,
		  float* new_positions, float** positions_dev,
		  float** fitnesses_dev, float** velocities_dev, float** personal_bests_dev,
		  float** personal_best_fitnesses_dev, float** gbest_dev,
		  float** gbest_fitnesses_dev, int get_gbest, int free_cuda_memory,
		  int run_init_population, float** previous_gbest_fitness,
		  FILE* arquivo_threads, FILE* arquivo_gbest, float delta_fitness_weight,
		  float time_spent_weight){
	clock_t begin = clock();
	struct timespec rawtime;

	clock_gettime(CLOCK_MONOTONIC_RAW, &rawtime);
	unsigned int time_nsec = rawtime.tv_nsec;

	int* active_indexes_dev;
	cudaMalloc((void**) &active_indexes_dev, N * sizeof(int));
	cudaMemcpy(active_indexes_dev, active_indexes, N * sizeof(int), cudaMemcpyHostToDevice);

	int* N_dev;
	cudaMalloc((void**) &N_dev, sizeof(int));
	cudaMemcpy(N_dev, &N, sizeof(int), cudaMemcpyHostToDevice);

	int* D_dev;
	cudaMalloc((void**) &D_dev, sizeof(int));
	cudaMemcpy(D_dev, &D, sizeof(int), cudaMemcpyHostToDevice);

	float* c1_dev;
	cudaMalloc((void**) &c1_dev, sizeof(float));
	cudaMemcpy(c1_dev, c1, sizeof(float), cudaMemcpyHostToDevice);

	float* c2_dev;
	cudaMalloc((void**) &c2_dev, sizeof(float));
	cudaMemcpy(c2_dev, c2, sizeof(float), cudaMemcpyHostToDevice);

	float* inertia_weight_dev;
	cudaMalloc((void**) &inertia_weight_dev, sizeof(float));
	cudaMemcpy(inertia_weight_dev, inertia_weight, sizeof(float), cudaMemcpyHostToDevice);

	unsigned int* time_nsec_dev;
	cudaMalloc((void**) &time_nsec_dev, sizeof(unsigned int));
	cudaMemcpy(time_nsec_dev, &time_nsec, sizeof(unsigned int), cudaMemcpyHostToDevice);

	int* number_new_individuals_dev;
	cudaMalloc((void**) &number_new_individuals_dev, sizeof(int));
	cudaMemcpy(number_new_individuals_dev, &number_new_individuals, sizeof(int), cudaMemcpyHostToDevice);

	int allocated_threads_get_gbest = N < MAX_THREAD_PER_BLOCK ? N : MAX_THREAD_PER_BLOCK;
	int allocated_threads_dimensions = D < MAX_THREAD_PER_BLOCK ? D : MAX_THREAD_PER_BLOCK;
	if(run_init_population == 1){
		cudaMalloc((void**) positions_dev, max_N * D * sizeof(float));
		cudaMalloc((void**) fitnesses_dev, max_N * sizeof(float));
		cudaMalloc((void**) velocities_dev, max_N * D * sizeof(float));
		cudaMalloc((void**) personal_bests_dev, max_N * D * sizeof(float));
		cudaMalloc((void**) personal_best_fitnesses_dev, max_N * sizeof(float));
		cudaMalloc((void**) gbest_dev, D * sizeof(float));
		cudaMalloc((void**) gbest_fitnesses_dev, sizeof(float));
		init_population<<<N, allocated_threads_dimensions>>>(*positions_dev, *velocities_dev,
			*fitnesses_dev, *personal_bests_dev, *personal_best_fitnesses_dev, time_nsec_dev,
			*gbest_dev, *gbest_fitnesses_dev, D_dev, active_indexes_dev);
		get_global_best<<<1, allocated_threads_get_gbest, N * sizeof(int)>>>(*positions_dev,
			*fitnesses_dev, *gbest_dev, *gbest_fitnesses_dev, D_dev, N_dev, active_indexes_dev);
	}else{
		for(int j = 0; j < number_new_individuals; j++){
			int index_new_element = active_indexes[j];
			cudaMemcpy((*positions_dev) + (index_new_element * D),
				new_positions + (j * D), D * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy((*personal_bests_dev) + (index_new_element * D),
				new_positions + (j * D), D * sizeof(float), cudaMemcpyHostToDevice);
		}
		calculate_new_fitnesses<<<number_new_individuals, 1>>>(D_dev,
			*positions_dev, *fitnesses_dev, *personal_best_fitnesses_dev,
			active_indexes_dev);
		reset_velocity_new_individuals<<<number_new_individuals, allocated_threads_dimensions>>>(D_dev,
			*velocities_dev, active_indexes_dev);
		get_global_best<<<1, allocated_threads_get_gbest, N * sizeof(int)>>>(*positions_dev,
			*fitnesses_dev, *gbest_dev, *gbest_fitnesses_dev, D_dev, N_dev, active_indexes_dev);
	}

	/*cudaMemcpy(positions, *positions_dev, max_N * D * sizeof(float),
				   cudaMemcpyDeviceToHost);
	cudaMemcpy(fitnesses, *fitnesses_dev, max_N * sizeof(float),
		   	   cudaMemcpyDeviceToHost);
	for(int i = 0; i < N; i++){	
		for(int j = 0; j < D; j++){
			printf("%f ", positions[active_indexes[i] * D + j]);
		}
		printf("= %f\n", fitnesses[active_indexes[i]]);
	}
	cudaMemcpy(gbest, *gbest_dev, D * sizeof(float),
		   	   cudaMemcpyDeviceToHost);
	cudaMemcpy(gbest_fitness, *gbest_fitnesses_dev, sizeof(float),
		   	   cudaMemcpyDeviceToHost);
	printf("GBEST\n");
	for(int i = 0; i < D; i++){
		printf("%f ", *(gbest+i));
	}
	printf("%f\n", *gbest_fitness);
	printf("=================================\n");*/
	for(int i = 0; i < n_iterations; i++){
		clock_gettime(CLOCK_MONOTONIC_RAW, &rawtime);
		time_nsec = rawtime.tv_nsec;
		cudaMemcpy(time_nsec_dev, &time_nsec, sizeof(unsigned int), cudaMemcpyHostToDevice);
		update_particle<<<N, allocated_threads_dimensions>>>(*positions_dev, *velocities_dev,
			*fitnesses_dev, *personal_bests_dev, *personal_best_fitnesses_dev, time_nsec_dev, *gbest_dev,
			inertia_weight_dev, c1_dev, c2_dev, D_dev, active_indexes_dev);
		get_global_best<<<1, allocated_threads_get_gbest, N * sizeof(int)>>>(*positions_dev, *fitnesses_dev,
			*gbest_dev, *gbest_fitnesses_dev, D_dev, N_dev, active_indexes_dev);
		
		*inertia_weight += w_update_value;
		*c1 += c1_update_value;
		*c2 += c2_update_value;
		cudaMemcpy(inertia_weight_dev, inertia_weight, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(c1_dev, c1, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(c2_dev, c2, sizeof(float), cudaMemcpyHostToDevice);
		
		/*cudaMemcpy(positions, *positions_dev, max_N * D * sizeof(float),
				   cudaMemcpyDeviceToHost);
		cudaMemcpy(fitnesses, *fitnesses_dev, max_N * sizeof(float),
			   	   cudaMemcpyDeviceToHost);
		for(int k = 0; k < N; k++){		
			for(int j = 0; j < D; j++){
				printf("%f ", positions[active_indexes[k] * D + j]);
			}
			printf("= %f\n", fitnesses[active_indexes[k]]);
		}
		printf("\n")
		cudaMemcpy(gbest_fitness, *gbest_fitnesses_dev, sizeof(float),
			   	   cudaMemcpyDeviceToHost);
		printf("%f\n", *gbest_fitness);
		printf("=================================\n");*/
	}

	cudaMemcpy(positions, *positions_dev, max_N * D * sizeof(float),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(fitnesses, *fitnesses_dev, max_N * sizeof(float),
		   	   cudaMemcpyDeviceToHost);
	cudaMemcpy(gbest_fitness, *gbest_fitnesses_dev, sizeof(float),
		   	   cudaMemcpyDeviceToHost);
	/*for(int i = 0; i < N; i++){		
		for(int j = 0; j < D; j++){
			printf("%f ", positions[i * D + j]);
		}
		printf("= %f\n", fitnesses[i]);
	}*/
	if(get_gbest == 1){
		cudaMemcpy(gbest, *gbest_dev, D * sizeof(float),
		   	   cudaMemcpyDeviceToHost);
		printf("GBEST\n");
		for(int i = 0; i < D; i++){
			printf("%f ", *(gbest+i));
		}
		printf("%f\n", *gbest_fitness);
		printf("=================================\n");
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	cudaFree(active_indexes_dev);
	cudaFree(N_dev);
	cudaFree(D_dev);
	cudaFree(c1_dev);
	cudaFree(c2_dev);
	cudaFree(inertia_weight_dev);
	cudaFree(time_nsec_dev);
	cudaFree(number_new_individuals_dev);
	float delta_gbest_fitness;
	if(*previous_gbest_fitness == NULL){
		*previous_gbest_fitness = (float*) malloc(sizeof(float));
		*(*previous_gbest_fitness) = *gbest_fitness;
		delta_gbest_fitness = *gbest_fitness;
	}else{
		delta_gbest_fitness = *(*previous_gbest_fitness) - *gbest_fitness;
		*(*previous_gbest_fitness) = *gbest_fitness;
	}

	fprintf(arquivo_threads, "%d\n", N);
	fprintf(arquivo_gbest, "%f\n", *gbest_fitness);

	if(free_cuda_memory == 1){
		cudaFree(positions_dev);
		cudaFree(velocities_dev);
		cudaFree(fitnesses_dev);
		cudaFree(personal_bests_dev);
		cudaFree(personal_best_fitnesses_dev);
		cudaFree(gbest_dev);
		cudaFree(gbest_fitnesses_dev);
	}
	return ((delta_fitness_weight / (abs(delta_gbest_fitness) + 1)) + (time_spent_weight * time_spent)) / (delta_fitness_weight + time_spent_weight);
}

/*int main(){
	float* posit;
    cudaMalloc((void**) &posit, 10 * sizeof(float));
}*/

/*
Algoritmo:

- Criar vetores (como ponteiros, que poderao ter tamanhos diferentes) serao
recipientes para os resultados de cada iteracao externa.
- Alocar espaco na gpu p inicializar populacao inicial.
- Salvar resultado nos vetores do inicio.
- Passar para run esses vetores, os parametros, etc.
- Dentro de run, alocar espaco interno baseado no tamanho da populacao e numero
de dimensoes e copiar os valores da populacao, etc.
- Calcular tudo.
- Copiar de volta p os vetores recipientes.

- Criar funcao que recebe populacao, vetor com historico de fitness (soma) de
cada individuo, tamanho da proxima populacao e, se for maior que a atual,
sorteia (roleta) os que irao gerar outro individuo, e se for menor, seleciona
alguem p apagar.
- Individuos acumulam fitnesses com somas, mas tem que ter um grau de
esquecimento (sera).
- Passar para funcao run a populacao apos modificacao.
*/
/*int main(){

	for(int i=0; i<30; i++){
		int initial_pop_size = 100;
		int D = 100;

		//Vetores que vao guardar os valores computados apos cada execucao com uma nova populacao.
		float* positions = (float*) malloc(initial_pop_size * D * sizeof(float));
		float* velocities = (float*) malloc(initial_pop_size * D * sizeof(float));
		float* personal_bests = (float*) malloc(initial_pop_size * D * sizeof(float));
		float* fitnesses = (float*) malloc(initial_pop_size * sizeof(float));
		float* personal_bests_fitnesses = (float*) malloc(initial_pop_size * sizeof(float));
		float* gbest = (float*) malloc(initial_pop_size * sizeof(float));
		float* gbest_fitness = (float*) malloc(sizeof(float));

		float* inertia_weight = (float*) malloc(sizeof(float));
		*inertia_weight = INIT_W;
		float* c1 = (float*) malloc(sizeof(float));
		*c1 = INIT_C1;
		float* c2 = (float*) malloc(sizeof(float));
		*c2 = INIT_C2;

		float c1_update_value = (FINAL_C1 - INIT_C1) / MAX_ITERATIONS;
		float c2_update_value = (FINAL_C2 - INIT_C2) / MAX_ITERATIONS;
		float w_update_value = (FINAL_W - INIT_W) / MAX_ITERATIONS;

		run(initial_pop_size, D, positions, velocities,
						   personal_bests, fitnesses, personal_bests_fitnesses, gbest,
						   gbest_fitness, inertia_weight, w_update_value, c1,
						   c1_update_value, c2, c2_update_value, 1, MAX_ITERATIONS);*/

		/*printf("%f\n", run(initial_pop_size, D, positions, velocities,
						   personal_bests, fitnesses, personal_bests_fitnesses, gbest,
						   gbest_fitness, inertia_weight, w_update_value, c1,
						   c1_update_value, c2, c2_update_value, 1, MAX_ITERATIONS/20)
		for(int i = 0; i < 0; i++){
			printf("%f\n", run(initial_pop_size, D, positions, velocities,
						   personal_bests, fitnesses, personal_bests_fitnesses, gbest,
						   gbest_fitness, inertia_weight, w_update_value, c1,
						   c1_update_value, c2, c2_update_value, 0, MAX_ITERATIONS/20));
		}*/
		/*free(positions);
	    free(velocities);
	    free(personal_bests);
	    free(fitnesses);
	    free(personal_bests_fitnesses);
	    free(gbest);
	    free(gbest_fitness);
	    free(inertia_weight);
	    free(c1);
	    free(c2);
	}
}*/

/*
int main(){
	int D_values_length = 4;
	int N_values_length = 2;
	int repetitions = 10;
	int D_values[] = {800, 1600, 3200, 6400};
	int N_values[] = {1280, 2560};
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

				float *gbest_fitnesses_dev_ptr;
				cudaMalloc((void**) &gbest_fitnesses_dev_ptr, sizeof(float));	

				clock_gettime(CLOCK_MONOTONIC_RAW, &rawtime);
				unsigned int time_nsec = rawtime.tv_nsec;

				int allocated_threads_get_gbest = N < MAX_THREAD_PER_BLOCK ? N : MAX_THREAD_PER_BLOCK;
				int allocated_threads_dimensions = D < MAX_THREAD_PER_BLOCK ? D : MAX_THREAD_PER_BLOCK;
				init_population<<<N, allocated_threads_dimensions>>>(positions_ptr, velocities_ptr,
					fitnesses_ptr, personal_bests_ptr, personal_best_fitness_ptr, time_nsec,
					gbest_dev_ptr,gbest_fitnesses_dev_ptr, D, N);
				get_global_best<<<1, allocated_threads_get_gbest, N * sizeof(int)>>>(positions_ptr, fitnesses_ptr, gbest_dev_ptr, gbest_fitnesses_dev_ptr, D, N);

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
						gbest_dev_ptr, gbest_fitnesses_dev_ptr, D, N);
					inertia_weight += (FINAL_W - INIT_W) / MAX_ITERATIONS;
					c1 += (FINAL_C1 - INIT_C1) / MAX_ITERATIONS;
					c2 += (FINAL_C2 - INIT_C2) / MAX_ITERATIONS;
				}

				cudaMemcpy(&positions, positions_ptr, N * D * sizeof(float),
						   cudaMemcpyDeviceToHost);
				cudaMemcpy(&fitnesses, fitnesses_ptr, N * sizeof(float),
					   	   cudaMemcpyDeviceToHost);
				cudaMemcpy(&gbest_fitness, gbest_fitnesses_dev_ptr, sizeof(float),
						   cudaMemcpyDeviceToHost);
				cudaMemcpy(&gbest, gbest_dev_ptr, D * sizeof(float),
					   	   cudaMemcpyDeviceToHost);
					
				cudaFree(positions_ptr);
				cudaFree(velocities_ptr);
				cudaFree(fitnesses_ptr);
				cudaFree(personal_bests_ptr);
				cudaFree(personal_best_fitness_ptr);
				cudaFree(gbest_dev_ptr);
				cudaFree(gbest_fitnesses_dev_ptr);
				clock_t end = clock();
				double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
				avg_fitness_result += -gbest_fitness;
				avg_time_result += time_spent;
				avg_general_score += sqrt(pow(gbest_fitness, 2) + pow(time_spent, 2));

			}
			printf("%d\t%d\t%lf\t%f\t%lf\n", D, N, avg_time_result / repetitions,
																																				 / repetitions, avg_general_score);
		}
		printf("\n");
	}	
}
*/