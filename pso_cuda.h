#ifndef __PSO_CUDA_H__
#define __PSO_CUDA_H__

float run(int N, int max_N, int D, float* positions, float* fitnesses,
		  float* gbest, float* gbest_fitness,
		  float* inertia_weight, float w_update_value, float* c1,
		  float c1_update_value, float* c2, float c2_update_value,
		  int n_iterations, int* active_indexes, int number_new_individuals,
		  float* new_positions, float** positions_dev,
		  float** fitnesses_dev, float** velocities_dev, float** personal_bests_dev,
		  float** personal_best_fitnesses_dev, float** gbest_dev,
		  float** gbest_fitnesses_dev, int get_gbest, int free_cuda_memory,
		  int run_init_population);
#endif