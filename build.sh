g++ -c stack.c &&
g++ -c linked_list.c &&
nvcc -c pso_cuda.cu -gencode arch=compute_20,code=sm_21 &&
g++ -c simulated_annealing_hyperheuristic.c &&
g++ -L/usr/local/cuda/lib64 -o main stack.o linked_list.o pso_cuda.o simulated_annealing_hyperheuristic.o -lcuda -lcudart -lm -lstdc++
