g++ -c stack.c &&
g++ -c linked_list.c &&
g++ -c simulated_annealing_hyperheuristic.c &&
nvcc -c pso_cuda.cu -gencode arch=compute_20,code=sm_21 &&
g++ -L/usr/local/cuda/lib64 -o main stack.o linked_list.o simulated_annealing_hyperheuristic.o pso_cuda.o -lcuda -lcudart -lm -lstdc++
