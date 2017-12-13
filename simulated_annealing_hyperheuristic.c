#define POS_MAX 100
#define INIT_W 0.9
#define FINAL_W 0.4
#define INIT_C1 2.0
#define FINAL_C1 2.0
#define INIT_C2 2.0
#define FINAL_C2 2.0
#define VEL_CLAMPING_FACTOR 0.8
#define MAX_INDIVIDUALS 2048
//Este código implementa o algoritmo Simulated Annealing.
#include <stdio.h>
//A biblioteca math.h possui a funcao "pow", que calcula potências.
#include <math.h>
//Estes includes sao necessários para gerar números aleatórios.
#include <time.h>
#include <stdlib.h>

#include "pso_cuda.h"
#include "linked_list.h"
#include "stack.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//Using Box-Muller Transform
double gauss(double mu, double sigma)
{
    static const double two_pi = 2.0*M_PI;
    double u1 = rand() * (1.0 / RAND_MAX);
    double u2 = rand() * (1.0 / RAND_MAX);
    double z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    return z0 * sigma + mu;
}

int* convert_linked_list_vector(struct linked_list_element* linked_list_e,
                                int num_elements){
    int* vector = (int*) malloc(num_elements * sizeof(int));
    struct linked_list_element* current_element = linked_list_e;
    int i;
    for(i = 0; i < num_elements; i++){
        *(vector + i) = (*current_element).content;
        current_element = (*current_element).next_linked_list_element;
    }
    return vector;
}

void copiar_vetor(float origem[], float destino[], int num_dimensoes){
    int i;
    for(i = 0; i < num_dimensoes; i++){
        destino[i] = origem[i];
    }
}

/*number_new_individuals must be positive (generate new individuals) or negative
  (kill already existent individuals).*/
/*Testing with the simplest case: the fitness itself defines the chance of
an individual being selected or not to be removed ot to create a new one.*/
float* modify_population(int number_new_individuals, float* population,
                         float* fitnesses, int current_pop_size,
                         struct linked_list_element** active_indexes,
                         struct stack_element** free_indexes, int dim,
                         float q, float v){
    float* new_positions = NULL;
    if(number_new_individuals > 0){
        int j, i, r, chosen_index;
        float fitness_sum = 0;
        struct linked_list_element* current_element = *active_indexes;
        for(j = 0; j < current_pop_size; j++){
            fitness_sum += 1.0 / (pow(abs(*(fitnesses + (*current_element).content)), q) + 1);
            current_element = (*current_element).next_linked_list_element;
        }
        new_positions = (float*) malloc(number_new_individuals * dim * sizeof(float));
        float roullete_number, partial_roullete_sum, perturbation;
        for(i = 0; i < number_new_individuals; i++){
            //printf("(*(*active_indexes)).content %d\n", (*(*active_indexes)).content);
            //printf("(*((*(*active_indexes)).next_linked_list_element)).content %d\n", (*((*(*active_indexes)).next_linked_list_element)).content);
            //printf("(*((*((*(*active_indexes)).next_linked_list_element)).next_linked_list_element)).content %d\n", (*((*((*(*active_indexes)).next_linked_list_element)).next_linked_list_element)).content);
            current_element = *active_indexes;
            for(j = 0; j < i; j++){
                current_element = (*current_element).next_linked_list_element;
            }
            r = rand();
            roullete_number = (float)(r % 10000) / 10000;
            partial_roullete_sum = 0;
            for(j = 0; j < current_pop_size; j++){
                partial_roullete_sum += (1.0 / pow(abs(*(fitnesses + (*current_element).content)), q) + 1.0) / fitness_sum;
                //printf("%f %f %f %f\n", roullete_number, partial_roullete_sum, *(fitnesses + (*current_element).content), fitness_sum);
                if(roullete_number < partial_roullete_sum){
                    chosen_index = j;
                    break;
                }
                current_element = (*current_element).next_linked_list_element;
            }
            for(j = 0; j < dim; j++){
                r = gauss(0, 1);
                perturbation = r * v;
                *(new_positions + (i * dim) + j) =
                    *(population + (chosen_index * dim) + j) + perturbation;
            }
            //printf("(*(*free_indexes)).content %d\n", (*(*free_indexes)).content);
            struct stack_element* new_index = pop(free_indexes);
            struct linked_list_element* ll_new_index = (struct linked_list_element*) malloc(sizeof(struct linked_list_element));
            (*ll_new_index).content = (*new_index).content;
            (*ll_new_index).next_linked_list_element = NULL;
            //printf("(*(*active_indexes)).content %d\n", (*(*active_indexes)).content);
            //printf("(*((*(*active_indexes)).next_linked_list_element)).content %d\n", (*((*(*active_indexes)).next_linked_list_element)).content );
            //printf("(*((*((*(*active_indexes)).next_linked_list_element)).next_linked_list_element)).content %d\n", (*((*((*(*active_indexes)).next_linked_list_element)).next_linked_list_element)).content);
            insert(active_indexes, ll_new_index, 0);
            //printf("(*(*active_indexes)).content %d\n", (*(*active_indexes)).content);
            //printf("(*((*(*active_indexes)).next_linked_list_element)).content %d\n", (*((*(*active_indexes)).next_linked_list_element)).content );
            //printf("(*((*((*(*active_indexes)).next_linked_list_element)).next_linked_list_element)).content %d\n", (*((*((*(*active_indexes)).next_linked_list_element)).next_linked_list_element)).content);
            //COMO LIBERAR ESTA MERDA SEM DAR PAU???
            //free(new_index);
        }
    }else if(number_new_individuals < 0){
        int i, j, r, chosen_index;
        float fitness_sum = 0;
        struct linked_list_element* current_element = *active_indexes;
        for(j = 0; j < current_pop_size; j++){
            fitness_sum += pow(abs(*(fitnesses + (*current_element).content)), q);
            current_element = (*current_element).next_linked_list_element;
        }
        float roullete_number, partial_roullete_sum;
        for(i = 0; i < -number_new_individuals; i++){
            r = rand();
            roullete_number = (float)(r % 10000) / 10000;
            partial_roullete_sum = 0;
            chosen_index = -1;
            current_element = *active_indexes;
            for(j = 0; j < current_pop_size - i; j++){
                partial_roullete_sum += pow(abs(*(fitnesses + (*current_element).content)), q) / fitness_sum;
                if(roullete_number < partial_roullete_sum){
                    chosen_index = j;
                    break;
                }
                current_element = (*current_element).next_linked_list_element;
            }
            //printf("CHOSEN INDEX: %d\n", chosen_index);
            struct linked_list_element* removed = remove_e(active_indexes, chosen_index);
            struct stack_element* s_removed = (struct stack_element*) malloc(sizeof(struct stack_element));
            (*s_removed).content = (*removed).content;
            (*s_removed).next_stack_element = NULL;
            fitness_sum -= pow(abs(*(fitnesses + (*s_removed).content)), q);
            push(free_indexes, s_removed);
            //COMO LIBERAR ESTA MERDA SEM DAR PAU???
            //free(removed);
        }
    }
    return new_positions;
}

void rollback_active_free_indexes(struct linked_list_element** active_indexes,
                                  struct stack_element** free_indexes,
                                  int number_new_individuals){
    if(number_new_individuals > 0){
        for(int i = 0; i < number_new_individuals; i++){
            struct linked_list_element* removed = remove_e(active_indexes, 0);
            struct stack_element* s_removed = (struct stack_element*) malloc(sizeof(struct stack_element));
            (*s_removed).content = (*removed).content;
            (*s_removed).next_stack_element = NULL;
            push(free_indexes, s_removed);
        }
    }else if(number_new_individuals < 0){
        for(int i = 0; i < -number_new_individuals; i++){
            struct stack_element* new_index = pop(free_indexes);
            struct linked_list_element* ll_new_index = (struct linked_list_element*) malloc(sizeof(struct linked_list_element));
            (*ll_new_index).content = (*new_index).content;
            (*ll_new_index).next_linked_list_element = NULL;
            insert(active_indexes, ll_new_index, 0);
        }
    }
}

/*Essa funcao possui a funcionalidade de gerar uma solucao nova a partir da
anterior (recebida como entrada) e, se for o caso, substituir o vetor da
solucao corrente com a nova.*/
float gerar_e_analisar_vizinho(float* solucao, float qualidade_solucao_corrente,
                               float temperatura, int num_dimensoes, int D,
                               float* positions, float* fitnesses,
                               float* gbest, float* gbest_fitness, float* inertia_weight,
                               float w_update_value, float* c1, float c1_update_value, float* c2,
                               float c2_update_value, int run_init_population, int n_iterations,
                               float** positions_dev,
                               float** fitness_dev, float** velocities_dev, float** personal_bests_dev,
                               float** personal_best_fitness_dev, float** gbest_dev,
                               float** gbest_fitness_dev,
                               struct linked_list_element** active_indexes,
                               struct stack_element** free_indexes, float q,
                               float v, float** previous_gbest_fitness,
                               FILE* arquivo_threads, FILE* qualidade_solucoes,
                               FILE* arquivo_gbest, int free_cuda_memory, float delta_fitness_weight,
                               float time_spent_weight){
    float nova_solucao[num_dimensoes];
    int i, r;
    for(i = 0; i < num_dimensoes; i++){
        r = rand();
        nova_solucao[i] = solucao[i] + (gauss(0, 1) * temperatura);
        //I'm considering here only one dimension (number of individuals)
        if(nova_solucao[i] < 2){
            nova_solucao[i] = 2;
        }else if(nova_solucao[i] > MAX_INDIVIDUALS){
            nova_solucao[i] = MAX_INDIVIDUALS;
        }
    }
    printf("POP: %d\n", (int)roundf(nova_solucao[0]));
    int number_new_individuals = (int)roundf(nova_solucao[0]) - (int)roundf(solucao[0]); 
    float* new_positions = modify_population(number_new_individuals, positions,
                                             fitnesses, (int)solucao[0],
                                             active_indexes, free_indexes, D,
                                             q, v);
    int* active_indexes_vec = convert_linked_list_vector(*active_indexes, (int)roundf(nova_solucao[0]));
    float qualidade_solucao_nova = run((int)roundf(nova_solucao[0]),
        MAX_INDIVIDUALS, D, positions, fitnesses,
        gbest, gbest_fitness,
        inertia_weight, w_update_value, c1,
        c1_update_value, c2, c2_update_value,
        n_iterations, active_indexes_vec, number_new_individuals,
        new_positions, positions_dev,
        fitness_dev, velocities_dev, personal_bests_dev,
        personal_best_fitness_dev, gbest_dev,
        gbest_fitness_dev, 0, free_cuda_memory, 0, previous_gbest_fitness,
        arquivo_threads, arquivo_gbest, delta_fitness_weight, time_spent_weight);
    printf("Qualidade: %f\n", qualidade_solucao_nova);
    fprintf(qualidade_solucoes, "%f\n", qualidade_solucao_nova);
    free(new_positions);

    if(qualidade_solucao_nova <= qualidade_solucao_corrente){
        copiar_vetor(nova_solucao, solucao, num_dimensoes);
        return qualidade_solucao_nova;
    }else{
        float delta_qualidade = qualidade_solucao_nova - qualidade_solucao_corrente;
        float probabilidade_escolha = exp(-delta_qualidade / temperatura);
        float sorteio = ((float)(rand() % 10000)) / 10000;
        if(sorteio < probabilidade_escolha){
            copiar_vetor(nova_solucao, solucao, num_dimensoes);
            return qualidade_solucao_nova;
        }else{
            rollback_active_free_indexes(active_indexes, free_indexes,
                number_new_individuals);
            return qualidade_solucao_corrente;
        }
    }
}

void inicializar_solucao(float* solucao, float min_value, float max_value,
                         int num_dimensoes){
    int i, r;
    for(i = 0; i < num_dimensoes; i++){
        r = rand();
        solucao[i] = ((float)(r % 10000) / 10000) * (max_value - min_value) + min_value;
    }
}

/*Executando o procedimento de busca durante um numero maximo de iteracoes para um dada temperatura fixa.*/
float executar_temperatura_fixa(float* solucao, float qualidade_solucao_corrente,
                               int num_dimensoes, int temperatura,
                               int maximo_iteracoes, int D,
                               float* positions, float* fitnesses,
                               float* gbest, float* gbest_fitness, float* inertia_weight,
                               float w_update_value, float* c1, float c1_update_value, float* c2,
                               float c2_update_value, int run_init_population, int n_iterations,
                               float** positions_dev,
                               float** fitness_dev, float** velocities_dev, float** personal_bests_dev,
                               float** personal_best_fitness_dev, float** gbest_dev,
                               float** gbest_fitness_dev, struct linked_list_element** active_indexes,
                               struct stack_element** free_indexes, float q,
                               float v, float** previous_gbest_fitness,
                               FILE* arquivo_threads, FILE* qualidade_solucoes,
                               FILE* arquivo_gbest, int free_cuda_memory,
                               float delta_fitness_weight, float time_spent_weight){
    int i;
    for(i = 0; i < maximo_iteracoes; i++){
        qualidade_solucao_corrente = gerar_e_analisar_vizinho(solucao, qualidade_solucao_corrente,
            temperatura, num_dimensoes, D,
            positions, fitnesses,
            gbest, gbest_fitness, inertia_weight,
            w_update_value, c1, c1_update_value, c2,
            c2_update_value, run_init_population, n_iterations,
            positions_dev, fitness_dev, velocities_dev, personal_bests_dev,
            personal_best_fitness_dev, gbest_dev, gbest_fitness_dev,
            active_indexes, free_indexes, q, v, previous_gbest_fitness,
            arquivo_threads, qualidade_solucoes, arquivo_gbest, free_cuda_memory,
            delta_fitness_weight, time_spent_weight);
    }
    return qualidade_solucao_corrente;
}

//Atualizando melhor solucao
float atualizar_melhor_solucao(float* solucao, float qualidade_solucao_corrente,
                               float melhor_solucao[], float qualidade_melhor_solucao,
                               int num_dimensoes){
    if(qualidade_solucao_corrente < qualidade_melhor_solucao){
        copiar_vetor(solucao, melhor_solucao, num_dimensoes);
        return qualidade_solucao_corrente;
    }
    return qualidade_melhor_solucao;
}

//Atualizando temperatura
float calcular_nova_temperatura(float temperatura, float alfa){
    return temperatura * alfa;
}

int main(){

    float temperatura = 1000;
    int maximo_iteracoes = 10;
    float alfa = 0.6;
    int temperatura_minima = 1;
    int D = 1000;
    int iteracoes_metaheuristica = 100;
    int q = 1;
    int v = 5;
    float delta_fitness_weight = 1;
    float time_spent_weight = 1;

    FILE* arquivo_threads = fopen("threads.txt", "w");
    FILE* qualidade_solucoes = fopen("qualidades.txt", "w");
    FILE* arquivo_gbest = fopen("gbest.txt", "w");

    srand(time(NULL));
    int num_dimensoes = 1;

    //Parâmetros

    float solucao[num_dimensoes];
    float qualidade_solucao_corrente;
    float melhor_solucao[num_dimensoes];
    float qualidade_melhor_solucao;

    inicializar_solucao(solucao, 0, temperatura, num_dimensoes);

    int initial_pop_size = roundf(solucao[0]);
    printf("%d\n", initial_pop_size);
    //Vetores que vao guardar os valores computados apos cada execucao com uma nova populacao.
    float* positions = (float*) malloc(MAX_INDIVIDUALS * D * sizeof(float));
    float* fitnesses = (float*) malloc(MAX_INDIVIDUALS * sizeof(float));
    float* gbest = (float*) malloc(D * sizeof(float));
    float* gbest_fitness = (float*) malloc(sizeof(float));
    float** previous_gbest_fitness = (float**) malloc(sizeof(float*));

    float* positions_dev;
    float* fitness_dev;
    float* velocities_dev;
    float* personal_bests_dev;
    float* personal_best_fitness_dev;
    float* gbest_dev;
    float* gbest_fitness_dev;

    struct linked_list_element* active_indexes = NULL;
    struct stack_element* free_indexes = NULL;
    struct linked_list_element* list_active_index = (struct linked_list_element*) malloc(initial_pop_size * sizeof(struct linked_list_element));
    struct stack_element* list_free_elem = (struct stack_element*) malloc((MAX_INDIVIDUALS - initial_pop_size) * sizeof(struct stack_element));
    int i, j = 0;
    for(i = 0; i < MAX_INDIVIDUALS; i++){
        if(i < initial_pop_size){
            (*(list_active_index + i)).content = i;
            insert(&active_indexes, list_active_index + i, i);
        }else{
            (*(list_free_elem + i - initial_pop_size)).content = MAX_INDIVIDUALS - j - 1;
            push(&free_indexes, list_free_elem + i - initial_pop_size);
            j++;
        }
    }
    /*printf("LINKED LIST\n");
    struct linked_list_element* active_indexes_current = active_indexes;
    for(i = 0; i < initial_pop_size; i++){
        printf("%d ", (*active_indexes_current).content);
        active_indexes_current = (*active_indexes_current).next_linked_list_element;
    }
    printf("\n");
    struct stack_element* current_element = free_indexes;
    for(i = 0; i < MAX_INDIVIDUALS - initial_pop_size; i++){
        printf("%d ", (*current_element).content);
        current_element = (*current_element).next_stack_element;
    }
    printf("\n");*/
    
    float* inertia_weight = (float*) malloc(sizeof(float));
    *inertia_weight = INIT_W;
    float* c1 = (float*) malloc(sizeof(float));
    *c1 = INIT_C1;
    float* c2 = (float*) malloc(sizeof(float));
    *c2 = INIT_C2;

    int max_iterations = (int)((((log(temperatura_minima / temperatura)) / log(alfa)) + 1) * iteracoes_metaheuristica * maximo_iteracoes);
    float c1_update_value = (float)(FINAL_C1 - INIT_C1) / max_iterations;
    float c2_update_value = (float)(FINAL_C2 - INIT_C2) / max_iterations;
    float w_update_value = (float)(FINAL_W - INIT_W) / max_iterations;
    printf("%d %f %f %f\n", max_iterations, c1_update_value, c2_update_value, w_update_value);

    int* active_indexes_vec = convert_linked_list_vector(active_indexes, initial_pop_size);
    qualidade_solucao_corrente = run((int)roundf(solucao[0]),
                                     MAX_INDIVIDUALS, D, positions, fitnesses,
                                     gbest, gbest_fitness,
                                     inertia_weight, w_update_value, c1,
                                     c1_update_value, c2, c2_update_value,
                                     iteracoes_metaheuristica, active_indexes_vec, 0,
                                     NULL, &positions_dev,
                                     &fitness_dev, &velocities_dev, &personal_bests_dev,
                                     &personal_best_fitness_dev, &gbest_dev,
                                     &gbest_fitness_dev, 0, 0, 1, previous_gbest_fitness,
                                     arquivo_threads, arquivo_gbest, delta_fitness_weight,
                                     time_spent_weight);
    copiar_vetor(solucao, melhor_solucao, num_dimensoes);
    qualidade_melhor_solucao = qualidade_solucao_corrente;
    printf("Solucao inicial:\n");
    for(i = 0; i < num_dimensoes; i++){
        printf("%f ", melhor_solucao[i]);
    }
    printf("\n");
    float proxima_temperatura;
    while(temperatura > temperatura_minima){
        proxima_temperatura = calcular_nova_temperatura(temperatura, alfa);
        if(proxima_temperatura > temperatura_minima){
            qualidade_solucao_corrente = executar_temperatura_fixa(solucao,
                qualidade_solucao_corrente,
                num_dimensoes, temperatura, maximo_iteracoes, D,
                positions, fitnesses,
                gbest, gbest_fitness, inertia_weight,
                w_update_value, c1, c1_update_value, c2,
                c2_update_value, 0, iteracoes_metaheuristica, &positions_dev,
                &fitness_dev, &velocities_dev, &personal_bests_dev,
                &personal_best_fitness_dev, &gbest_dev,
                &gbest_fitness_dev, &active_indexes, &free_indexes, q, v,
                previous_gbest_fitness, arquivo_threads, qualidade_solucoes,
                arquivo_gbest, 0, delta_fitness_weight, time_spent_weight);
        }else{
            qualidade_solucao_corrente = executar_temperatura_fixa(solucao,
                qualidade_solucao_corrente,
                num_dimensoes, temperatura, maximo_iteracoes, D,
                positions, fitnesses,
                gbest, gbest_fitness, inertia_weight,
                w_update_value, c1, c1_update_value, c2,
                c2_update_value, 0, iteracoes_metaheuristica, &positions_dev,
                &fitness_dev, &velocities_dev, &personal_bests_dev,
                &personal_best_fitness_dev, &gbest_dev,
                &gbest_fitness_dev, &active_indexes, &free_indexes, 1, 5,
                previous_gbest_fitness, arquivo_threads, qualidade_solucoes,
                arquivo_gbest, 1, delta_fitness_weight, time_spent_weight);
        }
        temperatura = calcular_nova_temperatura(temperatura, alfa);
        qualidade_melhor_solucao = atualizar_melhor_solucao(solucao, qualidade_solucao_corrente,
                                                        melhor_solucao, qualidade_melhor_solucao,
                                                        num_dimensoes);
        for(i = 0; i < num_dimensoes; i++){
            printf("%f ", melhor_solucao[i]);
        }
        printf(" = %f\n", qualidade_melhor_solucao);
        printf("%f %f %f\n", *c1, *c2, *inertia_weight);
    }
    printf("Solucao final:\n");
    for(i = 0; i < num_dimensoes; i++){
        printf("%f ", melhor_solucao[i]);
    }
    printf("\n");
    free(positions);
    free(fitnesses);
    free(gbest);
    free(gbest_fitness);
    free(previous_gbest_fitness);
    free(inertia_weight);
    free(c1);
    free(c2);
    fclose(arquivo_threads);
    fclose(qualidade_solucoes);
    fclose(arquivo_gbest);
  return 0;
}