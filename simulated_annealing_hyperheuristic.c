#define POS_MAX 100
#define INIT_W 0.9
#define FINAL_W 0.4
#define INIT_C1 2
#define FINAL_C1 2
#define INIT_C2 2
#define FINAL_C2 2
#define VEL_CLAMPING_FACTOR 0.8
#define MAX_ITERATIONS 5000
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

int* convert_linked_list_vector(struct linked_list_element* linked_list_element,
                                int num_elements){
    int* vector = (int*) malloc(num_elements * sizeof(int));
    struct linked_list_element* current_element = linked_list_element;
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
                         struct linked_list_element* active_indexes,
                         struct stack_element* free_indexes, int dim,
                         float q, float v){
    float* new_positions = NULL;
    int j;
    float fitness_sum = 0;
    for(j = 0; j < current_pop_size; j++){
        fitness_sum += pow(*(fitnesses + j), q);
    }
    if(number_new_individuals > 0){
        new_positions = (float*) malloc(number_new_individuals * dim * sizeof(float));
        int i, r, chosen_index;
        float roullete_number, partial_roullete_sum, perturbation;
        for(i = 0; i < number_new_individuals; i++){
            r = rand();
            roullete_number = (float)(r % 10000) / 10000;
            partial_roullete_sum = 0;
            for(j = 0; j < current_pop_size; j++){
                partial_roullete_sum += pow(*(fitnesses + j), q) / fitness_sum;
                if(roullete_number < partial_roullete_sum){
                    chosen_index = j;
                    break;
                }
            }
            for(j = 0; j < dim; j++){
                r = rand();
                perturbation = (((float)(r % 10000) / 10000) * v) - (v / 2);
                *((new_positions + (i * dim)) + j) =
                    *((population + (chosen_index * dim)) + j) + perturbation;
            }
            struct stack_element* new_index = pop(&free_indexes);
            struct linked_list_element ll_new_index;
            ll_new_index.content = (*new_index).content;
            ll_new_index.next_linked_list_element = NULL;
            insert(&active_indexes, &ll_new_index, 0);
        }
    }else if(number_new_individuals < 0){
        int i, j, r, chosen_index;
        float roullete_number, partial_roullete_sum;
        for(i = 0; i < -number_new_individuals; i++){
            r = rand();
            roullete_number = (float)(r % 10000) / 10000;
            partial_roullete_sum = 0;
            for(j = 0; j < current_pop_size; j++){
                partial_roullete_sum += 1.0 - (pow(*(fitnesses + j), q) / fitness_sum);
                if(roullete_number < partial_roullete_sum){
                    chosen_index = j;
                    break;
                }
            }
            struct linked_list_element* removed = remove_e(&active_indexes, chosen_index);
            struct stack_element s_removed;
            s_removed.content = (*removed).content;
            s_removed.next_stack_element = NULL;
            push(&free_indexes, &s_removed);
        }
    }
    return new_positions;
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
                               struct linked_list_element* active_indexes,
                               struct stack_element* free_indexes, float q,
                               float v){
    float nova_solucao[num_dimensoes];
    int i, r;
    for(i = 0; i < num_dimensoes; i++){
        r = rand();
        nova_solucao[i] = solucao[i] + (((float)(r % 10000) / 10000) * temperatura) - (temperatura / 2);
        printf("===%f\n", nova_solucao[i]);
        //I'm considering here only one dimension (number of individuals)
        if(nova_solucao[i] < 1){
            nova_solucao[i] = 1;
        }else if(nova_solucao[i] > MAX_INDIVIDUALS){
            nova_solucao[i] = MAX_INDIVIDUALS;
        }


        //REMEMBER TO CHANGE THIS - ONLY FOR TESTING
        nova_solucao[i] = 7;
    }
    printf("POP: %d\n", (int)roundf(nova_solucao[0]));

    int number_new_individuals = nova_solucao[0] - solucao[0];
    float* new_positions = modify_population(number_new_individuals, positions,
                                             fitnesses, (int)solucao[0],
                                             active_indexes, free_indexes, D,
                                             q, v);
    printf("####################\n");
    int* active_indexes_vec = convert_linked_list_vector(active_indexes, (int)roundf(nova_solucao[0]));
    for(i = 0; i < (int)roundf(nova_solucao[0]); i++){
        printf("%d ", *(active_indexes_vec+i));
    }
    printf("\n");
    float qualidade_solucao_nova = run((int)roundf(nova_solucao[0]),
        MAX_INDIVIDUALS, D, positions, fitnesses,
        gbest, gbest_fitness,
        inertia_weight, w_update_value, c1,
        c1_update_value, c2, c2_update_value,
        n_iterations, active_indexes_vec, number_new_individuals,
        new_positions, positions_dev,
        fitness_dev, velocities_dev, personal_bests_dev,
        personal_best_fitness_dev, gbest_dev,
        gbest_fitness_dev, 0, 0, 0);
    printf("Qualidade: %f\n", qualidade_solucao_nova);
    if(qualidade_solucao_nova < qualidade_solucao_corrente){
          copiar_vetor(nova_solucao, solucao, num_dimensoes);
          return qualidade_solucao_nova;
    }else{
          float delta_qualidade = qualidade_solucao_nova - qualidade_solucao_corrente;
          float probabilidade_escolha = exp(-delta_qualidade / temperatura);
          float sorteio = rand() / 10000;
          if(sorteio < probabilidade_escolha){
              copiar_vetor(nova_solucao, solucao, num_dimensoes);
              return qualidade_solucao_nova;
          }else{
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
        solucao[i] = 6;
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
                               float** gbest_fitness_dev, struct linked_list_element* active_indexes,
                               struct stack_element* free_indexes, float q,
                               float v){
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
            active_indexes, free_indexes, q, v);
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

    srand(time(NULL));
    int num_dimensoes = 1;

    //Parâmetros
    float temperatura = 1000;
    int maximo_iteracoes = 10;
    float alfa = 0.8;

    float solucao[num_dimensoes];
    float qualidade_solucao_corrente;
    float melhor_solucao[num_dimensoes];
    float qualidade_melhor_solucao;

    inicializar_solucao(solucao, 0, temperatura, num_dimensoes);

    int D = 2;
    int initial_pop_size = roundf(solucao[0]);
    printf("%d\n", initial_pop_size);
    //Vetores que vao guardar os valores computados apos cada execucao com uma nova populacao.
    float* positions = (float*) malloc(initial_pop_size * D * sizeof(float));
    float* fitnesses = (float*) malloc(initial_pop_size * sizeof(float));
    float* gbest = (float*) malloc(initial_pop_size * sizeof(float));
    float* gbest_fitness = (float*) malloc(sizeof(float));

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
    int i;
    for(i = 0; i < MAX_INDIVIDUALS; i++){
        if(i < initial_pop_size){
            (*(list_active_index + i)).content = i;
            void insert(struct linked_list_element **first_e, struct linked_list_element *e, int index);
            insert(&active_indexes, list_active_index + i, i);
        }else{
            (*(list_free_elem + i - initial_pop_size)).content = MAX_INDIVIDUALS - i;
            push(&free_indexes, list_free_elem + i - initial_pop_size);
        }
    }
    
    float* inertia_weight = (float*) malloc(sizeof(float));
    *inertia_weight = INIT_W;
    float* c1 = (float*) malloc(sizeof(float));
    *c1 = INIT_C1;
    float* c2 = (float*) malloc(sizeof(float));
    *c2 = INIT_C2;

    float c1_update_value = (FINAL_C1 - INIT_C1) / MAX_ITERATIONS;
    float c2_update_value = (FINAL_C2 - INIT_C2) / MAX_ITERATIONS;
    float w_update_value = (FINAL_W - INIT_W) / MAX_ITERATIONS;

    int* active_indexes_vec = convert_linked_list_vector(active_indexes, initial_pop_size);
    qualidade_solucao_corrente = run((int)roundf(solucao[0]),
                                     MAX_INDIVIDUALS, D, positions, fitnesses,
                                     gbest, gbest_fitness,
                                     inertia_weight, w_update_value, c1,
                                     c1_update_value, c2, c2_update_value,
                                     100, active_indexes_vec, 0,
                                     NULL, &positions_dev,
                                     &fitness_dev, &velocities_dev, &personal_bests_dev,
                                     &personal_best_fitness_dev, &gbest_dev,
                                     &gbest_fitness_dev, 1, 0, 1);
    copiar_vetor(solucao, melhor_solucao, num_dimensoes);
    qualidade_melhor_solucao = qualidade_solucao_corrente;
    printf("Solucao inicial:\n");
    for(i = 0; i < num_dimensoes; i++){
        printf("%f ", melhor_solucao[i]);
    }
    printf("\n");
    while(temperatura > 0.0001){
        qualidade_solucao_corrente = executar_temperatura_fixa(solucao,
            qualidade_solucao_corrente,
            num_dimensoes, temperatura, 100, D,
            positions, fitnesses,
            gbest, gbest_fitness, inertia_weight,
            w_update_value, c1, c1_update_value, c2,
            c2_update_value, 0, 100, &positions_dev,
            &fitness_dev, &velocities_dev, &personal_bests_dev,
            &personal_best_fitness_dev, &gbest_dev,
            &gbest_fitness_dev, active_indexes, free_indexes, 1, 10);
        temperatura = calcular_nova_temperatura(temperatura, alfa);
        qualidade_melhor_solucao = atualizar_melhor_solucao(solucao, qualidade_solucao_corrente,
                                                        melhor_solucao, qualidade_melhor_solucao,
                                                        num_dimensoes);
        for(i = 0; i < num_dimensoes; i++){
            printf("%f ", melhor_solucao[i]);
        }
        printf(" = %f\n", qualidade_melhor_solucao);
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
    free(inertia_weight);
    free(c1);
    free(c2);
  return 0;
}