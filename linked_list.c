#include <stdio.h>
#include <stdlib.h>

#include "linked_list.h"

void insert(struct linked_list_element **first_e, struct linked_list_element *e, int index){
	struct linked_list_element *current_e = NULL;
	if(*first_e == NULL){
		*first_e = e;
	}else{
		current_e = *first_e;
		if(index == 0){
			(*e).next_linked_list_element = current_e;
			*first_e = e;
		}else{
			int i;
			for(i = 0; i < index - 1; i++){
				current_e = (*current_e).next_linked_list_element;
			}
			(*e).next_linked_list_element = (*current_e).next_linked_list_element;
			(*current_e).next_linked_list_element = e;
		}
	}
}

//LIBERAR MEMORIA
struct linked_list_element* remove_e(struct linked_list_element **first_e, int index){
	struct linked_list_element *current_e = *first_e;
	struct linked_list_element *e_removed = NULL;
	if(index == 0){
		e_removed = *first_e;
		*first_e = (*e_removed).next_linked_list_element;
		(*e_removed).next_linked_list_element = NULL;
	}else{
		int i;
		for(i = 0; i < index - 1; i++){
			current_e = (*current_e).next_linked_list_element;
		}
		e_removed = (*current_e).next_linked_list_element;
		(*current_e).next_linked_list_element = (*((*current_e).next_linked_list_element)).next_linked_list_element;
		(*e_removed).next_linked_list_element = NULL;
	}
	return e_removed;
}

/*int main(){
	struct linked_list_element *e = malloc(4 * sizeof(struct linked_list_element));
	for(int i = 0; i < 4; i++){
		(*(e + i)).next_linked_list_element = NULL;
		(*(e + i)).position = (float*) malloc(10 * sizeof(float));
		(*(e + i)).fitness = 10 + i;
		(*(e + i)).velocity = (float*) malloc(10 * sizeof(float));
		(*(e + i)).personal_best = (float*) malloc(10 * sizeof(float));
		(*(e + i)).personal_best_fitness = 10;
	}

	struct linked_list_element *linked_list;
	linked_list = NULL;
	insert(&linked_list, e, 0);
	insert(&linked_list, e + 1, 1);
	insert(&linked_list, e + 2, 2);
	insert(&linked_list, e + 3, 0);

	printf("%f\n", (*linked_list).fitness);
	printf("%f\n", (*(*linked_list).next_linked_list_element).fitness);
	printf("%f\n", (*(*(*linked_list).next_linked_list_element).next_linked_list_element).fitness);
	printf("%f\n", (*(*(*(*linked_list).next_linked_list_element).next_linked_list_element).next_linked_list_element).fitness);

	printf("#####\n");
	remove_e(&linked_list, 0);
	printf("%f\n", (*linked_list).fitness);
	printf("%f\n", (*(*linked_list).next_linked_list_element).fitness);
	printf("%f\n", (*(*(*linked_list).next_linked_list_element).next_linked_list_element).fitness);
	printf("++++++\n");
	remove_e(&linked_list, 2);
	printf("%f\n", (*linked_list).fitness);
	printf("%f\n", (*(*linked_list).next_linked_list_element).fitness);
	printf("++++++\n");
	remove_e(&linked_list, 1);
	printf("%f\n", (*linked_list).fitness);
	printf("++++++\n");
	remove_e(&linked_list, 0);
	free(e);
}*/