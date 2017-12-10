#include <stdio.h>
#include <stdlib.h>

#include "stack.h"

void push(struct stack_element **first_e, struct stack_element *e){
	(*e).next_stack_element = *first_e;
	*first_e = e;
}

//LIBERAR MEMORIA
struct stack_element* pop(struct stack_element **first_e){
	struct stack_element* popped_e = *first_e;
	*first_e = (*popped_e).next_stack_element;
	(*popped_e).next_stack_element = NULL;
	return popped_e;
}

/*int main(){
	struct stack_element e;
	struct stack_element e2;
	e.next_stack_element = &e2;
	e.content = 0;
	e2.content = 1;

	struct stack_element *first_e = &e;

	struct stack_element e3;
	e3.content = 2;
	push(&first_e, &e3);

	printf("%d\n", (*first_e).content);
	struct stack_element pe = pop(&first_e);
	printf("%d\n", pe.content);
	printf("%d\n", (*first_e).content);
	pop(&first_e);
	printf("%d\n", (*first_e).content);
}*/