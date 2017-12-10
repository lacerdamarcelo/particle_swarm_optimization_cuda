#ifndef __STACK_H__
#define __STACK_H__

struct stack_element{
	int content;
	struct stack_element *next_stack_element;
};

void push(struct stack_element **first_e, struct stack_element *e);
struct stack_element* pop(struct stack_element **first_e);
#endif
