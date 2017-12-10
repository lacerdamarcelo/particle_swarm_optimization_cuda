#ifndef __LINKED_LIST_H__
#define __LINKED_LIST_H__

struct linked_list_element{
	int content;
	struct linked_list_element *next_linked_list_element;
};

void insert(struct linked_list_element **first_e, struct linked_list_element *e, int index);
struct linked_list_element* remove_e(struct linked_list_element **first_e, int index);
#endif