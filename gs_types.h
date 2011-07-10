#ifndef _GS_TYPES_H_
#define _GS_TYPES_H_

#include <Eina.h>

typedef char* element_id_t;
typedef char* gs_key_t;
typedef double real_t;
typedef Eina_Bool bool_t;

#define GS_TRUE EINA_TRUE
#define GS_FALSE EINA_FALSE

/*
 * Objects
 */
typedef struct _element element_t;
typedef struct _node node_t;
typedef struct _edge edge_t;
typedef struct _graph graph_t;
typedef struct _source source_t;
typedef struct _sink sink_t;

typedef Eina_Iterator iterator_t;

/*
 * Callbacks
 */
typedef void (*graph_cb_t)(graph_t *node, void **data);
typedef void (*node_cb_t)(node_t *node, void **data);
typedef void (*edge_cb_t)(edge_t *edge, void **data);
typedef void (*key_cb_t)(element_t *element, key_t *key, void **data);

#define NODE_CALLBACK(function) ((node_cb_t)function)
#define EDGE_CALLBACK(function) ((edge_cb_t)function)
#define KEY_CALLBACK(function) ((key_cb_t)function)

#endif /* _GS_TYPES_H_ */
