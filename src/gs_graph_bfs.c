
#include "gs_graph_bfs.h"

typedef struct _graph_iterator_bfs graph_iterator_bfs;

static const char GRAPH_BFS_ITERATOR_MAGIC_STR[] = "Graph BFS Iterator";

#define CHECK_GRAPH_BFS_ITERATOR_MAGIC(d, ...)			\
  do {								\
    if (!EINA_MAGIC_CHECK(d, EINA_MAGIC_GRAPH_BFS_ITERATOR)) {	\
      EINA_MAGIC_FAIL(d, EINA_MAGIC_GRAPH_BFS_ITERATOR);	\
      return __VA_ARGS__;					\
    }								\
  } while(0)

struct _graph_iterator_bfs {
  Eina_Iterator parent;

  graph_t *graph;
  iterator_t *nodes;
  Eina_List *candidats;
  Eina_List *closed;

  EINA_MAGIC
};

/**********************************************************************
 * PRIVATE
 */

GSAPI static Eina_Bool
_bfs_next(graph_iterator_bfs *iterator,
			    void **data)
{
  node_t *next;

  if(iterator->candidats == NULL &&
     iterator->nodes     == NULL)
    return EINA_FALSE;

  do {
    if(eina_list_count(iterator->candidats) > 0) {
      next = (node_t*) eina_list_data_get(iterator->candidats);
      EINA_LOG_DBG("candidats : %s", gs_element_id_get(GS_ELEMENT(next)));

      iterator->candidats = eina_list_remove_list(iterator->candidats,
						  iterator->candidats);
    }
    else {
      next = gs_iterator_next_node(iterator->nodes);

      if(next == NULL) {
	gs_iterator_free(iterator->nodes);
	iterator->nodes = NULL;
      }
      else
	EINA_LOG_DBG("next  node : %s", gs_element_id_get(GS_ELEMENT(next)));
    }
  } while( next != NULL && 
	   eina_list_data_find(iterator->closed, next) != NULL);

  if(next != NULL) {
    iterator_t *edges;
    edge_t *edge;
    node_t *op;

    EINA_LOG_DBG("next found : %s", gs_element_id_get(GS_ELEMENT(next)));

    iterator->closed = eina_list_append(iterator->closed, next);
    *data = next;

    edges = gs_node_edge_iterator_new(next);
    edge  = gs_iterator_next_edge(edges);

    while(edge != NULL) {
      op = gs_edge_oposite_get(edge, next);

      if(eina_list_data_find(iterator->closed, op) == NULL &&
	 eina_list_data_find(iterator->candidats, op) == NULL)
	iterator->candidats = eina_list_append(iterator->candidats, op);

      edge  = gs_iterator_next_edge(edges);
    }

    gs_iterator_free(edges);
    
    return EINA_TRUE;
  } else {
    return EINA_FALSE;
  }
}

GSAPI static void*
_bfs_get_container(graph_iterator_bfs *iterator)
{
  return iterator->graph->nodes;
}

GSAPI static void
_bfs_free(graph_iterator_bfs *iterator)
{
  if(iterator->nodes != NULL)
    gs_iterator_free(iterator->nodes);

  EINA_MAGIC_SET(iterator, EINA_MAGIC_NONE);
  EINA_MAGIC_SET(&iterator->parent, EINA_MAGIC_NONE);

  iterator->nodes = NULL;
  iterator->graph = NULL;
  
  eina_list_free(iterator->candidats);
  eina_list_free(iterator->closed);

  free(iterator);
}

GSAPI static Eina_Bool
_bfs_lock(graph_iterator_bfs *iterator)
{

}

GSAPI static Eina_Bool
_bfs_unlock(graph_iterator_bfs *iterator)
{

}

GSAPI static inline graph_iterator_bfs *
_gs_graph_iterator_bfs_create(const graph_t *graph)
{
  graph_iterator_bfs *iterator;
  iterator = (graph_iterator_bfs*) malloc(sizeof(graph_iterator_bfs));
  
  EINA_MAGIC_SET(iterator, GRAPH_BFS_ITERATOR_MAGIC);
  EINA_MAGIC_SET(&iterator->parent, EINA_MAGIC_ITERATOR);

  iterator->parent.next = FUNC_ITERATOR_NEXT(_bfs_next);
  iterator->parent.get_container = FUNC_ITERATOR_GET_CONTAINER(_bfs_get_container);
  iterator->parent.free = FUNC_ITERATOR_FREE(_bfs_free);
  iterator->parent.lock = FUNC_ITERATOR_LOCK(_bfs_lock);
  iterator->parent.unlock = FUNC_ITERATOR_LOCK(_bfs_unlock);

  iterator->nodes = gs_graph_node_iterator_new(graph);
  iterator->graph = graph;
  iterator->candidats = NULL;
  iterator->closed = NULL;

  return iterator;
}

/**********************************************************************
 * PUBLIC
 */

GSAPI iterator_t*
gs_graph_iterator_bfs_new(const graph_t *graph)
{
  graph_iterator_bfs *iterator;
  iterator = _gs_graph_iterator_bfs_create(graph);

  return (iterator_t*) iterator;
}

GSAPI iterator_t*
gs_graph_iterator_bfs_new_from_root(const graph_t *graph,
			  element_id_t root)
{
  graph_iterator_bfs *iterator;

  if(root != NULL) {
    node_t *r;
    r = gs_graph_node_get(graph, root);
    
    if(r == NULL)
      ERROR(GS_ERROR_NODE_NOT_FOUND);
    
    iterator = _gs_graph_iterator_bfs_create(graph);
    iterator->candidats = eina_list_append(iterator->candidats, r);

    return (iterator_t*) iterator;
  }

  return NULL;
}
