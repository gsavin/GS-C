#include "gs_graph_bfs.h"

typedef struct _graph_iterator_bfs graph_iterator_bfs;

static const char GRAPH_BFS_ITERATOR_MAGIC_STR[] = "Graph BFS Iterator";

#define CHECK_GRAPH_BFS_ITERATOR_MAGIC(d, ...)			\
  do {								\
    if (!EINA_MAGIC_CHECK(d, GRAPH_BFS_ITERATOR_MAGIC)) {	\
      EINA_MAGIC_FAIL(d, GRAPH_BFS_ITERATOR_MAGIC);		\
      return __VA_ARGS__;					\
    }								\
  } while(0)

struct _graph_iterator_bfs {
  Eina_Iterator parent;

  graph_t     *graph;
  iterator_t  *nodes;
  Eina_List   *candidats;
  Eina_Hash   *candidats_hash;
  Eina_Hash   *closed;
  unsigned int depth;
  unsigned int max_depth;
  unsigned int circle_remaining;
  unsigned int next_circle_size;
  bool_t       make_hop;

  EINA_MAGIC
};

/**********************************************************************
 * PRIVATE
 */

static const bool_t not_null = GS_TRUE;

GSAPI static void
_bfs_hash_free(void *data)
{
  // Nothing to do
}

GSAPI static Eina_Bool
_bfs_next(graph_iterator_bfs *iterator,
	  void              **data)
{
  node_t *next;

  if(iterator->candidats == NULL &&
     iterator->nodes     == NULL)
    return EINA_FALSE;

  do {
    if (eina_list_count(iterator->candidats) > 0) {
      next = (node_t*) eina_list_data_get(iterator->candidats);

      iterator->candidats = eina_list_remove_list(iterator->candidats,
						  iterator->candidats);
      eina_hash_del_by_key(iterator->candidats_hash, &next);

      //EINA_LOG_DBG("next from candidats: \"%s\"", gs_element_id_get(GS_ELEMENT(next)));
    }
    else if (iterator->make_hop) {
      next = gs_iterator_next_node(iterator->nodes);
      iterator->circle_remaining = 1;
      iterator->depth = 0;

      if(next == NULL) {
	gs_iterator_free(iterator->nodes);
	iterator->nodes = NULL;

	//EINA_LOG_DBG("next from hop: \"%s\"", gs_element_id_get(GS_ELEMENT(next)));
      }
    }
    else {
      next = NULL;
    }
  } while (next != NULL && 
	   eina_hash_find(iterator->closed, &next) != NULL);

  if(next != NULL) {
    iterator_t *edges;
    edge_t *edge;
    node_t *op;

    //EINA_LOG_DBG("next found : %s", gs_element_id_get(GS_ELEMENT(next)));

    eina_hash_add(iterator->closed, &next, &not_null);
    *data = next;

    edges = gs_node_edge_iterator_new(next);
    edge  = gs_iterator_next_edge(edges);

    while(edge != NULL) {
      op = gs_edge_oposite_get(edge, next);
      
      if(eina_hash_find(iterator->closed, &op) == NULL &&
	 eina_hash_find(iterator->candidats_hash, &op) == NULL) {
	iterator->candidats = eina_list_append(iterator->candidats, op);
	iterator->next_circle_size++;
	eina_hash_add(iterator->candidats_hash, &op, &not_null);
      }

      edge  = gs_iterator_next_edge(edges);
    }

    iterator->circle_remaining--;
    gs_iterator_free(edges);

    if (iterator->circle_remaining == 0) {
      iterator->circle_remaining = iterator->next_circle_size;
      iterator->next_circle_size = 0;

      iterator->depth++;
      if (iterator->depth > iterator->max_depth)
	iterator->max_depth = iterator->depth;
      
      //EINA_LOG_DBG("current depth : %d, %d remaining", iterator->depth,
      //		   iterator->circle_remaining);
    }
    
    return EINA_TRUE;
  }
  else
    return EINA_FALSE;
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

  EINA_MAGIC_SET(iterator,          EINA_MAGIC_NONE);
  EINA_MAGIC_SET(&iterator->parent, EINA_MAGIC_NONE);

  iterator->nodes = NULL;
  iterator->graph = NULL;
  
  eina_list_free(iterator->candidats);
  eina_hash_free(iterator->candidats_hash);
  eina_hash_free(iterator->closed);

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
  
  EINA_MAGIC_SET(iterator,          GRAPH_BFS_ITERATOR_MAGIC);
  EINA_MAGIC_SET(&iterator->parent, EINA_MAGIC_ITERATOR);

  iterator->parent.next          = FUNC_ITERATOR_NEXT(_bfs_next);
  iterator->parent.get_container = FUNC_ITERATOR_GET_CONTAINER(_bfs_get_container);
  iterator->parent.free          = FUNC_ITERATOR_FREE(_bfs_free);
  iterator->parent.lock          = FUNC_ITERATOR_LOCK(_bfs_lock);
  iterator->parent.unlock        = FUNC_ITERATOR_LOCK(_bfs_unlock);

  iterator->nodes            = gs_graph_node_iterator_new(graph);
  iterator->graph            = graph;
  iterator->candidats        = NULL;
  iterator->candidats_hash   = eina_hash_pointer_new(EINA_FREE_CB(_bfs_hash_free));
  iterator->closed           = eina_hash_pointer_new(EINA_FREE_CB(_bfs_hash_free));
  iterator->depth            = 0;
  iterator->max_depth        = 0;
  iterator->circle_remaining = 0;
  iterator->next_circle_size = 0;
  iterator->make_hop         = GS_FALSE;

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
				    const node_t *root,
				    bool_t make_hop)
{
  graph_iterator_bfs *iterator;

  if(root == NULL)
    ERROR(GS_ERROR_NULL_POINTER);
    
  iterator = _gs_graph_iterator_bfs_create(graph);
  iterator->candidats        = eina_list_append(iterator->candidats, root);
  iterator->circle_remaining = 1;
  iterator->make_hop         = make_hop;

  return (iterator_t*) iterator;
}

GSAPI iterator_t*
gs_graph_iterator_bfs_new_from_root_id(const graph_t *graph,
				       element_id_t root,
				       bool_t make_hop)
{
  graph_iterator_bfs *iterator;
  node_t *r;

  if (root == NULL)
    ERROR(GS_ERROR_NULL_POINTER);

  r = gs_graph_node_get(graph, root);
    
  if(r == NULL)
    ERROR(GS_ERROR_NODE_NOT_FOUND);
    
  iterator = _gs_graph_iterator_bfs_create(graph);
  iterator->candidats        = eina_list_append(iterator->candidats, r);
  iterator->circle_remaining = 1;
  iterator->make_hop         = make_hop;

  return (iterator_t*) iterator;
}

GSAPI unsigned int
gs_graph_iterator_bfs_depth(const iterator_t *iterator)
{
  if(iterator)
    CHECK_GRAPH_BFS_ITERATOR_MAGIC((graph_iterator_bfs*) iterator, NULL);

  return ((graph_iterator_bfs*) iterator)->depth - 1;
}

GSAPI unsigned int
gs_graph_iterator_bfs_depth_max(const iterator_t *iterator)
{
  if(iterator)
    CHECK_GRAPH_BFS_ITERATOR_MAGIC((graph_iterator_bfs*) iterator, NULL);

  return ((graph_iterator_bfs*) iterator)->max_depth - 1;
}

