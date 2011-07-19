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
  Eina_List   *candidats;
  Eina_Hash   *closed;
  unsigned int depth_max;

  EINA_MAGIC
};

/**********************************************************************
 * PRIVATE
 */

static const bool_t not_null = GS_TRUE;

GSAPI static void
_bfs_hash_closed_free(int *data)
{
  free(data);
}

GSAPI static inline int
_bfs_get_depth(graph_iterator_bfs *iterator,
	       node_t             *node)
{
  int *d;
  d = (int*) eina_hash_find(iterator->closed, &node);

  if (d == NULL)
    return -1;
  
  return d [0];
}

GSAPI static inline void
_bfs_set_depth(graph_iterator_bfs *iterator,
	       node_t *node,
	       int depth)
{
  int *d;
  d    = (int*) malloc(sizeof(int));
  d[0] = depth;

  eina_hash_add(iterator->closed, &node, d);
}

GSAPI static inline void
_bfs_add_neighbors(graph_iterator_bfs *iterator,
		   node_t             *node)
{
  iterator_t *edges;
  edge_t     *edge;
  node_t     *op;
  int         depth;

  edges = gs_node_edge_iterator_new(node);
  edge  = gs_iterator_next_edge(edges);
  depth = -1;

  while (edge != NULL) {
    op = GS_EDGE_OPOSITE_OF(edge, node);

    if (eina_hash_find(iterator->closed, &op) == NULL) {
      iterator->candidats = eina_list_append(iterator->candidats, op);
      
      if (depth < 0)
	depth = _bfs_get_depth(iterator, node) + 1;

      _bfs_set_depth(iterator, op, depth);
    }
    
    edge = gs_iterator_next_edge(edges);
  }
  
  gs_iterator_free(edges);

  if (depth > 0 && depth > iterator->depth_max)
    iterator->depth_max = depth;
}

GSAPI static Eina_Bool
_bfs_next(graph_iterator_bfs *iterator,
	  void              **data)
{
  if (eina_list_count(iterator->candidats) > 0) {
    node_t *next;
    iterator_t *edges;
    edge_t     *edge;
    node_t     *op;
    int         depth;
    
    next = eina_list_data_get(iterator->candidats);
    iterator->candidats = eina_list_remove_list(iterator->candidats, iterator->candidats);

    //_bfs_add_neighbors(iterator, next);

    edges = gs_node_edge_iterator_new(next);
    edge  = gs_iterator_next_edge(edges);
    depth = -1;
    
    while (edge != NULL) {
      op = GS_EDGE_OPOSITE_OF(edge, next);
      
      if (eina_hash_find(iterator->closed, &op) == NULL) {
	iterator->candidats = eina_list_append(iterator->candidats, op);
	
	if (depth < 0)
	  depth = _bfs_get_depth(iterator, next) + 1;

	_bfs_set_depth(iterator, op, depth);
      }
      
      edge = gs_iterator_next_edge(edges);
    }
  
    gs_iterator_free(edges);

    if (depth > 0 && depth > iterator->depth_max)
      iterator->depth_max = depth;

    *data = next;
    
    return EINA_TRUE;
  }
  else {
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
  EINA_MAGIC_SET(iterator,          EINA_MAGIC_NONE);
  EINA_MAGIC_SET(&iterator->parent, EINA_MAGIC_NONE);

  iterator->graph = NULL;
  
  eina_list_free(iterator->candidats);
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

  iterator->graph            = graph;
  iterator->candidats        = NULL;
  iterator->closed           = eina_hash_pointer_new(EINA_FREE_CB(_bfs_hash_closed_free));
  iterator->depth_max        = 0;

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
				    const node_t *root)
{
  graph_iterator_bfs *iterator;

  if(root == NULL)
    ERROR(GS_ERROR_NULL_POINTER);
    
  iterator = _gs_graph_iterator_bfs_create(graph);
  iterator->candidats        = eina_list_append(iterator->candidats, root);
  _bfs_set_depth(iterator, root, 0);

  return (iterator_t*) iterator;
}

GSAPI iterator_t*
gs_graph_iterator_bfs_new_from_root_id(const graph_t *graph,
				       element_id_t root)
{
  graph_iterator_bfs *iterator;
  node_t *r;

  if (root == NULL)
    ERROR(GS_ERROR_NULL_POINTER);

  r = gs_graph_node_get(graph, root);
    
  if(r == NULL)
    ERROR(GS_ERROR_NODE_NOT_FOUND);
    
  iterator            = _gs_graph_iterator_bfs_create(graph);
  iterator->candidats = eina_list_append(iterator->candidats, r);
  _bfs_set_depth(iterator, r, 0);

  return (iterator_t*) iterator;
}

GSAPI void
gs_graph_iterator_bfs_reset_from_root(iterator_t   *iterator,
				      const node_t *root)
{
  if(iterator) {
    graph_iterator_bfs *bfs;
    CHECK_GRAPH_BFS_ITERATOR_MAGIC((graph_iterator_bfs*) iterator, NULL);
    bfs = (graph_iterator_bfs*) iterator;

    eina_hash_free_buckets(bfs->closed);
    eina_list_free(bfs->candidats);
    bfs->depth_max = 0;
    bfs->candidats = eina_list_append(bfs->candidats, root);
    _bfs_set_depth(bfs, root, 0);
  }
}

GSAPI unsigned int
gs_graph_iterator_bfs_depth_max(const iterator_t *iterator)
{
  if(iterator)
    CHECK_GRAPH_BFS_ITERATOR_MAGIC((graph_iterator_bfs*) iterator, NULL);

  return ((graph_iterator_bfs*) iterator)->depth_max;
}

