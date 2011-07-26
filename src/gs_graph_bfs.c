#include "gs_graph_bfs.h"

typedef struct _graph_iterator_bfs GSIteratorBFS;

struct _graph_iterator_bfs {
  GSIterator   parent;
  GSGraph     *graph;
  GList       *candidats;
  GHashTable  *closed;
  unsigned int depth_max;
};

/**********************************************************************
 * PRIVATE
 */

static const gsboolean not_null = GS_TRUE;

GSAPI static void
_bfs_hash_closed_free(int *data)
{
  free(data);
}

GSAPI static inline int
_bfs_get_depth(GSIteratorBFS *iterator,
	       GSNode        *node)
{
  int *d;
  d = (int*) g_hash_table_lookup(iterator->closed, &node);

  if (d == NULL)
    return -1;
  
  return d [0];
}

GSAPI static inline void
_bfs_set_depth(GSIteratorBFS *iterator,
	       const GSNode  *node,
	       int            depth)
{
  int *d;
  d    = (int*) malloc(sizeof(int));
  d[0] = depth;

  g_hash_table_insert(iterator->closed, node, d);
}

GSAPI static inline void
_bfs_add_neighbors(GSIteratorBFS *iterator,
		   GSNode        *node)
{
  GSIterator *edges;
  GSEdge     *edge;
  GSNode     *op;
  int         depth;

  edges = gs_node_edge_iterator_new(node);
  edge  = gs_iterator_next_edge(edges);
  depth = -1;

  while (edge != NULL) {
    op = GS_EDGE_OPOSITE_OF(edge, node);

    if (g_hash_table_lookup(iterator->closed, op) == NULL) {
      iterator->candidats = g_list_append(iterator->candidats, op);
      
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

GSAPI static gsboolean
_bfs_next(GSIteratorBFS *iterator,
	  void         **data)
{
  if (iterator->candidats) {
    GSNode     *next;
    GSIterator *edges;
    GSEdge     *edge;
    GSNode     *op;
    int         depth;
    
    next = (GSNode*) iterator->candidats->data;
    iterator->candidats = g_list_delete_link(iterator->candidats, iterator->candidats);

    //_bfs_add_neighbors(iterator, next);

    edges = gs_node_edge_iterator_new(next);
    edge  = gs_iterator_next_edge(edges);
    depth = -1;
    
    while (edge != NULL) {
      op = GS_EDGE_OPOSITE_OF(edge, next);
      
      if (g_hash_table_lookup(iterator->closed, op) == NULL) {
	iterator->candidats = g_list_append(iterator->candidats, op);
	
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
    
    return GS_TRUE;
  }
  else {
    return GS_FALSE;
  }
}

GSAPI static void
_bfs_free(GSIteratorBFS *iterator)
{
  iterator->graph = NULL;
  
  g_list_free(iterator->candidats);
  g_hash_table_destroy(iterator->closed);

  free(iterator);
}

GSAPI static inline GSIteratorBFS*
_gs_graph_iterator_bfs_create(const GSGraph *graph)
{
  GSIteratorBFS *iterator;
  iterator = (GSIteratorBFS*) malloc(sizeof(GSIteratorBFS));

  iterator->parent.__next    = (GSIteratorNextCB) _bfs_next;
  iterator->parent.__free    = (GSIteratorFreeCB) _bfs_free;

  iterator->graph            = graph;
  iterator->candidats        = NULL;
  iterator->closed           = g_hash_table_new_full(g_direct_hash,
						     g_direct_equal,
						     NULL,
						     (GDestroyNotify) _bfs_hash_closed_free);
  iterator->depth_max        = 0;

  return iterator;
}

/**********************************************************************
 * PUBLIC
 */

GSAPI GSIterator*
gs_graph_iterator_bfs_new(const GSGraph *graph)
{
  GSIteratorBFS *iterator;
  iterator = _gs_graph_iterator_bfs_create(graph);

  return (GSIterator*) iterator;
}

GSAPI GSIterator*
gs_graph_iterator_bfs_new_from_root(const GSGraph *graph,
				    const GSNode *root)
{
  GSIteratorBFS *iterator;

  if(root == NULL)
    ERROR(GS_ERROR_NULL_POINTER);
    
  iterator = _gs_graph_iterator_bfs_create(graph);
  iterator->candidats = g_list_append(iterator->candidats, root);
  _bfs_set_depth(iterator, root, 0);

  return (GSIterator*) iterator;
}

GSAPI GSIterator*
gs_graph_iterator_bfs_new_from_root_id(const GSGraph *graph,
				       gsid           root)
{
  GSIteratorBFS *iterator;
  GSNode        *r;

  if (root == NULL)
    ERROR(GS_ERROR_NULL_POINTER);

  r = gs_graph_node_get(graph, root);
    
  if(r == NULL)
    ERROR(GS_ERROR_NODE_NOT_FOUND);
    
  iterator            = _gs_graph_iterator_bfs_create(graph);
  iterator->candidats = g_list_append(NULL, r);
  _bfs_set_depth(iterator, r, 0);

  return (GSIterator*) iterator;
}

GSAPI void
gs_graph_iterator_bfs_reset_from_root(GSIterator   *iterator,
				      const GSNode *root)
{
  if(iterator) {
    GSIteratorBFS *bfs;
    bfs = (GSIteratorBFS*) iterator;

    g_hash_table_remove_all(bfs->closed);
    g_list_free(bfs->candidats);
    bfs->depth_max = 0;
    bfs->candidats = g_list_append(NULL, root);
    _bfs_set_depth(bfs, root, 0);
  }
}

GSAPI unsigned int
gs_graph_iterator_bfs_depth_max(const GSIterator *iterator)
{
  return ((GSIteratorBFS*) iterator)->depth_max;
}

