#include "gs_iterator.h"
#include <stdlib.h>

/**********************************************************************
 * PRIVATE
 */

struct _list_iterator {
  GSIterator parent;
  GList     *list;
};

GSAPI static gboolean
_list_next(struct _list_iterator *iterator,
	   void                 **data)
{
  if (!iterator->list)
    return GS_FALSE;

  *data = (void*) iterator->list->data;
  iterator->list = iterator->list->next;

  return GS_TRUE;
}

/**********************************************************************
 * PUBLIC
 */

GSAPI GSIterator*
gs_iterator_list_new(GList *list)
{
  struct _list_iterator *it;
  
  it = (struct _list_iterator*) malloc(sizeof(struct _list_iterator));
  it->parent.__next = (GSIteratorNextCB) _list_next;
  it->parent.__free = NULL;

  return (GSIterator*) it;
}

GSAPI void
gs_iterator_free(GSIterator *iterator)
{
  if (iterator->__free)
    iterator->__free(iterator);
  
  free(iterator);
}

GSAPI inline void*
gs_iterator_next(GSIterator *iterator)
{
  void **data;

  if (iterator->__next(iterator, data))
    return *data;

  return NULL;
}

GSAPI inline GSNode*
gs_iterator_next_node(GSIterator *iterator)
{
  return (GSNode*) gs_iterator_next(iterator);
}

GSAPI inline GSEdge*
gs_iterator_next_edge(GSIterator *iterator)
{
  return (GSEdge*) gs_iterator_next(iterator);
}
