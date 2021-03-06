#ifndef _GS_ITERATOR_H_
#define _GS_ITERATOR_H_

#include "config.h"
#include "gs_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  typedef gsboolean (*GSIteratorNextCB)(GSIterator *iterator, void **data);
  typedef void (*GSIteratorFreeCB)(GSIterator *iterator);

  struct _iterator {
    GSIteratorNextCB __next;
    GSIteratorFreeCB __free;
  };

  GSAPI GSIterator*
  gs_iterator_list_new(GList *list);

  GSAPI void
  gs_iterator_free(GSIterator *iterator);
  
  GSAPI inline void*
  gs_iterator_next(GSIterator *iterator);

  GSAPI inline GSNode*
  gs_iterator_next_node(GSIterator *iterator);

  GSAPI inline GSEdge*
  gs_iterator_next_edge(GSIterator *iterator);

#ifdef __cplusplus
}
#endif

#endif /* _GS_ITERATOR_H_ */
