#ifndef _GS_TYPES_H_
#define _GS_TYPES_H_

#include <glib.h>

#ifdef __cplusplus
extern "C" {
#endif

  typedef gchar*   gsid;
  typedef gchar*   gskey;
  typedef double   gsreal;
  typedef gboolean gsboolean;
  
#define GS_TRUE TRUE
#define GS_FALSE FALSE

  /*
   * Objects
   */
  typedef struct _element  GSElement;
  typedef struct _node     GSNode;
  typedef struct _edge     GSEdge;
  typedef struct _graph    GSGraph;
  typedef struct _source   GSSource;
  typedef struct _sink     GSSink;
  typedef struct _iterator GSIterator;

  /*
   * Callbacks
   */
  typedef void (*GSGraphCB)(GSGraph *node, void **data);
  typedef void (*GSNodeCB)(gsid id, GSNode *node, void *data);
  typedef void (*GSEdgeCB)(gsid id, GSEdge *edge, void *data);
  typedef void (*GSKeyCB)(GSElement *element, gskey *key, void **data);
  
#define NODE_CALLBACK(function) ((GSNodeCB)function)
#define EDGE_CALLBACK(function) ((GSEdgeCB)function)
#define KEY_CALLBACK(function) ((GSKeyCB)function)

#include "gs_iterator.h"

#ifdef __cplusplus
}
#endif

#endif /* _GS_TYPES_H_ */
