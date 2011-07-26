#include "gs_graph.h"
#include "gs_test.h"

int
main(int argc, char **argv)
{
  GSGraph *g;
  GSNode *a, *b, *c;
  GSEdge *ab, *ac;

  BEGIN("create");
  g = gs_graph_create("g");
  if (g)
    DONE;
  else
    FAILED;

  BEGIN("add nodes");
  a = gs_graph_node_add(g, "a");
  b = gs_graph_node_add(g, "b");
  c = gs_graph_node_add(g, "c");
  
  if (a && b && c)
    DONE;
  else
    FAILED;

  BEGIN("add edges");
  ab = gs_graph_edge_add(g, "ab", "a", "b", GS_FALSE);
  ac = gs_graph_edge_add(g, "ac", "a", "c", GS_TRUE);

  if (ab && ac)
    DONE;
  else
    FAILED;

  BEGIN("check nodes (id)");
  if (strcmp(gs_element_id_get(GS_ELEMENT(a)), "a") ||
      strcmp(gs_element_id_get(GS_ELEMENT(b)), "b") ||
      strcmp(gs_element_id_get(GS_ELEMENT(c)), "c") )
    FAILED;
  else
    DONE;

  BEGIN("check edges (id)");
  if (strcmp(gs_element_id_get(GS_ELEMENT(ab)), "ab") ||
      strcmp(gs_element_id_get(GS_ELEMENT(ac)), "ac") )
    FAILED;
  else
    DONE;

  BEGIN("check edges (nodes)");
  if (ab->source != a || ab->target != b || ab->directed ||
      ac->source != a || ac->target != c || !ac->directed )
    FAILED;
  else
    DONE;

  BEGIN("destroy");
  gs_graph_destroy(g);
  DONE;

  return 0;
}
