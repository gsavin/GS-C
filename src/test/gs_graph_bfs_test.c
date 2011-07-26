#include "gs_graph_bfs.h"
#include "gs_test.h"

#define AN(id) gs_graph_node_add(g, id)
#define AE(src,trg) gs_graph_edge_add(g, src"_"trg, src, trg, GS_FALSE)
int
main(int argc, char **argv)
{
  GSGraph    *g;
  GSIterator *iterator;
  GSNode     *node;
  int         i;
  gsid        id;
  char       *path [] = {"root", "a", "b", "c", "b_1", "b_2", "b_2_1", "b_2_2", "b_2_3"};

  BEGIN("create graph");
  g = gs_graph_create("g");
  if (!g)
    FAILED;

  AN("root");

  AN("a");
  AN("b");
  AN("c");

  AN("b_1");
  AN("b_2");

  AN("b_2_1");
  AN("b_2_2");
  AN("b_2_3");
  
  AE("root", "a");
  AE("root", "b");
  AE("root", "c");

  AE("b", "b_1");
  AE("b", "b_2");

  AE("b_2", "b_2_1");
  AE("b_2", "b_2_2");
  AE("b_2", "b_2_3");

  DONE;

  BEGIN("check iterator bfs");
 
  i = 0;
  iterator = gs_graph_iterator_bfs_new_from_root_id(g, "root");
 
  while ((node = gs_iterator_next_node(iterator))) {
    id = gs_element_id_get(GS_ELEMENT(node));
    
    if (strcmp(id, path [i]))
      FAILED;

    i++;
  }

  DONE;

  gs_iterator_free(iterator);
  gs_graph_destroy(g);

  return 0;
}
