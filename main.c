#include "graphstream.h"

static void print_id_cb(void *n, void **data)
{
  printf("- %s \"%s\"\n", *data, gs_element_id_get(GS_ELEMENT(n)));
}

#define NODE_SIZE 100000
#define EDGE_SIZE 10000

static void
create_graph(int r)
{
  graph_t *g;
  int i;
  char **node_ids, **edge_ids;

  node_ids = (char**) malloc(NODE_SIZE*sizeof(char*));
  edge_ids = (char**) malloc(EDGE_SIZE*sizeof(char*));

  srand(time(NULL));

  g = gs_graph_create("g");

  for(i=0; i<NODE_SIZE; i++) {
    char *id;
    id = (char*) malloc(5*sizeof(char));
    sprintf(id, "n%04d", i);
    node_ids[i] = id;

    gs_graph_node_add(g, id);
  }

  for(i=0; i<EDGE_SIZE; i++) {
    char *id;
    int s, t;

    s = rand() % NODE_SIZE;
    do {
      t = rand() % NODE_SIZE;
    } while(t == s);

    id = (char*) malloc(5*sizeof(char));
    sprintf(id, "e%04d", i);
    edge_ids [i] = id;

    gs_graph_edge_add(g, id, node_ids[s], node_ids[t], GS_FALSE);
  }

  gs_graph_destroy(g);

  for(i=0; i<NODE_SIZE; i++)
    free(node_ids[i]);
  for(i=0; i<EDGE_SIZE; i++)
    free(edge_ids[i]);

  free(node_ids);
  free(edge_ids);
}

int
main(int argc, char **argv)
{
  Eina_Benchmark *test;
  Eina_Array     *ea;

  if (!gs_init())
    return EXIT_FAILURE;

   test = eina_benchmark_new("test", "creation");
   if (!test)
     goto shutdown_gs;

   eina_benchmark_register(test, "work-1", EINA_BENCHMARK(create_graph), 1, 100, 1);

   ea = eina_benchmark_run(test);

   eina_benchmark_free(test);
   gs_shutdown();

   return EXIT_SUCCESS;

  shutdown_gs:
   gs_shutdown();

   return EXIT_FAILURE;
}
