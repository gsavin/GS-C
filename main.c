#include "graphstream.h"
#include "gs_stream_dgs.h"
#include "gs_graph_bfs.h"

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

void
benchmark()
{
  Eina_Benchmark *test;
  Eina_Array     *ea;

   test = eina_benchmark_new("test", "creation");

   if (!test)
     return;

   eina_benchmark_register(test, "work-1", EINA_BENCHMARK(create_graph), 1, 100, 1);

   ea = eina_benchmark_run(test);

   eina_benchmark_free(test);
}

void
test_dgs()
{
  graph_t *g;
  source_dgs_t *in;
  sink_dgs_t *out;
  
  g = gs_graph_create("g");
  in = gs_stream_source_file_dgs_open("sample.dgs");
  out = gs_stream_sink_file_dgs_open("sample.out.dgs");

  EINA_LOG_DBG("opened");

  gs_stream_source_sink_add(GS_SOURCE(in),
			    GS_SINK(g));

  gs_stream_source_sink_add(GS_SOURCE(g),
			    GS_SINK(out));

  while(gs_stream_source_file_dgs_next(in))
    ;

  gs_stream_source_file_dgs_close(in);
  gs_stream_sink_file_dgs_close(out);
  gs_graph_destroy(g);
}

void
test_bfs()
{
  graph_t *g;
  iterator_t *it;
  node_t *n;

  g = gs_graph_create("g");
  gs_graph_node_add(g, "A");
  gs_graph_node_add(g, "A1");
  gs_graph_node_add(g, "A2");
  gs_graph_node_add(g, "A3");
  gs_graph_node_add(g, "A21");
  gs_graph_node_add(g, "A22");
  gs_graph_node_add(g, "A221");
  gs_graph_node_add(g, "A222");

  gs_graph_edge_add(g, "01", "A", "A1", GS_FALSE);
  gs_graph_edge_add(g, "02", "A", "A2", GS_FALSE);
  gs_graph_edge_add(g, "03", "A", "A3", GS_FALSE);

  gs_graph_edge_add(g, "04", "A2", "A21", GS_FALSE);
  gs_graph_edge_add(g, "05", "A2", "A22", GS_FALSE);
  gs_graph_edge_add(g, "06", "A22", "A221", GS_FALSE);
  gs_graph_edge_add(g, "07", "A22", "A222", GS_FALSE);

  it = gs_graph_iterator_bfs_new_from_root(g, "A");
  n = gs_iterator_next_node(it);

  while(n != NULL) {
    printf("- \"%s\"\n", gs_element_id_get(GS_ELEMENT(n)));
    n = gs_iterator_next_node(it);
  }

  printf("Max depth : %d\n", gs_graph_iterator_bfs_max_depth(it));

  gs_iterator_free(it);
  gs_graph_destroy(g);
}

int
main(int argc, char **argv)
{
  if (!gs_init())
    return EXIT_FAILURE;
  
  test_bfs();

  gs_shutdown();

  return EXIT_SUCCESS;

  shutdown_gs:
   gs_shutdown();

   return EXIT_FAILURE;
}
