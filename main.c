#include "graphstream.h"
#include "gs_stream_dgs.h"
#include "gs_graph_bfs.h"
#include "gs_algorithm_diameter.h"
#include "gs_matrix.h"

static void print_id_cb(void *n, void **data)
{
  printf("- %s \"%s\"\n", *data, gs_element_id_get(GS_ELEMENT(n)));
}

#define NODE_SIZE 100000
#define EDGE_SIZE 10000

#define START_TIMER(start) ((start) = clock())
#define STOP_TIMER(start, elapsed) (elapsed) = ((double)(clock() - (start)) / CLOCKS_PER_SEC)

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
  iterator_t *it1, *it2;
  node_t *n;
  int d, t;
  source_dgs_t *in;
  clock_t t1, t2;

  t1 = clock();

  d = 0;
  g = gs_graph_create("g");
  
  in = gs_stream_source_file_dgs_open("sample.dgs");

  gs_stream_source_sink_add(GS_SOURCE(in),
			    GS_SINK(g));

  while(gs_stream_source_file_dgs_next(in))
    ;

  gs_stream_source_file_dgs_close(in);

  t2 = clock();

  printf("read in %dms\n", (t2-t1) / (CLOCKS_PER_SEC / 1000));

  /*
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
  */

  it1 = gs_graph_node_iterator_new(g);
  n = gs_iterator_next_node(it1);
  it2 = gs_graph_iterator_bfs_new_from_root(g, n);

  while(n != NULL) {

    while (gs_iterator_next_node(it2) != NULL)
      ;
    
    t = gs_graph_iterator_bfs_depth_max(it2);

    if (t > d)
      d = t;

    //printf("[%s] %d\n", gs_element_id_get(GS_ELEMENT(n)), t);

    n = gs_iterator_next_node(it1);
    gs_graph_iterator_bfs_reset_from_root(it2, n);
  }

  gs_iterator_free(it2);

  t1 = clock();

  printf("computed in %dms\n", (t1-t2) / (CLOCKS_PER_SEC / 1000));
  printf("Max depth : %d\n", d);

  gs_iterator_free(it1);
  gs_graph_destroy(g);
}

void 
test_diameter()
{
  graph_t *g;
  int d;
  source_dgs_t *in;
  
  g = gs_graph_create("g");
  in = gs_stream_source_file_dgs_open("test.dgs");

  gs_stream_source_sink_add(GS_SOURCE(in),
			    GS_SINK(g));

  while(gs_stream_source_file_dgs_next(in))
    ;

  gs_stream_source_file_dgs_close(in);

  d = gs_algorithm_diameter(g);
  gs_graph_destroy(g);
  
  printf("Diameter : %d\n", d);
}

void
test_matrix()
{
  matrix_t *m;
  source_dgs_t *in;
  
  m = gs_matrix_new();
  in = gs_stream_source_file_dgs_open("sample.dgs");

  gs_stream_source_sink_add(GS_SOURCE(in),
			    GS_SINK(m));

  while(gs_stream_source_file_dgs_next(in))
    ;

  gs_stream_source_file_dgs_close(in);

  gs_matrix_print(m, stdout);

  gs_matrix_destroy(m);
}

void
test_matrix_bfs()
{
  matrix_t *m;
  iterator_t *it;
  int idx, d, t;
  source_dgs_t *in;
  clock_t c;
  double read_time, compute_time;

  d = 0;
  m = gs_matrix_new();

  START_TIMER(c);
  
  in = gs_stream_source_file_dgs_open("sample.dgs");

  gs_stream_source_sink_add(GS_SOURCE(in),
			    GS_SINK(m));

  while(gs_stream_source_file_dgs_next(in))
    ;

  gs_stream_source_file_dgs_close(in);

  STOP_TIMER(c, read_time);
  
  /*
  gs_matrix_node_add(m, "A");
  gs_matrix_node_add(m, "A1");
  gs_matrix_node_add(m, "A2");
  gs_matrix_node_add(m, "A3");
  gs_matrix_node_add(m, "A21");
  gs_matrix_node_add(m, "A22");
  gs_matrix_node_add(m, "A221");
  gs_matrix_node_add(m, "A222");

  gs_matrix_edge_add(m, "01", "A", "A1", GS_FALSE);
  gs_matrix_edge_add(m, "02", "A", "A2", GS_FALSE);
  gs_matrix_edge_add(m, "03", "A", "A3", GS_FALSE);

  gs_matrix_edge_add(m, "04", "A2", "A21", GS_FALSE);
  gs_matrix_edge_add(m, "05", "A2", "A22", GS_FALSE);
  gs_matrix_edge_add(m, "06", "A22", "A221", GS_FALSE);
  gs_matrix_edge_add(m, "07", "A22", "A222", GS_FALSE);
  
  */
  //  gs_matrix_print(m, stdout);
  
  //printf("---- start %d ----\n", 0);
  //it = gs_matrix_iterator_bfs_new_from_index(m, 0);

  printf("%d nodes; %d edges\n", m->nodes, m->edges);

  START_TIMER(c);
  d = gs_matrix_unweighted_eccentricity(m, -1);
  STOP_TIMER(c, compute_time);

  printf("read     in %.2f s\ncomputed in %.2f s\n", read_time, compute_time);
  printf("Max depth : %d\n", d);

  gs_matrix_destroy(m);
}

int
main(int argc, char **argv)
{
  if (!gs_init())
    return EXIT_FAILURE;
  
  test_matrix_bfs();

  gs_shutdown();

  return EXIT_SUCCESS;

  shutdown_gs:
   gs_shutdown();

   return EXIT_FAILURE;
}
