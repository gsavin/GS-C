#include "gs_stream_dgs.h"
#include "gs_graph.h"
#include "gs_test.h"

#define AN(id) gs_graph_node_add(g, id)
#define AE(src,trg) gs_graph_edge_add(g, src"_"trg, src, trg, GS_FALSE)

int
main(int argc, char **argv)
{
  GSGraph     *g;
  GSSinkDGS   *out;
  GSSourceDGS *in;

  BEGIN("create graph");
  g = gs_graph_create("g");
  DONE;

  BEGIN("create sink");
  out = gs_stream_sink_file_dgs_open("foo.dgs");
  if (out)
    DONE;
  else 
    FAILED;

  BEGIN("connect");
  gs_stream_source_sink_add(GS_SOURCE(g),
                            GS_SINK(out));
  DONE;

  BEGIN("write");
  AN("A");
  AN("B");
  AN("C");

  AE("A", "B");
  AE("A", "C");
  AE("B", "C");
  gs_stream_sink_file_dgs_close(out);
  DONE;

  gs_graph_destroy(g);
  g = gs_graph_create("g");

  BEGIN("create source");
  in = gs_stream_source_file_dgs_open("foo.dgs");
  if (in)
    DONE;
  else 
    FAILED;

  BEGIN("connect");
  gs_stream_source_sink_add(GS_SOURCE(in),
                            GS_SINK(g));
  DONE;
  
  BEGIN("read");
  while (gs_stream_source_file_dgs_next(in))
    ;

  gs_stream_source_file_dgs_close(in);

  DONE;
  
  BEGIN("check nodes");
  
  if (gs_graph_node_get_count(g) != 3)
    FAILED;

  if (gs_graph_node_get(g, "A") == NULL ||
      gs_graph_node_get(g, "B") == NULL ||
      gs_graph_node_get(g, "C") == NULL)
    FAILED;

  DONE;

  BEGIN("check edges");
  
  if (gs_graph_edge_get_count(g) != 3)
    FAILED;

  if (gs_graph_edge_get(g, "A_B") == NULL ||
      gs_graph_edge_get(g, "B_C") == NULL ||
      gs_graph_edge_get(g, "A_C") == NULL)
    FAILED;

  DONE;

  gs_graph_destroy(g);

  return 0;
}
