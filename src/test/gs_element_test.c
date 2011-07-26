#include "gs_element.h"
#include "gs_test.h"

int
main(int argc, char **argv)
{
  element_t *e;
  int attribute_int;
  real_t attribute_real;
  char *attribute_string;

  attribute_int = 12345;
  attribute_real = 123.45;
  attribute_string = "12345";

  BEGIN("allocation");
  e = (element_t*) malloc(sizeof(element_t));
  DONE;

  BEGIN("init");
  gs_element_init(e, "test");
  DONE;

  BEGIN("add attribute int");
  gs_element_attribute_add_int(e, "int", attribute_int);
  DONE;

  BEGIN("add attribute real");
  gs_element_attribute_add_real(e, "real", attribute_real);
  DONE;

  BEGIN("add attribute string");
  gs_element_attribute_add_string(e, "string", attribute_string);
  DONE;

  BEGIN("check int value");
  if (gs_element_attribute_get_int(e, "int") == attribute_int &&
      gs_element_attribute_get_int(e, "int") != 67890)
    DONE;
  else
    FAILED;

  BEGIN("check real value");
  if (gs_element_attribute_get_real(e, "real") == attribute_real &&
      gs_element_attribute_get_real(e, "real") != 67.890)
    DONE;
  else
    FAILED;

  BEGIN("check string value");
  if (!strcmp(gs_element_attribute_get_string(e, "string"), "12345") &&
      strcmp(gs_element_attribute_get_string(e, "string"), "1435"))
    DONE;
  else
    FAILED;

  BEGIN("finalize");
  gs_element_finalize(e);
  DONE;

  BEGIN("free");
  free(e);
  DONE;

  return 0;
}
