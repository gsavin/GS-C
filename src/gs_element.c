#include "gs_element.h"

/**********************************************************************
 * PRIVATE
 */

typedef struct {
  union {
    int    int_value;
    gsreal real_value;
    char  *string_value;
    void  *default_value;
  };
} attribute_value;

GSAPI static inline attribute_value *
_gs_element_attribute_new_value()
{
  attribute_value *value;
  value = (attribute_value*) malloc(sizeof(attribute_value));
  return value;
}

GSAPI static void
_gs_element_value_free(void *data)
{
  free(data);
}

GSAPI static inline attribute_value*
_gs_element_attribute_get(const GSElement *element,
			  const gskey      key)
{
  attribute_value *value;
  value = (attribute_value*) g_hash_table_lookup(e->attributes, key);

  if(value == NULL)
    ERROR(GS_ERROR_UNKNOWN_ATTRIBUTE);
}

GSAPI static inline void
_gs_element_attribute_add(const GSElement *element,
			  const gskey      key,
			  attribute_value *value)
{
  g_hash_table_insert(e->attributes, key, value);
}

/**********************************************************************
 * PUBLIC
 */

GSAPI void
gs_element_init(GSElement *element,
		gsid       id)
{
  element->id = id;
  element->attributes = g_hash_table_new_full(g_str_hash, g_str_equal,
					      NULL, _gs_element_value_free);
}

GSAPI void
gs_element_finalize(GSElement *element)
{
  g_hash_table_destroy(element->attributes);
}

GSAPI gsid
gs_element_id_get(const GSElement *element)
{
  return element->id;
}

GSAPI void
gs_element_attribute_add(const GSElement *element,
			 const gskey      key,
			 void            *value)
{
  attribute_value *av;
  av = _gs_element_attribute_new_value();
  av->default_value = value;

  _gs_element_attribute_add(element, key, av);
}

GSAPI void
gs_element_attribute_add_int(const GSElement *element,
			     const gskey      key,
			     int              value)
{
  attribute_value *av;
  av = _gs_element_attribute_new_value();
  av->int_value = value;

  _gs_element_attribute_add(element, key, av);
}

GSAPI void
gs_element_attribute_add_real(const GSElement *element,
			      const gskey      key,
			      gsreal           value)
{
  attribute_value *av;
  av = _gs_element_attribute_new_value();
  av->real_value = value;

  _gs_element_attribute_add(element, key, av);
}

GSAPI void
gs_element_attribute_add_string(const GSElement *element,
				const gskey      key,
				char            *value)
{
  attribute_value *av;
  av = _gs_element_attribute_new_value();
  av->string_value = value;

  _gs_element_attribute_add(element, key, av);
}

GSAPI void *
gs_element_attribute_change(const GSElement *element,
			    const gskey      key,
			    void            *value)
{
  attribute_value *av;
  void *old;

  av = _gs_element_attribute_get(element, key);
  old = av->default_value;
  av->default_value = value;

  return old;
}

GSAPI int
gs_element_attribute_change_int(const GSElement *element,
				const gskey      key,
				int              value)
{
  attribute_value *av;
  int old;

  av = _gs_element_attribute_get(element, key);
  old = av->int_value;
  av->int_value = value;

  return old;
}

GSAPI real_t
gs_element_attribute_change_real(const GSElement *element,
				 const gskey      key,
				 gsreal           value)
{
  attribute_value *av;
  gsreal old;

  av = _gs_element_attribute_get(element, key);
  old = av->real_value;
  av->real_value = value;

  return old;
}

GSAPI char *
gs_element_attribute_change_string(const GSElement *element,
				   const gskey      key,
				   char            *value)
{
  attribute_value *av;
  char *old;

  av = _gs_element_attribute_get(element, key);
  old = av->string_value;
  av->string_value = value;

  return old;
}

GSAPI void
gs_element_attribute_delete(const GSElement *element,
			    const gskey      key)
{
  if(g_hash_table_remove(element->attributes, key)!=GS_TRUE) {
    fprintf(stderr, "error while deleting attribute\n");
  }
}

GSAPI void *
gs_element_attribute_get(const GSElement *element,
			 const gskey      key)
{
  attribute_value *value;
  value = _gs_element_attribute_get(element, key);

  return value->default_value;
}

GSAPI int
gs_element_attribute_get_int(const GSElement *element,
			     const gskey      key)
{
  attribute_value *value;
  value = _gs_element_attribute_get(element, key);

  return value->int_value;
}

GSAPI gsreal
gs_element_attribute_get_real(const GSElement *element,
			      const gskey      key)
{
  attribute_value *value;
  value = _gs_element_attribute_get(element, key);

  return value->real_value;
}

GSAPI char *
gs_element_attribute_get_string(const GSElement *element,
				const gskey      key)
{
  attribute_value *value;
  value = _gs_element_attribute_get(element, key);

  return value->string_value;
}

GSAPI GSIterator *
gs_element_attribute_key_iterator_new(const GSElement *e)
{

}

GSAPI gskey *
gs_element_attribute_key_iterator_next(const GSIterator *it)
{

}

GSAPI void
gs_element_attribute_key_foreach(const GSElement *element,
				 GSKeyCB          callback,
				 void           **data)
{

}
