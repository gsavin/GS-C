#include "gs_element.h"

#ifndef ELEMENT_HASH_FUNCTION
#define ELEMENT_HASH_FUNCTION eina_hash_string_djb2_new
#endif

/**********************************************************************
 * PRIVATE
 */

typedef struct {
  union {
    int int_value;
    real_t real_value;
    char *string_value;
    void *default_value;
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

GSAPI static inline attribute_value *
_gs_element_attribute_get(const element_t *e, const gs_key_t key)
{
  attribute_value *value;
  value = (attribute_value*) eina_hash_find(e->attributes, key);

  if(value == NULL)
    ERROR(GS_ERROR_UNKNOWN_ATTRIBUTE);
}

GSAPI static inline void
_gs_element_attribute_add(const element_t *e, const gs_key_t key, attribute_value *value)
{
  if(eina_hash_add(e->attributes, key, value) != EINA_TRUE)
    ERROR(GS_ERROR_UNKNOWN);
}

/**********************************************************************
 * PUBLIC
 */

GSAPI void
gs_element_init(element_t *e, element_id_t id)
{
  e->id = id;
  e->attributes = ELEMENT_HASH_FUNCTION(_gs_element_value_free);
}

GSAPI void
gs_element_finalize(element_t *e)
{
  eina_hash_free(e->attributes);
}

GSAPI element_id_t
gs_element_id_get(const element_t *element)
{
  return element->id;
}

GSAPI void
gs_element_attribute_add(const element_t *e, const gs_key_t key, void *value)
{
  attribute_value *av;
  av = _gs_element_attribute_new_value();
  av->default_value = value;

  _gs_element_attribute_add(e, key, av);
}

GSAPI void
gs_element_attribute_add_int(const element_t *e, const gs_key_t key, int value)
{
  attribute_value *av;
  av = _gs_element_attribute_new_value();
  av->int_value = value;

  _gs_element_attribute_add(e, key, av);
}

GSAPI void
gs_element_attribute_add_real(const element_t *e, const gs_key_t key, real_t value)
{
  attribute_value *av;
  av = _gs_element_attribute_new_value();
  av->real_value = value;

  _gs_element_attribute_add(e, key, av);
}

GSAPI void
gs_element_attribute_add_string(const element_t *e, const gs_key_t key, char *value)
{
  attribute_value *av;
  av = _gs_element_attribute_new_value();
  av->string_value = value;

  _gs_element_attribute_add(e, key, av);
}

GSAPI void *
gs_element_attribute_change(const element_t *e, const gs_key_t key, void *value)
{
  attribute_value *av;
  void *old;

  av = _gs_element_attribute_get(e, key);
  old = av->default_value;
  av->default_value = value;

  return old;
}

GSAPI int
gs_element_attribute_change_int(const element_t *e, const gs_key_t key, int value)
{
  attribute_value *av;
  int old;

  av = _gs_element_attribute_get(e, key);
  old = av->int_value;
  av->int_value = value;

  return old;
}

GSAPI real_t
gs_element_attribute_change_real(const element_t *e, const gs_key_t key, real_t value)
{
  attribute_value *av;
  real_t old;

  av = _gs_element_attribute_get(e, key);
  old = av->real_value;
  av->real_value = value;

  return old;
}

GSAPI char *
gs_element_attribute_change_string(const element_t *e, const gs_key_t key, char *value)
{
  attribute_value *av;
  char *old;

  av = _gs_element_attribute_get(e, key);
  old = av->string_value;
  av->string_value = value;

  return old;
}

GSAPI void
gs_element_attribute_delete(const element_t *e,const gs_key_t key)
{
  if(eina_hash_del(e->attributes, key, NULL)!=EINA_TRUE) {
    fprintf(stderr, "error while deleting attribute\n");
  }
}

GSAPI void *
gs_element_attribute_get(const element_t *e, const gs_key_t key)
{
  attribute_value *value;
  value = _gs_element_attribute_get(e, key);

  return value->default_value;
}

GSAPI int
gs_element_attribute_get_int(const element_t *e, const gs_key_t key)
{
  attribute_value *value;
  value = _gs_element_attribute_get(e, key);

  return value->int_value;
}

GSAPI real_t
gs_element_attribute_get_real(const element_t *e, const gs_key_t key)
{
  attribute_value *value;
  value = _gs_element_attribute_get(e, key);

  return value->real_value;
}

GSAPI char *
gs_element_attribute_get_string(const element_t *e, const gs_key_t key)
{
  attribute_value *value;
  value = _gs_element_attribute_get(e, key);

  return value->string_value;
}

GSAPI iterator_t *
gs_element_attribute_key_iterator_new(const element_t *e)
{

}

GSAPI gs_key_t *
gs_element_attribute_key_iterator_next(const iterator_t *it)
{

}

GSAPI void
gs_element_attribute_key_foreach(const element_t *e,
				 key_cb_t callback,
				 void **data)
{

}
