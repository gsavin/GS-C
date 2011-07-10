#ifndef _GS_ELEMENT_H_
#define _GS_ELEMENT_H_

#include "gs_types.h"
#include "gs_common.h"

struct _element {
  object_t parent;
  element_id_t id;
  Eina_Hash *attributes;
};

#define GS_ELEMENT(e) ((element_t*)CHECK_TYPE(e,ELEMENT_TYPE))

GSAPI void gs_element_init(element_t*,
			   element_id_t);

GSAPI void gs_element_finalize(element_t*);

GSAPI element_id_t gs_element_id_get(const element_t*);

GSAPI void gs_element_attribute_add(const element_t*,
				    const gs_key_t,
				    void*);
GSAPI void gs_element_attribute_add_int(const element_t*,
					const gs_key_t,
					int);

GSAPI void gs_element_attribute_add_real(const element_t*,
					 const gs_key_t,
					 real_t);

GSAPI void gs_element_attribute_add_string(const element_t*,
					   const gs_key_t,
					   char*);

GSAPI void *gs_element_attribute_change(const element_t*,
					const gs_key_t,
					void*);

GSAPI int gs_element_attribute_change_int(const element_t*,
					  const gs_key_t,
					  int);

GSAPI real_t gs_element_attribute_change_real(const element_t*,
					      const gs_key_t,
					      real_t);

GSAPI char *gs_element_attribute_change_string(const element_t*,
					       const gs_key_t,
					       char*);

GSAPI void gs_element_attribute_delete(const element_t*,
				       const gs_key_t);

GSAPI void *gs_element_attribute_get(const element_t*,
				     const gs_key_t);

GSAPI int gs_element_attribute_get_int(const element_t*,
				       const gs_key_t);

GSAPI real_t gs_element_attribute_get_real(const element_t*,
					   const gs_key_t);

GSAPI char *gs_element_attribute_get_string(const element_t*,
					    const gs_key_t);

GSAPI iterator_t *gs_element_attribute_key_iterator_new(const element_t *e);

GSAPI gs_key_t *gs_element_attribute_key_iterator_next(const iterator_t *it);

GSAPI void gs_element_attribute_key_foreach(const element_t *e,
					    key_cb_t callback,
					    void **data);

#endif /* _GS_ELEMENT_H_ */
