#ifndef _GS_ELEMENT_H_
#define _GS_ELEMENT_H_

#include "gs_types.h"
#include "gs_common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct _element {
  GSObject    parent;
  gsid        id;
  GHashTable *attributes;
};

#define GS_ELEMENT(e) ((GSElement*)CHECK_TYPE(e,ELEMENT_TYPE))

GSAPI void gs_element_init(GSElement *element,
			   gsid       id);

GSAPI void gs_element_finalize(GSElement *element);

GSAPI gsid gs_element_id_get(const GSElement *element);

GSAPI void gs_element_attribute_add(const GSElement *element,
				    const gskey      key,
				    void            *data);

GSAPI void gs_element_attribute_add_int(const GSElement *element,
					const gskey      key,
					int              data);

GSAPI void gs_element_attribute_add_real(const GSElement *element,
					 const gskey      key,
					 gsreal           data);

GSAPI void gs_element_attribute_add_string(const GSElement *element,
					   const gskey      key,
					   char*);

GSAPI void *gs_element_attribute_change(const GSElement *element,
					const gskey      key,
					void            *data);

GSAPI int gs_element_attribute_change_int(const GSElement *element,
					  const gskey      key,
					  int              data);

GSAPI gsreal gs_element_attribute_change_real(const GSElement *element,
					      const gskey      key,
					      gsreal           data);

GSAPI char *gs_element_attribute_change_string(const GSElement *element,
					       const gskey      key,
					       char            *data);

GSAPI void gs_element_attribute_delete(const GSElement *element,
				       const gskey      key);

GSAPI void *gs_element_attribute_get(const GSElement *element,
				     const gskey      key);

GSAPI int gs_element_attribute_get_int(const GSElement *element,
				       const gskey      key);

GSAPI gsreal gs_element_attribute_get_real(const GSElement *element,
					   const gskey      key);

GSAPI char *gs_element_attribute_get_string(const GSElement *element,
					    const gskey      key);

GSAPI GSIterator *gs_element_attribute_key_iterator_new(const GSElement *element);

GSAPI gskey *gs_element_attribute_key_iterator_next(const GSIterator *it);

GSAPI void gs_element_attribute_key_foreach(const GSElement *element,
					    GSKeyCB          callback,
					    void           **data);

#ifdef __cplusplus
}
#endif

#endif /* _GS_ELEMENT_H_ */
