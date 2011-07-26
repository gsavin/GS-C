#ifndef _ERROR_H_
#define _ERROR_H_

#define GS_ERROR_UNKNOWN 0x000000001
#define GS_ERROR_UNKNOWN_ATTRIBUTE 0x000000002
#define GS_ERROR_INVALID_TYPE 0x00000003
#define GS_ERROR_ID_ALREADY_IN_USE 0x00000004
#define GS_ERROR_NODE_NOT_FOUND 0x00000005
#define GS_ERROR_EDGE_NOT_FOUND 0x00000006
#define GS_ERROR_CAN_NOT_ADD_SINK 0x00000007
#define GS_ERROR_CAN_NOT_DELETE_SINK 0x00000008
#define GS_ERROR_IO 0x00000009
#define GS_ERROR_BUFFER_ERROR 0x00000010
#define GS_ERROR_INVALID_ID_FORMAT 0x00000011
#define GS_ERROR_NULL_POINTER 0x00000012

#define ERROR(e)					\
  do {							\
    g_error("#%d", e);				\
    exit(e);						\
  } while(0)

#define ERROR_ERRNO(e)				\
  do {						\
    g_error("%s", strerror(errno));	\
    exit(e);					\
  } while(0)

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>

#endif /* _ERROR_H_ */
