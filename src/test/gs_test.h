#ifndef _GS_TEST_H_
#define _GS_TEST_H_

#define BEGIN(what)					\
  do {							\
    fprintf(stdout, " * %-40s", what);			\
    fflush(stdout);					\
  } while(0)

#define DONE					\
  do {						\
    fprintf(stdout, "[done]\n");		\
    fflush(stdout);				\
  } while(0)

#define FAILED					\
  do {						\
    fprintf(stdout, "[failed]\n");		\
    fflush(stdout);				\
    exit(1);					\
  } while(0)

#endif /* _GS_TEST_H_ */
