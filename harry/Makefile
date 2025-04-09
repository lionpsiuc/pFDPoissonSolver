CC = mpicc

CFLAGS = -g -Wall #-D_CC_OVERLAP

LDFLAGS = -lm

POISSOBJS1D = jacobi_1d.o

POISSOBJS2D = jacobi2d.o

EXECS = poiss1d poiss2d

all: $(EXECS)

poiss1d: main_1d.o $(POISSOBJS1D)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

poiss2d: main_2d.o $(POISSOBJS2D)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tags:
	etags *.c *.h

.PHONY: clean tags tests

clean:
	$(RM) *.o $(EXECS) TAGS tags
