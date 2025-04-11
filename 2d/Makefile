CC = mpicc

CFLAGS = -Wall -Wextra

LDFLAGS = -lm

POISSOBJS2D = aux.o decomp2d.o gatherwrite.o jacobi.o

EXECS = poiss2d

all: $(EXECS)

poiss2d: main.o $(POISSOBJS2D)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) *.o $(EXECS) *.png *.txt
