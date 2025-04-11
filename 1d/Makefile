CC = mpicc

CFLAGS = -O3 -Wall -Wextra

LDFLAGS = -lm

POISSOBJS1D = aux.o decomp1d.o gatherwrite.o jacobi.o

EXECS = poiss1d

all: $(EXECS)

poiss1d: main.o $(POISSOBJS1D)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) *.o $(EXECS) *.png *.txt
