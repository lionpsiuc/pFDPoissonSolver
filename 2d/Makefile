CC = mpicc

CFLAGS = -I./include -O3 -Wall -Wextra

LDFLAGS = -lm

SRCDIR = src
BUILDDIR = build
BINDIR = bin

# Create object files from source files
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))

EXECS = $(BINDIR)/poiss2d

# Make sure the build and bin directories exist
$(shell mkdir -p $(BUILDDIR) $(BINDIR))

all: $(EXECS)

# Rule to compile the executable
$(BINDIR)/poiss2d: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to compile object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to run the application with four processors and a grid size of 31
run4: $(EXECS)
	mpirun -np 4 $(BINDIR)/poiss2d

# Rule to run the application with 16 processors and a grid size of 31
run16: $(EXECS)
	mpirun -np 16 $(BINDIR)/poiss2d

# Generate a heatmap from the output files
heatmap:
	gnuplot scripts/heatmap.gp

.PHONY: clean run heatmap
clean:
	$(RM) -r $(BUILDDIR)/* $(BINDIR)/*
