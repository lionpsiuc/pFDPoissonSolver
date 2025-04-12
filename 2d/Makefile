CC = mpicc

CFLAGS = -I./include -O3 -Wall -Wextra

LDFLAGS = -lm

SRCDIR = src
BUILDDIR = build
BINDIR = bin

# Create object files from source files
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))

EXECS = $(BINDIR)/poiss1d

# Make sure the build and bin directories exist
$(shell mkdir -p $(BUILDDIR) $(BINDIR))

all: $(EXECS)

# Rule to compile the executable
$(BINDIR)/poiss1d: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to compile object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to run the application with four processors
run4: $(EXECS)
	mpirun -np 4 $(BINDIR)/poiss1d

# Rule to run the application with 16 processors
run16: $(EXECS)
	mpirun -np 16 $(BINDIR)/poiss1d

# Generate a heatmap from the output files
heatmap:
	gnuplot scripts/heatmap.gp

.PHONY: clean run heatmap
clean:
	$(RM) -r $(BUILDDIR)/* $(BINDIR)/* *.png *.txt
