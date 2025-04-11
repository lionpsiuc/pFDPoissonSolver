CC = mpicc

CFLAGS = -O3 -Wall -Wextra -I./include

LDFLAGS = -lm

SRCDIR = src
BUILDDIR = build
BINDIR = bin

# Create object file names from source files
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))

EXECS = $(BINDIR)/poiss1d

# Make sure the build and bin directories exist
$(shell mkdir -p $(BUILDDIR) $(BINDIR))

all: $(EXECS)

# Rule to compile the executable
$(BINDIR)/poiss1d: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to compile object files from C files
$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to run the application with 4 processors
run: $(EXECS)
	mpirun -np 4 $(BINDIR)/poiss1d

# Generate a heatmap from the output files
heatmap:
	gnuplot scripts/heatmap.gp

.PHONY: clean run heatmap
clean:
	$(RM) -r $(BUILDDIR)/* $(BINDIR)/* *.png *.txt
