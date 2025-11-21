# Makefile

# Compiler
HIPCC = hipcc

# Source and target
SRC = fused_bucketized_test.hip
TARGET = applications_fused_bucketized

# Compiler flags
CFLAGS = -O3

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CFLAGS) -o $@ $<

# Clean rule
clean:
	rm -f $(TARGET)


