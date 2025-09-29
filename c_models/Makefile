CC = gcc
CFLAGS = -O2 -std=c99 -Wno-unused-variable -Wno-unused-const-variable
TARGET = test_yolov8
SOURCES = test.c yolov8_regular.c

# Default target
all: $(TARGET)

# Build the test executable
$(TARGET): $(SOURCES)
	@echo "Compiling YOLOv8 quantized model test..."
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) -lm
	@echo "Build completed: $(TARGET)"

# Run the test
run: $(TARGET)
	@echo "Running YOLOv8 quantized model test..."
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)
	@echo "Cleaned build artifacts"

# Quick test (compile and run)
test: clean $(TARGET) run

# Show file sizes
info:
	@echo "=== Model Files Info ==="
	@ls -lh *.c
	@echo ""
	@echo "=== Executable Info ==="
	@if [ -f $(TARGET) ]; then ls -lh $(TARGET); else echo "$(TARGET) not built yet"; fi

.PHONY: all run clean test info