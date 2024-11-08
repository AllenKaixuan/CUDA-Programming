NVCC = nvcc

TARGET = hw2_ex2
SRC = hw2_ex2_template.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET) $(arg1) $(arg2) $(arg3)

clean:
	rm -f $(TARGET)

.PHONY: all run clean