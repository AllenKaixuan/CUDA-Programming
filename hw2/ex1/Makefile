NVCC = nvcc

TARGET = hw2_ex1
SRC = hw2_ex1_template.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC)
# arg is the inputLength
run: $(TARGET)
	./$(TARGET) $(arg)		

clean:
	rm -f $(TARGET)

.PHONY: all run clean