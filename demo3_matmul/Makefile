EXECUTABLE := matmul

ARCH=$(shell uname | sed -e 's/-.*//g')

NVCC=nvcc
NVCCFLAGS=-O3 #-m64 

.PHONY: clean run

default: $(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE)

run: $(EXECUTABLE)
	./$(EXECUTABLE)

$(EXECUTABLE): matmul.cu
		$(NVCC) $(NVCCFLAGS) $^ -o $@

