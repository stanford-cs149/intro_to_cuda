EXECUTABLE := error

ARCH=$(shell uname | sed -e 's/-.*//g')

NVCC=nvcc
NVCCFLAGS=-O3 #-m64 

.PHONY: clean run

default: $(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE)

run: $(EXECUTABLE)
	./$(EXECUTABLE)

$(EXECUTABLE): error.cu
		$(NVCC) $(NVCCFLAGS) $^ -o $@

