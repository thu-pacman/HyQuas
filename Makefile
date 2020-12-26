CC := mpicc
CXX := mpicxx
CUDA := nvcc
GENCODE_SM70 := -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS := $(GENCODE_SM70)

CUTT_DIR := third-party/cutt/cutt
CXX_FLAGS := -std=c++14 -O2 -g -Wall -I$(CUTT_DIR)/include -Isrc -I$(CUDA_HOME)/include
CUDA_FLAGS := -Xcompiler -std=c++14 -O2 -g -I$(CUTT_DIR)/include -Isrc -I$(CUDA_HOME)/include
CUDA_LFLAGS := -lmpi -lcudart -L$(CUTT_DIR)/lib -lcutt -lm
HEADERS := $(wildcard src/*.h)
SOURCES_CXX := $(wildcard src/*.cpp)
SOURCES_CUDA := $(wildcard src/*.cu)
OBJECTS := $(SOURCES_CXX:src/%.cpp=build/%.o) $(SOURCES_CUDA:src/%.cu=build/%.o)
DEFS := -DBACKEND=1 -DSHOW_SCHEDULE -DSHOW_SUMMARY -DDISABLE_ASSERT

all: build/main

build/main: $(CUTT_DIR)/lib/libcutt.a main.cpp $(OBJECTS)
	$(CXX) -o build/main main.cpp $(OBJECTS) $(CXX_FLAGS) $(CUDA_LFLAGS) $(DEFS)

build/%.o: src/%.cpp src/*.h
	$(CXX) $< -c $(CXX_FLAGS) $(DEFS) -o build/$*.o 

build/%.o: src/%.cu src/*.h
	$(CUDA) $< -c $(CUDA_FLAGS) $(DEFS) -o build/$*.o

.PHONY: clean
clean:
	rm -f build/*.o build/main
