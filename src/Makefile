TARGET = sharpen
CXX = nvcc
CXXFLAGS = -O2

all:
	$(CXX) main.cpp cpu_sharpen.cpp cuda_sharpen_naive.cu cuda_sharpen_shared.cu -o $(TARGET)

clean:
	rm -f $(TARGET)
