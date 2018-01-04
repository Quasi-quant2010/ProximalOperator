CXX = g++
#-std=c++0x
CXXFLAGS = -std=c++11 -Ofast -fopenmp -pthread -I/usr/local/eigen -I/usr/local/gsl/include -I/usr/local/boost/include
LDFLAGS = -L/usr/local/gsl/lib -L/usr/local/boost/lib
LDLIBS = -lgsl -lopenblas -lm
TARGET = ProxSGD ProxSVRG
SRCS_ProxSGD = ProxSGD.cc src/utils.cc src/read_file_sparse.cc src/ProxSGD_sparse.cc
OBJS_ProxSGD = $(SRCS_ProxSGD:.cc=.o)
SRCS_ProxSVRG = ProxSVRG.cc src/utils.cc src/read_file_sparse.cc src/ProxSVRG_sparse.cc
OBJS_ProxSVRG = $(SRCS_ProxSVRG:.cc=.o)

all: $(TARGET)

#$(TARGET): $(OBJS)
#	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)

ProxSGD: $(OBJS_ProxSGD)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)

ProxSVRG: $(OBJS_ProxSVRG)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS_ProxSGD) $(OBJS_ProxSVRG)
