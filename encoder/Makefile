# #include <execution> requires tbb/blocked_range.h which is provided by libtbb-dev

CFLAGS ?= -O2 -Wall
CXXFLAGS ?= -std=c++17 -O2 -Wall

#LDFLAGS = -ltbb

pcmenc: pcmenc.o resample.o FileReader.o Args.o
	g++ $(CXXFLAGS) $? -o $@ -ltbb


clean:
	rm -f *.o pcmenc
