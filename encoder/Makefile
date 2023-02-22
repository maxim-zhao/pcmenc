# To build you will need to install:
# make
# g++
# libtbb-dev

CFLAGS ?= -Ofast -Wall
CXXFLAGS ?= -std=c++17 -Ofast -Wall

pcmenc: pcmenc.o resample.o FileReader.o Args.o
	g++ $(CXXFLAGS) $^ -o $@ -ltbb


clean:
	rm -f *.o pcmenc
