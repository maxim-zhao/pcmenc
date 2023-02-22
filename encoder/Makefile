# To build you will need to install:
# make
# g++
# libtbb-dev

CFLAGS ?= -Ofast -Wall -Wextra -Werror
CXXFLAGS ?= -std=c++17 -Ofast -Wall -Wextra -Werror

pcmenc: pcmenc.o resample.o FileReader.o Args.o
	g++ $(CXXFLAGS) $^ -o $@ -ltbb


clean:
	rm -f *.o pcmenc
