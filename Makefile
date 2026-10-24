OBJS = neuron.o layer.o neuralnet.o activation.o main.o
TEST_OBJS = neuron.o layer.o neuralnet.o activation.o tests/test_nn.o
CC = g++
AR = ar
DIRECTORY = nnl
INCLUDE = /usr/include/
LIB = /usr/lib/
LIBNAME = libnn.so
ARNAME = libnn.a
OPTS = -fPIC -Wall -O2


all: clean $(OBJS)
	$(CC) -o demo $(OBJS) -Wall -O2
	$(CC) -shared $(OPTS) -o $(LIBNAME) neuron.o layer.o neuralnet.o activation.o 
	$(AR) cr $(ARNAME) neuron.o layer.o neuralnet.o activation.o 
	
main.o:
	$(CC) -c main.cpp -Wall -O2

neuron.o:
	$(CC) $(OPTS) -c neuron.cpp

layer.o:
	$(CC) $(OPTS) -c layer.cpp

neuralnet.o:
	$(CC) $(OPTS) -c neuralnet.cpp

activation.o:
	$(CC) $(OPTS) -c activation.cpp

tests/test_nn.o:
	$(CC) -c tests/test_nn.cpp -Wall -O2 -I. -o tests/test_nn.o

test_nn: $(TEST_OBJS)
	$(CC) -o test_nn $(TEST_OBJS) -Wall -O2

test: test_nn
	./test_nn

clean:
	rm -f *.o *.so *.a tests/test_nn.o test_nn

install:
	mkdir $(INCLUDE)$(DIRECTORY)
	cp *.h $(INCLUDE)$(DIRECTORY)
	cp $(LIBNAME) $(LIB)
	cp $(ARNAME) $(LIB)
