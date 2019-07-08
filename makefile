
SOURCES= WaveEquation.cpp \
		 ParallelRoutines.cpp \
		 MatrixHelper.cpp \

DEPS = WaveEquation.h \
	   ParallelRoutines.h \
       MatrixHelper.h


%.o: %.cpp 
	mpic++ -c -o $@ $< -I.

paralaaompi : $(SOURCES)
	mpic++ -std=c++11 -stdlib=libc++ -o paralaaompi $(SOURCES) -I.

