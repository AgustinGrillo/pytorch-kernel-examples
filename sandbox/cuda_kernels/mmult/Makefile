test: test.cu mmult.cu mmult.hpp
	nvcc test.cu mmult.cu -o test


MKLROOT=/opt/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH=${MKLROOT}/lib/intel64:$LD_LIBRARY_PATH
mkl: test.cu mmult.cu mmult.hpp
	# nvcc -DMKL -I${MKLROOT}/include test.cu mmult.cu -o test --lib ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a --lib ${MKLROOT}/lib/intel64/libmkl_sequential.a --lib ${MKLROOT}/lib/intel64/libmkl_core.a -lpthread -lm -ldl 
	nvcc -DMKL -I${MKLROOT}/include test.cu mmult.cu -o test -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

mkl_run: mkl
	./test

benchmark: benchmark.cu mmult.cu mmult.hpp
	# nvcc -DMKL -O3 --use_fast_math -Xcompiler -I${MKLROOT}/include benchmark.cu mmult.cu -o benchmark -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lcublas
	nvcc -DMKL -O3 --use_fast_math -Xcompiler -I${MKLROOT}/include benchmark.cu mmult.cu -o benchmark -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -lcublas
	# nvcc -O3 --use_fast_math -Xcompiler -fopenmp matmul_forward.cu -o matmul_forward -lcublas -lcublasLt

benchmark_run: benchmark
	./benchmark

bench_mkl: benchmark_mkl.cpp
	# https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html#gs.afg5u1
	 g++ -g -march=native -O3 -std=c++20 benchmark_mkl.cpp -o benchmark_mkl -m64 -I${MKLROOT}/include -Wl,--start-group ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_gnu_thread.a ${MKLROOT}/lib/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl
	 # g++ -g -march=native -O3 -std=c++20 benchmark_mkl.cpp -o benchmark_mkl -m64 -I${MKLROOT}/include -Wl,--start-group ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_intel_thread.a ${MKLROOT}/lib/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

clean:
	rm -f test benchmark


