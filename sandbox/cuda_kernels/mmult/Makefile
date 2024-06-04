test: test.cu mmult.cu mmult.hpp
	nvcc test.cu mmult.cu -o test


MKL_ROOT=/opt/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH=${MKL_ROOT}/lib/intel64:$LD_LIBRARY_PATH
mkl: test.cu mmult.cu mmult.hpp
	# nvcc -DMKL -I${MKL_ROOT}/include test.cu mmult.cu -o test --lib ${MKL_ROOT}/lib/intel64/libmkl_intel_ilp64.a --lib ${MKL_ROOT}/lib/intel64/libmkl_sequential.a --lib ${MKL_ROOT}/lib/intel64/libmkl_core.a -lpthread -lm -ldl 
	nvcc -DMKL -I${MKL_ROOT}/include test.cu mmult.cu -o test -L${MKL_ROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

mkl_run: mkl
	./test

benchmark: benchmark.cu mmult.cu mmult.hpp
	nvcc -DMKL -O3 --use_fast_math -Xcompiler -I${MKL_ROOT}/include benchmark.cu mmult.cu -o benchmark -L${MKL_ROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lcublas
	# nvcc -O3 --use_fast_math -Xcompiler -fopenmp matmul_forward.cu -o matmul_forward -lcublas -lcublasLt

benchmark_run: benchmark
	./benchmark

clean:
	rm -f test benchmark


