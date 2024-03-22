// All kernels are implemented here as template functions.
#pragma once

// for thread_spawns and parallelising for loops
#include<tbb/parallel_for.h>
#include<tbb/blocked_range2d.h>

// for avx instrinsics (used in divide and conquer)
#include<immintrin.h>

#include <iostream>

#define at(M, r, c) M[r*MAT_SIZE + c]

// LVL 1. NAIVE
template<typename T, int MAT_SIZE>
inline void matMul1(T* A, T* B, T* C) {
	for (int m = 0; m < MAT_SIZE; ++m) {
		for (int n = 0; n < MAT_SIZE; ++n) {
			for (int k = 0; k < MAT_SIZE; ++k) {
				at(C, m, n) += at(A, m, k) * at(B, k, n);
			}
		}
	}
}

// LVL 2. LOOP REARAANGE (CACHE AWARE)
template<typename T, int MAT_SIZE>
inline void matMul2(T* A, T* B, T* C) {
	for (int m = 0; m < MAT_SIZE; ++m) 
		for (int k = 0; k < MAT_SIZE; ++k) 
			for (int n = 0; n < MAT_SIZE; ++n) 
				at(C, m, n) += at(A, m, k) * at(B, k, n);
}

// LVL 3. compiler flags -O3 -O2

// LVL 4. PARALLEL LOOPS
template<typename T, int MAT_SIZE>
inline void matMul4(T *A, T *B, T *C){
	
	oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, MAT_SIZE, 1), [&](const oneapi::tbb::blocked_range<int> &r){
		for (int m = r.begin(); m < r.end(); ++m) 
			for (int k = 0; k < MAT_SIZE; ++k) 
				for (int n = 0; n < MAT_SIZE; ++n) 
					at(C, m, n) += at(A, m, k) * at(B, k, n);
	});
}


// // LVL 5. TILED AND PARALLEL LOOPS
template<typename T, int MAT_SIZE, int TILE>
inline void matMul5(T *A, T *B, T *C){
	/*
	we'll tile k loop. k loop goes till TILE SIZE, but loop will run k/TILESIZE times.
	ex->
		for(int i= 0; i< n; ++i){ ... } 
		
		runs for the same number of times as

		for(int i_t= 0; i_t< n; i_t+= TILE_SIZE)
			for(int i= i_t; i< t_t + TILE_SIZE; ++i)

	Tiled loops allow better cache locality
	*/ 

	// Tiling on all loops, parallelising first 2.
	oneapi::tbb::parallel_for(oneapi::tbb::blocked_range2d<int>(0, MAT_SIZE, TILE, 0, MAT_SIZE, TILE), 
		[&](const oneapi::tbb::blocked_range2d<int> &r){
			for(int mt = r.rows().begin(); mt < r.rows().end(); mt += TILE){
			int m_end = std::min( MAT_SIZE, mt + TILE );

			for(int nt = r.cols().begin(); nt < r.cols().end(); nt += TILE){
				int n_end = std::min( MAT_SIZE, nt + TILE);

				for(int kt = 0; kt < MAT_SIZE; kt += TILE){
					int k_end = std::min( MAT_SIZE, kt + TILE );

					for(int m= mt; m< m_end; ++m)
						for(int k= kt; k< k_end; ++k)
							for(int n= nt; n< n_end; ++n)
								at(C, m, n)+= at(A, m, k)*at(B, k, n);

				}
			}
		}
	});

	

}

// https://www.youtube.com/watch?v=eweD5_mV7h4 - for divide and conquer matMul
template< typename T, int MAT_SIZE, int THRESHOLD>
inline void matMul6(T *A, T *B, T *C, int N){
	if(N<= THRESHOLD){
		for (int m = 0; m < THRESHOLD; ++m) {
			for (int k = 0; k < THRESHOLD; ++k) {
				for (int n = 0; n < THRESHOLD; ++n) {
					at(C, m, n) += at(A, m, k) * at(B, k, n);
				}
			}
		}
	}
	else{
		// for MatMul recursive. r, c will be 1 or 0.
		#define infer(M, r, c) (M + (r*MAT_SIZE + c)*N/2)

		oneapi::tbb::task_group grp;
		grp.run([&](){ 
			matMul6<T, MAT_SIZE, THRESHOLD>(infer(A, 0, 0), infer(B, 0, 0), infer(C, 0, 0), N/2);
		});
		grp.run([&](){ 
			matMul6<T, MAT_SIZE, THRESHOLD>(infer(A, 0, 0), infer(B, 0, 1), infer(C, 0, 1), N/2);
		});
		grp.run([&](){
			matMul6<T, MAT_SIZE, THRESHOLD>(infer(A, 1, 0), infer(B, 0, 0), infer(C, 1, 0), N/2);
		});
			matMul6<T, MAT_SIZE, THRESHOLD>(infer(A, 1, 0), infer(B, 0, 1), infer(C, 1, 1), N/2);
		
		grp.wait();

		grp.run([&](){ 
			matMul6<T, MAT_SIZE, THRESHOLD>(infer(A, 0, 1), infer(B, 1, 0), infer(C, 0, 0), N/2);
		});
		grp.run([&](){ 
			matMul6<T, MAT_SIZE, THRESHOLD>(infer(A, 0, 1), infer(B, 1, 1), infer(C, 0, 1), N/2);
		});
		grp.run([&](){ 
			matMul6<T, MAT_SIZE, THRESHOLD>(infer(A, 1, 1), infer(B, 1, 0), infer(C, 1, 0), N/2);
		});
			matMul6<T, MAT_SIZE, THRESHOLD>(infer(A, 1, 1), infer(B, 1, 1), infer(C, 1, 1), N/2);

		grp.wait();
	}	
}

template< typename T, int MAT_SIZE, int THRESHOLD>
inline void matMul8(T *A, T *B, T *C, int N){
	if(N== THRESHOLD){
		for (int m = 0; m < THRESHOLD; ++m) {
			for (int k = 0; k < THRESHOLD; ++k) {
				__m256 a_vec= _mm256_broadcast_ss(&at(A, m, k)); // ss-> single precision floating point. mm-> multimedia extension. broadcast-> fill register with same values
				for (int n = 0; n < THRESHOLD; n+= 16) {

					// n + 0
					__m256 b_vec0= _mm256_load_ps(&at(B, k, n)); // ps-> packed single precision floating point. pd-> double. ph-> half
					__m256 c_vec0= _mm256_load_ps(&at(C, m, n));

					// multiply
					c_vec0= _mm256_fmadd_ps(a_vec, b_vec0, c_vec0);

					_mm256_store_ps(&at(C, m, n), c_vec0);

					// n + 8
					__m256 b_vec1= _mm256_load_ps(&at(B, k, n + 8)); // ps-> packed single precision floating point. pd-> double. ph-> half
					__m256 c_vec1= _mm256_load_ps(&at(C, m, n + 8));

					// multiply 
					c_vec1= _mm256_fmadd_ps(a_vec, b_vec1, c_vec1);

					_mm256_store_ps(&at(C, m, n + 8), c_vec1);
				}
			}
		}
		
	}
	else{
		// for MatMul recursive. r, c will be 1 or 0.
		#define infer(M, r, c) (M + (r*MAT_SIZE + c)*N/2)

		oneapi::tbb::task_group grp;
		grp.run([&](){ 
			matMul8<T, MAT_SIZE, THRESHOLD>(infer(A, 0, 0), infer(B, 0, 0), infer(C, 0, 0), N/2);
		});
		grp.run([&](){ 
			matMul8<T, MAT_SIZE, THRESHOLD>(infer(A, 0, 0), infer(B, 0, 1), infer(C, 0, 1), N/2);
		});
		grp.run([&](){
			matMul8<T, MAT_SIZE, THRESHOLD>(infer(A, 1, 0), infer(B, 0, 0), infer(C, 1, 0), N/2);
		});
			matMul8<T, MAT_SIZE, THRESHOLD>(infer(A, 1, 0), infer(B, 0, 1), infer(C, 1, 1), N/2);
		
		grp.wait();

		grp.run([&](){ 
			matMul8<T, MAT_SIZE, THRESHOLD>(infer(A, 0, 1), infer(B, 1, 0), infer(C, 0, 0), N/2);
		});
		grp.run([&](){ 
			matMul8<T, MAT_SIZE, THRESHOLD>(infer(A, 0, 1), infer(B, 1, 1), infer(C, 0, 1), N/2);
		});
		grp.run([&](){ 
			matMul8<T, MAT_SIZE, THRESHOLD>(infer(A, 1, 1), infer(B, 1, 0), infer(C, 1, 0), N/2);
		});
			matMul8<T, MAT_SIZE, THRESHOLD>(infer(A, 1, 1), infer(B, 1, 1), infer(C, 1, 1), N/2);

		grp.wait();
	}	
}

