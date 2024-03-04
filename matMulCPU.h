// matMulCPU.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include<tbb/parallel_for.h>
#include<tbb/blocked_range2d.h>
#include<immintrin.h>

// ye matMul recursive k lie hai. likh ke samajh. r aur c ki value -> 0 ya 1 hi hogi
#define infer(M, r, c) (M + (r*MAT_SIZE + c)*N/2)
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

	for (int m = 0; m < MAT_SIZE; ++m) {
		for (int k = 0; k < MAT_SIZE; ++k) {
			for (int n = 0; n < MAT_SIZE; ++n) {
				at(C, m, n) += at(A, m, k) * at(B, k, n);
			}
		}
	}
}

// LVL 3. optimisation is compiler flags laga -O3 -O2

// LVL 4. PARALLEL LOOPS
template<typename T, int MAT_SIZE>
inline void matMul4(T *A, T *B, T *C){
	
	oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, MAT_SIZE, 1), [&](const oneapi::tbb::blocked_range<int> &r){
		
		for (int m = r.begin(); m < r.end(); ++m) {
			for (int k = 0; k < MAT_SIZE; ++k) {
				for (int n = 0; n < MAT_SIZE; ++n) {
					at(C, m, n) += at(A, m, k) * at(B, k, n);
				}
			}
		}
	});
}


// // LVL 5. TILED AND PARALLEL LOOPS
template<typename T, int MAT_SIZE, int TILE>
inline void matMul5(T *A, T *B, T *C){
	// k ko tile krunga. k ki ek loop m TILE SIZE tk hi jayenge, lekin k/TILESIZE times loop chalaenge. code samajh
	//ex->
		//for(int i= 0; i< n; ++i){ ... } 
		
				// runs for the same number of times as

					// for(int i_t= 0; i_t< n; i_t+= TILE_SIZE)
						//for(int i= i_t; i< t_t + TILE_SIZE; ++i)

						// code likhne ke kayi tarike hoskte hain. point wo nhi h. ek loop chale de 0->tile_size aur indexing m i_t + i krde. wo bhi sahi h 

	// samajh nhi aya to fir se padh

	
	// saari loop tile krke, aur upar ki 2 parallelise krunga
	oneapi::tbb::parallel_for(oneapi::tbb::blocked_range2d<int>(0, MAT_SIZE, TILE, 0, MAT_SIZE, TILE), 
		[&](const oneapi::tbb::blocked_range2d<int> &r){
			for(int mt = r.rows().begin(); mt < r.rows().end(); mt += TILE){
			int m_end = std::min( MAT_SIZE, mt + TILE );

			for(int nt = r.cols().begin(); nt < r.cols().end(); nt += TILE){
				int n_end = std::min( MAT_SIZE, nt + TILE);

				for(int kt = 0; kt < MAT_SIZE; kt += TILE){
					int k_end = std::min( MAT_SIZE, kt + TILE );

					for(int m= mt; m< m_end; ++m){
						for(int k= kt; k< k_end; ++k){
							for(int n= nt; n< n_end; ++n){
								at(C, m, n)+= at(A, m, k)*at(B, k, n);
							}
						}
					}

				}
			}
		}
	});

	

}

// https://www.youtube.com/watch?v=eweD5_mV7h4 - yaha s smjh recursion
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
			// ise parallise nhi kia. is thread me bhi to kuchh chale. basic optimisation
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
			// ise parallise nhi kia. is thread me bhi to kuchh chale. basic optimisation
			matMul6<T, MAT_SIZE, THRESHOLD>(infer(A, 1, 1), infer(B, 1, 1), infer(C, 1, 1), N/2);

		grp.wait();
	}	
}

template< typename T, int MAT_SIZE, int THRESHOLD>
inline void matMul8(T *A, T *B, T *C, int N){
	if(N== THRESHOLD){
		for (int m = 0; m < THRESHOLD; ++m) {
			for (int k = 0; k < THRESHOLD; ++k) {
				__m256 a_vec= _mm256_broadcast_ss(&at(A, m, k)); // ss-> single precision floating point. mm-> multimedia extension. broadcast-> pure register m same value bhar de
				for (int n = 0; n < THRESHOLD; n+= 16) {

					// n + 0
					__m256 b_vec0= _mm256_load_ps(&at(B, k, n)); // ps-> packed single precision floating point. pd-> double. ph-> half
					__m256 c_vec0= _mm256_load_ps(&at(C, m, n));

					// multiply krde
					c_vec0= _mm256_fmadd_ps(a_vec, b_vec0, c_vec0);

					_mm256_store_ps(&at(C, m, n), c_vec0);

					// n + 8
					__m256 b_vec1= _mm256_load_ps(&at(B, k, n + 8)); // ps-> packed single precision floating point. pd-> double. ph-> half
					__m256 c_vec1= _mm256_load_ps(&at(C, m, n + 8));

					// multiply krde
					c_vec1= _mm256_fmadd_ps(a_vec, b_vec1, c_vec1);

					_mm256_store_ps(&at(C, m, n + 8), c_vec1);
				}
			}
		}
		
	}
	else{
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
			// ise parallise nhi kia. is thread me bhi to kuchh chale. basic optimisation
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
			// ise parallise nhi kia. is thread me bhi to kuchh chale. basic optimisation
			matMul8<T, MAT_SIZE, THRESHOLD>(infer(A, 1, 1), infer(B, 1, 1), infer(C, 1, 1), N/2);

		grp.wait();
	}	
}

