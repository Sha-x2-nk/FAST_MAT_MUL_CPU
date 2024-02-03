// matMulCPU.cpp : Defines the entry point for the application.
//

#include "matMulCPU.h"
#include<chrono>
#include<iostream>
#define A_VAL 2.0f
#define B_VAL 3.0f

using namespace std;

template< typename T, int MAT_SIZE>
inline void initMat(T* A, T val) {
	for (int i = 0; i < MAT_SIZE * MAT_SIZE; ++i)
		A[i] = val;
}

template< typename T, int MAT_SIZE>
inline bool checkC(T* C) {
	const T C_VAL = A_VAL * B_VAL * MAT_SIZE;
	
	for (int i = 0; i < MAT_SIZE * MAT_SIZE; ++i)
		if (C[i] != C_VAL)		return false;

	return true;
}

template<typename T, int MAT_SIZE>
void benchTimes(T *A, T *B, T *C, int idx, int iters){
	std::chrono::high_resolution_clock::time_point t0, t1;
	std::chrono::duration< double > fs;
	std::chrono::microseconds d; 
	std::chrono::duration< double > totfs= static_cast<std::chrono::duration< double >>(0);
	for(int i= 0; i< iters; ++i){
		
		switch(idx){
			case 0: t0 = std::chrono::high_resolution_clock::now(); matMul1<float, MAT_SIZE>(A, B, C); t1 = std::chrono::high_resolution_clock::now(); break;
			case 1: t0 = std::chrono::high_resolution_clock::now(); matMul2<float, MAT_SIZE>(A, B, C); t1 = std::chrono::high_resolution_clock::now(); break;
			case 2: t0 = std::chrono::high_resolution_clock::now(); matMul4<float, MAT_SIZE>(A, B, C); t1 = std::chrono::high_resolution_clock::now(); break;
			case 3: t0 = std::chrono::high_resolution_clock::now(); matMul5<float, MAT_SIZE, 64>(A, B, C); t1 = std::chrono::high_resolution_clock::now(); break; // 64 best aya
			case 4: t0 = std::chrono::high_resolution_clock::now(); matMul6<float, MAT_SIZE, 64>(A, B, C, MAT_SIZE); t1 = std::chrono::high_resolution_clock::now(); break; // 64 best aya
			case 5: t0 = std::chrono::high_resolution_clock::now(); matMul7<float, MAT_SIZE, 64>(A, B, C, MAT_SIZE); t1 = std::chrono::high_resolution_clock::now(); break; // 64 best aya
			case 6: t0 = std::chrono::high_resolution_clock::now(); matMul641<MAT_SIZE>(A, B, C); t1 = std::chrono::high_resolution_clock::now(); break;
		}
		
		fs = t1 - t0;
		totfs+= fs;
		// result checking
		if( checkC<float, MAT_SIZE>(C)== false ){ printf("\nINCORRECT RESULT."); return; }
		initMat<float, MAT_SIZE>(C, 0); // reset kia 0. kyunki C += ho rha h. nhi kia to wrong ans ayega
	}
	printf("\nRESULTS CORRECT. ");
	totfs/= iters;
	d = std::chrono::duration_cast<std::chrono::microseconds>(totfs);
	std::cout << (d.count())/1000000.0 << "s";
}

int main(int argc, char *args[])
{
	int bench_idx= std::stoi(args[1]);


	const int MAT_SIZE = 4096; 

	printf("\nINITIALISING ARRAYS.. ");
	auto t0 = std::chrono::high_resolution_clock::now();
	float* A = (float*)malloc(MAT_SIZE * MAT_SIZE * sizeof(float)); initMat<float, MAT_SIZE>(A, A_VAL);
	float* B = (float*)malloc(MAT_SIZE * MAT_SIZE * sizeof(float)); initMat<float, MAT_SIZE>(B, B_VAL);
	float* C = (float*)malloc(MAT_SIZE * MAT_SIZE * sizeof(float)); initMat<float, MAT_SIZE>(C, 0);
	auto t1 = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration< double > fs = t1 - t0;
	std::chrono::milliseconds d = std::chrono::duration_cast<std::chrono::milliseconds>(fs);
	printf("DONE. ");
	std::cout << d.count() << "ms\n";

	benchTimes<float, MAT_SIZE>(A, B, C, bench_idx, 1); // matMul naive
	return 0;

}
