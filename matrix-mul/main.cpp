//
//  main.cpp
//  matrix-mul
//
//  Created by Brandon Liu on 5/13/20.
//  Copyright © 2020 Brandon Liu. All rights reserved.
//  Referenced  What Every Programmer Should Know About Memory by Ulrich Drepper, 2007

#include <iostream>
#include <pmmintrin.h>
#include <mkl.h>
#include <iomanip>
#include <chrono>
using namespace std;
#define SIZE 4
/*int16_t mul1[SIZE][SIZE] = {
    {10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {14, 15, 16, 17, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {20, 2,  5,   8, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {8,  18, 14,  6, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {14, 15, 16, 17, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {20, 2,  5,   8, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {8,  18, 14,  6, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 18, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {14, 15, 16, 17, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {20, 2,  5,   8, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 17,},
    {8,  18, 14,  6, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 16, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {14, 15, 16, 17, 10, 11, 12, 13, 10,  6, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {20, 2,  5,   8, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10,  3, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {8,  18, 14,  6, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {14, 15, 16, 17,  1, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {20, 2,  5,   8, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12,  5, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {8,  18, 14,  6, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,  0, 11, 12, 13, 10, 14, 12, 13, 10, 11, 12, 13,},
    {10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {14, 15, 16, 17, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {20, 2,  5,   8, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {8,  18, 14,  6, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 19,},
    {10, 11, 12, 13, 10,  9, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {14, 15, 16, 17, 10, 11, 12, 13,  2, 11, 12, 13, 10,  6, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10,  8, 12, 13,},
    {20, 2,  5,   8, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {8,  18, 14,  6, 10, 11, 10, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 21, 13,},
    {10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {14, 15, 16, 17, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {20, 2,  5,   8, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
    {8,  18, 14,  6, 10, 11, 12,  6, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,},
};*/

int16_t mul1[SIZE][SIZE] = {
    {10, 11, 12, 13},
    {14, 15, 16, 17},
    {20, 2,  5,   8},
    {8,  18, 14,  6}
};

// Straight-forward implementation
void matmulNaive(int16_t mul1[SIZE][SIZE], int16_t mul2[SIZE][SIZE], int16_t res[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            for (int k = 0; k < SIZE; ++k) {
                res[i][j] += mul1[i][k] * mul2[k][j];
            }
        }
    }
}

// Transposed mul2 to be row major, should be faster
void matmulTransposed(int16_t mul1[SIZE][SIZE], int16_t mul2[SIZE][SIZE], int16_t res[SIZE][SIZE]) {
    int16_t tmp[SIZE][SIZE];
    for(int i = 0; i < SIZE; i++) {
        for(int j = 0; j < SIZE; j++) {
            tmp[i][j] = mul2[j][i]; // transpose mul2
        }
    }
    for(int i = 0; i < SIZE; i++) {
        for(int j = 0; j < SIZE; j++) {
            for(int k = 0; k < SIZE; k++) {
                res[i][j] += mul1[i][k] * tmp[j][k];
            }
        }
    }
}

void matmulBlocked(int16_t mul1[SIZE][SIZE], int16_t mul2[SIZE][SIZE], int16_t res[SIZE][SIZE]) {
    int16_t* rres;
    int16_t* rmul1;
    int16_t* rmul2;
    // if my cache line size if 64 bytes, and in16_t is 2 bytes, then stride length is 64/2 = 32
    int stride = 32;
    for(int i = 0; i < SIZE; i += stride) {
        for(int j = 0; j < SIZE; j += stride) {
            for(int k = 0; k < SIZE; k += stride) {
                rres = &res[i][j];
                rmul1 = &mul1[i][k];
                for(int i2 = 0; i2 < stride; i2++, rres+=SIZE, rmul1+=SIZE) {
                    rmul2 = &mul2[k][j];
                    for(int k2 = 0; k2 < stride; k2++, rmul2 +=SIZE) {
                        for(int j2 = 0; j2 < stride; j2++) {
                            rres[j2] += rmul1[k2] * rmul2[j2];
                        }
                    }
                }
            }
        }
    }
}

void printMatrix(int16_t m[SIZE][SIZE]) {
    for(int i = 0; i < SIZE; i++) {
        for(int j = 0; j < SIZE; j++) {
            printf("%d ", m[i][j]);
        }
        printf("\n");
    }
}

// Timer functions
chrono::time_point<chrono::high_resolution_clock> start, stop;
void startTimer() {
    start = chrono::high_resolution_clock::now();
}
void stopTimer() {
    stop = chrono::high_resolution_clock::now();
}
void printTime() {
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(stop - start).count();
    time_taken *= 1e-9;
    cout << "Time taken is: " << fixed << time_taken << setprecision(9);
    cout << " seconds" << endl;
}

int main(int argc, const char * argv[]) {
    int16_t res[SIZE][SIZE];
    int nruns = 10000;
    
    // Naive matrix multiply
    startTimer();
    for(int i = 0; i < nruns; i++) {
        memset(res, 0, SIZE*SIZE*2); // memset to reuse res
        matmulNaive(mul1, mul1, res);
    }
    stopTimer();
    printMatrix(res);
    printTime();
    
    // Transposed matrix multiply
    startTimer();
    for(int i = 0; i < nruns; i++) {
        memset(res, 0, SIZE*SIZE*2); // memset to reuse res
        matmulTransposed(mul1, mul1, res);
    }
    stopTimer();
    printMatrix(res);
    printTime();


    // Blocked matrix multiply
    startTimer();
    for(int i = 0; i < nruns; i++) {
        memset(res, 0, SIZE*SIZE*2); // memset to reuse res
        matmulBlocked(mul1, mul1, res);
    }
    stopTimer();
    printMatrix(res);
    printTime();
    return 0;
}

