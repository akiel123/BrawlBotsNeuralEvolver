#pragma once
#include "bot.h"
#define MAX_STEPS 20000;

//Not used
struct DBattle {
	DNeuralBot b1 = DNeuralBot(&DNeuralNetwork());
	DNeuralBot b2 = DNeuralBot(&DNeuralNetwork());
	int steps = MAX_STEPS;
	int step = 0;

	__device__ DBattle(DNeuralBot bot1, DNeuralBot bot2) {
		b1 = bot1;
		b2 = bot2;
	}
};

__global__ void OneOnOneBattles(float* pw, float* pt,
	DNeuralNetwork* nnmodel, int* weightCount, int* networkCount,
	int* battleCount, float* resultsPersonal, float* resultsLegacy,
	int* nnIDs);
__device__ int* battle(DNeuralNetwork nncontender, DNeuralNetwork nnopponent);
__device__ float* getPersonalFitness(float* pf, int ID, int count, int* ids);
__device__ float* getLegacyFitness(float* lf, int ID, int count, int* ids);
