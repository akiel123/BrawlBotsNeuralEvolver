#pragma once
#include "NeuralNetwork.h"
#include <curand.h>
#include "Arena.h"

#define nTrivial 3
#define nObs 4
#define nDetailsObs 8
#define nMemory 10
#define VIEWTYPE_AMOUNT 5

struct BotAction {
	float acceleration;
	float rotation;
	int shoot;
	BotAction(float acc, float rot, int shooting) {
		acceleration = acc;
		rotation = rot;
		shoot = shooting;
	}
};
__device__ void BotUpdate(SNeuralBot b, SNeuralNetwork nnwrk, SArena a);
__device__ float GetVelX(SNeuralBot b);
__device__ float GetVelY(SNeuralBot b);
__device__ int BotCollides(SNeuralBot bo, SBullet bu);
__device__ int BotCollides(SNeuralBot bo, SAmmo a);
__device__ float pointDistance(float x, float y, float x1, float y1, float x2, float y2);