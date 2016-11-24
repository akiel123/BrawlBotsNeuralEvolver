#pragma once
#include "NeuralNetwork.h"
#include <curand.h>
#include "Arena.h"

#define nTrivial 3
#define nObs 4
#define nDetailsObs 8
#define nMemory 10
#define VIEWTYPE_AMOUNT 5

struct DBotAction {
	float acceleration;
	float rotation;
	int shoot;
	__device__ DBotAction(float acc, float rot, int shooting) {
		acceleration = acc;
		rotation = rot;
		shoot = shooting;
	}
};
__device__ void BotUpdate(DNeuralBot b, DNeuralNetwork nnwrk, DArena a);
__device__ float GetVelX(DNeuralBot b);
__device__ float GetVelY(DNeuralBot b);
__device__ int BotCollides(DNeuralBot bo, DBullet bu);
__device__ int BotCollides(DNeuralBot bo, DAmmo a);
__device__ float pointDistance(float x, float y, float x1, float y1, float x2, float y2);