#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "Arena.h"
#include "NeuralNetwork.h"
#include "NeuralBot.h"
#include <thrust/device_vector.h>

__device__ thrust::device_vector<float> GetInputs(SNeuralBot b, thrust::device_vector<float> memory, SArena a);
__device__ float viewTypeToDouble(ViewType t);
__device__ BotAction GetBotAction(SNeuralBot b, SNeuralNetwork nnwrk);