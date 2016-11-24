#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "Arena.h"
#include "NeuralNetwork.h"
#include "NeuralBot.h"
#include <thrust/device_vector.h>

__device__ float* GetInputs(DNeuralBot b, float* memory, DArena a);
__device__ float viewTypeToDouble(ViewType t);
__device__ BotAction GetBotAction(DNeuralBot b, DNeuralNetwork nnwrk);