#include "bot.h"
#include <vector>
#include "Arena.h"
#include "NeuralNetwork.h"
#include "NeuralBot.h"

__device__ float* GetInputs(DNeuralBot b, float* memory, DArena a) {
	float* inp;
	inp[0] = ARENAW; //Arena width
	inp[1] = ARENAH; //Arena height

	DSightingBatch v = AView(a, b.posx, b.posy, b.rot, b.id);
	int random = randomPromille() * 1000 * v.nSightings;
	for (int i = 0; i < nObs && v.nSightings != 0; i++) {
		DSighting s = v.sights[(random + i) % v.nSightings];
		for (int j = 0; j < nDetailsObs; j++) {
			switch (j) {
			case 0:
				inp[i * nDetailsObs + j] = s.distance / ARENADIAG;
				break;
			case 1:
				inp[i * nDetailsObs + j] = fmodf(s.angle, M_2PI) / M_2PI;
				break;
			case 2:
				inp[i * nDetailsObs + j] = s.velocityx / MAX_VELOCITY;
				break;
			case 3:
				inp[i * nDetailsObs + j] = s.velocityy/ MAX_VELOCITY;
				break;
			case 4:
				inp[i * nDetailsObs + j] = viewTypeToDouble(s.t);
				break;
			case 5:
				inp[i * nDetailsObs + j] = s.id / RAND_MAX;
				break;
			case 6:
				inp[i * nDetailsObs + j] = s.posx / ARENAW;
				break;
			case 7:
				inp[i * nDetailsObs + j] = s.posx / ARENAH;
				break;
			default:
				break;
			}
		}
	}
	for (int i = 0; i < nMemory; i++) {
		inp[nTrivial + nObs * nDetailsObs + i] = memory[i];
	}

	//advanced variables - currently considered constants
	//inp[x] = super.getViewAngle(); 
	//inp[x] = super.getRadius();

	return inp;
}

__device__ float viewTypeToDouble(ViewType t) {
	switch (t)
	{
	case ViewType::FRIEND:
		return 1 / VIEWTYPE_AMOUNT;
	case ViewType::ENEMY:
		return 2 / VIEWTYPE_AMOUNT;
	case ViewType::WALL:
		return 3 / VIEWTYPE_AMOUNT;
	case ViewType::BULLET:
		return 4 / VIEWTYPE_AMOUNT;
	case ViewType::AMMO:
		return 5 / VIEWTYPE_AMOUNT;
	default:
		return 0;
	}
}

__device__ BotAction GetBotAction(DNeuralBot b, DNeuralNetwork nnwrk) {
	b.lastOutput = NNGetOutput(nnwrk, b.lastOutput);
	return BotAction(b.lastOutput[0], b.lastOutput[1], b.lastOutput[2] > 0.5);
}