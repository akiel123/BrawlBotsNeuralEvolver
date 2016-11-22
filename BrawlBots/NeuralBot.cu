#include "bot.h"
#include <vector>
#include "Arena.h"
#include "NeuralNetwork.h"
#include "NeuralBot.h"

__device__ thrust::device_vector<float> GetInputs(SNeuralBot b, thrust::device_vector<float> memory, SArena a) {
	thrust::device_vector<float> inp;
	inp.reserve(nTrivial + nObs * nDetailsObs + nMemory);
	inp.push_back(0); //Arena width
	inp.push_back(0); //Arena height

	thrust::device_vector<SSighting> v = AView(a, b.posx, b.posy, b.rot, b.id);
	int random = randomPromille() * 1000 * v.size();
	for (int i = 0; i < nObs && v.size() != 0; i++) {
		SSighting s = v[(random + i) % v.size()];
		for (int j = 0; j < nDetailsObs; j++) {
			switch (j) {
			case 1:
				inp.push_back(s.distance / ARENADIAG);
				break;
			case 2:
				inp.push_back(fmodf(s.angle, M_2PI) / M_2PI);
				break;
			case 3:
				inp.push_back(s.velocityx / MAX_VELOCITY);
				break;
			case 4:
				inp.push_back(s.velocityy/ MAX_VELOCITY);
				break;
			case 5:
				inp.push_back(viewTypeToDouble(s.t));
				break;
			case 6:
				inp.push_back(s.id / RAND_MAX);
				break;
			case 7:
				inp.push_back(s.posx / ARENAW);
				break;
			case 8:
				inp.push_back(s.posx / ARENAH);
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

__device__ BotAction GetBotAction(SNeuralBot b, SNeuralNetwork nnwrk) {
	b.lastOutput = NNGetOutput(nnwrk, b.lastOutput);
	return BotAction(b.lastOutput[0], b.lastOutput[1], b.lastOutput[2] > 0.5);
}