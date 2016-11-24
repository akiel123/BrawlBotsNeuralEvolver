#pragma once
#include "NeuralNetwork.h"
#include <cstdlib>

struct HIndividual {
	HNeuralNetwork nnwrk = HNeuralNetwork(0,0);
	float personalFitness;
	float legacyFitness;
	int name;
	
	HIndividual() {
		name = rand();
		nnwrk = HNeuralNetwork(39, 13);
		personalFitness = 0;
		legacyFitness = 0;
	}
	HIndividual(HNeuralNetwork parent) {
		name = rand();
		nnwrk = NNGetMutatedCopy(parent);
		personalFitness = 0;
		legacyFitness = 0;
	}
	HIndividual(HNeuralNetwork parent, float lFitness) {
		name = rand();
		nnwrk =  NNGetMutatedCopy(parent);
		personalFitness = 0;
		legacyFitness = lFitness;
	}
	HIndividual(HNeuralNetwork parent, float lFitness, float pFitness, int Name) {
		name = Name;
		nnwrk = NNGetMutatedCopy(parent);
		personalFitness = pFitness;
		legacyFitness = lFitness;
	}
};

struct less_than_key_pf {
	inline bool operator() (const HIndividual &i1, const HIndividual &i2) {
		return(i1.personalFitness < i2.personalFitness);
	}
};
struct less_than_key_lf {
	inline bool operator() (const HIndividual &i1, const HIndividual &i2) {
		return(i1.legacyFitness < i2.legacyFitness);
	}
};

void ISortByLegacyFitness(std::vector<HIndividual> ids);
void ISortByPersonalFitness(std::vector<HIndividual> ids);