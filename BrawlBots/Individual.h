#pragma once
#include "NeuralNetwork.h"
#include <cstdlib>

struct SIndividual {
	SNeuralNetwork nnwrk = SNeuralNetwork(0,0);
	float personalFitness;
	float legacyFitness;
	int name;
	
	SIndividual() {
		name = rand();
		nnwrk = SNeuralNetwork(39, 13);
		personalFitness = 0;
		legacyFitness = 0;
	}
	SIndividual(SNeuralNetwork parent) {
		name = rand();
		nnwrk = NNGetMutatedCopy(parent);
		personalFitness = 0;
		legacyFitness = 0;
	}
	SIndividual(SNeuralNetwork parent, float lFitness) {
		name = rand();
		nnwrk =  NNGetMutatedCopy(parent);
		personalFitness = 0;
		legacyFitness = lFitness;
	}
	SIndividual(SNeuralNetwork parent, float lFitness, float pFitness, int Name) {
		name = Name;
		nnwrk = NNGetMutatedCopy(parent);
		personalFitness = pFitness;
		legacyFitness = lFitness;
	}
};

struct less_than_key_pf {
	inline bool operator() (const SIndividual &i1, const SIndividual &i2) {
		return(i1.personalFitness < i2.personalFitness);
	}
};
struct less_than_key_lf {
	inline bool operator() (const SIndividual &i1, const SIndividual &i2) {
		return(i1.legacyFitness < i2.legacyFitness);
	}
};

void ISortByLegacyFitness(std::vector<SIndividual> ids);
void ISortByPersonalFitness(std::vector<SIndividual> ids);