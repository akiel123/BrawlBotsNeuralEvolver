#pragma once
#include "Individual.h"
#include <cstdlib>

struct HGeneration {
	std::vector<HIndividual> *ids;
	int generationNumber;
	int generationID;
	
	HGeneration(int nIndividual) {
		generationID = rand();
		for (int i = 0; i < nIndividual; i++) {
			ids->push_back(HIndividual());
		}
		generationNumber = 0;
	}
	HGeneration(std::vector<HIndividual> *individuals) {
		generationID = rand();
		ids = individuals;
		generationNumber = 0;
	}
	HGeneration(std::vector<HIndividual> *indvs, int generationNumber, int generationID) {
		generationID = generationID;
		ids = indvs;
		generationNumber = generationNumber;
	}

};

void GSortByLegacyFitness(HGeneration g);
void GSortByPersonalFitness(HGeneration g);
void GEvolveGeneration(HGeneration g);
void GEvolveGenerationToPopulationNr(HGeneration g, int n);
void GEvolveGenerationSelectiveToPopulationNr(HGeneration g, int n, float thresh);