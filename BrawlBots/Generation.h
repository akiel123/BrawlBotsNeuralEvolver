#pragma once
#include "Individual.h"
#include <cstdlib>

struct SGeneration {
	std::vector<SIndividual> *ids;
	int generationNumber;
	int generationID;
	
	SGeneration(int nIndividual) {
		generationID = rand();
		for (int i = 0; i < nIndividual; i++) {
			ids->push_back(SIndividual());
		}
		generationNumber = 0;
	}
	SGeneration(std::vector<SIndividual> *individuals) {
		generationID = rand();
		ids = individuals;
		generationNumber = 0;
	}
	SGeneration(std::vector<SIndividual> *indvs, int generationNumber, int generationID) {
		generationID = generationID;
		ids = indvs;
		generationNumber = generationNumber;
	}

};

void GSortByLegacyFitness(SGeneration g);
void GSortByPersonalFitness(SGeneration g);
void GEvolveGeneration(SGeneration g);
void GEvolveGenerationToPopulationNr(SGeneration g, int n);
void GEvolveGenerationSelectiveToPopulationNr(SGeneration g, int n, float thresh);