#include "Generation.h"
#include "Individual.h"
#include <iostream>

void GSortByLegacyFitness(HGeneration g) {
	ISortByLegacyFitness((*g.ids));
}
void GSortByPersonalFitness(HGeneration g) {
	ISortByPersonalFitness((*g.ids));
}

void GEvolveGeneration(HGeneration g) {
	GSortByPersonalFitness(g);
	int nId = (int)(g.ids->size / 2);
	std::vector<HIndividual> newGen;
	int babies = 0;
	int count = 0;
	int i = g.ids->size;
	while (babies < nId && count++ < nId * 4) {
		while (((rand() / RAND_MAX) * ((i / g.ids->size) * 1) + 1.5) > 1 && babies < nId) {
			newGen.push_back(HIndividual((*g.ids)[i]));
			babies++;
		}

		i = (i - 1) % g.ids->size;
	}

	GSortByLegacyFitness(g);
	nId = g.ids->size;
	i = g.ids->size;
	while (babies < nId && count++ < nId * 6) {
		while (((rand() / RAND_MAX) * ((i / g.ids->size) * 3.7) + 1.3) > 1 && babies < nId) {
			newGen.push_back(HIndividual((*g.ids)[i]));
			babies++;
		}

		i = (i - 1) % g.ids->size;
	}
	if (babies < nId) {
		for (int j = 0; j < nId - babies; j++) {
			newGen.push_back(HIndividual((*g.ids)[i]));
			babies++;
		}
	}

	g.generationNumber++;
	g.ids = &newGen;
	g.generationID = rand();
}

void GEvolveGenerationToPopulationNr(HGeneration g, int n) {
	GSortByPersonalFitness(g);
	int nId = n / 2;
	std::vector<HIndividual> newGen;
	int babies = 0;
	int count = 0;
	int i = g.ids->size;

	for (int j = 0; j < 5 && j < g.ids->size; j++) {
		newGen.push_back(HIndividual((*g.ids)[i]));
		babies++;
	}
	while (babies < nId && count++ < nId * 4) {
		while (((rand() / RAND_MAX) * ((i / g.ids->size) * 1) + 1.5) > 1 && babies < nId) {
			newGen.push_back(HIndividual((*g.ids)[i]));
			babies++;
		}

		i = (i - 1) % g.ids->size;
	}

	GSortByLegacyFitness(g);
	nId = n;
	i = g.ids->size - 1;
	while (babies < nId && count++ < nId * 6) {
		while (((rand() / RAND_MAX) * ((i / g.ids->size) * 3.7) + 1.3) > 1 && babies < nId) {
			newGen.push_back(HIndividual((*g.ids)[i]));
			babies++;
		}

		i = (i - 1) % g.ids->size;
	}
	if (babies < nId) {
		for (int j = 0; j < nId - babies; j++) {
			newGen.push_back(HIndividual((*g.ids)[i]));
			babies++;
		}
	}

	g.generationNumber++;
	g.ids = &newGen;
	g.generationID = rand();
}
void GEvolveGenerationSelectiveToPopulationNr(HGeneration g, int n, float thresh) {
	std::vector<HIndividual> idslist;
	for (int i = 0; i < g.ids->size; i++) {
		if ((*g.ids)[i].personalFitness > thresh) idslist.push_back((*g.ids)[i]);
	}

	if (idslist.size() < 3) {
		GEvolveGenerationToPopulationNr(g, n);
		std::cout << "Proficient population not large enough to evolve selectively \n";
		return;
	}

	g.ids = &idslist;
	GEvolveGenerationToPopulationNr(g, n);
}