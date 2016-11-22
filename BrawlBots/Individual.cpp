#include "Individual.h"
#include <algorithm>

void ISortByLegacyFitness(std::vector<SIndividual> ids) {
	if (ids.size < 1) {
		ids.push_back(SIndividual());
	}
	std::sort(ids.begin, ids.end, less_than_key_lf());
}
void ISortByPersonalFitness(std::vector<SIndividual> ids) {
	if (ids.size < 1) {
		ids.push_back(SIndividual());
	}
	std::sort(ids.begin, ids.end, less_than_key_pf());
}