#include <iostream>
#include "general_functions.h"

using namespace std;

// DEFINES THE GENERAL FUNCTIONS 

int get_random_integer(const int MIN, const int MAX) { return (rand() % (MAX - MIN + 1)) + MIN; }

double step(double x) { return (x > 0) ? 1 : 0; } // One of the activation functions used by the neurons

double dtanh(double x) {
	double e = 2.7182818;
	return 4 / pow(pow(e, x * -1) + pow(e, x), 2);
}