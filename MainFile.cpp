#include <iostream>
#include <time.h> // using for randomization
#include "network.h"

using namespace std;


int main() {
	srand(time(NULL)); // initializes randomness

	neuralNetwork line_network(1, 1, 1, 1, 1);

	networkTrainer fifty_output(line_network, vector<double>{0, 1}, vector<double>{0, 0.5});

	fifty_output.train_network();

	return 0;
}
