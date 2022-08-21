#include <iostream>
#include "network.h"

using namespace std;


networkTrainer::networkTrainer(neuralNetwork& the_neural_network, vector<vector<double>> expected_inputs, vector<vector<double>> expected_outputs) {
	neural_network = &the_neural_network;

	expected_input_output_values[iinput_expectation] = expected_inputs;
	expected_input_output_values[ioutput_expectation] = expected_outputs;
}

void networkTrainer:: train_network(int cycles) {
	const int NUMBER_OF_LAYERS = neural_network->hidden_layers.size() + 2;
	networks_slopes = vector<vector<vector<double>>>(3);
	for (int loop = 0; loop < cycles; loop++) {
		;
	}
	// publish_networks_slopes();
	networks_slopes.clear();
	return;
}

void networkTrainer::publish_networks_slopes() {
	neuralNetwork& publisher_network = *neural_network;
	publisher_network.input_layer.adjust_weights(networks_slopes[0]); // publishes the input layer's weights
	for (int i = 0; i < publisher_network.hidden_layers.size(); i++) { publisher_network.hidden_layers[i].adjust_weights(networks_slopes[i + 1]); } // publishes hidden layer weights
	publisher_network.output_layer.adjust_weights(networks_slopes[networks_slopes.size() - 1]); // publishes output layer's weights

	return;
}

double networkTrainer::get_layers_cost(hiddenLayer ACTUAL_LAYER, vector<double> const LAYERS_EXPECTED_VALUES) {
	double cost = 0;
	
	for (int i = 0; i < ACTUAL_LAYER.layer_neurons.size(); i++) { 
		double neurons_value = ACTUAL_LAYER.layer_neurons[i].get_activated_value();
		cost += pow(neurons_value - LAYERS_EXPECTED_VALUES[i], 2);
	}
	return cost;
}

double networkTrainer::get_layers_cost(outputLayer ACTUAL_LAYER, vector<double> const LAYERS_EXPECTED_VALUES) {
	double cost = 0;

	for (int i = 0; i < ACTUAL_LAYER.layer_neurons.size(); i++) {
		double neurons_value = ACTUAL_LAYER.layer_neurons[i].get_activated_value();
		cost += pow(neurons_value - LAYERS_EXPECTED_VALUES[i], 2);
	}
	return cost;
}

double networkTrainer::get_cost_weight_slope(double expected_value, neuron n) {
	return n.get_activated_value() * dtanh(n.get_z()) * (2 * (n.get_activated_value() - expected_value);
}