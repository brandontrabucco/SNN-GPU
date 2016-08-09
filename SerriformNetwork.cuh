/*
 * SawtoothNetwork.h
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#ifndef SERRIFORMNETWORK_H_
#define SERRIFORMNETWORK_H_

#include <vector>
#include <cuda.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include "Neuron.cuh"
using namespace std;

__global__ void forwardPass(Neuron **neurons, double *connections, double *activations, int size, int cycles);
__global__ void backwardPass(Neuron **neurons, double *weightedError, double *errorSum, double learningRate, int connections, int size, int cycles);

class SerriformNetwork {
private:
	const int maxBlocks = 256;
	const int maxThreads = 256;
	int inputSize;
	int overlap;
	double learningRate;
	double decayRate;
	double **error;
	vector<vector<Neuron> > layers;
	int getPreviousNeurons();
	int getPreviousNeurons(int l);
public:
	SerriformNetwork(int is, int o, double l, double d);
	virtual ~SerriformNetwork();
	void addLayer(int size);
	vector<double> classify(vector<double> input);
	vector<double> train(vector<double> input, vector<double> target);
};

#endif /* SERRIFORMNETWORK_H_ */
