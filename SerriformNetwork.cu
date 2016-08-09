/*
 * SerriformNetwork.cpp
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#include "SerriformNetwork.cuh"

__global__ void forwardPass(Neuron **neurons, double *connections, double *activations, int size, int cycles) {
	int maxId = gridDim.x * blockDim.x;
	for (int i = 0; i < (cycles); i++) {
		int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
		if (idx < size) {
			activations[idx] = neurons[idx]->forward(connections);
		}
	}
}

__global__ void backwardPass(Neuron **neurons, double *weightedError, double *errorSum, double learningRate, int connections, int size, int cycles) {
	int maxId = gridDim.x * blockDim.x;
	for (int i = 0; i < (cycles); i++) {
		int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
		if (idx < size) {
			double *contribution = neurons[idx]->backward(weightedError[idx], learningRate);
			for (int j = 0; j < connections; j++) {
				errorSum[j] += contribution[j];
			}
		}
	}
}

SerriformNetwork::SerriformNetwork(int is, int o, double l, double d) {
	// TODO Auto-generated constructor stub
	inputSize = is;
	overlap = o;
	learningRate = l;
	decayRate = d;
}

SerriformNetwork::~SerriformNetwork() {
	// TODO Auto-generated destructor stub
	for (int i = 0; i < layers.size(); i++) {
		free(error[i]);
	} free(error);
}

int SerriformNetwork::getPreviousNeurons() {
	int sum = 0;
	for (int i = ((int)layers.size() - 1); i > ((int)layers.size() - overlap - 2); i--) {
		if (i == -1) sum += inputSize;
		else if (i >= 0) {
			sum += (int)layers[i].size();
		}
	}
	return sum;
}

int SerriformNetwork::getPreviousNeurons(int l) {
	int sum = 0;
	for (int i = (l - 1); i > (l - overlap - 2); i--) {
		if (i == -1) sum += inputSize;
		else if (i >= 0) {
			sum += (int)layers[i].size();
		} //cout << i << " " << sum << endl;
	}
	return sum;
}

void SerriformNetwork::addLayer(int size) {
	vector<Neuron> buffer;
	for (int i = 0; i < size; i++) {
		buffer.push_back(Neuron(getPreviousNeurons()));
	} layers.push_back(buffer);
	if (layers.size() > 1) error = (double **)realloc(error, (sizeof(double *) * layers.size()));
	else error = (double **)malloc(sizeof(double *) * layers.size());
	error[layers.size() - 1] = (double *)calloc(size, sizeof(double));
}

vector<double> SerriformNetwork::classify(vector<double> input) {
	double *output = (double *)malloc(sizeof(double) * layers[layers.size() - 1].size());
	// calculate activations in reverse order from top
	for (int i = (layers.size() - 1); i >= 0; i--) {
		// sum the input from all previous layer neurons
		double *connections;
		cudaMalloc((void **)&connections, (sizeof(double) * getPreviousNeurons(i)));
		int offset = 0;
		for (int k = (i - overlap - 1); k < i; k++) {
			if (k == -1) {
				cudaMemcpy(&connections[0], &input[0], (sizeof(double) * input.size()), cudaMemcpyHostToDevice);
				offset += inputSize;
			} else if (k >= 0) for (int l = 0; l < (int)layers[k].size(); l++) {
				cudaMemcpy(&connections[offset], &(layers[k][l].activation), sizeof(double), cudaMemcpyHostToDevice);
				offset++;
			}
		} double *activations;
		cudaMalloc((void **)&activations, (sizeof(double) * layers[i].size()));

		Neuron **deviceNeurons, **buffer = (Neuron **)malloc(sizeof(Neuron *) * layers[i].size());
		for (int j = 0; j < layers[i].size(); j++) {
			cudaMemcpy(&(layers[i][j].impulse[0]), &connections[0], (sizeof(double) * layers[i][j].connections), cudaMemcpyDeviceToHost);
		}
		cudaMalloc((void **)&deviceNeurons, sizeof(Neuron *) * layers[i].size());
		for (int j = 0; j < layers[i].size(); j++) {
			Neuron *dn = Neuron::copyToGPU(&layers[i][j]);
			cudaMemcpy(&deviceNeurons[j], &dn, sizeof(Neuron *), cudaMemcpyHostToDevice);
		} forwardPass<<<maxBlocks, maxThreads>>>(deviceNeurons, connections, activations, layers[i].size(), ceil((double)layers[i].size() / (double)(maxBlocks * maxThreads)));
		cudaDeviceSynchronize();

		cudaMemcpy(&buffer[0], &deviceNeurons[0], (sizeof(Neuron *) * layers[i].size()), cudaMemcpyDeviceToHost);
		for (int j = 0; j < layers[i].size(); j++) {
			layers[i][j] = *Neuron::copyFromGPU(buffer[j]);
		} if (i == (layers.size() - 1)) cudaMemcpy(&output[0], &activations[0], (sizeof(double) * layers[layers.size() - 1].size()), cudaMemcpyDeviceToHost);

		cudaFree(deviceNeurons);
		cudaFree(activations);
		cudaFree(connections);
		free(buffer);
	} vector<double> result(&output[0], &output[layers[layers.size() - 1].size()]);
	free(output);
	return result;
}

vector<double> SerriformNetwork::train(vector<double> input, vector<double> target) {
	double *output = (double *)malloc(sizeof(double) * layers[layers.size() - 1].size());
	if ((int)input.size() == inputSize && (int)target.size() == ((int)layers[layers.size() - 1].size())) {
		// calculate activations in reverse order from top
		for (int i = ((int)layers.size() - 1); i >= 0; i--) {// sum the input from all previous layer neurons
			int connectionSize = getPreviousNeurons(i);
			double *connections;
			cudaMalloc((void **)&connections, (sizeof(double) * connectionSize));
			int offset = 0;
			for (int k = (i - overlap - 1); k < i; k++) {
				if (k == -1) {
					cudaMemcpy(&connections[0], &input[0], (sizeof(double) * input.size()), cudaMemcpyHostToDevice);
					offset += inputSize;
				} else if (k >= 0) for (int l = 0; l < (int)layers[k].size(); l++) {
					cudaMemcpy(&connections[offset], &(layers[k][l].activation), sizeof(double), cudaMemcpyHostToDevice);
					offset++;
				}
			}

			double *activations;
			cudaMalloc((void **)&activations, (sizeof(double) * layers[i].size()));
			Neuron **deviceNeurons, **buffer = (Neuron **)malloc(sizeof(Neuron *) * layers[i].size());
			cudaMalloc((void **)&deviceNeurons, sizeof(Neuron *) * layers[i].size());
			for (int j = 0; j < layers[i].size(); j++) {
				cudaMemcpy(&(layers[i][j].impulse[0]), &connections[0], (sizeof(double) * layers[i][j].connections), cudaMemcpyDeviceToHost);
				Neuron *dn = Neuron::copyToGPU(&layers[i][j]);
				cudaMemcpy(&deviceNeurons[j], &dn, sizeof(Neuron *), cudaMemcpyHostToDevice);
			} forwardPass<<<maxBlocks, maxThreads>>>(deviceNeurons, connections, activations, layers[i].size(), ceil((double)layers[i].size() / (double)(maxBlocks * maxThreads)));
			cudaDeviceSynchronize();

			double *weightedError, *errorSum;
			cudaMalloc((void **)&weightedError, (sizeof(double) * layers[i].size()));
			cudaMalloc((void **)&errorSum, (sizeof(double) * connectionSize));
			if (i > 0) cudaMemcpy(&errorSum[0], &(error[i - 1][0]), (sizeof(double) * connectionSize), cudaMemcpyHostToDevice);
			else cudaMemset(&errorSum[0], 0.0, (sizeof(double) * connectionSize));
			if (i == (layers.size() - 1)) {
				cudaMemcpy(&output[0], &activations[0], (sizeof(double) * layers[layers.size() - 1].size()), cudaMemcpyDeviceToHost);
				for (int j = 0; j < layers[layers.size() - 1].size(); j++) {
					double e = (output[j] - target[j]);
					output[j] = e;
					error[i][j] = e;
				} free(output);
			} cudaMemcpy(&weightedError[0], &error[i][0], (sizeof(double) * layers[i].size()), cudaMemcpyHostToDevice);
			backwardPass<<<maxBlocks, maxThreads>>>(deviceNeurons, weightedError, errorSum, learningRate, connectionSize, layers[i].size(), ceil((double)layers[i].size() / (double)(maxBlocks * maxThreads)));
			cudaDeviceSynchronize();

			offset = 0;
			if (i > 0) {
				for (int k = (i - overlap - 1); k < i; k++) {
					if (k == -1) {
						// input layer
						offset += inputSize;
					} else if (k >= 0) {
						cudaMemcpy(&error[k][0], &errorSum[offset], (sizeof(double) * layers[k].size()), cudaMemcpyDeviceToHost);
						offset += layers[k].size();
					}
				}
			} cudaMemcpy(&buffer[0], &deviceNeurons[0], (sizeof(Neuron *) * layers[i].size()), cudaMemcpyDeviceToHost);

			for (int j = 0; j < layers[i].size(); j++) {
				Neuron temp = *Neuron::copyFromGPU(buffer[j]);
				layers[i][j] = temp;
			} free(buffer);

			cudaFree(connections);
			cudaFree(activations);
			cudaFree(deviceNeurons);
			cudaFree(weightedError);
			cudaFree(errorSum);
			cudaDeviceSynchronize();

		} vector<double> result(&error[layers.size() - 1][0], &error[layers.size() - 1][layers[layers.size() - 1].size()]);
		learningRate *= decayRate;
		return result;
	}
	else return vector<double>();
}
