#include "nn.h"
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

#define MAXCHAR 1000

NeuralNetwork::NeuralNetwork(int input, int hidden, int output) : m_input(input), m_hidden(hidden), m_output(output)
{
	Matrix hidden_layer(m_hidden, m_input);
	Matrix output_layer(m_output, m_hidden);

	hidden_layer.randomize(m_hidden);
	output_layer.randomize(m_output);

	m_hidden_weights = hidden_layer;
	m_output_weights = output_layer;
}

NeuralNetwork::NeuralNetwork(std::string file_string)
{
	char entry[MAXCHAR];
	chdir(file_string.c_str());
	FILE *descriptor = fopen("descriptor", "r");

	fgets(entry, MAXCHAR, descriptor);
	m_input = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	m_hidden = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	m_output = atoi(entry);

	fclose(descriptor);

	m_hidden_weights = Matrix("hidden");
	m_output_weights = Matrix("output");

	printf("Successfully loaded network from '%s'\n", file_string.c_str());
	chdir("-"); // Go back to the original directory
}

void NeuralNetwork::train(Matrix _input, Matrix _output)
{
	Matrix input_calculations = m_hidden_weights;
	Matrix output_calculations = m_output_weights;
	Matrix hidden_errors = m_output_weights;

	// Feed Forward
	input_calculations.dot(_input);
	input_calculations.apply(sigmoid);
	output_calculations.dot(input_calculations);
	output_calculations.apply(sigmoid);

	// Find Errors
	_output.subtract(output_calculations);
	hidden_errors.transpose();
	hidden_errors.dot(_output);

	// Feed Backward
	// Output Weights
	output_calculations.apply(sigmoid_prime);
	_output.multiply(output_calculations);
	input_calculations.transpose();
	_output.dot(input_calculations);
	_output.scale(m_learning_rate /  m_batch_size);
	m_output_weights.add(_output);

	// Hidden Weights
	input_calculations.transpose();
	input_calculations.apply(sigmoid_prime);
	input_calculations.multiply(hidden_errors);
	_input.transpose();
	input_calculations.dot(_input);
	input_calculations.scale(m_learning_rate /  m_batch_size);
	m_hidden_weights.add(input_calculations);
}

void NeuralNetwork::train_batch_imgs(const std::vector<Img> &imgs)
{
	double real_learning_rate = m_learning_rate;
	for (int i = 0; i < imgs.size(); i++)
	{
		Img cur_img = imgs[i];
		cur_img.img_data.flatten(true); // false = flatten to single row vector
		Matrix output(2, 1);
		output.m_entries[cur_img.label][0] = 1;
		train(cur_img.img_data, output);
	}
}

void NeuralNetwork::train_model(const std::vector<Img> &imgs, uint16_t epochs, uint16_t batch_size, double learning_rate)
{
	m_learning_rate = learning_rate;
	m_batch_size = batch_size;
	train_batch_imgs(imgs);
	return;
		
	std::vector<std::vector<Img>> bunches;
	if (batch_size == 0)
	{
		m_batch_size = imgs.size();
		bunches.push_back(imgs);
	}
	else
	{
		for (size_t i = 0; i < imgs.size(); i += batch_size)
		{
			auto last = std::min(imgs.size(), i + batch_size);
			bunches.emplace_back(imgs.begin() + i, imgs.begin() + last);
		}
	}

	for (size_t epoch = 0; epoch < epochs; epoch++)
	{
		for (auto it = bunches.begin(); it != bunches.end(); ++it)
		{
			train_batch_imgs(*it);
		}
	}
	
}

double NeuralNetwork::predict_batch_imgs(const std::vector<Img> &imgs)
{
	int n_correct = 0;
	for (int i = 0; i < imgs.size(); i++)
	{
		Img cur_img = imgs[i];
		cur_img.img_data.flatten(true);
		Matrix prediction = predict(cur_img.img_data);

		std::cout << "0 - " << prediction.m_entries[0][0] << " | 1 - " << prediction.m_entries[1][0]
				  << " | argmax - " << prediction.max_value() << " | result - " << cur_img.label << std::endl;
		n_correct += prediction.max_value() == cur_img.label;
	}
	return 1.0 * n_correct / imgs.size();
}

Matrix NeuralNetwork::predict(const Matrix &input_data)
{
	Matrix input_calculations = m_hidden_weights;
	Matrix output_calculations = m_output_weights;

	input_calculations.dot(input_data);
	input_calculations.apply(sigmoid);
	output_calculations.dot(input_calculations);
	output_calculations.apply(sigmoid);
	output_calculations.soft_max();
	return output_calculations;
}

void NeuralNetwork::save(std::string file_string)
{
	mkdir(file_string.c_str(), 0777);
	chdir(file_string.c_str());
	FILE *descriptor = fopen("descriptor", "w");
	fprintf(descriptor, "%d\n", m_input);
	fprintf(descriptor, "%d\n", m_hidden);
	fprintf(descriptor, "%d\n", m_output);
	fclose(descriptor);
	m_hidden_weights.save("hidden");
	m_output_weights.save("output");
	printf("Successfully written to '%s'\n", file_string.c_str());
	chdir("-");
}

void NeuralNetwork::print() const
{
	printf("# of inputs: %d\n", m_input);
	printf("# of hidden: %d\n", m_hidden);
	printf("# of outputs: %d\n", m_output);
	printf("Hidden Weights: \n");
	m_hidden_weights.print();
	printf("Output Weights: \n");
	m_output_weights.print();
}