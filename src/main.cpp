#include <iostream>
#include <fstream>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <mutex>
#include <thread>

#include "img.h"
#include "matrix.h"
#include "nn.h"
#include "preprocess.h"
#include "lodepng.h"

using preprocess::flat_image_t;
using preprocess::image_t;
using preprocess::PartType;

// #define PROCESSING
#define TRAINING
// #define TUNING
#define TESTING

const char *files[] = {
	"../data/processed images/training_data.csv",
	"../data/processed images/validation_data.csv"
};

std::vector<int> epoch_sizes =        {60, 85, 100, 125};
std::vector<int> hidden_nodes_sizes = {200, 250, 300, 350};
std::vector<double> learning_rates =  {0.05, 0.08, 0.12, 0.15};

struct hyperparameters
{
	uint16_t hidden_nodes;
	double learning_rate;
};

std::mutex file_mutex;
std::vector<Img> train_imgs, test_imgs;
std::vector<hyperparameters> possible_hp_combos;
std::vector<std::thread> threads;

void save_score(const double score, const uint16_t current_epoch, const hyperparameters& params)
{
	file_mutex.lock();	
	std::cout << "Trained Network - Epochs : " << current_epoch << " Hidden Nodes : " << params.hidden_nodes << " Learning Rate : " << params.learning_rate << std::endl;
	chdir("../data/scores");
	FILE *score_matrix = fopen("score_matrix.csv", "a");
	fprintf(score_matrix, "%1.5f, ", score);
	fprintf(score_matrix, "%d, ", params.hidden_nodes);
	fprintf(score_matrix, "%d, ", current_epoch);
	fprintf(score_matrix, "%1.5f\n", params.learning_rate);
	fclose(score_matrix);
	file_mutex.unlock();
}

void train_and_save(const hyperparameters& params)
{
	NeuralNetwork net = NeuralNetwork(64, params.hidden_nodes, 2);
	for (int i = 0; i < epoch_sizes[3] + 1; i++)
	{
		net.train_model(train_imgs, 1, 1, params.learning_rate);
		for (int saved_epoch : epoch_sizes)
		{
			if (i == saved_epoch)
			{
				double score = net.predict_batch_imgs(test_imgs);
				save_score(score, i, params);
			}
		}
	}
	
}

void populate_hp_combos(int idx)
{
	if (idx == 0x10) return;
	hyperparameters hp = { hidden_nodes_sizes[(idx & 0xC) >> 0x2], learning_rates[idx & 0x3] };
	possible_hp_combos.push_back(hp);
	populate_hp_combos(++idx);
}

int main(int argc, char *argv[])
{

#ifdef PROCESSING
	std::vector<uint16_t> index_list(DATASET_SIZE);
	std::iota(index_list.begin(), index_list.end(), 1);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine random_engine(seed);
	std::shuffle(index_list.begin(), index_list.end(), random_engine);

	std::string in_filename, out_filename;
	PartType part_type;

	preprocess::clear_files(files, 2);

	for (size_t i = 0; i <= DATASET_SIZE; i++)
	{
		in_filename = "../data/images/" + std::to_string(index_list[i]) + ".PNG";
		out_filename = ((i + 1) <= TRAIN_SIZE) ? files[Training] : files[Validation];
		part_type = (index_list[i] <= 400) ? PartType::BadPart : PartType::GoodPart;

		std::cout << std::setw(3) << i + 1 << " - Saving Part " << std::setw(3) << index_list[i] << " As A " << (part_type ? "Good Part" : " Bad Part") << " Into " << out_filename << std::endl;

		preprocess::process_condensed(in_filename.c_str(), out_filename.c_str(), part_type);
	}
#endif


#ifdef TRAINING
	srand(42);

	std::vector<Img> imgs = load_csv(files[0]);
	NeuralNetwork net = NeuralNetwork(64, 200, 2);
	net.train_model(imgs, 60, 1, 0.15);
	// net.save("../data/network");
#endif

#ifdef TUNING
	mkdir("../data/scores", 0777);
	FILE *score_matrix = fopen("../data/scores/score_matrix.csv", "w");
	fprintf(score_matrix, "Score, ");
	fprintf(score_matrix, "Hidden_nodes, ");
	fprintf(score_matrix, "Epochs, ");
	fprintf(score_matrix, "Learning_rate\n");
	fclose(score_matrix);

	srand(42);

	train_imgs = load_csv(files[0]);
	test_imgs = load_csv(files[1]);

	populate_hp_combos(0);
	for (const hyperparameters& hp : possible_hp_combos)
	{
		threads.push_back(std::thread(train_and_save, hp));
	}

	for(std::thread& th : threads)
	{
		th.join();
	}
	
#endif

#ifdef TESTING
	imgs = load_csv(files[1]);
	double score = net.predict_batch_imgs(imgs);
	printf("Score: %1.5f\n", score);
	
#endif

	return 0;
}