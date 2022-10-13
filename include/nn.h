#ifndef NN_H
#define NN_H

#include "matrix.h"
#include "img.h"

class NeuralNetwork
{
private:
    void train(Matrix _input, Matrix _output);
    Matrix predict(const Matrix &input_data);
    void train_batch_imgs(const std::vector<Img>& imgs);
    Matrix predict_img(Img img);

public:
    NeuralNetwork(std::string file_string);
    NeuralNetwork(int input, int hidden, int output);
    ~NeuralNetwork(){};

	void train_model(const std::vector<Img>& imgs, uint16_t epochs, uint16_t batch_size, double learning_rate);
    double predict_batch_imgs(const std::vector<Img>& imgs);
    void save(std::string file_string);
    void print() const;

    int m_input;
    int m_hidden;
    int m_output;
    double m_learning_rate = 0.1;
    int m_batch_size;
    Matrix m_hidden_weights;
    Matrix m_output_weights;
};

#endif // NN_H