#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <string>
#include <functional>
#include <vector>

using function_t = std::function<double(double)>;

class Matrix
{
    using matrix_t = std::vector<std::vector<double>>;
public:
    Matrix() {}
    Matrix(const uint16_t _rows, const uint16_t _columns);
    Matrix(const Matrix &mat);
    Matrix(std::string file_string);
    ~Matrix(){};

    void print() const;
    void save(std::string file_string);
    void randomize(uint16_t n);
    uint32_t max_value();
    void flatten(bool axis);

    void dot(const Matrix &mat);
    void apply(function_t func);
    void add(const Matrix &mat);
    void subtract(const Matrix &mat);
    void multiply(const Matrix &mat);
    void scale(const double n);
    void transpose();

    Matrix soft_max();

    inline uint16_t rows() const { return m_entries.size(); }
    inline uint16_t cols() const { return m_entries[0].size(); }
    inline bool check_dimensions(const Matrix &mat) { return rows() == mat.rows() && cols() == mat.cols(); }

    matrix_t m_entries;
};

double sigmoid(double input);
double sigmoid_prime(double input);

#endif // MATRIX_H