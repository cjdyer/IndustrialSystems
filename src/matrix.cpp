#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include <iostream>

#define MAXCHAR 100

Matrix::Matrix(const uint16_t _rows, const uint16_t _columns)
	: m_entries(_rows, std::vector<double>(_columns)) {}

Matrix::Matrix(const Matrix &mat) : m_entries(mat.m_entries) {}

Matrix::Matrix(std::string file_string)
{
	FILE *file = fopen(file_string.c_str(), "r");

	char entry[MAXCHAR];
	fgets(entry, MAXCHAR, file);
	int row_size = atoi(entry);
	fgets(entry, MAXCHAR, file);
	int col_size = atoi(entry);

	*this = Matrix(row_size, col_size);

	for (int i = 0; i < rows(); i++) // Use an iterator instead
	{
		for (int j = 0; j < cols(); j++)
		{
			fgets(entry, MAXCHAR, file);
			m_entries[i][j] = std::strtod(entry, NULL);
		}
	}

	printf("Sucessfully loaded matrix from %s\n", file_string.c_str());
	fclose(file);
}

void Matrix::print() const
{
	printf("Rows: %d Columns: %d\n", rows(), cols());

	for (int i = 0; i < rows(); i++)
	{
		for (int j = 0; j < cols(); j++)
		{
			printf("%1.3f ", m_entries[i][j]);
		}

		printf("\n");
	}
}

void Matrix::save(std::string file_string)
{
	FILE *file = fopen(file_string.c_str(), "w");

	fprintf(file, "%d\n", rows());
	fprintf(file, "%d\n", cols());

	for (int i = 0; i < rows(); i++)
	{
		for (int j = 0; j < cols(); j++)
		{
			fprintf(file, "%.6f\n", m_entries[i][j]);
		}
	}

	printf("Successfully saved matrix to %s\n", file_string.c_str());
	fclose(file);
}

void Matrix::randomize(uint16_t n)
{
	const double min = -1.0 / sqrt(n);
	const int scaled_difference = (min - 1.0 / sqrt(n)) * 10000;

	for (int i = 0; i < rows(); i++)
	{
		for (int j = 0; j < cols(); j++)
		{
			m_entries[i][j] = min + (1.0 * (rand() % scaled_difference) / 10000);
		}
	}
}

uint32_t Matrix::max_value()
{
	double max_score = 0;
	uint32_t max_idx = 0;

	for (int i = 0; i < rows(); i++)
	{
		if (m_entries[i][0] > max_score)
		{
			max_score = m_entries[i][0];
			max_idx = i;
		}
	}

	return max_idx;
}

void Matrix::flatten(bool axis)
{
	// Axis = 0 -> Column Vector, Axis = 1 -> Row Vector
	const uint32_t rows_size = (((rows() * cols()) - 1) * axis) + 1;
	const uint32_t cols_size = (((rows() * cols()) - 1) * !axis) + 1;

	matrix_t temp_mat(rows_size, std::vector<double>(cols_size));

	for (int i = 0; i < rows(); i++)
	{
		const uint32_t col_index = i * cols();
		for (int j = 0; j < cols(); j++)
		{
			const uint32_t absolute_index = col_index + j;
			temp_mat[absolute_index * axis][absolute_index * !axis] = m_entries[i][j];
		}
	}

	m_entries = temp_mat;
}

void Matrix::multiply(const Matrix &mat)
{
	if (!check_dimensions(mat))
		exit(1);

	for (int i = 0; i < rows(); i++)
	{
		for (int j = 0; j < cols(); j++)
		{
			m_entries[i][j] *= mat.m_entries[i][j];
		}
	}
}

void Matrix::add(const Matrix &mat)
{
	if (!check_dimensions(mat))
		exit(1);

	for (int i = 0; i < rows(); i++)
	{
		for (int j = 0; j < cols(); j++)
		{
			m_entries[i][j] += mat.m_entries[i][j];
		}
	}
}

void Matrix::subtract(const Matrix &mat)
{
	if (!check_dimensions(mat))
		exit(1);

	for (int i = 0; i < rows(); i++)
	{
		for (int j = 0; j < cols(); j++)
		{
			m_entries[i][j] -= mat.m_entries[i][j];
		}
	}
}

void Matrix::apply(function_t func)
{
	for (int i = 0; i < rows(); i++)
	{
		for (int j = 0; j < cols(); j++)
		{
			m_entries[i][j] = func(m_entries[i][j]);
		}
	}
}

void Matrix::dot(const Matrix &mat)
{
	if (cols() != mat.rows())
		exit(1);

	matrix_t temp_mat(rows(), std::vector<double>(mat.cols()));
	for (int i = 0; i < rows(); i++)
	{
		for (int j = 0; j < mat.cols(); j++)
		{
			for (int k = 0; k < mat.rows(); k++)
			{
				temp_mat[i][j] += m_entries[i][k] * mat.m_entries[k][j];
			}
		}
	}
	m_entries = temp_mat;
}

void Matrix::scale(const double n)
{
	for (int i = 0; i < rows(); i++)
	{
		for (int j = 0; j < cols(); j++)
		{
			m_entries[i][j] *= n;
		}
	}
}

void Matrix::transpose()
{
	matrix_t temp_mat(cols(), std::vector<double>(rows()));

	for (int i = 0; i < rows(); i++)
	{
		for (int j = 0; j < cols(); j++)
		{
			temp_mat[j][i] = m_entries[i][j];
		}
	}

	m_entries = temp_mat;
}

double sigmoid(double input)
{
	return 1.0 / (1.0 + exp(-input));
}

double sigmoid_prime(double input)
{
	return exp(input) / pow(exp(input) + 1, 2.0);
}

Matrix Matrix::soft_max()
{
	double total = 0;

	for (int i = 0; i < rows(); i++)
	{
		for (int j = 0; j < cols(); j++)
		{
			total += exp(m_entries[i][j]);
		}
	}

	Matrix mat = Matrix(rows(), cols());

	for (int i = 0; i < mat.rows(); i++)
	{
		for (int j = 0; j < mat.cols(); j++)
		{
			mat.m_entries[i][j] = exp(m_entries[i][j]) / total;
		}
	}

	return mat;
}