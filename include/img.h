#ifndef IMG_H
#define IMG_H

#include "matrix.h"
#include <vector>

struct Img
{
	Matrix img_data;
	bool label;
	
	void print() const;
};

std::vector<Img> load_csv(const char *_file_name);

#endif // IMG_H