#include "img.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

std::vector<Img> load_csv(const char *_file_name)
{
	std::vector<Img> imgs;

	std::string line, word;
	std::fstream file(_file_name, std::ios::in);

	if (file.is_open())
	{
		while (getline(file, line))
		{
			std::stringstream str(line);
			getline(str, word, ',');
			
			Matrix img_data = Matrix(8, 8);
			// Matrix img_data = Matrix(9, 9);
			bool label = atoi(word.c_str());
			uint8_t x, y = 0;

			while (getline(str, word, ','))
			{
				img_data.m_entries[y][x] = atoi(word.c_str()) / 256.0;
				x = (x == 7 ?  0 : (x + 1));
				// x = (x == 8 ?  0 : (x + 1));
				y = (x == 0) + y;
				if (y == 8) break;
			}

			imgs.push_back({img_data, label});
		}
	}

	return imgs;
}

void Img::print() const
{
	img_data.print();
	std::cout << "Img Label: " << label << std::endl;
}