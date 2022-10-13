#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "lodepng.h"
#include <cstring>

#define DATASET_SIZE 600
#define TRAINING_SPLIT 0.85
#define VALIDATION_SPLIT 0.15

#define TRAIN_SIZE DATASET_SIZE * TRAINING_SPLIT
#define VALID_SIZE DATASET_SIZE * VALIDATION_SPLIT

enum
{
	Training,
	Validation
};

namespace preprocess
{
    struct dimension_t
    {
        uint16_t width;
        uint16_t height;

        dimension_t(const uint16_t _w = 0, const uint16_t _h = 0) : width(_w), height(_h) {}

        dimension_t &operator/=(const uint8_t _scalar)
        {
            this->width /= _scalar;
            this->height /= _scalar;
            return *this;
        }
    };
    
    typedef std::vector<std::vector<uint8_t>> image_t;
    typedef std::vector<uint8_t> flat_image_t;
    typedef std::pair<int16_t, int16_t> pixel_index_t;

    enum PartType
    {
        BadPart,
        GoodPart
    };

    dimension_t image_size;

    void loadFile(flat_image_t &_buffer, const char *_file_name)
    {
        std::ifstream file(_file_name, std::ios::in | std::ios::binary | std::ios::ate);

        std::streamsize size = 0;
        if (file.seekg(0, std::ios::end).good())
            size = file.tellg();
        if (file.seekg(0, std::ios::beg).good())
            size -= file.tellg();

        if (size > 0)
        {
            _buffer.resize((size_t)size);
            file.read((char *)(&_buffer[0]), size);
        }
        else
            _buffer.clear();
    }

    void open_image(flat_image_t &_image, const char *_file_name)
    {
        flat_image_t buffer;
        loadFile(buffer, _file_name);
        decodePNG(_image, image_size.width, image_size.height, buffer.empty() ? 0 : &buffer[0], (size_t)buffer.size());
    }

    void image_to_greyscale(const flat_image_t *_input_image, image_t &_output_image)
    {
        _output_image.resize(image_size.height, flat_image_t(image_size.width));
        for (size_t y = 0; y < image_size.height; y++)
        {
            uint32_t y_offset = y * image_size.width * 4;
            for (size_t x = 0; x < image_size.width; x++)
            {
                _output_image[y][x] = _input_image->at(y_offset + (x * 4));
            }
        }
    }

    void down_sample_by_average(image_t &_image, const uint8_t _sample_area)
    {
        image_t tmp_image(image_size.height / _sample_area, flat_image_t(image_size.width / _sample_area));
        uint16_t sample_area_sq = _sample_area * _sample_area;

        for (size_t y = 0; y < image_size.height; y += _sample_area)
        {
            for (size_t x = 0; x < image_size.width; x += _sample_area)
            {
                uint16_t sampled_total = 0;

                for (size_t sample_y = 0; sample_y < _sample_area; sample_y++)
                {
                    for (size_t sample_x = 0; sample_x < _sample_area; sample_x++)
                    {
                        sampled_total += _image[y + sample_y][x + sample_x];
                    }
                }

                tmp_image[y / _sample_area][x / _sample_area] = sampled_total / sample_area_sq;
            }
        }

        image_size /= _sample_area;
        _image.resize(image_size.height, flat_image_t(image_size.width));
        _image = tmp_image; // Check if assigning resizes, may not need the line above
    }

    void threshold_image(image_t &_image, const uint8_t _threshold_value)
    {
        for (auto &i : _image)
        {
            std::replace_if(
                i.begin(), i.end(), [&](unsigned char &val)
                { return val < _threshold_value; },
                1);
            std::replace_if(
                i.begin(), i.end(), [&](unsigned char &val)
                { return val >= _threshold_value; },
                0);
        }
    }

    pixel_index_t operator-(const pixel_index_t &p1, const pixel_index_t &p2)
    {
        return {p1.first - p2.first, p1.second - p2.second};
    }

    pixel_index_t operator+(const std::pair<double, double> &p1, const pixel_index_t &p2)
    {
        return {p1.first + p2.first, p1.second + p2.second};
    }

    pixel_index_t translate_pixel(const pixel_index_t _desired_pos, const pixel_index_t _old_center, const pixel_index_t _new_center, const double _sin_angle, const double _cos_angle)
    {
        pixel_index_t translated_pos = _desired_pos - _new_center;
        // Test if this needs to be a double,double
        std::pair<double, double> rotated_pos = {_cos_angle * translated_pos.first - _sin_angle * translated_pos.second,
                                                 _sin_angle * translated_pos.first + _cos_angle * translated_pos.second};
        return rotated_pos + _old_center;
    }

    void crop_to_corners(image_t &_image, const image_t &_threshold_image, const dimension_t _new_dim = {50, 35})
    {
        pixel_index_t corner_pos[4];
        image_t tmp_image(_new_dim.height, flat_image_t(_new_dim.width));

        // Scan from left to right
        [&]
        {
            for (size_t x = 0; x < image_size.width; x++)
            {
                for (size_t y = 0; y < image_size.height; y++)
                {
                    if (int(_threshold_image[y][x]) == 1)
                    {
                        corner_pos[0] = {x, y};
                        return;
                    }
                }
            }
        }();

        // Scan from top to bottom
        [&]
        {
            for (size_t y = 0; y < image_size.height; y++)
            {
                for (size_t x = image_size.width - 1; x > 0; x--)
                {
                    if (int(_threshold_image[y][x]) == 1)
                    {
                        corner_pos[1] = {x, y};
                        return;
                    }
                }
            }
        }();

        // Scan from right to left
        [&]
        {
            for (size_t x = image_size.width - 1; x > 0; x--)
            {
                for (size_t y = image_size.height - 1; y > 0; y--)
                {
                    if (int(_threshold_image[y][x]) == 1)
                    {
                        corner_pos[2] = {x, y};
                        return;
                    }
                }
            }
        }();

        // Scan from bottom to top
        [&]
        {
            for (size_t y = image_size.height - 1; y > 0; y--)
            {
                for (size_t x = 0; x < image_size.width; x++)
                {
                    if (int(_threshold_image[y][x]) == 1)
                    {
                        corner_pos[3] = {x, y};
                        return;
                    }
                }
            }
        }();

        pixel_index_t center = {(corner_pos[0].first + corner_pos[1].first + corner_pos[2].first + corner_pos[3].first) / 4,
                                (corner_pos[0].second + corner_pos[1].second + corner_pos[2].second + corner_pos[3].second) / 4};
        double angle = std::atan2(corner_pos[1].second - corner_pos[0].second, corner_pos[1].first - corner_pos[0].first);
        double sin_angle = std::sin(angle);
        double cos_angle = std::cos(angle);

        for (size_t y = 0; y < _new_dim.height; y++)
        {
            for (size_t x = 0; x < _new_dim.width; x++)
            {
                pixel_index_t input_index = translate_pixel({x, y}, center, {_new_dim.width / 2, _new_dim.height / 2}, sin_angle, cos_angle);
                tmp_image[y][x] = _image[input_index.second][input_index.first];
            }
        }

        image_size = _new_dim;
        _image.resize(image_size.height, flat_image_t(image_size.width));
        _image = tmp_image;
    }

    void save_to_file(const image_t &_image, const char *_file_name, const PartType _part_type)
    {
        std::ofstream output_file;
        output_file.open(_file_name, std::ofstream::app);
        output_file << int(_part_type);
        for (size_t y = 0; y < image_size.height; y++)
        {
            for (size_t x = 0; x < image_size.width; x++)
            {
                output_file << "," << int(_image[y][x]);
            }
        }
        for (size_t i = 0; i < 11; i++)
        {
            output_file << "," << 0;
        }
        
        output_file << "\n";
        output_file.close();
    }

    void clear_files(const char **_files, const uint8_t _num_files)
    {
        for (int i = 0; i < _num_files; i++)
        {
            std::ofstream file;
            file.open(_files[i], std::ofstream::trunc);
            file.close();
        }
    }

    // Save to BMP is used purely for debugging purposes
    // Retrieved from: https://stackoverflow.com/questions/2654480/writing-bmp-image-in-pure-c-c-without-other-libraries
    void save_to_bmp(const image_t &_image)
    {
        FILE *f;
        unsigned char *img = NULL;
        int filesize = 54 + 3 * image_size.width * image_size.height;

        img = (unsigned char *)std::malloc(3 * image_size.width * image_size.height);
        std::memset(img, 0, 3 * image_size.width * image_size.height);
        int x, y, v;

        for (int y = 0; y < image_size.height; y++)
        {
            for (int x = 0; x < image_size.width; x++)
            {
                v = _image[y][x];
                if (v > 255)
                    v = 255;
                img[(x + y * image_size.width) * 3 + 2] = (unsigned char)(v);
                img[(x + y * image_size.width) * 3 + 1] = (unsigned char)(v);
                img[(x + y * image_size.width) * 3 + 0] = (unsigned char)(v);
            }
        }

        unsigned char bmpfileheader[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
        unsigned char bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};
        unsigned char bmppad[3] = {0, 0, 0};

        bmpfileheader[2] = (unsigned char)(filesize);
        bmpfileheader[3] = (unsigned char)(filesize >> 8);
        bmpfileheader[4] = (unsigned char)(filesize >> 16);
        bmpfileheader[5] = (unsigned char)(filesize >> 24);

        bmpinfoheader[4] = (unsigned char)(image_size.width);
        bmpinfoheader[5] = (unsigned char)(image_size.width >> 8);
        bmpinfoheader[6] = (unsigned char)(image_size.width >> 16);
        bmpinfoheader[7] = (unsigned char)(image_size.width >> 24);
        bmpinfoheader[8] = (unsigned char)(image_size.height);
        bmpinfoheader[9] = (unsigned char)(image_size.height >> 8);
        bmpinfoheader[10] = (unsigned char)(image_size.height >> 16);
        bmpinfoheader[11] = (unsigned char)(image_size.height >> 24);

        f = fopen("../output.bmp", "wb");
        fwrite(bmpfileheader, 1, 14, f);
        fwrite(bmpinfoheader, 1, 40, f);
        for (int i = 0; i < image_size.height; i++)
        {
            fwrite(img + (image_size.width * (image_size.height - i - 1) * 3), 3, image_size.width, f);
            fwrite(bmppad, 1, (4 - (image_size.width * 3) % 4) % 4, f);
        }

        free(img);
        fclose(f);
    }

    void process_condensed(const char *_in_file_name, const char *_out_file_name, const PartType _part_type)
    {
        flat_image_t buffer;
        image_t image, thresh_image;

        open_image(buffer, _in_file_name);
        image_to_greyscale(&buffer, image);
        down_sample_by_average(image, 10);
        thresh_image = image;
        threshold_image(thresh_image, 90);
        crop_to_corners(image, thresh_image);
        down_sample_by_average(image, 5);
        save_to_file(image, _out_file_name, _part_type);
    }
} // namespace preprocess

#endif // PREPROCESS_H