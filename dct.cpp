#include <CL/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "dpc_common.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#pragma pack(push, 1)

// This is the data structure which is going to represent one pixel value in RGB
// format
typedef struct
{
  unsigned char blue;
  unsigned char green;
  unsigned char red;
} rgb;

// This block is only used when build for Structure of Arays (SOA) with Array
// Notation
typedef struct
{
  unsigned char *blue;
  unsigned char *green;
  unsigned char *red;
} SOA_rgb;

#pragma pack(pop)

using namespace dpc_common;
using namespace sycl;

constexpr int block_dims = 8;
constexpr int block_size = 64;

// 创建 8x8 的DCT矩阵
void CreateDCT(float matrix[block_size])
{
  int temp[block_dims];
  for (int i = 0; i < block_dims; ++i)
    temp[i] = i;
  for (int i = 0; i < block_dims; ++i)
  {
    for (int j = 0; j < block_dims; ++j)
    {
      if (i == 0)
        matrix[(i * block_dims) + j] = (1 / sycl::sqrt((float)block_dims));
      else
        matrix[(i * block_dims) + j] =
            sycl::sqrt((float)2 / block_dims) *
            sycl::cos(((((float)2 * temp[j]) + 1) * i * 3.14f) /
                      (2 * block_dims));
    }
  }
}

// 8x8 矩阵转置
void MatrixTranspose(float x[block_size], float xinv[block_size])
{
  for (int i = 0; i < block_dims; ++i)
  {
    for (int j = 0; j < block_dims; ++j)
      xinv[(j * block_dims) + i] = x[(i * block_dims) + j];
  }
}

// 矩阵相乘
SYCL_EXTERNAL void MatrixMultiply(float x[block_size], float y[block_size],
                                  float xy[block_size])
{
  for (int i = 0; i < block_dims; ++i)
  {
    for (int j = 0; j < block_dims; ++j)
    {
      xy[(i * block_dims) + j] = 0;
      for (int k = 0; k < block_dims; ++k)
        xy[(i * block_dims) + j] +=
            (x[(i * block_dims) + k] * y[(k * block_dims) + j]);
    }
  }
}

// 8x8 块上执行 DCT 算法
SYCL_EXTERNAL void ProcessBlock(rgb *indataset, rgb *outdataset,
                                float dct[block_size], float dctinv[block_size],
                                int start_index, int width)
{
  float interim[block_size], product[block_size], red_input[block_size],
      blue_input[block_size], green_input[block_size], temp[block_size];

  // 90% 量化
  float quant[64] = {3, 2, 2, 3, 5, 8, 10, 12,
                     2, 2, 3, 4, 5, 12, 12, 11,
                     3, 3, 3, 5, 8, 11, 14, 11,
                     3, 3, 4, 6, 10, 17, 16, 12,
                     4, 4, 7, 11, 14, 22, 21, 15,
                     5, 7, 11, 13, 16, 12, 23, 18,
                     10, 13, 16, 17, 21, 24, 24, 21,
                     14, 18, 19, 20, 22, 20, 20, 20};
  
  // 处理红色通道

  // 转换像素范围从 [0, 255] 到 [-128, 127]
  for (int i = 0; i < block_size; ++i)
  {
    int pixel_index = i / block_dims * width + i % block_dims;
    red_input[i] = indataset[start_index + pixel_index].red;
    red_input[i] -= 128;
  }

  // 计算DCT
  MatrixMultiply(dct, red_input, temp);
  MatrixMultiply(temp, dctinv, interim);

  
  // 使用量化矩阵舍弃高频数据
  for (int i = 0; i < block_size; ++i)
    interim[i] = sycl::floor((interim[i] / quant[i]) + 0.5f);

  // 反量化
  for (int i = 0; i < block_size; ++i)
    interim[i] = sycl::floor((interim[i] * quant[i]) + 0.5f);

  // 计算IDCT
  MatrixMultiply(dctinv, interim, temp);
  MatrixMultiply(temp, dct, product);

  // 转换像素范围写入输出图片
  for (int i = 0; i < block_size; ++i)
  {
    int pixel_index = i / block_dims * width + i % block_dims;
    float temp = (product[i] + 128);
    outdataset[start_index + pixel_index].red =
        (temp > 255.f) ? 255 : (unsigned char)temp;
  }

  // 处理蓝色通道

  // 转换像素范围从 [0, 255] 到 [-128, 127]
  for (int i = 0; i < block_size; ++i)
  {
    int pixel_index = i / block_dims * width + i % block_dims;
    blue_input[i] = indataset[start_index + pixel_index].blue;
    blue_input[i] -= 128;
  }

  // 计算DCT
  MatrixMultiply(dct, blue_input, temp);
  MatrixMultiply(temp, dctinv, interim);

  // 使用量化矩阵舍弃高频数据
  for (int i = 0; i < block_size; ++i)
    interim[i] = sycl::floor((interim[i] / quant[i]) + 0.5f);

  // 反量化
  //for (int i = 0; i < block_size; ++i)
  //  interim[i] = sycl::floor((interim[i] * quant[i]) + 0.5f);

  // 计算IDCT
  MatrixMultiply(dctinv, interim, temp);
  MatrixMultiply(temp, dct, product);

  // 转换像素范围写入输出图片
  for (int i = 0; i < block_size; ++i)
  {
    int pixel_index = i / block_dims * width + i % block_dims;
    float temp = product[i] + 128;
    outdataset[start_index + pixel_index].blue =
        (temp > 255.f) ? 255 : (unsigned char)temp;
  }

  // 处理绿色通道

  // 转换像素范围从 [0, 255] 到 [-128, 127]
  for (int i = 0; i < block_size; ++i)
  {
    int pixel_index = i / block_dims * width + i % block_dims;
    green_input[i] = indataset[start_index + pixel_index].green;
    green_input[i] -= 128;
  }

  // 计算DCT
  MatrixMultiply(dct, green_input, temp);
  MatrixMultiply(temp, dctinv, interim);

  // 使用量化矩阵舍弃高频数据
  //for (int i = 0; i < block_size; ++i)
  //  interim[i] = sycl::floor((interim[i] / quant[i]) + 0.5f);

  // 反量化
  //for (int i = 0; i < block_size; ++i)
  //  interim[i] = sycl::floor((interim[i] * quant[i]) + 0.5f);

  // 计算IDCT
  MatrixMultiply(dctinv, interim, temp);
  MatrixMultiply(temp, dct, product);

  // 转换像素范围写入输出图片
  for (int i = 0; i < block_size; ++i)
  {
    int pixel_index = i / block_dims * width + i % block_dims;
    float temp = product[i] + 128;
    outdataset[start_index + pixel_index].green =
        (temp > 255.f) ? 255 : (unsigned char)temp;
  }
}

// 将图像划分为 8x8 块大小处理
void ProcessImage(rgb *indataset, rgb *outdataset, int width, int height)
{
  sycl::queue q(default_selector_v, exception_handler);
  std::cout << "Running on "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  try
  {
    int image_size = width * height;
    float dct[block_size], dctinv[block_size];

    // 创建 8x8 的 DCT 矩阵
    CreateDCT(dct);
    // 创建 DCT 矩阵的转置
    MatrixTranspose(dct, dctinv);

    buffer indata_buf(indataset, range<1>(image_size));
    buffer outdata_buf(outdataset, range<1>(image_size));
    buffer dct_buf(dct, range<1>(block_size));
    buffer dctinv_buf(dctinv, range<1>(block_size));

    q.submit([&](handler &h)
             {
      auto i_acc = indata_buf.get_access(h,read_only);
      auto o_acc = outdata_buf.get_access(h);
      auto d_acc = dct_buf.get_access(h,read_only);
      auto di_acc = dctinv_buf.get_access(h,read_only);

      // 并行执行 8x8 块计算
      h.parallel_for(
          range<2>(width / block_dims, height / block_dims), [=](auto idx) {
            int start_index = idx[0] * block_dims + idx[1] * block_dims * width;
            ProcessBlock(i_acc.get_pointer(), o_acc.get_pointer(),
                         d_acc.get_pointer(), di_acc.get_pointer(), start_index,
                         width);
          }); });
    q.wait_and_throw();
  }
  catch (sycl::exception e)
  {
    std::cout << "SYCL exception caught: " << e.what() << "\n";
    exit(1);
  }
}

// 读写位图
int ReadProcessWrite(char *input, char *output)
{
  double timersecs;

  // 读取需要处理的位图数据
  int image_width = 0, image_height = 0, num_channels = 0;
  rgb *indata = (rgb *)stbi_load(input,
                                 &image_width, &image_height,
                                 &num_channels, STBI_rgb);
  // 未读取到数据
  if (!indata)
  {
    std::cout << "The input file could not be opened. Program will now exit\n";
    return 1;
  }
  else if (num_channels != 3)
  {
    std::cout
        << "The input file must be an RGB bmp image. Program will now exit\n";
    return 1;
  }
  else if (image_width % block_dims != 0 || image_height % block_dims != 0)
  {
    std::cout
        << "The input image must have dimensions which are a multiple of 8\n";
    return 1;
  }

  // 输出读取位图信息
  std::cout << "Filename: " << input << "\n"
            << " Width: " << image_width << "\n"
            << " Height: " << image_height << "\n\n";

  // 创建相同大小的输出数组
  rgb *outdata = (rgb *)malloc(image_width * image_height * sizeof(rgb));

  std::cout << "Start image processing ...\n";
  {
    TimeInterval t;
    ProcessImage(indata, outdata, image_width, image_height);
    timersecs = t.Elapsed();
  }
  std::cout << "--The processing time is " << timersecs << " seconds\n\n";

  // 输出位图
  stbi_write_bmp(output, image_width, image_height, 3, outdata);
  std::cout << "DCT successfully completed.\n"
            << "The processed image has been written to "
            << output << "\n";

  // 释放数据空间
  stbi_image_free(indata);
  std::free(outdata);

  return 0;
}

int main(int argc, char *argv[])
{
  // 判断输入的参数是否满足要求
  if (argc < 3)
  {
    std::cout << "Program usage is <modified_program> <inputfile.bmp> "
                 "<outputfile.bmp>\n";
    return 1;
  }
  return ReadProcessWrite(argv[1], argv[2]);
}
