#include <CL/sycl.hpp>

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

int *buildGaussKern(int winSize, int sigma)
{
  int wincenter, x;
  float sum = 0.0f;
  wincenter = winSize / 2;
  float *kern = (float *)malloc(winSize * sizeof(float));
  int *ikern = (int *)malloc(winSize * sizeof(int));
  float SQRT_2PI = 2.506628274631f;
  float sigmaMul2PI = 1.0f / (sigma * SQRT_2PI);
  float divSigmaPow2 = 1.0f / (2.0f * sigma * sigma);
  for (x = 0; x < wincenter + 1; x++)
  {
    kern[wincenter - x] = kern[wincenter + x] = exp(-(x * x) * divSigmaPow2) * sigmaMul2PI;
    sum += kern[wincenter - x] + ((x != 0) ? kern[wincenter + x] : 0.0);
  }
  sum = 1.0f / sum;
  for (x = 0; x < winSize; x++)
  {
    kern[x] *= sum;
    ikern[x] = kern[x] * 256.0f;
  }
  free(kern);
  return ikern;
}

void GaussBlur(rgb *pixels, unsigned int width, unsigned int height, int sigma)
{
  unsigned int winsize = (1 + (((int)ceil(3 * sigma)) * 2));
  int *gaussKern = buildGaussKern(winsize, sigma);
  unsigned int halfsize = winsize / 2;

  rgb *tmpBuffer = (rgb *)malloc(width * height * sizeof(rgb));

  for (unsigned int h = 0; h < height; h++)
  {
    unsigned int rowWidth = h * width;

    for (unsigned int w = 0; w < width; w += 1)
    {
      unsigned int rowR = 0;
      unsigned int rowG = 0;
      unsigned int rowB = 0;
      int *gaussKernPtr = gaussKern;
      int whalfsize = w + width - halfsize;
      unsigned int curPos = rowWidth + w;
      for (unsigned int k = 1; k < winsize; k += 1)
      {
        unsigned int pos = rowWidth + ((k + whalfsize) % width);
        int fkern = *gaussKernPtr++;
        rowR += (pixels[pos].red * fkern);
        rowG += (pixels[pos].green * fkern);
        rowB += (pixels[pos].blue * fkern);
      }

      tmpBuffer[curPos].red = ((unsigned char)(rowR >> 8));
      tmpBuffer[curPos].green = ((unsigned char)(rowG >> 8));
      tmpBuffer[curPos].blue = ((unsigned char)(rowB >> 8));
    }
  }
  for (unsigned int w = 0; w < width; w++)
  {
    for (unsigned int h = 0; h < height; h++)
    {
      unsigned int colR = 0;
      unsigned int colG = 0;
      unsigned int colB = 0;
      int hhalfsize = h + height - halfsize;
      for (unsigned int k = 0; k < winsize; k++)
      {
        colR += tmpBuffer[((k + hhalfsize) % height) * width + w].red * gaussKern[k];
        colG += tmpBuffer[((k + hhalfsize) % height) * width + w].green * gaussKern[k];
        colB += tmpBuffer[((k + hhalfsize) % height) * width + w].blue * gaussKern[k];
      }
      pixels[h * width + w].red = (unsigned char)(colR >> 8);
      pixels[h * width + w].green = (unsigned char)(colG >> 8);
      pixels[h * width + w].blue = (unsigned char)(colB >> 8);
    }
  }
  free(tmpBuffer);
  free(gaussKern);
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

  // 读取需要处理的位图数据
  int image_width = 0, image_height = 0, num_channels = 0;

  rgb *indata = (rgb *)stbi_load(argv[1],
                                 &image_width, &image_height,
                                 &num_channels, STBI_rgb);

  // 输出读取位图信息
  std::cout << "Filename: " << argv[1] << "\n"
            << "Width: " << image_width << "\n"
            << "Height: " << image_height << "\n\n";

  GaussBlur(indata, image_width, image_height, 2);

      // 输出位图
      stbi_write_bmp(argv[2], image_width, image_height, 3, indata);
  std::cout << "DCT successfully completed.\n"
            << "The processed image has been written to "
            << argv[2] << "\n";

  // 释放数据空间
  stbi_image_free(indata);
}
