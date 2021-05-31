/*
Fractal code for CS 4380 / CS 5351

Copyright (c) 2019 Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is *not* permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <sys/time.h>
#include "BMP43805351.h"

unsigned char* GPU_Init(const int gpu_frames, const int width);
void GPU_Exec(const int start_frame, const int gpu_frames, const int width, unsigned char* d_pic);
void GPU_Fini(const int gpu_frames, const int width, unsigned char* pic, unsigned char* d_pic);

static void fractal(const int start_frame, const int cpu_frames, const int width, unsigned char* const pic)
{
  // todo: use OpenMP to parallelize the for-row loop with default(none) and do not specify a schedule

  const double Delta = 0.002;
  const double xMid = 0.2315059;
  const double yMid = 0.5214880;

  // compute pixels of each frame
  double delta = Delta;
  for (int frame = start_frame; frame < cpu_frames; frame++) {  // frames

    delta = Delta * pow(0.98, frame);

    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;

    #pragma omp parallel for default(none) shared(pic,frame)
    for (int row = 0; row < width; row++) {  // rows
      const double cy = yMin + row * dw;
      for (int col = 0; col < width; col++) {  // columns
        const double cx = xMin + col * dw;
        double x = cx;
        double y = cy;
        double x2, y2;
        int depth = 256;
        do {
          x2 = x * x;
          y2 = y * y;
          y = 2.0 * x * y + cy;
          x = x2 - y2 + cx;
          depth--;
        } while ((depth > 0) && ((x2 + y2) < 5.0));
        pic[frame * width * width + row * width + col] = (unsigned char)depth;
      }
    }
    //delta *= 0.98;
  }
}

int main(int argc, char *argv[])
{
  // set up MPI
  int comm_sz, my_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) printf("Fractal v1.9\n");

  // check command line
  if (argc != 4) {fprintf(stderr, "USAGE: %s frame_width cpu_frames gpu_frames\n", argv[0]); exit(-1);}
  const int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "ERROR: frame_width must be at least 10\n"); exit(-1);}
  const int cpu_frames = atoi(argv[2]) / comm_sz;
  if (cpu_frames < 0) {fprintf(stderr, "ERROR: cpu_frames must be at least 0\n"); exit(-1);}
  const int gpu_frames = atoi(argv[3]) / comm_sz;
  if (gpu_frames < 0) {fprintf(stderr, "ERROR: gpu_frames must be at least 0\n"); exit(-1);}
  const int frames = cpu_frames + gpu_frames;
  if (frames < 1) {fprintf(stderr, "ERROR: total number of frames must be at least 1\n"); exit(-1);}

  const int cpu_start_frame = my_rank * frames;
  const int gpu_start_frame = cpu_start_frame + cpu_frames;

  if (my_rank == 0) {
    printf("cpu_frames: %d\n", cpu_frames * comm_sz);
    printf("gpu_frames: %d\n", gpu_frames * comm_sz);
    printf("frames: %d\n", frames * comm_sz);
    printf("width: %d\n", width);
    printf("MPI tasks: %d\n", comm_sz);
  }

  // allocate picture arrays
  unsigned char* pic = new unsigned char [frames * width * width];
  unsigned char* d_pic = GPU_Init(gpu_frames, width);
  unsigned char* full_pic = NULL;
  if (my_rank == 0) full_pic = new unsigned char [frames * comm_sz * width * width];

  // start time
  timeval start, end;
  MPI_Barrier(MPI_COMM_WORLD);  // for better timing
  gettimeofday(&start, NULL);

  // asynchronously compute the requested frames on the GPU
  GPU_Exec(gpu_start_frame, gpu_frames, width, d_pic);

  // compute the remaining frames on the CPU
  fractal(cpu_start_frame, cpu_frames, width, pic);

  // copy the GPU's result into the appropriate location of the CPU's pic array
  GPU_Fini(gpu_frames, width, &pic[cpu_frames * width * width], d_pic);

  // gather the resulting frames

  // todo: gather the results into full_pic on compute node 0
  MPI_Gather(&pic[cpu_start_frame], frames, MPI_CHAR, full_pic, frames, MPI_CHAR, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    gettimeofday(&end, NULL);
    const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("compute time: %.4f s\n", runtime);

    // write result to BMP files
    if ((width <= 257) && (frames * comm_sz <= 60)) {
      for (int frame = 0; frame < frames * comm_sz; frame++) {
        BMP24 bmp(0, 0, width - 1, width - 1);
        for (int y = 0; y < width - 1; y++) {
          for (int x = 0; x < width - 1; x++) {
            const int p = full_pic[frame * width * width + y * width + x];
            const int e = full_pic[frame * width * width + y * width + (x + 1)];
            const int s = full_pic[frame * width * width + (y + 1) * width + x];
            const int dx = std::min(2 * std::abs(e - p), 255);
            const int dy = std::min(2 * std::abs(s - p), 255);
            bmp.dot(x, y, dx * 0x000100 + dy * 0x000001);
          }
        }
        char name[32];
        sprintf(name, "fractal%d.bmp", frame + 1000);
        bmp.save(name);
      }
    }

    delete [] full_pic;
  }

  MPI_Finalize();
  delete [] pic;
  return 0;
}

