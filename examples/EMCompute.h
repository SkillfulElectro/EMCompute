/*MIT License

Copyright (c) 2024 ElectroMutex

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#ifndef EMCOMPUTE_H
#define EMCOMPUTE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * NOTE : on linux machines memory leak might happen if you use
 * vulkan backend until NVIDIA drivers for linux get fixed .
 *
 *
 * computing backends of the api
 */
typedef enum GPUComputingBackend {
  /**
   * targets all of the backends
   */
  all = 0,
  /**
   * default backend
   */
  default_backend = 1,
  /**
   * Supported on Windows, Linux/Android, and macOS/iOS via Vulkan Portability (with the Vulkan feature enabled)
   */
  vulkan = 2,
  /**
   * Supported on Linux/Android, the web through webassembly via WebGL, and Windows and macOS/iOS via ANGLE
   */
  opengl = 3,
  /**
   * MacOS & iOS only
   */
  metal = 4,
  /**
   * Windows +10 only
   */
  direct_x12 = 5,
  /**
   * browser WebGPU
   */
  webgpu = 6,
  /**
   * targets VULKAN METALDX12 BROWSER_WEBGPU
   */
  highest_support = 7,
  /**
   * targets OpenGL backend
   */
  lowest_support = 8,
} GPUComputingBackend;

/**
 * this settings used to tell gpu pre information about
 * our work
 */
typedef enum GPUMemorySettings {
  /**
   * our app needs to me more performant instead being
   * cable of allocating too much memory on gpu side
   */
  prefer_performance = 0,
  /**
   * our app will need to allocate memory on gpu side
   */
  prefer_memory = 1,
  /**
   * will be supported in next versions , by default for now it is set to
   * prefer_memory . in next versions you can tell how much memory
   * you need to allocate on gpu side
   */
  custom_memory = 3,
} GPUMemorySettings;

/**
 * this enum is used to tell to API
 * to setup GPU resources based on power saving rules or
 * not
 */
typedef enum GPUPowerSettings {
  /**
   * power and performance does not matter
   */
  none = 0,
  /**
   * choose based on the power saving rules
   */
  LowPower = 1,
  /**
   * performance is more important
   */
  HighPerformance = 2,
} GPUPowerSettings;

/**
 * this enum affects speed of the api
 * by setting how much gpu resources
 * are needed directly , if you take
 * too much which your hardware cannot
 * provide , panic happens
 */
typedef enum GPUSpeedSettings {
  /**
   * the lowest resources , supported on all backends
   */
  lowest_speed = 0,
  /**
   * low resources , supported on all backends expect webgl2
   * which our api does not aim to support for now
   */
  low_speed = 1,
  /**
   * the default
   */
  default_speed = 2,
  /**
   * will be supported in next versions , for now it is equal to
   * low_speed
   */
  custom_speed = 3,
} GPUSpeedSettings;

/**
 * as config field you have to provide GPUComputingConfig which
 * represent settings which you wanted
 */
typedef struct GPUComputingConfig {
  /**
   * set backend which you want
   */
  enum GPUComputingBackend backend;
  /**
   * set power settings which meets your needs
   */
  enum GPUPowerSettings power;
  /**
   * set speed settings which matches your needs
   */
  enum GPUSpeedSettings speed;
  /**
   * tell to gpu about your memory usage
   */
  enum GPUMemorySettings memory;
} GPUComputingConfig;

/**
 * CKernel which will represent your GPU task
 * like how Manifest.xml does in an android
 * project
 */
typedef struct CKernel {
  /**
   * set max number of workgroups in x dimension
   */
  uint32_t x;
  /**
   * set max number of workgroups in y dimension
   */
  uint32_t y;
  /**
   * set max number of workgroups in z dimension
   */
  uint32_t z;
  /**
   * this is a kernel code which must be in wgsl for now
   * more shading languages will be supported in the future
   */
  const char *code;
  /**
   * this part in the code , tell to the api which
   * function in the code must be called by gpu
   * when the task is sent to gpu
   */
  const char *code_entry_point;
  /**
   * by setting config you can customize behavior of the
   * gpu
   */
  struct GPUComputingConfig config;
  uintptr_t index;
} CKernel;

/**
 * this struct is for passing
 * data based on its bind index
 * in gpu side
 */
typedef struct DataBinder {
  /**
   * bind index of data in gpu side
   */
  uint32_t bind;
  /**
   * because data must be in uint8_t (u8 in Rust)
   * in C you have to pass the data len this way
   *
   * sizeof(your type) * real_len_of_your_array / sizeof(uint8_t)
   */
  uintptr_t data_len;
  /**
   * pointer to your data  in memory , it must be
   * uint8_t* (*mut u8 in Rust side)
   * in gpu side the type of this data will
   * be set based on CKernel code you provided
   */
  uint8_t *data;
} DataBinder;

/**
 * all DataBinder types which have
 * the same @group index in your kernel
 * code must all be gathered in this
 * type
 */
typedef struct GroupOfBinders {
  /**
   * index of group in your kernel code
   */
  uint32_t group;
  /**
   * pointer to array which all of the
   * DataBinders from same group
   * are gathered in
   */
  struct DataBinder *datas;
  /**
   * len of datas array
   */
  uintptr_t datas_len;
} GroupOfBinders;

/**
 * because setting CKernel config can be annoying if you just
 * want to do simple task , this function provides general
 * config which will meet most of your needs
 */
void set_kernel_default_config(struct CKernel *kernel);

/**
 * the simple and compact function for sending
 * your computing task to the gpu side
 *
 * kernel para = CKernel type which acts as Manifest for your gpu task
 * data_for_gpu = pointer to array of GroupOfBinders which contains data which must be sent to gpu
 * gpu_data_len = len of the array of the GroupOfBinders
 *
 * unlike CUDA , you dont need to copy data to gpu manually , this function does it for you
 * in the most performant possible way
 *
 * if you find any bug or any problem , help us to fix it -> https://github.com/SkillfulElectro/EMCompute.git
 */
int32_t compute(struct CKernel kernel,
                struct GroupOfBinders *data_for_gpu,
                uintptr_t gpu_data_len);

/**
 * since version 2.0.0 api does
 * caching for gpu resources on the memory .
 * the api does deallocates the caches
 * automatically , but in some cases
 * you might want to do it manually
 * so just call this free_compute_cache();
 */
void free_compute_cache(void);

#endif  /* EMCOMPUTE_H */
