# EMCompute
[![Build](https://github.com/SkillfulElectro/EMCompute/actions/workflows/rust.yml/badge.svg)](https://github.com/SkillfulElectro/EMCompute/actions/workflows/rust.yml)
- this library tries to take computing tasks on GPU for parallel processing in the simplest possible way for Rust/C/C++ and other languages which can work with C API
- this project is successor to https://github.com/SkillfulElectro/core-compute.git and https://github.com/SkillfulElectro/core-compute_native.git

## Why EMCompute?
- its fast
- its simple 
- its cross-platform
- its configurable
- it supports shading languages

## Getting started
- if you want to use it in Rust refer to https://crates.io/crates/EMCompute and check out https://docs.rs/EMCompute/latest/EMCompute/
- for using with C/C++ and Cython check out https://github.com/SkillfulElectro/EMCompute.git . for getting prebuilt binaries for your OS check the latest action artifacts it will contain .h , .hpp and .pyx header files and prebuilt binaries (you can read the comments for better understanding)

### Tutorial
- first things which you have to create is struct of type CKernel which stands for Computing Kernel , this struct will act as an manifest of your task which must be done by GPU 
```c
typedef struct CKernel {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  const char *code;
  const char *code_entry_point;
  struct GPUComputingConfig config;
} CKernel;
```
- x , y and z fields are used to specify max number of workgroups in each dimension . (and in your kernel code you will specify how much threads each workgroup must have)
- field code must contain your wgsl compute shader code (other shading languages will be supported soon)
- code_entry_point field will must be set to a function which must be called by GPU for your task 
- config field will tell to API how GPU must treat with our tasks 
- for setting it manually check the comments on header files or https://docs.rs/EMCompute/latest/EMCompute/ but for making it easier you can use : 
```c 
void set_kernel_default_config(struct CKernel *kernel);
```
- you pass pointer of your CKernel var and its config will be set , its useful because it will meet needs of most of our tasks
- now its gathering data time for GPU for that you have to use DataBinder and GroupOfBinders structs 
```c 
typedef struct DataBinder {
  uint32_t bind;
  uintptr_t data_len;
  uint8_t *data;
} DataBinder;
```
- in bind field you will provide bind index which in your kernel code exists to data goes there 
- data_len field must be : sizeof(your type) * real_len_of_your_array / sizeof(uint8_t)
- data field must be a pointer to array of your data 
- now in GroupOfBinders you will set the group index and a pointer to all DataBinders which are in same group 
```c 
typedef struct GroupOfBinders {
  uint32_t group;
  struct DataBinder *datas;
  uintptr_t datas_len;
} GroupOfBinders;
```
- now we have to create an array for GroupOfBinders and pass it to the compute function and done we are finished
```c 
int32_t compute(struct CKernel kernel,
                struct GroupOfBinders *data_for_gpu,
                uintptr_t gpu_data_len);
```
- it will return number which if not 0 ; error happened
- as an example : 
```main.c 
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>  
#include "EMCompute.h"

int main() {
  CKernel kernel;
  kernel.x = 60000;  
  kernel.y = 1000;
  kernel.z = 100;

  kernel.code = 
    "@group(0)@binding(0) var<storage, read_write> v_indices: array<u32>; "
    "@group(0)@binding(1) var<storage, read_write> v_indices0: array<u32>; "
    "@compute @workgroup_size(10 , 1 , 1)" 
    "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) { "
    "  let idx = global_id.x % 60000; "
    "   "
    "v_indices[idx] = v_indices[idx] + v_indices0[idx]; "
    "  "
    "}";

  set_kernel_default_config(&kernel);
  kernel.code_entry_point = "main";

  // Initialize data
  uint32_t example_data[60000];
  uint32_t example_data0[60000];

  for (int i = 0; i < 60000; ++i) {
    example_data[i] = 1;
    example_data0[i] = 1;
  }

  // Bind data
  DataBinder data;
  data.bind = 0;
  data.data = (uint8_t *)example_data;
  data.data_len = sizeof(uint32_t)*60000/sizeof(uint8_t);

  DataBinder data0;
  data0.bind = 1;
  data0.data = (uint8_t *)example_data0;
  data0.data_len = sizeof(uint32_t)*60000/sizeof(uint8_t);

  DataBinder group0[] = {data, data0};
  GroupOfBinders wrapper;
  wrapper.group = 0;
  wrapper.datas = group0;
  wrapper.datas_len = 2;

  GroupOfBinders groups[] = {wrapper};

  // for (int i = 0 ; i< 1000000 ;++i){
    compute(kernel, groups, 1);
  // }

  // Check results
  printf("example_data[4]: %d\n", example_data[4]);
  printf("example_data0[4]: %d\n", example_data0[4]);

  return 0;
}
```
- check out example https://github.com/SkillfulElectro/EMCompute/tree/main/examples and Goodluck :)

## Contribution
- if you find any problem or bug , ill be happy with your pull req or issue report 