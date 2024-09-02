#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>  
#include "EMCompute.h"

int main() {
  // Define the kernel
  CKernel kernel;
  kernel.x = 60000;  // Number of workgroups in the x dimension
  kernel.y = 1000;
  kernel.z = 100;

  // WGSL code to perform element-wise addition of example_data and example_data0
  const char* code = 
    "@group(0)@binding(0) var<storage, read_write> v_indices: array<u32>; "
    "@group(0)@binding(1) var<storage, read> v_indices0: array<u32>; "
    "@compute @workgroup_size(10 , 1 , 1)" 
    "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) { "
    "  let idx = global_id.x % 60000; "
    "   "
    "v_indices[idx] = v_indices[idx] + v_indices0[idx]; "
    "  "
    "}";

  uintptr_t index = set_kernel_default_config(&kernel);
  kernel.kernel_code_index = register_computing_kernel_code(index , code , "main");



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
    compute(&kernel, groups, 1);
    //free_compute_cache();
  // }
  

  // Check results
  printf("example_data[4]: %d\n", example_data[50000]);
  printf("example_data0[4]: %d\n", example_data0[4]);

  free_compute_cache();

  return 0;
}

