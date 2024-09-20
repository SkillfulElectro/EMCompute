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

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

/// computing backends of the api
enum class GPUComputingBackend {
  /// targets all of the backends
  all = 0,
  /// default backend
  default_backend = 1,
  /// Supported on Windows, Linux/Android, and macOS/iOS via Vulkan Portability (with the Vulkan feature enabled)
  vulkan = 2,
  /// Supported on Linux/Android, the web through webassembly via WebGL, and Windows and macOS/iOS via ANGLE
  opengl = 3,
  /// MacOS & iOS only
  metal = 4,
  /// Windows +10 only
  direct_x12 = 5,
  /// browser WebGPU
  webgpu = 6,
  /// targets VULKAN METALDX12 BROWSER_WEBGPU
  highest_support = 7,
  /// targets OpenGL backend
  lowest_support = 8,
};

/// Computing devices types
enum class GPUDeviceType {
  Other = 0,
  IntegratedGpu = 1,
  DiscreteGpu = 2,
  VirtualGpu = 3,
  Cpu = 4,
};

/// this settings used to tell gpu pre information about
/// our work
enum class GPUMemorySettings {
  /// our app needs to me more performant instead being
  /// cable of allocating too much memory on gpu side
  prefer_performance = 0,
  /// our app will need to allocate memory on gpu side
  prefer_memory = 1,
  /// if you set this , you have to set customize.gpu_memory_custom
  /// this variable will be used for memory allocation in gpu
  /// it sets min and max of memory you need in gpu side
  custom_memory = 3,
};

/// this enum is used to tell to API
/// to setup GPU resources based on power saving rules or
/// not
enum class GPUPowerSettings {
  /// power and performance does not matter
  none = 0,
  /// choose based on the power saving rules
  LowPower = 1,
  /// performance is more important
  HighPerformance = 2,
};

/// this enum affects speed of the api
/// by setting how much gpu resources
/// are needed directly , if you take
/// too much which your hardware cannot
/// provide , panic happens
enum class GPUSpeedSettings {
  /// the lowest resources , supported on all backends
  lowest_speed = 0,
  /// low resources , supported on all backends expect webgl2
  /// which our api does not aim to support for now
  low_speed = 1,
  /// the default
  default_speed = 2,
  /// will be supported in next versions , for now it is equal to
  /// low_speed
  custom_speed = 3,
};

/// as config field you have to provide GPUComputingConfig which
/// represent settings which you wanted
struct GPUComputingConfig {
  /// set backend which you want
  GPUComputingBackend backend;
  /// set power settings which meets your needs
  GPUPowerSettings power;
  /// set speed settings which matches your needs
  GPUSpeedSettings speed;
  /// tell to gpu about your memory usage
  GPUMemorySettings memory;
  /// Optional Setting : if you know index of
  /// your prefered gpu device in the list
  /// gpu devices with the same backend , you can
  /// set it , to API gets resources from
  /// that gpu and use it for computing task
  /// ```text
  /// get_computing_gpu_infos function can be used to get list of them
  /// free_gpu_devices_infos function must be used from C side of the program to deallocate
  /// recived gpu infos , in Rust RAII will take care of it
  /// ```
  /// if it sets to negative value , API will automatically choose the gpu device
  int64_t gpu_index_in_backend_group;
};

/// this struct is used for advance customizations refered as
/// custom_speed settings
struct GPUSpeedCustom {
  uint32_t max_texture_dimension_1d;
  uint32_t max_texture_dimension_2d;
  uint32_t max_texture_dimension_3d;
  uint32_t max_texture_array_layers;
  uint32_t max_bind_groups;
  uint32_t max_bindings_per_bind_group;
  uint32_t max_dynamic_uniform_buffers_per_pipeline_layout;
  uint32_t max_dynamic_storage_buffers_per_pipeline_layout;
  uint32_t max_sampled_textures_per_shader_stage;
  uint32_t max_samplers_per_shader_stage;
  uint32_t max_storage_buffers_per_shader_stage;
  uint32_t max_storage_textures_per_shader_stage;
  uint32_t max_uniform_buffers_per_shader_stage;
  uint32_t max_uniform_buffer_binding_size;
  uint32_t max_storage_buffer_binding_size;
  uint32_t max_vertex_buffers;
  uint64_t max_buffer_size;
  uint32_t max_vertex_attributes;
  uint32_t max_vertex_buffer_array_stride;
  uint32_t min_uniform_buffer_offset_alignment;
  uint32_t min_storage_buffer_offset_alignment;
  uint32_t max_inter_stage_shader_components;
  uint32_t max_color_attachments;
  uint32_t max_color_attachment_bytes_per_sample;
  uint32_t max_compute_workgroup_storage_size;
  uint32_t max_compute_invocations_per_workgroup;
  uint32_t max_compute_workgroup_size_x;
  uint32_t max_compute_workgroup_size_y;
  uint32_t max_compute_workgroup_size_z;
  uint32_t max_compute_workgroups_per_dimension;
  uint32_t min_subgroup_size;
  uint32_t max_subgroup_size;
  uint32_t max_push_constant_size;
  uint32_t max_non_sampler_bindings;
};

/// with this struct you set min - max of
/// memory you will need in gpu side
struct GPUMemoryCustom {
  /// min mem needed in gpu side
  uint64_t min;
  /// max mem needed in gpu side
  uint64_t max;
};

/// this struct represents custom settings
struct GPUCustomSettings {
  /// this variable keeps custom speed settings
  GPUSpeedCustom gpu_speed_custom;
  /// this variable keeps memory custom settings
  GPUMemoryCustom gpu_memory_custom;
};

/// CKernel which will represent your GPU task
/// like how Manifest.xml does in an android
/// project
struct CKernel {
  /// set max number of workgroups in x dimension
  uint32_t x;
  /// set max number of workgroups in y dimension
  uint32_t y;
  /// set max number of workgroups in z dimension
  uint32_t z;
  /// since v4.0.0 instead of directly passing
  /// kernel code , you have to pass return
  /// value of register_computing_kernel_code
  /// to this field
  uintptr_t kernel_code_index;
  /// since v4.0.0 instead of directly passing
  /// configs of your computing task
  /// you have to create_computing_gpu_resources
  /// return value to this field
  uintptr_t config_index;
};

/// this struct is for passing
/// data based on its bind index
/// in gpu side
struct DataBinder {
  /// bind index of data in gpu side
  uint32_t bind;
  /// because data must be in uint8_t (u8 in Rust)
  /// in C you have to pass the data len this way
  ///
  /// sizeof(your type) * real_len_of_your_array / sizeof(uint8_t)
  uintptr_t data_len;
  /// address of pointer (since v5.0.0) which holds your data in memory , it must be
  /// uint8_t** (*mut *mut u8 in Rust side)
  /// in gpu side the type of this data will
  /// be set based on CKernel code you provided
  uint8_t **data;
};

/// all DataBinder types which have
/// the same @group index in your kernel
/// code must all be gathered in this
/// type
struct GroupOfBinders {
  /// index of group in your kernel code
  uint32_t group;
  /// pointer to array which all of the
  /// DataBinders from same group
  /// are gathered in
  DataBinder *datas;
  /// len of datas array
  uintptr_t datas_len;
};

/// this struct is used for storing information about
/// each device
struct GPUDeviceInfo {
  /// name of the device
  const char *name;
  /// vendor ID of the device
  uint32_t vendor;
  /// device id of the device
  uint32_t device;
  /// type of the device , GPUDeviceType
  GPUDeviceType device_type;
  /// driver name
  const char *driver;
  /// driver information
  const char *driver_info;
  /// corresponding GPUComputingBackend
  GPUComputingBackend backend;
};

/// this function stores an dynamic array of GPUDeviceInfo with len ,
/// it must be freed with free_gpu_devices_infos function after usage
struct GPUDevices {
  /// len of the dyn array
  uintptr_t len;
  /// pointer to the GPUDeviceInfo array
  GPUDeviceInfo *infos;
};

extern "C" {

/// since v4.0.0 you must create_computing_gpu_resources
/// it will return gpu_res_descriptor as uintptr_t (usize)
/// and you have to pass it as config_index value to
/// CKernel variable
uintptr_t create_computing_gpu_resources(GPUComputingConfig config, GPUCustomSettings customize);

/// since v4.0.0 your kernel code must be registered before
/// you want to use it . gpu_res_index is gpu resource descriptor
/// which you get from create_computing_gpu_resources .
uintptr_t register_computing_kernel_code(uintptr_t gpu_res_index,
                                         const char *code,
                                         const char *entry_point);

/// when your work fully finished with kernel codes and you
/// wont need to use them anymore , you can use this
/// function to cleanup all the mess which they created from memory
void free_compute_kernel_codes(uintptr_t gpu_res_index);

/// because setting CKernel config can be annoying if you just
/// want to do simple task , this function provides general
/// config which will meet most of your needs . since v4.0.0
/// this function calls create_computing_gpu_resources automatically
/// and assign its return value to config_index of your CKernel variable .
/// only use this function once in your programs , instead of using this
/// many times and causing memory leaks (well all that mem can be freed by free_compute_cache function)
/// use config_index field of CKernel variable
uintptr_t set_kernel_default_config(CKernel *kernel);

/// the simple and compact function for sending
/// your computing task to the gpu side
///
/// kernel para = CKernel type which acts as Manifest for your gpu task
/// data_for_gpu = pointer to array of GroupOfBinders which contains data which must be sent to gpu
/// gpu_data_len = len of the array of the GroupOfBinders
///
/// unlike CUDA , you dont need to copy data to gpu manually , this function does it for you
/// in the most performant possible way
///
/// if you find any bug or any problem , help us to fix it -> https://github.com/SkillfulElectro/EMCompute.git
int32_t compute(CKernel *kernel,
                GroupOfBinders *data_for_gpu,
                uintptr_t gpu_data_len);

/// since version 2.0.0 api does
/// caching for gpu resources on the memory .
/// the api does deallocate the caches
/// automatically , but in some cases
/// you might want to do it manually
/// so just call free_compute_cache();
void free_compute_cache();

/// this function returns GPUDevices of passed GPUComputingBackend
GPUDevices get_computing_gpu_infos(GPUComputingBackend backend);

/// this function is used for deallocating GPUDevices type from C side
void free_gpu_devices_infos(GPUDevices *devices);

}  // extern "C"

#endif  // EMCOMPUTE_H
