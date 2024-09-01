# MIT License

# Copyright (c) 2024 ElectroMutex

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t, intptr_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, uintptr_t
cdef extern from *:
  ctypedef bint bool
  ctypedef struct va_list

cdef extern from *:

  # NOTE : on linux machines memory leak might happen if you use
  # vulkan backend until NVIDIA drivers for linux get fixed .
  #
  #
  # computing backends of the api
  cdef enum GPUComputingBackend:
    # targets all of the backends
    all # = 0,
    # default backend
    default_backend # = 1,
    # Supported on Windows, Linux/Android, and macOS/iOS via Vulkan Portability (with the Vulkan feature enabled)
    vulkan # = 2,
    # Supported on Linux/Android, the web through webassembly via WebGL, and Windows and macOS/iOS via ANGLE
    opengl # = 3,
    # MacOS & iOS only
    metal # = 4,
    # Windows +10 only
    direct_x12 # = 5,
    # browser WebGPU
    webgpu # = 6,
    # targets VULKAN METALDX12 BROWSER_WEBGPU
    highest_support # = 7,
    # targets OpenGL backend
    lowest_support # = 8,

  # this settings used to tell gpu pre information about
  # our work
  cdef enum GPUMemorySettings:
    # our app needs to me more performant instead being
    # cable of allocating too much memory on gpu side
    prefer_performance # = 0,
    # our app will need to allocate memory on gpu side
    prefer_memory # = 1,
    # if you set this , you have to set customize.gpu_memory_custom
    # this variable will be used for memory allocation in gpu
    # it sets min and max of memory you need in gpu side
    custom_memory # = 3,

  # this enum is used to tell to API
  # to setup GPU resources based on power saving rules or
  # not
  cdef enum GPUPowerSettings:
    # power and performance does not matter
    none # = 0,
    # choose based on the power saving rules
    LowPower # = 1,
    # performance is more important
    HighPerformance # = 2,

  # this enum affects speed of the api
  # by setting how much gpu resources
  # are needed directly , if you take
  # too much which your hardware cannot
  # provide , panic happens
  cdef enum GPUSpeedSettings:
    # the lowest resources , supported on all backends
    lowest_speed # = 0,
    # low resources , supported on all backends expect webgl2
    # which our api does not aim to support for now
    low_speed # = 1,
    # the default
    default_speed # = 2,
    # will be supported in next versions , for now it is equal to
    # low_speed
    custom_speed # = 3,

  # as config field you have to provide GPUComputingConfig which
  # represent settings which you wanted
  cdef struct GPUComputingConfig:
    # set backend which you want
    GPUComputingBackend backend;
    # set power settings which meets your needs
    GPUPowerSettings power;
    # set speed settings which matches your needs
    GPUSpeedSettings speed;
    # tell to gpu about your memory usage
    GPUMemorySettings memory;

  # this struct is used for advance customizations refered as
  # custom_speed settings
  cdef struct GPUSpeedCustom:
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

  # with this struct you set min - max of
  # memory you will need in gpu side
  cdef struct GPUMemoryCustom:
    # min mem needed in gpu side
    uint64_t min;
    # max mem needed in gpu side
    uint64_t max;

  # this struct represents custom settings
  cdef struct GPUCustomSettings:
    # this variable keeps custom speed settings
    GPUSpeedCustom gpu_speed_custom;
    # this variable keeps memory custom settings
    GPUMemoryCustom gpu_memory_custom;

  # CKernel which will represent your GPU task
  # like how Manifest.xml does in an android
  # project
  cdef struct CKernel:
    # set max number of workgroups in x dimension
    uint32_t x;
    # set max number of workgroups in y dimension
    uint32_t y;
    # set max number of workgroups in z dimension
    uint32_t z;
    # this is a kernel code which must be in wgsl for now
    # more shading languages will be supported in the future
    const char *code;
    # this part in the code , tell to the api which
    # function in the code must be called by gpu
    # when the task is sent to gpu
    const char *code_entry_point;
    # by setting config you can customize behavior of the
    # gpu
    GPUComputingConfig config;
    # since v3.0.0 when you set any of configs to
    # custom , you can set custom configs to it
    # by setting your customizations on equivalent field of
    # customize
    GPUCustomSettings customize;
    # since v3.0.0 caching method is based
    # on the setting_cache_index
    # this changes happened to caching
    # by granting the full control to the user .
    # when you for first time use a config and customize , api will
    # store them on dynamic array and set this index automatically .
    # NOTE : 1. if you use a config for first time use negative index - 2. if
    # you used a config before keep its index to use it
    int32_t setting_cache_index;

  # this struct is for passing
  # data based on its bind index
  # in gpu side
  cdef struct DataBinder:
    # bind index of data in gpu side
    uint32_t bind;
    # because data must be in uint8_t (u8 in Rust)
    # in C you have to pass the data len this way
    #
    # sizeof(your type) * real_len_of_your_array / sizeof(uint8_t)
    uintptr_t data_len;
    # pointer to your data  in memory , it must be
    # uint8_t* (*mut u8 in Rust side)
    # in gpu side the type of this data will
    # be set based on CKernel code you provided
    uint8_t *data;

  # all DataBinder types which have
  # the same @group index in your kernel
  # code must all be gathered in this
  # type
  cdef struct GroupOfBinders:
    # index of group in your kernel code
    uint32_t group;
    # pointer to array which all of the
    # DataBinders from same group
    # are gathered in
    DataBinder *datas;
    # len of datas array
    uintptr_t datas_len;

  # because setting CKernel config can be annoying if you just
  # want to do simple task , this function provides general
  # config which will meet most of your needs . this function
  # sets setting_cache_index to -1 , if you used this function before
  # instead of using this function again . you can use the index from
  # before . also if you dont use that index it will cause extra gpu
  # resource creation
  void set_kernel_default_config(CKernel *kernel);

  # the simple and compact function for sending
  # your computing task to the gpu side
  #
  # kernel para = CKernel type which acts as Manifest for your gpu task
  # data_for_gpu = pointer to array of GroupOfBinders which contains data which must be sent to gpu
  # gpu_data_len = len of the array of the GroupOfBinders
  #
  # unlike CUDA , you dont need to copy data to gpu manually , this function does it for you
  # in the most performant possible way
  #
  # if you find any bug or any problem , help us to fix it -> https://github.com/SkillfulElectro/EMCompute.git
  int32_t compute(CKernel *kernel,
                  GroupOfBinders *data_for_gpu,
                  uintptr_t gpu_data_len);

  # since version 2.0.0 api does
  # caching for gpu resources on the memory .
  # the api does deallocate the caches
  # automatically , but in some cases
  # you might want to do it manually
  # so just call free_compute_cache();
  void free_compute_cache();
