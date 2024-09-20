#[repr(C)]
#[derive(Clone , Debug)]
/// computing backends of the api 
pub enum GPUComputingBackend {
    /// targets all of the backends 
    all = 0 , 
    /// default backend
    default_backend = 1 ,
    /// Supported on Windows, Linux/Android, and macOS/iOS via Vulkan Portability (with the Vulkan feature enabled)
    vulkan = 2,
    /// Supported on Linux/Android, the web through webassembly via WebGL, and Windows and macOS/iOS via ANGLE
    opengl = 3 ,
    /// MacOS & iOS only
    metal = 4 ,
    /// Windows +10 only
    direct_x12 = 5,
    /// browser WebGPU
    webgpu = 6 ,
    /// targets VULKAN METALDX12 BROWSER_WEBGPU
    highest_support = 7 ,
    /// targets OpenGL backend
    lowest_support = 8 ,
}

#[repr(C)]
#[derive(Clone , Debug)]
/// this enum is used to tell to API
/// to setup GPU resources based on power saving rules or
/// not
pub enum GPUPowerSettings {
    /// power and performance does not matter
    none = 0 ,
    /// choose based on the power saving rules
    LowPower = 1 ,
    /// performance is more important
    HighPerformance = 2 ,
}

#[repr(C)]
#[derive(Clone , Debug)]
/// this settings used to tell gpu pre information about 
/// our work 
pub enum GPUMemorySettings {
    /// our app needs to me more performant instead being 
    /// cable of allocating too much memory on gpu side
    prefer_performance = 0 ,
    /// our app will need to allocate memory on gpu side 
    prefer_memory = 1 ,
    /// if you set this , you have to set customize.gpu_memory_custom 
    /// this variable will be used for memory allocation in gpu 
    /// it sets min and max of memory you need in gpu side 
    custom_memory = 3 ,
}

#[repr(C)]
#[derive(Clone , Debug)]
/// this enum affects speed of the api 
/// by setting how much gpu resources
/// are needed directly , if you take 
/// too much which your hardware cannot 
/// provide , panic happens
pub enum GPUSpeedSettings {
    /// the lowest resources , supported on all backends
    lowest_speed = 0 ,
    /// low resources , supported on all backends expect webgl2
    /// which our api does not aim to support for now 
    low_speed = 1 ,
    /// the default
    default_speed = 2 ,
    /// will be supported in next versions , for now it is equal to 
    /// low_speed
    custom_speed = 3 ,
}

#[repr(C)]
#[derive(Debug, Clone)]
/// as config field you have to provide GPUComputingConfig which
/// represent settings which you wanted
pub struct GPUComputingConfig {
    /// set backend which you want 
    pub backend : GPUComputingBackend ,
    /// set power settings which meets your needs 
    pub power : GPUPowerSettings ,
    /// set speed settings which matches your needs
    pub speed : GPUSpeedSettings ,
    /// tell to gpu about your memory usage 
    pub memory : GPUMemorySettings ,
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
    pub gpu_index_in_backend_group : i64 ,
}
