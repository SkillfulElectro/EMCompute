use crate::c_char;

use crate::
{GPUComputingBackend , 
    GPUPowerSettings , 
    GPUSpeedSettings , 
    GPUMemorySettings , 
    GPUComputingConfig};



#[repr(C)]
#[derive(Clone , Debug)]
/// Computing devices types
pub enum GPUDeviceType {
    Other = 0,
    IntegratedGpu = 1,
    DiscreteGpu = 2,
    VirtualGpu = 3,
    Cpu = 4,
}

impl From<wgpu::DeviceType> for GPUDeviceType {
    fn from(item : wgpu::DeviceType) -> Self {
        match item {
            wgpu::DeviceType::Other => GPUDeviceType::Other ,
            wgpu::DeviceType::IntegratedGpu => GPUDeviceType::IntegratedGpu ,
            wgpu::DeviceType::DiscreteGpu => GPUDeviceType::DiscreteGpu ,
            wgpu::DeviceType::VirtualGpu => GPUDeviceType::VirtualGpu ,
            wgpu::DeviceType::Cpu => GPUDeviceType::Cpu ,
        }
    }
}

#[repr(C)]
#[derive(Clone , Debug)]
/// this struct is used for storing information about
/// each device 
pub struct GPUDeviceInfo {
    /// name of the device
    pub name: *const c_char,
    /// vendor ID of the device 
    pub vendor: u32,
    /// device id of the device 
    pub device: u32,
    /// type of the device , GPUDeviceType
    pub device_type: GPUDeviceType,
    /// driver name 
    pub driver: *const c_char,
    /// driver information
    pub driver_info: *const c_char,
    /// corresponding GPUComputingBackend
    pub backend: GPUComputingBackend,
}

impl Drop for GPUDeviceInfo {
    fn drop(&mut self){
        unsafe {
            let name = std::ffi::CString::from_raw(self.name as *mut c_char);
            let driver = std::ffi::CString::from_raw(self.driver as *mut c_char);
            let driver_info = std::ffi::CString::from_raw(self.driver_info as *mut c_char);
        }
    }
}

#[repr(C)]
#[derive(Clone , Debug)]
/// this function stores an dynamic array of GPUDeviceInfo with len , 
/// it must be freed with free_gpu_devices_infos function after usage
pub struct GPUDevices {
    /// len of the dyn array
    pub len : usize ,
    /// pointer to the GPUDeviceInfo array
    pub infos : *mut GPUDeviceInfo ,
}

// Rust RAII
impl Drop for GPUDevices {
    fn drop(&mut self) {
        unsafe {
            let _tmp_vec = Vec::from_raw_parts(self.infos , self.len , self.len);
        }
    }
}

fn wgpu_backend_to_gpucomputingbackend(backend : wgpu::Backend) -> GPUComputingBackend {
    match backend {
        wgpu::Backend::Vulkan => {
            GPUComputingBackend::vulkan
        },
        wgpu::Backend::Gl => {
            GPUComputingBackend::opengl
        },
        wgpu::Backend::Empty => {
            GPUComputingBackend::default_backend
        },
        wgpu::Backend::Metal => {
            GPUComputingBackend::metal
        },
        wgpu::Backend::Dx12 => {
            GPUComputingBackend::direct_x12
        },
        wgpu::Backend::BrowserWebGpu => {
            GPUComputingBackend::webgpu
        },
    }
}

#[no_mangle]
/// this function returns GPUDevices of passed GPUComputingBackend
pub extern "C" fn get_computing_gpu_infos(backend : GPUComputingBackend) -> GPUDevices {

    let backender = match backend {
        GPUComputingBackend::vulkan => {
            wgpu::Backends::VULKAN
        },
        GPUComputingBackend::opengl => {
            wgpu::Backends::GL
        },
        GPUComputingBackend::all => {
            wgpu::Backends::all()
        },
        GPUComputingBackend::default_backend => {
            wgpu::Backends::default()
        },
        GPUComputingBackend::metal => {
            wgpu::Backends::METAL 
        },
        GPUComputingBackend::direct_x12 => {
            wgpu::Backends::DX12
        },
        GPUComputingBackend::highest_support => {
            wgpu::Backends::PRIMARY
        },
        GPUComputingBackend::lowest_support => {
            wgpu::Backends::SECONDARY
        },
        GPUComputingBackend::webgpu => {
            wgpu::Backends::BROWSER_WEBGPU
        },
    };

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor{
        backends : backender ,
        ..Default::default()
    });

    let adapters = instance.enumerate_adapters(backender);
    let mut devices_keeper : Vec<GPUDeviceInfo> = Vec::new();
    for adapter in adapters {
        let info = adapter.get_info();
        let name = std::ffi::CString::new(info.name).unwrap();
        let driver = std::ffi::CString::new(info.driver).unwrap();
        let driver_info = std::ffi::CString::new(info.driver_info).unwrap();
        devices_keeper.push(GPUDeviceInfo{
            vendor : info.vendor ,
            device : info.device ,
            name : name.into_raw() ,
            driver : driver.into_raw() ,
            driver_info : driver_info.into_raw() ,
            device_type : GPUDeviceType::from(info.device_type) ,
            backend : wgpu_backend_to_gpucomputingbackend(info.backend) ,
        });

    }

    let len = devices_keeper.len();
    let ptr = devices_keeper.as_ptr();
    std::mem::forget(devices_keeper);
    GPUDevices {
        len : len ,
        infos : ptr as *mut GPUDeviceInfo ,
    }
}

#[no_mangle]
/// this function is used for deallocating GPUDevices type from C side
pub extern "C" fn free_gpu_devices_infos(devices : *mut GPUDevices) {
    
    unsafe {
        let mut devices = &mut *devices;
        let tmp_vec = Vec::from_raw_parts(devices.infos , devices.len , devices.len);
    }
    /**/
}
