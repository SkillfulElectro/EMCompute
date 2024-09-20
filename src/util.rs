use crate::c_char;
use crate::CStr;


pub fn cchar_as_string(cstri : *const c_char) -> Option<String> {
    unsafe {
        if cstri.is_null() {
            None    
        } else {
            Some(CStr::from_ptr(cstri).to_string_lossy().into_owned())
        }
    }
}

