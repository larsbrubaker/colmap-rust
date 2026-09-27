// The About sheet's view of the GPU the viewport renders with. `backend_info` turns wgpu's
// `AdapterInfo` into readable labels and `BackendInfo::summary` joins the non-empty ones, so a
// browser adapter (wgpu's WebGPU backend reports only `GPUAdapterInfo.description` as the name,
// often empty, and `DeviceType::Other`) reads "WebGPU (browser)" rather than "(BrowserWebGpu,
// Other)".

/// The GPU the viewport renders with, as readable labels. Empty fields are unknown and are
/// left out of [`BackendInfo::summary`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendInfo {
    /// Adapter name, e.g. "Apple M2 Pro"; on the web, the browser's adapter description.
    pub adapter_name: String,
    /// Graphics API, e.g. "Metal", "Vulkan", "WebGPU (browser)".
    pub backend: String,
    /// Adapter class, e.g. "integrated GPU"; empty when the platform does not say.
    pub device_type: String,
    /// Driver name and version, e.g. "NVIDIA 550.54"; empty when not reported.
    pub driver: String,
}

impl BackendInfo {
    /// One line for the About sheet: the non-empty fields, comma-separated.
    pub fn summary(&self) -> String {
        let parts: Vec<&str> = [
            self.adapter_name.as_str(),
            self.backend.as_str(),
            self.device_type.as_str(),
            self.driver.as_str(),
        ]
        .into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
        if parts.is_empty() {
            "unknown adapter".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Readable name of a wgpu backend.
pub fn backend_label(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Noop => "no-op",
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Dx12 => "Direct3D 12",
        wgpu::Backend::Gl => "OpenGL / WebGL",
        wgpu::Backend::BrowserWebGpu => "WebGPU (browser)",
    }
}

/// Readable adapter class; empty for `Other`, which carries no information.
pub fn device_type_label(device_type: wgpu::DeviceType) -> &'static str {
    match device_type {
        wgpu::DeviceType::Other => "",
        wgpu::DeviceType::IntegratedGpu => "integrated GPU",
        wgpu::DeviceType::DiscreteGpu => "discrete GPU",
        wgpu::DeviceType::VirtualGpu => "virtual GPU",
        wgpu::DeviceType::Cpu => "software (CPU)",
    }
}

/// The About sheet's view of a wgpu adapter.
pub fn backend_info(info: &wgpu::AdapterInfo) -> BackendInfo {
    let driver = [info.driver.trim(), info.driver_info.trim()]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    BackendInfo {
        adapter_name: info.name.trim().to_string(),
        backend: backend_label(info.backend).to_string(),
        device_type: device_type_label(info.device_type).to_string(),
        driver,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(name: &str, backend: wgpu::Backend, ty: wgpu::DeviceType) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_string(),
            vendor: 0,
            device: 0,
            device_type: ty,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend,
            subgroup_min_size: 4,
            subgroup_max_size: 128,
            transient_saves_memory: false,
        }
    }

    #[test]
    fn browser_adapter_with_no_description_reads_webgpu_browser() {
        let info = backend_info(&adapter(
            "",
            wgpu::Backend::BrowserWebGpu,
            wgpu::DeviceType::Other,
        ));
        assert_eq!(info.summary(), "WebGPU (browser)");
    }

    #[test]
    fn browser_adapter_description_is_shown_when_present() {
        let info = backend_info(&adapter(
            "ANGLE Metal Renderer: Apple M2 Pro",
            wgpu::Backend::BrowserWebGpu,
            wgpu::DeviceType::Other,
        ));
        assert_eq!(
            info.summary(),
            "ANGLE Metal Renderer: Apple M2 Pro, WebGPU (browser)"
        );
    }

    #[test]
    fn native_adapter_lists_name_api_class_and_driver() {
        let mut raw = adapter(
            "NVIDIA GeForce RTX 4070",
            wgpu::Backend::Vulkan,
            wgpu::DeviceType::DiscreteGpu,
        );
        raw.driver = "NVIDIA".to_string();
        raw.driver_info = "550.54".to_string();
        assert_eq!(
            backend_info(&raw).summary(),
            "NVIDIA GeForce RTX 4070, Vulkan, discrete GPU, NVIDIA 550.54"
        );
        let metal = adapter(
            "Apple M2 Pro",
            wgpu::Backend::Metal,
            wgpu::DeviceType::IntegratedGpu,
        );
        assert_eq!(
            backend_info(&metal).summary(),
            "Apple M2 Pro, Metal, integrated GPU"
        );
    }

    #[test]
    fn whitespace_only_fields_are_omitted_and_all_empty_is_named() {
        let info = BackendInfo {
            adapter_name: "  ".to_string(),
            backend: String::new(),
            device_type: String::new(),
            driver: String::new(),
        };
        assert_eq!(info.summary(), "unknown adapter");
    }
}
