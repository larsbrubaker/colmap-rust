//! The compute seam's value types: port of colmap-sharp `ColmapSharp/Compute/ComputeDescriptors.cs`.
//! Buffer kinds, binding types, a kernel's bind group layout and source, and one bind group entry.
//! The device contract that uses them is `device.rs`.

/// What a buffer can be bound as. Port of `ComputeBufferKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeBufferKind {
    /// `var<storage, read>` or `var<storage, read_write>`.
    Storage,
    /// `var<uniform>`.
    Uniform,
}

/// How a kernel binds one buffer. Port of `ComputeBindingType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeBindingType {
    /// `var<uniform>`: needs a [`ComputeBufferKind::Uniform`] buffer.
    Uniform,
    /// `var<storage, read>`: needs a [`ComputeBufferKind::Storage`] buffer.
    ReadOnlyStorage,
    /// `var<storage, read_write>`: needs a [`ComputeBufferKind::Storage`] buffer, and may not be
    /// bound anywhere else in the same dispatch.
    Storage,
}

impl ComputeBindingType {
    /// The buffer kind a binding of this type needs.
    pub fn buffer_kind(self) -> ComputeBufferKind {
        match self {
            ComputeBindingType::Uniform => ComputeBufferKind::Uniform,
            ComputeBindingType::ReadOnlyStorage | ComputeBindingType::Storage => {
                ComputeBufferKind::Storage
            }
        }
    }
}

/// One `@group(g) @binding(b)` of a kernel's layout. Port of `ComputeKernelBinding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeKernelBinding {
    pub group: u32,
    pub binding: u32,
    pub ty: ComputeBindingType,
}

impl ComputeKernelBinding {
    pub const fn new(group: u32, binding: u32, ty: ComputeBindingType) -> Self {
        Self { group, binding, ty }
    }
}

/// A kernel to compile: WGSL source, entry point and an explicit bind group layout.
/// Port of `ComputeKernelDescriptor`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeKernelDescriptor {
    /// Debug name, carried into error messages.
    pub label: String,
    /// The WGSL module.
    pub source: String,
    /// The `@compute` function to run.
    pub entry_point: String,
    /// Every binding the WGSL declares. Groups run 0, 1, ... with no gap.
    pub bindings: Vec<ComputeKernelBinding>,
}

/// One entry of a bind group: `size` bytes of `buffer` from `offset`, at binding `binding`.
/// Port of `ComputeBufferBinding`; generic over the device's buffer handle.
#[derive(Debug)]
pub struct ComputeBufferBinding<'a, B> {
    pub binding: u32,
    pub buffer: &'a B,
    pub offset: u64,
    pub size: u64,
}

impl<'a, B> ComputeBufferBinding<'a, B> {
    pub fn new(binding: u32, buffer: &'a B, offset: u64, size: u64) -> Self {
        Self {
            binding,
            buffer,
            offset,
            size,
        }
    }
}

impl<B> Clone for ComputeBufferBinding<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for ComputeBufferBinding<'_, B> {}
