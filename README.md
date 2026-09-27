# colmap-rust

A pure Rust port of [COLMAP](https://github.com/colmap/colmap) 4.2.0 — Structure-from-Motion and
Multi-View Stereo: photos → camera poses → sparse point cloud → dense depth → mesh.

- **Pure Rust, MIT, commercial-safe.** No C/C++, no native dependencies; the core builds for
  WebAssembly.
- **Built on [agg-gui](https://github.com/larsbrubaker/agg-gui).** One app, rendered with
  agg-gui's wgpu renderer, runs natively on Windows, macOS and Linux and in the browser
  (WebGPU).
- **COLMAP's tests are the spec.** Every `*_test.cc` is ported 1:1, backed by fixtures from the
  pinned `pycolmap` wheel.

**Live demo:** https://larsbrubaker.github.io/colmap-rust/ (requires a WebGPU browser)

Status: just started — see [PORTING_PLAN.md](PORTING_PLAN.md). Sister project:
[colmap-sharp](https://github.com/larsbrubaker/colmap-sharp) (the same port in C#).

## Build

```bash
cargo run -p colmap-native      # desktop app
cargo test --workspace          # tests
web/build.sh                    # wasm bundle into web/
```

## License

MIT (see [LICENSE](LICENSE)); ported third-party code keeps its notices in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
