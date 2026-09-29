# burnin

GPU burn-in and stress testing. burnin keeps a GPU busy with large matrix multiplies and checks that every result it computes is correct, so you can find faulty hardware before it goes into service.

> **Status: early prototype.** It tests one NVIDIA GPU at a time on Linux. See the [roadmap](#roadmap) for what's coming.

## How it works

1. burnin fills most of the GPU's memory with result matrices.
2. Every result is the product of the same two inputs, so all of them should be bit-for-bit identical.
3. Each pass recomputes a reference result, then recomputes the rest in chunks and compares each chunk against the reference on the GPU.
4. Any difference means the hardware got a calculation wrong.

Chunks are sized to take about 1.5 seconds each. That keeps progress lines, Ctrl-C and error reports prompt on both slow and fast GPUs.

## Requirements

- Linux on x86_64 or aarch64.
- An NVIDIA GPU and driver.
- The CUDA 13 libraries cuBLAS and NVRTC. CUDA 12 may work but hasn't been tested yet.

The build itself doesn't need the CUDA toolkit: burnin loads the CUDA libraries when it starts.

## Build

```sh
cargo build --release
```

The binary is `target/release/burnin`.

## Usage

```sh
burnin list                     # GPUs burnin can see
burnin probe                    # device, memory and telemetry details
burnin run 10m                  # stress GPU 0 for ten minutes
burnin run 1h -d 1 -p fp64      # GPU 1, double precision, one hour
burnin run 30m -m 50%           # use half of the usable memory
burnin run 30s --inject-fault   # corrupt one result on purpose to check detection
```

Durations accept seconds, or units such as `90s`, `10m` and `2h`. Memory accepts a percentage such as `90%`, or a size such as `16G`, `512M` or `4096` (MiB when there's no unit).

Exit status:
- `0` when every result matched.
- `1` when mismatches were found.
- `2` on error.

### Unified-memory GPUs

On GPUs that share system RAM with the CPU, such as the GB10 in DGX Spark, CUDA's figure for free memory leaves out reclaimable page cache. burnin sizes its memory from the system's available memory instead, and leaves a reserve for the OS. `burnin probe` shows both figures and the budget it would use.

## Roadmap

- Every GPU in a machine at once, each in its own process so a hung GPU can be stopped without losing the others.
- A watchdog that flags GPUs that stop making progress.
- More precisions: TF32, FP16, BF16 and FP8.
- Telemetry in the report: temperature, power, clocks, throttling, ECC errors and driver error events.
- JSON output for automation.
- Prebuilt x86_64 and aarch64 release binaries.
- An Apple Silicon backend using Metal.

## Contributing

Development works on any platform. The CUDA backend is Linux-only, but it can be type-checked from elsewhere:

```sh
cargo test
cargo clippy --all-targets
rustup target add aarch64-unknown-linux-gnu
cargo check --target aarch64-unknown-linux-gnu
```

GPU runs need a Linux machine with an NVIDIA GPU. When you report a problem, please include the output of `burnin probe`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
