# burnin

GPU burn-in and stress testing. burnin keeps every GPU in a machine busy with large matrix multiplies and checks that every result is correct, so you can find faulty hardware before it goes into service.

> **Status: early prototype.** It tests NVIDIA GPUs on Linux. See the [roadmap](#roadmap) for what's coming.

## How it works

1. burnin fills most of the GPU's memory with result matrices.
2. Every result is the product of the same two inputs, so all of them should be bit-for-bit identical.
3. Each pass recomputes a reference result, then recomputes the rest in chunks and compares each chunk against the reference on the GPU.
4. Any difference means the hardware got a calculation wrong.

Chunks are sized to take about 1.5 seconds each. That keeps progress lines, Ctrl-C and error reports prompt on both slow and fast GPUs.

Every GPU is tested at the same time, each in its own child process. A supervisor prints each GPU's progress and warns when a GPU goes quiet. If a GPU makes no progress for `--hang-timeout` (3 minutes by default), the supervisor gives up on it and kills its process, and the other GPUs carry on. When the run ends, each GPU gets its own verdict:

| Verdict | Meaning |
|---|---|
| `PASS` | Every result matched. |
| `FAIL` | Some results differed, the GPU hit an error during the run, or it reported uncorrected ECC memory errors or a critical driver error (Xid). |
| `HUNG` | The GPU stopped making progress, or didn't finish its last chunk within the grace period after the run ended. |
| `ERROR` | The GPU couldn't be set up, so it wasn't tested. |

Progress lines show each GPU's temperature, power, SM clock and any throttling. The summary adds the peak temperature, average and peak power, average and lowest clock, and every throttle reason seen. It also lists ECC memory errors and critical driver errors (Xid events) that occurred during the run. Corrected ECC errors are reported but don't fail a GPU. Fields a GPU doesn't report are left out; `burnin probe` shows which ones a GPU supports.

`--isolation thread` tests every GPU in threads of a single process instead. That's easier to debug, but a hung GPU can only be reported, not stopped.

## Requirements

- Linux on x86_64 or aarch64.
- An NVIDIA GPU and driver.
- The CUDA 13 libraries cuBLAS and NVRTC, plus cuBLASLt for `fp8`. CUDA 12 may work but hasn't been tested yet.

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
burnin run 10m                  # stress every GPU for ten minutes
burnin run 1h -d 1 -p fp64      # GPU 1 only, double precision, one hour
burnin run 30m -p bf16          # bfloat16 on the tensor cores
burnin run 30m -d 0,2           # GPUs 0 and 2
burnin run 30m -m 50%           # use half of the usable memory
burnin run 30s --inject-fault   # corrupt one result on purpose to check detection
```

Durations accept seconds, or units such as `90s`, `10m` and `2h`. Memory accepts a percentage such as `90%`, or a size such as `16G`, `512M` or `4096` (MiB when there's no unit).

`-p` picks the precision of the matrix multiplies. The lower precisions run on the tensor cores, so each needs a GPU with a new enough compute capability. A GPU that's too old for the chosen precision isn't tested, and is reported as `ERROR`.

| Precision | Inputs | Results | Needs |
|---|---|---|---|
| `fp32` (default) | FP32 | FP32 | Any GPU |
| `tf32` | FP32, multiplied as TF32 | FP32 | Compute capability 8.0 (Ampere) or newer |
| `fp16` | FP16 | FP16 | Compute capability 7.0 (Volta) or newer |
| `bf16` | BF16 | BF16 | Compute capability 8.0 (Ampere) or newer |
| `fp64` | FP64 | FP64 | Any GPU |
| `fp8` | FP8 E4M3 | FP32 | Compute capability 8.9 (Ada) or newer, and a matrix size that's a multiple of 16 |

`fp16`, `bf16` and `fp8` accumulate in FP32. `burnin list` shows each GPU's compute capability, for example `sm_89` for 8.9.

Exit status:
- `0` when every GPU passed.
- `1` when any GPU failed or hung.
- `2` when a GPU couldn't be tested, or burnin itself hit an error.

### JSON output

`burnin run --format json` writes [JSON Lines](https://jsonlines.org) to stdout, one object per event, instead of text. Each object has an `event` field:

| Event | When |
|---|---|
| `start` | The run begins: burnin version, precision and GPU count. |
| `gpu` | Once for each GPU to be tested. |
| `ready` | A GPU is set up and testing has begun. |
| `running` | Every GPU is ready and the clock has started. |
| `progress` | Every `--report-every`: pass, throughput, mismatches and telemetry for each GPU. |
| `mismatch`, `hardware`, `stalled`, `recovered`, `hung`, `error` | As they happen. |
| `summary` | Always last: the overall `result` and `exit_status`, and each GPU's verdict, throughput, telemetry and hardware errors. |

To act only on the outcome, read the last line:

```sh
burnin run 10m --format json | tail -n 1 | jq '.gpus[] | {gpu, verdict, detail}'
```

Errors that stop burnin before testing starts still go to stderr as text, with exit status 2.

### Unified-memory GPUs

On GPUs that share system RAM with the CPU, such as the GB10 in DGX Spark, CUDA's figure for free memory leaves out reclaimable page cache. burnin sizes its memory from the system's available memory instead, and leaves a reserve for the OS. If several such GPUs are tested at once, they split that memory between them. `burnin probe` shows both figures and the budget it would use.

## Roadmap

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
