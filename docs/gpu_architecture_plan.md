# Mini-GPU RTL Bring-Up Plan

## Current Status

- [x] 32-bit ISA encoding exists.
- [x] Instruction decoder exists.
- [x] Per-lane register files exist.
- [x] Integer execute units exist.
- [x] FP32 add/sub unit exists.
- [x] FP32 multiply unit exists.
- [x] Shared FPU path exists.
- [x] Warp/block/SM wrappers exist.
- [x] Scheduled `mini_gpu_core` exists.
- [x] Multi-core `mini_gpu` wrapper exists.
- [x] Four-bank shared memory exists.
- [x] Shared memory arbitration exists.
- [x] Linear programs can run to `EXIT`.
- [x] Active lanes execute in SIMT style.
- [x] Thread metadata instructions exist: `TID`, `TIDX`, `LID`, `WID`, `BID`,
  `BDIM`, `GDIM`.
- [x] `LDG`, `STG`, `LDS`, and `STS` issue per-lane memory requests.
- [x] `LDC` constant-table loads exist.
- [x] MVP control flow exists: `PUSHM`, `PRED`, `PREDN`, `POPM`, `BRA`, `BZ`,
  `BNZ`.
- [x] Multi-warp and multi-block scheduling exists.
- [x] Block-scope `BAR` exists.
- [x] Basys3 smoke top exists.
- [x] Basys3 synthesis/implementation path exists.
- [ ] UART command front end exists.
- [ ] Full host-controlled Basys3 top exists.
- [ ] SHA256 validation path exists.
- [ ] ISA conversion support exists for numeric int/float conversion.
- [ ] PyTorch runtime path is connected to hardware.

## Hardware Interfaces

- [x] Program load: `prog_we`, `prog_addr`, `prog_wdata`.
- [x] Constant load: `const_we`, `const_addr`, `const_wdata`.
- [x] Launch control: `launch`, `base_pc`, `active_mask`, `block_dim`,
  `grid_dim`.
- [x] Core memory request/response interface.
- [x] Wrapper-level shared memory.
- [x] Status outputs: `busy`, `done`, `error`, `unsupported`,
  `divide_by_zero`, `pc`.
- [x] Debug writeback readout.
- [ ] UART program load command.
- [ ] UART data memory write command in hardware top.
- [ ] UART data memory read command in hardware top.
- [ ] UART launch command.
- [ ] UART status read command.

## Tests

- [x] Instruction decode tests.
- [x] Thread tests.
- [x] Execute tests.
- [x] Memory tests.
- [x] SM/core scheduler tests.
- [x] File-backed full-GPU program test path.
- [x] Vector-add program files.
- [x] Basys3 top smoke test.
- [ ] UART hardware command tests.
- [ ] End-to-end host-to-board program load test.
- [ ] End-to-end host-to-board data copy test.
- [ ] End-to-end host-to-board kernel launch test.

## Basys3 Goals

- [x] Fit one practical smoke design on Basys3.
- [x] Keep FP32 multiply in hardware.
- [x] Use shared FPUs to reduce LUT pressure.
- [x] Use DSPs for integer multiply where possible.
- [ ] Measure final LUT/DSP/BRAM use after UART front end.
- [ ] Confirm timing after UART front end.
- [ ] Confirm timing after SHA256 validation.

## Future Plans

- [ ] Finish UART command front end.
- [ ] Connect `gpu_comm` commands to the hardware top.
- [ ] Add host-loadable program memory flow.
- [ ] Add host-loadable constant memory flow.
- [ ] Add host-readable result memory flow.
- [ ] Add SHA256 validation flow.
- [ ] Add more board-visible debug/status modes.
- [ ] Add small neural-network kernels: vector add, ReLU, dot product, tiny MLP.
- [ ] Connect PyTorch `PrivateUse1` backend to the runtime.
- [ ] Add selected conversion instructions if software lowering needs them.
- [ ] Add larger FPGA parameter variants for more warps/cores/FPUs.
