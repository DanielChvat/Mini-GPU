# Mini-GPU RTL Bring-Up Plan

## Summary

This is the living roadmap for bringing the Mini-GPU RTL up on a Basys3 FPGA.
The immediate milestone is "one instruction first": prove that an instruction can
be fetched, issued to the existing 4-lane SIMT execution path, retired, and made
visible through status/readback signals. After that, expand into memory
operations, UART loading, scheduling, and finally the host runtime/PyTorch
`privateuse1` path.

Current RTL already has decode, per-lane register files, integer/float execute
units, warp/block/SM wrappers, a 4-bank scratchpad memory, a scheduled GPU core,
and a multi-core GPU wrapper with integrated shared memory arbitration. Missing
pieces are UART command front end, full Basys3 integration around UART/control,
shared/global memory separation, and host runtime integration.

## Current Status

- `mini_gpu_core` can load and run linear programs ending in `EXIT`.
- Active lanes execute one shared instruction in SIMT style.
- Thread metadata instructions are implemented for the current one-block core:
  `TID`, `TIDX`, `LID`, `WID`, `BID`, `BDIM`, and `GDIM`.
- `LID`/`TID` allow active lanes to produce different register values from the
  same instruction stream.
- `LDG/STG/LDS/STS` now issue per-lane memory requests through the core boundary.
  The memory module is external/shared and is connected beside the core, not
  instantiated privately inside each core.
- The full `mini_gpu` wrapper now instantiates the shared 4-bank memory and
  arbitrates memory traffic from multiple cores into the 4-lane memory module.
- `LDC` is implemented through a small per-core constant table loaded with
  `const_we`, `const_addr`, and `const_wdata`.
- The core accumulates per-lane memory completions and writebacks, so bank
  conflicts that complete lanes over multiple cycles retire as one instruction.
- MVP control flow is implemented for the single-warp core: `PUSHM`, `PRED`,
  `PREDN`, `POPM`, `BRA`, `BZ`, and `BNZ`. Predication updates the active lane
  mask, and branches use PC-relative `imm14` offsets.
- `mini_gpu_core` now supports parameterized multi-warp and multi-block
  scheduling with per-warp PC, active mask, mask stack, done state, and
  round-robin ready-warp selection.
- `BAR` now waits at block scope: each warp parks at the barrier until the other
  non-exited warps in that block arrive, then all waiting warps advance.
- `mini_gpu` wraps multiple scheduled cores and assigns each core its own base
  block ID while broadcasting program and constant-table loads.

## Key Changes

- `mini_gpu_core` owns the current instruction memory, constant memory,
  launch/reset/status, per-warp state, and writeback capture. `mini_gpu`
  broadcasts program loads into each core's replicated instruction memory.
- The scheduler runs:
  `IDLE -> CLEAR -> SCHEDULE -> FETCH -> ISSUE -> WAIT_DONE -> SCHEDULE/DONE`.
- Linear programs ending in `EXIT`, memory operations, predication, branches,
  `BAR`, and thread metadata instructions are covered by RTL tests.
- A Basys3 smoke top preloads a tiny program, launches it after reset, and maps
  `done`, `busy`, `error`, and low result bits onto LEDs.
- Next, add UART command handling around program/constant/data loading and then
  build the host runtime path.

## Public Interfaces

- Program load: `prog_we`, `prog_addr`, `prog_wdata`; these currently write the
  per-core instruction memories through the `mini_gpu` broadcast bus.
- Launch control: `launch`, `base_pc`, `active_mask`, `block_dim`, `grid_dim`.
- Core memory interface: per-lane `mem_req_valid`, `mem_req_write`,
  `mem_req_addr`, `mem_req_wdata`, `mem_req_ready`, `mem_resp_valid`, and
  `mem_resp_rdata`.
- Full GPU wrapper memory: `mini_gpu` owns the shared `global_memory`; tests and
  future UART/control logic can load/read that memory through wrapper-level
  control paths rather than core-private memory ports.
- Status: `busy`, `done`, `error`, `unsupported`, `divide_by_zero`, `pc`.
- Debug/readback: last writeback lane mask, destination register, and per-lane
  writeback data.
- Preserve the current 32-bit ISA encoding in `hardware/rtl/include/minigpu_isa.vh`
  and `compiler/isa_to_bin.py`.
- Keep the existing `gpu_comm` UART packet API as the future host boundary unless
  hardware command semantics require a deliberate protocol revision.

## Test Plan

- Add RTL tests for the new core:
  - single `MOVI` fetch/issue/retire captures expected per-lane writeback
  - two-instruction `MOVI` then `ADD` retires in order
  - unsupported opcode sets error/status
  - active mask disables inactive lanes
  - `LDC` reads the per-core constant table
  - multi-warp/multi-block scheduling preserves `TID`, `WID`, and `BID`
  - block-scope `BAR` releases only after all active warps in the block arrive
  - `mini_gpu` launches multiple scheduled cores with distinct base block IDs
- After memory integration:
  - `STG` writes lane values to banked memory
  - `LDG` reads back into registers
  - same-bank conflicts stall/retry instead of dropping lanes
  - integrated `mini_gpu` vector-add test seeds A/B, launches two cores, and
    checks C in shared memory
- After control-flow integration:
  - `PUSHM/PRED/POPM` masks a subset of lanes
  - `PREDN` activates the inverse subset after restoring the prior mask
  - `BZ/BRA` update the program counter and skip the expected instructions
- For file-backed full-GPU simulations:
  - assemble `.asm` programs with `hardware/tools/asm_to_tb_hex.py`
  - run `mini_gpu_program_tb` with `+program_hex=<path>` to load instructions
    from a `$readmemh` file
  - use optional sidecar files with `+const_file=<path>`, `+mem_init=<path>`,
    and `+mem_expect=<path>` for constant-table values, initial data memory,
    and expected memory checks
- For Basys3 bring-up:
  - synthesize/place/route without timing/resource surprises
  - LED smoke program reaches `done`
  - UART launch/read-status loop works after the command front end lands

## Assumptions

- First milestone is simulation-first one-instruction execution, followed by a
  minimal Basys3 LED smoke test.
- Basys3 constraints/project files are not currently present and should be added
  when the smoke top is taken into Vivado.
- The first useful neural-network demo should target small fixed-shape kernels
  such as vector add, ReLU, dot products, and tiny MNIST MLP steps before broader
  PyTorch coverage.
