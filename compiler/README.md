# Mini-GPU Compiler Encoding

Mini-GPU uses a fixed-width 32-bit instruction format. The encoding is intentionally RISC-V-like: simple register fields, a small immediate, and PC-relative branches.

## Instruction Word

| Bits | Field | Width | Description |
|---|---:|---:|---|
| `[31:26]` | `opcode` | 6 | Instruction opcode |
| `[25:22]` | `rd` | 4 | Destination register |
| `[21:18]` | `rs1` | 4 | First source register / base register |
| `[17:14]` | `rs2` | 4 | Second source register / store value |
| `[13:0]` | `imm14` | 14 | Signed immediate, branch offset, or constant ID |

Registers are `r0` through `r15`. Immediates are signed 14-bit values. Raw `.bin` output is big-endian per instruction word, so hex word `0C000000` is emitted as bytes `0C 00 00 00`.

For register-to-register typed ALU instructions, `imm14[2:0]` carries a data-format tag. Existing un-suffixed integer assembly keeps encoding as `I32`.

| Format | ID | Meaning |
|---|---:|---|
| `I32` | `0` | signed 32-bit integer |
| `I16` | `1` | signed 16-bit integer in low register bits |
| `I8` | `2` | signed 8-bit integer in low register bits |
| `FP32` | `3` | 32-bit floating point |
| `FP16` | `4` | 16-bit floating point in low register bits |
| `FP8` | `5` | 8-bit floating point in low register bits |

The CUDA subset compiler tracks scalar and pointer element types for `int`, `int16_t`, `int8_t`, `float`, `half`, and `fp8_e4m3`. It emits typed IR such as `add.fp32` or `add.i8`, which lowers to ISA suffixes such as `FADD.FP32` or `ADD.I8`.

Typed memory operations are represented in IR, for example `load_global.fp16`, but the current ISA still encodes loads/stores as `LDG/STG` without a width field. Packed byte/halfword/fp8 memory behavior must be added in the memory unit before these types are fully correct end to end.

## Instruction Forms

All forms use the same 32-bit layout. Unused fields are encoded as zero.

### Register

Used for register-to-register ALU operations.

```text
ADD rd, rs1, rs2
ADD.I16 rd, rs1, rs2
MUL rd, rs1, rs2
FADD.FP32 rd, rs1, rs2
SLT rd, rs1, rs2
```

| Field | Value |
|---|---|
| `opcode` | operation opcode |
| `rd` | destination register |
| `rs1` | first input register |
| `rs2` | second input register |
| `imm14` | `0` for un-suffixed `I32`, otherwise low bits hold the format ID |

Example:

```text
ADD r6, r3, r0
```

### Immediate

Used when one operand is a small constant.

```text
ADDI rd, rs1, imm
MOVI rd, imm
```

| Field | `ADDI rd, rs1, imm` | `MOVI rd, imm` |
|---|---|---|
| `opcode` | `ADDI` | `MOVI` |
| `rd` | destination register | destination register |
| `rs1` | input register | `0` |
| `rs2` | `0` | `0` |
| `imm14` | signed immediate | signed immediate |

Example:

```text
MOVI r5, 4
```

### Constant Load

Used for kernel arguments, shared memory bases, and other constant-table values.

```text
LDC rd, ARG_A
LDC rd, SHARED_TILE
```

| Field | Value |
|---|---|
| `opcode` | `LDC` |
| `rd` | destination register |
| `rs1` | `0` |
| `rs2` | `0` |
| `imm14` | constant ID |

The assembler assigns IDs to symbolic constants. `ARG_*` IDs start at `0`; `SHARED_*` IDs start at `32`.

### Load

Used for global and shared memory loads.

```text
LDG rd, [rs1 + imm]
LDS rd, [rs1 + imm]
```

| Field | Value |
|---|---|
| `opcode` | `LDG` or `LDS` |
| `rd` | destination register |
| `rs1` | base address register |
| `rs2` | `0` |
| `imm14` | signed offset |

### Store

Used for global and shared memory stores.

```text
STG [rs1 + imm], rs2
STS [rs1 + imm], rs2
```

| Field | Value |
|---|---|
| `opcode` | `STG` or `STS` |
| `rd` | `0` |
| `rs1` | base address register |
| `rs2` | value register |
| `imm14` | signed offset |

### Thread Metadata

Used to read execution metadata.

```text
TID rd
TIDX rd
BID rd
BDIM rd
```

| Field | Value |
|---|---|
| `opcode` | metadata opcode |
| `rd` | destination register |
| `rs1` | `0` |
| `rs2` | `0` |
| `imm14` | `0` |

### Predication

Used for active-mask control.

```text
PUSHM
PRED rs1
POPM
```

| Instruction | Field use |
|---|---|
| `PUSHM` | opcode only |
| `PRED rs1` | `rs1` holds the per-lane predicate |
| `POPM` | opcode only |

### Branch

Branches use signed PC-relative instruction offsets.

```text
BRA label
BZ rs1, label
BNZ rs1, label
```

| Field | Value |
|---|---|
| `opcode` | branch opcode |
| `rd` | `0` |
| `rs1` | condition register for `BZ`/`BNZ`, otherwise `0` |
| `rs2` | `0` |
| `imm14` | `target_pc - (pc + 1)` |

### Control

Used for synchronization and kernel exit.

```text
BAR
EXIT
NOP
```

Only the opcode field is used.

## Opcode Table

| Opcode | Mnemonic | Opcode | Mnemonic |
|---:|---|---:|---|
| `0x00` | `NOP` | `0x01` | `MOV` |
| `0x02` | `MOVI` | `0x03` | `LDC` |
| `0x04` | `ADD` | `0x05` | `ADDI` |
| `0x06` | `SUB` | `0x07` | `SUBI` |
| `0x08` | `MUL` | `0x09` | `MULI` |
| `0x0A` | `DIV` | `0x0B` | `MOD` |
| `0x0C` | `AND` | `0x0D` | `ANDI` |
| `0x0E` | `OR` | `0x0F` | `ORI` |
| `0x10` | `XOR` | `0x11` | `XORI` |
| `0x12` | `NOT` | `0x13` | `SHL` |
| `0x14` | `SHLI` | `0x15` | `SHR` |
| `0x16` | `SHRI` | `0x17` | `SLT` |
| `0x18` | `SLE` | `0x19` | `SGT` |
| `0x1A` | `SGE` | `0x1B` | `SEQ` |
| `0x1C` | `SNE` | `0x1D` | `FADD` |
| `0x1E` | `FSUB` | `0x1F` | `FMUL` |
| `0x20` | `LDG` | `0x21` | `STG` |
| `0x22` | `LDS` | `0x23` | `STS` |
| `0x24` | `FDIV` | `0x28` | `TID` |
| `0x29` | `TIDX` | `0x2A` | `BID` |
| `0x2B` | `BDIM` | `0x2C` | `GDIM` |
| `0x2D` | `LID` | `0x2E` | `WID` |
| `0x30` | `PUSHM` | `0x31` | `PRED` |
| `0x32` | `POPM` | `0x33` | `PREDN` |
| `0x38` | `BRA` | `0x39` | `BZ` |
| `0x3A` | `BNZ` | `0x3B` | `BAR` |
| `0x3C` | `EXIT` |  |  |

## Mnemonic Names

| Mnemonic | Instruction name | Behavior |
|---|---|---|
| `NOP` | No Operation | Does nothing. |
| `MOV` | Move Register | Copies `rs1` into `rd`. |
| `MOVI` | Move Immediate | Writes `imm14` into `rd`. |
| `LDC` | Load Constant | Loads a constant-table entry into `rd`. |
| `ADD` | Add | Computes `rd = rs1 + rs2`. |
| `ADDI` | Add Immediate | Computes `rd = rs1 + imm14`. |
| `SUB` | Subtract | Computes `rd = rs1 - rs2`. |
| `SUBI` | Subtract Immediate | Computes `rd = rs1 - imm14`. |
| `MUL` | Multiply | Computes `rd = rs1 * rs2`. |
| `MULI` | Multiply Immediate | Computes `rd = rs1 * imm14`. |
| `DIV` | Divide | Computes `rd = rs1 / rs2`. |
| `MOD` | Modulo | Computes `rd = rs1 % rs2`. |
| `AND` | Bitwise And | Computes `rd = rs1 & rs2`. |
| `ANDI` | Bitwise And Immediate | Computes `rd = rs1 & imm14`. |
| `OR` | Bitwise Or | Computes `rd = rs1 | rs2`. |
| `ORI` | Bitwise Or Immediate | Computes `rd = rs1 | imm14`. |
| `XOR` | Bitwise Exclusive Or | Computes `rd = rs1 ^ rs2`. |
| `XORI` | Bitwise Exclusive Or Immediate | Computes `rd = rs1 ^ imm14`. |
| `NOT` | Bitwise Not | Computes `rd = ~rs1`. |
| `SHL` | Shift Left | Computes `rd = rs1 << rs2`. |
| `SHLI` | Shift Left Immediate | Computes `rd = rs1 << imm14`. |
| `SHR` | Shift Right | Computes `rd = rs1 >> rs2`. |
| `SHRI` | Shift Right Immediate | Computes `rd = rs1 >> imm14`. |
| `SLT` | Set Less Than | Writes `1` if `rs1 < rs2`, else `0`. |
| `SLE` | Set Less Than or Equal | Writes `1` if `rs1 <= rs2`, else `0`. |
| `SGT` | Set Greater Than | Writes `1` if `rs1 > rs2`, else `0`. |
| `SGE` | Set Greater Than or Equal | Writes `1` if `rs1 >= rs2`, else `0`. |
| `SEQ` | Set Equal | Writes `1` if `rs1 == rs2`, else `0`. |
| `SNE` | Set Not Equal | Writes `1` if `rs1 != rs2`, else `0`. |
| `FADD` | Floating Add | Computes typed floating-point `rd = rs1 + rs2`. |
| `FSUB` | Floating Subtract | Computes typed floating-point `rd = rs1 - rs2`. |
| `FMUL` | Floating Multiply | Computes typed floating-point `rd = rs1 * rs2`. |
| `FDIV` | Floating Divide | Computes typed floating-point `rd = rs1 / rs2`. |
| `LDG` | Load Global | Loads global memory `[rs1 + imm14]` into `rd`. |
| `STG` | Store Global | Stores `rs2` to global memory `[rs1 + imm14]`. |
| `LDS` | Load Shared | Loads shared memory `[rs1 + imm14]` into `rd`. |
| `STS` | Store Shared | Stores `rs2` to shared memory `[rs1 + imm14]`. |
| `TID` | Thread Global ID | Writes the global linear thread ID into `rd`. |
| `TIDX` | Thread Index X | Writes `threadIdx.x` into `rd`. |
| `BID` | Block ID | Writes `blockIdx.x` into `rd`. |
| `BDIM` | Block Dimension | Writes `blockDim.x` into `rd`. |
| `GDIM` | Grid Dimension | Writes `gridDim.x` into `rd`. |
| `LID` | Lane ID | Writes the lane ID inside the current warp into `rd`. |
| `WID` | Warp ID | Writes the current warp ID into `rd`. |
| `PUSHM` | Push Active Mask | Saves the current active-lane mask. |
| `PRED` | Predicate Active Mask | Applies `active_mask &= rs1`. |
| `POPM` | Pop Active Mask | Restores the previous active-lane mask. |
| `PREDN` | Predicate Active Mask Not | Applies `active_mask &= !rs1`. |
| `BRA` | Branch Always | Jumps to a PC-relative target. |
| `BZ` | Branch If Zero | Branches if `rs1 == 0`. |
| `BNZ` | Branch If Not Zero | Branches if `rs1 != 0`. |
| `BAR` | Barrier | Synchronizes threads in the current block. |
| `EXIT` | Exit Kernel | Ends execution for the current warp/thread. |
