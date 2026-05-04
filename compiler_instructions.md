# CUDA-Oriented ISA and Compiler Plan for a Basys3 SIMT FPGA Accelerator

## Goal

We want to build a small SIMT-style GPU/accelerator on a Basys3 FPGA and create a compiler that can take a restricted CUDA-like `.cu` file and lower it to our custom ISA.

The compiler flow is:

```text
CUDA-like .cu source
        ↓
Clang CUDA parser
        ↓
Clang AST
        ↓
Custom AST visitor
        ↓
Mini-GPU IR
        ↓
Mini-GPU ISA assembly
        ↓
Machine code
        ↓
UART loader
        ↓
FPGA instruction memory
```

The goal is not to support full CUDA. The goal is to support a useful CUDA subset well enough that common kernels like vector add, ReLU, elementwise math, reductions, and small matrix multiplication can compile to our FPGA accelerator.

A good way to describe the project is:

> We use Clang to parse a restricted CUDA-like `.cu` source file and extract the kernel AST. Our compiler lowers CUDA thread indexing, arithmetic, array accesses, predicated conditionals, simple loops, and memory operations into a custom Mini-GPU IR. The IR is then lowered into our SIMT ISA and loaded over UART into the FPGA instruction memory.

## 1. Why Clang AST → IR → ISA Makes Sense

Using Clang lets us avoid writing a CUDA parser from scratch.

Clang can parse source code like:

```cuda
__global__ void vec_add(int* A, int* B, int* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}
```

Clang gives us an AST representing the kernel, variables, expressions, array accesses, and control flow.

However, Clang giving us an AST does not mean every CUDA program automatically works. It only means the syntax can be parsed. Our compiler backend still needs to support the specific CUDA constructs in the AST.

So we support a restricted CUDA subset.

Supported initially:

```text
- __global__ kernel functions
- int / int16_t scalar variables
- pointer kernel arguments
- threadIdx.x
- blockIdx.x
- blockDim.x
- gridDim.x, optional
- basic arithmetic: +, -, *, &, |, ^, <<, >>
- comparisons: <, <=, >, >=, ==, !=
- array indexing: A[i]
- if statements using predication
- simple for loops
- global memory loads/stores
- optional shared memory later
- optional __syncthreads() later
```

Not supported initially:

```text
- Full CUDA runtime
- malloc/free inside kernels
- recursion
- function pointers
- templates/classes
- floating point, unless implemented
- texture memory
- atomics
- warp intrinsics
- dynamic parallelism
- arbitrary divergent control flow
- complex pointer aliasing
```

## 2. Compiler Architecture

The compiler should not emit ISA directly from the AST. Instead:

```text
AST → Mini-GPU IR → ISA
```

This is cleaner because the AST is syntax-oriented, while the ISA is hardware-oriented.

Example CUDA source:

```cuda
C[tid] = A[tid] + B[tid];
```

The AST will contain nodes like:

```text
BinaryOperator '='
    LHS: ArraySubscriptExpr C[tid]
    RHS: BinaryOperator '+'
        LHS: ArraySubscriptExpr A[tid]
        RHS: ArraySubscriptExpr B[tid]
```

The compiler lowers this into IR:

```text
%a_addr = add %A, %tid
%a_val  = load %a_addr

%b_addr = add %B, %tid
%b_val  = load %b_addr

%sum = add %a_val, %b_val

%c_addr = add %C, %tid
store %c_addr, %sum
```

Then the IR lowers into ISA:

```asm
ADD r6, rA, rtid
LDG r7, [r6 + 0]

ADD r8, rB, rtid
LDG r9, [r8 + 0]

ADD r10, r7, r9

ADD r11, rC, rtid
STG [r11 + 0], r10
```

## 3. Mini-GPU IR

The IR should be small and hardware-friendly.

Possible IR instructions:

```text
tid          dst
thread_idx   dst
block_idx    dst
block_dim    dst
grid_dim     dst

arg          dst, arg_name
const        dst, value

add          dst, a, b
sub          dst, a, b
mul          dst, a, b
mac          dst, a, b
and          dst, a, b
or           dst, a, b
xor          dst, a, b
not          dst, a
shl          dst, a, b
shr          dst, a, b

lt           dst, a, b
le           dst, a, b
eq           dst, a, b
ne           dst, a, b

load_global  dst, ptr
store_global ptr, value

load_shared  dst, ptr
store_shared ptr, value

pred_begin   cond
pred_end

label        name
branch       label
branch_zero  cond, label
branch_nzero cond, label

barrier
return
```

Example IR for vector add:

```text
%tid = global_tid
%N   = arg N

%cond = lt %tid, %N

pred_begin %cond
    %A_base = arg A
    %B_base = arg B
    %C_base = arg C

    %a_addr = add %A_base, %tid
    %b_addr = add %B_base, %tid
    %c_addr = add %C_base, %tid

    %a_val = load_global %a_addr
    %b_val = load_global %b_addr
    %sum   = add %a_val, %b_val

    store_global %c_addr, %sum
pred_end

return
```

## 4. CUDA Execution Model Support

To make the ISA work with CUDA as well as possible, the ISA should be designed around the CUDA execution model.

CUDA uses:

```cuda
threadIdx.x
blockIdx.x
blockDim.x
gridDim.x
```

The most common indexing pattern is:

```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

Our ISA should support this directly.

Recommended thread/block metadata instructions:

```asm
TID    rd      ; rd = global linear thread id
TIDX   rd      ; rd = threadIdx.x
BID    rd      ; rd = blockIdx.x
BDIM   rd      ; rd = blockDim.x
GDIM   rd      ; rd = gridDim.x
LID    rd      ; rd = lane id inside warp
WID    rd      ; rd = warp id
```

Then the compiler can lower:

```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

directly to:

```asm
TID r0
```

or literally to:

```asm
BID  r0
BDIM r1
TIDX r2
MUL  r3, r0, r1
ADD  r4, r3, r2
```

For the MVP, `TID r0` is much better.

## 5. Blocks and Warps

To match CUDA, the runtime should support grid/block launches:

```python
gpu.launch(kernel, grid_dim=2, block_dim=4, args=[A, B, C, N])
```

Example:

```text
warp_size = 4
gridDim.x = 2
blockDim.x = 4
total threads = 8
```

Then:

```text
Block 0:
    Warp 0:
        threadIdx.x = 0, 1, 2, 3
        global tid  = 0, 1, 2, 3

Block 1:
    Warp 1:
        threadIdx.x = 0, 1, 2, 3
        global tid  = 4, 5, 6, 7
```

If `blockDim.x > warp_size`, one block has multiple warps.

Example:

```text
warp_size = 4
blockDim.x = 8
gridDim.x = 2
```

Then:

```text
Block 0:
    Warp 0: threadIdx.x = 0, 1, 2, 3
    Warp 1: threadIdx.x = 4, 5, 6, 7

Block 1:
    Warp 2: threadIdx.x = 0, 1, 2, 3
    Warp 3: threadIdx.x = 4, 5, 6, 7
```

Each warp state should store:

```text
valid
state
pc
block_id
warp_id_in_block
base_thread_idx
active_mask
wait_reason
```

A possible warp state entry:

```text
Warp State Entry:
    warp_id
    state
    pc
    block_id
    warp_id_in_block
    base_thread_idx
    active_mask
```

The hardware computes:

```text
threadIdx.x = base_thread_idx + lane_id
global_tid  = block_id * block_dim + threadIdx.x
```

## 6. CUDA Memory Spaces

CUDA has several memory spaces:

```text
- Global memory
- Shared memory
- Constant memory
- Local memory
- Registers
```

For our FPGA design, map them like this:

| CUDA Concept | FPGA Hardware Mapping |
|---|---|
| Registers | Per-lane register file |
| Global memory | Banked scratchpad BRAM global region |
| Shared memory | Per-block shared scratchpad BRAM region |
| Constant memory | Constant/argument BRAM |
| Local memory | Avoid initially, or use spill region in BRAM |

Recommended memory instructions:

```asm
LDG rd, [rs1 + imm]      ; load from global memory
STG [rs1 + imm], rs2     ; store to global memory

LDS rd, [rs1 + imm]      ; load from shared memory
STS [rs1 + imm], rs2     ; store to shared memory

LDC rd, const_id         ; load kernel arg or constant
```

This is better than only having generic `LOAD` and `STORE`, because CUDA has different memory spaces.

Example:

```cuda
int x = A[i];
```

lowers to:

```asm
LDC r1, ARG_A
ADD r2, r1, ri
LDG r3, [r2 + 0]
```

Example:

```cuda
__shared__ int tile[16];
tile[threadIdx.x] = A[i];
```

could lower to:

```asm
TIDX r0
LDC  r1, ARG_A
ADD  r2, r1, ri
LDG  r3, [r2 + 0]

LDC  r4, SHARED_BASE
ADD  r5, r4, r0
STS  [r5 + 0], r3
```

Even if global and shared memory are physically both BRAM regions, keeping the ISA instructions separate helps the compiler and makes the design more CUDA-like.

## 7. Predication and Active Masks

CUDA has divergent branches:

```cuda
if (tid < N) {
    C[tid] = A[tid] + B[tid];
}
```

A real GPU uses active masks and reconvergence to handle divergence.

For the MVP, use predication with a mask stack:

```asm
SLT   r2, r_tid, r_N
PUSHM
PRED  r2

; if body

POPM
```

Recommended mask instructions:

```asm
PUSHM       ; save current active mask
PRED rs     ; active_mask = active_mask & per_lane_bool(rs)
POPM        ; restore previous active mask
```

This is important because without `POPM`, once a lane is disabled it would stay disabled forever.

For example:

```cuda
if (tid < N) {
    C[tid] = A[tid] + B[tid];
}
D[tid] = 5;
```

The code after the `if` should execute with the original mask restored. That is why we need `PUSHM` and `POPM`.

Lowering:

```asm
TID r0
LDC r1, ARG_N
SLT r2, r0, r1

PUSHM
PRED r2

; body of if

POPM

; code after if
```

## 8. Branches and Loops

CUDA kernels commonly use loops:

```cuda
for (int k = 0; k < K; k++) {
    sum += A[k] * B[k];
}
```

The ISA needs branches:

```asm
BRA target       ; unconditional branch
BZ  rs, target   ; branch if rs == 0
BNZ rs, target   ; branch if rs != 0
```

Loop lowering example:

```asm
MOVI r_k, 0

loop:
SLT  r_cond, r_k, r_K
BZ   r_cond, end

; loop body

ADDI r_k, r_k, 1
BRA  loop

end:
```

For the MVP, support uniform loops first, where all lanes loop the same number of times.

This covers matrix multiplication inner loops.

Avoid arbitrary per-lane divergent loops initially.

## 9. Recommended CUDA-Friendly ISA

### 9.1 Thread and Block Metadata

```asm
TID    rd      ; global linear thread id
TIDX   rd      ; threadIdx.x
BID    rd      ; blockIdx.x
BDIM   rd      ; blockDim.x
GDIM   rd      ; gridDim.x
LID    rd      ; lane id inside warp
WID    rd      ; warp id
```

### 9.2 Data Movement

```asm
MOV    rd, rs
MOVI   rd, imm
LDC    rd, const_id
```

### 9.3 Integer / Fixed-Point ALU

```asm
ADD    rd, rs1, rs2
ADDI   rd, rs1, imm

SUB    rd, rs1, rs2
SUBI   rd, rs1, imm

MUL    rd, rs1, rs2
MULI   rd, rs1, imm

MAC    rd, rs1, rs2      ; rd = rd + rs1 * rs2
```

For fixed-point, define `MUL` and `MAC` clearly.

Example:

```text
If using Q8.8 fixed-point:
    MUL computes (rs1 * rs2) >> 8
    MAC computes rd = rd + ((rs1 * rs2) >> 8)
```

### 9.4 Bitwise and Shift

```asm
AND    rd, rs1, rs2
ANDI   rd, rs1, imm

OR     rd, rs1, rs2
ORI    rd, rs1, imm

XOR    rd, rs1, rs2
XORI   rd, rs1, imm

NOT    rd, rs

SHL    rd, rs1, rs2
SHR    rd, rs1, rs2

SHLI   rd, rs1, imm
SHRI   rd, rs1, imm
```

These are useful for index math.

Example:

```cuda
int row = tid >> 2;
int col = tid & 3;
```

lowers to:

```asm
TID  r0
SHRI r1, r0, 2
ANDI r2, r0, 3
```

### 9.5 Comparisons

```asm
SLT    rd, rs1, rs2      ; rd = rs1 < rs2
SLE    rd, rs1, rs2      ; rd = rs1 <= rs2
SGT    rd, rs1, rs2      ; rd = rs1 > rs2
SGE    rd, rs1, rs2      ; rd = rs1 >= rs2
SEQ    rd, rs1, rs2      ; rd = rs1 == rs2
SNE    rd, rs1, rs2      ; rd = rs1 != rs2
```

For MVP hardware, you can implement only:

```asm
SLT
SEQ
```

and have the compiler synthesize the others, but implementing all six makes the compiler much simpler.

### 9.6 Memory Instructions

```asm
LDG    rd, [rs1 + imm]      ; global memory load
STG    [rs1 + imm], rs2     ; global memory store

LDS    rd, [rs1 + imm]      ; shared memory load
STS    [rs1 + imm], rs2     ; shared memory store

LDC    rd, const_id         ; load kernel arg / constant
```

### 9.7 Predication / Masks

```asm
PUSHM
PRED   rs
POPM
```

Optional:

```asm
PREDN  rs      ; predicate on not rs
```

### 9.8 Control Flow

```asm
BRA    target
BZ     rs, target
BNZ    rs, target
BAR
EXIT
```

`BAR` maps to CUDA:

```cuda
__syncthreads();
```

For MVP, if each block has one warp, `BAR` can be a no-op. Later, it can become a real per-block barrier.

## 10. Instruction Format

A 4-bit opcode gives only 16 instructions, which is too small for a CUDA-friendly ISA.

Use a 6-bit opcode.

Suggested 32-bit instruction format:

```text
[31:26] OPCODE
[25:22] RD
[21:18] RS1
[17:14] RS2
[13:0]  IMM14
```

This gives:

```text
64 possible opcodes
16 physical registers
14-bit immediate
```

For larger immediates, either:

1. Keep memory small enough that 14 bits is enough.
2. Add a `LUI` instruction later.
3. Let the runtime patch small addresses into constants.
4. Use constant memory / argument table with `LDC`.

Example encodings conceptually:

```asm
ADD r6, r3, r0
```

```text
opcode = ADD
rd     = 6
rs1    = 3
rs2    = 0
imm    = 0
```

```asm
LDG r9, [r6 + 0]
```

```text
opcode = LDG
rd     = 9
rs1    = 6
imm    = 0
```

```asm
STG [r8 + 0], r11
```

```text
opcode = STG
rs1    = 8
rs2    = 11
imm    = 0
```

## 11. Memory System and SIMT Loads

A SIMT memory load is not a single scalar load. It is one instruction that creates multiple per-lane memory requests.

Example:

```asm
LDG r7, [r4 + 0]
```

means:

```text
for each active lane:
    r7[lane] = global_memory[r4[lane] + 0]
```

Example register values:

```text
r4:
    lane 0 = 0
    lane 1 = 1
    lane 2 = 2
    lane 3 = 3
```

Then:

```text
lane 0 loads memory[0]
lane 1 loads memory[1]
lane 2 loads memory[2]
lane 3 loads memory[3]
```

If memory contains:

```text
memory[0] = 10
memory[1] = 20
memory[2] = 30
memory[3] = 40
```

then after the load:

```text
r7:
    lane 0 = 10
    lane 1 = 20
    lane 2 = 30
    lane 3 = 40
```

The load/store unit does:

```text
1. Take the SIMT load instruction.
2. Read the address register from each active lane.
3. Compute effective addresses.
4. Map addresses to memory banks.
5. Detect bank conflicts.
6. Issue BRAM reads.
7. Collect returned data.
8. Write data back into the per-lane register file.
9. Mark the warp ready again.
```

## 12. Banked BRAM Memory

Use striped banking:

```text
bank_id   = address % NUM_BANKS
bank_addr = address / NUM_BANKS
```

For 4 banks:

```text
bank_id   = address[1:0]
bank_addr = address >> 2
```

Consecutive addresses map across different banks:

```text
Address 0 -> Bank 0
Address 1 -> Bank 1
Address 2 -> Bank 2
Address 3 -> Bank 3
Address 4 -> Bank 0
Address 5 -> Bank 1
...
```

Good access pattern:

```text
lane 0 address = 0 -> bank 0
lane 1 address = 1 -> bank 1
lane 2 address = 2 -> bank 2
lane 3 address = 3 -> bank 3
```

All lanes can be served in parallel.

Bad access pattern:

```text
lane 0 address = 0  -> bank 0
lane 1 address = 4  -> bank 0
lane 2 address = 8  -> bank 0
lane 3 address = 12 -> bank 0
```

All lanes want bank 0. That is a bank conflict.

For the MVP:

```text
If no bank conflict:
    serve all active lanes in parallel.

If bank conflict:
    serialize the conflicting requests.
```

The warp state during a memory request:

```text
READY -> RUNNING -> WAIT_MEM -> READY
```

While one warp waits for memory, the scheduler can run another ready warp.

## 13. Memory Access / Load-Store Pipeline

A detailed memory pipeline:

```text
SIMT LDG/STG instruction
        ↓
Address Generation Unit
        ↓
Predicate Filter / Active Lane Mask
        ↓
Memory Request Builder
        ↓
Coalescer / Request Merger
        ↓
Bank Mapper
        ↓
Conflict Detector
        ↓
Pending Load / Store Queue
        ↓
BRAM Bank Controller
        ↓
Response Collector
        ↓
Writeback Arbiter
        ↓
Per-lane Register File
```

Important structures:

```text
Pending Load Table Entry:
    valid
    warp_id
    dest_reg
    active_mask
    lane_addresses
    completed_mask
    lane_data
```

Example pending load:

```text
warp_id        = 0
dest_reg       = r7
active_mask    = 1111
lane0_address  = 0
lane1_address  = 1
lane2_address  = 2
lane3_address  = 3
completed_mask = 0000
```

When data returns:

```text
lane0_data = 10
lane1_data = 20
lane2_data = 30
lane3_data = 40
```

the writeback unit writes:

```text
regfile[warp0][lane0][r7] = 10
regfile[warp0][lane1][r7] = 20
regfile[warp0][lane2][r7] = 30
regfile[warp0][lane3][r7] = 40
```

Then:

```text
warp0.state = READY
```

## 14. Cache / Buffer Strategy

A full cache is probably too much for the MVP.

Instead, implement simple load/store buffers:

```text
- Load coalescing buffer
- Tiny line buffer / read cache
- Store buffer / write combiner
```

### Load Coalescing Buffer

Detect when lanes access consecutive addresses:

```text
lane addresses = [0, 1, 2, 3]
```

Then issue one coalesced access across four banks.

This is not a full cache. It is a request combiner.

### Tiny Line Buffer

Store the last loaded line:

```text
valid
base_address
data[0:3]
```

If the next load hits the same line, return data quickly.

Example:

```text
line base = 0
line data = memory[0], memory[1], memory[2], memory[3]
```

If a later instruction loads `memory[2]`, it can hit in the line buffer.

This is optional.

### Store Buffer

For stores, collect adjacent lane stores:

```text
lane 0 stores address 32
lane 1 stores address 33
lane 2 stores address 34
lane 3 stores address 35
```

Then issue them together to the banks.

Again, this is simpler than a real cache.

Suggested MVP statement:

> Instead of implementing a full cache, the MVP uses a banked scratchpad memory with a simple coalescing buffer and optional tiny line buffer. Consecutive per-lane accesses are merged into a single banked transaction, while bank conflicts are serialized.

## 15. Example: CUDA Vector Add End-to-End

Input CUDA-like source:

```cuda
__global__ void vec_add(int* A, int* B, int* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

Clang parses it into an AST.

The compiler recognizes:

```cuda
blockIdx.x * blockDim.x + threadIdx.x
```

and lowers it to:

```text
%tid = global_tid
```

Mini-GPU IR:

```text
kernel vec_add(A, B, C, N)

%tid = global_tid
%N_val = arg N

%cond = lt %tid, %N_val

pred_begin %cond
    %A_base = arg A
    %B_base = arg B
    %C_base = arg C

    %a_addr = add %A_base, %tid
    %b_addr = add %B_base, %tid
    %c_addr = add %C_base, %tid

    %a_val = load_global %a_addr
    %b_val = load_global %b_addr

    %sum = add %a_val, %b_val

    store_global %c_addr, %sum
pred_end

return
```

Lowered ISA:

```asm
TID   r0
LDC   r1, ARG_N
SLT   r2, r0, r1

PUSHM
PRED  r2

LDC   r3, ARG_A
LDC   r4, ARG_B
LDC   r5, ARG_C

ADD   r6, r3, r0
ADD   r7, r4, r0
ADD   r8, r5, r0

LDG   r9,  [r6 + 0]
LDG   r10, [r7 + 0]
ADD   r11, r9, r10

STG   [r8 + 0], r11

POPM
EXIT
```

Example runtime values:

```text
A base = 0
B base = 16
C base = 32
N      = 6
```

Assume:

```text
A = [10, 20, 30, 40, 50, 60]
B = [1,  2,  3,  4,  5,  6]
```

Launch:

```text
warp_size = 4
gridDim.x = 2
blockDim.x = 4
total hardware threads = 8
valid N = 6
```

Warps:

```text
Warp 0:
    block_id = 0
    threadIdx.x = 0,1,2,3
    global tid = 0,1,2,3
    active_mask = 1111

Warp 1:
    block_id = 1
    threadIdx.x = 0,1,2,3
    global tid = 4,5,6,7
    active_mask initially = 1111
```

For warp 1:

```text
tid = [4, 5, 6, 7]
N = 6
tid < N = [1, 1, 0, 0]
```

After:

```asm
SLT r2, r0, r1
PUSHM
PRED r2
```

the active mask becomes:

```text
0011
```

Only lanes 0 and 1 store results.

Final output:

```text
C = [11, 22, 33, 44, 55, 66]
```

## 16. Example: CUDA ReLU

CUDA source:

```cuda
__global__ void relu(int* X, int* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        int x = X[i];

        if (x < 0) {
            Y[i] = 0;
        } else {
            Y[i] = x;
        }
    }
}
```

Lowered ISA using predication:

```asm
TID   r0
LDC   r1, ARG_N
SLT   r2, r0, r1

PUSHM
PRED  r2

LDC   r3, ARG_X
LDC   r4, ARG_Y

ADD   r5, r3, r0
ADD   r6, r4, r0

LDG   r7, [r5 + 0]      ; x

MOVI  r8, 0
SLT   r9, r7, r8        ; x < 0

; if x < 0: Y[i] = 0
PUSHM
PRED  r9
STG   [r6 + 0], r8
POPM

; else: Y[i] = x
NOT   r10, r9
PUSHM
PRED  r10
STG   [r6 + 0], r7
POPM

POPM
EXIT
```

This is not optimal, but it is simple and compiler-friendly.

## 17. Example: CUDA Matrix Multiply

Example CUDA source for a small fixed `N = 4` matrix multiply:

```cuda
__global__ void matmul4(int* A, int* B, int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int row = tid >> 2;
    int col = tid & 3;

    int sum = 0;

    for (int k = 0; k < 4; k++) {
        sum += A[row * 4 + k] * B[k * 4 + col];
    }

    C[tid] = sum;
}
```

Lowered ISA:

```asm
TID   r0

SHRI  r1, r0, 2        ; row = tid / 4
ANDI  r2, r0, 3        ; col = tid % 4

MOVI  r3, 0            ; sum = 0
MOVI  r4, 0            ; k = 0

loop:
SLTI  r5, r4, 4
BZ    r5, end

; A index = row * 4 + k
SHLI  r6, r1, 2
ADD   r6, r6, r4

; B index = k * 4 + col
SHLI  r7, r4, 2
ADD   r7, r7, r2

LDC   r8, ARG_A
LDC   r9, ARG_B

ADD   r10, r8, r6
ADD   r11, r9, r7

LDG   r12, [r10 + 0]
LDG   r13, [r11 + 0]

MAC   r3, r12, r13

ADDI  r4, r4, 1
BRA   loop

end:
LDC   r14, ARG_C
ADD   r15, r14, r0
STG   [r15 + 0], r3

EXIT
```

This requires:

```text
TID
SHRI
ANDI
MOVI
SLTI or SLT with immediate
BZ
SHLI
ADD
LDC
LDG
MAC
ADDI
BRA
STG
EXIT
```

This shows why immediate instructions and branch instructions are important.

## 18. Runtime and Kernel Arguments

CUDA kernels have arguments:

```cuda
__global__ void vec_add(int* A, int* B, int* C, int N)
```

We do not need a full ABI.

Simplest approach:

1. Host allocates device memory.
2. Host knows base addresses.
3. Host writes kernel arguments into a constant/argument table.
4. Kernel uses `LDC` to load arguments.

Example argument table:

```text
ARG_A = 0
ARG_B = 16
ARG_C = 32
ARG_N = 6
```

Then:

```asm
LDC r3, ARG_A
LDC r4, ARG_B
LDC r5, ARG_C
LDC r1, ARG_N
```

The host runtime sends:

```text
WRITE_DATA
WRITE_PROGRAM
WRITE_ARG_TABLE
LAUNCH
READ_STATUS
READ_DATA
```

Suggested UART commands:

```text
WRITE_DATA
READ_DATA
WRITE_PROGRAM
WRITE_ARGS
LAUNCH
READ_STATUS
RESET
WRITE_HASH
VALIDATE
```

## 19. Clang Frontend Strategy

Use Clang to parse `.cu` files.

For early development, create a fake CUDA header:

```c
#define __global__ __attribute__((global))
#define __device__
#define __host__

typedef struct {
    int x;
    int y;
    int z;
} dim3;

extern const dim3 threadIdx;
extern const dim3 blockIdx;
extern const dim3 blockDim;
extern const dim3 gridDim;

void __syncthreads();
```

Then the source can use real CUDA-like syntax:

```cuda
__global__ void vec_add(int* A, int* B, int* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}
```

The compiler’s AST visitor should recognize:

```text
FunctionDecl with __global__
ParmVarDecl kernel args
MemberExpr threadIdx.x
MemberExpr blockIdx.x
MemberExpr blockDim.x
ArraySubscriptExpr A[i]
IfStmt
ForStmt
BinaryOperator
IntegerLiteral
DeclRefExpr
CallExpr __syncthreads
```

Special CUDA lowering patterns:

```text
blockIdx.x * blockDim.x + threadIdx.x → TID
threadIdx.x → TIDX
blockIdx.x → BID
blockDim.x → BDIM
gridDim.x → GDIM
__syncthreads() → BAR
```

## 20. Register Allocation

Start with a simple register allocator.

Map each IR temporary to a physical register:

```text
%tid     → r0
%N       → r1
%cond    → r2
%A_base  → r3
%B_base  → r4
%C_base  → r5
%a_addr  → r6
%b_addr  → r7
%c_addr  → r8
%a_val   → r9
%b_val   → r10
%sum     → r11
```

If the kernel needs more registers than available, reject it:

```text
Compiler error: kernel requires too many registers for Mini-GPU MVP.
```

This is acceptable for the capstone.

Later, add liveness analysis to reuse registers.

## 21. Hardware Design Around Compiler Needs

The ISA should be regular and easy to generate from a compiler.

Good design principles:

```text
- Use 3-register ALU instructions: rd = rs1 op rs2.
- Include immediate variants: ADDI, SHLI, SHRI, ANDI.
- Use consistent memory format: LDG rd, [rs + imm].
- Provide direct CUDA metadata instructions: TID, TIDX, BID, BDIM.
- Provide LDC for kernel args/constants.
- Provide PUSHM/PRED/POPM for if-statements.
- Provide BRA/BZ/BNZ for loops.
- Provide MAC for matrix multiplication and ML kernels.
```

Avoid weird special-case instructions unless they dramatically simplify CUDA lowering.

## 22. Suggested MVP Hardware Parameters

```text
Board: Basys3 FPGA
Warp size: 4 lanes
Resident warps: 4
Registers per thread: 16
Data width: 16-bit integer or Q8.8 fixed-point
Instruction width: 32 bits
Opcode width: 6 bits
Memory: banked BRAM scratchpad
Banks: 4
Host interface: UART
Compiler input: restricted CUDA-like .cu
Compiler flow: Clang AST → Mini-GPU IR → ISA → machine code
```

## 23. Suggested MVP Demo Kernels

Start with:

```text
1. vector add
2. scalar multiply
3. ReLU
4. elementwise multiply
5. small matrix multiply
```

Then add:

```text
6. simple reduction
7. tiled matrix multiply using shared memory
8. small neural network layer
```

## 24. Final Project Description

A strong project description:

> This project implements a CUDA-oriented SIMT accelerator on a Basys3 FPGA. The hardware includes a warp scheduler, per-lane register files, a 4-lane SIMT execution core, predicated execution, banked BRAM memory, a load/store pipeline, and a UART host interface. On the software side, a Clang-based compiler parses a restricted CUDA-like `.cu` file, lowers the kernel AST into a custom Mini-GPU IR, then emits our custom SIMT ISA. The ISA is designed around CUDA concepts such as thread/block IDs, global/shared memory operations, predication, barriers, and branch support, allowing common CUDA kernels to run on the FPGA accelerator.

Another shorter version:

> We are building a small CUDA-like SIMT accelerator on FPGA with a custom compiler. The compiler uses Clang to parse CUDA-like `.cu` kernels, lowers the AST to a Mini-GPU IR, and emits a CUDA-oriented ISA with thread/block ID instructions, memory-space-specific loads/stores, predication, branching, and fixed-point arithmetic. The FPGA executes these kernels using a 4-lane warp model, banked BRAM scratchpad memory, a warp scheduler, and UART-based host communication.

## 25. Key Takeaway

To make the ISA work with CUDA as well as possible, design it around the concepts CUDA kernels actually use:

```text
1. Thread and block IDs:
       TID, TIDX, BID, BDIM, GDIM

2. Memory spaces:
       LDG/STG for global memory
       LDS/STS for shared memory
       LDC for constants/kernel args

3. Predicated SIMT control:
       PUSHM, PRED, POPM

4. Loops and branches:
       BRA, BZ, BNZ

5. Useful compiler-friendly ALU ops:
       ADD, ADDI, MUL, MAC, SHLI, SHRI, ANDI

6. Synchronization:
       BAR for __syncthreads()

7. Simple host/runtime model:
       argument table, program memory, banked data memory, UART launch
```

This gives us a realistic CUDA subset while keeping the FPGA design manageable.