`ifndef MINIGPU_ISA_VH
`define MINIGPU_ISA_VH

// Keep these opcode values in lockstep with compiler/isa_to_bin.py.
`define MGPU_OP_NOP    6'h00
`define MGPU_OP_MOV    6'h01
`define MGPU_OP_MOVI   6'h02
`define MGPU_OP_LDC    6'h03
`define MGPU_OP_ADD    6'h04
`define MGPU_OP_ADDI   6'h05
`define MGPU_OP_SUB    6'h06
`define MGPU_OP_SUBI   6'h07
`define MGPU_OP_MUL    6'h08
`define MGPU_OP_MULI   6'h09
`define MGPU_OP_DIV    6'h0a
`define MGPU_OP_MOD    6'h0b
`define MGPU_OP_AND    6'h0c
`define MGPU_OP_ANDI   6'h0d
`define MGPU_OP_OR     6'h0e
`define MGPU_OP_ORI    6'h0f
`define MGPU_OP_XOR    6'h10
`define MGPU_OP_XORI   6'h11
`define MGPU_OP_NOT    6'h12
`define MGPU_OP_SHL    6'h13
`define MGPU_OP_SHLI   6'h14
`define MGPU_OP_SHR    6'h15
`define MGPU_OP_SHRI   6'h16
`define MGPU_OP_SLT    6'h17
`define MGPU_OP_SLE    6'h18
`define MGPU_OP_SGT    6'h19
`define MGPU_OP_SGE    6'h1a
`define MGPU_OP_SEQ    6'h1b
`define MGPU_OP_SNE    6'h1c
`define MGPU_OP_FADD   6'h1d
`define MGPU_OP_FSUB   6'h1e
`define MGPU_OP_FMUL   6'h1f
`define MGPU_OP_LDG    6'h20
`define MGPU_OP_STG    6'h21
`define MGPU_OP_LDS    6'h22
`define MGPU_OP_STS    6'h23
`define MGPU_OP_FDIV   6'h24
`define MGPU_OP_TID    6'h28
`define MGPU_OP_TIDX   6'h29
`define MGPU_OP_BID    6'h2a
`define MGPU_OP_BDIM   6'h2b
`define MGPU_OP_GDIM   6'h2c
`define MGPU_OP_LID    6'h2d
`define MGPU_OP_WID    6'h2e
`define MGPU_OP_PUSHM  6'h30
`define MGPU_OP_PRED   6'h31
`define MGPU_OP_POPM   6'h32
`define MGPU_OP_PREDN  6'h33
`define MGPU_OP_BRA    6'h38
`define MGPU_OP_BZ     6'h39
`define MGPU_OP_BNZ    6'h3a
`define MGPU_OP_BAR    6'h3b
`define MGPU_OP_EXIT   6'h3c

// Register-register typed ALU ops store their format in imm14[2:0].
`define MGPU_FMT_I32   3'h0
`define MGPU_FMT_I16   3'h1
`define MGPU_FMT_I8    3'h2
`define MGPU_FMT_FP32  3'h3
`define MGPU_FMT_FP16  3'h4
`define MGPU_FMT_FP8   3'h5

`endif
