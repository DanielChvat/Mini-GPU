`timescale 1ns/1ps

module instruction_decode (
    input  wire [31:0] instr,
    output wire [5:0]  opcode,
    output wire [3:0]  rd,
    output wire [3:0]  rs1,
    output wire [3:0]  rs2,
    output wire [13:0] imm14,
    output wire [2:0]  fmt,
    output wire [31:0] imm_sext
);
    assign opcode = instr[31:26];
    assign rd = instr[25:22];
    assign rs1 = instr[21:18];
    assign rs2 = instr[17:14];
    assign imm14 = instr[13:0];
    assign fmt = instr[2:0];
    assign imm_sext = {{18{instr[13]}}, instr[13:0]};
endmodule
