`timescale 1ns/1ps

`include "minigpu_isa.vh"

module thread #(
    parameter WIDTH = 32
) (
    input  wire             clk,
    input  wire             rst,
    input  wire             instr_valid,
    input  wire [31:0]      instr,
    output wire             supported,
    output wire             divide_by_zero,
    output wire             execute_busy,
    output wire             execute_done,
    output wire             writeback_enable,
    output wire [3:0]       writeback_addr,
    output wire [WIDTH-1:0] writeback_data
);
    wire [5:0] opcode;
    wire [3:0] rd;
    wire [3:0] rs1;
    wire [3:0] rs2;
    wire [13:0] imm14;
    wire [2:0] fmt;
    wire [31:0] imm_sext;

    wire [WIDTH-1:0] lhs;
    wire [WIDTH-1:0] rhs;
    wire [WIDTH-1:0] exec_result;
    wire exec_supported;
    wire exec_divide_by_zero;
    reg pending_writes_register;
    reg [3:0] pending_writeback_addr;

    wire issue_execute = instr_valid && !execute_busy;

    instruction_decode decoder (
        .instr(instr),
        .opcode(opcode),
        .rd(rd),
        .rs1(rs1),
        .rs2(rs2),
        .imm14(imm14),
        .fmt(fmt),
        .imm_sext(imm_sext)
    );

    regfile #(.WIDTH(WIDTH)) registers (
        .clk(clk),
        .rst(rst),
        .write_enable(writeback_enable),
        .write_addr(writeback_addr),
        .write_data(writeback_data),
        .read_addr_a(rs1),
        .read_addr_b(rs2),
        .read_data_a(lhs),
        .read_data_b(rhs)
    );

    execute #(.WIDTH(WIDTH)) execute_unit (
        .clk(clk),
        .rst(rst),
        .start(issue_execute),
        .opcode(opcode),
        .lhs(lhs),
        .rhs(rhs),
        .imm14(imm14),
        .result(exec_result),
        .supported(exec_supported),
        .divide_by_zero(exec_divide_by_zero),
        .busy(execute_busy),
        .done(execute_done)
    );

    assign supported = exec_supported;
    assign divide_by_zero = exec_divide_by_zero;
    assign writeback_enable = execute_done && exec_supported && pending_writes_register;
    assign writeback_addr = pending_writeback_addr;
    assign writeback_data = exec_result;

    always @(posedge clk) begin
        if (rst) begin
            pending_writes_register <= 1'b0;
            pending_writeback_addr <= 4'b0;
        end else if (issue_execute) begin
            pending_writes_register <= writes_register(opcode);
            pending_writeback_addr <= rd;
        end
    end

    function writes_register;
        input [5:0] value_opcode;
        begin
            case (value_opcode)
                `MGPU_OP_MOV,
                `MGPU_OP_MOVI,
                `MGPU_OP_ADD,
                `MGPU_OP_ADDI,
                `MGPU_OP_SUB,
                `MGPU_OP_SUBI,
                `MGPU_OP_MUL,
                `MGPU_OP_MULI,
                `MGPU_OP_DIV,
                `MGPU_OP_MOD,
                `MGPU_OP_AND,
                `MGPU_OP_ANDI,
                `MGPU_OP_OR,
                `MGPU_OP_ORI,
                `MGPU_OP_XOR,
                `MGPU_OP_XORI,
                `MGPU_OP_NOT,
                `MGPU_OP_SHL,
                `MGPU_OP_SHLI,
                `MGPU_OP_SHR,
                `MGPU_OP_SHRI,
                `MGPU_OP_SLT,
                `MGPU_OP_SLE,
                `MGPU_OP_SGT,
                `MGPU_OP_SGE,
                `MGPU_OP_SEQ,
                `MGPU_OP_SNE,
                `MGPU_OP_FADD,
                `MGPU_OP_FSUB,
                `MGPU_OP_FMUL,
                `MGPU_OP_FDIV: writes_register = 1'b1;
                default: writes_register = 1'b0;
            endcase
        end
    endfunction
endmodule
