`timescale 1ns/1ps

`include "minigpu_isa.vh"

module thread #(
    parameter WIDTH = 32,
    parameter ADDR_WIDTH = 16,
    parameter ENABLE_FLOAT_ADD = 1,
    parameter ENABLE_FLOAT_MUL = 1,
    parameter ENABLE_FLOAT_DIV = 1,
    parameter FLOAT_FP32_ONLY = 0,
    parameter USE_SHARED_FLOAT = 0
) (
    input  wire             clk,
    input  wire             rst,
    input  wire             instr_valid,
    input  wire [31:0]      instr,
    input  wire [WIDTH-1:0] thread_id,
    input  wire [WIDTH-1:0] lane_id,
    input  wire [WIDTH-1:0] warp_id,
    input  wire [WIDTH-1:0] block_id,
    input  wire [WIDTH-1:0] block_dim,
    input  wire [WIDTH-1:0] grid_dim,
    input  wire [WIDTH-1:0] const_data,
    output wire             mem_req_valid,
    output wire             mem_req_write,
    output wire [ADDR_WIDTH-1:0] mem_req_addr,
    output wire [WIDTH-1:0] mem_req_wdata,
    input  wire             mem_req_ready,
    input  wire             mem_resp_valid,
    input  wire [WIDTH-1:0] mem_resp_rdata,
    output wire             float_req_valid,
    output wire [5:0]       float_req_opcode,
    output wire [2:0]       float_req_fmt,
    output wire [WIDTH-1:0] float_req_lhs,
    output wire [WIDTH-1:0] float_req_rhs,
    input  wire             float_req_ready,
    input  wire             float_resp_valid,
    input  wire [WIDTH-1:0] float_resp_result,
    input  wire             float_resp_supported,
    input  wire             float_resp_divide_by_zero,
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
    wire exec_busy;
    wire exec_done;
    wire memory_opcode = is_memory_opcode(opcode);
    wire float_opcode = is_float_opcode(opcode);
    wire thread_busy;
    reg pending_writes_register;
    reg [3:0] pending_writeback_addr;
    reg mem_busy;
    reg mem_req_sent;
    reg mem_pending_load;
    reg mem_done;
    reg [3:0] mem_pending_writeback_addr;
    reg [ADDR_WIDTH-1:0] mem_addr_r;
    reg [WIDTH-1:0] mem_wdata_r;
    reg [WIDTH-1:0] mem_rdata_r;
    reg float_busy;
    reg float_req_sent;
    reg float_done;
    reg float_supported_r;
    reg float_divide_by_zero_r;
    reg [5:0] float_opcode_r;
    reg [2:0] float_fmt_r;
    reg [3:0] float_pending_writeback_addr;
    reg [WIDTH-1:0] float_lhs_r;
    reg [WIDTH-1:0] float_rhs_r;
    reg [WIDTH-1:0] float_result_r;

    wire issue_float = USE_SHARED_FLOAT && instr_valid && !thread_busy && float_opcode;
    wire issue_execute = instr_valid && !thread_busy && !memory_opcode &&
                         !(USE_SHARED_FLOAT && float_opcode);
    wire issue_memory = instr_valid && !thread_busy && memory_opcode;

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

    execute #(
        .WIDTH(WIDTH),
        .ENABLE_FLOAT_ADD(USE_SHARED_FLOAT ? 0 : ENABLE_FLOAT_ADD),
        .ENABLE_FLOAT_MUL(USE_SHARED_FLOAT ? 0 : ENABLE_FLOAT_MUL),
        .ENABLE_FLOAT_DIV(USE_SHARED_FLOAT ? 0 : ENABLE_FLOAT_DIV),
        .FLOAT_FP32_ONLY(FLOAT_FP32_ONLY)
    ) execute_unit (
        .clk(clk),
        .rst(rst),
        .start(issue_execute),
        .opcode(opcode),
        .lhs(lhs),
        .rhs(rhs),
        .imm14(imm14),
        .thread_id(thread_id),
        .lane_id(lane_id),
        .warp_id(warp_id),
        .block_id(block_id),
        .block_dim(block_dim),
        .grid_dim(grid_dim),
        .const_data(const_data),
        .result(exec_result),
        .supported(exec_supported),
        .divide_by_zero(exec_divide_by_zero),
        .busy(exec_busy),
        .done(exec_done)
    );

    assign thread_busy = exec_busy || mem_busy || float_busy;
    assign execute_busy = thread_busy;
    assign execute_done = exec_done || mem_done || float_done;
    assign supported = mem_done ? 1'b1 :
                       (float_done ? float_supported_r : exec_supported);
    assign divide_by_zero = float_done ? float_divide_by_zero_r : exec_divide_by_zero;
    assign writeback_enable = (exec_done && exec_supported && pending_writes_register) ||
                              (mem_done && mem_pending_load) ||
                              (float_done && float_supported_r);
    assign writeback_addr = mem_done ? mem_pending_writeback_addr :
                            (float_done ? float_pending_writeback_addr : pending_writeback_addr);
    assign writeback_data = mem_done ? mem_rdata_r :
                            (float_done ? float_result_r : exec_result);
    assign mem_req_valid = mem_busy && !mem_req_sent;
    assign mem_req_write = !mem_pending_load;
    assign mem_req_addr = mem_addr_r;
    assign mem_req_wdata = mem_wdata_r;
    assign float_req_valid = float_busy && !float_req_sent;
    assign float_req_opcode = float_opcode_r;
    assign float_req_fmt = float_fmt_r;
    assign float_req_lhs = float_lhs_r;
    assign float_req_rhs = float_rhs_r;

    always @(posedge clk) begin
        if (rst) begin
            pending_writes_register <= 1'b0;
            pending_writeback_addr <= 4'b0;
            mem_busy <= 1'b0;
            mem_req_sent <= 1'b0;
            mem_pending_load <= 1'b0;
            mem_done <= 1'b0;
            mem_pending_writeback_addr <= 4'b0;
            mem_addr_r <= {ADDR_WIDTH{1'b0}};
            mem_wdata_r <= {WIDTH{1'b0}};
            mem_rdata_r <= {WIDTH{1'b0}};
            float_busy <= 1'b0;
            float_req_sent <= 1'b0;
            float_done <= 1'b0;
            float_supported_r <= 1'b0;
            float_divide_by_zero_r <= 1'b0;
            float_opcode_r <= 6'b0;
            float_fmt_r <= 3'b0;
            float_pending_writeback_addr <= 4'b0;
            float_lhs_r <= {WIDTH{1'b0}};
            float_rhs_r <= {WIDTH{1'b0}};
            float_result_r <= {WIDTH{1'b0}};
        end else begin
            mem_done <= 1'b0;
            float_done <= 1'b0;

            if (issue_memory) begin
                mem_busy <= 1'b1;
                mem_req_sent <= 1'b0;
                mem_pending_load <= is_load_opcode(opcode);
                mem_pending_writeback_addr <= rd;
                mem_addr_r <= lhs[ADDR_WIDTH-1:0] + imm_sext[ADDR_WIDTH-1:0];
                mem_wdata_r <= rhs;
            end else if (mem_busy && !mem_req_sent) begin
                if (mem_req_ready) begin
                    mem_req_sent <= 1'b1;
                    if (!mem_pending_load) begin
                        mem_busy <= 1'b0;
                        mem_done <= 1'b1;
                    end
                end
            end else if (mem_busy && mem_req_sent && mem_pending_load) begin
                if (mem_resp_valid) begin
                    mem_rdata_r <= mem_resp_rdata;
                    mem_busy <= 1'b0;
                    mem_done <= 1'b1;
                end
            end

            if (issue_float) begin
                float_busy <= 1'b1;
                float_req_sent <= 1'b0;
                float_opcode_r <= opcode;
                float_fmt_r <= fmt;
                float_pending_writeback_addr <= rd;
                float_lhs_r <= lhs;
                float_rhs_r <= rhs;
            end else if (float_busy && !float_req_sent) begin
                if (float_req_ready) begin
                    float_req_sent <= 1'b1;
                end
            end else if (float_busy && float_req_sent) begin
                if (float_resp_valid) begin
                    float_busy <= 1'b0;
                    float_done <= 1'b1;
                    float_result_r <= float_resp_result;
                    float_supported_r <= float_resp_supported;
                    float_divide_by_zero_r <= float_resp_divide_by_zero;
                end
            end

            if (issue_execute) begin
                pending_writes_register <= writes_register(opcode);
                pending_writeback_addr <= rd;
            end
        end
    end

    function is_memory_opcode;
        input [5:0] value_opcode;
        begin
            case (value_opcode)
                `MGPU_OP_LDG,
                `MGPU_OP_STG,
                `MGPU_OP_LDS,
                `MGPU_OP_STS: is_memory_opcode = 1'b1;
                default: is_memory_opcode = 1'b0;
            endcase
        end
    endfunction

    function is_load_opcode;
        input [5:0] value_opcode;
        begin
            case (value_opcode)
                `MGPU_OP_LDG,
                `MGPU_OP_LDS: is_load_opcode = 1'b1;
                default: is_load_opcode = 1'b0;
            endcase
        end
    endfunction

    function is_float_opcode;
        input [5:0] value_opcode;
        begin
            case (value_opcode)
                `MGPU_OP_FADD,
                `MGPU_OP_FSUB,
                `MGPU_OP_FMUL,
                `MGPU_OP_FDIV: is_float_opcode = 1'b1;
                default: is_float_opcode = 1'b0;
            endcase
        end
    endfunction

    function writes_register;
        input [5:0] value_opcode;
        begin
            case (value_opcode)
                `MGPU_OP_MOV,
                `MGPU_OP_MOVI,
                `MGPU_OP_LDC,
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
                `MGPU_OP_FDIV,
                `MGPU_OP_TID,
                `MGPU_OP_TIDX,
                `MGPU_OP_BID,
                `MGPU_OP_BDIM,
                `MGPU_OP_GDIM,
                `MGPU_OP_LID,
                `MGPU_OP_WID: writes_register = 1'b1;
                default: writes_register = 1'b0;
            endcase
        end
    endfunction
endmodule
