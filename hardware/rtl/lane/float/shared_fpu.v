`timescale 1ns/1ps

`include "minigpu_isa.vh"

module shared_fpu #(
    parameter ENABLE_FLOAT_ADD = 1,
    parameter ENABLE_FLOAT_MUL = 1,
    parameter ENABLE_FLOAT_DIV = 1,
    parameter FLOAT_FP32_ONLY = 0,
    parameter LATENCY = 32,
    parameter TAG_WIDTH = 1
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [5:0]  opcode,
    input  wire [2:0]  fmt,
    input  wire [31:0] lhs,
    input  wire [31:0] rhs,
    input  wire [TAG_WIDTH-1:0] tag_in,
    output reg  [31:0] result,
    output reg         supported,
    output reg         divide_by_zero,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output reg         busy,
    output reg         done
);
    localparam ADD_SUB_LATENCY = 30;
    localparam MUL_LATENCY = 6;
    localparam ADD_SUB_DELAY = (LATENCY > ADD_SUB_LATENCY) ? (LATENCY - ADD_SUB_LATENCY) : 0;
    localparam MUL_DELAY = (LATENCY > MUL_LATENCY) ? (LATENCY - MUL_LATENCY) : 0;
    localparam ADD_SUB_DELAY_SLOTS = (ADD_SUB_DELAY == 0) ? 1 : ADD_SUB_DELAY;
    localparam MUL_DELAY_SLOTS = (MUL_DELAY == 0) ? 1 : MUL_DELAY;

    wire [31:0] add_sub_result;
    wire [31:0] mul_result;
    wire [31:0] div_result;
    wire add_sub_supported;
    wire mul_supported;
    wire div_supported;
    wire div_zero;

    reg [LATENCY-1:0] valid_pipe;
    reg [(LATENCY*6)-1:0] opcode_pipe;
    reg [(LATENCY*TAG_WIDTH)-1:0] tag_pipe;
    reg [(ADD_SUB_DELAY_SLOTS*32)-1:0] add_sub_result_delay;
    reg [ADD_SUB_DELAY_SLOTS-1:0] add_sub_supported_delay;
    reg [(MUL_DELAY_SLOTS*32)-1:0] mul_result_delay;
    reg [MUL_DELAY_SLOTS-1:0] mul_supported_delay;

    wire [31:0] aligned_add_sub_result = (ADD_SUB_DELAY == 0)
        ? add_sub_result
        : add_sub_result_delay[((ADD_SUB_DELAY-1)*32) +: 32];
    wire aligned_add_sub_supported = (ADD_SUB_DELAY == 0)
        ? add_sub_supported
        : add_sub_supported_delay[ADD_SUB_DELAY-1];
    wire [31:0] aligned_mul_result = (MUL_DELAY == 0)
        ? mul_result
        : mul_result_delay[((MUL_DELAY-1)*32) +: 32];
    wire aligned_mul_supported = (MUL_DELAY == 0)
        ? mul_supported
        : mul_supported_delay[MUL_DELAY-1];

    generate
        if (ENABLE_FLOAT_ADD) begin : gen_add_sub
            float_add_sub #(
                .FP32_ONLY(FLOAT_FP32_ONLY)
            ) add_sub_unit (
                .clk(clk),
                .rst(rst),
                .fmt(fmt),
                .subtract(opcode == `MGPU_OP_FSUB),
                .lhs(lhs),
                .rhs(rhs),
                .result(add_sub_result),
                .supported(add_sub_supported)
            );
        end else begin : gen_no_add_sub
            assign add_sub_result = 32'b0;
            assign add_sub_supported = 1'b0;
        end

        if (ENABLE_FLOAT_MUL) begin : gen_mul
            float_mul #(
                .FP32_ONLY(FLOAT_FP32_ONLY)
            ) mul_unit (
                .clk(clk),
                .rst(rst),
                .fmt(fmt),
                .lhs(lhs),
                .rhs(rhs),
                .result(mul_result),
                .supported(mul_supported)
            );
        end else begin : gen_no_mul
            assign mul_result = 32'b0;
            assign mul_supported = 1'b0;
        end

        if (ENABLE_FLOAT_DIV) begin : gen_div
            float_div #(
                .FP32_ONLY(FLOAT_FP32_ONLY)
            ) div_unit (
                .fmt(fmt),
                .lhs(lhs),
                .rhs(rhs),
                .result(div_result),
                .supported(div_supported),
                .divide_by_zero(div_zero)
            );
        end else begin : gen_no_div
            assign div_result = 32'b0;
            assign div_supported = 1'b0;
            assign div_zero = 1'b0;
        end
    endgenerate

    integer delay_index;

    always @(posedge clk) begin
        if (rst) begin
            result <= 32'b0;
            supported <= 1'b0;
            divide_by_zero <= 1'b0;
            tag_out <= {TAG_WIDTH{1'b0}};
            busy <= 1'b0;
            done <= 1'b0;
            valid_pipe <= {LATENCY{1'b0}};
            opcode_pipe <= {(LATENCY*6){1'b0}};
            tag_pipe <= {(LATENCY*TAG_WIDTH){1'b0}};
            add_sub_result_delay <= {(ADD_SUB_DELAY_SLOTS*32){1'b0}};
            add_sub_supported_delay <= {ADD_SUB_DELAY_SLOTS{1'b0}};
            mul_result_delay <= {(MUL_DELAY_SLOTS*32){1'b0}};
            mul_supported_delay <= {MUL_DELAY_SLOTS{1'b0}};
        end else begin
            valid_pipe <= {valid_pipe[LATENCY-2:0], start};
            opcode_pipe <= {opcode_pipe[((LATENCY-1)*6)-1:0], opcode};
            tag_pipe <= {tag_pipe[((LATENCY-1)*TAG_WIDTH)-1:0], tag_in};

            if (ADD_SUB_DELAY != 0) begin
                add_sub_result_delay[31:0] <= add_sub_result;
                add_sub_supported_delay[0] <= add_sub_supported;
                for (delay_index = 1; delay_index < ADD_SUB_DELAY; delay_index = delay_index + 1) begin
                    add_sub_result_delay[(delay_index*32) +: 32] <= add_sub_result_delay[((delay_index-1)*32) +: 32];
                    add_sub_supported_delay[delay_index] <= add_sub_supported_delay[delay_index-1];
                end
            end

            if (MUL_DELAY != 0) begin
                mul_result_delay[31:0] <= mul_result;
                mul_supported_delay[0] <= mul_supported;
                for (delay_index = 1; delay_index < MUL_DELAY; delay_index = delay_index + 1) begin
                    mul_result_delay[(delay_index*32) +: 32] <= mul_result_delay[((delay_index-1)*32) +: 32];
                    mul_supported_delay[delay_index] <= mul_supported_delay[delay_index-1];
                end
            end

            busy <= 1'b0;
            done <= valid_pipe[LATENCY-1];
            tag_out <= tag_pipe[((LATENCY-1)*TAG_WIDTH) +: TAG_WIDTH];
            divide_by_zero <= 1'b0;
            result <= 32'b0;
            supported <= 1'b0;

            case (opcode_pipe[((LATENCY-1)*6) +: 6])
                `MGPU_OP_FADD,
                `MGPU_OP_FSUB: begin
                    result <= aligned_add_sub_result;
                    supported <= aligned_add_sub_supported;
                end
                `MGPU_OP_FMUL: begin
                    result <= aligned_mul_result;
                    supported <= aligned_mul_supported;
                end
                `MGPU_OP_FDIV: begin
                    result <= div_result;
                    supported <= div_supported;
                    divide_by_zero <= div_zero;
                end
                default: begin
                    result <= 32'b0;
                    supported <= 1'b0;
                end
            endcase
        end
    end
endmodule
