`timescale 1ns/1ps

`include "minigpu_isa.vh"

module shared_fpu #(
    parameter ENABLE_FLOAT_ADD = 1,
    parameter ENABLE_FLOAT_MUL = 1,
    parameter ENABLE_FLOAT_DIV = 1,
    parameter FLOAT_FP32_ONLY = 0,
    parameter LATENCY = 4
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [5:0]  opcode,
    input  wire [2:0]  fmt,
    input  wire [31:0] lhs,
    input  wire [31:0] rhs,
    output reg  [31:0] result,
    output reg         supported,
    output reg         divide_by_zero,
    output reg         busy,
    output reg         done
);
    reg [5:0] opcode_r;
    reg [2:0] fmt_r;
    reg [31:0] lhs_r;
    reg [31:0] rhs_r;
    reg [7:0] cycles_left;

    wire [31:0] add_sub_result;
    wire [31:0] mul_result;
    wire [31:0] div_result;
    wire add_sub_supported;
    wire mul_supported;
    wire div_supported;
    wire div_zero;

    generate
        if (ENABLE_FLOAT_ADD) begin : gen_add_sub
            float_add_sub #(
                .FP32_ONLY(FLOAT_FP32_ONLY)
            ) add_sub_unit (
                .clk(clk),
                .rst(rst),
                .fmt(start ? fmt : fmt_r),
                .subtract(start ? (opcode == `MGPU_OP_FSUB) : (opcode_r == `MGPU_OP_FSUB)),
                .lhs(start ? lhs : lhs_r),
                .rhs(start ? rhs : rhs_r),
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
                .fmt(start ? fmt : fmt_r),
                .lhs(start ? lhs : lhs_r),
                .rhs(start ? rhs : rhs_r),
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
                .fmt(fmt_r),
                .lhs(lhs_r),
                .rhs(rhs_r),
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

    always @(posedge clk) begin
        if (rst) begin
            result <= 32'b0;
            supported <= 1'b0;
            divide_by_zero <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
            opcode_r <= 6'b0;
            fmt_r <= 3'b0;
            lhs_r <= 32'b0;
            rhs_r <= 32'b0;
            cycles_left <= 8'b0;
        end else begin
            done <= 1'b0;

            if (start && !busy) begin
                opcode_r <= opcode;
                fmt_r <= fmt;
                lhs_r <= lhs;
                rhs_r <= rhs;
                cycles_left <= LATENCY[7:0];
                busy <= 1'b1;
            end else if (busy) begin
                if (cycles_left <= 8'd1) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    divide_by_zero <= 1'b0;
                    result <= 32'b0;
                    supported <= 1'b0;

                    case (opcode_r)
                        `MGPU_OP_FADD,
                        `MGPU_OP_FSUB: begin
                            result <= add_sub_result;
                            supported <= add_sub_supported;
                        end
                        `MGPU_OP_FMUL: begin
                            result <= mul_result;
                            supported <= mul_supported;
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
                end else begin
                    cycles_left <= cycles_left - 8'd1;
                end
            end
        end
    end
endmodule
