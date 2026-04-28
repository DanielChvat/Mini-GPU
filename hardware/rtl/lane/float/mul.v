`timescale 1ns/1ps

`include "minigpu_isa.vh"

module float_mul (
    input  wire [2:0]  fmt,
    input  wire [31:0] lhs,
    input  wire [31:0] rhs,
    output wire [31:0] result,
    output wire        supported
);
    assign supported = (fmt == `MGPU_FMT_FP32) || (fmt == `MGPU_FMT_FP16) || (fmt == `MGPU_FMT_FP8);
    assign result = supported ? float_mul_any(lhs, rhs, fmt) : 32'b0;

    function [31:0] float_mul_any;
        input [31:0] a;
        input [31:0] b;
        input [2:0] value_fmt;
        begin
            case (value_fmt)
                `MGPU_FMT_FP16: float_mul_any = fp_mul_core(a, b, 5, 10, 15);
                `MGPU_FMT_FP8:  float_mul_any = fp_mul_core(a, b, 4, 3, 7);
                default:       float_mul_any = fp_mul_core(a, b, 8, 23, 127);
            endcase
        end
    endfunction

    function [31:0] fp_mul_core;
        input [31:0] a;
        input [31:0] b;
        input integer exp_bits;
        input integer mant_bits;
        input integer bias;
        integer total_bits;
        integer max_exp;
        integer exp_a;
        integer exp_b;
        integer exp_r;
        integer shift;
        reg sign_r;
        reg [31:0] frac_mask;
        reg [31:0] mant_a;
        reg [31:0] mant_b;
        reg [63:0] product;
        reg [31:0] mant_r;
        begin
            total_bits = 1 + exp_bits + mant_bits;
            max_exp = (1 << exp_bits) - 1;
            frac_mask = (32'h1 << mant_bits) - 1;
            sign_r = ((a >> (total_bits - 1)) ^ (b >> (total_bits - 1))) & 1'b1;
            exp_a = (a >> mant_bits) & max_exp;
            exp_b = (b >> mant_bits) & max_exp;
            mant_a = a & frac_mask;
            mant_b = b & frac_mask;

            if ((exp_a == 0 && mant_a == 0) || (exp_b == 0 && mant_b == 0)) begin
                fp_mul_core = 32'b0;
            end else begin
                mant_a = mant_a | (32'h1 << mant_bits);
                mant_b = mant_b | (32'h1 << mant_bits);
                exp_r = exp_a + exp_b - bias;
                product = mant_a * mant_b;

                if ((product >> ((2 * mant_bits) + 1)) != 0) begin
                    shift = mant_bits + 1;
                    exp_r = exp_r + 1;
                end else begin
                    shift = mant_bits;
                end

                mant_r = (product >> shift) & ((32'h1 << (mant_bits + 1)) - 1);
                fp_mul_core = pack_float(sign_r, exp_r, mant_r & frac_mask, exp_bits, mant_bits);
            end
        end
    endfunction

    function [31:0] pack_float;
        input sign;
        input integer exponent;
        input [31:0] fraction;
        input integer exp_bits;
        input integer mant_bits;
        integer total_bits;
        integer max_exp;
        reg [31:0] frac_mask;
        begin
            total_bits = 1 + exp_bits + mant_bits;
            max_exp = (1 << exp_bits) - 1;
            frac_mask = (32'h1 << mant_bits) - 1;
            if (exponent >= max_exp) begin
                pack_float = ({31'b0, sign} << (total_bits - 1)) | (max_exp << mant_bits);
            end else if (exponent <= 0) begin
                pack_float = 32'b0;
            end else begin
                pack_float = ({31'b0, sign} << (total_bits - 1)) | (exponent << mant_bits) | (fraction & frac_mask);
            end
        end
    endfunction
endmodule
