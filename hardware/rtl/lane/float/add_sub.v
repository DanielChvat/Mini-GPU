`timescale 1ns/1ps

`include "minigpu_isa.vh"

module float_add_sub (
    input  wire [2:0]  fmt,
    input  wire        subtract,
    input  wire [31:0] lhs,
    input  wire [31:0] rhs,
    output wire [31:0] result,
    output wire        supported
);
    assign supported = is_float_format(fmt);
    assign result = supported ? float_add_any(lhs, rhs, subtract, fmt) : 32'b0;

    function is_float_format;
        input [2:0] value_fmt;
        begin
            is_float_format = (value_fmt == `MGPU_FMT_FP32) ||
                              (value_fmt == `MGPU_FMT_FP16) ||
                              (value_fmt == `MGPU_FMT_FP8);
        end
    endfunction

    function [31:0] float_add_any;
        input [31:0] a;
        input [31:0] b;
        input sub;
        input [2:0] value_fmt;
        begin
            case (value_fmt)
                `MGPU_FMT_FP16: float_add_any = fp_add_core(a, b, sub, 5, 10, 15);
                `MGPU_FMT_FP8:  float_add_any = fp_add_core(a, b, sub, 4, 3, 7);
                default:       float_add_any = fp_add_core(a, b, sub, 8, 23, 127);
            endcase
        end
    endfunction

    function [31:0] fp_add_core;
        input [31:0] a;
        input [31:0] b;
        input sub;
        input integer exp_bits;
        input integer mant_bits;
        input integer bias;
        integer total_bits;
        integer max_exp;
        integer exp_a;
        integer exp_b;
        integer exp_r;
        integer shift;
        reg sign_a;
        reg sign_b;
        reg sign_r;
        reg [31:0] frac_mask;
        reg [31:0] mant_a;
        reg [31:0] mant_b;
        reg [32:0] mant_r;
        begin
            total_bits = 1 + exp_bits + mant_bits;
            max_exp = (1 << exp_bits) - 1;
            frac_mask = (32'h1 << mant_bits) - 1;
            sign_a = (a >> (total_bits - 1)) & 1'b1;
            sign_b = ((b >> (total_bits - 1)) & 1'b1) ^ sub;
            exp_a = (a >> mant_bits) & max_exp;
            exp_b = (b >> mant_bits) & max_exp;
            mant_a = a & frac_mask;
            mant_b = b & frac_mask;

            if (exp_a == 0 && mant_a == 0) begin
                fp_add_core = pack_float(sign_b, exp_b, mant_b, exp_bits, mant_bits);
            end else if (exp_b == 0 && mant_b == 0) begin
                fp_add_core = pack_float(sign_a, exp_a, mant_a, exp_bits, mant_bits);
            end else begin
                if (exp_a != 0) mant_a = mant_a | (32'h1 << mant_bits);
                if (exp_b != 0) mant_b = mant_b | (32'h1 << mant_bits);

                if (exp_a > exp_b) begin
                    shift = exp_a - exp_b;
                    mant_b = (shift > 31) ? 32'b0 : (mant_b >> shift);
                    exp_r = exp_a;
                end else begin
                    shift = exp_b - exp_a;
                    mant_a = (shift > 31) ? 32'b0 : (mant_a >> shift);
                    exp_r = exp_b;
                end

                if (sign_a == sign_b) begin
                    mant_r = mant_a + mant_b;
                    sign_r = sign_a;
                    if (mant_r >> (mant_bits + 1)) begin
                        mant_r = mant_r >> 1;
                        exp_r = exp_r + 1;
                    end
                end else if (mant_a >= mant_b) begin
                    mant_r = mant_a - mant_b;
                    sign_r = sign_a;
                end else begin
                    mant_r = mant_b - mant_a;
                    sign_r = sign_b;
                end

                while ((mant_r != 0) && ((mant_r >> mant_bits) == 0) && (exp_r > 0)) begin
                    mant_r = mant_r << 1;
                    exp_r = exp_r - 1;
                end

                if (mant_r == 0) begin
                    fp_add_core = 32'b0;
                end else begin
                    fp_add_core = pack_float(sign_r, exp_r, mant_r[31:0] & frac_mask, exp_bits, mant_bits);
                end
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
