`timescale 1ns/1ps

`include "minigpu_isa.vh"

module float_mul #(
    parameter FP32_ONLY = 0
) (
    input  wire        clk,
    input  wire        rst,
    input  wire [2:0]  fmt,
    input  wire [31:0] lhs,
    input  wire [31:0] rhs,
    output reg  [31:0] result,
    output reg         supported
);
    reg [31:0] stage0_lhs;
    reg [31:0] stage0_rhs;
    reg [2:0] stage0_fmt;
    reg stage0_supported;

    reg stage1_sign;
    reg [2:0] stage1_fmt;
    reg stage1_supported;
    reg [9:0] stage1_exp_r;
    reg [31:0] stage1_lhs_mant;
    reg [31:0] stage1_rhs_mant;

    reg stage2_sign;
    reg [2:0] stage2_fmt;
    reg stage2_supported;
    reg [9:0] stage2_exp_r;
    (* use_dsp = "yes" *) reg [63:0] stage2_product;

    reg stage3_sign;
    reg [2:0] stage3_fmt;
    reg stage3_supported;
    reg [9:0] stage3_exp_r;
    reg [31:0] stage3_fraction;

    reg [31:0] stage4_result;
    reg stage4_supported;

    always @(posedge clk) begin
        if (rst) begin
            stage0_lhs <= 32'b0;
            stage0_rhs <= 32'b0;
            stage0_fmt <= 3'b0;
            stage0_supported <= 1'b0;
            stage1_sign <= 1'b0;
            stage1_fmt <= 3'b0;
            stage1_supported <= 1'b0;
            stage1_exp_r <= 10'b0;
            stage1_lhs_mant <= 32'b0;
            stage1_rhs_mant <= 32'b0;
            stage2_sign <= 1'b0;
            stage2_fmt <= 3'b0;
            stage2_supported <= 1'b0;
            stage2_exp_r <= 10'b0;
            stage2_product <= 64'b0;
            stage3_sign <= 1'b0;
            stage3_fmt <= 3'b0;
            stage3_supported <= 1'b0;
            stage3_exp_r <= 10'b0;
            stage3_fraction <= 32'b0;
            stage4_result <= 32'b0;
            stage4_supported <= 1'b0;
            result <= 32'b0;
            supported <= 1'b0;
        end else begin
            stage0_lhs <= lhs;
            stage0_rhs <= rhs;
            stage0_fmt <= fmt;
            stage0_supported <= fmt_supported(fmt);

            stage1_sign <= sign_for(stage0_lhs, stage0_rhs, stage0_fmt);
            stage1_fmt <= stage0_fmt;
            stage1_supported <= stage0_supported && !zero_operand(stage0_lhs, stage0_fmt) &&
                                !zero_operand(stage0_rhs, stage0_fmt);
            stage1_exp_r <= exp_for(stage0_lhs, stage0_fmt) +
                            exp_for(stage0_rhs, stage0_fmt) -
                            bias_for(stage0_fmt);
            stage1_lhs_mant <= mant_for(stage0_lhs, stage0_fmt);
            stage1_rhs_mant <= mant_for(stage0_rhs, stage0_fmt);

            stage2_sign <= stage1_sign;
            stage2_fmt <= stage1_fmt;
            stage2_supported <= stage1_supported;
            stage2_exp_r <= stage1_exp_r;
            stage2_product <= stage1_lhs_mant * stage1_rhs_mant;

            stage3_sign <= stage2_sign;
            stage3_fmt <= stage2_fmt;
            stage3_supported <= stage2_supported;
            stage3_exp_r <= normalized_exp(stage2_exp_r, stage2_product, stage2_fmt);
            stage3_fraction <= normalized_fraction(stage2_product, stage2_fmt);

            stage4_supported <= stage3_supported;
            stage4_result <= stage3_supported
                ? pack_float(stage3_sign, stage3_exp_r, stage3_fraction,
                             exp_bits_for(stage3_fmt), mant_bits_for(stage3_fmt))
                : 32'b0;

            supported <= stage4_supported;
            result <= stage4_result;
        end
    end

    function fmt_supported;
        input [2:0] value_fmt;
        begin
            fmt_supported = FP32_ONLY
                ? (value_fmt == `MGPU_FMT_FP32)
                : ((value_fmt == `MGPU_FMT_FP32) ||
                   (value_fmt == `MGPU_FMT_FP16) ||
                   (value_fmt == `MGPU_FMT_FP8));
        end
    endfunction

    function [5:0] exp_bits_for;
        input [2:0] value_fmt;
        begin
            case (value_fmt)
                `MGPU_FMT_FP16: exp_bits_for = 6'd5;
                `MGPU_FMT_FP8:  exp_bits_for = 6'd4;
                default:       exp_bits_for = 6'd8;
            endcase
        end
    endfunction

    function [5:0] mant_bits_for;
        input [2:0] value_fmt;
        begin
            case (value_fmt)
                `MGPU_FMT_FP16: mant_bits_for = 6'd10;
                `MGPU_FMT_FP8:  mant_bits_for = 6'd3;
                default:       mant_bits_for = 6'd23;
            endcase
        end
    endfunction

    function [9:0] bias_for;
        input [2:0] value_fmt;
        begin
            case (value_fmt)
                `MGPU_FMT_FP16: bias_for = 10'd15;
                `MGPU_FMT_FP8:  bias_for = 10'd7;
                default:       bias_for = 10'd127;
            endcase
        end
    endfunction

    function sign_for;
        input [31:0] a;
        input [31:0] b;
        input [2:0] value_fmt;
        integer total_bits;
        begin
            total_bits = 1 + exp_bits_for(value_fmt) + mant_bits_for(value_fmt);
            sign_for = ((a >> (total_bits - 1)) ^ (b >> (total_bits - 1))) & 1'b1;
        end
    endfunction

    function [9:0] exp_for;
        input [31:0] value;
        input [2:0] value_fmt;
        integer exp_bits;
        integer mant_bits;
        integer max_exp;
        begin
            exp_bits = exp_bits_for(value_fmt);
            mant_bits = mant_bits_for(value_fmt);
            max_exp = (1 << exp_bits) - 1;
            exp_for = (value >> mant_bits) & max_exp;
        end
    endfunction

    function [31:0] mant_for;
        input [31:0] value;
        input [2:0] value_fmt;
        integer mant_bits;
        reg [31:0] frac_mask;
        begin
            mant_bits = mant_bits_for(value_fmt);
            frac_mask = (32'h1 << mant_bits) - 1;
            mant_for = (value & frac_mask) | (32'h1 << mant_bits);
        end
    endfunction

    function zero_operand;
        input [31:0] value;
        input [2:0] value_fmt;
        begin
            zero_operand = (exp_for(value, value_fmt) == 0) &&
                           ((value & ((32'h1 << mant_bits_for(value_fmt)) - 1)) == 0);
        end
    endfunction

    function product_needs_shift;
        input [63:0] product;
        input [2:0] value_fmt;
        integer mant_bits;
        begin
            mant_bits = mant_bits_for(value_fmt);
            product_needs_shift = (product >> ((2 * mant_bits) + 1)) != 0;
        end
    endfunction

    function [9:0] normalized_exp;
        input [9:0] exponent;
        input [63:0] product;
        input [2:0] value_fmt;
        begin
            normalized_exp = product_needs_shift(product, value_fmt)
                ? (exponent + 10'd1)
                : exponent;
        end
    endfunction

    function [31:0] normalized_fraction;
        input [63:0] product;
        input [2:0] value_fmt;
        integer mant_bits;
        integer shift;
        reg [31:0] mant_r;
        reg [31:0] frac_mask;
        begin
            mant_bits = mant_bits_for(value_fmt);
            frac_mask = (32'h1 << mant_bits) - 1;
            shift = product_needs_shift(product, value_fmt)
                ? (mant_bits + 1)
                : mant_bits;

            mant_r = (product >> shift) & ((32'h1 << (mant_bits + 1)) - 1);
            normalized_fraction = mant_r & frac_mask;
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
