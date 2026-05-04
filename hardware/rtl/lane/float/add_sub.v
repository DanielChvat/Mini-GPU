`timescale 1ns/1ps

`include "minigpu_isa.vh"

module float_add_sub #(
    parameter FP32_ONLY = 0,
    parameter NORMALIZE_STAGES = 24
) (
    input  wire        clk,
    input  wire        rst,
    input  wire [2:0]  fmt,
    input  wire        subtract,
    input  wire [31:0] lhs,
    input  wire [31:0] rhs,
    output reg  [31:0] result,
    output reg         supported
);
    reg [31:0] s0_lhs;
    reg [31:0] s0_rhs;
    reg [2:0] s0_fmt;
    reg s0_subtract;
    reg s0_supported;

    reg s1_sign_a;
    reg s1_sign_b;
    reg [9:0] s1_exp_a;
    reg [9:0] s1_exp_b;
    reg [31:0] s1_mant_a;
    reg [31:0] s1_mant_b;
    reg [2:0] s1_fmt;
    reg s1_supported;
    reg s1_a_zero;
    reg s1_b_zero;

    reg s2_sign_a;
    reg s2_sign_b;
    reg [9:0] s2_exp;
    reg [31:0] s2_mant_a;
    reg [31:0] s2_mant_b;
    reg [2:0] s2_fmt;
    reg s2_supported;
    reg s2_prepacked;
    reg [31:0] s2_result;

    reg [32:0] s3_mant;
    reg [9:0] s3_exp;
    reg [2:0] s3_fmt;
    reg s3_sign;
    reg s3_supported;
    reg s3_prepacked;
    reg [31:0] s3_result;

    reg [32:0] norm_mant [0:NORMALIZE_STAGES];
    reg [9:0] norm_exp [0:NORMALIZE_STAGES];
    reg [2:0] norm_fmt [0:NORMALIZE_STAGES];
    reg norm_sign [0:NORMALIZE_STAGES];
    reg norm_supported [0:NORMALIZE_STAGES];
    reg norm_prepacked [0:NORMALIZE_STAGES];
    reg [31:0] norm_result [0:NORMALIZE_STAGES];

    integer stage;

    always @(posedge clk) begin
        if (rst) begin
            s0_lhs <= 32'b0;
            s0_rhs <= 32'b0;
            s0_fmt <= 3'b0;
            s0_subtract <= 1'b0;
            s0_supported <= 1'b0;
            s1_sign_a <= 1'b0;
            s1_sign_b <= 1'b0;
            s1_exp_a <= 10'b0;
            s1_exp_b <= 10'b0;
            s1_mant_a <= 32'b0;
            s1_mant_b <= 32'b0;
            s1_fmt <= 3'b0;
            s1_supported <= 1'b0;
            s1_a_zero <= 1'b0;
            s1_b_zero <= 1'b0;
            s2_sign_a <= 1'b0;
            s2_sign_b <= 1'b0;
            s2_exp <= 10'b0;
            s2_mant_a <= 32'b0;
            s2_mant_b <= 32'b0;
            s2_fmt <= 3'b0;
            s2_supported <= 1'b0;
            s2_prepacked <= 1'b0;
            s2_result <= 32'b0;
            s3_mant <= 33'b0;
            s3_exp <= 10'b0;
            s3_fmt <= 3'b0;
            s3_sign <= 1'b0;
            s3_supported <= 1'b0;
            s3_prepacked <= 1'b0;
            s3_result <= 32'b0;
            result <= 32'b0;
            supported <= 1'b0;

            for (stage = 0; stage <= NORMALIZE_STAGES; stage = stage + 1) begin
                norm_mant[stage] <= 33'b0;
                norm_exp[stage] <= 10'b0;
                norm_fmt[stage] <= 3'b0;
                norm_sign[stage] <= 1'b0;
                norm_supported[stage] <= 1'b0;
                norm_prepacked[stage] <= 1'b0;
                norm_result[stage] <= 32'b0;
            end
        end else begin
            s0_lhs <= lhs;
            s0_rhs <= rhs;
            s0_fmt <= fmt;
            s0_subtract <= subtract;
            s0_supported <= fmt_supported(fmt);

            s1_sign_a <= sign_for(s0_lhs, s0_fmt);
            s1_sign_b <= sign_for(s0_rhs, s0_fmt) ^ s0_subtract;
            s1_exp_a <= exp_for(s0_lhs, s0_fmt);
            s1_exp_b <= exp_for(s0_rhs, s0_fmt);
            s1_mant_a <= frac_for(s0_lhs, s0_fmt);
            s1_mant_b <= frac_for(s0_rhs, s0_fmt);
            s1_fmt <= s0_fmt;
            s1_supported <= s0_supported;
            s1_a_zero <= (exp_for(s0_lhs, s0_fmt) == 10'd0) &&
                         (frac_for(s0_lhs, s0_fmt) == 32'd0);
            s1_b_zero <= (exp_for(s0_rhs, s0_fmt) == 10'd0) &&
                         (frac_for(s0_rhs, s0_fmt) == 32'd0);

            s2_fmt <= s1_fmt;
            s2_supported <= s1_supported;
            s2_prepacked <= !s1_supported || s1_a_zero || s1_b_zero;
            s2_result <= prepacked_result(
                s1_sign_a,
                s1_sign_b,
                s1_exp_a,
                s1_exp_b,
                s1_mant_a,
                s1_mant_b,
                s1_fmt,
                s1_supported,
                s1_a_zero
            );
            s2_sign_a <= s1_sign_a;
            s2_sign_b <= s1_sign_b;
            s2_exp <= aligned_exp(s1_exp_a, s1_exp_b);
            s2_mant_a <= aligned_mant(s1_mant_a, s1_exp_a, s1_exp_b, s1_fmt);
            s2_mant_b <= aligned_mant(s1_mant_b, s1_exp_b, s1_exp_a, s1_fmt);

            s3_fmt <= s2_fmt;
            s3_supported <= s2_supported;
            s3_prepacked <= s2_prepacked;
            s3_result <= s2_result;
            s3_exp <= s2_exp;
            if (s2_sign_a == s2_sign_b) begin
                s3_mant <= s2_mant_a + s2_mant_b;
                s3_sign <= s2_sign_a;
                if ((s2_mant_a + s2_mant_b) >> (mant_bits_for(s2_fmt) + 1)) begin
                    s3_mant <= (s2_mant_a + s2_mant_b) >> 1;
                    s3_exp <= s2_exp + 10'd1;
                end
            end else if (s2_mant_a >= s2_mant_b) begin
                s3_mant <= s2_mant_a - s2_mant_b;
                s3_sign <= s2_sign_a;
            end else begin
                s3_mant <= s2_mant_b - s2_mant_a;
                s3_sign <= s2_sign_b;
            end

            norm_mant[0] <= s3_mant;
            norm_exp[0] <= s3_exp;
            norm_fmt[0] <= s3_fmt;
            norm_sign[0] <= s3_sign;
            norm_supported[0] <= s3_supported;
            norm_prepacked[0] <= s3_prepacked;
            norm_result[0] <= s3_result;

            for (stage = 1; stage <= NORMALIZE_STAGES; stage = stage + 1) begin
                norm_fmt[stage] <= norm_fmt[stage - 1];
                norm_sign[stage] <= norm_sign[stage - 1];
                norm_supported[stage] <= norm_supported[stage - 1];
                norm_prepacked[stage] <= norm_prepacked[stage - 1];
                norm_result[stage] <= norm_result[stage - 1];

                if (!norm_prepacked[stage - 1] &&
                    (norm_mant[stage - 1] != 33'b0) &&
                    ((norm_mant[stage - 1] >> mant_bits_for(norm_fmt[stage - 1])) == 0) &&
                    (norm_exp[stage - 1] > 10'd0)) begin
                    norm_mant[stage] <= norm_mant[stage - 1] << 1;
                    norm_exp[stage] <= norm_exp[stage - 1] - 10'd1;
                end else begin
                    norm_mant[stage] <= norm_mant[stage - 1];
                    norm_exp[stage] <= norm_exp[stage - 1];
                end
            end

            supported <= norm_supported[NORMALIZE_STAGES];
            result <= final_result(
                norm_sign[NORMALIZE_STAGES],
                norm_exp[NORMALIZE_STAGES],
                norm_mant[NORMALIZE_STAGES],
                norm_fmt[NORMALIZE_STAGES],
                norm_supported[NORMALIZE_STAGES],
                norm_prepacked[NORMALIZE_STAGES],
                norm_result[NORMALIZE_STAGES]
            );
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

    function sign_for;
        input [31:0] value;
        input [2:0] value_fmt;
        begin
            sign_for = (value >> (exp_bits_for(value_fmt) + mant_bits_for(value_fmt))) & 1'b1;
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

    function [31:0] frac_for;
        input [31:0] value;
        input [2:0] value_fmt;
        integer mant_bits;
        reg [31:0] frac_mask;
        begin
            mant_bits = mant_bits_for(value_fmt);
            frac_mask = (32'h1 << mant_bits) - 1;
            frac_for = value & frac_mask;
        end
    endfunction

    function [9:0] aligned_exp;
        input [9:0] exp_a;
        input [9:0] exp_b;
        begin
            aligned_exp = (exp_a > exp_b) ? exp_a : exp_b;
        end
    endfunction

    function [31:0] aligned_mant;
        input [31:0] mantissa;
        input [9:0] self_exp;
        input [9:0] other_exp;
        input [2:0] value_fmt;
        integer shift;
        reg [31:0] mant_with_hidden;
        begin
            mant_with_hidden = (self_exp != 10'd0)
                ? (mantissa | (32'h1 << mant_bits_for(value_fmt)))
                : mantissa;
            if (self_exp >= other_exp) begin
                aligned_mant = mant_with_hidden;
            end else begin
                shift = other_exp - self_exp;
                aligned_mant = (shift > 31) ? 32'b0 : (mant_with_hidden >> shift);
            end
        end
    endfunction

    function [31:0] prepacked_result;
        input sign_a;
        input sign_b;
        input [9:0] exp_a;
        input [9:0] exp_b;
        input [31:0] mant_a;
        input [31:0] mant_b;
        input [2:0] value_fmt;
        input value_supported;
        input a_is_zero;
        begin
            if (!value_supported) begin
                prepacked_result = 32'b0;
            end else if (a_is_zero) begin
                prepacked_result = pack_float(sign_b, exp_b, mant_b, value_fmt);
            end else begin
                prepacked_result = pack_float(sign_a, exp_a, mant_a, value_fmt);
            end
        end
    endfunction

    function [31:0] final_result;
        input sign;
        input [9:0] exponent;
        input [32:0] mantissa;
        input [2:0] value_fmt;
        input value_supported;
        input value_prepacked;
        input [31:0] value_result;
        reg [31:0] frac_mask;
        begin
            frac_mask = (32'h1 << mant_bits_for(value_fmt)) - 1;
            if (!value_supported) begin
                final_result = 32'b0;
            end else if (value_prepacked) begin
                final_result = value_result;
            end else if (mantissa == 33'b0) begin
                final_result = 32'b0;
            end else begin
                final_result = pack_float(sign, exponent, mantissa[31:0] & frac_mask, value_fmt);
            end
        end
    endfunction

    function [31:0] pack_float;
        input sign;
        input integer exponent;
        input [31:0] fraction;
        input [2:0] value_fmt;
        integer exp_bits;
        integer mant_bits;
        integer total_bits;
        integer max_exp;
        reg [31:0] frac_mask;
        begin
            exp_bits = exp_bits_for(value_fmt);
            mant_bits = mant_bits_for(value_fmt);
            total_bits = 1 + exp_bits + mant_bits;
            max_exp = (1 << exp_bits) - 1;
            frac_mask = (32'h1 << mant_bits) - 1;
            if (exponent >= max_exp) begin
                pack_float = ({31'b0, sign} << (total_bits - 1)) | (max_exp << mant_bits);
            end else if (exponent <= 0) begin
                pack_float = 32'b0;
            end else begin
                pack_float = ({31'b0, sign} << (total_bits - 1)) |
                             (exponent << mant_bits) |
                             (fraction & frac_mask);
            end
        end
    endfunction
endmodule
