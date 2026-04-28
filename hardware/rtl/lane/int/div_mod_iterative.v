`timescale 1ns/1ps

`include "minigpu_isa.vh"

module int_div_mod_iterative #(
    parameter WIDTH = 32
) (
    input  wire             clk,
    input  wire             rst,
    input  wire             start,
    input  wire [2:0]       fmt,
    input  wire             modulo,
    input  wire [WIDTH-1:0] lhs,
    input  wire [WIDTH-1:0] rhs,
    output reg  [WIDTH-1:0] result,
    output reg              divide_by_zero,
    output reg              busy,
    output reg              done
);
    reg [2:0] fmt_r;
    reg modulo_r;
    reg quotient_negative;
    reg remainder_negative;
    reg [WIDTH-1:0] dividend_shift;
    reg [WIDTH-1:0] divisor_abs;
    reg [WIDTH-1:0] quotient;
    reg [WIDTH:0] remainder;
    reg [5:0] bits_left;

    wire [WIDTH:0] shifted_remainder = {remainder[WIDTH-1:0], dividend_shift[WIDTH-1]};
    wire subtract_ok = shifted_remainder >= {1'b0, divisor_abs};
    wire [WIDTH:0] next_remainder = subtract_ok ? (shifted_remainder - {1'b0, divisor_abs}) : shifted_remainder;
    wire [WIDTH-1:0] next_quotient = {quotient[WIDTH-2:0], subtract_ok};
    wire [WIDTH-1:0] next_dividend_shift = {dividend_shift[WIDTH-2:0], 1'b0};

    always @(posedge clk) begin
        if (rst) begin
            result <= {WIDTH{1'b0}};
            divide_by_zero <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
            fmt_r <= `MGPU_FMT_I32;
            modulo_r <= 1'b0;
            quotient_negative <= 1'b0;
            remainder_negative <= 1'b0;
            dividend_shift <= {WIDTH{1'b0}};
            divisor_abs <= {WIDTH{1'b0}};
            quotient <= {WIDTH{1'b0}};
            remainder <= {(WIDTH+1){1'b0}};
            bits_left <= 6'b0;
        end else begin
            done <= 1'b0;

            if (start && !busy) begin
                fmt_r <= fmt;
                modulo_r <= modulo;
                divide_by_zero <= (sign_extend_int(rhs, fmt) == {WIDTH{1'b0}});
                result <= {WIDTH{1'b0}};

                if (sign_extend_int(rhs, fmt) == {WIDTH{1'b0}}) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end else begin
                    busy <= 1'b1;
                    dividend_shift <= align_dividend(abs_signed(sign_extend_int(lhs, fmt)), fmt);
                    divisor_abs <= abs_signed(sign_extend_int(rhs, fmt));
                    quotient <= {WIDTH{1'b0}};
                    remainder <= {(WIDTH+1){1'b0}};
                    quotient_negative <= sign_bit(lhs, fmt) ^ sign_bit(rhs, fmt);
                    remainder_negative <= sign_bit(lhs, fmt);
                    bits_left <= operand_bits(fmt);
                end
            end else if (busy) begin
                dividend_shift <= next_dividend_shift;
                quotient <= next_quotient;
                remainder <= next_remainder;

                if (bits_left <= 6'd1) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    if (modulo_r) begin
                        result <= narrow_int_result(remainder_negative ? negate(next_remainder[WIDTH-1:0]) : next_remainder[WIDTH-1:0], fmt_r);
                    end else begin
                        result <= narrow_int_result(quotient_negative ? negate(next_quotient) : next_quotient, fmt_r);
                    end
                end else begin
                    bits_left <= bits_left - 6'd1;
                end
            end
        end
    end

    function [5:0] operand_bits;
        input [2:0] value_fmt;
        begin
            case (value_fmt)
                `MGPU_FMT_I8: operand_bits = 6'd8;
                `MGPU_FMT_I16: operand_bits = 6'd16;
                default: operand_bits = 6'd32;
            endcase
        end
    endfunction

    function [WIDTH-1:0] align_dividend;
        input [WIDTH-1:0] value;
        input [2:0] value_fmt;
        begin
            case (value_fmt)
                `MGPU_FMT_I8: align_dividend = value << (WIDTH - 8);
                `MGPU_FMT_I16: align_dividend = value << (WIDTH - 16);
                default: align_dividend = value;
            endcase
        end
    endfunction

    function signed [WIDTH-1:0] sign_extend_int;
        input [WIDTH-1:0] value;
        input [2:0] value_fmt;
        begin
            case (value_fmt)
                `MGPU_FMT_I8:  sign_extend_int = {{(WIDTH-8){value[7]}}, value[7:0]};
                `MGPU_FMT_I16: sign_extend_int = {{(WIDTH-16){value[15]}}, value[15:0]};
                default:       sign_extend_int = value;
            endcase
        end
    endfunction

    function sign_bit;
        input [WIDTH-1:0] value;
        input [2:0] value_fmt;
        begin
            case (value_fmt)
                `MGPU_FMT_I8: sign_bit = value[7];
                `MGPU_FMT_I16: sign_bit = value[15];
                default: sign_bit = value[WIDTH-1];
            endcase
        end
    endfunction

    function [WIDTH-1:0] abs_signed;
        input signed [WIDTH-1:0] value;
        begin
            abs_signed = value[WIDTH-1] ? negate(value) : value;
        end
    endfunction

    function [WIDTH-1:0] negate;
        input [WIDTH-1:0] value;
        begin
            negate = (~value) + {{(WIDTH-1){1'b0}}, 1'b1};
        end
    endfunction

    function [WIDTH-1:0] narrow_int_result;
        input [WIDTH-1:0] value;
        input [2:0] value_fmt;
        begin
            case (value_fmt)
                `MGPU_FMT_I8:  narrow_int_result = {{(WIDTH-8){value[7]}}, value[7:0]};
                `MGPU_FMT_I16: narrow_int_result = {{(WIDTH-16){value[15]}}, value[15:0]};
                default:       narrow_int_result = value;
            endcase
        end
    endfunction
endmodule
