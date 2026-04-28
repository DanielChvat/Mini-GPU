`timescale 1ns/1ps

`include "minigpu_isa.vh"

module int_mul #(
    parameter WIDTH = 32
) (
    input  wire [2:0]       fmt,
    input  wire [WIDTH-1:0] lhs,
    input  wire [WIDTH-1:0] rhs,
    output wire [WIDTH-1:0] result
);
    wire signed [WIDTH-1:0] lhs_ext = sign_extend_int(lhs, fmt);
    wire signed [WIDTH-1:0] rhs_ext = sign_extend_int(rhs, fmt);
    wire signed [(2*WIDTH)-1:0] product = lhs_ext * rhs_ext;

    assign result = narrow_int_result(product, fmt);

    function [WIDTH-1:0] sign_extend_int;
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

    function [WIDTH-1:0] narrow_int_result;
        input signed [(2*WIDTH)-1:0] value;
        input [2:0] value_fmt;
        begin
            case (value_fmt)
                `MGPU_FMT_I8:  narrow_int_result = {{(WIDTH-8){value[7]}}, value[7:0]};
                `MGPU_FMT_I16: narrow_int_result = {{(WIDTH-16){value[15]}}, value[15:0]};
                default:       narrow_int_result = value[WIDTH-1:0];
            endcase
        end
    endfunction
endmodule
