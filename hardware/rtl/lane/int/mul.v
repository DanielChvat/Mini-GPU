`timescale 1ns/1ps

`include "minigpu_isa.vh"

module int_mul #(
    parameter WIDTH = 32
) (
    input  wire                   clk,
    input  wire                   rst,
    input  wire [2:0]       fmt,
    input  wire [WIDTH-1:0] lhs,
    input  wire [WIDTH-1:0] rhs,
    output reg  [WIDTH-1:0] result
);
    reg [2:0] stage0_fmt;
    reg signed [WIDTH-1:0] stage0_lhs;
    reg signed [WIDTH-1:0] stage0_rhs;
    reg [2:0] stage1_fmt;
    (* use_dsp = "yes" *) reg signed [(2*WIDTH)-1:0] stage1_product;

    always @(posedge clk) begin
        if (rst) begin
            stage0_fmt <= 3'b0;
            stage0_lhs <= {WIDTH{1'b0}};
            stage0_rhs <= {WIDTH{1'b0}};
            stage1_fmt <= 3'b0;
            stage1_product <= {(2*WIDTH){1'b0}};
            result <= {WIDTH{1'b0}};
        end else begin
            stage0_fmt <= fmt;
            stage0_lhs <= sign_extend_int(lhs, fmt);
            stage0_rhs <= sign_extend_int(rhs, fmt);
            stage1_fmt <= stage0_fmt;
            stage1_product <= stage0_lhs * stage0_rhs;
            result <= narrow_int_result(stage1_product, stage1_fmt);
        end
    end

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
