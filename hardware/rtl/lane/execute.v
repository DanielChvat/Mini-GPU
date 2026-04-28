`timescale 1ns/1ps

`include "minigpu_isa.vh"

module execute #(
    parameter WIDTH = 32,
    parameter MUL_LATENCY = 2,
    parameter DIV_LATENCY = 16,
    parameter FLOAT_LATENCY = 4
) (
    input  wire             clk,
    input  wire             rst,
    input  wire             start,
    input  wire [5:0]       opcode,
    input  wire [WIDTH-1:0] lhs,
    input  wire [WIDTH-1:0] rhs,
    input  wire [13:0]      imm14,
    output reg  [WIDTH-1:0] result,
    output reg              supported,
    output reg              divide_by_zero,
    output reg              busy,
    output reg              done
);
    localparam CLASS_FAST  = 2'd0;
    localparam CLASS_MUL   = 2'd1;
    localparam CLASS_DIV   = 2'd2;
    localparam CLASS_FLOAT = 2'd3;

    reg [5:0] opcode_r;
    reg [WIDTH-1:0] lhs_r;
    reg [WIDTH-1:0] rhs_r;
    reg [13:0] imm14_r;
    reg [1:0] op_class;
    reg [7:0] cycles_left;

    wire [2:0] fmt = imm14_r[2:0];
    wire signed [WIDTH-1:0] imm_signed = {{(WIDTH-14){imm14_r[13]}}, imm14_r};
    wire [4:0] rhs_shift = rhs_r[4:0];
    wire [4:0] imm_shift = imm_signed[4:0];

    wire [WIDTH-1:0] mul_result;
    reg div_start;
    wire [WIDTH-1:0] div_iter_result;
    wire div_iter_zero;
    wire div_iter_busy;
    wire div_iter_done;
    wire [31:0] float_add_result;
    wire [31:0] float_sub_result;
    wire [31:0] float_mul_result;
    wire [31:0] float_div_result;
    wire float_add_supported;
    wire float_sub_supported;
    wire float_mul_supported;
    wire float_div_supported;
    wire float_div_zero;

    int_mul #(.WIDTH(WIDTH)) mul_unit (
        .fmt(fmt),
        .lhs(lhs_r),
        .rhs(rhs_r),
        .result(mul_result)
    );

    int_div_mod_iterative #(.WIDTH(WIDTH)) div_unit (
        .clk(clk),
        .rst(rst),
        .start(div_start),
        .fmt(fmt),
        .modulo(opcode_r == `MGPU_OP_MOD),
        .lhs(lhs_r),
        .rhs(rhs_r),
        .result(div_iter_result),
        .divide_by_zero(div_iter_zero),
        .busy(div_iter_busy),
        .done(div_iter_done)
    );

    float_add_sub float_add_unit (
        .fmt(fmt),
        .subtract(1'b0),
        .lhs(lhs_r),
        .rhs(rhs_r),
        .result(float_add_result),
        .supported(float_add_supported)
    );

    float_add_sub float_sub_unit (
        .fmt(fmt),
        .subtract(1'b1),
        .lhs(lhs_r),
        .rhs(rhs_r),
        .result(float_sub_result),
        .supported(float_sub_supported)
    );

    float_mul float_mul_unit (
        .fmt(fmt),
        .lhs(lhs_r),
        .rhs(rhs_r),
        .result(float_mul_result),
        .supported(float_mul_supported)
    );

    float_div float_div_unit (
        .fmt(fmt),
        .lhs(lhs_r),
        .rhs(rhs_r),
        .result(float_div_result),
        .supported(float_div_supported),
        .divide_by_zero(float_div_zero)
    );

    always @(posedge clk) begin
        if (rst) begin
            result <= {WIDTH{1'b0}};
            supported <= 1'b0;
            divide_by_zero <= 1'b0;
            div_start <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
            opcode_r <= 6'b0;
            lhs_r <= {WIDTH{1'b0}};
            rhs_r <= {WIDTH{1'b0}};
            imm14_r <= 14'b0;
            op_class <= CLASS_FAST;
            cycles_left <= 8'b0;
        end else begin
            done <= 1'b0;
            div_start <= 1'b0;

            if (start && !busy) begin
                opcode_r <= opcode;
                lhs_r <= lhs;
                rhs_r <= rhs;
                imm14_r <= imm14;
                op_class <= opcode_class(opcode);
                cycles_left <= latency_for(opcode);
                busy <= 1'b1;
                if (opcode_class(opcode) == CLASS_DIV) begin
                    div_start <= 1'b1;
                end
            end else if (busy) begin
                if (op_class == CLASS_DIV) begin
                    if (div_iter_done) begin
                        result <= div_iter_result;
                        supported <= 1'b1;
                        divide_by_zero <= div_iter_zero;
                        busy <= 1'b0;
                        done <= 1'b1;
                    end
                end else if (cycles_left <= 8'd1) begin
                    finish_operation();
                    busy <= 1'b0;
                    done <= 1'b1;
                end else begin
                    cycles_left <= cycles_left - 8'd1;
                end
            end
        end
    end

    task finish_operation;
        begin
            result <= {WIDTH{1'b0}};
            supported <= 1'b1;
            divide_by_zero <= 1'b0;

            if (is_typed_int_opcode(opcode_r) && !is_integer_format(fmt)) begin
                supported <= 1'b0;
            end else begin
                case (op_class)
                    CLASS_MUL: begin
                        result <= mul_result;
                    end

                    CLASS_FLOAT: begin
                        case (opcode_r)
                            `MGPU_OP_FADD: begin
                                result <= float_add_result;
                                supported <= float_add_supported;
                            end
                            `MGPU_OP_FSUB: begin
                                result <= float_sub_result;
                                supported <= float_sub_supported;
                            end
                            `MGPU_OP_FMUL: begin
                                result <= float_mul_result;
                                supported <= float_mul_supported;
                            end
                            `MGPU_OP_FDIV: begin
                                result <= float_div_result;
                                supported <= float_div_supported;
                                divide_by_zero <= float_div_zero;
                            end
                            default: supported <= 1'b0;
                        endcase
                    end

                    default: begin
                        finish_fast_operation();
                    end
                endcase
            end
        end
    endtask

    task finish_fast_operation;
        begin
            case (opcode_r)
                `MGPU_OP_NOP:  result <= {WIDTH{1'b0}};
                `MGPU_OP_MOV:  result <= lhs_r;
                `MGPU_OP_MOVI: result <= imm_signed;
                `MGPU_OP_ADD:  result <= narrow_int_result(sign_extend_int(lhs_r, fmt) + sign_extend_int(rhs_r, fmt), fmt);
                `MGPU_OP_ADDI: result <= lhs_r + imm_signed;
                `MGPU_OP_SUB:  result <= narrow_int_result(sign_extend_int(lhs_r, fmt) - sign_extend_int(rhs_r, fmt), fmt);
                `MGPU_OP_SUBI: result <= lhs_r - imm_signed;
                `MGPU_OP_AND:  result <= narrow_int_result(sign_extend_int(lhs_r, fmt) & sign_extend_int(rhs_r, fmt), fmt);
                `MGPU_OP_ANDI: result <= lhs_r & imm_signed;
                `MGPU_OP_OR:   result <= narrow_int_result(sign_extend_int(lhs_r, fmt) | sign_extend_int(rhs_r, fmt), fmt);
                `MGPU_OP_ORI:  result <= lhs_r | imm_signed;
                `MGPU_OP_XOR:  result <= narrow_int_result(sign_extend_int(lhs_r, fmt) ^ sign_extend_int(rhs_r, fmt), fmt);
                `MGPU_OP_XORI: result <= lhs_r ^ imm_signed;
                `MGPU_OP_NOT:  result <= ~lhs_r;
                `MGPU_OP_SHL:  result <= narrow_int_result(sign_extend_int(lhs_r, fmt) << rhs_shift, fmt);
                `MGPU_OP_SHLI: result <= lhs_r << imm_shift;
                `MGPU_OP_SHR:  result <= narrow_int_result(sign_extend_int(lhs_r, fmt) >>> rhs_shift, fmt);
                `MGPU_OP_SHRI: result <= lhs_r >> imm_shift;
                `MGPU_OP_SLT:  result <= signed_compare_result(3'd0, lhs_r, rhs_r, fmt);
                `MGPU_OP_SLE:  result <= signed_compare_result(3'd1, lhs_r, rhs_r, fmt);
                `MGPU_OP_SGT:  result <= signed_compare_result(3'd2, lhs_r, rhs_r, fmt);
                `MGPU_OP_SGE:  result <= signed_compare_result(3'd3, lhs_r, rhs_r, fmt);
                `MGPU_OP_SEQ:  result <= (sign_extend_int(lhs_r, fmt) == sign_extend_int(rhs_r, fmt)) ? {{(WIDTH-1){1'b0}}, 1'b1} : {WIDTH{1'b0}};
                `MGPU_OP_SNE:  result <= (sign_extend_int(lhs_r, fmt) != sign_extend_int(rhs_r, fmt)) ? {{(WIDTH-1){1'b0}}, 1'b1} : {WIDTH{1'b0}};
                default: begin
                    supported <= 1'b0;
                    result <= {WIDTH{1'b0}};
                end
            endcase
        end
    endtask

    function [1:0] opcode_class;
        input [5:0] value_opcode;
        begin
            case (value_opcode)
                `MGPU_OP_MUL,
                `MGPU_OP_MULI: opcode_class = CLASS_MUL;
                `MGPU_OP_DIV,
                `MGPU_OP_MOD: opcode_class = CLASS_DIV;
                `MGPU_OP_FADD,
                `MGPU_OP_FSUB,
                `MGPU_OP_FMUL,
                `MGPU_OP_FDIV: opcode_class = CLASS_FLOAT;
                default: opcode_class = CLASS_FAST;
            endcase
        end
    endfunction

    function [7:0] latency_for;
        input [5:0] value_opcode;
        begin
            case (opcode_class(value_opcode))
                CLASS_MUL: latency_for = MUL_LATENCY[7:0];
                CLASS_DIV: latency_for = 8'd1;
                CLASS_FLOAT: latency_for = FLOAT_LATENCY[7:0];
                default: latency_for = 8'd1;
            endcase
        end
    endfunction

    function is_integer_format;
        input [2:0] value_fmt;
        begin
            is_integer_format = (value_fmt == `MGPU_FMT_I32) ||
                                (value_fmt == `MGPU_FMT_I16) ||
                                (value_fmt == `MGPU_FMT_I8);
        end
    endfunction

    function is_typed_int_opcode;
        input [5:0] value_opcode;
        begin
            case (value_opcode)
                `MGPU_OP_ADD,
                `MGPU_OP_SUB,
                `MGPU_OP_MUL,
                `MGPU_OP_DIV,
                `MGPU_OP_MOD,
                `MGPU_OP_AND,
                `MGPU_OP_OR,
                `MGPU_OP_XOR,
                `MGPU_OP_SHL,
                `MGPU_OP_SHR,
                `MGPU_OP_SLT,
                `MGPU_OP_SLE,
                `MGPU_OP_SGT,
                `MGPU_OP_SGE,
                `MGPU_OP_SEQ,
                `MGPU_OP_SNE: is_typed_int_opcode = 1'b1;
                default: is_typed_int_opcode = 1'b0;
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

    function [WIDTH-1:0] narrow_int_result;
        input signed [WIDTH-1:0] value;
        input [2:0] value_fmt;
        begin
            case (value_fmt)
                `MGPU_FMT_I8:  narrow_int_result = {{(WIDTH-8){value[7]}}, value[7:0]};
                `MGPU_FMT_I16: narrow_int_result = {{(WIDTH-16){value[15]}}, value[15:0]};
                default:       narrow_int_result = value;
            endcase
        end
    endfunction

    function [WIDTH-1:0] signed_compare_result;
        input [2:0] compare_op;
        input [WIDTH-1:0] a;
        input [WIDTH-1:0] b;
        input [2:0] value_fmt;
        reg signed [WIDTH-1:0] a_ext;
        reg signed [WIDTH-1:0] b_ext;
        begin
            a_ext = sign_extend_int(a, value_fmt);
            b_ext = sign_extend_int(b, value_fmt);
            case (compare_op)
                3'd0: signed_compare_result = (a_ext <  b_ext) ? {{(WIDTH-1){1'b0}}, 1'b1} : {WIDTH{1'b0}};
                3'd1: signed_compare_result = (a_ext <= b_ext) ? {{(WIDTH-1){1'b0}}, 1'b1} : {WIDTH{1'b0}};
                3'd2: signed_compare_result = (a_ext >  b_ext) ? {{(WIDTH-1){1'b0}}, 1'b1} : {WIDTH{1'b0}};
                3'd3: signed_compare_result = (a_ext >= b_ext) ? {{(WIDTH-1){1'b0}}, 1'b1} : {WIDTH{1'b0}};
                default: signed_compare_result = {WIDTH{1'b0}};
            endcase
        end
    endfunction
endmodule
