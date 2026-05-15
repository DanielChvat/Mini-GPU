`timescale 1ns/1ps

`include "minigpu_isa.vh"

module shared_fpu_pipeline_tb;
    reg clk = 1'b0;
    reg rst = 1'b1;
    reg start = 1'b0;
    reg [5:0] opcode = `MGPU_OP_NOP;
    reg [2:0] fmt = `MGPU_FMT_FP32;
    reg [31:0] lhs = 32'b0;
    reg [31:0] rhs = 32'b0;
    reg [3:0] tag_in = 4'b0;

    wire [31:0] result;
    wire supported;
    wire divide_by_zero;
    wire [3:0] tag_out;
    wire busy;
    wire done;

    integer done_count = 0;
    integer cycle = 0;

    shared_fpu #(
        .ENABLE_FLOAT_ADD(1),
        .ENABLE_FLOAT_MUL(1),
        .ENABLE_FLOAT_DIV(0),
        .FLOAT_FP32_ONLY(0),
        .LATENCY(32),
        .TAG_WIDTH(4)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .opcode(opcode),
        .fmt(fmt),
        .lhs(lhs),
        .rhs(rhs),
        .tag_in(tag_in),
        .result(result),
        .supported(supported),
        .divide_by_zero(divide_by_zero),
        .tag_out(tag_out),
        .busy(busy),
        .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        repeat (4) @(posedge clk);
        rst <= 1'b0;

        issue(`MGPU_OP_FADD, 32'h3f800000, 32'h40000000, 4'b0001);
        issue(`MGPU_OP_FMUL, 32'h40400000, 32'h40800000, 4'b0010);
        start <= 1'b0;

        repeat (80) @(posedge clk);
        if (done_count != 2) begin
            $display("shared_fpu_pipeline_tb FAIL done_count=%0d", done_count);
            $finish(1);
        end
        $display("shared_fpu_pipeline_tb PASS");
        $finish;
    end

    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (done) begin
            done_count <= done_count + 1;
            if (!supported || divide_by_zero) begin
                $display("shared_fpu_pipeline_tb FAIL bad status tag=%b result=%h supported=%b div0=%b",
                         tag_out, result, supported, divide_by_zero);
                $finish(1);
            end
            if (tag_out == 4'b0001 && result != 32'h40400000) begin
                $display("shared_fpu_pipeline_tb FAIL add result=%h", result);
                $finish(1);
            end
            if (tag_out == 4'b0010 && result != 32'h41400000) begin
                $display("shared_fpu_pipeline_tb FAIL mul result=%h", result);
                $finish(1);
            end
        end
    end

    task issue;
        input [5:0] op;
        input [31:0] a;
        input [31:0] b;
        input [3:0] tag;
        begin
            @(negedge clk);
            start <= 1'b1;
            opcode <= op;
            lhs <= a;
            rhs <= b;
            tag_in <= tag;
            @(posedge clk);
        end
    endtask
endmodule
