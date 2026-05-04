`timescale 1ns/1ps

`include "minigpu_isa.vh"

module execute_tb;
    reg clk;
    reg rst;
    reg start;
    reg [5:0] opcode;
    reg [31:0] lhs;
    reg [31:0] rhs;
    reg [13:0] imm14;
    wire [31:0] result;
    wire supported;
    wire divide_by_zero;
    wire busy;
    wire done;

    execute #(
        .MUL_LATENCY(4),
        .DIV_LATENCY(4),
        .FLOAT_LATENCY(32)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .opcode(opcode),
        .lhs(lhs),
        .rhs(rhs),
        .imm14(imm14),
        .thread_id(32'd0),
        .lane_id(32'd0),
        .warp_id(32'd0),
        .block_id(32'd0),
        .block_dim(32'd4),
        .grid_dim(32'd1),
        .const_data(32'h0000cafe),
        .result(result),
        .supported(supported),
        .divide_by_zero(divide_by_zero),
        .busy(busy),
        .done(done)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task issue_and_wait;
        input [5:0] op;
        input [31:0] a;
        input [31:0] b;
        input [13:0] imm;
        begin
            @(negedge clk);
            opcode = op;
            lhs = a;
            rhs = b;
            imm14 = imm;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            while (!done) begin
                @(negedge clk);
            end
            #1;
            if (!supported) begin
                $display("FAIL unsupported op=0x%02h", op);
                $finish;
            end
        end
    endtask

    task check_result;
        input [5:0] op;
        input [31:0] a;
        input [31:0] b;
        input [13:0] imm;
        input [31:0] expected;
        begin
            issue_and_wait(op, a, b, imm);
            $display("EXE op=0x%02h lhs=0x%08h rhs=0x%08h imm=0x%04h -> result=0x%08h expected=0x%08h div0=%0d",
                     op, a, b, imm, result, expected, divide_by_zero);
            if (result !== expected) begin
                $display("FAIL op=0x%02h got=0x%08h expected=0x%08h", op, result, expected);
                $finish;
            end
        end
    endtask

    initial begin
        rst = 1'b1;
        start = 1'b0;
        opcode = 6'b0;
        lhs = 32'b0;
        rhs = 32'b0;
        imm14 = 14'b0;

        repeat (2) @(posedge clk);
        rst = 1'b0;

        check_result(`MGPU_OP_ADD, 32'd40, 32'd2, `MGPU_FMT_I32, 32'd42);
        check_result(`MGPU_OP_LDC, 32'd0, 32'd0, 14'd7, 32'h0000cafe);
        check_result(`MGPU_OP_ADD, 32'd127, 32'd1, `MGPU_FMT_I8, 32'hffffff80);
        check_result(`MGPU_OP_MUL, 32'd7, 32'd6, `MGPU_FMT_I32, 32'd42);
        check_result(`MGPU_OP_DIV, 32'd84, 32'd2, `MGPU_FMT_I32, 32'd42);
        check_result(`MGPU_OP_DIV, 32'hfffffff7, 32'd3, `MGPU_FMT_I32, 32'hfffffffd);
        check_result(`MGPU_OP_MOD, 32'hfffffff6, 32'd3, `MGPU_FMT_I32, 32'hffffffff);
        check_result(`MGPU_OP_DIV, 32'hfff7, 32'd3, `MGPU_FMT_I16, 32'hfffffffd);
        check_result(`MGPU_OP_MOD, 32'h00f6, 32'd3, `MGPU_FMT_I8, 32'hffffffff);

        issue_and_wait(`MGPU_OP_DIV, 32'd1, 32'd0, `MGPU_FMT_I32);
        $display("EXE op=DIV lhs=1 rhs=0 -> result=0x%08h div0=%0d expected result=0 div0=1",
                 result, divide_by_zero);
        if (!divide_by_zero || result !== 32'd0) begin
            $display("FAIL divide by zero result=0x%08h div0=%0d", result, divide_by_zero);
            $finish;
        end

        check_result(`MGPU_OP_FADD, 32'h3f800000, 32'h40000000, `MGPU_FMT_FP32, 32'h40400000);

        $display("execute_tb PASS");
        $finish;
    end
endmodule
