`timescale 1ns/1ps

`include "minigpu_isa.vh"

module thread_tb;
    reg clk;
    reg rst;
    reg instr_valid;
    reg [31:0] instr;
    wire supported;
    wire divide_by_zero;
    wire execute_busy;
    wire execute_done;
    wire writeback_enable;
    wire [3:0] writeback_addr;
    wire [31:0] writeback_data;

    thread dut (
        .clk(clk),
        .rst(rst),
        .instr_valid(instr_valid),
        .instr(instr),
        .supported(supported),
        .divide_by_zero(divide_by_zero),
        .execute_busy(execute_busy),
        .execute_done(execute_done),
        .writeback_enable(writeback_enable),
        .writeback_addr(writeback_addr),
        .writeback_data(writeback_data)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task issue_expect_writeback;
        input [31:0] value;
        input [3:0] expected_addr;
        input [31:0] expected_data;
        begin
            @(negedge clk);
            instr = value;
            instr_valid = 1'b1;
            @(negedge clk);
            instr_valid = 1'b0;
            while (!execute_done) begin
                @(negedge clk);
            end
            #1;
            if (!supported) begin
                $display("FAIL unsupported instruction 0x%08h", value);
                $finish;
            end
            $display("THREAD instr=0x%08h -> wb_en=%0d wb_addr=%0d wb_data=0x%08h expected addr=%0d data=0x%08h",
                     value,
                     writeback_enable,
                     writeback_addr,
                     writeback_data,
                     expected_addr,
                     expected_data);
            if (!writeback_enable || writeback_addr !== expected_addr || writeback_data !== expected_data) begin
                $display("FAIL writeback got en=%0d addr=%0d data=0x%08h expected addr=%0d data=0x%08h",
                         writeback_enable, writeback_addr, writeback_data, expected_addr, expected_data);
                $finish;
            end
            @(posedge clk);
            #1;
        end
    endtask

    task issue_unsupported;
        input [31:0] value;
        begin
            @(negedge clk);
            instr = value;
            instr_valid = 1'b1;
            @(negedge clk);
            instr_valid = 1'b0;
            while (!execute_done) begin
                @(negedge clk);
            end
            #1;
            $display("THREAD unsupported instr=0x%08h -> supported=%0d wb_en=%0d",
                     value, supported, writeback_enable);
            if (supported || writeback_enable) begin
                $display("FAIL instruction should be unsupported/no-writeback 0x%08h", value);
                $finish;
            end
        end
    endtask

    function [31:0] pack_instr;
        input [5:0] op;
        input [3:0] rd;
        input [3:0] rs1;
        input [3:0] rs2;
        input [13:0] imm14;
        begin
            pack_instr = {op, rd, rs1, rs2, imm14};
        end
    endfunction

    initial begin
        rst = 1'b1;
        instr_valid = 1'b0;
        instr = 32'b0;
        repeat (2) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);

        issue_expect_writeback(pack_instr(`MGPU_OP_MOVI, 4'd1, 4'd0, 4'd0, 14'd40), 4'd1, 32'd40);

        issue_expect_writeback(pack_instr(`MGPU_OP_MOVI, 4'd2, 4'd0, 4'd0, 14'd2), 4'd2, 32'd2);

        issue_expect_writeback(pack_instr(`MGPU_OP_ADD, 4'd3, 4'd1, 4'd2, 14'd0), 4'd3, 32'd42);

        issue_expect_writeback(pack_instr(`MGPU_OP_MUL, 4'd4, 4'd3, 4'd2, 14'd0), 4'd4, 32'd84);

        issue_expect_writeback(pack_instr(`MGPU_OP_ADD, 4'd5, 4'd1, 4'd2, {11'b0, `MGPU_FMT_I8}), 4'd5, 32'd42);

        issue_expect_writeback(pack_instr(`MGPU_OP_MOVI, 4'd6, 4'd0, 4'd0, 14'h3fff), 4'd6, 32'hffffffff);

        issue_expect_writeback(pack_instr(`MGPU_OP_SLT, 4'd7, 4'd6, 4'd2, 14'd0), 4'd7, 32'd1);

        issue_expect_writeback(pack_instr(`MGPU_OP_MOVI, 4'd8, 4'd0, 4'd0, 14'h0038), 4'd8, 32'h00000038);

        issue_expect_writeback(pack_instr(`MGPU_OP_MOVI, 4'd9, 4'd0, 4'd0, 14'h0040), 4'd9, 32'h00000040);

        issue_expect_writeback(pack_instr(`MGPU_OP_FADD, 4'd10, 4'd8, 4'd9, {11'b0, `MGPU_FMT_FP8}), 4'd10, 32'h00000044);

        issue_unsupported(pack_instr(`MGPU_OP_LDG, 4'd11, 4'd0, 4'd0, 14'd0));

        $display("thread_tb PASS");
        $finish;
    end
endmodule
