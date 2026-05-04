`timescale 1ns/1ps

`include "minigpu_isa.vh"

module sm_tb;
    localparam WIDTH = 32;
    localparam WARP_SIZE = 4;
    localparam NUM_WARPS = 4;
    localparam NUM_BLOCKS = 1;
    localparam WARP_ID_WIDTH = 2;
    localparam BLOCK_ID_WIDTH = 1;

    reg clk;
    reg rst;
    reg instr_valid;
    reg [BLOCK_ID_WIDTH-1:0] issue_block_id;
    reg [WARP_ID_WIDTH-1:0] issue_warp_id;
    reg [31:0] instr;
    reg [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0] active_masks;
    wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0] supported_mask;
    wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0] divide_by_zero_mask;
    wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0] busy_mask;
    wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0] done_mask;
    wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0] writeback_enable_mask;
    wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE*4)-1:0] writeback_addr;
    wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE*WIDTH)-1:0] writeback_data;
    wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0] mem_req_valid;
    wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0] mem_req_write;
    wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE*16)-1:0] mem_req_addr;
    wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE*WIDTH)-1:0] mem_req_wdata;
    wire [NUM_BLOCKS-1:0] block_busy;
    wire [NUM_BLOCKS-1:0] block_done;
    wire sm_busy;
    wire sm_done;

    sm #(
        .WIDTH(WIDTH),
        .WARP_SIZE(WARP_SIZE),
        .NUM_WARPS(NUM_WARPS),
        .NUM_BLOCKS(NUM_BLOCKS),
        .WARP_ID_WIDTH(WARP_ID_WIDTH),
        .BLOCK_ID_WIDTH(BLOCK_ID_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .instr_valid(instr_valid),
        .issue_block_id(issue_block_id),
        .issue_warp_id(issue_warp_id),
        .instr(instr),
        .active_masks(active_masks),
        .base_block_id(32'd0),
        .block_dim(32'd16),
        .grid_dim(32'd1),
        .const_data(32'h0000cafe),
        .mem_req_valid(mem_req_valid),
        .mem_req_write(mem_req_write),
        .mem_req_addr(mem_req_addr),
        .mem_req_wdata(mem_req_wdata),
        .mem_req_ready({(NUM_BLOCKS*NUM_WARPS*WARP_SIZE){1'b0}}),
        .mem_resp_valid({(NUM_BLOCKS*NUM_WARPS*WARP_SIZE){1'b0}}),
        .mem_resp_rdata({(NUM_BLOCKS*NUM_WARPS*WARP_SIZE*WIDTH){1'b0}}),
        .supported_mask(supported_mask),
        .divide_by_zero_mask(divide_by_zero_mask),
        .busy_mask(busy_mask),
        .done_mask(done_mask),
        .writeback_enable_mask(writeback_enable_mask),
        .writeback_addr(writeback_addr),
        .writeback_data(writeback_data),
        .block_busy(block_busy),
        .block_done(block_done),
        .sm_busy(sm_busy),
        .sm_done(sm_done)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

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

    task issue_to_warp;
        input [WARP_ID_WIDTH-1:0] warp_id;
        input [31:0] value;
        input [(WARP_SIZE)-1:0] expected_done;
        input [3:0] expected_wb_addr;
        input [WIDTH-1:0] expected_wb_data;
        integer base;
        integer lane;
        begin
            base = warp_id * WARP_SIZE;
            @(negedge clk);
            issue_block_id = 1'b0;
            issue_warp_id = warp_id;
            instr = value;
            instr_valid = 1'b1;
            @(negedge clk);
            instr_valid = 1'b0;
            while (done_mask[base +: WARP_SIZE] != expected_done) begin
                @(negedge clk);
            end
            #1;
            $display("SM issue: block=0 warp=%0d instr=0x%08h active=%b -> done=%b supported=%b wb_en=%b",
                     warp_id,
                     value,
                     active_masks[base +: WARP_SIZE],
                     done_mask[base +: WARP_SIZE],
                     supported_mask[base +: WARP_SIZE],
                     writeback_enable_mask[base +: WARP_SIZE]);
            if (supported_mask[base +: WARP_SIZE] != expected_done) begin
                $display("FAIL warp %0d supported=%b expected=%b",
                         warp_id, supported_mask[base +: WARP_SIZE], expected_done);
                $finish;
            end
            if (writeback_enable_mask[base +: WARP_SIZE] != expected_done) begin
                $display("FAIL warp %0d writeback enable=%b expected=%b",
                         warp_id, writeback_enable_mask[base +: WARP_SIZE], expected_done);
                $finish;
            end
            for (lane = 0; lane < WARP_SIZE; lane = lane + 1) begin
                if (expected_done[lane]) begin
                    $display("SM writeback: warp=%0d lane=%0d addr=%0d data=0x%08h expected addr=%0d data=0x%08h",
                             warp_id,
                             lane,
                             writeback_addr[((base + lane)*4) +: 4],
                             writeback_data[((base + lane)*WIDTH) +: WIDTH],
                             expected_wb_addr,
                             expected_wb_data);
                    if (writeback_addr[((base + lane)*4) +: 4] !== expected_wb_addr ||
                        writeback_data[((base + lane)*WIDTH) +: WIDTH] !== expected_wb_data) begin
                        $display("FAIL warp=%0d lane=%0d wb addr=%0d data=0x%08h expected addr=%0d data=0x%08h",
                                 warp_id,
                                 lane,
                                 writeback_addr[((base + lane)*4) +: 4],
                                 writeback_data[((base + lane)*WIDTH) +: WIDTH],
                                 expected_wb_addr,
                                 expected_wb_data);
                        $finish;
                    end
                end
            end
            @(posedge clk);
        end
    endtask

    initial begin
        rst = 1'b1;
        instr_valid = 1'b0;
        issue_block_id = 1'b0;
        issue_warp_id = 2'b0;
        instr = 32'b0;
        active_masks = {NUM_BLOCKS*NUM_WARPS*WARP_SIZE{1'b1}};

        repeat (2) @(posedge clk);
        rst = 1'b0;

        issue_to_warp(2'd0, pack_instr(`MGPU_OP_MOVI, 4'd1, 4'd0, 4'd0, 14'd7), 4'b1111, 4'd1, 32'd7);
        issue_to_warp(2'd0, pack_instr(`MGPU_OP_ADD, 4'd2, 4'd1, 4'd1, 14'd0), 4'b1111, 4'd2, 32'd14);

        issue_to_warp(2'd1, pack_instr(`MGPU_OP_MOVI, 4'd1, 4'd0, 4'd0, 14'd11), 4'b1111, 4'd1, 32'd11);
        issue_to_warp(2'd1, pack_instr(`MGPU_OP_ADD, 4'd2, 4'd1, 4'd1, 14'd0), 4'b1111, 4'd2, 32'd22);
        issue_to_warp(2'd0, pack_instr(`MGPU_OP_ADD, 4'd3, 4'd1, 4'd2, 14'd0), 4'b1111, 4'd3, 32'd21);

        $display("sm_tb PASS");
        $finish;
    end
endmodule
