`timescale 1ns/1ps

module warp #(
    parameter WIDTH = 32,
    parameter WARP_SIZE = 4
) (
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         instr_valid,
    input  wire [31:0]                  instr,
    input  wire [WARP_SIZE-1:0]         active_mask,
    output wire [WARP_SIZE-1:0]         supported_mask,
    output wire [WARP_SIZE-1:0]         divide_by_zero_mask,
    output wire [WARP_SIZE-1:0]         busy_mask,
    output wire [WARP_SIZE-1:0]         done_mask,
    output wire [WARP_SIZE-1:0]         writeback_enable_mask,
    output wire [(WARP_SIZE*4)-1:0]     writeback_addr,
    output wire [(WARP_SIZE*WIDTH)-1:0] writeback_data,
    output wire                         warp_busy,
    output wire                         warp_done
);
    genvar lane;
    generate
        for (lane = 0; lane < WARP_SIZE; lane = lane + 1) begin : lanes
            thread #(.WIDTH(WIDTH)) thread_lane (
                .clk(clk),
                .rst(rst),
                .instr_valid(instr_valid && active_mask[lane]),
                .instr(instr),
                .supported(supported_mask[lane]),
                .divide_by_zero(divide_by_zero_mask[lane]),
                .execute_busy(busy_mask[lane]),
                .execute_done(done_mask[lane]),
                .writeback_enable(writeback_enable_mask[lane]),
                .writeback_addr(writeback_addr[(lane*4) +: 4]),
                .writeback_data(writeback_data[(lane*WIDTH) +: WIDTH])
            );
        end
    endgenerate

    assign warp_busy = |(busy_mask & active_mask);
    assign warp_done = |active_mask && ((done_mask & active_mask) == active_mask);
endmodule
