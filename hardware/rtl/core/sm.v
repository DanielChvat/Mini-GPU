`timescale 1ns/1ps

module sm #(
    parameter WIDTH = 32,
    parameter WARP_SIZE = 4,
    parameter NUM_WARPS = 4,
    parameter NUM_BLOCKS = 1,
    parameter WARP_ID_WIDTH = 2,
    parameter BLOCK_ID_WIDTH = 1
) (
    input  wire                                             clk,
    input  wire                                             rst,
    input  wire                                             instr_valid,
    input  wire [BLOCK_ID_WIDTH-1:0]                         issue_block_id,
    input  wire [WARP_ID_WIDTH-1:0]                          issue_warp_id,
    input  wire [31:0]                                      instr,
    input  wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0]       active_masks,
    output wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0]       supported_mask,
    output wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0]       divide_by_zero_mask,
    output wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0]       busy_mask,
    output wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0]       done_mask,
    output wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE)-1:0]       writeback_enable_mask,
    output wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE*4)-1:0]     writeback_addr,
    output wire [(NUM_BLOCKS*NUM_WARPS*WARP_SIZE*WIDTH)-1:0] writeback_data,
    output wire [NUM_BLOCKS-1:0]                             block_busy,
    output wire [NUM_BLOCKS-1:0]                             block_done,
    output wire                                             sm_busy,
    output wire                                             sm_done
);
    localparam BLOCK_MASK_WIDTH = NUM_WARPS * WARP_SIZE;
    localparam BLOCK_WB_ADDR_WIDTH = NUM_WARPS * WARP_SIZE * 4;
    localparam BLOCK_WB_DATA_WIDTH = NUM_WARPS * WARP_SIZE * WIDTH;
    wire [(NUM_BLOCKS*NUM_WARPS)-1:0] warp_busy_unused;
    wire [(NUM_BLOCKS*NUM_WARPS)-1:0] warp_done_unused;

    genvar block_id;
    generate
        for (block_id = 0; block_id < NUM_BLOCKS; block_id = block_id + 1) begin : blocks
            block #(
                .WIDTH(WIDTH),
                .WARP_SIZE(WARP_SIZE),
                .NUM_WARPS(NUM_WARPS),
                .WARP_ID_WIDTH(WARP_ID_WIDTH)
            ) block_unit (
                .clk(clk),
                .rst(rst),
                .instr_valid(instr_valid && (issue_block_id == block_id[BLOCK_ID_WIDTH-1:0])),
                .issue_warp_id(issue_warp_id),
                .instr(instr),
                .active_masks(active_masks[(block_id*BLOCK_MASK_WIDTH) +: BLOCK_MASK_WIDTH]),
                .supported_mask(supported_mask[(block_id*BLOCK_MASK_WIDTH) +: BLOCK_MASK_WIDTH]),
                .divide_by_zero_mask(divide_by_zero_mask[(block_id*BLOCK_MASK_WIDTH) +: BLOCK_MASK_WIDTH]),
                .busy_mask(busy_mask[(block_id*BLOCK_MASK_WIDTH) +: BLOCK_MASK_WIDTH]),
                .done_mask(done_mask[(block_id*BLOCK_MASK_WIDTH) +: BLOCK_MASK_WIDTH]),
                .writeback_enable_mask(writeback_enable_mask[(block_id*BLOCK_MASK_WIDTH) +: BLOCK_MASK_WIDTH]),
                .writeback_addr(writeback_addr[(block_id*BLOCK_WB_ADDR_WIDTH) +: BLOCK_WB_ADDR_WIDTH]),
                .writeback_data(writeback_data[(block_id*BLOCK_WB_DATA_WIDTH) +: BLOCK_WB_DATA_WIDTH]),
                .warp_busy(warp_busy_unused[(block_id*NUM_WARPS) +: NUM_WARPS]),
                .warp_done(warp_done_unused[(block_id*NUM_WARPS) +: NUM_WARPS]),
                .block_busy(block_busy[block_id]),
                .block_done(block_done[block_id])
            );
        end
    endgenerate

    assign sm_busy = |block_busy;
    assign sm_done = &block_done;
endmodule
