`timescale 1ns/1ps

module block #(
    parameter WIDTH = 32,
    parameter WARP_SIZE = 4,
    parameter NUM_WARPS = 4,
    parameter WARP_ID_WIDTH = 2
) (
    input  wire                                    clk,
    input  wire                                    rst,
    input  wire                                    instr_valid,
    input  wire [WARP_ID_WIDTH-1:0]                issue_warp_id,
    input  wire [31:0]                             instr,
    input  wire [(NUM_WARPS*WARP_SIZE)-1:0]        active_masks,
    output wire [(NUM_WARPS*WARP_SIZE)-1:0]        supported_mask,
    output wire [(NUM_WARPS*WARP_SIZE)-1:0]        divide_by_zero_mask,
    output wire [(NUM_WARPS*WARP_SIZE)-1:0]        busy_mask,
    output wire [(NUM_WARPS*WARP_SIZE)-1:0]        done_mask,
    output wire [(NUM_WARPS*WARP_SIZE)-1:0]        writeback_enable_mask,
    output wire [(NUM_WARPS*WARP_SIZE*4)-1:0]      writeback_addr,
    output wire [(NUM_WARPS*WARP_SIZE*WIDTH)-1:0]  writeback_data,
    output wire [NUM_WARPS-1:0]                    warp_busy,
    output wire [NUM_WARPS-1:0]                    warp_done,
    output wire                                    block_busy,
    output wire                                    block_done
);
    genvar warp_id;
    generate
        for (warp_id = 0; warp_id < NUM_WARPS; warp_id = warp_id + 1) begin : warps
            warp #(
                .WIDTH(WIDTH),
                .WARP_SIZE(WARP_SIZE)
            ) warp_unit (
                .clk(clk),
                .rst(rst),
                .instr_valid(instr_valid && (issue_warp_id == warp_id[WARP_ID_WIDTH-1:0])),
                .instr(instr),
                .active_mask(active_masks[(warp_id*WARP_SIZE) +: WARP_SIZE]),
                .supported_mask(supported_mask[(warp_id*WARP_SIZE) +: WARP_SIZE]),
                .divide_by_zero_mask(divide_by_zero_mask[(warp_id*WARP_SIZE) +: WARP_SIZE]),
                .busy_mask(busy_mask[(warp_id*WARP_SIZE) +: WARP_SIZE]),
                .done_mask(done_mask[(warp_id*WARP_SIZE) +: WARP_SIZE]),
                .writeback_enable_mask(writeback_enable_mask[(warp_id*WARP_SIZE) +: WARP_SIZE]),
                .writeback_addr(writeback_addr[(warp_id*WARP_SIZE*4) +: (WARP_SIZE*4)]),
                .writeback_data(writeback_data[(warp_id*WARP_SIZE*WIDTH) +: (WARP_SIZE*WIDTH)]),
                .warp_busy(warp_busy[warp_id]),
                .warp_done(warp_done[warp_id])
            );
        end
    endgenerate

    assign block_busy = |warp_busy;
    assign block_done = &warp_done;
endmodule
