`timescale 1ns/1ps

module block #(
    parameter WIDTH = 32,
    parameter WARP_SIZE = 4,
    parameter NUM_WARPS = 4,
    parameter WARP_ID_WIDTH = 2,
    parameter ADDR_WIDTH = 16,
    parameter ENABLE_FLOAT_ADD = 1,
    parameter ENABLE_FLOAT_MUL = 1,
    parameter ENABLE_FLOAT_DIV = 1,
    parameter FLOAT_FP32_ONLY = 0,
    parameter USE_SHARED_FLOAT = 0,
    parameter SHARED_FLOAT_UNITS = 1
) (
    input  wire                                    clk,
    input  wire                                    rst,
    input  wire                                    instr_valid,
    input  wire [WARP_ID_WIDTH-1:0]                issue_warp_id,
    input  wire [31:0]                             instr,
    input  wire [(NUM_WARPS*WARP_SIZE)-1:0]        active_masks,
    input  wire [WIDTH-1:0]                        block_id,
    input  wire [WIDTH-1:0]                        block_dim,
    input  wire [WIDTH-1:0]                        grid_dim,
    input  wire [WIDTH-1:0]                        const_data,
    output wire [(NUM_WARPS*WARP_SIZE)-1:0]        mem_req_valid,
    output wire [(NUM_WARPS*WARP_SIZE)-1:0]        mem_req_write,
    output wire [(NUM_WARPS*WARP_SIZE*ADDR_WIDTH)-1:0] mem_req_addr,
    output wire [(NUM_WARPS*WARP_SIZE*WIDTH)-1:0]  mem_req_wdata,
    input  wire [(NUM_WARPS*WARP_SIZE)-1:0]        mem_req_ready,
    input  wire [(NUM_WARPS*WARP_SIZE)-1:0]        mem_resp_valid,
    input  wire [(NUM_WARPS*WARP_SIZE*WIDTH)-1:0]  mem_resp_rdata,
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
            wire [WIDTH-1:0] warp_id_value = warp_id;
            warp #(
                .WIDTH(WIDTH),
                .WARP_SIZE(WARP_SIZE),
                .ADDR_WIDTH(ADDR_WIDTH),
                .ENABLE_FLOAT_ADD(ENABLE_FLOAT_ADD),
                .ENABLE_FLOAT_MUL(ENABLE_FLOAT_MUL),
                .ENABLE_FLOAT_DIV(ENABLE_FLOAT_DIV),
                .FLOAT_FP32_ONLY(FLOAT_FP32_ONLY),
                .USE_SHARED_FLOAT(USE_SHARED_FLOAT),
                .SHARED_FLOAT_UNITS(SHARED_FLOAT_UNITS)
            ) warp_unit (
                .clk(clk),
                .rst(rst),
                .instr_valid(instr_valid && (issue_warp_id == warp_id[WARP_ID_WIDTH-1:0])),
                .instr(instr),
                .active_mask(active_masks[(warp_id*WARP_SIZE) +: WARP_SIZE]),
                .warp_id(warp_id_value),
                .block_id(block_id),
                .block_dim(block_dim),
                .grid_dim(grid_dim),
                .const_data(const_data),
                .mem_req_valid(mem_req_valid[(warp_id*WARP_SIZE) +: WARP_SIZE]),
                .mem_req_write(mem_req_write[(warp_id*WARP_SIZE) +: WARP_SIZE]),
                .mem_req_addr(mem_req_addr[(warp_id*WARP_SIZE*ADDR_WIDTH) +: (WARP_SIZE*ADDR_WIDTH)]),
                .mem_req_wdata(mem_req_wdata[(warp_id*WARP_SIZE*WIDTH) +: (WARP_SIZE*WIDTH)]),
                .mem_req_ready(mem_req_ready[(warp_id*WARP_SIZE) +: WARP_SIZE]),
                .mem_resp_valid(mem_resp_valid[(warp_id*WARP_SIZE) +: WARP_SIZE]),
                .mem_resp_rdata(mem_resp_rdata[(warp_id*WARP_SIZE*WIDTH) +: (WARP_SIZE*WIDTH)]),
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
