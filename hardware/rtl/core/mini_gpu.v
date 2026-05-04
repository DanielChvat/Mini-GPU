`timescale 1ns/1ps

module mini_gpu #(
    parameter WIDTH = 32,
    parameter WARP_SIZE = 4,
    parameter NUM_CORES = 2,
    parameter NUM_WARPS_PER_CORE = 1,
    parameter WARP_ID_WIDTH = 1,
    parameter PROG_ADDR_WIDTH = 8,
    parameter ADDR_WIDTH = 16,
    parameter CONST_ADDR_WIDTH = 8,
    parameter MEMORY_BANK_DEPTH = 8192,
    parameter ENABLE_FLOAT_ADD = 1,
    parameter ENABLE_FLOAT_MUL = 1,
    parameter ENABLE_FLOAT_DIV = 1,
    parameter FLOAT_FP32_ONLY = 0,
    parameter USE_SHARED_FLOAT = 0,
    parameter SHARED_FLOAT_UNITS = 1
) (
    input  wire                         clk,
    input  wire                         rst,

    input  wire                         prog_we,
    input  wire [PROG_ADDR_WIDTH-1:0]   prog_addr,
    input  wire [31:0]                  prog_wdata,

    input  wire                         const_we,
    input  wire [CONST_ADDR_WIDTH-1:0]  const_addr,
    input  wire [WIDTH-1:0]             const_wdata,

    input  wire                         launch,
    input  wire [PROG_ADDR_WIDTH-1:0]   base_pc,
    input  wire [WARP_SIZE-1:0]         active_mask,
    input  wire [WIDTH-1:0]             block_dim,
    input  wire [WIDTH-1:0]             grid_dim,

    output wire [NUM_CORES-1:0]                         core_busy,
    output wire [NUM_CORES-1:0]                         core_done,
    output wire [NUM_CORES-1:0]                         core_error,
    output wire [NUM_CORES-1:0]                         core_unsupported,
    output wire [NUM_CORES-1:0]                         core_divide_by_zero,
    output wire [(NUM_CORES*PROG_ADDR_WIDTH)-1:0]       core_pc,
    output wire [(NUM_CORES*32)-1:0]                    core_current_instr,
    output wire [(NUM_CORES*32)-1:0]                    core_last_instr,
    output wire [(NUM_CORES*16)-1:0]                    core_retired_count,
    output wire [(NUM_CORES*WARP_SIZE)-1:0]             core_last_writeback_mask,
    output wire [(NUM_CORES*4)-1:0]                     core_last_writeback_addr,
    output wire [(NUM_CORES*WARP_SIZE*WIDTH)-1:0]       core_last_writeback_data,

    output wire                                         busy,
    output wire                                         done,
    output wire                                         error,
    output wire                                         unsupported,
    output wire                                         divide_by_zero
);
    localparam CORE_INDEX_WIDTH = (NUM_CORES <= 1) ? 1 : clog2(NUM_CORES);

    wire [(NUM_CORES*WARP_SIZE)-1:0]             core_mem_req_valid;
    wire [(NUM_CORES*WARP_SIZE)-1:0]             core_mem_req_write;
    wire [(NUM_CORES*WARP_SIZE*ADDR_WIDTH)-1:0]  core_mem_req_addr;
    wire [(NUM_CORES*WARP_SIZE*WIDTH)-1:0]       core_mem_req_wdata;
    reg  [(NUM_CORES*WARP_SIZE)-1:0]             core_mem_req_ready;
    reg  [(NUM_CORES*WARP_SIZE)-1:0]             core_mem_resp_valid;
    reg  [(NUM_CORES*WARP_SIZE*WIDTH)-1:0]       core_mem_resp_rdata;

    reg [CORE_INDEX_WIDTH-1:0] arb_core;
    reg [CORE_INDEX_WIDTH-1:0] rr_core;
    reg [CORE_INDEX_WIDTH-1:0] resp_core_q;
    reg arb_valid;
    reg [WARP_SIZE-1:0] memory_req_valid;
    reg [WARP_SIZE-1:0] memory_req_write;
    reg [(WARP_SIZE*ADDR_WIDTH)-1:0] memory_req_addr;
    reg [(WARP_SIZE*WIDTH)-1:0] memory_req_wdata;
    wire [WARP_SIZE-1:0] memory_req_ready;
    wire [WARP_SIZE-1:0] memory_resp_valid;
    wire [(WARP_SIZE*WIDTH)-1:0] memory_resp_rdata;

    integer arb_scan;
    integer arb_candidate;

    always @* begin
        arb_valid = 1'b0;
        arb_core = rr_core;
        for (arb_scan = 0; arb_scan < NUM_CORES; arb_scan = arb_scan + 1) begin
            arb_candidate = wrap_core_index(rr_core, arb_scan);
            if (!arb_valid &&
                |core_mem_req_valid[(arb_candidate*WARP_SIZE) +: WARP_SIZE]) begin
                arb_core = arb_candidate[CORE_INDEX_WIDTH-1:0];
                arb_valid = 1'b1;
            end
        end
    end

    always @* begin
        memory_req_valid = {WARP_SIZE{1'b0}};
        memory_req_write = {WARP_SIZE{1'b0}};
        memory_req_addr = {(WARP_SIZE*ADDR_WIDTH){1'b0}};
        memory_req_wdata = {(WARP_SIZE*WIDTH){1'b0}};
        core_mem_req_ready = {(NUM_CORES*WARP_SIZE){1'b0}};

        if (arb_valid) begin
            memory_req_valid = core_mem_req_valid[(arb_core*WARP_SIZE) +: WARP_SIZE];
            memory_req_write = core_mem_req_write[(arb_core*WARP_SIZE) +: WARP_SIZE];
            memory_req_addr = core_mem_req_addr[(arb_core*WARP_SIZE*ADDR_WIDTH) +: (WARP_SIZE*ADDR_WIDTH)];
            memory_req_wdata = core_mem_req_wdata[(arb_core*WARP_SIZE*WIDTH) +: (WARP_SIZE*WIDTH)];
            core_mem_req_ready[(arb_core*WARP_SIZE) +: WARP_SIZE] = memory_req_ready;
        end
    end

    always @* begin
        core_mem_resp_valid = {(NUM_CORES*WARP_SIZE){1'b0}};
        core_mem_resp_rdata = {(NUM_CORES*WARP_SIZE*WIDTH){1'b0}};
        core_mem_resp_valid[(resp_core_q*WARP_SIZE) +: WARP_SIZE] = memory_resp_valid;
        core_mem_resp_rdata[(resp_core_q*WARP_SIZE*WIDTH) +: (WARP_SIZE*WIDTH)] = memory_resp_rdata;
    end

    always @(posedge clk) begin
        if (rst) begin
            rr_core <= {CORE_INDEX_WIDTH{1'b0}};
            resp_core_q <= {CORE_INDEX_WIDTH{1'b0}};
        end else begin
            if (arb_valid && |(memory_req_valid & memory_req_ready)) begin
                rr_core <= next_core_index(arb_core);
            end

            if (arb_valid && |(memory_req_valid & memory_req_ready & ~memory_req_write)) begin
                resp_core_q <= arb_core;
            end
        end
    end

    memory #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(WIDTH),
        .BANK_DEPTH(MEMORY_BANK_DEPTH)
    ) global_memory (
        .clk(clk),
        .rst(rst),
        .req_valid(memory_req_valid),
        .req_write(memory_req_write),
        .req_addr(memory_req_addr),
        .req_wdata(memory_req_wdata),
        .req_ready(memory_req_ready),
        .resp_valid(memory_resp_valid),
        .resp_rdata(memory_resp_rdata)
    );

    genvar core_id;
    generate
        for (core_id = 0; core_id < NUM_CORES; core_id = core_id + 1) begin : cores
            wire [WIDTH-1:0] core_block_id = core_id;

            mini_gpu_core #(
                .WIDTH(WIDTH),
                .WARP_SIZE(WARP_SIZE),
                .PROG_ADDR_WIDTH(PROG_ADDR_WIDTH),
                .ADDR_WIDTH(ADDR_WIDTH),
                .CONST_ADDR_WIDTH(CONST_ADDR_WIDTH),
                .NUM_WARPS(NUM_WARPS_PER_CORE),
                .NUM_BLOCKS(1),
                .WARP_ID_WIDTH(WARP_ID_WIDTH),
                .BLOCK_ID_WIDTH(1),
                .ENABLE_FLOAT_ADD(ENABLE_FLOAT_ADD),
                .ENABLE_FLOAT_MUL(ENABLE_FLOAT_MUL),
                .ENABLE_FLOAT_DIV(ENABLE_FLOAT_DIV),
                .FLOAT_FP32_ONLY(FLOAT_FP32_ONLY),
                .USE_SHARED_FLOAT(USE_SHARED_FLOAT),
                .SHARED_FLOAT_UNITS(SHARED_FLOAT_UNITS)
            ) core (
                .clk(clk),
                .rst(rst),
                .prog_we(prog_we),
                .prog_addr(prog_addr),
                .prog_wdata(prog_wdata),
                .const_we(const_we),
                .const_addr(const_addr),
                .const_wdata(const_wdata),
                .launch(launch),
                .base_pc(base_pc),
                .active_mask(active_mask),
                .base_block_id(core_block_id),
                .block_dim(block_dim),
                .grid_dim(grid_dim),
                .mem_req_valid(core_mem_req_valid[(core_id*WARP_SIZE) +: WARP_SIZE]),
                .mem_req_write(core_mem_req_write[(core_id*WARP_SIZE) +: WARP_SIZE]),
                .mem_req_addr(core_mem_req_addr[(core_id*WARP_SIZE*ADDR_WIDTH) +: (WARP_SIZE*ADDR_WIDTH)]),
                .mem_req_wdata(core_mem_req_wdata[(core_id*WARP_SIZE*WIDTH) +: (WARP_SIZE*WIDTH)]),
                .mem_req_ready(core_mem_req_ready[(core_id*WARP_SIZE) +: WARP_SIZE]),
                .mem_resp_valid(core_mem_resp_valid[(core_id*WARP_SIZE) +: WARP_SIZE]),
                .mem_resp_rdata(core_mem_resp_rdata[(core_id*WARP_SIZE*WIDTH) +: (WARP_SIZE*WIDTH)]),
                .busy(core_busy[core_id]),
                .done(core_done[core_id]),
                .error(core_error[core_id]),
                .unsupported(core_unsupported[core_id]),
                .divide_by_zero(core_divide_by_zero[core_id]),
                .pc(core_pc[(core_id*PROG_ADDR_WIDTH) +: PROG_ADDR_WIDTH]),
                .current_instr(core_current_instr[(core_id*32) +: 32]),
                .last_instr(core_last_instr[(core_id*32) +: 32]),
                .retired_count(core_retired_count[(core_id*16) +: 16]),
                .last_writeback_mask(core_last_writeback_mask[(core_id*WARP_SIZE) +: WARP_SIZE]),
                .last_writeback_addr(core_last_writeback_addr[(core_id*4) +: 4]),
                .last_writeback_data(core_last_writeback_data[(core_id*WARP_SIZE*WIDTH) +: (WARP_SIZE*WIDTH)])
            );
        end
    endgenerate

    assign busy = |core_busy;
    assign done = &core_done;
    assign error = |core_error;
    assign unsupported = |core_unsupported;
    assign divide_by_zero = |core_divide_by_zero;

    function [CORE_INDEX_WIDTH-1:0] next_core_index;
        input [CORE_INDEX_WIDTH-1:0] value;
        begin
            if (value == (NUM_CORES - 1)) begin
                next_core_index = {CORE_INDEX_WIDTH{1'b0}};
            end else begin
                next_core_index = value + {{(CORE_INDEX_WIDTH-1){1'b0}}, 1'b1};
            end
        end
    endfunction

    function integer wrap_core_index;
        input integer base;
        input integer offset;
        integer candidate;
        begin
            candidate = base + offset;
            if (candidate >= NUM_CORES) begin
                candidate = candidate - NUM_CORES;
            end
            wrap_core_index = candidate;
        end
    endfunction

    function integer clog2;
        input integer value;
        integer shifted;
        begin
            shifted = value - 1;
            for (clog2 = 0; shifted > 0; clog2 = clog2 + 1) begin
                shifted = shifted >> 1;
            end
        end
    endfunction
endmodule
