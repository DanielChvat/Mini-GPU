`timescale 1ns/1ps
`default_nettype none

module bus_controller #(
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32
) (
    input  wire        clk,
    input  wire        rst,

    // Host memory bus
    input  wire [3:0]  host_mem_req_valid,
    input  wire [3:0]  host_mem_req_write,
    input  wire [(4*ADDR_WIDTH)-1:0] host_mem_req_addr,
    input  wire [(4*DATA_WIDTH)-1:0] host_mem_req_wdata,
    output wire [3:0]  host_mem_req_ready,
    output wire [3:0]  host_mem_resp_valid,
    output wire [(4*DATA_WIDTH)-1:0] host_mem_resp_rdata,

    // Host status flags
    output wire        host_mem_write_done,
    output wire        host_mem_write_fail,
    input  wire        host_memory_status_consumed,

    // Core memory bus
    input  wire [3:0]  core_mem_req_valid,
    input  wire [3:0]  core_mem_req_write,
    input  wire [(4*ADDR_WIDTH)-1:0] core_mem_req_addr,
    input  wire [(4*DATA_WIDTH)-1:0] core_mem_req_wdata,
    output wire [3:0]  core_mem_req_ready,
    output wire [3:0]  core_mem_resp_valid,
    output wire [(4*DATA_WIDTH)-1:0] core_mem_resp_rdata,

    // Core status flags
    output wire        core_mem_write_done,
    output wire        core_mem_write_fail,
    input  wire        core_memory_status_consumed,

    // Merged output bus (to security_gate)
    output wire [3:0]  merged_mem_req_valid,
    output wire [3:0]  merged_mem_req_write,
    output wire [(4*ADDR_WIDTH)-1:0] merged_mem_req_addr,
    output wire [(4*DATA_WIDTH)-1:0] merged_mem_req_wdata,
    input  wire [3:0]  merged_mem_req_ready,
    input  wire [3:0]  merged_mem_resp_valid,
    input  wire [(4*DATA_WIDTH)-1:0] merged_mem_resp_rdata,

    // Status flags from security_gate (demuxed to requesters)
    input  wire        merged_mem_write_done,
    input  wire        merged_mem_write_fail,

    // Consumed flag to security_gate (muxed from requesters)
    output wire        merged_memory_status_consumed,

    // Request origin ID
    output wire [1:0]  memory_request_id
);

    localparam REQ_HOST  = 2'd0;
    localparam REQ_CORE  = 2'd1;
    localparam REQ_SEC   = 2'd2;
    localparam REQ_DEBUG = 2'd3;

    // =========================================================================
    // Arbitration: host priority
    // =========================================================================
    wire host_active = |host_mem_req_valid;
    wire core_active = |core_mem_req_valid;

    wire select_host = host_active;
    wire select_core = core_active && !host_active;

    // =========================================================================
    // Request mux (combinational)
    // =========================================================================
    assign merged_mem_req_valid = select_host ? host_mem_req_valid :
                                  select_core ? core_mem_req_valid : 4'b0;

    assign merged_mem_req_write = select_host ? host_mem_req_write :
                                  select_core ? core_mem_req_write : 4'b0;

    assign merged_mem_req_addr  = select_host ? host_mem_req_addr :
                                  select_core ? core_mem_req_addr : {(4*ADDR_WIDTH){1'b0}};

    assign merged_mem_req_wdata = select_host ? host_mem_req_wdata :
                                  select_core ? core_mem_req_wdata : {(4*DATA_WIDTH){1'b0}};

    assign memory_request_id = select_host ? REQ_HOST :
                               select_core ? REQ_CORE : REQ_HOST;

    // =========================================================================
    // Ready routing
    // =========================================================================
    assign host_mem_req_ready = select_host ? merged_mem_req_ready : 4'b0;
    assign core_mem_req_ready = select_core ? merged_mem_req_ready : 4'b0;

    // =========================================================================
    // Response routing: latch who made the request
    // =========================================================================
    reg [1:0] resp_owner_q;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            resp_owner_q <= REQ_HOST;
        end else if (select_host || select_core) begin
            resp_owner_q <= select_host ? REQ_HOST : REQ_CORE;
        end
    end

    assign host_mem_resp_valid = (resp_owner_q == REQ_HOST) ? merged_mem_resp_valid : 4'b0;
    assign core_mem_resp_valid = (resp_owner_q == REQ_CORE) ? merged_mem_resp_valid : 4'b0;

    assign host_mem_resp_rdata = merged_mem_resp_rdata;
    assign core_mem_resp_rdata = merged_mem_resp_rdata;

    // =========================================================================
    // Status flag demux
    // =========================================================================
    assign host_mem_write_done = (resp_owner_q == REQ_HOST) ? merged_mem_write_done : 1'b0;
    assign host_mem_write_fail = (resp_owner_q == REQ_HOST) ? merged_mem_write_fail : 1'b0;
    assign core_mem_write_done = (resp_owner_q == REQ_CORE) ? merged_mem_write_done : 1'b0;
    assign core_mem_write_fail = (resp_owner_q == REQ_CORE) ? merged_mem_write_fail : 1'b0;

    assign merged_memory_status_consumed = host_memory_status_consumed | core_memory_status_consumed;

endmodule

`default_nettype wire
