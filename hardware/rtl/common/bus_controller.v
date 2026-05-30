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
    input  wire [(4*4)-1:0] host_mem_req_wmask,
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
    input  wire [(4*4)-1:0] core_mem_req_wmask,
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
    output wire [(4*4)-1:0] merged_mem_req_wmask,
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

    assign merged_mem_req_wmask = select_host ? host_mem_req_wmask :
                                  select_core ? core_mem_req_wmask : {(4*4){1'b0}};

    assign memory_request_id = {1'b0, select_core};

    // =========================================================================
    // Ready routing
    // =========================================================================
    assign host_mem_req_ready = select_host ? merged_mem_req_ready : 4'b0;
    assign core_mem_req_ready = select_core ? merged_mem_req_ready : 4'b0;

    // =========================================================================
    // Response routing: latch who made the request
    // =========================================================================
    reg resp_owner_core_q;
    wire request_accepted = |(merged_mem_req_valid & merged_mem_req_ready);
    wire request_is_read = |(merged_mem_req_valid & ~merged_mem_req_write);
    wire request_is_write = |(merged_mem_req_valid & merged_mem_req_write);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            resp_owner_core_q <= 1'b0;
        end else if (request_accepted && request_is_read) begin
            resp_owner_core_q <= select_core;
        end
    end

    assign host_mem_resp_valid = resp_owner_core_q ? 4'b0 : merged_mem_resp_valid;
    assign core_mem_resp_valid = resp_owner_core_q ? merged_mem_resp_valid : 4'b0;

    assign host_mem_resp_rdata = merged_mem_resp_rdata;
    assign core_mem_resp_rdata = merged_mem_resp_rdata;

    // =========================================================================
    // Status flag demux
    // =========================================================================
    reg write_owner_core_q;
    reg host_mem_write_done_q;
    reg host_mem_write_fail_q;
    reg core_mem_write_done_q;
    reg core_mem_write_fail_q;

    wire write_status_owner_core =
        (request_accepted && request_is_write) ? select_core : write_owner_core_q;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            write_owner_core_q <= 1'b0;
            host_mem_write_done_q <= 1'b0;
            host_mem_write_fail_q <= 1'b0;
            core_mem_write_done_q <= 1'b0;
            core_mem_write_fail_q <= 1'b0;
        end else begin
            if (request_accepted && request_is_write) begin
                write_owner_core_q <= select_core;
            end

            if (host_memory_status_consumed) begin
                host_mem_write_done_q <= 1'b0;
                host_mem_write_fail_q <= 1'b0;
            end
            if (core_memory_status_consumed) begin
                core_mem_write_done_q <= 1'b0;
                core_mem_write_fail_q <= 1'b0;
            end

            if (merged_mem_write_done) begin
                if (write_status_owner_core) begin
                    core_mem_write_done_q <= 1'b1;
                end else begin
                    host_mem_write_done_q <= 1'b1;
                end
            end

            if (merged_mem_write_fail) begin
                if (write_status_owner_core) begin
                    core_mem_write_fail_q <= 1'b1;
                end else begin
                    host_mem_write_fail_q <= 1'b1;
                end
            end
        end
    end

    assign host_mem_write_done = host_mem_write_done_q;
    assign host_mem_write_fail = host_mem_write_fail_q;
    assign core_mem_write_done = core_mem_write_done_q;
    assign core_mem_write_fail = core_mem_write_fail_q;

    assign merged_memory_status_consumed = host_memory_status_consumed | core_memory_status_consumed;

endmodule

`default_nettype wire