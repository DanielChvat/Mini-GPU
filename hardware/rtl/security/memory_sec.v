`timescale 1ns/1ps

module memory_sec #(
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32,
    parameter BANK_DEPTH = 8192
) (
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      locked,
    input  wire [ADDR_WIDTH-1:0]     protected_base,
    input  wire [ADDR_WIDTH-1:0]     protected_limit,
    input  wire [3:0]                req_valid,
    input  wire [3:0]                req_write,
    input  wire [(4*ADDR_WIDTH)-1:0] req_addr,
    input  wire [(4*DATA_WIDTH)-1:0] req_wdata,
    output wire [3:0]                req_ready,
    output wire [3:0]                resp_valid,
    output wire [(4*DATA_WIDTH)-1:0] resp_rdata,
    output wire [3:0]                write_blocked
);
    wire [3:0] inner_req_valid;
    wire [3:0] inner_req_ready;

    wire [ADDR_WIDTH-1:0] lane_addr0 = req_addr[(0*ADDR_WIDTH) +: ADDR_WIDTH];
    wire [ADDR_WIDTH-1:0] lane_addr1 = req_addr[(1*ADDR_WIDTH) +: ADDR_WIDTH];
    wire [ADDR_WIDTH-1:0] lane_addr2 = req_addr[(2*ADDR_WIDTH) +: ADDR_WIDTH];
    wire [ADDR_WIDTH-1:0] lane_addr3 = req_addr[(3*ADDR_WIDTH) +: ADDR_WIDTH];

    wire protected_region_valid = protected_base < protected_limit;

    wire lane_protected0 = protected_region_valid &&
                           (lane_addr0 >= protected_base) &&
                           (lane_addr0 < protected_limit);
    wire lane_protected1 = protected_region_valid &&
                           (lane_addr1 >= protected_base) &&
                           (lane_addr1 < protected_limit);
    wire lane_protected2 = protected_region_valid &&
                           (lane_addr2 >= protected_base) &&
                           (lane_addr2 < protected_limit);
    wire lane_protected3 = protected_region_valid &&
                           (lane_addr3 >= protected_base) &&
                           (lane_addr3 < protected_limit);

    assign write_blocked[0] = locked && req_valid[0] && req_write[0] && lane_protected0;
    assign write_blocked[1] = locked && req_valid[1] && req_write[1] && lane_protected1;
    assign write_blocked[2] = locked && req_valid[2] && req_write[2] && lane_protected2;
    assign write_blocked[3] = locked && req_valid[3] && req_write[3] && lane_protected3;

    assign inner_req_valid = req_valid & ~write_blocked;
    assign req_ready = (inner_req_ready & ~write_blocked) | write_blocked;

    memory #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .BANK_DEPTH(BANK_DEPTH)
    ) mem (
        .clk(clk),
        .rst(rst),
        .req_valid(inner_req_valid),
        .req_write(req_write),
        .req_addr(req_addr),
        .req_wdata(req_wdata),
        .req_ready(inner_req_ready),
        .resp_valid(resp_valid),
        .resp_rdata(resp_rdata)
    );
endmodule
