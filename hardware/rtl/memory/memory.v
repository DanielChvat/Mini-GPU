`timescale 1ns/1ps

module memory #(
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32,
    // Basys3 / xc7a35t has 50 36Kb BRAM tiles. Vivado rounds this true
    // dual-port RAM up to a power-of-two depth, so 8192 words per bank uses
    // 8 RAMB36 blocks per bank, or 32 total across the four banks.
    parameter BANK_DEPTH = 8192
) (
    input  wire                      clk,
    input  wire                      rst,
    input  wire [3:0]                req_valid,
    input  wire [3:0]                req_write,
    input  wire [(4*ADDR_WIDTH)-1:0] req_addr,
    input  wire [(4*DATA_WIDTH)-1:0] req_wdata,
    input  wire [(4*4)-1:0]          req_wmask,
    output reg  [3:0]                req_ready,
    output reg  [3:0]                resp_valid,
    output reg  [(4*DATA_WIDTH)-1:0] resp_rdata
);
    localparam BANK_BITS = 2;
    localparam BANK_ADDR_WIDTH = clog2(BANK_DEPTH);
    localparam NO_LANE = 3'd4;

    wire [ADDR_WIDTH-1:0] req_lane_addr0 = req_addr[(0*ADDR_WIDTH) +: ADDR_WIDTH];
    wire [ADDR_WIDTH-1:0] req_lane_addr1 = req_addr[(1*ADDR_WIDTH) +: ADDR_WIDTH];
    wire [ADDR_WIDTH-1:0] req_lane_addr2 = req_addr[(2*ADDR_WIDTH) +: ADDR_WIDTH];
    wire [ADDR_WIDTH-1:0] req_lane_addr3 = req_addr[(3*ADDR_WIDTH) +: ADDR_WIDTH];

    wire [1:0] req_lane_bank0 = req_lane_addr0[1:0];
    wire [1:0] req_lane_bank1 = req_lane_addr1[1:0];
    wire [1:0] req_lane_bank2 = req_lane_addr2[1:0];
    wire [1:0] req_lane_bank3 = req_lane_addr3[1:0];

    wire [BANK_ADDR_WIDTH-1:0] req_lane_index0 = req_lane_addr0[BANK_BITS +: BANK_ADDR_WIDTH];
    wire [BANK_ADDR_WIDTH-1:0] req_lane_index1 = req_lane_addr1[BANK_BITS +: BANK_ADDR_WIDTH];
    wire [BANK_ADDR_WIDTH-1:0] req_lane_index2 = req_lane_addr2[BANK_BITS +: BANK_ADDR_WIDTH];
    wire [BANK_ADDR_WIDTH-1:0] req_lane_index3 = req_lane_addr3[BANK_BITS +: BANK_ADDR_WIDTH];

    reg [3:0] pipe_valid;
    reg [3:0] pipe_write;
    reg [(4*ADDR_WIDTH)-1:0] pipe_addr;
    reg [(4*DATA_WIDTH)-1:0] pipe_wdata;
    reg [(4*4)-1:0] pipe_wmask;

    wire [ADDR_WIDTH-1:0] lane_addr0 = pipe_addr[(0*ADDR_WIDTH) +: ADDR_WIDTH];
    wire [ADDR_WIDTH-1:0] lane_addr1 = pipe_addr[(1*ADDR_WIDTH) +: ADDR_WIDTH];
    wire [ADDR_WIDTH-1:0] lane_addr2 = pipe_addr[(2*ADDR_WIDTH) +: ADDR_WIDTH];
    wire [ADDR_WIDTH-1:0] lane_addr3 = pipe_addr[(3*ADDR_WIDTH) +: ADDR_WIDTH];

    wire [DATA_WIDTH-1:0] lane_wdata0 = pipe_wdata[(0*DATA_WIDTH) +: DATA_WIDTH];
    wire [DATA_WIDTH-1:0] lane_wdata1 = pipe_wdata[(1*DATA_WIDTH) +: DATA_WIDTH];
    wire [DATA_WIDTH-1:0] lane_wdata2 = pipe_wdata[(2*DATA_WIDTH) +: DATA_WIDTH];
    wire [DATA_WIDTH-1:0] lane_wdata3 = pipe_wdata[(3*DATA_WIDTH) +: DATA_WIDTH];

    wire [3:0] lane_wmask0 = pipe_wmask[(0*4) +: 4];
    wire [3:0] lane_wmask1 = pipe_wmask[(1*4) +: 4];
    wire [3:0] lane_wmask2 = pipe_wmask[(2*4) +: 4];
    wire [3:0] lane_wmask3 = pipe_wmask[(3*4) +: 4];

    wire [1:0] lane_bank0 = lane_addr0[1:0];
    wire [1:0] lane_bank1 = lane_addr1[1:0];
    wire [1:0] lane_bank2 = lane_addr2[1:0];
    wire [1:0] lane_bank3 = lane_addr3[1:0];

    wire [BANK_ADDR_WIDTH-1:0] lane_index0 = lane_addr0[BANK_BITS +: BANK_ADDR_WIDTH];
    wire [BANK_ADDR_WIDTH-1:0] lane_index1 = lane_addr1[BANK_BITS +: BANK_ADDR_WIDTH];
    wire [BANK_ADDR_WIDTH-1:0] lane_index2 = lane_addr2[BANK_BITS +: BANK_ADDR_WIDTH];
    wire [BANK_ADDR_WIDTH-1:0] lane_index3 = lane_addr3[BANK_BITS +: BANK_ADDR_WIDTH];

    reg b0_en_a;
    reg b0_we_a;
    reg [3:0] b0_wem_a;
    reg [BANK_ADDR_WIDTH-1:0] b0_addr_a;
    reg [DATA_WIDTH-1:0] b0_din_a;
    wire [DATA_WIDTH-1:0] b0_dout_a;
    reg b0_en_b;
    reg b0_we_b;
    reg [3:0] b0_wem_b;
    reg [BANK_ADDR_WIDTH-1:0] b0_addr_b;
    reg [DATA_WIDTH-1:0] b0_din_b;
    wire [DATA_WIDTH-1:0] b0_dout_b;

    reg b1_en_a;
    reg b1_we_a;
    reg [3:0] b1_wem_a;
    reg [BANK_ADDR_WIDTH-1:0] b1_addr_a;
    reg [DATA_WIDTH-1:0] b1_din_a;
    wire [DATA_WIDTH-1:0] b1_dout_a;
    reg b1_en_b;
    reg b1_we_b;
    reg [3:0] b1_wem_b;
    reg [BANK_ADDR_WIDTH-1:0] b1_addr_b;
    reg [DATA_WIDTH-1:0] b1_din_b;
    wire [DATA_WIDTH-1:0] b1_dout_b;

    reg b2_en_a;
    reg b2_we_a;
    reg [3:0] b2_wem_a;
    reg [BANK_ADDR_WIDTH-1:0] b2_addr_a;
    reg [DATA_WIDTH-1:0] b2_din_a;
    wire [DATA_WIDTH-1:0] b2_dout_a;
    reg b2_en_b;
    reg b2_we_b;
    reg [3:0] b2_wem_b;
    reg [BANK_ADDR_WIDTH-1:0] b2_addr_b;
    reg [DATA_WIDTH-1:0] b2_din_b;
    wire [DATA_WIDTH-1:0] b2_dout_b;

    reg b3_en_a;
    reg b3_we_a;
    reg [3:0] b3_wem_a;
    reg [BANK_ADDR_WIDTH-1:0] b3_addr_a;
    reg [DATA_WIDTH-1:0] b3_din_a;
    wire [DATA_WIDTH-1:0] b3_dout_a;
    reg b3_en_b;
    reg b3_we_b;
    reg [3:0] b3_wem_b;
    reg [BANK_ADDR_WIDTH-1:0] b3_addr_b;
    reg [DATA_WIDTH-1:0] b3_din_b;
    wire [DATA_WIDTH-1:0] b3_dout_b;

    reg [2:0] b0_lane_a;
    reg [2:0] b0_lane_b;
    reg [2:0] b1_lane_a;
    reg [2:0] b1_lane_b;
    reg [2:0] b2_lane_a;
    reg [2:0] b2_lane_b;
    reg [2:0] b3_lane_a;
    reg [2:0] b3_lane_b;

    reg b0_read_a_q;
    reg b0_read_b_q;
    reg [2:0] b0_lane_a_q;
    reg [2:0] b0_lane_b_q;
    reg b1_read_a_q;
    reg b1_read_b_q;
    reg [2:0] b1_lane_a_q;
    reg [2:0] b1_lane_b_q;
    reg b2_read_a_q;
    reg b2_read_b_q;
    reg [2:0] b2_lane_a_q;
    reg [2:0] b2_lane_b_q;
    reg b3_read_a_q;
    reg b3_read_b_q;
    reg [2:0] b3_lane_a_q;
    reg [2:0] b3_lane_b_q;

    memory_bank #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(BANK_ADDR_WIDTH), .DEPTH(BANK_DEPTH)) bank0 (
        .clk(clk),
        .en_a(b0_en_a),
        .we_a(b0_we_a),
        .wem_a(b0_wem_a),
        .addr_a(b0_addr_a),
        .din_a(b0_din_a),
        .dout_a(b0_dout_a),
        .en_b(b0_en_b),
        .we_b(b0_we_b),
        .wem_b(b0_wem_b),
        .addr_b(b0_addr_b),
        .din_b(b0_din_b),
        .dout_b(b0_dout_b)
    );

    memory_bank #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(BANK_ADDR_WIDTH), .DEPTH(BANK_DEPTH)) bank1 (
        .clk(clk),
        .en_a(b1_en_a),
        .we_a(b1_we_a),
        .wem_a(b1_wem_a),
        .addr_a(b1_addr_a),
        .din_a(b1_din_a),
        .dout_a(b1_dout_a),
        .en_b(b1_en_b),
        .we_b(b1_we_b),
        .wem_b(b1_wem_b),
        .addr_b(b1_addr_b),
        .din_b(b1_din_b),
        .dout_b(b1_dout_b)
    );

    memory_bank #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(BANK_ADDR_WIDTH), .DEPTH(BANK_DEPTH)) bank2 (
        .clk(clk),
        .en_a(b2_en_a),
        .we_a(b2_we_a),
        .wem_a(b2_wem_a),
        .addr_a(b2_addr_a),
        .din_a(b2_din_a),
        .dout_a(b2_dout_a),
        .en_b(b2_en_b),
        .we_b(b2_we_b),
        .wem_b(b2_wem_b),
        .addr_b(b2_addr_b),
        .din_b(b2_din_b),
        .dout_b(b2_dout_b)
    );

    memory_bank #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(BANK_ADDR_WIDTH), .DEPTH(BANK_DEPTH)) bank3 (
        .clk(clk),
        .en_a(b3_en_a),
        .we_a(b3_we_a),
        .wem_a(b3_wem_a),
        .addr_a(b3_addr_a),
        .din_a(b3_din_a),
        .dout_a(b3_dout_a),
        .en_b(b3_en_b),
        .we_b(b3_we_b),
        .wem_b(b3_wem_b),
        .addr_b(b3_addr_b),
        .din_b(b3_din_b),
        .dout_b(b3_dout_b)
    );

    always @* begin
        req_ready[0] = req_valid[0];
        req_ready[1] = req_valid[1] && (same_prior_count(1) < 2);
        req_ready[2] = req_valid[2] && (same_prior_count(2) < 2);
        req_ready[3] = req_valid[3] && (same_prior_count(3) < 2);

        clear_ports();

        assign_lane_to_port(3'd0, lane_bank0, lane_index0, pipe_write[0], lane_wdata0, lane_wmask0, pipe_valid[0]);
        assign_lane_to_port(3'd1, lane_bank1, lane_index1, pipe_write[1], lane_wdata1, lane_wmask1, pipe_valid[1]);
        assign_lane_to_port(3'd2, lane_bank2, lane_index2, pipe_write[2], lane_wdata2, lane_wmask2, pipe_valid[2]);
        assign_lane_to_port(3'd3, lane_bank3, lane_index3, pipe_write[3], lane_wdata3, lane_wmask3, pipe_valid[3]);
    end

    always @(posedge clk) begin
        if (rst) begin
            pipe_valid <= 4'b0000;
            pipe_write <= 4'b0000;
            pipe_addr <= {(4*ADDR_WIDTH){1'b0}};
            pipe_wdata <= {(4*DATA_WIDTH){1'b0}};
            pipe_wmask <= {(4*4){1'b0}};
            b0_read_a_q <= 1'b0;
            b0_read_b_q <= 1'b0;
            b0_lane_a_q <= NO_LANE;
            b0_lane_b_q <= NO_LANE;
            b1_read_a_q <= 1'b0;
            b1_read_b_q <= 1'b0;
            b1_lane_a_q <= NO_LANE;
            b1_lane_b_q <= NO_LANE;
            b2_read_a_q <= 1'b0;
            b2_read_b_q <= 1'b0;
            b2_lane_a_q <= NO_LANE;
            b2_lane_b_q <= NO_LANE;
            b3_read_a_q <= 1'b0;
            b3_read_b_q <= 1'b0;
            b3_lane_a_q <= NO_LANE;
            b3_lane_b_q <= NO_LANE;
        end else begin
            pipe_valid <= req_valid & req_ready;
            pipe_write <= req_write;
            pipe_addr <= req_addr;
            pipe_wdata <= req_wdata;
            pipe_wmask <= req_wmask;
            b0_read_a_q <= b0_en_a && !b0_we_a;
            b0_read_b_q <= b0_en_b && !b0_we_b;
            b0_lane_a_q <= b0_lane_a;
            b0_lane_b_q <= b0_lane_b;
            b1_read_a_q <= b1_en_a && !b1_we_a;
            b1_read_b_q <= b1_en_b && !b1_we_b;
            b1_lane_a_q <= b1_lane_a;
            b1_lane_b_q <= b1_lane_b;
            b2_read_a_q <= b2_en_a && !b2_we_a;
            b2_read_b_q <= b2_en_b && !b2_we_b;
            b2_lane_a_q <= b2_lane_a;
            b2_lane_b_q <= b2_lane_b;
            b3_read_a_q <= b3_en_a && !b3_we_a;
            b3_read_b_q <= b3_en_b && !b3_we_b;
            b3_lane_a_q <= b3_lane_a;
            b3_lane_b_q <= b3_lane_b;
        end
    end

    always @* begin
        resp_valid = 4'b0000;
        resp_rdata = {(4*DATA_WIDTH){1'b0}};

        route_response(b0_read_a_q, b0_lane_a_q, b0_dout_a);
        route_response(b0_read_b_q, b0_lane_b_q, b0_dout_b);
        route_response(b1_read_a_q, b1_lane_a_q, b1_dout_a);
        route_response(b1_read_b_q, b1_lane_b_q, b1_dout_b);
        route_response(b2_read_a_q, b2_lane_a_q, b2_dout_a);
        route_response(b2_read_b_q, b2_lane_b_q, b2_dout_b);
        route_response(b3_read_a_q, b3_lane_a_q, b3_dout_a);
        route_response(b3_read_b_q, b3_lane_b_q, b3_dout_b);
    end

    task clear_ports;
        begin
            b0_en_a = 1'b0; b0_we_a = 1'b0; b0_wem_a = 4'b0000; b0_addr_a = {BANK_ADDR_WIDTH{1'b0}}; b0_din_a = {DATA_WIDTH{1'b0}}; b0_lane_a = NO_LANE;
            b0_en_b = 1'b0; b0_we_b = 1'b0; b0_wem_b = 4'b0000; b0_addr_b = {BANK_ADDR_WIDTH{1'b0}}; b0_din_b = {DATA_WIDTH{1'b0}}; b0_lane_b = NO_LANE;
            b1_en_a = 1'b0; b1_we_a = 1'b0; b1_wem_a = 4'b0000; b1_addr_a = {BANK_ADDR_WIDTH{1'b0}}; b1_din_a = {DATA_WIDTH{1'b0}}; b1_lane_a = NO_LANE;
            b1_en_b = 1'b0; b1_we_b = 1'b0; b1_wem_b = 4'b0000; b1_addr_b = {BANK_ADDR_WIDTH{1'b0}}; b1_din_b = {DATA_WIDTH{1'b0}}; b1_lane_b = NO_LANE;
            b2_en_a = 1'b0; b2_we_a = 1'b0; b2_wem_a = 4'b0000; b2_addr_a = {BANK_ADDR_WIDTH{1'b0}}; b2_din_a = {DATA_WIDTH{1'b0}}; b2_lane_a = NO_LANE;
            b2_en_b = 1'b0; b2_we_b = 1'b0; b2_wem_b = 4'b0000; b2_addr_b = {BANK_ADDR_WIDTH{1'b0}}; b2_din_b = {DATA_WIDTH{1'b0}}; b2_lane_b = NO_LANE;
            b3_en_a = 1'b0; b3_we_a = 1'b0; b3_wem_a = 4'b0000; b3_addr_a = {BANK_ADDR_WIDTH{1'b0}}; b3_din_a = {DATA_WIDTH{1'b0}}; b3_lane_a = NO_LANE;
            b3_en_b = 1'b0; b3_we_b = 1'b0; b3_wem_b = 4'b0000; b3_addr_b = {BANK_ADDR_WIDTH{1'b0}}; b3_din_b = {DATA_WIDTH{1'b0}}; b3_lane_b = NO_LANE;
        end
    endtask

    task assign_lane_to_port;
        input [2:0] lane_id;
        input [1:0] bank_id;
        input [BANK_ADDR_WIDTH-1:0] bank_index;
        input write;
        input [DATA_WIDTH-1:0] data;
        input [3:0] wmask;
        input valid;
        begin
            if (valid) begin
                case (bank_id)
                    2'd0: begin
                        if (write && b0_en_a && b0_we_a && b0_addr_a == bank_index) begin
                            merge_write_data(b0_din_a, b0_wem_a, data, wmask);
                        end else if (write && b0_en_b && b0_we_b && b0_addr_b == bank_index) begin
                            merge_write_data(b0_din_b, b0_wem_b, data, wmask);
                        end else if (!b0_en_a) begin
                            b0_en_a = 1'b1; b0_we_a = write; b0_wem_a = wmask; b0_addr_a = bank_index; b0_din_a = data; b0_lane_a = lane_id;
                        end else begin
                            b0_en_b = 1'b1; b0_we_b = write; b0_wem_b = wmask; b0_addr_b = bank_index; b0_din_b = data; b0_lane_b = lane_id;
                        end
                    end
                    2'd1: begin
                        if (write && b1_en_a && b1_we_a && b1_addr_a == bank_index) begin
                            merge_write_data(b1_din_a, b1_wem_a, data, wmask);
                        end else if (write && b1_en_b && b1_we_b && b1_addr_b == bank_index) begin
                            merge_write_data(b1_din_b, b1_wem_b, data, wmask);
                        end else if (!b1_en_a) begin
                            b1_en_a = 1'b1; b1_we_a = write; b1_wem_a = wmask; b1_addr_a = bank_index; b1_din_a = data; b1_lane_a = lane_id;
                        end else begin
                            b1_en_b = 1'b1; b1_we_b = write; b1_wem_b = wmask; b1_addr_b = bank_index; b1_din_b = data; b1_lane_b = lane_id;
                        end
                    end
                    2'd2: begin
                        if (write && b2_en_a && b2_we_a && b2_addr_a == bank_index) begin
                            merge_write_data(b2_din_a, b2_wem_a, data, wmask);
                        end else if (write && b2_en_b && b2_we_b && b2_addr_b == bank_index) begin
                            merge_write_data(b2_din_b, b2_wem_b, data, wmask);
                        end else if (!b2_en_a) begin
                            b2_en_a = 1'b1; b2_we_a = write; b2_wem_a = wmask; b2_addr_a = bank_index; b2_din_a = data; b2_lane_a = lane_id;
                        end else begin
                            b2_en_b = 1'b1; b2_we_b = write; b2_wem_b = wmask; b2_addr_b = bank_index; b2_din_b = data; b2_lane_b = lane_id;
                        end
                    end
                    default: begin
                        if (write && b3_en_a && b3_we_a && b3_addr_a == bank_index) begin
                            merge_write_data(b3_din_a, b3_wem_a, data, wmask);
                        end else if (write && b3_en_b && b3_we_b && b3_addr_b == bank_index) begin
                            merge_write_data(b3_din_b, b3_wem_b, data, wmask);
                        end else if (!b3_en_a) begin
                            b3_en_a = 1'b1; b3_we_a = write; b3_wem_a = wmask; b3_addr_a = bank_index; b3_din_a = data; b3_lane_a = lane_id;
                        end else begin
                            b3_en_b = 1'b1; b3_we_b = write; b3_wem_b = wmask; b3_addr_b = bank_index; b3_din_b = data; b3_lane_b = lane_id;
                        end
                    end
                endcase
            end
        end
    endtask

    task merge_write_data;
        inout [DATA_WIDTH-1:0] dest;
        inout [3:0] dest_mask;
        input [DATA_WIDTH-1:0] src;
        input [3:0] src_mask;
        begin
            if (src_mask[0]) dest[7:0] = src[7:0];
            if (src_mask[1]) dest[15:8] = src[15:8];
            if (src_mask[2]) dest[23:16] = src[23:16];
            if (src_mask[3]) dest[31:24] = src[31:24];
            dest_mask = dest_mask | src_mask;
        end
    endtask

    task route_response;
        input valid;
        input [2:0] lane_id;
        input [DATA_WIDTH-1:0] data;
        begin
            if (valid && lane_id < 3'd4) begin
                resp_valid[lane_id] = 1'b1;
                resp_rdata[(lane_id*DATA_WIDTH) +: DATA_WIDTH] = data;
            end
        end
    endtask

    function [1:0] same_prior_count;
        input [1:0] lane_id;
        begin
            case (lane_id)
                2'd0: same_prior_count = 2'd0;
                2'd1: same_prior_count = prior_port_count(2'd1, req_lane_bank1, req_lane_index1, req_write[1]);
                2'd2: same_prior_count = prior_port_count(2'd2, req_lane_bank2, req_lane_index2, req_write[2]);
                default: same_prior_count = prior_port_count(2'd3, req_lane_bank3, req_lane_index3, req_write[3]);
            endcase
        end
    endfunction

    function [1:0] prior_port_count;
        input [1:0] lane_id;
        input [1:0] bank_id;
        input [BANK_ADDR_WIDTH-1:0] bank_index;
        input write;
        begin
            prior_port_count = 2'd0;
            if (lane_id > 2'd0 && prior_uses_port(req_valid[0], req_write[0], req_lane_bank0, req_lane_index0, write, bank_id, bank_index)) begin
                prior_port_count = prior_port_count + 2'd1;
            end
            if (lane_id > 2'd1 && prior_uses_port(req_valid[1], req_write[1], req_lane_bank1, req_lane_index1, write, bank_id, bank_index)) begin
                prior_port_count = prior_port_count + 2'd1;
            end
            if (lane_id > 2'd2 && prior_uses_port(req_valid[2], req_write[2], req_lane_bank2, req_lane_index2, write, bank_id, bank_index)) begin
                prior_port_count = prior_port_count + 2'd1;
            end
        end
    endfunction

    function prior_uses_port;
        input prior_valid;
        input prior_write;
        input [1:0] prior_bank;
        input [BANK_ADDR_WIDTH-1:0] prior_index;
        input current_write;
        input [1:0] current_bank;
        input [BANK_ADDR_WIDTH-1:0] current_index;
        begin
            prior_uses_port = prior_valid &&
                              (prior_bank == current_bank) &&
                              !(prior_write && current_write && (prior_index == current_index));
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
