`timescale 1ns/1ps

module memory_tb;
    localparam LANES = 4;
    localparam ADDR_WIDTH = 8;
    localparam DATA_WIDTH = 32;

    reg clk;
    reg rst;
    reg [LANES-1:0] req_valid;
    reg [LANES-1:0] req_write;
    reg [(LANES*ADDR_WIDTH)-1:0] req_addr;
    reg [(LANES*DATA_WIDTH)-1:0] req_wdata;
    wire [LANES-1:0] req_ready;
    wire [LANES-1:0] resp_valid;
    wire [(LANES*DATA_WIDTH)-1:0] resp_rdata;

    memory #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .BANK_DEPTH(64)
    ) dut (
        .clk(clk),
        .rst(rst),
        .req_valid(req_valid),
        .req_write(req_write),
        .req_addr(req_addr),
        .req_wdata(req_wdata),
        .req_ready(req_ready),
        .resp_valid(resp_valid),
        .resp_rdata(resp_rdata)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task set_lane;
        input integer lane;
        input valid;
        input write;
        input [ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] data;
        begin
            req_valid[lane] = valid;
            req_write[lane] = write;
            req_addr[(lane*ADDR_WIDTH) +: ADDR_WIDTH] = addr;
            req_wdata[(lane*DATA_WIDTH) +: DATA_WIDTH] = data;
        end
    endtask

    task expect_ready;
        input [LANES-1:0] expected;
        begin
            #1;
            $display("MEM ready: valid=%b write=%b addr={%0d,%0d,%0d,%0d} wdata={0x%08h,0x%08h,0x%08h,0x%08h} -> ready=%b expected=%b",
                     req_valid,
                     req_write,
                     req_addr[(3*ADDR_WIDTH) +: ADDR_WIDTH],
                     req_addr[(2*ADDR_WIDTH) +: ADDR_WIDTH],
                     req_addr[(1*ADDR_WIDTH) +: ADDR_WIDTH],
                     req_addr[(0*ADDR_WIDTH) +: ADDR_WIDTH],
                     req_wdata[(3*DATA_WIDTH) +: DATA_WIDTH],
                     req_wdata[(2*DATA_WIDTH) +: DATA_WIDTH],
                     req_wdata[(1*DATA_WIDTH) +: DATA_WIDTH],
                     req_wdata[(0*DATA_WIDTH) +: DATA_WIDTH],
                     req_ready,
                     expected);
            if (req_ready !== expected) begin
                $display("FAIL ready got=%b expected=%b", req_ready, expected);
                $finish;
            end
        end
    endtask

    task expect_resp;
        input integer lane;
        input [DATA_WIDTH-1:0] expected;
        begin
            $display("MEM resp: lane=%0d valid=%0d rdata=0x%08h expected=0x%08h",
                     lane, resp_valid[lane], resp_rdata[(lane*DATA_WIDTH) +: DATA_WIDTH], expected);
            if (!resp_valid[lane]) begin
                $display("FAIL lane %0d missing response", lane);
                $finish;
            end
            if (resp_rdata[(lane*DATA_WIDTH) +: DATA_WIDTH] !== expected) begin
                $display("FAIL lane %0d data got=0x%08h expected=0x%08h",
                         lane, resp_rdata[(lane*DATA_WIDTH) +: DATA_WIDTH], expected);
                $finish;
            end
        end
    endtask

    initial begin
        rst = 1'b1;
        req_valid = 4'b0;
        req_write = 4'b0;
        req_addr = {LANES*ADDR_WIDTH{1'b0}};
        req_wdata = {LANES*DATA_WIDTH{1'b0}};

        repeat (2) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);

        set_lane(0, 1'b1, 1'b1, 8'd0, 32'h00000010);
        set_lane(1, 1'b1, 1'b1, 8'd1, 32'h00000020);
        set_lane(2, 1'b1, 1'b1, 8'd2, 32'h00000030);
        set_lane(3, 1'b1, 1'b1, 8'd3, 32'h00000040);
        expect_ready(4'b1111);
        $display("MEM write accepted: consecutive bank writes");
        @(posedge clk);
        @(negedge clk);

        req_write = 4'b0;
        expect_ready(4'b1111);
        @(posedge clk);
        #1;
        expect_resp(0, 32'h00000010);
        expect_resp(1, 32'h00000020);
        expect_resp(2, 32'h00000030);
        expect_resp(3, 32'h00000040);

        set_lane(0, 1'b1, 1'b0, 8'd0, 32'b0);
        set_lane(1, 1'b1, 1'b0, 8'd4, 32'b0);
        set_lane(2, 1'b1, 1'b0, 8'd8, 32'b0);
        set_lane(3, 1'b1, 1'b0, 8'd12, 32'b0);
        expect_ready(4'b0011);
        @(posedge clk);
        #1;
        if (resp_valid !== 4'b0011) begin
            $display("FAIL conflict response mask got=%b expected=0011", resp_valid);
            $finish;
        end
        $display("MEM conflict read: addr={12,8,4,0} -> resp_valid=%b expected=0011", resp_valid);

        $display("memory_tb PASS");
        $finish;
    end
endmodule
