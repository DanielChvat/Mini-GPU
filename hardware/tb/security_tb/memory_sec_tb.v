`timescale 1ns/1ps

module memory_sec_tb;
    localparam LANES = 4;
    localparam ADDR_WIDTH = 8;
    localparam DATA_WIDTH = 32;

    reg clk;
    reg rst;
    reg locked;
    reg [ADDR_WIDTH-1:0] protected_base;
    reg [ADDR_WIDTH-1:0] protected_limit;
    reg [LANES-1:0] req_valid;
    reg [LANES-1:0] req_write;
    reg [(LANES*ADDR_WIDTH)-1:0] req_addr;
    reg [(LANES*DATA_WIDTH)-1:0] req_wdata;
    reg [(LANES*4)-1:0] req_wmask;
    wire [LANES-1:0] req_ready;
    wire [LANES-1:0] resp_valid;
    wire [(LANES*DATA_WIDTH)-1:0] resp_rdata;
    wire [LANES-1:0] write_blocked;

    memory_sec #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .BANK_DEPTH(64)
    ) dut (
        .clk(clk),
        .rst(rst),
        .locked(locked),
        .protected_base(protected_base),
        .protected_limit(protected_limit),
        .req_valid(req_valid),
        .req_write(req_write),
        .req_addr(req_addr),
        .req_wdata(req_wdata),
        .req_wmask(req_wmask),
        .req_ready(req_ready),
        .resp_valid(resp_valid),
        .resp_rdata(resp_rdata),
        .write_blocked(write_blocked)
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
            req_wmask[(lane*4) +: 4] = 4'b1111;  // Enable all 4 bytes
        end
    endtask

    task clear_req;
        begin
            req_valid = {LANES{1'b0}};
            req_write = {LANES{1'b0}};
            req_addr = {(LANES*ADDR_WIDTH){1'b0}};
            req_wdata = {(LANES*DATA_WIDTH){1'b0}};
            req_wmask = {(LANES*4){1'b0}};
        end
    endtask

    task expect_ready_blocked;
        input [LANES-1:0] expected_ready;
        input [LANES-1:0] expected_blocked;
        begin
            #1;
            if (req_ready !== expected_ready) begin
                $display("FAIL ready got=%b expected=%b", req_ready, expected_ready);
                $finish;
            end
            if (write_blocked !== expected_blocked) begin
                $display("FAIL write_blocked got=%b expected=%b", write_blocked, expected_blocked);
                $finish;
            end
        end
    endtask

    task write_one;
        input [ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] data;
        input [LANES-1:0] expected_blocked;
        begin
            clear_req();
            set_lane(0, 1'b1, 1'b1, addr, data);
            expect_ready_blocked(4'b0001, expected_blocked);
            @(posedge clk);
            @(negedge clk);
            clear_req();
        end
    endtask

    task read_one_expect;
        input [ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] expected;
        begin
            clear_req();
            set_lane(0, 1'b1, 1'b0, addr, {DATA_WIDTH{1'b0}});
            expect_ready_blocked(4'b0001, 4'b0000);
            @(posedge clk);
            #1;
            if (!resp_valid[0]) begin
                $display("FAIL read addr=%0d missing response", addr);
                $finish;
            end
            if (resp_rdata[0 +: DATA_WIDTH] !== expected) begin
                $display("FAIL read addr=%0d got=0x%08h expected=0x%08h",
                         addr, resp_rdata[0 +: DATA_WIDTH], expected);
                $finish;
            end
            @(negedge clk);
            clear_req();
        end
    endtask

    initial begin
        rst = 1'b1;
        locked = 1'b0;
        protected_base = 8'd8;
        protected_limit = 8'd16;
        clear_req();

        repeat (2) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);

        // Unlocked writes pass through both outside and inside the protected range.
        write_one(8'd4, 32'h11111111, 4'b0000);
        write_one(8'd10, 32'h22222222, 4'b0000);
        read_one_expect(8'd4, 32'h11111111);
        read_one_expect(8'd10, 32'h22222222);

        locked = 1'b1;

        // Locked writes outside the protected range still pass through.
        write_one(8'd20, 32'h33333333, 4'b0000);
        read_one_expect(8'd20, 32'h33333333);

        // Locked writes inside the protected range are accepted, flagged, and suppressed.
        write_one(8'd10, 32'hAAAAAAAA, 4'b0001);
        read_one_expect(8'd10, 32'h22222222);

        // Reads from protected addresses remain allowed while locked.
        read_one_expect(8'd10, 32'h22222222);

        // Mixed lanes suppress only blocked writes.
        locked = 1'b0;
        write_one(8'd9, 32'h99999999, 4'b0000);
        write_one(8'd12, 32'h12121212, 4'b0000);
        locked = 1'b1;
        clear_req();
        set_lane(0, 1'b1, 1'b1, 8'd9,  32'hBBBB0000);
        set_lane(1, 1'b1, 1'b1, 8'd17, 32'hBBBB1111);
        set_lane(2, 1'b1, 1'b1, 8'd12, 32'hBBBB2222);
        set_lane(3, 1'b1, 1'b1, 8'd21, 32'hBBBB3333);
        expect_ready_blocked(4'b1111, 4'b0101);
        @(posedge clk);
        @(negedge clk);
        read_one_expect(8'd9, 32'h99999999);
        read_one_expect(8'd17, 32'hBBBB1111);
        read_one_expect(8'd12, 32'h12121212);
        read_one_expect(8'd21, 32'hBBBB3333);

        // Empty protected regions block nothing.
        protected_base = 8'd16;
        protected_limit = 8'd8;
        write_one(8'd10, 32'hCCCCCCCC, 4'b0000);
        read_one_expect(8'd10, 32'hCCCCCCCC);

        $display("memory_sec_tb PASS");
        $finish;
    end
endmodule
