`timescale 1ns/1ps

module security_gate_tb;

    localparam ADDR_WIDTH = 16;
    localparam DATA_WIDTH = 32;
    localparam NUM_GOLDEN_HASHES = 4;
    localparam BANK_DEPTH = 64;

    // SHA-256 of 8 big-endian 32-bit words [0,1,2,...,7]
    localparam [255:0] HASH_8_WORDS =
        256'hbdb32f8604eafe89ad767fe7fe8ccd29ecc5d0de9b7a3c9d95e3cced553d625a;

    // SHA-256 of 4 big-endian 32-bit words [0,1,2,3]
    localparam [255:0] HASH_4_WORDS =
        256'h3067c72c5e501c31e3feca73f047dc341a956399ec705e0aee9efb17a1553578;

    // =========================================================================
    // Clock / Reset
    // =========================================================================
    reg clk;
    reg rst;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // =========================================================================
    // DUT Signals
    // =========================================================================
    reg         security_reset;
    reg  [3:0]  host_mem_req_valid;
    reg  [3:0]  host_mem_req_write;
    reg  [(4*ADDR_WIDTH)-1:0] host_mem_req_addr;
    reg  [(4*DATA_WIDTH)-1:0] host_mem_req_wdata;
    wire [3:0]  host_mem_req_ready;
    wire [3:0]  host_mem_resp_valid;
    wire [(4*DATA_WIDTH)-1:0] host_mem_resp_rdata;

    reg         validate_triggered;
    reg  [3:0]  validate_model_id;

    wire        mem_write_done;
    wire        mem_write_fail;
    reg         memory_status_consumed;

    wire        weight_locked;
    wire [ADDR_WIDTH-1:0] protected_base;
    wire [ADDR_WIDTH-1:0] protected_limit;

    wire [2:0]  security_state;
    wire        verified;
    wire        security_fault;

    // =========================================================================
    // Memory_sec wires
    // =========================================================================
    wire [3:0]  mem_req_valid;
    wire [3:0]  mem_req_write;
    wire [(4*ADDR_WIDTH)-1:0] mem_req_addr;
    wire [(4*DATA_WIDTH)-1:0] mem_req_wdata;
    wire [(4*4)-1:0] mem_req_wmask;
    wire [3:0]  mem_req_ready;
    wire [3:0]  mem_resp_valid;
    wire [(4*DATA_WIDTH)-1:0] mem_resp_rdata;
    wire [3:0]  write_blocked;

    // =========================================================================
    // DUT: Security Gate
    // =========================================================================
    security_gate #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_GOLDEN_HASHES(NUM_GOLDEN_HASHES)
    ) dut (
        .clk(clk),
        .rst(rst),
        .security_reset(security_reset),
        .host_mem_req_valid(host_mem_req_valid),
        .host_mem_req_write(host_mem_req_write),
        .host_mem_req_addr(host_mem_req_addr),
        .host_mem_req_wdata(host_mem_req_wdata),
        .host_mem_req_ready(host_mem_req_ready),
        .host_mem_resp_valid(host_mem_resp_valid),
        .host_mem_resp_rdata(host_mem_resp_rdata),
        .mem_req_valid(mem_req_valid),
        .mem_req_write(mem_req_write),
        .mem_req_addr(mem_req_addr),
        .mem_req_wdata(mem_req_wdata),
        .mem_req_wmask(mem_req_wmask),
        .mem_req_ready(mem_req_ready),
        .mem_resp_valid(mem_resp_valid),
        .mem_resp_rdata(mem_resp_rdata),
        .validate_triggered(validate_triggered),
        .validate_model_id(validate_model_id),
        .mem_write_done(mem_write_done),
        .mem_write_fail(mem_write_fail),
        .memory_status_consumed(memory_status_consumed),
        .weight_locked(weight_locked),
        .protected_base(protected_base),
        .protected_limit(protected_limit),
        .security_state(security_state),
        .verified(verified),
        .security_fault(security_fault)
    );

    // =========================================================================
    // Memory: memory_sec wrapping memory
    // =========================================================================
    memory_sec #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .BANK_DEPTH(BANK_DEPTH)
    ) mem (
        .clk(clk),
        .rst(rst),
        .locked(weight_locked),
        .protected_base(protected_base),
        .protected_limit(protected_limit),
        .req_valid(mem_req_valid),
        .req_write(mem_req_write),
        .req_addr(mem_req_addr),
        .req_wdata(mem_req_wdata),
        .req_wmask(mem_req_wmask),
        .req_ready(mem_req_ready),
        .resp_valid(mem_resp_valid),
        .resp_rdata(mem_resp_rdata),
        .write_blocked(write_blocked)
    );

    // =========================================================================
    // State Name Helper
    // =========================================================================
    reg [55:0] state_name;
    always @* begin
        case (security_state)
            3'd0: state_name = "IDLE   ";
            3'd1: state_name = "LOAD   ";
            3'd2: state_name = "FINAL  ";
            3'd3: state_name = "VERIFY ";
            3'd4: state_name = "LOCK   ";
            3'd5: state_name = "EXECUTE";
            3'd6: state_name = "ERROR  ";
            default: state_name = "UNKNOWN";
        endcase
    end

    // =========================================================================
    // Test Infrastructure
    // =========================================================================
    integer errors;
    integer test_num;

    task clear_inputs;
        begin
            host_mem_req_valid  <= 4'b0000;
            host_mem_req_write  <= 4'b0000;
            host_mem_req_addr   <= {(4*ADDR_WIDTH){1'b0}};
            host_mem_req_wdata  <= {(4*DATA_WIDTH){1'b0}};
            validate_triggered  <= 1'b0;
            validate_model_id   <= 4'b0;
            memory_status_consumed <= 1'b0;
            security_reset      <= 1'b0;
        end
    endtask

    task write_word;
        input [ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] data;
        begin
            @(negedge clk);
            host_mem_req_valid[0] <= 1'b1;
            host_mem_req_write[0] <= 1'b1;
            host_mem_req_addr[0 +: ADDR_WIDTH] <= addr;
            host_mem_req_wdata[0 +: DATA_WIDTH] <= data;

            // One-shot pulse like the comm_controller
            @(negedge clk);
            host_mem_req_valid[0] <= 1'b0;
            host_mem_req_write[0] <= 1'b0;

            // Wait for memory to process
            @(negedge clk);
        end
    endtask

    task read_word;
        input [ADDR_WIDTH-1:0] addr;
        output [DATA_WIDTH-1:0] data;
        begin
            @(negedge clk);
            host_mem_req_valid[0] <= 1'b1;
            host_mem_req_write[0] <= 1'b0;
            host_mem_req_addr[0 +: ADDR_WIDTH] <= addr;

            @(negedge clk);
            while (!host_mem_req_ready[0]) begin
                @(negedge clk);
            end

            host_mem_req_valid[0] <= 1'b0;

            while (!host_mem_resp_valid[0]) begin
                @(negedge clk);
            end
            data = host_mem_resp_rdata[0 +: DATA_WIDTH];
        end
    endtask

    task trigger_validate;
        input [3:0] model_id;
        begin
            @(negedge clk);
            validate_triggered <= 1'b1;
            validate_model_id  <= model_id;
            @(negedge clk);
            validate_triggered <= 1'b0;
        end
    endtask

    task wait_for_state;
        input [2:0] expected_state;
        integer timeout;
        begin
            timeout = 0;
            while (security_state !== expected_state && timeout < 10000) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (timeout >= 10000) begin
                $display("FAIL: Timeout waiting for state %0d, current state=%0d (%s)",
                         expected_state, security_state, state_name);
                errors = errors + 1;
            end
        end
    endtask

    task pulse_security_reset;
        begin
            @(negedge clk);
            security_reset <= 1'b1;
            @(negedge clk);
            security_reset <= 1'b0;
            @(negedge clk);
        end
    endtask

    task check;
        input [255:0] label;
        input integer actual;
        input integer expected;
        begin
            if (actual !== expected) begin
                $display("  FAIL %0s: got %0d, expected %0d", label, actual, expected);
                errors = errors + 1;
            end
        end
    endtask

    // =========================================================================
    // Load weights helper: write sequential words starting at address 0
    // =========================================================================
    task load_weights;
        input integer count;
        integer i;
        begin
            for (i = 0; i < count; i = i + 1) begin
                write_word(i[ADDR_WIDTH-1:0], i[DATA_WIDTH-1:0]);
            end
        end
    endtask

    // =========================================================================
    // Main Test Sequence
    // =========================================================================
    reg [DATA_WIDTH-1:0] read_data;

    initial begin
        $dumpfile("security_gate_tb.vcd");
        $dumpvars(0, security_gate_tb);

        errors = 0;
        rst = 1'b1;
        clear_inputs();

        // Load golden hash for model 0 = SHA-256 of 8 words [0..7]
        dut.golden_hash[0] = HASH_8_WORDS;
        // Load golden hash for model 1 = SHA-256 of 4 words [0..3]
        dut.golden_hash[1] = HASH_4_WORDS;

        #20;
        rst = 1'b0;

        // Wait for SHA pre-start to complete (sha_needs_start → sha_start pulse → SHA in S_COLLECT)
        repeat (3) @(negedge clk);

        // ==============================================================
        // Test 1: Happy path (8 weights, model 0)
        // ==============================================================
        test_num = 1;
        $display("\n=== Test %0d: Happy path (8 weights, model 0) ===", test_num);

        check("initial state", security_state, 3'd0);
        check("not locked", weight_locked, 0);
        check("not verified", verified, 0);
        check("no fault", security_fault, 0);

        load_weights(8);

        check("state after load", security_state, 3'd1);
        check("weight_count", dut.weight_count_reg, 8);

        trigger_validate(4'd0);

        wait_for_state(3'd5);

        check("state = EXECUTE", security_state, 3'd5);
        check("locked", weight_locked, 1);
        check("verified", verified, 1);
        check("no fault", security_fault, 0);
        check("protected_limit", protected_limit, 16'd8);

        // Verify weights readable
        read_word(16'd0, read_data);
        check("weight[0]", read_data, 32'd0);
        read_word(16'd3, read_data);
        check("weight[3]", read_data, 32'd3);
        read_word(16'd7, read_data);
        check("weight[7]", read_data, 32'd7);

        // Verify write to protected region is blocked by memory_sec
        write_word(16'd2, 32'hDEADBEEF);
        @(negedge clk);
        read_word(16'd2, read_data);
        check("protected write blocked", read_data, 32'd2);

        // Verify write outside protected region succeeds
        write_word(16'd10, 32'hCAFEBABE);
        @(negedge clk);
        read_word(16'd10, read_data);
        check("unprotected write ok", read_data, 32'hCAFEBABE);

        $display("  Test %0d: %s", test_num, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 2: Security reset from EXECUTE → IDLE → new load
        // ==============================================================
        test_num = 2;
        $display("\n=== Test %0d: Security reset from EXECUTE ===", test_num);

        pulse_security_reset();

        check("state after reset", security_state, 3'd0);
        check("unlocked after reset", weight_locked, 0);
        check("not verified after reset", verified, 0);
        check("no fault after reset", security_fault, 0);

        // Write to previously protected region should work now
        write_word(16'd0, 32'hAAAAAAAA);
        @(negedge clk);
        read_word(16'd0, read_data);
        check("write after unlock", read_data, 32'hAAAAAAAA);

        $display("  Test %0d: %s", test_num, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 3: Hash mismatch → ERROR
        // ==============================================================
        test_num = 3;
        $display("\n=== Test %0d: Hash mismatch (wrong model_id) ===", test_num);

        // Load 8 weights (hash matches model 0) but validate with model 1
        load_weights(8);
        trigger_validate(4'd1);

        wait_for_state(3'd6);
        @(negedge clk);

        check("state = ERROR", security_state, 3'd6);
        check("fault asserted", security_fault, 1);
        check("not verified", verified, 0);
        check("not locked", weight_locked, 0);
        check("mem_write_fail", mem_write_fail, 1);

        // Security reset recovers from error
        pulse_security_reset();
        check("state after reset", security_state, 3'd0);
        check("fault cleared", security_fault, 0);

        $display("  Test %0d: %s", test_num, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 4: Second successful load (4 weights, model 1)
        // ==============================================================
        test_num = 4;
        $display("\n=== Test %0d: Second load (4 weights, model 1) ===", test_num);

        load_weights(4);
        trigger_validate(4'd1);

        wait_for_state(3'd5);

        check("state = EXECUTE", security_state, 3'd5);
        check("locked", weight_locked, 1);
        check("verified", verified, 1);
        check("protected_limit", protected_limit, 16'd4);

        // Weight at addr 0 should be 0 (we loaded 0,1,2,3)
        read_word(16'd0, read_data);
        check("weight[0]", read_data, 32'd0);

        // Write to addr 5 (outside protected region) should work
        write_word(16'd5, 32'h12345678);
        @(negedge clk);
        read_word(16'd5, read_data);
        check("unprotected write ok", read_data, 32'h12345678);

        pulse_security_reset();

        $display("  Test %0d: %s", test_num, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Summary
        // ==============================================================
        $display("\n==========================================");
        if (errors == 0) begin
            $display("security_gate_tb: ALL TESTS PASSED");
        end else begin
            $display("security_gate_tb: %0d ERRORS", errors);
        end
        $display("==========================================\n");
        $finish;
    end

    // Timeout
    initial begin
        #5_000_000;
        $display("FAIL: Global timeout");
        $finish;
    end

endmodule
