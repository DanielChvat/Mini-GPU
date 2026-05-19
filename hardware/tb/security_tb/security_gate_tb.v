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
    localparam [(NUM_GOLDEN_HASHES*256)-1:0] GOLDEN_HASHES = {
        256'h0,
        256'h0,
        HASH_4_WORDS,
        HASH_8_WORDS
    };

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
    // Host Bus Signals (drives bus_controller host port)
    // =========================================================================
    reg  [3:0]  host_mem_req_valid;
    reg  [3:0]  host_mem_req_write;
    reg  [(4*ADDR_WIDTH)-1:0] host_mem_req_addr;
    reg  [(4*DATA_WIDTH)-1:0] host_mem_req_wdata;
    reg  [(4*4)-1:0] host_mem_req_wmask;
    wire [3:0]  host_mem_req_ready;
    wire [3:0]  host_mem_resp_valid;
    wire [(4*DATA_WIDTH)-1:0] host_mem_resp_rdata;
    wire        host_mem_write_done;
    wire        host_mem_write_fail;
    reg         host_memory_status_consumed;

    // =========================================================================
    // Core Bus Signals (drives bus_controller core port)
    // =========================================================================
    reg  [3:0]  core_mem_req_valid;
    reg  [3:0]  core_mem_req_write;
    reg  [(4*ADDR_WIDTH)-1:0] core_mem_req_addr;
    reg  [(4*DATA_WIDTH)-1:0] core_mem_req_wdata;
    reg  [(4*4)-1:0] core_mem_req_wmask;
    wire [3:0]  core_mem_req_ready;
    wire [3:0]  core_mem_resp_valid;
    wire [(4*DATA_WIDTH)-1:0] core_mem_resp_rdata;
    wire        core_mem_write_done;
    wire        core_mem_write_fail;
    reg         core_memory_status_consumed;

    // =========================================================================
    // Bus Controller → Security Gate wires
    // =========================================================================
    wire [3:0]  merged_mem_req_valid;
    wire [3:0]  merged_mem_req_write;
    wire [(4*ADDR_WIDTH)-1:0] merged_mem_req_addr;
    wire [(4*DATA_WIDTH)-1:0] merged_mem_req_wdata;
    wire [(4*4)-1:0] merged_mem_req_wmask;
    wire [3:0]  merged_mem_req_ready;
    wire [3:0]  merged_mem_resp_valid;
    wire [(4*DATA_WIDTH)-1:0] merged_mem_resp_rdata;
    wire        merged_mem_write_done;
    wire        merged_mem_write_fail;
    wire        merged_memory_status_consumed;
    wire [1:0]  memory_request_id;

    // =========================================================================
    // Security Gate → Memory_sec wires
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
    // Security Gate control/status
    // =========================================================================
    reg         security_reset;
    reg         validate_triggered;
    reg  [3:0]  validate_model_id;

    wire        mem_write_done_unused;
    wire        mem_write_fail_unused;
    wire        weight_locked;
    wire [ADDR_WIDTH-1:0] protected_base;
    wire [ADDR_WIDTH-1:0] protected_limit;
    wire [2:0]  security_state;
    wire        verified;
    wire        security_fault;

    // =========================================================================
    // DUT: Bus Controller
    // =========================================================================
    bus_controller #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) bus_ctrl (
        .clk(clk),
        .rst(rst),
        .host_mem_req_valid(host_mem_req_valid),
        .host_mem_req_write(host_mem_req_write),
        .host_mem_req_addr(host_mem_req_addr),
        .host_mem_req_wdata(host_mem_req_wdata),
        .host_mem_req_wmask(host_mem_req_wmask),
        .host_mem_req_ready(host_mem_req_ready),
        .host_mem_resp_valid(host_mem_resp_valid),
        .host_mem_resp_rdata(host_mem_resp_rdata),
        .host_mem_write_done(host_mem_write_done),
        .host_mem_write_fail(host_mem_write_fail),
        .host_memory_status_consumed(host_memory_status_consumed),
        .core_mem_req_valid(core_mem_req_valid),
        .core_mem_req_write(core_mem_req_write),
        .core_mem_req_addr(core_mem_req_addr),
        .core_mem_req_wdata(core_mem_req_wdata),
        .core_mem_req_wmask(core_mem_req_wmask),
        .core_mem_req_ready(core_mem_req_ready),
        .core_mem_resp_valid(core_mem_resp_valid),
        .core_mem_resp_rdata(core_mem_resp_rdata),
        .core_mem_write_done(core_mem_write_done),
        .core_mem_write_fail(core_mem_write_fail),
        .core_memory_status_consumed(core_memory_status_consumed),
        .merged_mem_req_valid(merged_mem_req_valid),
        .merged_mem_req_write(merged_mem_req_write),
        .merged_mem_req_addr(merged_mem_req_addr),
        .merged_mem_req_wdata(merged_mem_req_wdata),
        .merged_mem_req_wmask(merged_mem_req_wmask),
        .merged_mem_req_ready(merged_mem_req_ready),
        .merged_mem_resp_valid(merged_mem_resp_valid),
        .merged_mem_resp_rdata(merged_mem_resp_rdata),
        .merged_mem_write_done(merged_mem_write_done),
        .merged_mem_write_fail(merged_mem_write_fail),
        .merged_memory_status_consumed(merged_memory_status_consumed),
        .memory_request_id(memory_request_id)
    );

    // =========================================================================
    // DUT: Security Gate
    // =========================================================================
    security_gate #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_GOLDEN_HASHES(NUM_GOLDEN_HASHES),
        .GOLDEN_HASHES(GOLDEN_HASHES)
    ) dut (
        .clk(clk),
        .rst(rst),
        .security_reset(security_reset),
        .merged_mem_req_valid(merged_mem_req_valid),
        .merged_mem_req_write(merged_mem_req_write),
        .merged_mem_req_addr(merged_mem_req_addr),
        .merged_mem_req_wdata(merged_mem_req_wdata),
        .merged_mem_req_wmask(merged_mem_req_wmask),
        .merged_mem_req_ready(merged_mem_req_ready),
        .merged_mem_resp_valid(merged_mem_resp_valid),
        .merged_mem_resp_rdata(merged_mem_resp_rdata),
        .memory_request_id(memory_request_id),
        .mem_req_valid(mem_req_valid),
        .mem_req_write(mem_req_write),
        .mem_req_addr(mem_req_addr),
        .mem_req_wdata(mem_req_wdata),
        .mem_req_wmask(mem_req_wmask),
        .mem_req_ready(mem_req_ready),
        .mem_resp_valid(mem_resp_valid),
        .mem_resp_rdata(mem_resp_rdata),
        .write_blocked(write_blocked),
        .validate_triggered(validate_triggered),
        .validate_model_id(validate_model_id),
        .mem_write_done(merged_mem_write_done),
        .mem_write_fail(merged_mem_write_fail),
        .memory_status_consumed(merged_memory_status_consumed),
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
            host_mem_req_wmask  <= {4{4'b1111}};
            host_memory_status_consumed <= 1'b0;
            core_mem_req_valid  <= 4'b0000;
            core_mem_req_write  <= 4'b0000;
            core_mem_req_addr   <= {(4*ADDR_WIDTH){1'b0}};
            core_mem_req_wdata  <= {(4*DATA_WIDTH){1'b0}};
            core_mem_req_wmask  <= {4{4'b1111}};
            core_memory_status_consumed <= 1'b0;
            validate_triggered  <= 1'b0;
            validate_model_id   <= 4'b0;
            security_reset      <= 1'b0;
        end
    endtask

    // Host write (one-shot pulse on lane 0)
    task host_write_word;
        input [ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] data;
        begin
            @(negedge clk);
            host_mem_req_valid[0] <= 1'b1;
            host_mem_req_write[0] <= 1'b1;
            host_mem_req_addr[0 +: ADDR_WIDTH] <= addr;
            host_mem_req_wdata[0 +: DATA_WIDTH] <= data;
            @(negedge clk);
            host_mem_req_valid[0] <= 1'b0;
            host_mem_req_write[0] <= 1'b0;
            @(negedge clk);
        end
    endtask

    // Host read (lane 0)
    task host_read_word;
        input [ADDR_WIDTH-1:0] addr;
        output [DATA_WIDTH-1:0] data;
        begin
            @(negedge clk);
            host_mem_req_valid[0] <= 1'b1;
            host_mem_req_write[0] <= 1'b0;
            host_mem_req_addr[0 +: ADDR_WIDTH] <= addr;
            @(negedge clk);
            while (!host_mem_req_ready[0]) @(negedge clk);
            host_mem_req_valid[0] <= 1'b0;
            while (!host_mem_resp_valid[0]) @(negedge clk);
            data = host_mem_resp_rdata[0 +: DATA_WIDTH];
        end
    endtask

    // Core write (one-shot pulse on lane 0)
    task core_write_word;
        input [ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] data;
        begin
            @(negedge clk);
            core_mem_req_valid[0] <= 1'b1;
            core_mem_req_write[0] <= 1'b1;
            core_mem_req_addr[0 +: ADDR_WIDTH] <= addr;
            core_mem_req_wdata[0 +: DATA_WIDTH] <= data;
            @(negedge clk);
            core_mem_req_valid[0] <= 1'b0;
            core_mem_req_write[0] <= 1'b0;
            @(negedge clk);
        end
    endtask

    // Core read (lane 0)
    task core_read_word;
        input [ADDR_WIDTH-1:0] addr;
        output [DATA_WIDTH-1:0] data;
        begin
            @(negedge clk);
            core_mem_req_valid[0] <= 1'b1;
            core_mem_req_write[0] <= 1'b0;
            core_mem_req_addr[0 +: ADDR_WIDTH] <= addr;
            @(negedge clk);
            while (!core_mem_req_ready[0]) @(negedge clk);
            core_mem_req_valid[0] <= 1'b0;
            while (!core_mem_resp_valid[0]) @(negedge clk);
            data = core_mem_resp_rdata[0 +: DATA_WIDTH];
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
                $display("  FAIL %0s: got 0x%08x, expected 0x%08x", label, actual, expected);
                errors = errors + 1;
            end
        end
    endtask

    task load_weights_host;
        input integer count;
        integer i;
        begin
            for (i = 0; i < count; i = i + 1) begin
                host_write_word(i[ADDR_WIDTH-1:0], i[DATA_WIDTH-1:0]);
            end
        end
    endtask

    // =========================================================================
    // Main Test Sequence
    // =========================================================================
    reg [DATA_WIDTH-1:0] read_data;

    initial begin
        $dumpfile("./tmp/security_gate_tb.vcd");
        $dumpvars(0, security_gate_tb);

        errors = 0;
        rst = 1'b1;
        clear_inputs();

        #20;
        rst = 1'b0;

        // Wait for SHA pre-start
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

        load_weights_host(8);

        check("state after load", security_state, 3'd1);
        check("weight_count", dut.weight_count_reg, 8);

        trigger_validate(4'd0);
        wait_for_state(3'd5);

        check("state = EXECUTE", security_state, 3'd5);
        check("locked", weight_locked, 1);
        check("verified", verified, 1);
        check("no fault", security_fault, 0);
        check("protected_limit", protected_limit, 16'd8);

        // Verify write to protected region is blocked by memory_sec
        host_write_word(16'd2, 32'hDEADBEEF);
        @(negedge clk);

        // Verify write outside protected region succeeds
        host_write_word(16'd10, 32'hCAFEBABE);
        @(negedge clk);
        host_read_word(16'd10, read_data);
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
        host_write_word(16'd0, 32'hAAAAAAAA);
        @(negedge clk);
        host_read_word(16'd0, read_data);
        check("write after unlock", read_data, 32'hAAAAAAAA);

        $display("  Test %0d: %s", test_num, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 3: Hash mismatch → ERROR
        // ==============================================================
        test_num = 3;
        $display("\n=== Test %0d: Hash mismatch (wrong model_id) ===", test_num);

        load_weights_host(8);
        trigger_validate(4'd1);

        wait_for_state(3'd6);
        @(negedge clk);

        check("state = ERROR", security_state, 3'd6);
        check("fault asserted", security_fault, 1);
        check("not verified", verified, 0);
        check("not locked", weight_locked, 0);
        check("mem_write_fail", host_mem_write_fail, 1);

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

        load_weights_host(4);
        trigger_validate(4'd1);

        wait_for_state(3'd5);

        check("state = EXECUTE", security_state, 3'd5);
        check("locked", weight_locked, 1);
        check("verified", verified, 1);
        check("protected_limit", protected_limit, 16'd4);

        // Write to addr 5 (outside protected region) should work
        host_write_word(16'd5, 32'h12345678);
        @(negedge clk);
        host_read_word(16'd5, read_data);
        check("unprotected write ok", read_data, 32'h12345678);

        pulse_security_reset();

        $display("  Test %0d: %s", test_num, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 5: Host read protection in EXECUTE
        // ==============================================================
        test_num = 5;
        $display("\n=== Test %0d: Host read protection in EXECUTE ===", test_num);

        load_weights_host(8);
        trigger_validate(4'd0);
        wait_for_state(3'd5);

        // Host reads protected region → should get 0xDEADDEAD
        host_read_word(16'd0, read_data);
        check("host read protected[0]", read_data, 32'hDEADDEAD);
        host_read_word(16'd3, read_data);
        check("host read protected[3]", read_data, 32'hDEADDEAD);
        host_read_word(16'd7, read_data);
        check("host read protected[7]", read_data, 32'hDEADDEAD);

        // Host reads scratch (outside protected) → should succeed normally
        host_read_word(16'd10, read_data);
        check("host read scratch", read_data, 32'hCAFEBABE);

        $display("  Test %0d: %s", test_num, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 6: Core read in EXECUTE (allowed for protected region)
        // ==============================================================
        test_num = 6;
        $display("\n=== Test %0d: Core read protected in EXECUTE ===", test_num);

        // Core reads protected region → should get actual weight data
        core_read_word(16'd0, read_data);
        check("core read weight[0]", read_data, 32'd0);
        core_read_word(16'd3, read_data);
        check("core read weight[3]", read_data, 32'd3);
        core_read_word(16'd7, read_data);
        check("core read weight[7]", read_data, 32'd7);

        $display("  Test %0d: %s", test_num, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 7: Core write blocked in EXECUTE (protected region)
        // ==============================================================
        test_num = 7;
        $display("\n=== Test %0d: Core write blocked in EXECUTE ===", test_num);

        // Core tries to write to protected region
        core_write_word(16'd2, 32'hBADBAD00);
        @(negedge clk);

        // Read back via core (should still be original value)
        core_read_word(16'd2, read_data);
        check("core write blocked", read_data, 32'd2);

        $display("  Test %0d: %s", test_num, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 8: ERROR state total lockdown
        // ==============================================================
        test_num = 8;
        $display("\n=== Test %0d: ERROR state total lockdown ===", test_num);

        pulse_security_reset();

        // Load 8 weights, validate with wrong model → ERROR
        load_weights_host(8);
        trigger_validate(4'd1);
        wait_for_state(3'd6);
        @(negedge clk);

        check("state = ERROR", security_state, 3'd6);

        // Host reads protected → 0xDEADDEAD
        host_read_word(16'd0, read_data);
        check("host read protected in ERROR", read_data, 32'hDEADDEAD);

        // Core reads protected → 0xDEADDEAD
        core_read_word(16'd0, read_data);
        check("core read protected in ERROR", read_data, 32'hDEADDEAD);

        // Host writes are blocked (scratch region)
        host_write_word(16'd20, 32'hFEEDFACE);
        @(negedge clk);

        // Scratch reads still work
        host_read_word(16'd10, read_data);
        check("scratch read still works in ERROR", read_data, 32'hCAFEBABE);

        // Recovery
        pulse_security_reset();
        check("recovered from ERROR", security_state, 3'd0);

        $display("  Test %0d: %s", test_num, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 9: Core scratch access in EXECUTE
        // ==============================================================
        test_num = 9;
        $display("\n=== Test %0d: Core scratch access in EXECUTE ===", test_num);

        load_weights_host(4);
        trigger_validate(4'd1);
        wait_for_state(3'd5);

        // Core writes to scratch region (outside protected [0,4))
        core_write_word(16'd30, 32'hC0FFEE00);
        @(negedge clk);

        // Core reads scratch back
        core_read_word(16'd30, read_data);
        check("core scratch write+read", read_data, 32'hC0FFEE00);

        // Core reads another scratch location (addr 10 was written to 0xCAFEBABE in test 1)
        core_read_word(16'd10, read_data);
        check("core scratch read", read_data, 32'hCAFEBABE);

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
