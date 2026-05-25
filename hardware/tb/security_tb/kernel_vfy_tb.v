`timescale 1ns/1ps

module kernel_vfy_tb;

    localparam ADDR_WIDTH       = 16;
    localparam PROG_ADDR_WIDTH  = 12;
    localparam DATA_WIDTH       = 32;
    localparam NUM_GOLDEN_HASHES = 128;
    localparam KERNEL_ID_WIDTH  = 7;

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
    reg         cc_prog_we;
    reg  [ADDR_WIDTH-1:0] cc_prog_addr;
    reg  [31:0] cc_prog_wdata;
    reg         cc_launch_valid;
    reg         validate_triggered;
    reg  [KERNEL_ID_WIDTH-1:0] validate_kernel_id;
    reg         core_busy;

    wire        gpu_prog_we;
    wire [PROG_ADDR_WIDTH-1:0] gpu_prog_addr;
    wire [31:0] gpu_prog_wdata;
    wire        gpu_launch_valid;
    wire        prog_write_blocked;
    wire        validate_done;
    wire        validate_fail;
    wire [2:0]  kernel_state;
    wire        kernel_verified;
    wire        kernel_fault;

    // =========================================================================
    // DUT Instantiation
    // =========================================================================
    kernel_vfy #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .PROG_ADDR_WIDTH(PROG_ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_GOLDEN_HASHES(NUM_GOLDEN_HASHES),
        .KERNEL_ID_WIDTH(KERNEL_ID_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .security_reset(security_reset),
        .cc_prog_we(cc_prog_we),
        .cc_prog_addr(cc_prog_addr),
        .cc_prog_wdata(cc_prog_wdata),
        .gpu_prog_we(gpu_prog_we),
        .gpu_prog_addr(gpu_prog_addr),
        .gpu_prog_wdata(gpu_prog_wdata),
        .cc_launch_valid(cc_launch_valid),
        .gpu_launch_valid(gpu_launch_valid),
        .validate_triggered(validate_triggered),
        .validate_kernel_id(validate_kernel_id),
        .core_busy(core_busy),
        .prog_write_blocked(prog_write_blocked),
        .validate_done(validate_done),
        .validate_fail(validate_fail),
        .kernel_state(kernel_state),
        .kernel_verified(kernel_verified),
        .kernel_fault(kernel_fault)
    );

    // =========================================================================
    // Override golden hashes for testing
    // =========================================================================
    initial begin
        dut.hash_store.hash_mem[0] = HASH_8_WORDS;
        dut.hash_store.hash_mem[1] = HASH_4_WORDS;
        dut.hash_store.hash_mem[2] = 256'hDEADBEEF;
        dut.hash_store.hash_mem[3] = 256'h0;
    end

    // =========================================================================
    // Test Counters
    // =========================================================================
    integer test_num;
    integer pass_count;
    integer fail_count;

    // =========================================================================
    // Helper Tasks
    // =========================================================================

    task reset_dut;
    begin
        rst = 1'b1;
        security_reset = 1'b0;
        cc_prog_we = 1'b0;
        cc_prog_addr = {ADDR_WIDTH{1'b0}};
        cc_prog_wdata = 32'b0;
        cc_launch_valid = 1'b0;
        validate_triggered = 1'b0;
        validate_kernel_id = {KERNEL_ID_WIDTH{1'b0}};
        core_busy = 1'b0;
        @(posedge clk);
        @(posedge clk);
        rst = 1'b0;
        repeat (150000)@(posedge clk);
        @(posedge clk);
        #1;
    end
    endtask

    task write_prog_word(input [ADDR_WIDTH-1:0] addr, input [31:0] data);
    begin
        @(posedge clk);
        cc_prog_we    <= 1'b1;
        cc_prog_addr  <= addr;
        cc_prog_wdata <= data;
        @(posedge clk);
        cc_prog_we    <= 1'b0;
        #1; // let NBAs settle
    end
    endtask

    task trigger_validate(input [KERNEL_ID_WIDTH-1:0] kid);
    begin
        @(posedge clk);
        validate_triggered  <= 1'b1;
        validate_kernel_id  <= kid;
        @(posedge clk);
        validate_triggered  <= 1'b0;
        #1; // let NBAs settle
    end
    endtask

    task wait_for_state(input [2:0] target_state, input integer timeout);
        integer i;
    begin
        for (i = 0; i < timeout; i = i + 1) begin
            if (kernel_state == target_state) begin
                i = timeout;
            end else begin
                @(posedge clk);
                #1;
            end
        end
    end
    endtask

    task check(input [255:0] label, input condition);
    begin
        if (condition) begin
            $display("  [PASS] %0s", label);
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] %0s", label);
            fail_count = fail_count + 1;
        end
    end
    endtask

    // =========================================================================
    // Main Test
    // =========================================================================
    initial begin
        $dumpfile("kernel_vfy_tb.vcd");
        $dumpvars(0, kernel_vfy_tb);

        pass_count = 0;
        fail_count = 0;

        // ==============================================================
        // TEST 1: Happy path — IDLE → LOAD → FINALIZE → VERIFY → LOCK → EXECUTE
        // ==============================================================
        test_num = 1;
        $display("\n=== TEST %0d: Happy path (8 words, hash slot 0) ===", test_num);
        reset_dut;

        check("Initial state is IDLE", kernel_state == 3'd0);
        check("Not verified", kernel_verified == 1'b0);

        // Write 8 words [0..7]
        write_prog_word(16'h0000, 32'h00000000);
        check("State transitions to LOAD", kernel_state == 3'd1);

        write_prog_word(16'h0001, 32'h00000001);
        write_prog_word(16'h0002, 32'h00000002);
        write_prog_word(16'h0003, 32'h00000003);
        write_prog_word(16'h0004, 32'h00000004);
        write_prog_word(16'h0005, 32'h00000005);
        write_prog_word(16'h0006, 32'h00000006);
        write_prog_word(16'h0007, 32'h00000007);

        // Trigger validation with kernel_id=0
        trigger_validate(6'd0);
        check("State is FINALIZE", kernel_state == 3'd2);

        // Wait for SHA to finish and verification to complete
        wait_for_state(3'd5, 500); // S_EXECUTE
        check("Reached EXECUTE state", kernel_state == 3'd5);
        check("kernel_verified is set", kernel_verified == 1'b1);
        check("No fault", kernel_fault == 1'b0);

        // Test launch gating — should pass through in EXECUTE with verified
        @(posedge clk);
        cc_launch_valid <= 1'b1;
        #1;
        check("Launch passes through when verified", gpu_launch_valid == 1'b1);
        cc_launch_valid <= 1'b0;
        @(posedge clk);

        // ==============================================================
        // TEST 2: Hash mismatch → ERROR
        // ==============================================================
        test_num = 2;
        $display("\n=== TEST %0d: Hash mismatch ===", test_num);
        reset_dut;

        // Write 8 words but validate against slot 2 (wrong hash)
        write_prog_word(16'h0000, 32'h00000000);
        write_prog_word(16'h0001, 32'h00000001);
        write_prog_word(16'h0002, 32'h00000002);
        write_prog_word(16'h0003, 32'h00000003);
        write_prog_word(16'h0004, 32'h00000004);
        write_prog_word(16'h0005, 32'h00000005);
        write_prog_word(16'h0006, 32'h00000006);
        write_prog_word(16'h0007, 32'h00000007);

        trigger_validate(6'd2); // slot 2 has 0xDEADBEEF, won't match

        wait_for_state(3'd6, 500); // S_ERROR
        check("Reached ERROR state on mismatch", kernel_state == 3'd6);
        check("kernel_fault is set", kernel_fault == 1'b1);
        check("Not verified", kernel_verified == 1'b0);

        // ==============================================================
        // TEST 3: Launch gating — blocked before verification
        // ==============================================================
        test_num = 3;
        $display("\n=== TEST %0d: Launch gating ===", test_num);
        reset_dut;

        // In IDLE, launch should be blocked
        cc_launch_valid <= 1'b1;
        @(posedge clk);
        #1;
        check("Launch blocked in IDLE", gpu_launch_valid == 1'b0);

        // Start loading
        write_prog_word(16'h0000, 32'hAAAA);
        check("Launch blocked in LOAD", gpu_launch_valid == 1'b0);
        cc_launch_valid <= 1'b0;
        @(posedge clk);

        // ==============================================================
        // TEST 4: EXECUTE→LOAD transition (core idle)
        // ==============================================================
        test_num = 4;
        $display("\n=== TEST %0d: EXECUTE->LOAD transition (core idle) ===", test_num);
        reset_dut;

        // First: load and verify 4-word kernel in slot 1
        write_prog_word(16'h0000, 32'h00000000);
        write_prog_word(16'h0001, 32'h00000001);
        write_prog_word(16'h0002, 32'h00000002);
        write_prog_word(16'h0003, 32'h00000003);

        trigger_validate(6'd1);
        wait_for_state(3'd5, 500); // S_EXECUTE
        check("First kernel verified", kernel_verified == 1'b1);

        // Now send a new WRITE_PROGRAM while core is idle
        core_busy <= 1'b0;
        write_prog_word(16'h0000, 32'hAABBCCDD);
        check("Transitioned to LOAD", kernel_state == 3'd1);
        check("Verified cleared", kernel_verified == 1'b0);

        // Continue loading a new 8-word kernel
        write_prog_word(16'h0001, 32'h00000001);
        write_prog_word(16'h0002, 32'h00000002);
        write_prog_word(16'h0003, 32'h00000003);
        write_prog_word(16'h0004, 32'h00000004);
        write_prog_word(16'h0005, 32'h00000005);
        write_prog_word(16'h0006, 32'h00000006);
        write_prog_word(16'h0007, 32'h00000007);

        // This won't match hash slot 0 because first word was 0xAABBCCDD
        trigger_validate(6'd0);
        wait_for_state(3'd6, 500); // S_ERROR
        check("Mismatch on reloaded kernel -> ERROR", kernel_state == 3'd6);

        // ==============================================================
        // TEST 5: EXECUTE→LOAD rejection (core busy)
        // ==============================================================
        test_num = 5;
        $display("\n=== TEST %0d: EXECUTE->LOAD rejection (core busy) ===", test_num);
        reset_dut;

        // Load and verify 8-word kernel in slot 0
        write_prog_word(16'h0000, 32'h00000000);
        write_prog_word(16'h0001, 32'h00000001);
        write_prog_word(16'h0002, 32'h00000002);
        write_prog_word(16'h0003, 32'h00000003);
        write_prog_word(16'h0004, 32'h00000004);
        write_prog_word(16'h0005, 32'h00000005);
        write_prog_word(16'h0006, 32'h00000006);
        write_prog_word(16'h0007, 32'h00000007);

        trigger_validate(6'd0);
        wait_for_state(3'd5, 500); // S_EXECUTE
        check("Reached EXECUTE", kernel_state == 3'd5);

        // Set core_busy and attempt write
        core_busy <= 1'b1;
        @(posedge clk);
        cc_prog_we <= 1'b1;
        cc_prog_addr <= 16'h0000;
        cc_prog_wdata <= 32'hDEAD;
        #1;
        check("prog_write_blocked is high", prog_write_blocked == 1'b1);
        check("gpu_prog_we is blocked", gpu_prog_we == 1'b0);
        @(posedge clk);
        #1;
        check("Still in EXECUTE", kernel_state == 3'd5);
        check("Still verified", kernel_verified == 1'b1);
        cc_prog_we <= 1'b0;
        core_busy <= 1'b0;
        @(posedge clk);

        // ==============================================================
        // TEST 6: Security reset from ERROR → IDLE
        // ==============================================================
        test_num = 6;
        $display("\n=== TEST %0d: Security reset from ERROR ===", test_num);
        reset_dut;

        // Create an error state via hash mismatch
        write_prog_word(16'h0000, 32'hBAD00000);
        trigger_validate(6'd0); // won't match
        wait_for_state(3'd6, 500); // S_ERROR
        check("In ERROR state", kernel_state == 3'd6);
        check("Writes blocked in ERROR", prog_write_blocked == 1'b1);

        // Security reset
        @(posedge clk);
        security_reset <= 1'b1;
        @(posedge clk);
        security_reset <= 1'b0;
        repeat (150000)@(posedge clk);
        @(posedge clk);
        #1;
        check("Back to IDLE after security reset", kernel_state == 3'd0);
        check("Fault cleared", kernel_fault == 1'b0);
        check("Verified cleared", kernel_verified == 1'b0);

        // ==============================================================
        // TEST 7: validate_fail pulse in ERROR state
        // ==============================================================
        test_num = 7;
        $display("\n=== TEST %0d: validate_fail in ERROR state ===", test_num);
        reset_dut;

        write_prog_word(16'h0000, 32'hBAD00000);
        trigger_validate(6'd0);
        wait_for_state(3'd6, 500); // S_ERROR

        // Attempt validate while in ERROR
        trigger_validate(6'd0);
        check("Still in ERROR", kernel_state == 3'd6);

        // ==============================================================
        // TEST 8: Multiple kernel reload cycle
        // ==============================================================
        test_num = 8;
        $display("\n=== TEST %0d: Multiple kernel reload cycle ===", test_num);
        reset_dut;

        // Load kernel A (8 words, slot 0)
        write_prog_word(16'h0000, 32'h00000000);
        write_prog_word(16'h0001, 32'h00000001);
        write_prog_word(16'h0002, 32'h00000002);
        write_prog_word(16'h0003, 32'h00000003);
        write_prog_word(16'h0004, 32'h00000004);
        write_prog_word(16'h0005, 32'h00000005);
        write_prog_word(16'h0006, 32'h00000006);
        write_prog_word(16'h0007, 32'h00000007);
        trigger_validate(6'd0);
        wait_for_state(3'd5, 500);
        check("Kernel A verified", kernel_verified == 1'b1);

        // Reload kernel B (4 words, slot 1)
        // First word triggers EXECUTE→LOAD transition
        core_busy <= 1'b0;
        write_prog_word(16'h0000, 32'h00000000);
        check("Transitioned to LOAD for kernel B", kernel_state == 3'd1);

        // Wait for SHA engine to initialize and first word to be hashed
        repeat (5) @(posedge clk);
        #1;

        // Remaining 3 words
        write_prog_word(16'h0001, 32'h00000001);
        write_prog_word(16'h0002, 32'h00000002);
        write_prog_word(16'h0003, 32'h00000003);

        trigger_validate(6'd1);
        wait_for_state(3'd5, 500);
        check("Kernel B verified", kernel_verified == 1'b1);
        check("No fault after reload", kernel_fault == 1'b0);

        // =========================================================================
        // Summary
        // =========================================================================
        $display("\n============================================");
        $display("  %0d tests passed, %0d tests failed", pass_count, fail_count);
        $display("============================================\n");

        if (fail_count > 0) begin
            $display("SOME TESTS FAILED");
        end else begin
            $display("ALL TESTS PASSED");
        end

        $finish;
    end

endmodule
