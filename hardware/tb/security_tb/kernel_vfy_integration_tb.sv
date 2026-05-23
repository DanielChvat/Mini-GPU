`timescale 1ns/1ps

module kernel_vfy_integration_tb();

    localparam CLK_FREQ = 100_000_000;
    localparam BAUD_RATE = 10_000_000;
    localparam ADDR_WIDTH = 16;
    localparam DATA_WIDTH = 32;
    localparam PROG_ADDR_WIDTH = 12;
    localparam MEMORY_BANK_DEPTH = 64;
    localparam BAUD_PERIOD = 100;

    localparam COM_CMD_WRITE_PROGRAM  = 8'h03;
    localparam COM_CMD_VALIDATE       = 8'h07;
    localparam COM_CMD_ACK            = 8'h08;
    localparam COM_CMD_NAK            = 8'h09;
    localparam COM_CMD_SECURITY_RESET = 8'h06;

    // SHA-256 of 4 big-endian 32-bit words [0,1,2,3]
    localparam [255:0] HASH_4_WORDS =
        256'h3067c72c5e501c31e3feca73f047dc341a956399ec705e0aee9efb17a1553578;

    // SHA-256 of 8 big-endian 32-bit words [0,1,2,...,7]
    localparam [255:0] HASH_8_WORDS =
        256'hbdb32f8604eafe89ad767fe7fe8ccd29ecc5d0de9b7a3c9d95e3cced553d625a;

    // =========================================================================
    // Clock / Reset
    // =========================================================================
    reg clk;
    reg rst;
    reg rx_line;
    wire tx_line;

    always #5 clk = ~clk;

    // =========================================================================
    // Communication Controller Signals
    // =========================================================================
    wire [3:0]  data_mem_req_valid;
    wire [3:0]  data_mem_req_write;
    wire [(4*ADDR_WIDTH)-1:0] data_mem_req_addr;
    wire [(4*DATA_WIDTH)-1:0] data_mem_req_wdata;
    wire [3:0]  data_mem_req_ready;
    wire [3:0]  data_mem_resp_valid;
    wire [(4*DATA_WIDTH)-1:0] data_mem_resp_rdata;

    wire [7:0]  dbg_rx_cmd;
    wire [15:0] dbg_rx_addr;
    wire [15:0] dbg_rx_len;
    wire [5:0]  validate_kernel_id;
    wire        validate_triggered;
    wire        security_reset_triggered;

    wire        prog_we;
    wire [ADDR_WIDTH-1:0] prog_addr;
    wire [31:0] prog_wdata;
    wire        const_we;
    wire [ADDR_WIDTH-1:0] const_addr;
    wire [31:0] const_wdata;
    wire        launch_valid;
    wire [ADDR_WIDTH-1:0] launch_base_pc;
    wire [31:0] launch_grid_dim;
    wire [31:0] launch_block_dim;
    wire [31:0] launch_active_mask;
    reg  [31:0] status_word;

    wire        mem_write_done;
    wire        mem_write_fail;
    wire        memory_status_consumed;

    // =========================================================================
    // kernel_vfy signals
    // =========================================================================
    wire        kv_prog_we;
    wire [PROG_ADDR_WIDTH-1:0] kv_prog_addr;
    wire [31:0] kv_prog_wdata;
    wire        kv_launch_valid;
    wire        kv_prog_write_blocked;
    wire        kv_validate_done;
    wire        kv_validate_fail;
    wire [2:0]  kv_kernel_state;
    wire        kv_kernel_verified;
    wire        kv_kernel_fault;

    reg         core_busy;

    // Tie off data memory (not used in this test)
    assign data_mem_req_ready = 4'b1111;
    assign data_mem_resp_valid = 4'b0000;
    assign data_mem_resp_rdata = {(4*DATA_WIDTH){1'b0}};

    // Data memory write status: auto-ACK for non-program writes
    assign mem_write_done = |(data_mem_req_valid & data_mem_req_write & data_mem_req_ready);
    assign mem_write_fail = 1'b0;
    assign memory_status_consumed = 1'b0;

    // =========================================================================
    // Communication Controller
    // =========================================================================
    communication_controller #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .MEMORY_BANK_DEPTH(MEMORY_BANK_DEPTH)
    ) comm (
        .clk(clk),
        .rst(rst),
        .rx_line(rx_line),
        .tx_line(tx_line),
        .rsp_valid(1'b0),
        .rsp_data(32'b0),
        .rsp_ready(),
        .data_mem_req_valid(data_mem_req_valid),
        .data_mem_req_write(data_mem_req_write),
        .data_mem_req_addr(data_mem_req_addr),
        .data_mem_req_wdata(data_mem_req_wdata),
        .data_mem_req_ready(data_mem_req_ready),
        .data_mem_resp_valid(data_mem_resp_valid),
        .data_mem_resp_rdata(data_mem_resp_rdata),
        .mem_write_done(mem_write_done),
        .mem_write_fail(mem_write_fail),
        .memory_status_consumed(memory_status_consumed),
        .prog_we(prog_we),
        .prog_addr(prog_addr),
        .prog_wdata(prog_wdata),
        .const_we(const_we),
        .const_addr(const_addr),
        .const_wdata(const_wdata),
        .launch_valid(launch_valid),
        .launch_base_pc(launch_base_pc),
        .launch_grid_dim(launch_grid_dim),
        .launch_block_dim(launch_block_dim),
        .launch_active_mask(launch_active_mask),
        .status_word(status_word),
        .validate_kernel_id(validate_kernel_id),
        .validate_triggered(validate_triggered),
        .prog_write_blocked(kv_prog_write_blocked),
        .validate_done(kv_validate_done),
        .validate_fail(kv_validate_fail),
        .security_reset_triggered(security_reset_triggered),
        .dbg_rx_cmd(dbg_rx_cmd),
        .dbg_rx_addr(dbg_rx_addr),
        .dbg_rx_len(dbg_rx_len)
    );

    // =========================================================================
    // Kernel Verification Module
    // =========================================================================
    kernel_vfy #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .PROG_ADDR_WIDTH(PROG_ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_GOLDEN_HASHES(4),
        .KERNEL_ID_WIDTH(6)
    ) u_kernel_vfy (
        .clk(clk),
        .rst(rst),
        .security_reset(security_reset_triggered),
        .cc_prog_we(prog_we),
        .cc_prog_addr(prog_addr),
        .cc_prog_wdata(prog_wdata),
        .gpu_prog_we(kv_prog_we),
        .gpu_prog_addr(kv_prog_addr),
        .gpu_prog_wdata(kv_prog_wdata),
        .cc_launch_valid(launch_valid),
        .gpu_launch_valid(kv_launch_valid),
        .validate_triggered(validate_triggered),
        .validate_kernel_id(validate_kernel_id),
        .core_busy(core_busy),
        .prog_write_blocked(kv_prog_write_blocked),
        .validate_done(kv_validate_done),
        .validate_fail(kv_validate_fail),
        .kernel_state(kv_kernel_state),
        .kernel_verified(kv_kernel_verified),
        .kernel_fault(kv_kernel_fault)
    );

    // =========================================================================
    // Test Infrastructure
    // =========================================================================
    integer errors;
    integer testNum;
    integer timeout_count;
    reg [7:0] payload [0:255];

    // =========================================================================
    // UART Helpers
    // =========================================================================
    task send_byte(input [7:0] b);
        integer i;
        begin
            rx_line = 1'b0;
            #(BAUD_PERIOD);
            for (i = 0; i < 8; i = i + 1) begin
                rx_line = b[i];
                #(BAUD_PERIOD);
            end
            rx_line = 1'b1;
            #(BAUD_PERIOD);
        end
    endtask

    function [7:0] crc_calc;
        input [7:0] c;
        input [7:0] b;
        begin
            crc_calc = c ^ b;
        end
    endfunction

    task send_packet(
        input [7:0] cmd_i,
        input [15:0] addr_i,
        input [15:0] len_i
    );
        integer i;
        reg [7:0] crc;
        begin
            crc = 0;
            send_byte(8'hAA);
            send_byte(cmd_i);     crc = crc_calc(crc, cmd_i);
            send_byte(addr_i[15:8]); crc = crc_calc(crc, addr_i[15:8]);
            send_byte(addr_i[7:0]);  crc = crc_calc(crc, addr_i[7:0]);
            send_byte(len_i[15:8]);  crc = crc_calc(crc, len_i[15:8]);
            send_byte(len_i[7:0]);   crc = crc_calc(crc, len_i[7:0]);
            for (i = 0; i < len_i; i = i + 1) begin
                send_byte(payload[i]);
                crc = crc_calc(crc, payload[i]);
            end
            send_byte(crc);
        end
    endtask

    task expect_tx_done(
        input [7:0] exp_cmd
    );
        begin
            timeout_count = 0;
            while ((comm.tx_start !== 1'b1) && (timeout_count < 100000)) begin
                @(posedge clk);
                timeout_count = timeout_count + 1;
            end
            if (comm.tx_start !== 1'b1) begin
                $display("  TEST%0d FAIL: timed out waiting for TX start", testNum);
                errors = errors + 1;
            end else begin
                if (comm.tx_cmd !== exp_cmd) begin
                    $display("  TEST%0d FAIL: tx_cmd expected %h got %h", testNum, exp_cmd, comm.tx_cmd);
                    errors = errors + 1;
                end else begin
                    $display("  TEST%0d OK: received expected response %h", testNum, exp_cmd);
                end
            end
            #20;
        end
    endtask

    task wait_tx_idle;
        begin
            @(posedge clk);
            timeout_count = 0;
            while ((comm.tx_busy !== 1'b0) && (timeout_count < 200000)) begin
                @(posedge clk);
                timeout_count = timeout_count + 1;
            end
            if (comm.tx_busy !== 1'b0) begin
                $display("  TEST%0d FAIL: timed out waiting for TX idle", testNum);
                errors = errors + 1;
            end
            #1000;
        end
    endtask

    task wait_for_kernel_state;
        input [2:0] expected;
        begin
            timeout_count = 0;
            while ((kv_kernel_state !== expected) && (timeout_count < 100000)) begin
                @(posedge clk);
                timeout_count = timeout_count + 1;
            end
            if (kv_kernel_state !== expected) begin
                $display("  TEST%0d FAIL: timeout waiting for kernel state %0d, got %0d",
                         testNum, expected, kv_kernel_state);
                errors = errors + 1;
            end
        end
    endtask

    // =========================================================================
    // Payload Helpers
    // =========================================================================

    // Fill payload with N sequential words [0, 1, ..., N-1] in little-endian
    task fill_prog_payload;
        input integer num_words;
        integer w;
        begin
            for (w = 0; w < num_words; w = w + 1) begin
                payload[w*4 + 0] = w[7:0];
                payload[w*4 + 1] = w[15:8];
                payload[w*4 + 2] = w[23:16];
                payload[w*4 + 3] = w[31:24];
            end
        end
    endtask

    // =========================================================================
    // Main Test Sequence
    // =========================================================================
    initial begin
        $dumpfile("kernel_vfy_integration_tb.vcd");
        $dumpvars(0, kernel_vfy_integration_tb);

        clk = 0;
        rst = 1;
        rx_line = 1;
        errors = 0;
        testNum = 0;
        status_word = 32'b0;
        core_busy = 1'b0;

        // Load golden hashes for testing
        u_kernel_vfy.hash_store.hash_mem[0] = HASH_4_WORDS;
        u_kernel_vfy.hash_store.hash_mem[1] = HASH_8_WORDS;
        u_kernel_vfy.hash_store.hash_mem[2] = 256'h0;
        u_kernel_vfy.hash_store.hash_mem[3] = 256'h0;

        #20;
        rst = 0;
        #1000;

        // ==============================================================
        // Test 1: Upload 4-word kernel, validate → ACK, verify EXECUTE
        // ==============================================================
        testNum = 1;
        $display("\n=== Test %0d: Happy path (4-word kernel via UART, validate slot 0) ===", testNum);

        fill_prog_payload(4);

        // Send WRITE_PROGRAM packet with 4 words (16 bytes) at address 0
        fork
            send_packet(COM_CMD_WRITE_PROGRAM, 16'h0000, 16'd16);
            expect_tx_done(COM_CMD_ACK);
        join
        wait_tx_idle();

        if (kv_kernel_state !== 3'd1) begin
            $display("  TEST%0d FAIL: expected LOAD state after write, got %0d", testNum, kv_kernel_state);
            errors = errors + 1;
        end

        // Send VALIDATE command (kernel_id = 0 in addr[5:0])
        fork
            send_packet(COM_CMD_VALIDATE, 16'h0000, 16'd0);
            expect_tx_done(COM_CMD_ACK);
        join
        wait_tx_idle();

        wait_for_kernel_state(3'd5); // S_EXECUTE

        if (kv_kernel_verified !== 1'b1) begin
            $display("  TEST%0d FAIL: kernel_verified not set", testNum);
            errors = errors + 1;
        end
        if (kv_kernel_fault !== 1'b0) begin
            $display("  TEST%0d FAIL: kernel_fault set unexpectedly", testNum);
            errors = errors + 1;
        end

        $display("  Test %0d: %s", testNum, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 2: Hash mismatch → NAK for VALIDATE
        // ==============================================================
        testNum = 2;
        $display("\n=== Test %0d: Hash mismatch → NAK on VALIDATE ===", testNum);

        // Security reset first
        fork
            send_packet(COM_CMD_SECURITY_RESET, 16'h0000, 16'd0);
            expect_tx_done(COM_CMD_ACK);
        join
        wait_tx_idle();
        repeat (10) @(posedge clk);

        // Upload 4 words
        fill_prog_payload(4);
        fork
            send_packet(COM_CMD_WRITE_PROGRAM, 16'h0000, 16'd16);
            expect_tx_done(COM_CMD_ACK);
        join
        wait_tx_idle();

        // Validate against slot 2 (hash is 0, won't match)
        fork
            send_packet(COM_CMD_VALIDATE, 16'h0002, 16'd0);
            expect_tx_done(COM_CMD_NAK);
        join
        wait_tx_idle();

        wait_for_kernel_state(3'd6); // S_ERROR

        if (kv_kernel_fault !== 1'b1) begin
            $display("  TEST%0d FAIL: kernel_fault not asserted", testNum);
            errors = errors + 1;
        end

        $display("  Test %0d: %s", testNum, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 3: Recovery from ERROR via security reset, then re-verify
        // ==============================================================
        testNum = 3;
        $display("\n=== Test %0d: Recovery from ERROR, then successful verify ===", testNum);

        fork
            send_packet(COM_CMD_SECURITY_RESET, 16'h0000, 16'd0);
            expect_tx_done(COM_CMD_ACK);
        join
        wait_tx_idle();
        repeat (10) @(posedge clk);

        if (kv_kernel_state !== 3'd0) begin
            $display("  TEST%0d FAIL: not in IDLE after reset, got %0d", testNum, kv_kernel_state);
            errors = errors + 1;
        end

        // Re-upload and verify 8-word kernel in slot 1
        fill_prog_payload(8);
        fork
            send_packet(COM_CMD_WRITE_PROGRAM, 16'h0000, 16'd32);
            expect_tx_done(COM_CMD_ACK);
        join
        wait_tx_idle();

        fork
            send_packet(COM_CMD_VALIDATE, 16'h0001, 16'd0);
            expect_tx_done(COM_CMD_ACK);
        join
        wait_tx_idle();

        wait_for_kernel_state(3'd5);

        if (kv_kernel_verified !== 1'b1) begin
            $display("  TEST%0d FAIL: not verified after recovery+reload", testNum);
            errors = errors + 1;
        end

        $display("  Test %0d: %s", testNum, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 4: Write blocked when core is busy → NACK
        // ==============================================================
        testNum = 4;
        $display("\n=== Test %0d: WRITE_PROGRAM blocked when core busy (NACK) ===", testNum);

        // We're in EXECUTE from test 3, set core_busy
        core_busy = 1'b1;

        fill_prog_payload(4);
        fork
            send_packet(COM_CMD_WRITE_PROGRAM, 16'h0000, 16'd16);
            expect_tx_done(COM_CMD_NAK);
        join
        wait_tx_idle();

        // Should still be in EXECUTE, verified
        if (kv_kernel_state !== 3'd5) begin
            $display("  TEST%0d FAIL: not in EXECUTE, got %0d", testNum, kv_kernel_state);
            errors = errors + 1;
        end
        if (kv_kernel_verified !== 1'b1) begin
            $display("  TEST%0d FAIL: lost verification", testNum);
            errors = errors + 1;
        end

        core_busy = 1'b0;

        $display("  Test %0d: %s", testNum, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Summary
        // ==============================================================
        #1000;
        $display("\n==========================================");
        if (errors == 0) begin
            $display("kernel_vfy_integration_tb: ALL TESTS PASSED");
        end else begin
            $display("kernel_vfy_integration_tb: %0d ERRORS", errors);
        end
        $display("==========================================\n");
        $finish;
    end

    // Timeout
    initial begin
        #100_000_000;
        $display("FAIL: Global timeout");
        $finish;
    end

endmodule
