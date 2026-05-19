`timescale 1ns/1ps

module security_gate_integration_tb();

    localparam CLK_FREQ = 100_000_000;
    localparam BAUD_RATE = 10_000_000;
    localparam ADDR_WIDTH = 16;
    localparam DATA_WIDTH = 32;
    localparam MEMORY_BANK_DEPTH = 64;
    localparam BAUD_PERIOD = 100;

    localparam COM_CMD_WRITE_DATA     = 8'h01;
    localparam COM_CMD_READ_DATA      = 8'h02;
    localparam COM_CMD_VALIDATE       = 8'h07;
    localparam COM_CMD_ACK            = 8'h08;
    localparam COM_CMD_NAK            = 8'h09;
    localparam COM_CMD_SECURITY_RESET = 8'h06;

    // SHA-256 of 4 big-endian 32-bit words [0,1,2,3]
    localparam [255:0] HASH_4_WORDS =
        256'h3067c72c5e501c31e3feca73f047dc341a956399ec705e0aee9efb17a1553578;
    localparam [(4*256)-1:0] GOLDEN_HASHES = {
        256'h0,
        256'h0,
        256'h0,
        HASH_4_WORDS
    };

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
    wire [3:0]  validate_model_id;
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

    // Core bus (tied off for this integration test)
    wire [3:0]  core_mem_req_ready;
    wire [3:0]  core_mem_resp_valid;
    wire [(4*DATA_WIDTH)-1:0] core_mem_resp_rdata;
    wire        core_mem_write_done;
    wire        core_mem_write_fail;

    // =========================================================================
    // Security Gate → Memory_sec wires
    // =========================================================================
    wire [3:0]  sg_mem_req_valid;
    wire [3:0]  sg_mem_req_write;
    wire [(4*ADDR_WIDTH)-1:0] sg_mem_req_addr;
    wire [(4*DATA_WIDTH)-1:0] sg_mem_req_wdata;
    wire [(4*4)-1:0] sg_mem_req_wmask;
    wire [3:0]  sg_mem_req_ready;
    wire [3:0]  sg_mem_resp_valid;
    wire [(4*DATA_WIDTH)-1:0] sg_mem_resp_rdata;

    wire        weight_locked;
    wire [ADDR_WIDTH-1:0] protected_base;
    wire [ADDR_WIDTH-1:0] protected_limit;
    wire [2:0]  security_state;
    wire        verified;
    wire        security_fault;
    wire [3:0]  write_blocked;

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
        .validate_model_id(validate_model_id),
        .validate_triggered(validate_triggered),
        .security_reset_triggered(security_reset_triggered),
        .dbg_rx_cmd(dbg_rx_cmd),
        .dbg_rx_addr(dbg_rx_addr),
        .dbg_rx_len(dbg_rx_len)
    );

    // =========================================================================
    // Bus Controller
    // =========================================================================
    bus_controller #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) bus_ctrl (
        .clk(clk),
        .rst(rst),
        .host_mem_req_valid(data_mem_req_valid),
        .host_mem_req_write(data_mem_req_write),
        .host_mem_req_addr(data_mem_req_addr),
        .host_mem_req_wdata(data_mem_req_wdata),
        .host_mem_req_wmask({4{4'b1111}}),
        .host_mem_req_ready(data_mem_req_ready),
        .host_mem_resp_valid(data_mem_resp_valid),
        .host_mem_resp_rdata(data_mem_resp_rdata),
        .host_mem_write_done(mem_write_done),
        .host_mem_write_fail(mem_write_fail),
        .host_memory_status_consumed(memory_status_consumed),
        .core_mem_req_valid(4'b0000),
        .core_mem_req_write(4'b0000),
        .core_mem_req_addr({(4*ADDR_WIDTH){1'b0}}),
        .core_mem_req_wdata({(4*DATA_WIDTH){1'b0}}),
        .core_mem_req_wmask({(4*4){1'b0}}),
        .core_mem_req_ready(core_mem_req_ready),
        .core_mem_resp_valid(core_mem_resp_valid),
        .core_mem_resp_rdata(core_mem_resp_rdata),
        .core_mem_write_done(core_mem_write_done),
        .core_mem_write_fail(core_mem_write_fail),
        .core_memory_status_consumed(1'b0),
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
    // Security Gate
    // =========================================================================
    security_gate #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_GOLDEN_HASHES(4),
        .GOLDEN_HASHES(GOLDEN_HASHES)
    ) sec_gate (
        .clk(clk),
        .rst(rst),
        .security_reset(security_reset_triggered),
        .merged_mem_req_valid(merged_mem_req_valid),
        .merged_mem_req_write(merged_mem_req_write),
        .merged_mem_req_addr(merged_mem_req_addr),
        .merged_mem_req_wdata(merged_mem_req_wdata),
        .merged_mem_req_wmask(merged_mem_req_wmask),
        .merged_mem_req_ready(merged_mem_req_ready),
        .merged_mem_resp_valid(merged_mem_resp_valid),
        .merged_mem_resp_rdata(merged_mem_resp_rdata),
        .memory_request_id(memory_request_id),
        .mem_req_valid(sg_mem_req_valid),
        .mem_req_write(sg_mem_req_write),
        .mem_req_addr(sg_mem_req_addr),
        .mem_req_wdata(sg_mem_req_wdata),
        .mem_req_wmask(sg_mem_req_wmask),
        .mem_req_ready(sg_mem_req_ready),
        .mem_resp_valid(sg_mem_resp_valid),
        .mem_resp_rdata(sg_mem_resp_rdata),
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
    // Memory (memory_sec)
    // =========================================================================
    memory_sec #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .BANK_DEPTH(MEMORY_BANK_DEPTH)
    ) mem (
        .clk(clk),
        .rst(rst),
        .locked(weight_locked),
        .protected_base(protected_base),
        .protected_limit(protected_limit),
        .req_valid(sg_mem_req_valid),
        .req_write(sg_mem_req_write),
        .req_addr(sg_mem_req_addr),
        .req_wdata(sg_mem_req_wdata),
        .req_wmask(sg_mem_req_wmask),
        .req_ready(sg_mem_req_ready),
        .resp_valid(sg_mem_resp_valid),
        .resp_rdata(sg_mem_resp_rdata),
        .write_blocked(write_blocked)
    );

    // =========================================================================
    // Test Infrastructure
    // =========================================================================
    integer errors;
    integer testNum;
    integer timeout_count;
    reg [7:0] payload [0:255];

    // =========================================================================
    // UART Helpers (from communication_controller_tb)
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
        input [7:0] exp_cmd,
        input [15:0] exp_addr,
        input [15:0] exp_len
    );
        begin
            timeout_count = 0;
            while ((comm.tx_start !== 1'b1) && (timeout_count < 50000)) begin
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
                end
            end
            #20;
        end
    endtask

    task wait_tx_idle;
        begin
            @(posedge clk);
            timeout_count = 0;
            while ((comm.tx_busy !== 1'b0) && (timeout_count < 100000)) begin
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

    task wait_for_sec_state;
        input [2:0] expected;
        begin
            timeout_count = 0;
            while ((security_state !== expected) && (timeout_count < 50000)) begin
                @(posedge clk);
                timeout_count = timeout_count + 1;
            end
            if (security_state !== expected) begin
                $display("  TEST%0d FAIL: timeout waiting for security state %0d, got %0d",
                         testNum, expected, security_state);
                errors = errors + 1;
            end
        end
    endtask

    // =========================================================================
    // Payload Helpers
    // =========================================================================

    // Fill payload with 4 words [0, 1, 2, 3] in little-endian byte order
    // Word 0x00000000 → bytes [0x00, 0x00, 0x00, 0x00]
    // Word 0x00000001 → bytes [0x01, 0x00, 0x00, 0x00]
    // etc.
    task fill_weight_payload_4words;
        integer w;
        begin
            for (w = 0; w < 4; w = w + 1) begin
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
        $dumpfile("./tmp/security_gate_integration_tb.vcd");
        $dumpvars(0, security_gate_integration_tb);

        clk = 0;
        rst = 1;
        rx_line = 1;
        errors = 0;
        testNum = 0;
        status_word = 32'b0;

        #20;
        rst = 0;
        #1000;

        // ==============================================================
        // Test 1: Happy path - load 4 weights, validate, verify EXECUTE
        // ==============================================================
        testNum = 1;
        $display("\n=== Test %0d: Happy path (4 weights over UART, validate model 0) ===", testNum);

        fill_weight_payload_4words();

        fork
            send_packet(COM_CMD_WRITE_DATA, 16'h0000, 16'd16);
            expect_tx_done(COM_CMD_ACK, 16'h0000, 16'd0);
        join
        wait_tx_idle();

        if (security_state !== 3'd1) begin
            $display("  TEST%0d FAIL: expected LOAD state after write, got %0d", testNum, security_state);
            errors = errors + 1;
        end

        // Send VALIDATE command (model_id = 0, in addr field)
        fork
            send_packet(COM_CMD_VALIDATE, 16'h0000, 16'd0);
            expect_tx_done(COM_CMD_ACK, 16'h0000, 16'd0);
        join
        wait_tx_idle();

        wait_for_sec_state(3'd5);

        if (verified !== 1'b1) begin
            $display("  TEST%0d FAIL: verified not set", testNum);
            errors = errors + 1;
        end
        if (weight_locked !== 1'b1) begin
            $display("  TEST%0d FAIL: weight_locked not set", testNum);
            errors = errors + 1;
        end
        if (protected_limit !== 16'd4) begin
            $display("  TEST%0d FAIL: protected_limit expected 4, got %0d", testNum, protected_limit);
            errors = errors + 1;
        end

        $display("  Test %0d: %s", testNum, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 2: Host read protected region returns 0xDEADDEAD
        // ==============================================================
        testNum = 2;
        $display("\n=== Test %0d: Host read protected region returns DEADDEAD ===", testNum);

        // READ_DATA at address 0, 4 bytes (1 word) — protected, host blocked
        payload[0] = 8'h00; payload[1] = 8'h00; payload[2] = 8'h00; payload[3] = 8'h00;
        fork
            send_packet(COM_CMD_READ_DATA, 16'h0000, 16'd4);
            begin
                timeout_count = 0;
                while ((comm.tx_start !== 1'b1) && (timeout_count < 50000)) begin
                    @(posedge clk);
                    timeout_count = timeout_count + 1;
                end
                if (comm.tx_cmd !== COM_CMD_READ_DATA) begin
                    $display("  TEST%0d FAIL: expected READ_DATA response, got %h", testNum, comm.tx_cmd);
                    errors = errors + 1;
                end
                timeout_count = 0;
                while ((comm.tx_payload_valid !== 1'b1) && (timeout_count < 50000)) begin
                    @(posedge clk);
                    timeout_count = timeout_count + 1;
                end
                if (comm.tx_payload_data !== 32'hDEADDEAD) begin
                    $display("  TEST%0d FAIL: expected 0xDEADDEAD, got %h", testNum, comm.tx_payload_data);
                    errors = errors + 1;
                end
            end
        join
        wait_tx_idle();

        $display("  Test %0d: %s", testNum, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 3: Write to protected region blocked in EXECUTE → NAK
        // ==============================================================
        testNum = 3;
        $display("\n=== Test %0d: Write to protected region blocked (NAK) ===", testNum);

        payload[0] = 8'hDE; payload[1] = 8'hAD; payload[2] = 8'hBE; payload[3] = 8'hEF;
        fork
            send_packet(COM_CMD_WRITE_DATA, 16'h0000, 16'd4);
            expect_tx_done(COM_CMD_NAK, 16'h0000, 16'd0);
        join
        wait_tx_idle();

        // Read back — host reads protected region, gets 0xDEADDEAD regardless
        payload[0] = 0; payload[1] = 0; payload[2] = 0; payload[3] = 0;
        fork
            send_packet(COM_CMD_READ_DATA, 16'h0000, 16'd4);
            begin
                timeout_count = 0;
                while ((comm.tx_start !== 1'b1) && (timeout_count < 50000)) begin
                    @(posedge clk);
                    timeout_count = timeout_count + 1;
                end
                timeout_count = 0;
                while ((comm.tx_payload_valid !== 1'b1) && (timeout_count < 50000)) begin
                    @(posedge clk);
                    timeout_count = timeout_count + 1;
                end
                if (comm.tx_payload_data !== 32'hDEADDEAD) begin
                    $display("  TEST%0d FAIL: expected 0xDEADDEAD, got %h", testNum, comm.tx_payload_data);
                    errors = errors + 1;
                end
            end
        join
        wait_tx_idle();

        $display("  Test %0d: %s", testNum, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 4: Write outside protected region succeeds
        // ==============================================================
        testNum = 4;
        $display("\n=== Test %0d: Write outside protected region succeeds ===", testNum);

        // Write to address 0x0010 (word address 4, outside [0,4) protected range)
        payload[0] = 8'hCA; payload[1] = 8'hFE; payload[2] = 8'hBA; payload[3] = 8'hBE;
        fork
            send_packet(COM_CMD_WRITE_DATA, 16'h0010, 16'd4);
            expect_tx_done(COM_CMD_ACK, 16'h0010, 16'd0);
        join
        wait_tx_idle();

        // Read back
        payload[0] = 0; payload[1] = 0; payload[2] = 0; payload[3] = 0;
        fork
            send_packet(COM_CMD_READ_DATA, 16'h0010, 16'd4);
            begin
                timeout_count = 0;
                while ((comm.tx_start !== 1'b1) && (timeout_count < 50000)) begin
                    @(posedge clk);
                    timeout_count = timeout_count + 1;
                end
                timeout_count = 0;
                while ((comm.tx_payload_valid !== 1'b1) && (timeout_count < 50000)) begin
                    @(posedge clk);
                    timeout_count = timeout_count + 1;
                end
                if (comm.tx_payload_data !== 32'hBEBAFECA) begin
                    $display("  TEST%0d FAIL: unprotected write failed! Got %h, expected BEBAFECA", testNum, comm.tx_payload_data);
                    errors = errors + 1;
                end
            end
        join
        wait_tx_idle();

        $display("  Test %0d: %s", testNum, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 5: Security reset via command → back to IDLE
        // ==============================================================
        testNum = 5;
        $display("\n=== Test %0d: Security reset via CMD 0x06 ===", testNum);

        fork
            send_packet(COM_CMD_SECURITY_RESET, 16'h0000, 16'd0);
            expect_tx_done(COM_CMD_ACK, 16'h0000, 16'd0);
        join
        wait_tx_idle();

        // Wait a few cycles for security_reset_triggered to propagate
        repeat (5) @(posedge clk);

        if (security_state !== 3'd0) begin
            $display("  TEST%0d FAIL: expected IDLE after reset, got %0d", testNum, security_state);
            errors = errors + 1;
        end
        if (weight_locked !== 1'b0) begin
            $display("  TEST%0d FAIL: still locked after reset", testNum);
            errors = errors + 1;
        end

        $display("  Test %0d: %s", testNum, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 6: Hash mismatch → ERROR state
        // ==============================================================
        testNum = 6;
        $display("\n=== Test %0d: Hash mismatch → ERROR ===", testNum);

        // Load 4 weights but validate against model 1 (whose hash is all zeros)
        fill_weight_payload_4words();
        fork
            send_packet(COM_CMD_WRITE_DATA, 16'h0000, 16'd16);
            expect_tx_done(COM_CMD_ACK, 16'h0000, 16'd0);
        join
        wait_tx_idle();

        fork
            send_packet(COM_CMD_VALIDATE, 16'h0001, 16'd0);
            expect_tx_done(COM_CMD_ACK, 16'h0001, 16'd0);
        join
        wait_tx_idle();

        wait_for_sec_state(3'd6);

        if (security_fault !== 1'b1) begin
            $display("  TEST%0d FAIL: security_fault not asserted", testNum);
            errors = errors + 1;
        end

        $display("  Test %0d: %s", testNum, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Test 7: Recovery from ERROR via security reset command
        // ==============================================================
        testNum = 7;
        $display("\n=== Test %0d: Recovery from ERROR ===", testNum);

        fork
            send_packet(COM_CMD_SECURITY_RESET, 16'h0000, 16'd0);
            expect_tx_done(COM_CMD_ACK, 16'h0000, 16'd0);
        join
        wait_tx_idle();

        repeat (5) @(posedge clk);

        if (security_state !== 3'd0) begin
            $display("  TEST%0d FAIL: not in IDLE after recovery, got %0d", testNum, security_state);
            errors = errors + 1;
        end
        if (security_fault !== 1'b0) begin
            $display("  TEST%0d FAIL: fault still asserted", testNum);
            errors = errors + 1;
        end

        // Can load again after recovery
        fill_weight_payload_4words();
        fork
            send_packet(COM_CMD_WRITE_DATA, 16'h0000, 16'd16);
            expect_tx_done(COM_CMD_ACK, 16'h0000, 16'd0);
        join
        wait_tx_idle();

        if (security_state !== 3'd1) begin
            $display("  TEST%0d FAIL: not in LOAD after new write, got %0d", testNum, security_state);
            errors = errors + 1;
        end

        // Validate with correct hash this time
        fork
            send_packet(COM_CMD_VALIDATE, 16'h0000, 16'd0);
            expect_tx_done(COM_CMD_ACK, 16'h0000, 16'd0);
        join
        wait_tx_idle();

        wait_for_sec_state(3'd5);

        if (verified !== 1'b1) begin
            $display("  TEST%0d FAIL: not verified after recovery+reload", testNum);
            errors = errors + 1;
        end

        $display("  Test %0d: %s", testNum, (errors == 0) ? "PASS" : "FAIL");

        // ==============================================================
        // Summary
        // ==============================================================
        #1000;
        $display("\n==========================================");
        if (errors == 0) begin
            $display("security_gate_integration_tb: ALL TESTS PASSED");
        end else begin
            $display("security_gate_integration_tb: %0d ERRORS", errors);
        end
        $display("==========================================\n");
        $finish;
    end

    // Timeout
    initial begin
        #50_000_000;
        $display("FAIL: Global timeout");
        $finish;
    end

endmodule
