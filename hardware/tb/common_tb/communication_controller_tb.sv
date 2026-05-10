`timescale 1ns / 1ps

module communication_controller_tb();

localparam CLK_FREQ = 100_000_000;
localparam BAUD_RATE = 10_000_000;
localparam ADDR_WIDTH = 16;
localparam DATA_WIDTH = 32;
localparam MEMORY_BANK_DEPTH = 64;
localparam BAUD_PERIOD = 100; // ns for 10 Mbaud at 100 MHz

localparam COM_CMD_WRITE_DATA = 8'h01;
localparam COM_CMD_READ_DATA  = 8'h02;
localparam COM_CMD_ACK        = 8'h08;
localparam COM_CMD_NAK        = 8'h09;
localparam COM_CMD_VALIDATE   = 8'h07;

reg clk;
reg rst;
reg rx_line;
wire tx_line;

reg rsp_valid;
reg [31:0] rsp_data;
wire rsp_ready;

wire [7:0]  dbg_rx_cmd;
wire [15:0] dbg_rx_addr;
wire [15:0] dbg_rx_len;

wire [3:0]  validate_model_id;
wire        validate_triggered;

wire security_reset_triggered;

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
reg [31:0] status_word;

wire [3:0] data_mem_req_valid;
wire [3:0] data_mem_req_write;
wire [(4*ADDR_WIDTH)-1:0] data_mem_req_addr;
wire [(4*DATA_WIDTH)-1:0] data_mem_req_wdata;
wire [3:0] data_mem_req_ready;
wire [3:0] data_mem_resp_valid;
wire [(4*DATA_WIDTH)-1:0] data_mem_resp_rdata;

reg  mem_write_done;
reg  mem_write_fail;
reg  force_mem_write_fail;
wire memory_status_consumed;

integer testNum;
integer errors;
integer write_count;
integer timeout_count;
integer i;
reg [15:0] write_addr_seen [0:31];
reg [31:0] write_data_seen [0:31];
reg [7:0] payload [0:255];
reg validate_triggered_seen;

// ============================================================
// DUT
// ============================================================
communication_controller #(
    .CLK_FREQ(CLK_FREQ),
    .BAUD_RATE(BAUD_RATE),
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(DATA_WIDTH),
    .MEMORY_BANK_DEPTH(MEMORY_BANK_DEPTH)
) dut (
    .clk(clk),
    .rst(rst),
    .rx_line(rx_line),
    .tx_line(tx_line),
    .rsp_valid(rsp_valid),
    .rsp_data(rsp_data),
    .rsp_ready(rsp_ready),
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
    .dbg_rx_cmd(dbg_rx_cmd),
    .dbg_rx_addr(dbg_rx_addr),
    .dbg_rx_len(dbg_rx_len),
    .validate_model_id(validate_model_id),
    .validate_triggered(validate_triggered),
    .security_reset_triggered(security_reset_triggered),
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
    .status_word(status_word)
);

// External memory instance, like the top module should provide.
memory #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(DATA_WIDTH),
    .BANK_DEPTH(MEMORY_BANK_DEPTH)
) data_memory (
    .clk(clk),
    .rst(rst),
    .req_valid(data_mem_req_valid),
    .req_write(data_mem_req_write),
    .req_addr(data_mem_req_addr),
    .req_wdata(data_mem_req_wdata),
    .req_wmask({4{1'b1}}),
    .req_ready(data_mem_req_ready),
    .resp_valid(data_mem_resp_valid),
    .resp_rdata(data_mem_resp_rdata)
);

// ============================================================
// Clock (100 MHz)
// ============================================================
always #5 clk = ~clk;

// ============================================================
// Capture controller memory writes
// ============================================================
always @(posedge clk) begin
    if (rst) begin
        write_count <= 0;
    end else if (data_mem_req_valid[0] && data_mem_req_write[0]) begin
        write_addr_seen[write_count] <= data_mem_req_addr[0 +: ADDR_WIDTH];
        write_data_seen[write_count] <= data_mem_req_wdata[0 +: DATA_WIDTH];
        write_count <= write_count + 1;
    end
end

always @(posedge clk) begin
    if (rst) begin
        validate_triggered_seen <= 1'b0;
    end else if (validate_triggered) begin
        validate_triggered_seen <= 1'b1;
    end
end

always @(posedge clk or posedge rst) begin
    if (rst) begin
        mem_write_done <= 1'b0;
        mem_write_fail <= 1'b0;
        force_mem_write_fail <= 1'b0;
        status_word <= 32'b0;
    end else begin
        if (memory_status_consumed) begin
            mem_write_done <= 1'b0;
            mem_write_fail <= 1'b0;
        end else if (dut.state == 4'd13) begin
            // Respond immediately to WRITE_DATA packet completion.
            if (force_mem_write_fail) begin
                mem_write_fail <= 1'b1;
            end else begin
                mem_write_done <= 1'b1;
            end
        end
    end
end

// ============================================================
// UART bit helper (LSB-first like the other common testbenches)
// ============================================================
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

// ============================================================
// CRC helper
// ============================================================
function [7:0] crc_calc;
    input [7:0] c;
    input [7:0] b;
    begin
        crc_calc = c ^ b;
    end
endfunction

// ============================================================
// Packet sender
// ============================================================
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

        send_byte(cmd_i);
        crc = crc_calc(crc, cmd_i);

        send_byte(addr_i[15:8]);
        crc = crc_calc(crc, addr_i[15:8]);

        send_byte(addr_i[7:0]);
        crc = crc_calc(crc, addr_i[7:0]);

        send_byte(len_i[15:8]);
        crc = crc_calc(crc, len_i[15:8]);

        send_byte(len_i[7:0]);
        crc = crc_calc(crc, len_i[7:0]);

        for (i = 0; i < len_i; i = i + 1) begin
            send_byte(payload[i]);
            crc = crc_calc(crc, payload[i]);
        end

        send_byte(crc);
    end
endtask

// ============================================================
// Bad CRC packet sender
// ============================================================
task send_bad_crc_packet(
    input [7:0] cmd_i,
    input [15:0] addr_i,
    input [15:0] len_i
);
    integer i;
    reg [7:0] crc;
    begin
        crc = 0;

        send_byte(8'hAA);

        send_byte(cmd_i);
        crc = crc_calc(crc, cmd_i);

        send_byte(addr_i[15:8]);
        crc = crc_calc(crc, addr_i[15:8]);

        send_byte(addr_i[7:0]);
        crc = crc_calc(crc, addr_i[7:0]);

        send_byte(len_i[15:8]);
        crc = crc_calc(crc, len_i[15:8]);

        send_byte(len_i[7:0]);
        crc = crc_calc(crc, len_i[7:0]);

        for (i = 0; i < len_i; i = i + 1) begin
            send_byte(payload[i]);
            crc = crc_calc(crc, payload[i]);
        end

        send_byte(crc ^ 8'hFF);
    end
endtask

// ============================================================
// TX packet checker
// ============================================================
task expect_tx_done(
    input [7:0] exp_cmd,
    input [15:0] exp_addr,
    input [15:0] exp_len
);
    begin
        timeout_count = 0;
        while ((dut.tx_start !== 1'b1) && (timeout_count < 20000)) begin
            @(posedge clk);
            timeout_count = timeout_count + 1;
        end

        if (dut.tx_start !== 1'b1) begin
            $display("TEST%0d FAIL: timed out waiting for controller TX start", testNum);
            errors = errors + 1;
        end

        if (dut.tx_cmd !== exp_cmd) begin
            $display("TEST%0d FAIL: tx_cmd expected %h got %h", testNum, exp_cmd, dut.tx_cmd);
            errors = errors + 1;
        end

        if (dut.tx_addr !== exp_addr) begin
            $display("TEST%0d FAIL: tx_addr expected %h got %h", testNum, exp_addr, dut.tx_addr);
            errors = errors + 1;
        end

        if (dut.tx_len !== exp_len) begin
            $display("TEST%0d FAIL: tx_len expected %h got %h", testNum, exp_len, dut.tx_len);
            errors = errors + 1;
        end

        #20;
end
endtask

task expect_tx_packet_with_word(
    input [7:0] exp_cmd,
    input [15:0] exp_addr,
    input [15:0] exp_len,
    input [31:0] exp_word
);
    begin
        expect_tx_done(exp_cmd, exp_addr, exp_len);

        timeout_count = 0;
        while ((dut.tx_payload_valid !== 1'b1) && (timeout_count < 20000)) begin
            @(posedge clk);
            timeout_count = timeout_count + 1;
        end

        if (dut.tx_payload_valid !== 1'b1) begin
            $display("TEST%0d FAIL: timed out waiting for controller TX payload", testNum);
            errors = errors + 1;
        end

        if (dut.tx_payload_data !== exp_word) begin
            $display("TEST%0d FAIL: payload expected %h got %h", testNum, exp_word, dut.tx_payload_data);
            errors = errors + 1;
        end

        #20;
end
endtask

task expect_tx_packet_with_words4(
    input [7:0] exp_cmd,
    input [15:0] exp_addr,
    input [15:0] exp_len,
    input [31:0] exp_word0,
    input [31:0] exp_word1,
    input [31:0] exp_word2,
    input [31:0] exp_word3
);
    begin
        expect_tx_done(exp_cmd, exp_addr, exp_len);

        timeout_count = 0;
        while ((dut.tx_payload_valid !== 1'b1) && (timeout_count < 20000)) begin
            @(posedge clk);
            timeout_count = timeout_count + 1;
        end
        if (dut.tx_payload_data !== exp_word0) begin
            $display("TEST%0d FAIL: payload word0 expected %h got %h", testNum, exp_word0, dut.tx_payload_data);
            errors = errors + 1;
        end

        @(posedge clk);
        while (dut.tx_payload_valid === 1'b1) @(posedge clk);
        timeout_count = 0;
        while ((dut.tx_payload_valid !== 1'b1) && (timeout_count < 20000)) begin
            @(posedge clk);
            timeout_count = timeout_count + 1;
        end
        if (dut.tx_payload_data !== exp_word1) begin
            $display("TEST%0d FAIL: payload word1 expected %h got %h", testNum, exp_word1, dut.tx_payload_data);
            errors = errors + 1;
        end

        @(posedge clk);
        while (dut.tx_payload_valid === 1'b1) @(posedge clk);
        timeout_count = 0;
        while ((dut.tx_payload_valid !== 1'b1) && (timeout_count < 20000)) begin
            @(posedge clk);
            timeout_count = timeout_count + 1;
        end
        if (dut.tx_payload_data !== exp_word2) begin
            $display("TEST%0d FAIL: payload word2 expected %h got %h", testNum, exp_word2, dut.tx_payload_data);
            errors = errors + 1;
        end

        @(posedge clk);
        while (dut.tx_payload_valid === 1'b1) @(posedge clk);
        timeout_count = 0;
        while ((dut.tx_payload_valid !== 1'b1) && (timeout_count < 20000)) begin
            @(posedge clk);
            timeout_count = timeout_count + 1;
        end
        if (dut.tx_payload_data !== exp_word3) begin
            $display("TEST%0d FAIL: payload word3 expected %h got %h", testNum, exp_word3, dut.tx_payload_data);
            errors = errors + 1;
        end

        #20;
    end
endtask

task wait_tx_idle;
    begin
        @(posedge clk);
        timeout_count = 0;
        while ((dut.tx_busy !== 1'b0) && (timeout_count < 50000)) begin
            @(posedge clk);
            timeout_count = timeout_count + 1;
        end

        if (dut.tx_busy !== 1'b0) begin
            $display("TEST%0d FAIL: timed out waiting for controller TX idle", testNum);
            errors = errors + 1;
            rst = 1'b1;
            rx_line = 1'b1;
            repeat (5) @(posedge clk);
            rst = 1'b0;
            #1000;
        end

        #1000;
    end
endtask

task reset_dut;
    begin
        rst = 1'b1;
        rx_line = 1'b1;
        rsp_valid = 1'b0;
        repeat (5) @(posedge clk);
        rst = 1'b0;
        #1000;
    end
endtask

task fill_payload_counting(input integer count);
    integer j;
    begin
        for (j = 0; j < count; j = j + 1) begin
            payload[j] = j[7:0];
        end
    end
endtask

function [31:0] expected_counting_word;
    input integer word_index;
    begin
        expected_counting_word = {
            counting_byte(word_index*4 + 3),
            counting_byte(word_index*4 + 2),
            counting_byte(word_index*4 + 1),
            counting_byte(word_index*4 + 0)
        };
    end
endfunction

function [7:0] counting_byte;
    input integer value;
    begin
        counting_byte = value;
    end
endfunction

// ============================================================
// Test procedure
// ============================================================
initial begin
    #5000000;
    $display("COMMUNICATION CONTROLLER TESTS TIMED OUT");
    $finish;
end

initial begin
    $dumpfile("tmp/communication_controller.vcd");
    $dumpvars(1, communication_controller_tb);

    clk = 0;
    rst = 1;
    rx_line = 1;
    rsp_valid = 0;
    rsp_data = 0;
    errors = 0;
    testNum = 0;

    #20;
    rst = 0;
    #1000;

    // ========================================================
    // TEST 1: Multi-word WRITE_DATA writes sequential addresses
    // ========================================================
    testNum = 1;
    write_count = 0;

    payload[0] = 8'h11;
    payload[1] = 8'h22;
    payload[2] = 8'h33;
    payload[3] = 8'h44;
    payload[4] = 8'h55;
    payload[5] = 8'h66;
    payload[6] = 8'h77;
    payload[7] = 8'h88;

    fork
        send_packet(COM_CMD_WRITE_DATA, 16'h0010, 16'd8);
        expect_tx_done(COM_CMD_ACK, 16'h0010, 16'd0);
    join
    wait_tx_idle();

    if (write_count !== 2) begin
        $display("TEST1 FAIL: write_count expected 2 got %0d", write_count);
        errors = errors + 1;
    end

    if (write_addr_seen[0] !== 16'h0004 || write_data_seen[0] !== 32'h44332211) begin
        $display("TEST1 FAIL WORD0: addr=%h data=%h", write_addr_seen[0], write_data_seen[0]);
        errors = errors + 1;
    end

    if (write_addr_seen[1] !== 16'h0005 || write_data_seen[1] !== 32'h88776655) begin
        $display("TEST1 FAIL WORD1: addr=%h data=%h", write_addr_seen[1], write_data_seen[1]);
        errors = errors + 1;
    end

    // ========================================================
    // TEST 1B: WRITE_DATA NAKs when memory write fails
    // ========================================================
    testNum = 11;
    write_count = 0;
    force_mem_write_fail = 1'b1;

    payload[0] = 8'h11;
    payload[1] = 8'h22;
    payload[2] = 8'h33;
    payload[3] = 8'h44;
    payload[4] = 8'h55;
    payload[5] = 8'h66;
    payload[6] = 8'h77;
    payload[7] = 8'h88;

    fork
        send_packet(COM_CMD_WRITE_DATA, 16'h0100, 16'd8);
        expect_tx_done(COM_CMD_NAK, 16'h0100, 16'd0);
    join
    wait_tx_idle();
    force_mem_write_fail = 1'b0;

    if (write_count !== 2) begin
        $display("TEST1B FAIL: write_count expected 2 got %0d", write_count);
        errors = errors + 1;
    end

    // ========================================================
    // TEST 2: READ_DATA returns the word stored in memory
    // ========================================================
    testNum = 2;

    // READ_DATA uses len as the byte count to read. The RX parser also
    // consumes len payload bytes, so this test sends dummy payload bytes.
    payload[0] = 8'h00;
    payload[1] = 8'h00;
    payload[2] = 8'h00;
    payload[3] = 8'h00;

    fork
        send_packet(COM_CMD_READ_DATA, 16'h0010, 16'd4);
        expect_tx_packet_with_word(COM_CMD_READ_DATA, 16'h0010, 16'd4, 32'h44332211);
    join
    wait_tx_idle();

    // ========================================================
    // TEST 2B: READ_DATA can stream multiple response words
    // ========================================================
    testNum = 20;
    write_count = 0;
    fill_payload_counting(16);

    fork
        send_packet(COM_CMD_WRITE_DATA, 16'h0060, 16'd16);
        expect_tx_done(COM_CMD_ACK, 16'h0060, 16'd0);
    join
    wait_tx_idle();

    fill_payload_counting(16);
    fork
        send_packet(COM_CMD_READ_DATA, 16'h0060, 16'd16);
        expect_tx_packet_with_words4(
            COM_CMD_READ_DATA,
            16'h0060,
            16'd16,
            32'h03020100,
            32'h07060504,
            32'h0B0A0908,
            32'h0F0E0D0C
        );
    join
    wait_tx_idle();

    // ========================================================
    // TEST 3: Bad CRC produces NAK after streamed memory write
    // ========================================================
    testNum = 3;
    write_count = 0;

    payload[0] = 8'hAA;
    payload[1] = 8'hBB;
    payload[2] = 8'hCC;
    payload[3] = 8'hDD;

    fork
        send_bad_crc_packet(COM_CMD_WRITE_DATA, 16'h0020, 16'd4);
        expect_tx_done(COM_CMD_NAK, 16'h0020, 16'd0);
    join
    wait_tx_idle();

    if (write_count !== 1) begin
        $display("TEST3 FAIL: bad CRC write_count expected 1 got %0d", write_count);
        errors = errors + 1;
    end

    if (write_addr_seen[0] !== 16'h0008 || write_data_seen[0] !== 32'hDDCCBBAA) begin
        $display("TEST3 FAIL: streamed bad-CRC write addr=%h data=%h", write_addr_seen[0], write_data_seen[0]);
        errors = errors + 1;
    end

    // ========================================================
    // TEST 4: External RSP interface sends one response word
    // ========================================================
    testNum = 4;

    rsp_data = 32'hCAFEBABE;
    rsp_valid = 1'b1;
    timeout_count = 0;
    while ((rsp_ready !== 1'b1) && (timeout_count < 20000)) begin
        @(posedge clk);
        timeout_count = timeout_count + 1;
    end
    if (rsp_ready !== 1'b1) begin
        $display("TEST4 FAIL: timed out waiting for rsp_ready");
        errors = errors + 1;
    end
    #10;
    rsp_valid = 1'b0;

    fork
        expect_tx_packet_with_word(8'h0A, 16'h0000, 16'd4, 32'hCAFEBABE);
    join
    wait_tx_idle();

    // ========================================================
    // TEST 5: Zero-length WRITE_DATA ACKs and writes nothing
    // ========================================================
    testNum = 5;
    write_count = 0;

    fork
        send_packet(COM_CMD_WRITE_DATA, 16'h0030, 16'd0);
        expect_tx_done(COM_CMD_ACK, 16'h0030, 16'd0);
    join
    wait_tx_idle();

    if (write_count !== 0) begin
        $display("TEST5 FAIL: zero-length write_count expected 0 got %0d", write_count);
        errors = errors + 1;
    end

    // ========================================================
    // TEST 6: Zero-length READ_DATA returns header with no payload
    // ========================================================
    testNum = 6;

    fork
        send_packet(COM_CMD_READ_DATA, 16'h0030, 16'd0);
        expect_tx_done(COM_CMD_READ_DATA, 16'h0030, 16'd0);
    join
    wait_tx_idle();

    // ========================================================
    // TEST 7: Back-to-back writes after each ACK
    // ========================================================
    testNum = 7;
    write_count = 0;

    payload[0] = 8'h10;
    payload[1] = 8'h32;
    payload[2] = 8'h54;
    payload[3] = 8'h76;

    fork
        send_packet(COM_CMD_WRITE_DATA, 16'h0040, 16'd4);
        expect_tx_done(COM_CMD_ACK, 16'h0040, 16'd0);
    join
    wait_tx_idle();

    payload[0] = 8'h89;
    payload[1] = 8'hAB;
    payload[2] = 8'hCD;
    payload[3] = 8'hEF;

    fork
        send_packet(COM_CMD_WRITE_DATA, 16'h0044, 16'd4);
        expect_tx_done(COM_CMD_ACK, 16'h0044, 16'd0);
    join
    wait_tx_idle();

    if (write_count !== 2) begin
        $display("TEST7 FAIL: back-to-back write_count expected 2 got %0d", write_count);
        errors = errors + 1;
    end

    if (write_addr_seen[0] !== 16'h0010 || write_data_seen[0] !== 32'h76543210) begin
        $display("TEST7 FAIL WORD0: addr=%h data=%h", write_addr_seen[0], write_data_seen[0]);
        errors = errors + 1;
    end

    if (write_addr_seen[1] !== 16'h0011 || write_data_seen[1] !== 32'hEFCDAB89) begin
        $display("TEST7 FAIL WORD1: addr=%h data=%h", write_addr_seen[1], write_data_seen[1]);
        errors = errors + 1;
    end

    // ========================================================
    // TEST 8: Reset during packet reception, then recover
    // ========================================================
    testNum = 8;
    write_count = 0;

    send_byte(8'hAA);
    send_byte(COM_CMD_WRITE_DATA);
    send_byte(8'h00);
    send_byte(8'h50);
    reset_dut();

    payload[0] = 8'hCA;
    payload[1] = 8'hFE;
    payload[2] = 8'hBA;
    payload[3] = 8'hBE;

    fork
        send_packet(COM_CMD_WRITE_DATA, 16'h0050, 16'd4);
        expect_tx_done(COM_CMD_ACK, 16'h0050, 16'd0);
    join
    wait_tx_idle();

    if (write_count !== 1) begin
        $display("TEST8 FAIL: reset recovery write_count expected 1 got %0d", write_count);
        errors = errors + 1;
    end

    if (write_addr_seen[0] !== 16'h0014 || write_data_seen[0] !== 32'hBEBAFECA) begin
        $display("TEST8 FAIL: reset recovery addr=%h data=%h", write_addr_seen[0], write_data_seen[0]);
        errors = errors + 1;
    end

    // ========================================================
    // TEST 9: Long WRITE_DATA stream
    // ========================================================
    testNum = 9;
    write_count = 0;
    fill_payload_counting(64);

    fork
        send_packet(COM_CMD_WRITE_DATA, 16'h0080, 16'd64);
        expect_tx_done(COM_CMD_ACK, 16'h0080, 16'd0);
    join
    wait_tx_idle();

    if (write_count !== 16) begin
        $display("TEST9 FAIL: long write_count expected 16 got %0d", write_count);
        errors = errors + 1;
    end

    for (i = 0; i < 16; i = i + 1) begin
        if (write_addr_seen[i] !== (16'h0020 + i)) begin
            $display("TEST9 FAIL ADDR[%0d]: expected %h got %h", i, 16'h0020 + i, write_addr_seen[i]);
            errors = errors + 1;
        end

        if (write_data_seen[i] !== expected_counting_word(i)) begin
            $display("TEST9 FAIL DATA[%0d]: expected %h got %h", i, expected_counting_word(i), write_data_seen[i]);
            errors = errors + 1;
        end
    end

    // ========================================================
    // TEST 10: VALIDATE command triggers validation and sends ACK
    // ========================================================
    testNum = 10;
    validate_triggered_seen = 1'b0;

    fork
        send_packet(COM_CMD_VALIDATE, 16'h0005, 16'd0);
        expect_tx_done(COM_CMD_ACK, 16'h0005, 16'd0);
    join
    wait_tx_idle();

    if (validate_model_id !== 4'h5) begin
        $display("TEST10 FAIL: validate_model_id expected 4'h5 got %h", validate_model_id);
        errors = errors + 1;
    end

    if (!validate_triggered_seen) begin
        $display("TEST10 FAIL: validate_triggered was not pulsed");
        errors = errors + 1;
    end

    // ========================================================
    // DONE
    // ========================================================
    #1000;

    if (errors == 0)
        $display("COMMUNICATION CONTROLLER TESTS PASSED");
    else
        $display("COMMUNICATION CONTROLLER TESTS FAILED: %0d errors", errors);

    $finish;
end

endmodule
