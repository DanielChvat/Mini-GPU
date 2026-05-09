`timescale 1ns / 1ps

module serial_packet_rx_tb();

reg clk;
reg rst;
reg rx_line;

wire [31:0] word_data;
wire word_valid;
reg  word_taken;

wire [7:0] cmd;
wire [15:0] addr;
wire [15:0] len;

reg write_done;
reg write_fail;
wire send_ack;
wire send_nack;

integer testNum;
integer errors;
reg [7:0] payload [0:255];

// DUT
serial_packet_rx dut (
    .clk(clk),
    .rst(rst),
    .rx_line(rx_line),
    .word_data(word_data),
    .word_valid(word_valid),
    // .word_taken(word_taken),
    .cmd(cmd),
    .addr(addr),
    .len(len),
    .write_done(write_done),
    .write_fail(write_fail),
    .send_ack(send_ack),
    .send_nack(send_nack)
);

// ============================================================
// Clock (100MHz)
// ============================================================
always begin
    #5 clk = ~clk;
end

// ============================================================
// UART bit helper (LSB-first like UART)
// ============================================================
task send_byte(input [7:0] b);
    integer i;
    begin
        rx_line = 0;
        #(8680);

        for (i = 0; i < 8; i = i + 1) begin
            rx_line = b[i];
            #(8680);
        end

        rx_line = 1;
        #(8680);
    end
endtask

// ============================================================
// CRC helper (must match DUT XOR logic)
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

        // SOF
        send_byte(8'hAA);

        // CMD
        send_byte(cmd_i);
        crc = crc_calc(crc, cmd_i);

        // ADDR HI
        send_byte(addr_i[15:8]);
        crc = crc_calc(crc, addr_i[15:8]);

        // ADDR LO
        send_byte(addr_i[7:0]);
        crc = crc_calc(crc, addr_i[7:0]);

        // LEN HI
        send_byte(len_i[15:8]);
        crc = crc_calc(crc, len_i[15:8]);

        // LEN LO
        send_byte(len_i[7:0]);
        crc = crc_calc(crc, len_i[7:0]);

        // PAYLOAD
        for (i = 0; i < len_i; i = i + 1) begin
            send_byte(payload[i]);
            crc = crc_calc(crc, payload[i]);
        end

        // CRC
        send_byte(crc);
    end
endtask

// ============================================================
// Test procedure
// ============================================================
// reg [7:0] payload [0:255];

initial begin
    $dumpfile("rx_wave.vcd");
    $dumpvars(0, serial_packet_rx_tb);

    clk = 0;
    rst = 1;
    rx_line = 1;
    word_taken = 0;
    write_done = 0;
    write_fail = 0;
    errors = 0;


    testNum = 0;

    // ========================================================
    // RESET TEST
    // ========================================================
    testNum = 0;
    #10.1;
    rst = 0;

    if (cmd !== 0 || addr !== 0 || len !== 0 || word_valid !== 0 || send_ack !== 0 || send_nack !== 0)
        $display("RESET FAIL");

    // ========================================================
    // TEST 1: Small packet (fits in 1 word)
    // ========================================================
    testNum = 1;
    $display("Starting test %d", testNum);

    payload[0] = 8'h11;
    payload[1] = 8'h22;
    payload[2] = 8'h33;
    payload[3] = 8'h44;

    fork
        // thread 1
        send_packet(8'h01, 16'h1000, 4);

        // thread 2
        begin
            // wait for word
            wait(word_valid == 1);

            if (word_data !== 32'h44332211) begin
                $display("TEST1 FAIL: word_data=%h", word_data);
                errors = errors + 1;
            end
        end
    join

    write_done = 1;
    #30;    // note the delay needed to activate the write done flag 
    write_done = 0;

    

    // ========================================================
    // TEST 2: Multi-word packet
    // ========================================================
    testNum = 2;
    $display("Starting test %d", testNum);
    

    payload[7] = 8'hAA;
    payload[6] = 8'hBB;
    payload[5] = 8'hCC;
    payload[4] = 8'hDD;
    payload[3] = 8'h11;
    payload[2] = 8'h22;
    payload[1] = 8'h33;
    payload[0] = 8'h44;

    fork
        // thread 1
        send_packet(8'h02, 16'h2000, 8);

        // thread 2
        begin
            // first word
            wait(word_valid);
            if (word_data !== 32'h11223344) begin
                $display("TEST2 FAIL WORD1: word_data=%h", word_data);
                errors = errors + 1;
            end

            // second word
            #20;
            wait(word_valid);
            if (word_data !== 32'hAABBCCDD) begin
                $display("TEST2 FAIL WORD2: word_data=%h", word_data);
                errors = errors + 1;
            end
        end
    join

    

    // ========================================================
    // TEST 3: ACK behavior
    // ========================================================
    testNum = 3;
    $display("Starting test %d", testNum);

    write_done = 1;
    #20;
    write_done = 0;

    if (send_ack !== 1) begin
        $display("ACK FAIL");
        errors = errors + 1;
    end

    // ========================================================
    // TEST 4: NACK case (bad CRC)
    // ========================================================
    testNum = 4;
    $display("Starting test %d", testNum);

    // corrupt payload (wrong CRC)
    send_byte(8'hAA);
    send_byte(8'h05);
    send_byte(8'h00);
    send_byte(8'h10);
    send_byte(8'h00);
    send_byte(8'h01);
    send_byte(8'h01);
    fork
        // thread 1
        send_byte(8'hFF); // wrong CRC

        // thread 2
        begin
            wait(send_nack == 1);
            if (send_nack !== 1) begin
                $display("NACK FAIL");
                errors = errors + 1;
            end
        end
    join

    write_done = 1;
    #20;
    write_done = 0;

    // ========================================================
    // TEST 5: NACK after failed write
    // ========================================================
    testNum = 5;
    $display("Starting test %d", testNum);

    payload[0] = 8'h11;
    payload[1] = 8'h22;
    payload[2] = 8'h33;
    payload[3] = 8'h44;

    fork
        // thread 1
        send_packet(8'h01, 16'h1000, 4);

        // thread 2
        begin
            // wait for word
            wait(word_valid == 1);

            if (word_data !== 32'h44332211) begin
                $display("TEST5 FAIL: word_data=%h", word_data);
                errors = errors + 1;
            end
        end
    join

    write_done = 0;

    fork
        // thread 1
        write_fail = 1;

        // thread 2
        begin
            wait(send_nack == 1);
            if (send_nack !== 1) begin
                $display("NACK FAIL");
                errors = errors + 1;
            end
        end
    join


    // ========================================================
    // DONE
    // ========================================================
    #20;

    if (errors == 0)
        $display("ALL TESTS PASSED");
    else
        $display("TESTS FAILED: %0d errors", errors);

    $finish;
end

endmodule
