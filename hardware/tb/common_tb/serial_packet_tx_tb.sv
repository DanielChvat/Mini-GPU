`timescale 1ns / 1ps

module serial_packet_tx_tb();

reg clk;
reg rst;

// TX inputs
reg start;
reg [7:0]  cmd;
reg [15:0] addr;
reg [15:0] len;

reg [31:0] payload_data;
reg        payload_valid;
wire       payload_advance;

// TX outputs
wire tx_line;
wire busy;

integer errors;
integer testNum;

// DUT
serial_packet_tx dut (
    .clk(clk),
    .rst(rst),
    .start(start),
    .cmd(cmd),
    .addr(addr),
    .len(len),
    .payload_data(payload_data),
    .payload_valid(payload_valid),
    .payload_advance(payload_advance),
    .tx_line(tx_line),
    .busy(busy)
);

// ============================================================
// Clock (100 MHz)
// ============================================================
always #5 clk = ~clk;

// ============================================================
// Baud timing (match DUT)
// ============================================================
localparam BAUD_PERIOD = 8680; // ns for ~115200

// ============================================================
// Serial capture (bit → byte)
// ============================================================
reg [7:0] rx_shift;
reg [2:0] bit_cnt;
reg [7:0] rx_byte;
reg       byte_valid;
reg first;

initial begin
    bit_cnt = 0;
    rx_shift = 0;
    first = 1;
    byte_valid = 0;
    // payload_idx = 0;
end

always begin
    if(first)begin
        #500;
        first = 0;
    end
    #(BAUD_PERIOD);
    

    if (bit_cnt == 3'd7) begin
        rx_byte   = {tx_line, rx_shift[7:1]};
        byte_valid = 1;
        bit_cnt = 0;
    end else begin
        rx_shift = {tx_line, rx_shift[7:1]};
        bit_cnt  = bit_cnt + 1;
        byte_valid = 0;
    end
end

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
// Payload driver (respond to payload_advance)
// ============================================================
reg [31:0] payload_mem [0:3];
integer payload_idx = 0;

always @(posedge clk) begin
    if (payload_advance) begin
        payload_data  <= payload_mem[payload_idx];
        payload_valid <= 1;
        // payload_idx   <= payload_idx + 1;
    end else begin
        payload_valid <= 0;
    end
end

// ============================================================
// Packet checker
// ============================================================
task expect_packet(
    input [7:0]  exp_cmd,
    input [15:0] exp_addr,
    input [15:0] exp_len
);
    integer i;
    reg [7:0] crc;
    reg [7:0] b;
begin
    crc = 0;

    // SOF
    wait(byte_valid);
    if (rx_byte !== 8'hAA) begin
        errors++;
        $display("ERROR %d: SOF %h", errors, rx_byte);
        
    end

    // CMD
    #(BAUD_PERIOD);
    wait(byte_valid);
    if (rx_byte !== exp_cmd) begin
        errors++;
        $display("ERROR %d: CMD %h", errors, rx_byte);
    end
    crc = crc_calc(crc, rx_byte);

    // ADDR HI
    #(BAUD_PERIOD);
    wait(byte_valid);
    if (rx_byte !== exp_addr[15:8]) begin
        errors++;
        $display("ERROR %d: ADDR HI %h", errors, rx_byte);
    end
    crc = crc_calc(crc, rx_byte);

    // ADDR LO
    #(BAUD_PERIOD);
    wait(byte_valid);
    if (rx_byte !== exp_addr[7:0]) begin
        errors++;
        $display("ERROR %d: ADDR LO %h", errors, rx_byte);
    end
    crc = crc_calc(crc, rx_byte);

    // LEN HI
    #(BAUD_PERIOD);
    wait(byte_valid);
    if (rx_byte !== exp_len[15:8]) begin
        errors++;
        $display("ERROR %d: LEN HI %h", errors, rx_byte);
    end
    crc = crc_calc(crc, rx_byte);

    // LEN LO
    #(BAUD_PERIOD);
    wait(byte_valid);
    if (rx_byte !== exp_len[7:0]) begin
        errors++;
        $display("ERROR %d: LEN LO %h", errors, rx_byte);
    end
    crc = crc_calc(crc, rx_byte);

    // PAYLOAD (4 bytes assumed)
    for (i = 0; i < exp_len; i = i + 1) begin
        #(BAUD_PERIOD);
        wait(byte_valid);
        // if (rx_byte !== payload_mem[i][7:0]) begin
        //     errors++;
        //     $display("ERROR %d: PAYLOAD[%d] BYTE %h", errors, i, rx_byte);
        // end
        crc = crc_calc(crc, rx_byte);
    end

    // CRC
    #(BAUD_PERIOD);
    wait(byte_valid);
    if (rx_byte !== crc) begin
        errors++;
        $display("ERROR %d: CRC expected %h got %h", errors, crc, rx_byte);
    end
end
endtask

// ============================================================
// TESTS
// ============================================================
initial begin
    $dumpfile("tx_wave.vcd");
    $dumpvars(0, serial_packet_tx_tb);

    clk = 0;
    rst = 1;
    start = 0;
    payload_valid = 0;
    errors = 0;

    #20;
    rst = 0;

    // ============================================================
    // TEST 1: 4-byte payload
    // ============================================================
    testNum = 1;

    cmd  = 8'h01;
    addr = 16'h1000;
    len  = 4;

    payload_mem[0] = 32'h44332211;
    // payload_valid = 1;
    payload_idx = 0;

    #20;
    start = 1;
    #10;
    start = 0;
    #500;
    bit_cnt = 0;
    

    fork
        expect_packet(cmd, addr, len);
    join

    // ============================================================
    // TEST 2: zero-length payload
    // ============================================================
    testNum = 2;

    cmd  = 8'h02;
    addr = 16'h2000;
    len  = 0;

    #20;
    start = 1;
    #10;
    start = 0;
    #500;
    bit_cnt = 0;
    #(BAUD_PERIOD);
    fork
        expect_packet(cmd, addr, len);
    join

    // ============================================================
    // DONE
    // ============================================================
    #100;

    if (errors == 0)
        $display("ALL TX TESTS PASSED");
    else
        $display("TX TESTS FAILED: %0d errors", errors);

    $finish;
end

endmodule