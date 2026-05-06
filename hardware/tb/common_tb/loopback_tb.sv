`timescale 1ns / 1ps

module serial_loopback_tb();

reg clk_tx;
reg clk_rx;
reg rst;

// ================= TX signals =================
reg start;
reg [7:0]  tx_cmd;
reg [15:0] tx_addr;
reg [15:0] tx_len;

reg [31:0] payload_data;
reg        payload_valid;
wire       payload_advance;

wire tx_line;
wire tx_busy;

// ================= RX signals =================
wire [31:0] word_data;
wire        word_valid;
reg         word_taken;
reg ack_recieved;

wire [7:0]  rx_cmd;
wire [15:0] rx_addr;
wire [15:0] rx_len;

reg  write_done;
wire send_ack;
wire send_nack;

// ============================================================
// DUTs
// ============================================================

serial_packet_tx tx (
    .clk(clk_tx),
    .rst(rst),
    .start(start),
    .cmd(tx_cmd),
    .addr(tx_addr),
    .len(tx_len),
    .payload_data(payload_data),
    .payload_valid(payload_valid),
    .payload_advance(payload_advance),
    .tx_line(tx_line),
    .busy(tx_busy)
);

serial_packet_rx rx (
    .clk(clk_rx),
    .rst(rst),
    .rx_line(tx_line),   // LOOPBACK
    .word_data(word_data),
    .word_valid(word_valid),
    // .word_taken(word_taken),
    .cmd(rx_cmd),
    .addr(rx_addr),
    .len(rx_len),
    .write_done(write_done),
    .send_ack(send_ack),
    .send_nack(send_nack)
);

// ============================================================
// Clock
// ============================================================
reg first = 1;
always #5 clk_tx = ~clk_tx;
always begin
    if(first)begin
        first = 0;
        #500;
    end
    #5 clk_rx = ~clk_rx;
end

// ============================================================
// Payload driver
// ============================================================
reg [31:0] payload_mem;
reg payload_sent;

always @(posedge clk_tx) begin
    if (payload_advance) begin
        payload_data  <= payload_mem;
        payload_valid <= 1;
        // payload_sent  <= 1;
    end else begin
        payload_valid <= 0;
    end
end

/*
// ============================================================
// Wait for signal and timeout if it doesn't arrive
// ============================================================
// You must use 'automatic' when using 'ref' in a module-based task
task automatic wait_for_ack();
    begin : wait_block
        fork
            begin
                wait(send_ack == 1'b1);
                // $display("[%0t] Signal arrived!", $time);
                disable wait_block;
            end
            begin
                #(1000000);
                // $display("[%0t] Timeout reached!", $time);
                $display("FAIL: ACK not asserted");
                error++;
                disable wait_block;
            end
        join_any
        
    end
endtask
*/

// ============================================================
// Test
// ============================================================
integer errors;
integer testNum;

initial begin
    $dumpfile("loopback.vcd");
    $dumpvars(0, serial_loopback_tb);



    clk_tx = 0;
    clk_rx = 0;
    rst = 1;
    start = 0;
    payload_valid = 0;
    word_taken = 0;
    write_done = 0;
    payload_sent = 0;
    errors = 0;

    #20;
    rst = 0;

    // ============================================================
    // TEST 1: Basic packet
    // ============================================================
    testNum = 1;

    tx_cmd  = 8'h55;
    tx_addr = 16'h1234;
    tx_len  = 4;

    payload_mem = 32'hDEADBEEF;

    #20;
    start = 1;
    #10;
    start = 0;

    // Wait for RX to output word
    wait(word_valid);

    if (word_data !== 32'hDEADBEEF) begin
        $display("FAIL: Payload mismatch %h", word_data);
        errors++;
    end

    if (rx_cmd !== tx_cmd) begin
        $display("FAIL: CMD mismatch %h", rx_cmd);
        errors++;
    end

    if (rx_addr !== tx_addr) begin
        $display("FAIL: ADDR mismatch %h", rx_addr);
        errors++;
    end

    if (rx_len !== tx_len) begin
        $display("FAIL: LEN mismatch %h", rx_len);
        errors++;
    end

    // trigger ACK
    // wait_for_ack();
    write_done = 1;
    fork
        begin
            wait(send_ack == 1);
            ack_recieved = 1;
        end
        begin
            #(10000000);
            // $display("[%0t] Timeout reached!", $time);
            if(!ack_recieved)begin
                $display("FAIL: ACK not asserted");
                errors++;
                $finish;
            end
        end
    join
    write_done = 0;


    

    // ============================================================
    // TEST 2: Different payload
    // ============================================================
    testNum = 2;

    payload_sent = 0;

    tx_cmd  = 8'hA5;
    tx_addr = 16'hBEEF;
    tx_len  = 4;

    payload_mem = 32'h11223344;

    #50;
    start = 1;
    #10;
    start = 0;

    wait(word_valid);

    if (word_data !== 32'h11223344) begin
        $display("FAIL: Payload mismatch %h", word_data);
        errors++;
    end

    if (rx_cmd !== tx_cmd) begin
        $display("FAIL: CMD mismatch %h", rx_cmd);
        errors++;
    end

    if (rx_addr !== tx_addr) begin
        $display("FAIL: ADDR mismatch %h", rx_addr);
        errors++;
    end

    if (rx_len !== tx_len) begin
        $display("FAIL: LEN mismatch %h", rx_len);
        errors++;
    end

    // ============================================================
    // DONE
    // ============================================================

    #100;

    if (errors == 0)
        $display("LOOPBACK TEST PASSED");
    else
        $display("LOOPBACK FAILED: %0d errors", errors);

    $finish;
end

endmodule