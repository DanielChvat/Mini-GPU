`timescale 1ns/1ps

module encryption_security_controller_tb;
    localparam CLK_FREQ = 100_000_000;
    localparam BAUD_RATE = 10_000_000;
    localparam BAUD_PERIOD = 100;

    localparam SEC_CMD_INIT      = 8'hF0;
    localparam SEC_CMD_INIT_RSP  = 8'hF1;
    localparam SEC_CMD_HOST_AUTH = 8'hF2;
    localparam SEC_CMD_FPGA_AUTH = 8'hF3;

    localparam COM_CMD_WRITE_DATA = 8'h01;

    reg clk;
    reg rst;
    reg rx_line;
    wire tx_line;

    wire [31:0] rx_word_data;
    wire rx_word_valid;
    wire [7:0] rx_cmd;
    wire [15:0] rx_addr;
    wire [15:0] rx_len;
    reg write_done;
    reg write_fail;
    wire packet_done;
    wire send_ack;
    wire send_nack;

    reg tx_start;
    reg [7:0] tx_cmd;
    reg [15:0] tx_addr;
    reg [15:0] tx_len;
    reg [31:0] tx_payload_data;
    reg tx_payload_valid;
    wire tx_payload_advance;
    wire tx_busy;

    reg [7:0] payload [0:255];
    reg [7:0] rx_payload [0:255];
    reg [7:0] tag_payload [0:15];
    reg [(255*8)-1:0] payload_flat;
    reg hmac_start;
    reg [7:0] hmac_cmd;
    reg [15:0] hmac_addr;
    reg [15:0] hmac_len;
    reg [15:0] hmac_counter;
    wire [127:0] hmac_tag;
    wire hmac_tag_valid;
    wire hmac_busy;
    wire tb_sha_start;
    wire tb_sha_byte_valid;
    wire [7:0] tb_sha_byte_data;
    wire tb_sha_finish;
    wire tb_sha_ready;
    wire [255:0] tb_sha_digest;
    wire tb_sha_digest_valid;
    wire tb_sha_busy;
    wire tb_sha_error;
    reg tb_aes_enable;
    reg tb_aes_seed;
    reg tb_aes_load;
    reg tb_aes_ack;
    reg [127:0] tb_aes_in;
    wire [127:0] tb_aes_out;
    wire tb_aes_valid;
    integer i;
    integer errors;
    integer word_seen;
    reg packet_seen;

    encryption_security_controller #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) dut (
        .clk(clk),
        .rst(rst),
        .rx_line(rx_line),
        .tx_line(tx_line),
        .rx_word_data(rx_word_data),
        .rx_word_valid(rx_word_valid),
        .rx_cmd(rx_cmd),
        .rx_addr(rx_addr),
        .rx_len(rx_len),
        .write_done(write_done),
        .write_fail(write_fail),
        .packet_done(packet_done),
        .send_ack(send_ack),
        .send_nack(send_nack),
        .tx_start(tx_start),
        .tx_cmd(tx_cmd),
        .tx_addr(tx_addr),
        .tx_len(tx_len),
        .tx_payload_data(tx_payload_data),
        .tx_payload_valid(tx_payload_valid),
        .tx_payload_advance(tx_payload_advance),
        .tx_busy(tx_busy)
    );

    hmac_sha256_packet #(
        .MAX_PAYLOAD(255)
    ) tb_hmac (
        .clk(clk),
        .rst(rst),
        .start(hmac_start),
        .hmac_key(dut.hmac_key),
        .cmd(hmac_cmd),
        .addr(hmac_addr),
        .len(hmac_len),
        .packet_counter(hmac_counter),
        .payload_flat(payload_flat),
        .tag(hmac_tag),
        .tag_valid(hmac_tag_valid),
        .busy(hmac_busy),
        .sha_start(tb_sha_start),
        .sha_byte_valid(tb_sha_byte_valid),
        .sha_byte_data(tb_sha_byte_data),
        .sha_finish(tb_sha_finish),
        .sha_ready(tb_sha_ready),
        .sha_digest(tb_sha_digest),
        .sha_digest_valid(tb_sha_digest_valid)
    );

    sha256_byte_stream tb_sha (
        .clk(clk),
        .rst(rst),
        .start(tb_sha_start),
        .byte_valid(tb_sha_byte_valid),
        .byte_data(tb_sha_byte_data),
        .finish(tb_sha_finish),
        .busy(tb_sha_busy),
        .ready(tb_sha_ready),
        .digest(tb_sha_digest),
        .digest_valid(tb_sha_digest_valid),
        .error(tb_sha_error)
    );

    aes_ctr_wrapper tb_aes (
        .clk(clk),
        .rst(rst),
        .enable(tb_aes_enable),
        .key(dut.aes_key_h2f),
        .nonce({80'b0, hmac_counter}),
        .seed(tb_aes_seed),
        .data_in(tb_aes_in),
        .load(tb_aes_load),
        .data_out(tb_aes_out),
        .data_valid(tb_aes_valid),
        .data_ack(tb_aes_ack)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (rst) begin
            word_seen <= 0;
            packet_seen <= 1'b0;
        end else begin
            if (rx_word_valid) begin
                word_seen <= rx_word_data;
            end
            if (packet_done) begin
                packet_seen <= 1'b1;
            end
        end
    end

    task fail;
        input [1023:0] msg;
        begin
            $display("FAIL: %0s", msg);
            errors = errors + 1;
        end
    endtask

    function [7:0] crc_update;
        input [7:0] crc;
        input [7:0] data;
        begin
            crc_update = crc ^ data;
        end
    endfunction

    task send_uart_byte;
        input [7:0] data;
        integer bit_idx;
        begin
            rx_line = 1'b0;
            #(BAUD_PERIOD);
            for (bit_idx = 0; bit_idx < 8; bit_idx = bit_idx + 1) begin
                rx_line = data[bit_idx];
                #(BAUD_PERIOD);
            end
            rx_line = 1'b1;
            #(BAUD_PERIOD);
        end
    endtask

    task recv_uart_byte;
        output [7:0] data;
        integer bit_idx;
        begin
            @(negedge tx_line);
            #(BAUD_PERIOD + (BAUD_PERIOD / 2));
            for (bit_idx = 0; bit_idx < 8; bit_idx = bit_idx + 1) begin
                data[bit_idx] = tx_line;
                #(BAUD_PERIOD);
            end
            #(BAUD_PERIOD / 2);
        end
    endtask

    task send_plain_packet;
        input [7:0] cmd;
        input [15:0] addr;
        input [15:0] len;
        integer idx;
        reg [7:0] crc;
        begin
            crc = 8'h00;
            send_uart_byte(8'hAA);
            send_uart_byte(cmd); crc = crc_update(crc, cmd);
            send_uart_byte(addr[15:8]); crc = crc_update(crc, addr[15:8]);
            send_uart_byte(addr[7:0]); crc = crc_update(crc, addr[7:0]);
            send_uart_byte(len[15:8]); crc = crc_update(crc, len[15:8]);
            send_uart_byte(len[7:0]); crc = crc_update(crc, len[7:0]);
            for (idx = 0; idx < len; idx = idx + 1) begin
                send_uart_byte(payload[idx]);
                crc = crc_update(crc, payload[idx]);
            end
            send_uart_byte(crc);
        end
    endtask

    task recv_plain_packet;
        output [7:0] cmd;
        output [15:0] addr;
        output [15:0] len;
        integer idx;
        reg [7:0] b;
        reg [7:0] crc;
        begin
            crc = 8'h00;
            recv_uart_byte(b);
            if (b !== 8'hAA) fail("missing SOF");
            recv_uart_byte(cmd); crc = crc_update(crc, cmd);
            recv_uart_byte(b); addr[15:8] = b; crc = crc_update(crc, b);
            recv_uart_byte(b); addr[7:0] = b; crc = crc_update(crc, b);
            recv_uart_byte(b); len[15:8] = b; crc = crc_update(crc, b);
            recv_uart_byte(b); len[7:0] = b; crc = crc_update(crc, b);
            for (idx = 0; idx < len; idx = idx + 1) begin
                recv_uart_byte(rx_payload[idx]);
                crc = crc_update(crc, rx_payload[idx]);
            end
            recv_uart_byte(b);
            if (b !== crc) fail("bad plaintext crc");
        end
    endtask

    task send_secure_packet;
        input [7:0] cmd;
        input [15:0] addr;
        input [15:0] len;
        input [15:0] counter;
        integer idx;
        integer wait_count;
        begin
            send_uart_byte(8'hAA);
            send_uart_byte(cmd);
            send_uart_byte(addr[15:8]);
            send_uart_byte(addr[7:0]);
            send_uart_byte(len[15:8]);
            send_uart_byte(len[7:0]);
            send_uart_byte(counter[15:8]);
            send_uart_byte(counter[7:0]);
            for (idx = 0; idx < len; idx = idx + 1)
                send_uart_byte(payload[idx]);
            for (idx = 0; idx < 16; idx = idx + 1)
                send_uart_byte(tag_payload[idx]);
        end
    endtask

    task compute_tag;
        input [7:0] cmd;
        input [15:0] addr;
        input [15:0] len;
        input [15:0] counter;
        integer idx;
        integer wait_count;
        begin
            payload_flat = {(255*8){1'b0}};
            for (idx = 0; idx < len; idx = idx + 1)
                payload_flat[idx * 8 +: 8] = payload[idx];
            hmac_cmd = cmd;
            hmac_addr = addr;
            hmac_len = len;
            hmac_counter = counter;
            @(posedge clk);
            hmac_start = 1'b1;
            @(posedge clk);
            hmac_start = 1'b0;
            wait_count = 0;
            while (!hmac_tag_valid && wait_count < 20000) begin
                @(posedge clk);
                wait_count = wait_count + 1;
            end
            if (!hmac_tag_valid) begin
                fail("hmac compute timeout");
            end
            for (idx = 0; idx < 16; idx = idx + 1)
                tag_payload[idx] = hmac_tag[(15 - idx) * 8 +: 8];
        end
    endtask

    task encrypt_payload;
        input [15:0] len;
        input [15:0] counter;
        integer idx;
        begin
            hmac_counter = counter;
            tb_aes_in = 128'b0;
            for (idx = 0; idx < 16; idx = idx + 1) begin
                if (idx < len)
                    tb_aes_in[idx * 8 +: 8] = payload[idx];
            end
            @(posedge clk);
            tb_aes_seed = 1'b1;
            @(posedge clk);
            tb_aes_seed = 1'b0;
            repeat (80) @(posedge clk);
            tb_aes_load = 1'b1;
            @(posedge clk);
            tb_aes_load = 1'b0;
            while (!tb_aes_valid)
                @(posedge clk);
            for (idx = 0; idx < 16; idx = idx + 1) begin
                if (idx < len)
                    payload[idx] = tb_aes_out[idx * 8 +: 8];
            end
            tb_aes_ack = 1'b1;
            @(posedge clk);
            tb_aes_ack = 1'b0;
        end
    endtask

    initial begin
        $dumpfile("tmp/encryption_security_controller_tb.vcd");
        $dumpvars(0, encryption_security_controller_tb);

        clk = 1'b0;
        rst = 1'b1;
        rx_line = 1'b1;
        write_done = 1'b0;
        write_fail = 1'b0;
        tx_start = 1'b0;
        tx_cmd = 8'h00;
        tx_addr = 16'h0000;
        tx_len = 16'h0000;
        tx_payload_data = 32'h00000000;
        tx_payload_valid = 1'b0;
        tb_aes_enable = 1'b1;
        tb_aes_seed = 1'b0;
        tb_aes_load = 1'b0;
        tb_aes_ack = 1'b0;
        tb_aes_in = 128'b0;
        hmac_start = 1'b0;
        hmac_cmd = 8'h00;
        hmac_addr = 16'h0000;
        hmac_len = 16'h0000;
        hmac_counter = 16'h0000;
        payload_flat = {(255*8){1'b0}};
        errors = 0;
        for (i = 0; i < 256; i = i + 1) begin
            payload[i] = 8'h00;
            rx_payload[i] = 8'h00;
            if (i < 16)
                tag_payload[i] = 8'h00;
        end

        repeat (10) @(posedge clk);
        rst = 1'b0;
        repeat (2000) @(posedge clk);

        for (i = 0; i < 16; i = i + 1)
            payload[i] = i[7:0];
        fork
            begin
                send_plain_packet(SEC_CMD_INIT, 16'h0000, 16'd16);
            end
            begin
                reg [7:0] cmd;
                reg [15:0] addr;
                reg [15:0] len;
                recv_plain_packet(cmd, addr, len);
                if (cmd !== SEC_CMD_INIT_RSP) fail("expected INIT_RSP");
                if (len !== 16'd16) fail("INIT_RSP length");
            end
        join

        for (i = 0; i < 16; i = i + 1)
            payload[i] = 8'hA0 + i[7:0];
        fork
            begin
                send_plain_packet(SEC_CMD_HOST_AUTH, 16'h0000, 16'd16);
            end
            begin
                reg [7:0] cmd;
                reg [15:0] addr;
                reg [15:0] len;
                recv_plain_packet(cmd, addr, len);
                if (cmd !== SEC_CMD_FPGA_AUTH) fail("expected FPGA_AUTH");
                if (len !== 16'd16) fail("FPGA_AUTH length");
            end
        join

        payload[0] = 8'h44;
        payload[1] = 8'h33;
        payload[2] = 8'h22;
        payload[3] = 8'h11;
        encrypt_payload(16'd4, 16'd1);
        compute_tag(COM_CMD_WRITE_DATA, 16'h0040, 16'd4, 16'd1);
        packet_seen = 1'b0;
        send_secure_packet(COM_CMD_WRITE_DATA, 16'h0040, 16'd4, 16'd1);
        repeat (25000) @(posedge clk);
        if (!packet_seen) fail("secure packet was not released");
        if (word_seen !== 32'h11223344) fail("released word mismatch");

        packet_seen = 1'b0;
        send_secure_packet(COM_CMD_WRITE_DATA, 16'h0040, 16'd4, 16'd1);
        repeat (25000) @(posedge clk);
        if (packet_seen) fail("replayed packet was released");

        if (errors == 0)
            $display("encryption_security_controller_tb PASS");
        else
            $display("encryption_security_controller_tb FAIL errors=%0d", errors);
        $finish;
    end

    initial begin
        #20_000_000;
        fail("timeout");
        $finish;
    end
endmodule
