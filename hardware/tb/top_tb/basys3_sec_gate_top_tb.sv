`timescale 1ns/1ps

module basys3_sec_gate_top_tb;
    localparam integer CLK_FREQ = 100_000_000;
    localparam integer BAUD_RATE = 10_000_000;
    localparam integer BAUD_PERIOD_NS = 100;
    localparam integer ADDR_WIDTH = 16;
    localparam integer DATA_WIDTH = 32;
    localparam integer MEMORY_BANK_DEPTH = 64;
    localparam integer PROG_ADDR_WIDTH = 8;
    localparam integer CONST_ADDR_WIDTH = 8;
    localparam integer WARP_SIZE = 4;
    localparam integer NUM_GOLDEN_HASHES = 4;

    localparam [7:0] COM_CMD_WRITE_DATA     = 8'h01;
    localparam [7:0] COM_CMD_READ_DATA      = 8'h02;
    localparam [7:0] COM_CMD_SECURITY_RESET = 8'h06;
    localparam [7:0] COM_CMD_VALIDATE       = 8'h07;
    localparam [7:0] COM_CMD_ACK            = 8'h08;
    localparam [7:0] COM_CMD_NAK            = 8'h09;

    localparam [255:0] HASH_4_WORDS =
        256'h3067c72c5e501c31e3feca73f047dc341a956399ec705e0aee9efb17a1553578;
    localparam [255:0] HASH_8_WORDS =
        256'hbdb32f8604eafe89ad767fe7fe8ccd29ecc5d0de9b7a3c9d95e3cced553d625a;
    localparam [(NUM_GOLDEN_HASHES*256)-1:0] GOLDEN_HASHES = {
        256'h0,
        256'h0,
        HASH_8_WORDS,
        HASH_4_WORDS
    };

    reg CLK100MHZ;
    reg btnC;
    reg [15:0] sw;
    reg RsRx;
    wire RsTx;
    wire [15:0] led;
    wire [6:0] seg;
    wire dp;
    wire [3:0] an;

    integer errors;
    integer test_num;
    integer i;
    integer timeout_count;
    integer rx_len_seen;
    reg [7:0] payload [0:255];
    reg [7:0] rx_payload [0:255];

    basys3_sec_gate_top #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .MEMORY_BANK_DEPTH(MEMORY_BANK_DEPTH),
        .PROG_ADDR_WIDTH(PROG_ADDR_WIDTH),
        .CONST_ADDR_WIDTH(CONST_ADDR_WIDTH),
        .WARP_SIZE(WARP_SIZE),
        .NUM_CORES(1),
        .NUM_WARPS_PER_CORE(1),
        .WARP_ID_WIDTH(1),
        .NUM_GOLDEN_HASHES(NUM_GOLDEN_HASHES),
        .GOLDEN_HASHES(GOLDEN_HASHES)
    ) dut (
        .CLK100MHZ(CLK100MHZ),
        .btnC(btnC),
        .sw(sw),
        .RsRx(RsRx),
        .RsTx(RsTx),
        .led(led),
        .seg(seg),
        .dp(dp),
        .an(an)
    );

    always #5 CLK100MHZ = ~CLK100MHZ;

    task fail;
        input [1023:0] msg;
        begin
            $display("TEST%0d FAIL: %0s", test_num, msg);
            errors = errors + 1;
        end
    endtask

    task expect_eq32;
        input [1023:0] label;
        input [31:0] got;
        input [31:0] expected;
        begin
            if (got !== expected) begin
                $display("TEST%0d FAIL: %0s got=0x%08h expected=0x%08h",
                         test_num, label, got, expected);
                errors = errors + 1;
            end
        end
    endtask

    task expect_eq16;
        input [1023:0] label;
        input [15:0] got;
        input [15:0] expected;
        begin
            if (got !== expected) begin
                $display("TEST%0d FAIL: %0s got=0x%04h expected=0x%04h",
                         test_num, label, got, expected);
                errors = errors + 1;
            end
        end
    endtask

    task wait_not_reset;
        begin
            timeout_count = 0;
            while ((led[0] !== 1'b1) && (timeout_count < 80000)) begin
                @(posedge CLK100MHZ);
                timeout_count = timeout_count + 1;
            end
            if (led[0] !== 1'b1) begin
                fail("power-on reset did not release");
            end
        end
    endtask

    task wait_security_state;
        input [2:0] expected;
        begin
            timeout_count = 0;
            while ((dut.security_state !== expected) && (timeout_count < 100000)) begin
                @(posedge CLK100MHZ);
                timeout_count = timeout_count + 1;
            end
            if (dut.security_state !== expected) begin
                $display("TEST%0d FAIL: security_state got=%0d expected=%0d",
                         test_num, dut.security_state, expected);
                errors = errors + 1;
            end
        end
    endtask

    task send_uart_byte;
        input [7:0] data;
        integer bit_idx;
        begin
            RsRx = 1'b0;
            #(BAUD_PERIOD_NS);
            for (bit_idx = 0; bit_idx < 8; bit_idx = bit_idx + 1) begin
                RsRx = data[bit_idx];
                #(BAUD_PERIOD_NS);
            end
            RsRx = 1'b1;
            #(BAUD_PERIOD_NS);
        end
    endtask

    task recv_uart_byte;
        output [7:0] data;
        integer bit_idx;
        begin
            @(negedge RsTx);
            #(BAUD_PERIOD_NS + (BAUD_PERIOD_NS / 2));
            for (bit_idx = 0; bit_idx < 8; bit_idx = bit_idx + 1) begin
                data[bit_idx] = RsTx;
                #(BAUD_PERIOD_NS);
            end
            if (RsTx !== 1'b1) begin
                fail("UART TX stop bit was not high");
            end
            #(BAUD_PERIOD_NS / 2);
        end
    endtask

    function [7:0] crc_update;
        input [7:0] crc;
        input [7:0] data;
        begin
            crc_update = crc ^ data;
        end
    endfunction

    task clear_payload;
        begin
            for (i = 0; i < 256; i = i + 1) begin
                payload[i] = 8'h00;
                rx_payload[i] = 8'h00;
            end
        end
    endtask

    task put_word_le;
        input integer offset;
        input [31:0] word;
        begin
            payload[offset + 0] = word[7:0];
            payload[offset + 1] = word[15:8];
            payload[offset + 2] = word[23:16];
            payload[offset + 3] = word[31:24];
        end
    endtask

    function [31:0] get_rx_word_le;
        input integer offset;
        begin
            get_rx_word_le = {
                rx_payload[offset + 3],
                rx_payload[offset + 2],
                rx_payload[offset + 1],
                rx_payload[offset + 0]
            };
        end
    endfunction

    task send_packet;
        input [7:0] cmd;
        input [15:0] addr;
        input [15:0] len;
        integer idx;
        reg [7:0] crc;
        begin
            crc = 8'h00;
            send_uart_byte(8'haa);
            send_uart_byte(cmd);
            crc = crc_update(crc, cmd);
            send_uart_byte(addr[15:8]);
            crc = crc_update(crc, addr[15:8]);
            send_uart_byte(addr[7:0]);
            crc = crc_update(crc, addr[7:0]);
            send_uart_byte(len[15:8]);
            crc = crc_update(crc, len[15:8]);
            send_uart_byte(len[7:0]);
            crc = crc_update(crc, len[7:0]);
            for (idx = 0; idx < len; idx = idx + 1) begin
                send_uart_byte(payload[idx]);
                crc = crc_update(crc, payload[idx]);
            end
            send_uart_byte(crc);
        end
    endtask

    task recv_packet;
        output [7:0] cmd;
        output [15:0] addr;
        output [15:0] len;
        integer idx;
        reg [7:0] b;
        reg [7:0] crc;
        reg [7:0] got_crc;
        begin
            crc = 8'h00;
            recv_uart_byte(b);
            if (b !== 8'haa) begin
                $display("TEST%0d FAIL: response SOF got=0x%02h expected=0xaa", test_num, b);
                errors = errors + 1;
            end
            recv_uart_byte(cmd);
            crc = crc_update(crc, cmd);
            recv_uart_byte(b);
            addr[15:8] = b;
            crc = crc_update(crc, b);
            recv_uart_byte(b);
            addr[7:0] = b;
            crc = crc_update(crc, b);
            recv_uart_byte(b);
            len[15:8] = b;
            crc = crc_update(crc, b);
            recv_uart_byte(b);
            len[7:0] = b;
            crc = crc_update(crc, b);
            for (idx = 0; idx < len; idx = idx + 1) begin
                recv_uart_byte(rx_payload[idx]);
                crc = crc_update(crc, rx_payload[idx]);
            end
            recv_uart_byte(got_crc);
            if (got_crc !== crc) begin
                $display("TEST%0d FAIL: response CRC got=0x%02h expected=0x%02h",
                         test_num, got_crc, crc);
                errors = errors + 1;
            end
            rx_len_seen = len;
        end
    endtask

    task transact;
        input [7:0] tx_cmd;
        input [15:0] tx_addr;
        input [15:0] tx_len;
        input [7:0] exp_cmd;
        input [15:0] exp_addr;
        input [15:0] exp_len;
        reg [7:0] got_cmd;
        reg [15:0] got_addr;
        reg [15:0] got_len;
        begin
            fork
                send_packet(tx_cmd, tx_addr, tx_len);
                recv_packet(got_cmd, got_addr, got_len);
            join
            if (got_cmd !== exp_cmd) begin
                $display("TEST%0d FAIL: response cmd got=0x%02h expected=0x%02h",
                         test_num, got_cmd, exp_cmd);
                errors = errors + 1;
            end
            expect_eq16("response addr", got_addr, exp_addr);
            expect_eq16("response len", got_len, exp_len);
        end
    endtask

    task fill_weight_payload_4words;
        begin
            put_word_le(0, 32'd0);
            put_word_le(4, 32'd1);
            put_word_le(8, 32'd2);
            put_word_le(12, 32'd3);
        end
    endtask

    task load_four_weights;
        begin
            clear_payload();
            fill_weight_payload_4words();
            transact(COM_CMD_WRITE_DATA, 16'h0000, 16'd16, COM_CMD_ACK, 16'h0000, 16'd0);
        end
    endtask

    initial begin
        #40_000_000;
        $display("BASYS3 SEC GATE TOP TESTS TIMED OUT");
        $finish;
    end

    initial begin
        $dumpfile("tmp/basys3_sec_gate_top_tb.vcd");
        $dumpvars(1, basys3_sec_gate_top_tb);

        CLK100MHZ = 1'b0;
        btnC = 1'b1;
        sw = 16'h0000;
        RsRx = 1'b1;
        errors = 0;
        test_num = 0;
        clear_payload();

        repeat (8) @(posedge CLK100MHZ);
        btnC = 1'b0;
        wait_not_reset();

        test_num = 1;
        if (led[0] !== 1'b1) fail("LED0 should indicate reset released");
        if (led[1] !== 1'b1) fail("LED1 should mirror idle RsRx");
        if (dp !== 1'b1) fail("seven-segment decimal point should be off");
        if (an === 4'bxxxx || seg === 7'bxxxxxxx) fail("seven-segment outputs contain X");

        test_num = 2;
        load_four_weights();
        if (dut.security_state !== 3'd1) fail("security gate did not enter LOAD after weights");
        transact(COM_CMD_VALIDATE, 16'h0000, 16'd0, COM_CMD_ACK, 16'h0000, 16'd0);
        wait_security_state(3'd5);
        if (dut.verified !== 1'b1) fail("security gate did not verify model 0");
        expect_eq16("protected_limit", dut.protected_limit, 16'd4);

        test_num = 3;
        clear_payload();
        transact(COM_CMD_READ_DATA, 16'h0000, 16'd4, COM_CMD_READ_DATA, 16'h0000, 16'd4);
        expect_eq32("protected host read", get_rx_word_le(0), 32'hDEADDEAD);

        test_num = 4;
        clear_payload();
        put_word_le(0, 32'hDEADBEEF);
        transact(COM_CMD_WRITE_DATA, 16'h0000, 16'd4, COM_CMD_NAK, 16'h0000, 16'd0);

        test_num = 5;
        clear_payload();
        put_word_le(0, 32'hCAFEBABE);
        transact(COM_CMD_WRITE_DATA, 16'h0010, 16'd4, COM_CMD_ACK, 16'h0010, 16'd0);
        clear_payload();
        transact(COM_CMD_READ_DATA, 16'h0010, 16'd4, COM_CMD_READ_DATA, 16'h0010, 16'd4);
        expect_eq32("scratch write/read", get_rx_word_le(0), 32'hCAFEBABE);

        test_num = 6;
        transact(COM_CMD_SECURITY_RESET, 16'h0000, 16'd0, COM_CMD_ACK, 16'h0000, 16'd0);
        repeat (5) @(posedge CLK100MHZ);
        wait_security_state(3'd0);
        if (dut.verified !== 1'b0) fail("verified still set after security reset");
        if (dut.security_fault !== 1'b0) fail("security_fault set after security reset");

        test_num = 7;
        load_four_weights();
        transact(COM_CMD_VALIDATE, 16'h0001, 16'd0, COM_CMD_ACK, 16'h0001, 16'd0);
        wait_security_state(3'd6);
        if (dut.security_fault !== 1'b1) fail("wrong model id did not assert security_fault");

        $display("\n==========================================");
        if (errors == 0) begin
            $display("basys3_sec_gate_top_tb: ALL TESTS PASSED");
        end else begin
            $display("basys3_sec_gate_top_tb: %0d ERRORS", errors);
        end
        $display("==========================================\n");
        $finish;
    end
endmodule
