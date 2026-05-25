`timescale 1ns/1ps

`include "minigpu_isa.vh"

module basys3_security_top_tb;
    localparam integer CLK_FREQ = 100_000_000;
    localparam integer BAUD_RATE = 10_000_000;
    localparam integer BAUD_PERIOD_NS = 100;
    localparam integer ADDR_WIDTH = 16;
    localparam integer DATA_WIDTH = 32;
    localparam integer MEMORY_BANK_DEPTH = 64;
    localparam integer PROG_ADDR_WIDTH = 8;
    localparam integer CONST_ADDR_WIDTH = 8;
    localparam integer WARP_SIZE = 4;

    localparam [7:0] COM_CMD_WRITE_DATA      = 8'h01;
    localparam [7:0] COM_CMD_READ_DATA       = 8'h02;
    localparam [7:0] COM_CMD_WRITE_PROGRAM   = 8'h03;
    localparam [7:0] COM_CMD_LAUNCH          = 8'h04;
    localparam [7:0] COM_CMD_READ_STATUS     = 8'h05;
    localparam [7:0] COM_CMD_SECURITY_RESET  = 8'h06;
    localparam [7:0] COM_CMD_VALIDATE        = 8'h07;
    localparam [7:0] COM_CMD_ACK             = 8'h08;
    localparam [7:0] COM_CMD_NAK             = 8'h09;
    localparam [7:0] COM_CMD_WRITE_CONSTANTS = 8'h0b;

    // SHA-256 of the 12 vector-add kernel instruction words (big-endian)
    localparam [255:0] VECTOR_ADD_HASH =
        256'hae79b5465f8e02713971535685574c9774b58d60a5d78484b665d746b9e6928f;

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
    integer status_polls;
    integer rx_len_seen;
    reg [7:0] payload [0:255];
    reg [7:0] rx_payload [0:255];
    reg [31:0] status_word;

    basys3_security_top #(
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
        .WARP_ID_WIDTH(1)
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
        integer cycles;
        begin
            cycles = 0;
            while ((led[0] !== 1'b1) && (cycles < 80000)) begin
                @(posedge CLK100MHZ);
                cycles = cycles + 1;
            end
            if (led[0] !== 1'b1) begin
                fail("power-on reset did not release");
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

    task send_bad_crc_packet;
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
            send_uart_byte(crc ^ 8'hff);
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

    task transact_bad_crc;
        input [7:0] tx_cmd;
        input [15:0] tx_addr;
        input [15:0] tx_len;
        reg [7:0] got_cmd;
        reg [15:0] got_addr;
        reg [15:0] got_len;
        begin
            fork
                send_bad_crc_packet(tx_cmd, tx_addr, tx_len);
                recv_packet(got_cmd, got_addr, got_len);
            join
            if (got_cmd !== COM_CMD_NAK) begin
                $display("TEST%0d FAIL: bad CRC response cmd got=0x%02h expected=NAK", test_num, got_cmd);
                errors = errors + 1;
            end
            expect_eq16("bad CRC response addr", got_addr, tx_addr);
            expect_eq16("bad CRC response len", got_len, 16'd0);
        end
    endtask

    task write_words;
        input [15:0] byte_addr;
        input integer word_count;
        input [31:0] base_word;
        begin
            clear_payload();
            for (i = 0; i < word_count; i = i + 1) begin
                put_word_le(i * 4, base_word + i);
            end
            transact(COM_CMD_WRITE_DATA, byte_addr, word_count * 4, COM_CMD_ACK, byte_addr, 16'd0);
        end
    endtask

    task read_words_expect_incrementing;
        input [15:0] byte_addr;
        input integer word_count;
        input [31:0] base_word;
        begin
            clear_payload();
            transact(COM_CMD_READ_DATA, byte_addr, word_count * 4, COM_CMD_READ_DATA, byte_addr, word_count * 4);
            for (i = 0; i < word_count; i = i + 1) begin
                expect_eq32("READ_DATA word", get_rx_word_le(i * 4), base_word + i);
            end
        end
    endtask

    task read_vector_add_expect;
        input [15:0] byte_addr;
        input integer word_count;
        input [31:0] lhs_base;
        input [31:0] rhs_base;
        begin
            clear_payload();
            transact(COM_CMD_READ_DATA, byte_addr, word_count * 4, COM_CMD_READ_DATA, byte_addr, word_count * 4);
            for (i = 0; i < word_count; i = i + 1) begin
                expect_eq32("vector-add output word",
                            get_rx_word_le(i * 4),
                            lhs_base + rhs_base + (i * 2));
            end
        end
    endtask

    function [31:0] pack_instr;
        input [5:0] op;
        input [3:0] rd;
        input [3:0] rs1;
        input [3:0] rs2;
        input [13:0] imm14;
        begin
            pack_instr = {op, rd, rs1, rs2, imm14};
        end
    endfunction

    task write_program_word;
        input [15:0] instr_addr;
        input [31:0] instr;
        begin
            clear_payload();
            put_word_le(0, instr);
            transact(COM_CMD_WRITE_PROGRAM, instr_addr << 2, 16'd4,
                     COM_CMD_ACK, instr_addr << 2, 16'd0);
        end
    endtask

    task write_const_word;
        input [15:0] const_addr;
        input [31:0] data;
        begin
            clear_payload();
            put_word_le(0, data);
            transact(COM_CMD_WRITE_CONSTANTS, const_addr << 2, 16'd4,
                     COM_CMD_ACK, const_addr << 2, 16'd0);
        end
    endtask

    task read_status;
        begin
            clear_payload();
            transact(COM_CMD_READ_STATUS, 16'h0000, 16'd0,
                     COM_CMD_READ_STATUS, 16'h0000, 16'd4);
            status_word = get_rx_word_le(0);
        end
    endtask

    task poll_gpu_done;
        begin
            status_word = 32'h0;
            for (status_polls = 0; status_polls < 80; status_polls = status_polls + 1) begin
                read_status();
                if (status_word[0]) begin
                    status_polls = 80;
                end else begin
                    repeat (200) @(posedge CLK100MHZ);
                end
            end
            if (!status_word[0]) begin
                fail("GPU did not report done through READ_STATUS");
            end
            if (status_word[3:1] !== 3'b000) begin
                $display("TEST%0d FAIL: status error bits set status=0x%08h", test_num, status_word);
                errors = errors + 1;
            end
            if (status_word[15:8] == 8'd0) begin
                fail("retired instruction count did not increment");
            end
        end
    endtask

    // Upload the 12-instruction vector-add kernel
    task load_vector_add_kernel;
        begin
            write_program_word(16'd0, pack_instr(`MGPU_OP_LDC,  4'd10, 4'd0,  4'd0, 14'd0));
            write_program_word(16'd1, pack_instr(`MGPU_OP_LDC,  4'd11, 4'd0,  4'd0, 14'd1));
            write_program_word(16'd2, pack_instr(`MGPU_OP_LDC,  4'd12, 4'd0,  4'd0, 14'd2));
            write_program_word(16'd3, pack_instr(`MGPU_OP_TID,  4'd1,  4'd0,  4'd0, 14'd0));
            write_program_word(16'd4, pack_instr(`MGPU_OP_ADD,  4'd2,  4'd10, 4'd1, {11'b0, `MGPU_FMT_I32}));
            write_program_word(16'd5, pack_instr(`MGPU_OP_LDG,  4'd3,  4'd2,  4'd0, 14'd0));
            write_program_word(16'd6, pack_instr(`MGPU_OP_ADD,  4'd4,  4'd11, 4'd1, {11'b0, `MGPU_FMT_I32}));
            write_program_word(16'd7, pack_instr(`MGPU_OP_LDG,  4'd5,  4'd4,  4'd0, 14'd0));
            write_program_word(16'd8, pack_instr(`MGPU_OP_ADD,  4'd6,  4'd3,  4'd5, {11'b0, `MGPU_FMT_I32}));
            write_program_word(16'd9, pack_instr(`MGPU_OP_ADD,  4'd7,  4'd12, 4'd1, {11'b0, `MGPU_FMT_I32}));
            write_program_word(16'd10, pack_instr(`MGPU_OP_STG, 4'd0,  4'd7,  4'd6, 14'd0));
            write_program_word(16'd11, pack_instr(`MGPU_OP_EXIT, 4'd0,  4'd0,  4'd0, 14'd0));
        end
    endtask

    // Validate the loaded kernel with a given kernel_id, expect ACK
    task validate_kernel_expect_ack;
        input [6:0] kernel_id;
        begin
            clear_payload();
            transact(COM_CMD_VALIDATE, {9'b0, kernel_id}, 16'd0,
                     COM_CMD_ACK, {9'b0, kernel_id}, 16'd0);
        end
    endtask

    // Validate the loaded kernel with a given kernel_id, expect NAK
    task validate_kernel_expect_nak;
        input [6:0] kernel_id;
        begin
            clear_payload();
            transact(COM_CMD_VALIDATE, {9'b0, kernel_id}, 16'd0,
                     COM_CMD_NAK, {9'b0, kernel_id}, 16'd0);
        end
    endtask

    task launch_kernel_expect;
        input [7:0] expected_cmd;
        begin
            clear_payload();
            put_word_le(0, 32'd1);
            put_word_le(4, 32'd4);
            put_word_le(8, 32'h0000000f);
            transact(COM_CMD_LAUNCH, 16'h0000, 16'd12, expected_cmd, 16'h0000, 16'd0);
        end
    endtask

    task launch_kernel;
        begin
            launch_kernel_expect(COM_CMD_ACK);
        end
    endtask

    initial begin
        #40_000_000;
        $display("BASYS3 SECURITY TOP TESTS TIMED OUT");
        $finish;
    end

    initial begin
        $dumpfile("tmp/basys3_security_top_tb.vcd");
        $dumpvars(1, basys3_security_top_tb);

        CLK100MHZ = 1'b0;
        btnC = 1'b1;
        sw = 16'h0000;
        RsRx = 1'b1;
        errors = 0;
        test_num = 0;
        status_word = 32'h0;
        clear_payload();

        repeat (8) @(posedge CLK100MHZ);
        btnC = 1'b0;
        wait_not_reset();

        // Set golden hash for kernel_id=0 to the vector-add kernel hash
        dut.u_kernel_vfy.hash_store.hash_mem[0] = VECTOR_ADD_HASH;
        // Set a different hash for kernel_id=1 (will cause mismatch)
        dut.u_kernel_vfy.hash_store.hash_mem[1] = 256'hDEADBEEF;

        // =====================================================================
        // Test 1: initial state check after reset
        // =====================================================================
        test_num = 1;
        $display("TEST%0d: Initial state check", test_num);
        if (led[0] !== 1'b1) fail("LED0 should indicate reset released");
        if (led[1] !== 1'b1) fail("LED1 should mirror idle RsRx");
        if (dp !== 1'b1) fail("seven-segment decimal point should be off");
        if (an === 4'bxxxx || seg === 7'bxxxxxxx) fail("seven-segment outputs contain X");

        // =====================================================================
        // Test 2: writing and reading memory
        // =====================================================================
        test_num = 2;
        $display("TEST%0d: Write and read memory", test_num);
        write_words(16'h0010, 4, 32'h11223344);
        if (dut.write_count !== 16'd4) begin
            $display("TEST%0d FAIL: top write_count got=%0d expected=4", test_num, dut.write_count);
            errors = errors + 1;
        end
        expect_eq16("last_mem_addr after write", dut.last_mem_addr, 16'h0007);
        expect_eq32("last_mem_wdata after write", dut.last_mem_wdata, 32'h11223347);
        read_words_expect_incrementing(16'h0010, 4, 32'h11223344);
        if (dut.read_count < 16'd4) begin
            fail("top read_count did not reflect host reads");
        end
        expect_eq32("last_mem_rdata after read", dut.last_mem_rdata, 32'h11223347);

        // =====================================================================
        // Test 3: CRC error handling
        // =====================================================================
        test_num = 3;
        $display("TEST%0d: CRC error handling", test_num);
        clear_payload();
        put_word_le(0, 32'hcafebabe);
        transact_bad_crc(COM_CMD_WRITE_DATA, 16'h0040, 16'd4);

        if (dut.u_kernel_vfy.state_reg !== 3'd6) begin
            $display("TEST%0d FAIL: kernel_vfy state got=%0d expected=6 (ERROR)",
                     test_num, dut.u_kernel_vfy.state_reg);
            errors = errors + 1;
        end

        // Security reset should clear error state
        transact(COM_CMD_SECURITY_RESET, 16'h0000, 16'd0, COM_CMD_ACK, 16'h0000, 16'd0);
        // Allow a few cycles for security_reset to propagate
        repeat (150000) @(posedge CLK100MHZ);

        // =====================================================================
        // Test 4: Kernel verification - upload, validate (ACK), security reset
        // =====================================================================
        test_num = 4;
        $display("TEST%0d: Kernel verification (upload + validate ACK + security reset)", test_num);
        // Upload vector-add kernel (kernel_vfy goes IDLE→LOAD)
        load_vector_add_kernel();
        // Validate with kernel_id=0 (golden hash matches) → expect ACK
        validate_kernel_expect_ack(7'd0);
        // Check kernel_vfy reached EXECUTE state (state=5) and is verified
        if (dut.u_kernel_vfy.state_reg !== 3'd5) begin
            $display("TEST%0d FAIL: kernel_vfy state got=%0d expected=5 (EXECUTE)",
                     test_num, dut.u_kernel_vfy.state_reg);
            errors = errors + 1;
        end
        if (dut.u_kernel_vfy.verified_reg !== 1'b1) fail("kernel_verified should be 1 after validate ACK");
        // Security reset should clear verification state
        transact(COM_CMD_SECURITY_RESET, 16'h0000, 16'd0, COM_CMD_ACK, 16'h0000, 16'd0);
        // Allow a few cycles for security_reset to propagate
        repeat (150000) @(posedge CLK100MHZ);
        if (dut.u_kernel_vfy.state_reg !== 3'd0) begin
            $display("TEST%0d FAIL: kernel_vfy state after security reset got=%0d expected=0 (IDLE)",
                     test_num, dut.u_kernel_vfy.state_reg);
            errors = errors + 1;
        end
        if (dut.u_kernel_vfy.verified_reg !== 1'b0) fail("kernel_verified should be 0 after security reset");

        // =====================================================================
        // Test 5: check debug status and display
        // =====================================================================
        test_num = 5;
        $display("TEST%0d: Debug status and display", test_num);
        sw[1:0] = 2'd0;
        transact(COM_CMD_READ_STATUS, 16'h0000, 16'd0, COM_CMD_READ_STATUS, 16'h0000, 16'd4);
        if (led[15:8] !== COM_CMD_READ_STATUS) begin
            $display("TEST%0d FAIL: LED debug cmd got=0x%02h expected=0x%02h",
                     test_num, led[15:8], COM_CMD_READ_STATUS);
            errors = errors + 1;
        end
        sw[1:0] = 2'd3;
        repeat (4) @(posedge CLK100MHZ);
        if (seg === 7'bxxxxxxx || an === 4'bxxxx) fail("display outputs invalid when selecting last memory address");

        // =====================================================================
        // Test 6: Kernel validation mismatch (NAK)
        // =====================================================================
        test_num = 6;
        $display("TEST%0d: Kernel validation mismatch (NAK)", test_num);
        // Upload kernel, validate with wrong kernel_id (id=1 has wrong hash)
        load_vector_add_kernel();
        validate_kernel_expect_nak(7'd1);
        // kernel_vfy should be in ERROR state (state=6)
        if (dut.u_kernel_vfy.state_reg !== 3'd6) begin
            $display("TEST%0d FAIL: kernel_vfy state got=%0d expected=6 (ERROR)",
                     test_num, dut.u_kernel_vfy.state_reg);
            errors = errors + 1;
        end
        if (dut.u_kernel_vfy.fault_reg !== 1'b1) fail("kernel_fault should be set after mismatch");
        // Recover via security reset
        transact(COM_CMD_SECURITY_RESET, 16'h0000, 16'd0, COM_CMD_ACK, 16'h0000, 16'd0);
        repeat (150000) @(posedge CLK100MHZ);

        // =====================================================================
        // Test 7: Launch gating - launch blocked without verification
        // =====================================================================
        test_num = 7;
        $display("TEST%0d: Launch gating (blocked without verification)", test_num);
        // Upload kernel but do NOT validate
        load_vector_add_kernel();
        // kernel_vfy is in LOAD state, not verified
        // Set up data for vector add
        write_const_word(16'd0, 32'd0);
        write_const_word(16'd1, 32'd16);
        write_const_word(16'd2, 32'd32);
        write_words(16'h0000, 4, 32'd100);
        write_words(16'h0040, 4, 32'd7);
        // Launch should be rejected before validation.
        launch_kernel_expect(COM_CMD_NAK);
        // GPU should NOT have started since launch was gated
        repeat (100) @(posedge CLK100MHZ);
        read_status();
        // gpu_done (bit 0) should NOT be set since launch was blocked
        if (status_word[0] == 1'b1 && status_word[15:8] != 8'd0) begin
            fail("GPU should not have executed without kernel verification");
        end
        // Security reset to clean up
        transact(COM_CMD_SECURITY_RESET, 16'h0000, 16'd0, COM_CMD_ACK, 16'h0000, 16'd0);
        repeat (150000) @(posedge CLK100MHZ);

        // =====================================================================
        // Test 8: Full vector addition with kernel verification
        // =====================================================================
        test_num = 8;
        $display("TEST%0d: Full vector addition with kernel verification", test_num);
        // Upload vector-add kernel
        load_vector_add_kernel();
        // Validate with correct hash (kernel_id=0)
        validate_kernel_expect_ack(7'd0);
        // Set up constants and data
        write_const_word(16'd0, 32'd0);
        write_const_word(16'd1, 32'd16);
        write_const_word(16'd2, 32'd32);
        write_words(16'h0000, 4, 32'd100);
        write_words(16'h0040, 4, 32'd7);
        // Now launch should succeed (kernel is verified)
        launch_kernel();
        poll_gpu_done();
        read_vector_add_expect(16'h0080, 4, 32'd100, 32'd7);

        // =====================================================================
        // Test 9: Reset check
        // =====================================================================
        test_num = 9;
        $display("TEST%0d: Reset check", test_num);
        btnC = 1'b1;
        repeat (4) @(posedge CLK100MHZ);
        btnC = 1'b0;
        wait_not_reset();
        if (dut.write_count !== 16'd0 || dut.read_count !== 16'd0) begin
            $display("TEST%0d FAIL: reset did not clear top counters write=%0d read=%0d",
                     test_num, dut.write_count, dut.read_count);
            errors = errors + 1;
        end

        #1000;
        if (errors == 0) begin
            $display("BASYS3 SECURITY TOP TESTS PASSED");
        end else begin
            $display("BASYS3 SECURITY TOP TESTS FAILED: %0d errors", errors);
        end
        $finish;
    end
endmodule
