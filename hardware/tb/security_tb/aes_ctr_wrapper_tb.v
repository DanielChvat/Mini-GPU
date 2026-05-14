`timescale 1ns/1ps

module aes_ctr_wrapper_tb;

    reg          clk;
    reg          rst;
    reg          enable;
    reg  [127:0] key;
    reg  [95:0]  nonce;
    reg          seed;
    reg  [127:0] data_in;
    reg          load;
    wire [127:0] data_out;
    wire         data_valid;
    reg          data_ack;

    aes_ctr_wrapper dut (
        .clk       (clk),
        .rst       (rst),
        .enable    (enable),
        .key       (key),
        .nonce     (nonce),
        .seed      (seed),
        .data_in   (data_in),
        .load      (load),
        .data_out  (data_out),
        .data_valid(data_valid),
        .data_ack  (data_ack)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    localparam [127:0] KEY_A  = 128'h000102030405060708090a0b0c0d0e0f;
    localparam [127:0] KEY_B  = 128'hffffffffffffffffffffffffffffffff;
    localparam [95:0]  NONCE  = 96'h000000000000000000000001;
    localparam [127:0] PT_A   = 128'hdeadbeefcafebabe0123456789abcdef;

    localparam NUM_BLOCKS = 4;

    reg [127:0] blocks [0:NUM_BLOCKS-1];
    reg [127:0] first_ct;
    reg [127:0] rekey_ct;
    reg [127:0] determ_ct;
    reg [127:0] roundtrip_out;
    integer     i, j;
    reg         test_fail;

    task wait_valid;
        begin
            @(negedge clk);
            while (data_valid !== 1'b1)
                @(negedge clk);
        end
    endtask

    task ack_block;
        begin
            data_ack = 1'b1;
            @(posedge clk);
            data_ack = 1'b0;
        end
    endtask

    task seed_key;
        begin
            @(negedge clk);
            seed = 1'b1;
            @(negedge clk);
            seed = 1'b0;
            @(negedge clk);
            while (dut.state_reg != dut.W_IDLE)
                @(negedge clk);
        end
    endtask

    task load_block(input [127:0] pt);
        begin
            data_in = pt;
            load = 1'b1;
            @(posedge clk);
            load = 1'b0;
        end
    endtask

    initial begin
        rst      = 1'b1;
        enable   = 1'b0;
        key      = KEY_A;
        nonce    = NONCE;
        seed     = 1'b0;
        data_in  = 128'b0;
        load     = 1'b0;
        data_ack = 1'b0;

        repeat (2) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);

        // ---- Test 1: Reset ----
        if (data_valid !== 1'b0) begin
            $display("FAIL data_valid not 0 after reset");
            $finish;
        end
        $display("PASS reset test");

        // ---- Test 2: Enable gate ----
        seed_key;
        load_block(PT_A);

        repeat (100) @(posedge clk);
        @(negedge clk);

        if (data_valid !== 1'b0) begin
            $display("FAIL data_valid asserted with enable=0");
            $finish;
        end
        $display("PASS enable gate test");

        // ---- Test 3: Basic handshake ----
        enable = 1'b1;
        @(posedge clk);
        @(negedge clk);

        if (data_valid !== 1'b1) begin
            $display("FAIL data_valid did not assert when enable went high");
            $finish;
        end

        first_ct  = data_out;
        blocks[0] = data_out;

        ack_block;

        @(negedge clk);
        if (data_valid !== 1'b0) begin
            $display("FAIL data_valid did not deassert after ack");
            $finish;
        end
        $display("PASS handshake test");

        // ---- Test 4: Multiple blocks differ ----
        for (i = 1; i < NUM_BLOCKS; i = i + 1) begin
            load_block(PT_A);
            wait_valid;
            blocks[i] = data_out;
            ack_block;
        end

        test_fail = 1'b0;
        for (i = 0; i < NUM_BLOCKS; i = i + 1) begin
            for (j = i + 1; j < NUM_BLOCKS; j = j + 1) begin
                if (blocks[i] == blocks[j]) begin
                    $display("FAIL block[%0d] == block[%0d]", i, j);
                    test_fail = 1'b1;
                end
            end
        end
        if (test_fail) $finish;
        $display("PASS multiple blocks differ");

        // ---- Test 5: Re-key ----
        key = KEY_B;
        seed_key;
        load_block(PT_A);
        wait_valid;
        rekey_ct = data_out;

        test_fail = 1'b0;
        for (i = 0; i < NUM_BLOCKS; i = i + 1) begin
            if (rekey_ct == blocks[i]) begin
                $display("FAIL rekey block matches block[%0d]", i);
                test_fail = 1'b1;
            end
        end
        if (test_fail) $finish;
        $display("PASS re-key test");
        ack_block;

        // ---- Test 6: Determinism ----
        key = KEY_A;
        seed_key;
        load_block(PT_A);
        wait_valid;
        determ_ct = data_out;

        if (determ_ct !== first_ct) begin
            $display("FAIL deterministic: re-seed with KEY_A did not reproduce first ciphertext");
            $finish;
        end
        $display("PASS determinism test");
        ack_block;

        // ---- Test 7: CTR round-trip ----
        // Re-seed to reset counter, then "decrypt" the first ciphertext
        key = KEY_A;
        seed_key;
        load_block(first_ct);
        wait_valid;
        roundtrip_out = data_out;

        if (roundtrip_out !== PT_A) begin
            $display("FAIL CTR round-trip: expected %h, got %h", PT_A, roundtrip_out);
            $finish;
        end
        $display("PASS CTR round-trip test");
        ack_block;

        // ---- Test 8: Ack gated by enable ----
        load_block(PT_A);
        wait_valid;

        enable = 1'b0;
        @(posedge clk);

        data_ack = 1'b1;
        @(posedge clk);
        data_ack = 1'b0;
        @(posedge clk);

        enable = 1'b1;
        @(negedge clk);

        if (data_valid !== 1'b1) begin
            $display("FAIL ack with enable=0 should not have consumed block");
            $finish;
        end
        $display("PASS ack gated by enable");
        ack_block;

        // ---- Test 9: Load-before-seed rejection ----
        rst = 1'b1;
        repeat (2) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);

        load_block(PT_A);
        repeat (50) @(posedge clk);
        @(negedge clk);

        if (data_valid !== 1'b0) begin
            $display("FAIL load without seed should not produce output");
            $finish;
        end
        $display("PASS load-before-seed rejection");

        $display("aes_ctr_wrapper_tb PASS");
        $finish;
    end

    initial begin
        #50_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule
