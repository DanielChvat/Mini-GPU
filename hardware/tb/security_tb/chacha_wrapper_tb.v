`timescale 1ns/1ps

module chacha_wrapper_tb;

    reg          clk;
    reg          rst;
    reg          enable;
    reg  [255:0] key;
    reg          seed;
    wire [511:0] data_out;
    wire         data_valid;
    reg          data_ack;

    chacha_wrapper dut (
        .clk       (clk),
        .rst       (rst),
        .enable    (enable),
        .key       (key),
        .seed      (seed),
        .data_out  (data_out),
        .data_valid(data_valid),
        .data_ack  (data_ack)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    localparam [255:0] KEY_A = 256'h000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f;
    localparam [255:0] KEY_B = 256'hffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff;

    localparam NUM_BLOCKS = 4;

    reg [511:0] blocks [0:NUM_BLOCKS-1];
    reg [511:0] reseed_block;
    reg [511:0] deterministic_block;
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

    initial begin
        rst      = 1'b1;
        enable   = 1'b0;
        key      = KEY_A;
        seed     = 1'b0;
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
        seed = 1'b1;
        @(posedge clk);
        seed = 1'b0;

        repeat (50) @(posedge clk);
        @(negedge clk);

        if (data_valid !== 1'b0) begin
            $display("FAIL data_valid asserted with enable=0");
            $finish;
        end
        $display("PASS enable gate test");

        // ---- Test 3: Basic generation + handshake ----
        enable = 1'b1;
        @(posedge clk);
        @(negedge clk);

        if (data_valid !== 1'b1) begin
            $display("FAIL data_valid did not assert when enable went high");
            $finish;
        end

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

        // ---- Test 5: Re-seed with different key ----
        // Wrapper is in W_VALID after last ack. Seed directly (no ack needed).
        key  = KEY_B;
        seed = 1'b1;
        @(posedge clk);
        seed = 1'b0;

        wait_valid;
        reseed_block = data_out;

        test_fail = 1'b0;
        for (i = 0; i < NUM_BLOCKS; i = i + 1) begin
            if (reseed_block == blocks[i]) begin
                $display("FAIL reseed block matches block[%0d]", i);
                test_fail = 1'b1;
            end
        end
        if (test_fail) $finish;
        $display("PASS re-seed with different key");

        // ---- Test 6: Deterministic (same key -> same first block) ----
        // Wrapper is in W_VALID. Seed directly with KEY_A (no ack needed).
        key  = KEY_A;
        seed = 1'b1;
        @(posedge clk);
        seed = 1'b0;

        wait_valid;
        deterministic_block = data_out;

        if (deterministic_block !== blocks[0]) begin
            $display("FAIL deterministic: re-seed with KEY_A did not reproduce block[0]");
            $finish;
        end
        $display("PASS deterministic test");

        // ---- Test 7: Ack gated by enable ----
        // Wrapper is in W_VALID with deterministic_block. Disable enable, then ack.
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

        if (data_out !== deterministic_block) begin
            $display("FAIL data changed after ack with enable=0");
            $finish;
        end
        $display("PASS ack gated by enable");

        $display("chacha_wrapper_tb PASS");
        $finish;
    end

    initial begin
        #10_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule
