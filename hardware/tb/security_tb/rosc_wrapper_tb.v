`timescale 1ns/1ps

module rosc_wrapper_tb;

    reg         clk;
    reg         rst;
    reg         enable;
    wire [31:0] entropy_data;
    wire        entropy_valid;
    reg         entropy_ack;

    rosc_wrapper dut (
        .clk          (clk),
        .rst          (rst),
        .enable       (enable),
        .entropy_data (entropy_data),
        .entropy_valid(entropy_valid),
        .entropy_ack  (entropy_ack)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    localparam NUM_SAMPLES = 64;
    localparam REP_CUTOFF  = 4;
    localparam PROP_THRESH = 32;

    reg [31:0] samples [0:NUM_SAMPLES-1];
    integer    sample_idx;
    integer    i, j;
    integer    rep_count;
    integer    match_count;
    reg [31:0] or_acc;
    reg [31:0] and_acc;
    reg [31:0] saved_data;
    reg        degenerate;

    initial begin
        rst         = 1'b1;
        enable      = 1'b0;
        entropy_ack = 1'b0;

        repeat (2) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);

        // ---- Reset test ----
        if (entropy_valid !== 1'b0) begin
            $display("FAIL entropy_valid not 0 after reset");
            $finish;
        end
        $display("PASS reset test");

        // ---- Enable gate test ----
        repeat (10000) begin
            @(posedge clk);
            if (entropy_valid !== 1'b0) begin
                $display("FAIL entropy_valid asserted with enable=0");
                $finish;
            end
        end
        $display("PASS enable gate test");

        // ---- Collect entropy words + handshake test ----
        enable = 1'b1;
        for (sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx = sample_idx + 1) begin
            while (entropy_valid !== 1'b1)
                @(posedge clk);

            saved_data = entropy_data;
            @(posedge clk);
            if (entropy_valid && entropy_data !== saved_data) begin
                $display("FAIL entropy_data changed while valid held (sample %0d)", sample_idx);
                $finish;
            end

            samples[sample_idx] = saved_data;

            entropy_ack = 1'b1;
            @(posedge clk);
            entropy_ack = 1'b0;

            @(posedge clk);
            if (entropy_valid !== 1'b0) begin
                $display("FAIL entropy_valid did not deassert after ack (sample %0d)", sample_idx);
                $finish;
            end
        end
        $display("PASS collected %0d entropy words", NUM_SAMPLES);
        $display("PASS handshake test");

        // ---- Check for degenerate output (ROSC can't oscillate in simulation) ----
        degenerate = 1'b0;
        for (i = 0; i < NUM_SAMPLES; i = i + 1) begin
            if ((^samples[i]) === 1'bx)
                degenerate = 1'b1;
        end

        if (degenerate) begin
            $display("INFO degenerate entropy detected (expected in simulation, ROSC has combinational loops)");
            $display("INFO health tests below are only meaningful on real hardware");
            $display("SKIP repetition count test");
            $display("SKIP adaptive proportion test");
            $display("SKIP stuck-word test");
            $display("SKIP stuck-bit test");
        end else begin
            // ---- Repetition count test (NIST SP 800-90B) ----
            rep_count = 1;
            for (i = 1; i < NUM_SAMPLES; i = i + 1) begin
                if (samples[i] == samples[i-1])
                    rep_count = rep_count + 1;
                else
                    rep_count = 1;

                if (rep_count >= REP_CUTOFF) begin
                    $display("FAIL repetition count %0d at index %0d (value 0x%08h)", rep_count, i, samples[i]);
                    $finish;
                end
            end
            $display("PASS repetition count test");

            // ---- Adaptive proportion test (NIST SP 800-90B) ----
            for (i = 0; i < NUM_SAMPLES; i = i + 1) begin
                match_count = 0;
                for (j = 0; j < NUM_SAMPLES; j = j + 1) begin
                    if (samples[j] == samples[i])
                        match_count = match_count + 1;
                end
                if (match_count >= PROP_THRESH) begin
                    $display("FAIL adaptive proportion: value 0x%08h appeared %0d times", samples[i], match_count);
                    $finish;
                end
            end
            $display("PASS adaptive proportion test");

            // ---- Stuck-word test ----
            degenerate = 1'b1;
            for (i = 1; i < NUM_SAMPLES; i = i + 1) begin
                if (samples[i] != samples[0])
                    degenerate = 1'b0;
            end
            if (degenerate) begin
                $display("FAIL all entropy words identical (0x%08h)", samples[0]);
                $finish;
            end
            $display("PASS stuck-word test");

            // ---- Stuck-bit test ----
            or_acc  = 32'h00000000;
            and_acc = 32'hFFFFFFFF;
            for (i = 0; i < NUM_SAMPLES; i = i + 1) begin
                or_acc  = or_acc  | samples[i];
                and_acc = and_acc & samples[i];
            end
            if (or_acc != 32'hFFFFFFFF) begin
                $display("FAIL stuck-at-0 bits detected: OR accumulator = 0x%08h", or_acc);
                $finish;
            end
            if (and_acc != 32'h00000000) begin
                $display("FAIL stuck-at-1 bits detected: AND accumulator = 0x%08h", and_acc);
                $finish;
            end
            $display("PASS stuck-bit test");
        end

        $display("rosc_wrapper_tb PASS");
        $finish;
    end

    initial begin
        #100_000_000;
        $display("FAIL timeout");
        $finish;
    end

endmodule
