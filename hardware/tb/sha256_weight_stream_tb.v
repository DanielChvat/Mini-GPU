`timescale 1ns/1ps

module sha256_weight_stream_tb;
    reg clk;
    reg rst;
    reg start;
    reg word_valid;
    reg [31:0] word_data;
    reg finish;
    wire busy;
    wire ready;
    wire [255:0] digest;
    wire digest_valid;
    wire error;

    sha256_weight_stream dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .word_valid(word_valid),
        .word_data(word_data),
        .finish(finish),
        .busy(busy),
        .ready(ready),
        .digest(digest),
        .digest_valid(digest_valid),
        .error(error)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task begin_hash;
        begin
            @(negedge clk);
            start = 1'b1;
            word_valid = 1'b0;
            finish = 1'b0;
            word_data = 32'b0;
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    task send_word;
        input [31:0] data;
        begin
            while (!ready) begin
                @(negedge clk);
            end

            word_data = data;
            word_valid = 1'b1;
            finish = 1'b0;
            @(negedge clk);
            word_valid = 1'b0;
            word_data = 32'b0;
        end
    endtask

    task end_hash_expect;
        input [255:0] expected;
        begin
            while (!ready) begin
                @(negedge clk);
            end

            finish = 1'b1;
            @(negedge clk);
            finish = 1'b0;

            while (!digest_valid) begin
                @(negedge clk);
            end

            #1;
            if (digest !== expected) begin
                $display("FAIL digest got=%064h expected=%064h", digest, expected);
                $finish;
            end
            if (error) begin
                $display("FAIL unexpected error flag");
                $finish;
            end
        end
    endtask

    task run_range_test;
        input integer count;
        input [255:0] expected;
        integer i;
        begin
            begin_hash();
            for (i = 0; i < count; i = i + 1) begin
                send_word(i[31:0]);
            end
            end_hash_expect(expected);
        end
    endtask

    task run_one_word_test;
        begin
            begin_hash();
            send_word(32'h11223344);
            end_hash_expect(256'h1a835ed8734f86355ca5b835d824d486993aabf1913cd3a011b7446c0514b7c9);
        end
    endtask

    task run_backpressure_test;
        integer i;
        begin
            begin_hash();
            for (i = 0; i < 16; i = i + 1) begin
                send_word(i[31:0]);
            end

            #1;
            if (ready) begin
                $display("FAIL ready stayed high while full block was processing");
                $finish;
            end

            while (!ready) begin
                @(negedge clk);
            end
            end_hash_expect(256'h59d67963f3f53fd016156b50d83b8c83d4068f6e2f08bc9c87f0b49c20cf31f0);
        end
    endtask

    initial begin
        rst = 1'b1;
        start = 1'b0;
        word_valid = 1'b0;
        word_data = 32'b0;
        finish = 1'b0;

        // repeat (2) @(posedge clk);
        #10;
        rst = 1'b0;

        // Expected values are SHA-256 over 32-bit words serialized big-endian.
        begin_hash();
        end_hash_expect(256'he3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855);

        run_one_word_test();
        run_range_test(15, 256'hf00b4276a50df7a0bb077502c5f54d21faf7fbd9ada826220151d7555a4b8380);
        run_backpressure_test();
        run_range_test(20, 256'hbaabd3d13233879a8b39e016f96247df17579c3a3900ceed9aae6831faf9da87);

        $display("sha256_weight_stream_tb PASS");
        $finish;
    end
endmodule
