`timescale 1ns/1ps

module basys3_mini_gpu_top_tb;
    parameter SHARED_FLOAT_UNITS = 2;

    reg CLK100MHZ = 1'b0;
    reg btnC = 1'b1;
    reg [15:0] sw = 16'b0;
    wire [15:0] led;
    wire [6:0] seg;
    wire dp;
    wire [3:0] an;

    integer cycles;

    basys3_mini_gpu_top #(
        .SHARED_FLOAT_UNITS(SHARED_FLOAT_UNITS)
    ) dut (
        .CLK100MHZ(CLK100MHZ),
        .btnC(btnC),
        .sw(sw),
        .led(led),
        .seg(seg),
        .dp(dp),
        .an(an)
    );

    always #5 CLK100MHZ = ~CLK100MHZ;

    initial begin
        repeat (4) @(posedge CLK100MHZ);
        btnC = 1'b0;

        cycles = 0;
        while (!led[2] && !led[3] && cycles < 200000) begin
            @(posedge CLK100MHZ);
            cycles = cycles + 1;
        end

        if (!led[2] || led[3]) begin
            $display("FAIL basys3_mini_gpu_top pass=%0d fail=%0d led=0x%04h cycles=%0d run_cycles=%0d fpu_units=%0d",
                     led[2], led[3], led, cycles, dut.run_cycles, SHARED_FLOAT_UNITS);
            $finish;
        end

        $display("basys3_mini_gpu_top_tb PASS led=0x%04h cycles=%0d run_cycles=%0d fpu_units=%0d",
                 led, cycles, dut.run_cycles, SHARED_FLOAT_UNITS);
        $finish;
    end
endmodule
