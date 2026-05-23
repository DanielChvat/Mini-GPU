`timescale 1ns/1ps
`default_nettype none

module uart_tx_byte #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115200
) (
    input  wire      clk,
    input  wire      rst,
    input  wire [7:0] byte_data,
    input  wire      byte_valid,
    output reg       byte_ready,
    output reg       tx_line,
    output wire      busy
);

    localparam integer BAUD_DIV = CLK_FREQ / BAUD_RATE;

    reg [$clog2(BAUD_DIV)-1:0] baud_cnt;
    reg baud_tick;
    reg [9:0] tx_shift;
    reg [3:0] bit_cnt;
    reg sending;

    assign busy = sending;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            baud_cnt <= 0;
            baud_tick <= 1'b0;
        end else begin
            if (baud_cnt == BAUD_DIV - 1) begin
                baud_cnt <= 0;
                baud_tick <= 1'b1;
            end else begin
                baud_cnt <= baud_cnt + 1;
                baud_tick <= 1'b0;
            end
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            tx_line <= 1'b1;
            tx_shift <= 10'h3ff;
            bit_cnt <= 4'd0;
            sending <= 1'b0;
            byte_ready <= 1'b1;
        end else begin
            byte_ready <= !sending;

            if (!sending && byte_valid) begin
                tx_shift <= {1'b1, byte_data, 1'b0};
                bit_cnt <= 4'd0;
                sending <= 1'b1;
                byte_ready <= 1'b0;
            end else if (sending && baud_tick) begin
                tx_line <= tx_shift[0];
                if (bit_cnt == 4'd9) begin
                    sending <= 1'b0;
                    tx_line <= 1'b1;
                    byte_ready <= 1'b1;
                end else begin
                    bit_cnt <= bit_cnt + 4'd1;
                    tx_shift <= {1'b1, tx_shift[9:1]};
                end
            end
        end
    end

endmodule

`default_nettype wire
