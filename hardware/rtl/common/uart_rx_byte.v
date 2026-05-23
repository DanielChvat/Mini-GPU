`timescale 1ns/1ps
`default_nettype none

module uart_rx_byte #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115200
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       rx_line,
    output reg [7:0]  byte_data,
    output reg        byte_valid
);

    localparam integer BAUD_DIV = CLK_FREQ / BAUD_RATE;
    localparam integer HALF_BAUD_DIV = BAUD_DIV / 2;

    localparam UART_IDLE  = 2'd0;
    localparam UART_START = 2'd1;
    localparam UART_DATA  = 2'd2;
    localparam UART_STOP  = 2'd3;

    reg rx_sync1;
    reg rx_sync2;
    reg [1:0] uart_state;
    reg [$clog2(BAUD_DIV)-1:0] baud_cnt;
    reg [2:0] bit_cnt;
    reg [7:0] shift_reg;

    always @(posedge clk) begin
        rx_sync1 <= rx_line;
        rx_sync2 <= rx_sync1;
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            uart_state <= UART_IDLE;
            baud_cnt   <= 0;
            bit_cnt    <= 0;
            shift_reg  <= 0;
            byte_data  <= 0;
            byte_valid <= 0;
        end else begin
            byte_valid <= 1'b0;

            case (uart_state)
                UART_IDLE: begin
                    bit_cnt <= 0;
                    baud_cnt <= 0;
                    if (!rx_sync2)
                        uart_state <= UART_START;
                end

                UART_START: begin
                    if (baud_cnt == HALF_BAUD_DIV - 1) begin
                        baud_cnt <= 0;
                        if (!rx_sync2)
                            uart_state <= UART_DATA;
                        else
                            uart_state <= UART_IDLE;
                    end else begin
                        baud_cnt <= baud_cnt + 1;
                    end
                end

                UART_DATA: begin
                    if (baud_cnt == BAUD_DIV - 1) begin
                        baud_cnt <= 0;
                        shift_reg <= {rx_sync2, shift_reg[7:1]};
                        if (bit_cnt == 3'd7) begin
                            byte_data <= {rx_sync2, shift_reg[7:1]};
                            bit_cnt <= 0;
                            uart_state <= UART_STOP;
                        end else begin
                            bit_cnt <= bit_cnt + 1;
                        end
                    end else begin
                        baud_cnt <= baud_cnt + 1;
                    end
                end

                UART_STOP: begin
                    if (baud_cnt == BAUD_DIV - 1) begin
                        baud_cnt <= 0;
                        byte_valid <= rx_sync2;
                        uart_state <= UART_IDLE;
                    end else begin
                        baud_cnt <= baud_cnt + 1;
                    end
                end

                default: uart_state <= UART_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
