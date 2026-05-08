module serial_packet_rx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115200
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        rx_line,

    // 32-bit output to memory system
    output reg [31:0]  word_data,
    output reg         word_valid,
    // input  wire        word_taken,

    // packet-level outputs
    output reg [7:0]   cmd,
    output reg [15:0]  addr,
    output reg [15:0]  len,

    // control
    input  wire        write_done,

    // response signals
    output reg         packet_done,
    output reg         send_ack,
    output reg         send_nack
);

    // ============================================================
    // State machine declarations
    // ============================================================
    parameter IDLE = 4'd0, GET_CMD = 4'd1, GET_ADDR_HI = 4'd2, GET_ADDR_LO = 4'd3, GET_LEN_HI = 4'd4, GET_LEN_LO = 4'd5, GET_PAYLOAD = 4'd6, GET_CRC = 4'd7, WAIT_WRITE = 4'd8, SEND_ACK = 4'd9, SEND_NACK = 4'd10;
    reg [3:0] state;
    

    // ============================================================
    // UART RX
    // ============================================================
    localparam integer BAUD_DIV = CLK_FREQ / BAUD_RATE;
    localparam integer HALF_BAUD_DIV = BAUD_DIV / 2;

    // ============================================================
    // Input Synchronizer
    // ============================================================
    reg rx_sync1, rx_sync2;

    always @(posedge clk) begin
        rx_sync1 <= rx_line;
        rx_sync2 <= rx_sync1;
    end

    localparam UART_IDLE  = 2'd0;
    localparam UART_START = 2'd1;
    localparam UART_DATA  = 2'd2;
    localparam UART_STOP  = 2'd3;

    reg [1:0] uart_state;
    reg [$clog2(BAUD_DIV)-1:0] baud_cnt;
    reg [2:0] bit_cnt;
    reg [7:0] shift_reg;
    reg       byte_valid;
    reg [7:0] byte_data;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            uart_state <= UART_IDLE;
            baud_cnt   <= 0;
            bit_cnt    <= 0;
            shift_reg  <= 0;
            byte_valid <= 0;
            byte_data  <= 0;
        end else begin
            byte_valid <= 0;

            case (uart_state)
                UART_IDLE: begin
                    bit_cnt <= 0;
                    baud_cnt <= 0;
                    if (!rx_sync2) begin
                        uart_state <= UART_START;
                    end
                end

                UART_START: begin
                    if (baud_cnt == HALF_BAUD_DIV-1) begin
                        baud_cnt <= 0;
                        if (!rx_sync2) begin
                            uart_state <= UART_DATA;
                        end else begin
                            uart_state <= UART_IDLE;
                        end
                    end else begin
                        baud_cnt <= baud_cnt + 1;
                    end
                end

                UART_DATA: begin
                    if (baud_cnt == BAUD_DIV-1) begin
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
                    if (baud_cnt == BAUD_DIV-1) begin
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

    // ============================================================
    // Packet FSM
    // ============================================================
    

    reg [15:0] payload_count;
    reg [7:0]  crc;

    // 32-bit packing
    reg [1:0]  byte_index;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state         <= IDLE;
            cmd           <= 0;
            addr          <= 0;
            len           <= 0;
            payload_count <= 0;
            crc           <= 0;

            word_data     <= 0;
            word_valid    <= 0;
            byte_index    <= 0;

            packet_done   <= 0;
            send_ack      <= 0;
            send_nack     <= 0;
        end else begin
            // default strobes
            word_valid <= 0;
            packet_done <= 0;
            send_ack   <= 0;
            send_nack  <= 0;

            case(state)
                IDLE: begin
                    if (byte_valid && (byte_data == 8'hAA)) begin
                        state <= GET_CMD;
                        crc   <= 0;
                    end
                end

                // ------------------------------------------------
                WAIT_WRITE: begin
                    if (write_done)
                        state <= SEND_ACK;
                end

                // ------------------------------------------------
                SEND_ACK: begin
                    send_ack <= 1;
                    state    <= IDLE;
                end

                SEND_NACK: begin
                    send_nack <= 1;
                    state     <= IDLE;
                end

                // ------------------------------------------------
                default: begin
                    if (byte_valid) begin
                        case (state)
                        GET_CMD: begin
                            cmd   <= byte_data;
                            crc   <= byte_data;
                            state <= GET_ADDR_HI;
                        end

                        GET_ADDR_HI: begin
                            addr[15:8] <= byte_data;
                            crc        <= crc ^ byte_data;
                            state      <= GET_ADDR_LO;
                        end

                        GET_ADDR_LO: begin
                            addr[7:0] <= byte_data;
                            crc       <= crc ^ byte_data;
                            state     <= GET_LEN_HI;
                        end

                        GET_LEN_HI: begin
                            len[15:8] <= byte_data;
                            crc       <= crc ^ byte_data;
                            state     <= GET_LEN_LO;
                        end

                        GET_LEN_LO: begin
                            len[7:0]   <= byte_data;
                            crc        <= crc ^ byte_data;
                            payload_count <= 0;
                            byte_index <= 0;
                            word_data <= 0;

                            if ({len[15:8], byte_data} == 0)
                                state <= GET_CRC;
                            else
                                state <= GET_PAYLOAD;
                        end

                        // ------------------------------------------------
                        GET_PAYLOAD: begin
                            crc <= crc ^ byte_data;

                            // pack into 32-bit word
                            // word_data  <= {word_data[23:0], byte_data}; // small endian
                            word_data <= {byte_data, word_data[31:8]}; // big endian

                            if (byte_index == 2'd3) begin
                                word_valid <= 1;
                                byte_index <= 0;
                            end else
                                byte_index <= byte_index + 1;

                            payload_count <= payload_count + 1;

                            if (payload_count + 1 >= len) begin
                                state <= GET_CRC;
                                word_valid <= 1; // flush word even when payload isn't multiple of 4
                            end
                        end

                        // ------------------------------------------------
                        GET_CRC: begin
                            if (crc == byte_data) begin
                                state <= WAIT_WRITE;
                                packet_done <= 1;
                            end else
                                state <= SEND_NACK;
                        end

                        // ------------------------------------------------
                        default: state <= IDLE;

                        endcase
                    end
                end
            endcase
        end
    end

endmodule
