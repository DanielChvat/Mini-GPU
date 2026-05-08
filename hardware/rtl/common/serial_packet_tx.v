module serial_packet_tx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115200
)(
    input  wire        clk,
    input  wire        rst,

    // control
    input  wire        start,
    input  wire [7:0]  cmd,
    input  wire [15:0] addr,
    input  wire [15:0] len,

    // payload stream (32-bit words, little-endian like RX)
    input  wire [31:0] payload_data,
    input  wire        payload_valid,
    output reg         payload_advance,

    // UART output
    output reg         tx_line,
    output reg         busy
);

    // ============================================================
    // Baud Tick Generator
    // ============================================================
    localparam integer BAUD_DIV = CLK_FREQ / BAUD_RATE;

    reg [$clog2(BAUD_DIV)-1:0] baud_cnt;
    reg baud_tick;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            baud_cnt  <= 0;
            baud_tick <= 0;
        end else begin
            if (baud_cnt == BAUD_DIV-1) begin
                baud_cnt  <= 0;
                baud_tick <= 1;
            end else begin
                baud_cnt  <= baud_cnt + 1;
                baud_tick <= 0;
            end
        end
    end

    // ============================================================
    // FSM States
    // ============================================================
    localparam IDLE       = 4'd0,
               SEND_SOF   = 4'd1,
               SEND_CMD   = 4'd2,
               SEND_AH    = 4'd3,
               SEND_AL    = 4'd4,
               SEND_LH    = 4'd5,
               SEND_LL    = 4'd6,
               LOAD_PAY   = 4'd7,
               SEND_PAY   = 4'd8,
               SEND_CRC   = 4'd9,
               DONE       = 4'd10;

    reg [3:0] state;

    // ============================================================
    // TX shift logic (byte -> UART 8N1 frame)
    // ============================================================
    reg [9:0] tx_shift;
    reg [3:0] bit_cnt;
    reg       sending_byte;
    // reg       tx_busy;
    reg [7:0] tx_buffer;
    reg       buffer_valid;

    // ============================================================
    // Packet tracking
    // ============================================================
    reg [7:0]  crc;
    reg [15:0] payload_count;

    // payload word unpacking (little-endian like RX)
    reg [31:0] cur_word;
    reg [1:0]  byte_idx;

    reg [7:0] cur_byte;

    // ============================================================
    // Byte transmit helper
    // ============================================================
    task send_byte(input [7:0] b);
    begin
        if(!sending_byte) begin
            tx_shift    <= {1'b1, b, 1'b0};
            sending_byte<= 1;
            bit_cnt     <= 0;
        end else if (!buffer_valid) begin // this 
            tx_buffer <= b;
            buffer_valid <= 1;
        end // else: the FSM gate will prevent this case
            
    end
    endtask

    // ============================================================
    // Main FSM
    // ============================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state           <= IDLE;
            tx_line         <= 1'b1;
            busy            <= 0;

            crc             <= 0;
            payload_count   <= 0;

            sending_byte    <= 0;
            payload_advance <= 0;
            byte_idx        <= 0;
            buffer_valid    <= 0;
        end else begin
            // payload_advance <= 0;

            // ====================================================
            // Bit-level transmission
            // ====================================================
            if (baud_tick && (sending_byte || buffer_valid)) begin

                if(sending_byte) begin
                    tx_line   <= tx_shift[0];
                    
                    if (bit_cnt == 4'd9) begin
                        if(buffer_valid)begin
                            tx_shift <= {1'b1, tx_buffer, 1'b0};
                            buffer_valid <= 0;
                            bit_cnt <= 0;
                        end
                        else begin
                            sending_byte <= 0;
                            tx_line <= 1'b1;
                        end
                    end else begin
                        bit_cnt <= bit_cnt + 1;
                        tx_shift  <= {1'b1, tx_shift[9:1]};
                    end
                end 
                else begin // when buffer_valid is active, but sending_byte is not activated
                    tx_line   <= 1'b0;                      // start bit
                    tx_shift <= {1'b1, 1'b1, tx_buffer};    // stop bit followed by data bits
                    buffer_valid <= 0;
                    sending_byte <= 1;
                    bit_cnt <= 1;                           // already sent the first bit 
                end
            end

            // ====================================================
            // FSM (byte-level)
            // ====================================================
            if (!buffer_valid) begin
                case (state)

                IDLE: begin
                    // tx_line <= 1'b1;
                    busy    <= 0;

                    if (start) begin
                        crc           <= 0;
                        payload_count <= 0;
                        byte_idx      <= 0;
                        busy          <= 1;
                        state         <= SEND_SOF;
                    end
                end

                // ------------------------------------------------
                SEND_SOF: begin
                    send_byte(8'hAA);
                    state <= SEND_CMD;
                end

                SEND_CMD: begin
                    send_byte(cmd);
                    crc   <= crc ^ cmd;
                    state <= SEND_AH;
                end

                SEND_AH: begin
                    send_byte(addr[15:8]);
                    crc   <= crc ^ addr[15:8];
                    state <= SEND_AL;
                end

                SEND_AL: begin
                    send_byte(addr[7:0]);
                    crc   <= crc ^ addr[7:0];
                    state <= SEND_LH;
                end

                SEND_LH: begin
                    send_byte(len[15:8]);
                    crc   <= crc ^ len[15:8];
                    state <= SEND_LL;
                end

                SEND_LL: begin
                    send_byte(len[7:0]);
                    crc   <= crc ^ len[7:0];

                    if (len == 0)
                        state <= SEND_CRC;
                    else begin
                        state <= LOAD_PAY;
                        payload_advance <= 1;
                    end
                end

                // ------------------------------------------------
                LOAD_PAY: begin
                    if (payload_valid) begin
                        payload_advance <= 0;
                        cur_word <= payload_data;
                        byte_idx <= 0;
                        state    <= SEND_PAY; // step into byte emit
                    end
                end

                // byte emission phase (inline expansion)
                SEND_PAY: begin
                    case (byte_idx)
                        2'd0: cur_byte = cur_word[7:0];
                        2'd1: cur_byte = cur_word[15:8];
                        2'd2: cur_byte = cur_word[23:16];
                        2'd3: cur_byte = cur_word[31:24];
                    endcase

                    send_byte(cur_byte);
                    crc <= crc ^ cur_byte;
                    payload_count <= payload_count + 16'd1;

                    if (payload_count + 16'd1 >= len) begin
                        byte_idx <= 0;
                        state <= SEND_CRC;
                    end else if (byte_idx == 2'd3) begin
                        byte_idx <= 0;
                        state <= LOAD_PAY;
                        payload_advance <= 1;
                    end else begin
                        byte_idx <= byte_idx + 1;
                    end
                end

                // ------------------------------------------------
                SEND_CRC: begin
                    send_byte(crc);
                    state <= DONE;
                end

                DONE: begin
                    busy  <= 0;
                    state <= IDLE;
                end

                endcase
            end
        end
    end

endmodule
