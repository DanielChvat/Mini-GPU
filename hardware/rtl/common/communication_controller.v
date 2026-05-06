module communication_controller #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115200
)(
    input  wire clk,
    input  wire rst,

    // UART physical line
    input  wire rx_line,
    output wire tx_line,

    // ================= External RSP interface =================
    input  wire        rsp_valid,
    input  wire [31:0] rsp_data,
    output reg         rsp_ready,

    // ================= Debug / observability =================
    output wire [7:0]  dbg_rx_cmd,
    output wire [15:0] dbg_rx_addr,
    output wire [15:0] dbg_rx_len
);

    // ============================================================
    // CMD definitions
    // ============================================================
    localparam COM_CMD_ACK = 8'h08;
    localparam COM_CMD_NAK = 8'h09;
    localparam COM_CMD_RSP = 8'h0A;

    // ============================================================
    // RX wires
    // ============================================================
    wire [31:0] word_data;
    wire        word_valid;

    wire [7:0]  rx_cmd;
    wire [15:0] rx_addr;
    wire [15:0] rx_len;

    wire send_ack;
    wire send_nack;

    reg  write_done;

    assign dbg_rx_cmd  = rx_cmd;
    assign dbg_rx_addr = rx_addr;
    assign dbg_rx_len  = rx_len;

    // ============================================================
    // TX wires
    // ============================================================
    reg        tx_start;
    reg [7:0]  tx_cmd;
    reg [15:0] tx_addr;
    reg [15:0] tx_len;

    wire       tx_busy;

    reg [31:0] tx_payload_data;
    reg        tx_payload_valid;
    wire       tx_payload_advance;

    // ============================================================
    // Instantiate RX
    // ============================================================
    serial_packet_rx #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) u_rx (
        .clk(clk),
        .rst(rst),
        .rx_line(rx_line),

        .word_data(word_data),
        .word_valid(word_valid),

        .cmd(rx_cmd),
        .addr(rx_addr),
        .len(rx_len),

        .write_done(write_done),

        .send_ack(send_ack),
        .send_nack(send_nack)
    );

    // ============================================================
    // Instantiate TX
    // ============================================================
    serial_packet_tx #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) u_tx (
        .clk(clk),
        .rst(rst),

        .start(tx_start),
        .cmd(tx_cmd),
        .addr(tx_addr),
        .len(tx_len),

        .payload_data(tx_payload_data),
        .payload_valid(tx_payload_valid),
        .payload_advance(tx_payload_advance),

        .tx_line(tx_line),
        .busy(tx_busy)
    );

    // ============================================================
    // Edge detection
    // ============================================================
    reg send_ack_d, send_nack_d;

    wire ack_edge  = send_ack  & ~send_ack_d;
    wire nack_edge = send_nack & ~send_nack_d;

    always @(posedge clk) begin
        send_ack_d  <= send_ack;
        send_nack_d <= send_nack;
    end

    // ============================================================
    // FSM
    // ============================================================
    localparam IDLE            = 4'd0;
    localparam ISSUE_CMD       = 4'd1;
    localparam WAIT_TX         = 4'd2;

    localparam LOAD_RSP        = 4'd3;
    localparam SEND_RSP        = 4'd4;
    localparam SEND_RSP_PAYLOAD= 4'd5;

    reg [3:0] state;

    reg [7:0] pending_cmd;
    reg       req_valid;

    // RSP storage
    reg [31:0] rsp_buffer;

    // ============================================================
    // Main FSM
    // ============================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;

            tx_start <= 0;
            tx_cmd   <= 0;
            tx_addr  <= 0;
            tx_len   <= 0;

            tx_payload_data  <= 0;
            tx_payload_valid <= 0;

            req_valid <= 0;
            pending_cmd <= 0;

            rsp_ready <= 0;
            write_done <= 0;
        end else begin

            // defaults
            tx_start <= 0;
            tx_payload_valid <= 0;
            rsp_ready <= 0;
            write_done <= 0;

            // ====================================================
            // Capture ACK/NACK events (priority: NACK)
            // ====================================================
            if (!req_valid) begin
                if (nack_edge) begin
                    pending_cmd <= COM_CMD_NAK;
                    req_valid   <= 1;
                end
                else if (ack_edge) begin
                    pending_cmd <= COM_CMD_ACK;
                    req_valid   <= 1;
                end
            end

            case (state)

            // ----------------------------------------------------
            IDLE: begin

                // ==============================
                // PRIORITY 1: RSP request
                // ==============================
                if (rsp_valid && !tx_busy) begin
                    rsp_buffer <= rsp_data;
                    rsp_ready  <= 1;
                    state      <= LOAD_RSP;
                end

                // ==============================
                // PRIORITY 2: ACK/NACK
                // ==============================
                else if (req_valid && !tx_busy) begin
                    state <= ISSUE_CMD;
                end

                // ====================================================
                // TODO: Command handling (USER IMPLEMENTATION)
                // ====================================================
                // case (rx_cmd)
                //
                //   COM_CMD_WRITE_DATA:
                //       // write to BRAM
                //       // assert write_done when finished
                //
                //   COM_CMD_READ_DATA:
                //       // trigger rsp_valid + rsp_data
                //
                //   COM_CMD_LAUNCH:
                //       // trigger execution
                //
                // endcase

            end

            // ----------------------------------------------------
            ISSUE_CMD: begin
                tx_cmd  <= pending_cmd;
                tx_addr <= 0;
                tx_len  <= 0;

                tx_start <= 1;

                state <= WAIT_TX;
            end

            // ----------------------------------------------------
            LOAD_RSP: begin
                tx_cmd  <= COM_CMD_RSP;
                tx_addr <= 0;
                tx_len  <= 4;

                tx_start <= 1;

                state <= SEND_RSP;
            end

            // ----------------------------------------------------
            SEND_RSP: begin
                // wait for TX to request payload
                if (tx_payload_advance) begin
                    tx_payload_data  <= rsp_buffer;
                    tx_payload_valid <= 1;
                    state <= SEND_RSP_PAYLOAD;
                end
            end

            // ----------------------------------------------------
            SEND_RSP_PAYLOAD: begin
                // wait for TX to finish
                if (!tx_busy) begin
                    state <= IDLE;
                end
            end

            // ----------------------------------------------------
            WAIT_TX: begin
                if (!tx_busy) begin
                    req_valid <= 0;
                    state     <= IDLE;
                end
            end

            endcase
        end
    end

endmodule