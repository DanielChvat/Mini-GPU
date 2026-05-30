module communication_controller #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115200,
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32,
    parameter MEMORY_BANK_DEPTH = 8192,
    parameter KERNEL_ID_WIDTH = 7
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

    // ================= Memory write status interface =================
    input  wire        mem_write_done,
    input  wire        mem_write_fail,
    output reg         memory_status_consumed,

    // ================= External data memory interface =================
    output reg  [3:0] data_mem_req_valid,
    output reg  [3:0] data_mem_req_write,
    output reg  [(4*ADDR_WIDTH)-1:0] data_mem_req_addr,
    output reg  [(4*DATA_WIDTH)-1:0] data_mem_req_wdata,
    input  wire [3:0] data_mem_req_ready,
    input  wire [3:0] data_mem_resp_valid,
    input  wire [(4*DATA_WIDTH)-1:0] data_mem_resp_rdata,

    // ================= Program / launch interface =================
    output reg                         prog_we,
    output reg  [ADDR_WIDTH-1:0]       prog_addr,
    output reg  [31:0]                 prog_wdata,
    output reg                         const_we,
    output reg  [ADDR_WIDTH-1:0]       const_addr,
    output reg  [31:0]                 const_wdata,
    output reg                         launch_valid,
    output reg  [ADDR_WIDTH-1:0]       launch_base_pc,
    output reg  [31:0]                 launch_grid_dim,
    output reg  [31:0]                 launch_block_dim,
    output reg  [31:0]                 launch_active_mask,
    input  wire [31:0]                 status_word,
    input wire                         launch_ready,

    // ================= Validate interface =================
    output reg [KERNEL_ID_WIDTH-1:0]   validate_kernel_id,
    output reg         validate_triggered,

    // ================= Kernel verification status =================
    input  wire        prog_write_blocked,
    input  wire        validate_done,
    input  wire        validate_fail,

    // ================= Security reset interface =================
    output reg         security_reset_triggered,
    output reg         security_error,

    // ================= Debug / observability =================
    output wire [7:0]  dbg_rx_cmd,
    output wire [15:0] dbg_rx_addr,
    output wire [15:0] dbg_rx_len
);

    // ============================================================
    // CMD definitions
    // ============================================================
    
    localparam COM_CMD_WRITE_DATA    = 8'h01; // write data to data BRAM
    localparam COM_CMD_READ_DATA     = 8'h02; // read data from data BRAM

    localparam COM_CMD_WRITE_PROGRAM = 8'h03; // write program to instr BRAM
    localparam COM_CMD_LAUNCH        = 8'h04; // start execution
    localparam COM_CMD_READ_STATUS   = 8'h05; // read status register
    localparam COM_CMD_WRITE_CONSTANTS = 8'h0B; // write constant memory
    localparam COM_CMD_SECURITY_RESET = 8'h06; // reset security gate
    localparam COM_CMD_VALIDATE      = 8'h07; // trigger hash validation
    localparam COM_CMD_ACK           = 8'h08; // successful operation
    localparam COM_CMD_NAK           = 8'h09; // failed operation / bad packet
    localparam COM_CMD_RSP           = 8'h0A; // response packet

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
    wire packet_done;

    reg  write_done;
    reg  write_fail;

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
        .write_fail(write_fail),

        .packet_done(packet_done),
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
    localparam IDLE            = 4'd0; // wait for RSP data or pending ACK/NAK
    localparam ISSUE_CMD       = 4'd1; // start an ACK/NAK packet
    localparam WAIT_TX         = 4'd2; // wait for ACK/NAK transmit to finish

    localparam LOAD_RSP        = 4'd3; // prepare an external RSP packet
    localparam SEND_RSP        = 4'd4; // wait for TX to request RSP payload
    localparam SEND_RSP_PAYLOAD= 4'd5; // hold until RSP transmit completes
    localparam READ_START      = 4'd6; // start a memory read response packet
    localparam READ_WAIT       = 4'd7; // wait for memory read data
    localparam READ_SEND       = 4'd8; // present one read word to TX
    localparam READ_PAYLOAD    = 4'd9; // request next read word or finish packet
    localparam STATUS_START    = 4'd10; // start status response packet
    localparam STATUS_SEND     = 4'd11; // present status word to TX
    localparam STATUS_WAIT     = 4'd12; // wait for status response to finish
    localparam WAIT_WRITE_RESPONSE = 4'd13; // wait for write_done or write_fail from memory
    localparam WAIT_VALIDATE_KERNEL_RESPONSE = 4'd14; // wait for kernel_vfy validate_done or validate_fail

    reg [3:0] state;

    reg       pending_is_nack;
    reg [15:0] pending_addr;
    reg       req_valid;

    // RSP storage
    reg [31:0] rsp_buffer;

    // ============================================================
    // Data memory
    // ============================================================
    // The memory module is instantiated outside this controller. In the top
    // module, connect these ports directly to the external memory instance:
    //   communication_controller.data_mem_req_*  -> memory.req_*
    //   memory.req_ready                         -> communication_controller.data_mem_req_ready
    //   memory.resp_*                            -> communication_controller.data_mem_resp_*

    reg [15:0] write_words_seen;
    reg [15:0] read_bytes_left;
    reg [15:0] read_addr;
    reg [15:0] read_resp_addr;
    reg [15:0] read_resp_len;
    reg        write_done_hold;
    reg        write_fail_hold;
    reg        read_payload_requested;
    reg        read_sending_word;

    wire [ADDR_WIDTH-1:0] rx_word_addr =
        (rx_addr >> 2) + write_words_seen[ADDR_WIDTH-1:0];

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
            pending_is_nack <= 0;
            pending_addr <= 0;

            rsp_ready <= 0;
            write_done <= 0;
            write_fail <= 0;
            write_done_hold <= 0;
            write_fail_hold <= 0;
            memory_status_consumed <= 0;
            validate_triggered <= 0;
            security_reset_triggered <= 0;

            data_mem_req_valid <= 4'b0000;
            data_mem_req_write <= 4'b0000;
            data_mem_req_addr <= {(4*ADDR_WIDTH){1'b0}};
            data_mem_req_wdata <= {(4*DATA_WIDTH){1'b0}};
            prog_we <= 1'b0;
            prog_addr <= {ADDR_WIDTH{1'b0}};
            prog_wdata <= 32'b0;
            const_we <= 1'b0;
            const_addr <= {ADDR_WIDTH{1'b0}};
            const_wdata <= 32'b0;
            launch_valid <= 1'b0;
            launch_base_pc <= {ADDR_WIDTH{1'b0}};
            launch_grid_dim <= 32'b0;
            launch_block_dim <= 32'd4;
            launch_active_mask <= 32'h0000000f;
            write_words_seen <= 0;
            read_bytes_left <= 0;
            read_addr <= 0;
            read_resp_addr <= 0;
            read_resp_len <= 0;
            read_payload_requested <= 0;
            read_sending_word <= 0;
            validate_kernel_id <= {KERNEL_ID_WIDTH{1'b0}};
            security_error <= 0;
        end else begin

            // defaults
            tx_start <= 0;
            tx_payload_valid <= 0;
            rsp_ready <= 0;
            write_done <= write_done_hold;
            write_fail <= write_fail_hold;
            memory_status_consumed <= 0;
            data_mem_req_valid <= 4'b0000;
            data_mem_req_write <= 4'b0000;
            data_mem_req_addr <= {(4*ADDR_WIDTH){1'b0}};
            data_mem_req_wdata <= {(4*DATA_WIDTH){1'b0}};
            prog_we <= 1'b0;
            const_we <= 1'b0;
            launch_valid <= 1'b0;
            validate_triggered <= 1'b0;
            security_reset_triggered <= 1'b0;
            security_error <= 1'b0;

            // Remember when TX asks for read payload before memory data is ready.
            if ((state == READ_START) || (state == READ_WAIT) ||
                (state == READ_SEND) || (state == READ_PAYLOAD) ||
                (state == STATUS_START) || (state == STATUS_SEND) ||
                (state == STATUS_WAIT)) begin
                if (tx_payload_advance) begin
                    read_payload_requested <= 1'b1;
                end
            end else begin
                read_payload_requested <= 1'b0;
            end

            // Clear the held write/read-done flag after RX emits ACK/NAK status.
            if (nack_edge || ack_edge) begin
                write_done_hold <= 0;
                write_fail_hold <= 0;
            end

            // Restart write-word addressing whenever the current command is not a data write.
            if ((rx_cmd != COM_CMD_WRITE_DATA) &&
                (rx_cmd != COM_CMD_WRITE_PROGRAM) &&
                (rx_cmd != COM_CMD_WRITE_CONSTANTS) &&
                (rx_cmd != COM_CMD_LAUNCH)) begin
                write_words_seen <= 0;
            end

            // Stream each received WRITE_DATA word into data memory bank 0.
            if (word_valid && (rx_cmd == COM_CMD_WRITE_DATA)) begin
                data_mem_req_valid[0] <= 1'b1;
                data_mem_req_write[0] <= 1'b1;
                data_mem_req_addr[0 +: ADDR_WIDTH] <= rx_word_addr;
                data_mem_req_wdata[0 +: DATA_WIDTH] <= word_data;
                write_words_seen <= write_words_seen + 16'd1;
            end

            // Stream each received WRITE_PROGRAM word into instruction memory.
            if (word_valid && (rx_cmd == COM_CMD_WRITE_PROGRAM)) begin
                prog_we <= 1'b1;
                prog_addr <= rx_word_addr;
                prog_wdata <= word_data;
                write_words_seen <= write_words_seen + 16'd1;
            end

            // Stream each received WRITE_CONSTANTS word into constant memory.
            if (word_valid && (rx_cmd == COM_CMD_WRITE_CONSTANTS)) begin
                const_we <= 1'b1;
                const_addr <= rx_word_addr;
                const_wdata <= word_data;
                write_words_seen <= write_words_seen + 16'd1;
            end

            // Capture the three launch payload words: grid_dim, block_dim, active_mask.
            if (word_valid && (rx_cmd == COM_CMD_LAUNCH)) begin
                case (write_words_seen[1:0])
                    2'd0: launch_grid_dim <= word_data;
                    2'd1: launch_block_dim <= word_data;
                    default: launch_active_mask <= word_data;
                endcase
                write_words_seen <= write_words_seen + 16'd1;
            end

            // Mark write completion, or initialize a READ_DATA response sequence.
            if (packet_done && (rx_cmd == COM_CMD_WRITE_DATA)) begin
                write_words_seen <= 16'd0;
                state <= WAIT_WRITE_RESPONSE;
            end else if (packet_done && (rx_cmd == COM_CMD_WRITE_PROGRAM)) begin
                write_words_seen <= 16'd0;
                if (prog_write_blocked) begin
                    write_fail_hold <= 1'b1;
                end else begin
                    write_done_hold <= 1'b1;
                end
            end else if (packet_done && (rx_cmd == COM_CMD_WRITE_CONSTANTS)) begin
                write_done_hold <= 1'b1;
                write_words_seen <= 16'd0;
            end else if (packet_done && (rx_cmd == COM_CMD_LAUNCH)) begin
                if(!launch_ready) begin
                    write_fail_hold <= 1'b1;
                end else begin
                    launch_valid <= 1'b1;
                    launch_base_pc <= rx_addr[ADDR_WIDTH-1:0];
                    write_done_hold <= 1'b1;
                    write_words_seen <= 16'd0;
                end
            end else if (packet_done && (rx_cmd == COM_CMD_VALIDATE)) begin
                validate_kernel_id <= rx_addr[KERNEL_ID_WIDTH-1:0];
                validate_triggered <= 1'b1;
                state <= WAIT_VALIDATE_KERNEL_RESPONSE;
            end else if (packet_done && (rx_cmd == COM_CMD_SECURITY_RESET)) begin
                security_reset_triggered <= 1'b1;
                write_done_hold <= 1'b1;
            end else if (packet_done && (rx_cmd == COM_CMD_READ_DATA)) begin
                write_done_hold <= 1'b1;
                read_resp_addr <= rx_addr;
                read_resp_len <= rx_len;
                read_addr <= rx_addr >> 2;
                read_bytes_left <= rx_len;
                read_payload_requested <= 1'b0;
                state <= READ_START;
            end else if (packet_done && (rx_cmd == COM_CMD_READ_STATUS)) begin
                write_done_hold <= 1'b1;
                read_resp_addr <= rx_addr;
                read_resp_len <= 16'd4;
                rsp_buffer <= status_word;
                read_payload_requested <= 1'b0;
                state <= STATUS_START;
            end

            // ====================================================
            // Capture ACK/NACK events 
            // ====================================================
            if (!req_valid) begin
                if (nack_edge) begin
                    pending_is_nack <= 1'b1;
                    pending_addr <= rx_addr;
                    req_valid   <= 1;

                    security_error <= 1'b1;
                end
                else if (ack_edge && (rx_cmd != COM_CMD_READ_DATA) &&
                         (rx_cmd != COM_CMD_READ_STATUS)) begin
                    pending_is_nack <= 1'b0;
                    pending_addr <= rx_addr;
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

            end

            // ----------------------------------------------------
            ISSUE_CMD: begin
                tx_cmd  <= pending_is_nack ? COM_CMD_NAK : COM_CMD_ACK;
                tx_addr <= pending_addr;
                tx_len  <= 16'd0;

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
                tx_payload_valid <= 1;
            end

            // ----------------------------------------------------
            READ_START: begin
                tx_cmd <= COM_CMD_READ_DATA;
                tx_addr <= read_resp_addr;
                tx_len <= read_resp_len;
                tx_start <= 1'b1;

                if (read_bytes_left == 16'd0) begin
                    state <= READ_PAYLOAD;
                end else begin
                    data_mem_req_valid[0] <= 1'b1;
                    data_mem_req_write[0] <= 1'b0;
                    data_mem_req_addr[0 +: ADDR_WIDTH] <= read_addr[ADDR_WIDTH-1:0];
                    if (data_mem_req_ready[0]) begin
                        read_addr <= read_addr + 16'd1;
                        state <= READ_WAIT;
                    end
                end
            end

            // ----------------------------------------------------
            READ_WAIT: begin
                if (data_mem_resp_valid[0]) begin
                    rsp_buffer <= data_mem_resp_rdata[0 +: DATA_WIDTH];
                    state <= READ_SEND;
                end
            end

            // ----------------------------------------------------
            READ_SEND: begin
                if (!read_sending_word) begin
                    if (tx_payload_advance || read_payload_requested) begin
                        tx_payload_data <= rsp_buffer;
                        tx_payload_valid <= 1'b1;
                        read_sending_word <= 1'b1;
                    end
                end else begin
                    tx_payload_data <= rsp_buffer;
                    tx_payload_valid <= 1'b1;

                    if (!tx_payload_advance) begin
                        read_sending_word <= 1'b0;
                        read_payload_requested <= 1'b0;

                        if (read_bytes_left > 16'd4) begin
                            read_bytes_left <= read_bytes_left - 16'd4;
                        end else begin
                            read_bytes_left <= 16'd0;
                        end

                        state <= READ_PAYLOAD;
                    end
                end
            end

            // ----------------------------------------------------
            READ_PAYLOAD: begin
                if (read_bytes_left != 16'd0) begin
                    data_mem_req_valid[0] <= 1'b1;
                    data_mem_req_write[0] <= 1'b0;
                    data_mem_req_addr[0 +: ADDR_WIDTH] <= read_addr[ADDR_WIDTH-1:0];
                    if (data_mem_req_ready[0]) begin
                        read_addr <= read_addr + 16'd1;
                        state <= READ_WAIT;
                    end
                end else if (!tx_busy) begin
                    state <= IDLE;
                end
            end

            // ----------------------------------------------------
            WAIT_WRITE_RESPONSE: begin
                if (mem_write_done) begin
                    write_done_hold <= 1'b1;
                    memory_status_consumed <= 1'b1;
                    state <= IDLE;
                end else if (mem_write_fail) begin
                    write_fail_hold <= 1'b1;
                    memory_status_consumed <= 1'b1;
                    state <= IDLE;
                end
            end

            // ----------------------------------------------------
            STATUS_START: begin
                tx_cmd <= COM_CMD_READ_STATUS;
                tx_addr <= read_resp_addr;
                tx_len <= 16'd4;
                tx_start <= 1'b1;
                read_sending_word <= 1'b0;
                state <= STATUS_SEND;
            end

            // ----------------------------------------------------
            STATUS_SEND: begin
                if (!read_sending_word) begin
                    if (tx_payload_advance || read_payload_requested) begin
                        tx_payload_data <= rsp_buffer;
                        tx_payload_valid <= 1'b1;
                        read_sending_word <= 1'b1;
                    end
                end else begin
                    tx_payload_data <= rsp_buffer;
                    tx_payload_valid <= 1'b1;

                    if (!tx_payload_advance) begin
                        read_sending_word <= 1'b0;
                        read_payload_requested <= 1'b0;
                        state <= STATUS_WAIT;
                    end
                end
            end

            // ----------------------------------------------------
            STATUS_WAIT: begin
                if (!tx_busy) begin
                    state <= IDLE;
                end
            end

            // ----------------------------------------------------
            WAIT_VALIDATE_KERNEL_RESPONSE: begin
                if (validate_done) begin
                    write_done_hold <= 1'b1;
                    state <= IDLE;
                end else if (validate_fail) begin
                    write_fail_hold <= 1'b1;
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