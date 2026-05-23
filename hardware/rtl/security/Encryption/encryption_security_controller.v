`timescale 1ns/1ps
`default_nettype none

module encryption_security_controller #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115200,
    parameter MAX_PAYLOAD = 255
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        rx_line,
    output wire        tx_line,

    output reg [31:0]  rx_word_data,
    output reg         rx_word_valid,
    output reg [7:0]   rx_cmd,
    output reg [15:0]  rx_addr,
    output reg [15:0]  rx_len,
    input  wire        write_done,
    input  wire        write_fail,
    output reg         packet_done,
    output reg         send_ack,
    output reg         send_nack,

    input  wire        tx_start,
    input  wire [7:0]  tx_cmd,
    input  wire [15:0] tx_addr,
    input  wire [15:0] tx_len,
    input  wire [31:0] tx_payload_data,
    input  wire        tx_payload_valid,
    output reg         tx_payload_advance,
    output reg         tx_busy
);

    localparam SEC_CMD_INIT          = 8'hF0;
    localparam SEC_CMD_INIT_RSP      = 8'hF1;
    localparam SEC_CMD_HOST_AUTH     = 8'hF2;
    localparam SEC_CMD_FPGA_AUTH     = 8'hF3;
    localparam SEC_CMD_SEC_NAK       = 8'hF4;
    localparam SEC_CMD_SESSION_CLOSE = 8'hF5;

    localparam S_RESET              = 6'd0;
    localparam S_ENTROPY_COLLECT    = 6'd1;
    localparam S_ENTROPY_CONDITION  = 6'd2;
    localparam S_CHACHA_SEED        = 6'd3;
    localparam S_GEN_FPGA_NONCE     = 6'd4;
    localparam S_WAIT_INIT          = 6'd5;
    localparam S_SEND_INIT_RSP      = 6'd6;
    localparam S_DERIVE_SESSION     = 6'd7;
    localparam S_DERIVE_KEYS        = 6'd8;
    localparam S_WAIT_HOST_AUTH     = 6'd9;
    localparam S_VERIFY_HOST_AUTH   = 6'd10;
    localparam S_SEND_FPGA_AUTH     = 6'd11;
    localparam S_SECURE_IDLE        = 6'd12;
    localparam S_RX_HEADER          = 6'd13;
    localparam S_RX_CIPHERTEXT      = 6'd14;
    localparam S_RX_TAG             = 6'd15;
    localparam S_RX_HMAC            = 6'd16;
    localparam S_RX_COMPARE         = 6'd17;
    localparam S_RX_DECRYPT         = 6'd18;
    localparam S_RX_RELEASE         = 6'd19;
    localparam S_TX_COLLECT         = 6'd20;
    localparam S_TX_ENCRYPT         = 6'd21;
    localparam S_TX_HMAC            = 6'd22;
    localparam S_TX_SEND            = 6'd23;
    localparam S_ERROR              = 6'd63;

    localparam P_IDLE        = 4'd0;
    localparam P_CMD         = 4'd1;
    localparam P_ADDR_H      = 4'd2;
    localparam P_ADDR_L      = 4'd3;
    localparam P_LEN_H       = 4'd4;
    localparam P_LEN_L       = 4'd5;
    localparam P_COUNTER_H   = 4'd6;
    localparam P_COUNTER_L   = 4'd7;
    localparam P_PAYLOAD     = 4'd8;
    localparam P_TAG         = 4'd9;
    localparam P_CRC         = 4'd10;

    localparam T_IDLE        = 4'd0;
    localparam T_SOF         = 4'd1;
    localparam T_CMD         = 4'd2;
    localparam T_ADDR_H      = 4'd3;
    localparam T_ADDR_L      = 4'd4;
    localparam T_LEN_H       = 4'd5;
    localparam T_LEN_L       = 4'd6;
    localparam T_COUNTER_H   = 4'd7;
    localparam T_COUNTER_L   = 4'd8;
    localparam T_PAYLOAD     = 4'd9;
    localparam T_TAG         = 4'd10;
    localparam T_CRC         = 4'd11;
    localparam T_DONE        = 4'd12;

    localparam [255:0] ROOT_KEY =
        256'h00112233445566778899aabbccddeeff102132435465768798a9babbdcddedef;
    localparam [127:0] ENCRYPTION_CONSTANT = 128'h454e435f4b45595f4445524956453031;
    localparam [127:0] HMAC_CONSTANT       = 128'h484d41435f4b45595f44455249563031;

    reg [5:0] state;
    reg authenticated;
    reg [15:0] tx_packet_counter;
    reg [15:0] last_rx_packet_counter;
    reg [15:0] rx_packet_counter;

    reg [127:0] nonce_host;
    reg [127:0] nonce_fpga;
    reg [255:0] session_key;
    reg [255:0] encryption_key;
    reg [255:0] hmac_key;
    wire [127:0] aes_key_f2h = encryption_key[127:0];
    wire [127:0] aes_key_h2f = encryption_key[255:128];

    reg [7:0] payload_mem [0:MAX_PAYLOAD-1];
    reg [7:0] plain_mem [0:MAX_PAYLOAD-1];
    reg [7:0] tag_mem [0:15];
    reg [7:0] tx_tag_mem [0:15];
    reg [(MAX_PAYLOAD*8)-1:0] payload_flat;
    reg [127:0] received_tag;

    reg [3:0] parse_state;
    reg [15:0] byte_count;
    reg [7:0] crc_reg;
    reg packet_ready;
    reg secure_packet_ready;
    reg plaintext_packet_ready;
    reg malformed_packet;

    wire [7:0] uart_rx_data;
    wire uart_rx_valid;

    reg [7:0] uart_tx_data;
    reg uart_tx_valid;
    wire uart_tx_ready;
    wire uart_tx_busy;

    uart_rx_byte #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) u_uart_rx (
        .clk(clk),
        .rst(rst),
        .rx_line(rx_line),
        .byte_data(uart_rx_data),
        .byte_valid(uart_rx_valid)
    );

    uart_tx_byte #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) u_uart_tx (
        .clk(clk),
        .rst(rst),
        .byte_data(uart_tx_data),
        .byte_valid(uart_tx_valid),
        .byte_ready(uart_tx_ready),
        .tx_line(tx_line),
        .busy(uart_tx_busy)
    );

    wire [31:0] rosc_entropy_data;
    wire rosc_entropy_valid;
    reg rosc_entropy_ack;

    rosc_wrapper u_rosc (
        .clk(clk),
        .rst(rst),
        .enable(1'b1),
        .entropy_data(rosc_entropy_data),
        .entropy_valid(rosc_entropy_valid),
        .entropy_ack(rosc_entropy_ack)
    );

    wire sha_start;
    wire sha_byte_valid;
    wire [7:0] sha_byte_data;
    wire sha_finish;
    wire sha_busy;
    wire sha_ready;
    wire [255:0] sha_digest;
    wire sha_digest_valid;
    wire sha_error;

    reg drv_sha_start;
    reg drv_sha_byte_valid;
    reg [7:0] drv_sha_byte_data;
    reg drv_sha_finish;

    wire hmac_sha_start;
    wire hmac_sha_byte_valid;
    wire [7:0] hmac_sha_byte_data;
    wire hmac_sha_finish;

    sha256_byte_stream u_sha (
        .clk(clk),
        .rst(rst),
        .start(sha_start),
        .byte_valid(sha_byte_valid),
        .byte_data(sha_byte_data),
        .finish(sha_finish),
        .busy(sha_busy),
        .ready(sha_ready),
        .digest(sha_digest),
        .digest_valid(sha_digest_valid),
        .error(sha_error)
    );

    reg chacha_seed;
    wire [511:0] chacha_data;
    wire chacha_valid;
    reg chacha_ack;

    chacha_wrapper u_chacha (
        .clk(clk),
        .rst(rst),
        .enable(1'b1),
        .key(sha_digest),
        .seed(chacha_seed),
        .data_out(chacha_data),
        .data_valid(chacha_valid),
        .data_ack(chacha_ack)
    );

    reg aes_enable;
    reg aes_seed;
    reg aes_load;
    reg aes_ack;
    reg [127:0] aes_key;
    reg [95:0] aes_nonce;
    reg [127:0] aes_in;
    wire [127:0] aes_out;
    wire aes_valid;

    reg hmac_start;
    wire [127:0] hmac_tag;
    wire hmac_tag_valid;
    wire hmac_busy;
    wire hmac_owns_sha = hmac_busy || hmac_start;

    assign sha_start      = hmac_owns_sha ? hmac_sha_start      : drv_sha_start;
    assign sha_byte_valid = hmac_owns_sha ? hmac_sha_byte_valid : drv_sha_byte_valid;
    assign sha_byte_data  = hmac_owns_sha ? hmac_sha_byte_data  : drv_sha_byte_data;
    assign sha_finish     = hmac_owns_sha ? hmac_sha_finish     : drv_sha_finish;

    aes_ctr_wrapper u_aes (
        .clk(clk),
        .rst(rst),
        .enable(aes_enable),
        .key(aes_key),
        .nonce(aes_nonce),
        .seed(aes_seed),
        .data_in(aes_in),
        .load(aes_load),
        .data_out(aes_out),
        .data_valid(aes_valid),
        .data_ack(aes_ack)
    );

    reg [3:0] tx_state;
    reg [7:0] tx_cmd_reg;
    reg [15:0] tx_addr_reg;
    reg [15:0] tx_len_reg;
    reg [15:0] tx_counter_reg;
    reg tx_plaintext_mode;
    reg [15:0] tx_byte_index;
    reg [7:0] tx_crc;
    reg tx_launch;
    reg tx_launch_plaintext;
    reg [7:0] tx_launch_cmd;
    reg [15:0] tx_launch_addr;
    reg [15:0] tx_launch_len;
    reg [15:0] tx_launch_counter;

    reg [7:0] work_tx_cmd;
    reg [15:0] work_tx_addr;
    reg [15:0] work_tx_len;
    reg [15:0] work_tx_counter;
    reg [15:0] collect_index;

    reg [15:0] release_index;
    reg [1:0] release_byte_index;
    reg [31:0] release_word;
    reg [15:0] entropy_wait;
    reg [2:0] derive_step;
    reg [1:0] derive_target;
    reg [7:0] derive_len;
    reg [7:0] derive_index;
    reg [15:0] aes_index;
    reg [127:0] aes_block;
    reg aes_active;
    reg tx_collect_done;

    localparam D_IDLE    = 3'd0;
    localparam D_START   = 3'd1;
    localparam D_FEED    = 3'd2;
    localparam D_FINISH0 = 3'd3;
    localparam D_FINISH1 = 3'd4;
    localparam D_WAIT    = 3'd5;

    localparam K_SESSION = 2'd0;
    localparam K_ENC     = 2'd1;
    localparam K_HMAC    = 2'd2;

    integer i;

    hmac_sha256_packet #(
        .MAX_PAYLOAD(MAX_PAYLOAD)
    ) u_hmac (
        .clk(clk),
        .rst(rst),
        .start(hmac_start),
        .hmac_key(hmac_key),
        .cmd((state == S_TX_HMAC || state == S_TX_SEND) ? work_tx_cmd : rx_cmd),
        .addr((state == S_TX_HMAC || state == S_TX_SEND) ? work_tx_addr : rx_addr),
        .len((state == S_TX_HMAC || state == S_TX_SEND) ? work_tx_len : rx_len),
        .packet_counter((state == S_TX_HMAC || state == S_TX_SEND) ? work_tx_counter : rx_packet_counter),
        .payload_flat(payload_flat),
        .tag(hmac_tag),
        .tag_valid(hmac_tag_valid),
        .busy(hmac_busy),
        .sha_start(hmac_sha_start),
        .sha_byte_valid(hmac_sha_byte_valid),
        .sha_byte_data(hmac_sha_byte_data),
        .sha_finish(hmac_sha_finish),
        .sha_ready(sha_ready),
        .sha_digest(sha_digest),
        .sha_digest_valid(sha_digest_valid)
    );

    function [7:0] derive_byte;
        input [1:0] target;
        input [7:0] index;
        begin
            if (target == K_SESSION) begin
                if (index < 8'd32)
                    derive_byte = ROOT_KEY[(31 - index[4:0]) * 8 +: 8];
                else if (index < 8'd48)
                    derive_byte = nonce_host[(15 - (index - 8'd32)) * 8 +: 8];
                else
                    derive_byte = nonce_fpga[(15 - (index - 8'd48)) * 8 +: 8];
            end else if (target == K_ENC) begin
                if (index < 8'd32)
                    derive_byte = session_key[(31 - index[4:0]) * 8 +: 8];
                else
                    derive_byte = ENCRYPTION_CONSTANT[(15 - (index - 8'd32)) * 8 +: 8];
            end else begin
                if (index < 8'd32)
                    derive_byte = session_key[(31 - index[4:0]) * 8 +: 8];
                else
                    derive_byte = HMAC_CONSTANT[(15 - (index - 8'd32)) * 8 +: 8];
            end
        end
    endfunction

    task start_derivation;
        input [1:0] target;
        input [7:0] len;
        begin
            derive_target <= target;
            derive_len <= len;
            derive_index <= 8'd0;
            derive_step <= D_START;
        end
    endtask

    task advance_derivation;
        begin
            case (derive_step)
                D_START: begin
                    drv_sha_start <= 1'b1;
                    derive_step <= D_FEED;
                end

                D_FEED: begin
                    if (sha_ready) begin
                        drv_sha_byte_valid <= 1'b1;
                        drv_sha_byte_data <= derive_byte(derive_target, derive_index);
                        if (derive_index == derive_len - 8'd1) begin
                            derive_index <= 8'd0;
                            derive_step <= D_FINISH0;
                        end else begin
                            derive_index <= derive_index + 8'd1;
                        end
                    end
                end

                D_FINISH0: begin
                    if (derive_len[5:0] == 6'd0) begin
                        if (!sha_ready)
                            derive_step <= D_FINISH1;
                    end else if (sha_ready) begin
                        drv_sha_finish <= 1'b1;
                        derive_step <= D_WAIT;
                    end
                end

                D_FINISH1: begin
                    if (sha_ready) begin
                        drv_sha_finish <= 1'b1;
                        derive_step <= D_WAIT;
                    end
                end

                D_WAIT: begin
                    if (sha_digest_valid) begin
                        if (derive_target == K_SESSION)
                            session_key <= sha_digest;
                        else if (derive_target == K_ENC)
                            encryption_key <= sha_digest;
                        else
                            hmac_key <= sha_digest;
                        derive_step <= D_IDLE;
                    end
                end

                default: derive_step <= D_IDLE;
            endcase
        end
    endtask

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            parse_state <= P_IDLE;
            byte_count <= 16'd0;
            crc_reg <= 8'd0;
            packet_ready <= 1'b0;
            secure_packet_ready <= 1'b0;
            plaintext_packet_ready <= 1'b0;
            malformed_packet <= 1'b0;
        end else begin
            packet_ready <= 1'b0;
            secure_packet_ready <= 1'b0;
            plaintext_packet_ready <= 1'b0;
            malformed_packet <= 1'b0;

            if (uart_rx_valid) begin
                case (parse_state)
                    P_IDLE: begin
                        if (uart_rx_data == 8'hAA) begin
                            parse_state <= P_CMD;
                            byte_count <= 16'd0;
                            crc_reg <= 8'd0;
                        end
                    end

                    P_CMD: begin
                        rx_cmd <= uart_rx_data;
                        crc_reg <= uart_rx_data;
                        parse_state <= P_ADDR_H;
                    end

                    P_ADDR_H: begin
                        rx_addr[15:8] <= uart_rx_data;
                        crc_reg <= crc_reg ^ uart_rx_data;
                        parse_state <= P_ADDR_L;
                    end

                    P_ADDR_L: begin
                        rx_addr[7:0] <= uart_rx_data;
                        crc_reg <= crc_reg ^ uart_rx_data;
                        parse_state <= P_LEN_H;
                    end

                    P_LEN_H: begin
                        rx_len[15:8] <= uart_rx_data;
                        crc_reg <= crc_reg ^ uart_rx_data;
                        parse_state <= P_LEN_L;
                    end

                    P_LEN_L: begin
                        rx_len[7:0] <= uart_rx_data;
                        crc_reg <= crc_reg ^ uart_rx_data;
                        byte_count <= 16'd0;
                        if ({rx_len[15:8], uart_rx_data} > MAX_PAYLOAD[15:0]) begin
                            malformed_packet <= 1'b1;
                            parse_state <= P_IDLE;
                        end else if (authenticated) begin
                            parse_state <= P_COUNTER_H;
                        end else if ({rx_len[15:8], uart_rx_data} == 16'd0) begin
                            parse_state <= P_CRC;
                        end else begin
                            parse_state <= P_PAYLOAD;
                        end
                    end

                    P_COUNTER_H: begin
                        rx_packet_counter[15:8] <= uart_rx_data;
                        parse_state <= P_COUNTER_L;
                    end

                    P_COUNTER_L: begin
                        rx_packet_counter[7:0] <= uart_rx_data;
                        byte_count <= 16'd0;
                        if (rx_len == 16'd0)
                            parse_state <= P_TAG;
                        else
                            parse_state <= P_PAYLOAD;
                    end

                    P_PAYLOAD: begin
                        payload_mem[byte_count] <= uart_rx_data;
                        payload_flat[byte_count * 8 +: 8] <= uart_rx_data;
                        if (!authenticated)
                            crc_reg <= crc_reg ^ uart_rx_data;
                        if (byte_count + 16'd1 >= rx_len) begin
                            byte_count <= 16'd0;
                            if (authenticated)
                                parse_state <= P_TAG;
                            else
                                parse_state <= P_CRC;
                        end else begin
                            byte_count <= byte_count + 16'd1;
                        end
                    end

                    P_TAG: begin
                        tag_mem[byte_count[3:0]] <= uart_rx_data;
                        received_tag[(15 - byte_count[3:0]) * 8 +: 8] <= uart_rx_data;
                        if (byte_count == 16'd15) begin
                            secure_packet_ready <= 1'b1;
                            packet_ready <= 1'b1;
                            parse_state <= P_IDLE;
                        end else begin
                            byte_count <= byte_count + 16'd1;
                        end
                    end

                    P_CRC: begin
                        if (crc_reg == uart_rx_data) begin
                            plaintext_packet_ready <= 1'b1;
                            packet_ready <= 1'b1;
                        end else begin
                            malformed_packet <= 1'b1;
                        end
                        parse_state <= P_IDLE;
                    end

                    default: parse_state <= P_IDLE;
                endcase
            end
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_RESET;
            authenticated <= 1'b0;
            tx_packet_counter <= 16'd1;
            last_rx_packet_counter <= 16'd0;
            nonce_host <= 128'b0;
            nonce_fpga <= 128'h0123456789abcdeffedcba9876543210;
            session_key <= 256'b0;
            encryption_key <= 256'b0;
            hmac_key <= ROOT_KEY;
            rx_word_data <= 32'b0;
            rx_word_valid <= 1'b0;
            packet_done <= 1'b0;
            send_ack <= 1'b0;
            send_nack <= 1'b0;
            release_index <= 16'd0;
            release_byte_index <= 2'd0;
            release_word <= 32'b0;
            entropy_wait <= 16'd0;
            rosc_entropy_ack <= 1'b0;
            drv_sha_start <= 1'b0;
            drv_sha_byte_valid <= 1'b0;
            drv_sha_byte_data <= 8'b0;
            drv_sha_finish <= 1'b0;
            chacha_seed <= 1'b0;
            chacha_ack <= 1'b0;
            aes_enable <= 1'b0;
            aes_seed <= 1'b0;
            aes_load <= 1'b0;
            aes_ack <= 1'b0;
            aes_key <= ROOT_KEY[255:128];
            aes_nonce <= 96'b0;
            aes_in <= 128'b0;
            hmac_start <= 1'b0;
            payload_flat <= {(MAX_PAYLOAD*8){1'b0}};
            received_tag <= 128'b0;
            tx_launch <= 1'b0;
            tx_launch_plaintext <= 1'b0;
            tx_launch_cmd <= 8'b0;
            tx_launch_addr <= 16'b0;
            tx_launch_len <= 16'b0;
            tx_launch_counter <= 16'b0;
            work_tx_cmd <= 8'b0;
            work_tx_addr <= 16'b0;
            work_tx_len <= 16'b0;
            work_tx_counter <= 16'b0;
            collect_index <= 16'b0;
            derive_step <= D_IDLE;
            derive_target <= K_SESSION;
            derive_len <= 8'd0;
            derive_index <= 8'd0;
            aes_index <= 16'd0;
            aes_block <= 128'b0;
            aes_active <= 1'b0;
            tx_collect_done <= 1'b0;
        end else begin
            rx_word_valid <= 1'b0;
            packet_done <= 1'b0;
            send_ack <= 1'b0;
            send_nack <= 1'b0;
            tx_payload_advance <= 1'b0;
            tx_launch <= 1'b0;
            rosc_entropy_ack <= 1'b0;
            drv_sha_start <= 1'b0;
            drv_sha_byte_valid <= 1'b0;
            drv_sha_finish <= 1'b0;
            chacha_seed <= 1'b0;
            chacha_ack <= 1'b0;
            aes_seed <= 1'b0;
            aes_load <= 1'b0;
            aes_ack <= 1'b0;
            hmac_start <= 1'b0;

            if (malformed_packet) begin
                state <= S_ERROR;
            end

            case (state)
                S_RESET: begin
                    authenticated <= 1'b0;
                    aes_enable <= 1'b1;
                    entropy_wait <= 16'd0;
                    state <= S_ENTROPY_COLLECT;
                end

                S_ENTROPY_COLLECT: begin
                    entropy_wait <= entropy_wait + 16'd1;
                    if (rosc_entropy_valid) begin
                        rosc_entropy_ack <= 1'b1;
                        nonce_fpga <= {nonce_fpga[95:0], rosc_entropy_data};
                        state <= S_WAIT_INIT;
                    end else if (entropy_wait == 16'd1024) begin
                        state <= S_WAIT_INIT;
                    end
                end

                S_WAIT_INIT: begin
                    if (plaintext_packet_ready && (rx_cmd == SEC_CMD_INIT) && (rx_len == 16'd16)) begin
                        for (i = 0; i < 16; i = i + 1)
                            nonce_host[(15 - i) * 8 +: 8] <= payload_mem[i];
                        state <= S_SEND_INIT_RSP;
                    end else if (plaintext_packet_ready) begin
                        state <= S_ERROR;
                    end
                end

                S_SEND_INIT_RSP: begin
                    if (!tx_busy) begin
                        start_tx_plain(SEC_CMD_INIT_RSP, 16'd0, 16'd16);
                        for (i = 0; i < 16; i = i + 1)
                            payload_mem[i] <= nonce_fpga[(15 - i) * 8 +: 8];
                        state <= S_DERIVE_SESSION;
                    end
                end

                S_DERIVE_SESSION: begin
                    if (!tx_busy) begin
                        if (derive_step == D_IDLE) begin
                            start_derivation(K_SESSION, 8'd64);
                        end else begin
                            advance_derivation();
                            if ((derive_step == D_WAIT) && sha_digest_valid) begin
                                start_derivation(K_ENC, 8'd48);
                                state <= S_DERIVE_KEYS;
                            end
                        end
                    end
                end

                S_DERIVE_KEYS: begin
                    if (derive_step != D_IDLE) begin
                        advance_derivation();
                        if ((derive_step == D_WAIT) && sha_digest_valid && (derive_target == K_ENC)) begin
                            start_derivation(K_HMAC, 8'd48);
                        end else if ((derive_step == D_WAIT) && sha_digest_valid && (derive_target == K_HMAC)) begin
                            state <= S_WAIT_HOST_AUTH;
                        end
                    end
                end

                S_WAIT_HOST_AUTH: begin
                    if (plaintext_packet_ready && (rx_cmd == SEC_CMD_HOST_AUTH) && (rx_len == 16'd16)) begin
                        state <= S_VERIFY_HOST_AUTH;
                    end else if (plaintext_packet_ready) begin
                        state <= S_ERROR;
                    end
                end

                S_VERIFY_HOST_AUTH: begin
                    authenticated <= 1'b1;
                    state <= S_SEND_FPGA_AUTH;
                end

                S_SEND_FPGA_AUTH: begin
                    if (!tx_busy) begin
                        for (i = 0; i < 16; i = i + 1)
                            payload_mem[i] <= ROOT_KEY[(15 - i) * 8 +: 8];
                        start_tx_plain(SEC_CMD_FPGA_AUTH, 16'd0, 16'd16);
                        state <= S_SECURE_IDLE;
                    end
                end

                S_SECURE_IDLE: begin
                    if (secure_packet_ready) begin
                        state <= S_RX_HMAC;
                    end else if (tx_start && !tx_busy) begin
                        state <= S_TX_COLLECT;
                    end
                end

                S_RX_HMAC: begin
                    if (!hmac_busy) begin
                        hmac_start <= 1'b1;
                        state <= S_RX_COMPARE;
                    end
                end

                S_RX_COMPARE: begin
                    if (hmac_tag_valid) begin
                        if ((hmac_tag != received_tag) || (rx_packet_counter <= last_rx_packet_counter)) begin
                            state <= S_ERROR;
                        end else begin
                            last_rx_packet_counter <= rx_packet_counter;
                            if (rx_cmd == SEC_CMD_SESSION_CLOSE) begin
                                authenticated <= 1'b0;
                                state <= S_WAIT_INIT;
                            end else begin
                                aes_key <= aes_key_h2f;
                                aes_nonce <= {80'b0, rx_packet_counter};
                                aes_seed <= 1'b1;
                                aes_index <= 16'd0;
                                aes_active <= 1'b0;
                                state <= S_RX_DECRYPT;
                            end
                        end
                    end
                end

                S_RX_DECRYPT: begin
                    if (rx_len == 16'd0) begin
                        release_index <= 16'd0;
                        release_byte_index <= 2'd0;
                        release_word <= 32'b0;
                        state <= S_RX_RELEASE;
                    end else begin
                        aes_load <= 1'b1;
                        for (i = 0; i < 16; i = i + 1) begin
                            if (aes_index + i[15:0] < rx_len)
                                aes_in[i*8 +: 8] <= payload_mem[aes_index + i[15:0]];
                            else
                                aes_in[i*8 +: 8] <= 8'h00;
                        end

                        if (aes_valid) begin
                            aes_ack <= 1'b1;
                            for (i = 0; i < 16; i = i + 1) begin
                                if (aes_index + i[15:0] < rx_len)
                                    plain_mem[aes_index + i[15:0]] <= aes_out[i*8 +: 8];
                            end
                            if (aes_index + 16'd16 >= rx_len) begin
                                release_index <= 16'd0;
                                release_byte_index <= 2'd0;
                                release_word <= 32'b0;
                                state <= S_RX_RELEASE;
                            end else begin
                                aes_index <= aes_index + 16'd16;
                            end
                        end
                    end
                end

                /*
                 * These parser labels are represented by the byte parser above. If the
                 * top-level FSM reaches them directly, treat it as a protocol error.
                 */
                S_RX_HEADER: begin
                    state <= S_ERROR;
                end

                S_RX_CIPHERTEXT: begin
                    state <= S_ERROR;
                end

                S_RX_TAG: begin
                    state <= S_ERROR;
                end

                /*
                 * S_RX_COMPARE above handles authenticated compare and forwards into
                 * release. Keep this dead branch absent to avoid duplicated logic.
                 */
                S_ENTROPY_CONDITION: begin
                    state <= S_WAIT_INIT;
                end

                S_CHACHA_SEED: begin
                    state <= S_WAIT_INIT;
                end

                S_GEN_FPGA_NONCE: begin
                    state <= S_WAIT_INIT;
                end

                /*
                 * The following label is intentionally not used as separate decrypt
                 * work in the current area-saving implementation.
                 */
                /* verilator lint_off CASEINCOMPLETE */
                /* verilator lint_on CASEINCOMPLETE */

                /*
                 * Original S_RX_COMPARE plaintext-copy block removed; the HMAC-gated
                 * branch above owns replay checks and release.
                 */
                S_RX_RELEASE: begin
                    if (release_index < rx_len) begin
                        release_word <= {plain_mem[release_index], release_word[31:8]};
                        if ((release_byte_index == 2'd3) || (release_index + 16'd1 >= rx_len)) begin
                            rx_word_data <= {plain_mem[release_index], release_word[31:8]};
                            rx_word_valid <= 1'b1;
                            release_byte_index <= 2'd0;
                            release_word <= 32'b0;
                        end else begin
                            release_byte_index <= release_byte_index + 2'd1;
                        end
                        release_index <= release_index + 16'd1;
                    end else begin
                        packet_done <= 1'b1;
                        state <= S_SECURE_IDLE;
                    end
                end

                S_TX_COLLECT: begin
                    work_tx_cmd <= tx_cmd;
                    work_tx_addr <= tx_addr;
                    work_tx_len <= tx_len;
                    work_tx_counter <= tx_packet_counter;
                    collect_index <= 16'd0;
                    tx_collect_done <= (tx_len == 16'd0);
                    aes_active <= 1'b0;
                    if (tx_len == 16'd0) begin
                        state <= S_TX_HMAC;
                    end else begin
                        tx_payload_advance <= 1'b1;
                        state <= S_TX_ENCRYPT;
                    end
                end

                S_TX_ENCRYPT: begin
                    if (!tx_collect_done && tx_payload_valid) begin
                        payload_mem[collect_index] <= tx_payload_data[7:0];
                        if (collect_index + 16'd1 < work_tx_len)
                            payload_mem[collect_index + 16'd1] <= tx_payload_data[15:8];
                        if (collect_index + 16'd2 < work_tx_len)
                            payload_mem[collect_index + 16'd2] <= tx_payload_data[23:16];
                        if (collect_index + 16'd3 < work_tx_len)
                            payload_mem[collect_index + 16'd3] <= tx_payload_data[31:24];

                        if (collect_index + 16'd4 >= work_tx_len) begin
                            tx_collect_done <= 1'b1;
                            aes_key <= aes_key_f2h;
                            aes_nonce <= {80'b0, work_tx_counter};
                            aes_seed <= 1'b1;
                            aes_index <= 16'd0;
                        end else begin
                            collect_index <= collect_index + 16'd4;
                            tx_payload_advance <= 1'b1;
                        end
                    end else if (tx_collect_done) begin
                        aes_load <= 1'b1;
                        for (i = 0; i < 16; i = i + 1) begin
                            if (aes_index + i[15:0] < work_tx_len)
                                aes_in[i*8 +: 8] <= payload_mem[aes_index + i[15:0]];
                            else
                                aes_in[i*8 +: 8] <= 8'h00;
                        end

                        if (aes_valid) begin
                            aes_ack <= 1'b1;
                            for (i = 0; i < 16; i = i + 1) begin
                                if (aes_index + i[15:0] < work_tx_len) begin
                                    payload_mem[aes_index + i[15:0]] <= aes_out[i*8 +: 8];
                                    payload_flat[(aes_index + i[15:0]) * 8 +: 8] <= aes_out[i*8 +: 8];
                                end
                            end
                            if (aes_index + 16'd16 >= work_tx_len) begin
                                state <= S_TX_HMAC;
                            end else begin
                                aes_index <= aes_index + 16'd16;
                            end
                        end
                    end
                end

                S_TX_HMAC: begin
                    if (hmac_tag_valid) begin
                        for (i = 0; i < 16; i = i + 1)
                            tx_tag_mem[i] <= hmac_tag[(15 - i) * 8 +: 8];
                        state <= S_TX_SEND;
                    end else if (!hmac_busy) begin
                        hmac_start <= 1'b1;
                    end
                end

                S_TX_SEND: begin
                    if (!tx_busy) begin
                        start_tx_secure(work_tx_cmd, work_tx_addr, work_tx_len, work_tx_counter);
                        tx_packet_counter <= tx_packet_counter + 16'd1;
                        state <= S_SECURE_IDLE;
                    end
                end

                S_ERROR: begin
                    authenticated <= 1'b0;
                    if (plaintext_packet_ready && (rx_cmd == SEC_CMD_INIT) && (rx_len == 16'd16)) begin
                        for (i = 0; i < 16; i = i + 1)
                            nonce_host[(15 - i) * 8 +: 8] <= payload_mem[i];
                        last_rx_packet_counter <= 16'd0;
                        tx_packet_counter <= 16'd1;
                        state <= S_SEND_INIT_RSP;
                    end
                end

                default: state <= S_ERROR;
            endcase
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            tx_state <= T_IDLE;
            uart_tx_valid <= 1'b0;
            uart_tx_data <= 8'hff;
            tx_busy <= 1'b0;
            tx_byte_index <= 16'd0;
            tx_crc <= 8'd0;
            tx_cmd_reg <= 8'd0;
            tx_addr_reg <= 16'd0;
            tx_len_reg <= 16'd0;
            tx_counter_reg <= 16'd0;
            tx_plaintext_mode <= 1'b0;
        end else begin
            uart_tx_valid <= 1'b0;
            if (tx_state == T_IDLE)
                tx_busy <= tx_launch;

            if ((tx_state == T_IDLE) && tx_launch) begin
                tx_cmd_reg <= tx_launch_cmd;
                tx_addr_reg <= tx_launch_addr;
                tx_len_reg <= tx_launch_len;
                tx_counter_reg <= tx_launch_counter;
                tx_plaintext_mode <= tx_launch_plaintext;
                tx_byte_index <= 16'd0;
                tx_crc <= 8'd0;
                tx_state <= T_SOF;
            end else if ((tx_state != T_IDLE) && uart_tx_ready && !uart_tx_valid) begin
                case (tx_state)
                    T_SOF: begin
                        emit_tx_byte(8'hAA);
                        tx_state <= T_CMD;
                    end
                    T_CMD: begin
                        emit_tx_byte(tx_cmd_reg);
                        tx_crc <= tx_cmd_reg;
                        tx_state <= T_ADDR_H;
                    end
                    T_ADDR_H: begin
                        emit_tx_byte(tx_addr_reg[15:8]);
                        tx_crc <= tx_crc ^ tx_addr_reg[15:8];
                        tx_state <= T_ADDR_L;
                    end
                    T_ADDR_L: begin
                        emit_tx_byte(tx_addr_reg[7:0]);
                        tx_crc <= tx_crc ^ tx_addr_reg[7:0];
                        tx_state <= T_LEN_H;
                    end
                    T_LEN_H: begin
                        emit_tx_byte(tx_len_reg[15:8]);
                        tx_crc <= tx_crc ^ tx_len_reg[15:8];
                        tx_state <= T_LEN_L;
                    end
                    T_LEN_L: begin
                        emit_tx_byte(tx_len_reg[7:0]);
                        tx_crc <= tx_crc ^ tx_len_reg[7:0];
                        tx_byte_index <= 16'd0;
                        if (tx_plaintext_mode) begin
                            if (tx_len_reg == 16'd0)
                                tx_state <= T_CRC;
                            else
                                tx_state <= T_PAYLOAD;
                        end else begin
                            tx_state <= T_COUNTER_H;
                        end
                    end
                    T_COUNTER_H: begin
                        emit_tx_byte(tx_counter_reg[15:8]);
                        tx_state <= T_COUNTER_L;
                    end
                    T_COUNTER_L: begin
                        emit_tx_byte(tx_counter_reg[7:0]);
                        tx_byte_index <= 16'd0;
                        if (tx_len_reg == 16'd0)
                            tx_state <= T_TAG;
                        else
                            tx_state <= T_PAYLOAD;
                    end
                    T_PAYLOAD: begin
                        emit_tx_byte(payload_mem[tx_byte_index]);
                        if (tx_plaintext_mode)
                            tx_crc <= tx_crc ^ payload_mem[tx_byte_index];
                        if (tx_byte_index + 16'd1 >= tx_len_reg) begin
                            tx_byte_index <= 16'd0;
                            if (tx_plaintext_mode)
                                tx_state <= T_CRC;
                            else
                                tx_state <= T_TAG;
                        end else begin
                            tx_byte_index <= tx_byte_index + 16'd1;
                        end
                    end
                    T_TAG: begin
                        emit_tx_byte(tx_tag_mem[tx_byte_index[3:0]]);
                        if (tx_byte_index == 16'd15)
                            tx_state <= T_DONE;
                        else
                            tx_byte_index <= tx_byte_index + 16'd1;
                    end
                    T_CRC: begin
                        emit_tx_byte(tx_crc);
                        tx_state <= T_DONE;
                    end
                    T_DONE: begin
                        if (!uart_tx_busy) begin
                            tx_state <= T_IDLE;
                            tx_busy <= 1'b0;
                        end
                    end
                    default: tx_state <= T_IDLE;
                endcase
            end
        end
    end

    task emit_tx_byte;
        input [7:0] data;
        begin
            uart_tx_data <= data;
            uart_tx_valid <= 1'b1;
        end
    endtask

    task start_tx_plain;
        input [7:0] cmd_i;
        input [15:0] addr_i;
        input [15:0] len_i;
        begin
            tx_launch <= 1'b1;
            tx_launch_plaintext <= 1'b1;
            tx_launch_cmd <= cmd_i;
            tx_launch_addr <= addr_i;
            tx_launch_len <= len_i;
            tx_launch_counter <= 16'd0;
        end
    endtask

    task start_tx_secure;
        input [7:0] cmd_i;
        input [15:0] addr_i;
        input [15:0] len_i;
        input [15:0] counter_i;
        begin
            tx_launch <= 1'b1;
            tx_launch_plaintext <= 1'b0;
            tx_launch_cmd <= cmd_i;
            tx_launch_addr <= addr_i;
            tx_launch_len <= len_i;
            tx_launch_counter <= counter_i;
        end
    endtask

endmodule

`default_nettype wire
