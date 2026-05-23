`timescale 1ns/1ps
`default_nettype none

module hmac_sha256_packet #(
    parameter MAX_PAYLOAD = 255
) (
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         start,
    input  wire [255:0]                 hmac_key,
    input  wire [7:0]                   cmd,
    input  wire [15:0]                  addr,
    input  wire [15:0]                  len,
    input  wire [15:0]                  packet_counter,
    input  wire [(MAX_PAYLOAD*8)-1:0]   payload_flat,
    output reg  [127:0]                 tag,
    output reg                          tag_valid,
    output wire                         busy,

    output reg                          sha_start,
    output reg                          sha_byte_valid,
    output reg  [7:0]                   sha_byte_data,
    output reg                          sha_finish,
    input  wire                         sha_ready,
    input  wire [255:0]                 sha_digest,
    input  wire                         sha_digest_valid
);

    localparam H_IDLE         = 4'd0;
    localparam H_INNER_START  = 4'd1;
    localparam H_INNER_FEED   = 4'd2;
    localparam H_INNER_FINISH = 4'd3;
    localparam H_INNER_WAIT   = 4'd4;
    localparam H_OUTER_START  = 4'd5;
    localparam H_OUTER_FEED   = 4'd6;
    localparam H_OUTER_FINISH = 4'd7;
    localparam H_OUTER_WAIT   = 4'd8;
    localparam H_INNER_FINISH_WAIT = 4'd9;
    localparam H_OUTER_FINISH_WAIT = 4'd10;

    reg [3:0] state;
    reg [15:0] feed_index;
    reg [15:0] inner_len;
    reg [255:0] inner_digest;

    assign busy = (state != H_IDLE);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= H_IDLE;
            feed_index <= 16'd0;
            inner_len <= 16'd0;
            inner_digest <= 256'b0;
            tag <= 128'b0;
            tag_valid <= 1'b0;
            sha_start <= 1'b0;
            sha_byte_valid <= 1'b0;
            sha_byte_data <= 8'b0;
            sha_finish <= 1'b0;
        end else begin
            tag_valid <= 1'b0;
            sha_start <= 1'b0;
            sha_byte_valid <= 1'b0;
            sha_finish <= 1'b0;

            case (state)
                H_IDLE: begin
                    if (start) begin
                        inner_len <= len + 16'd7;
                        feed_index <= 16'd0;
                        state <= H_INNER_START;
                    end
                end

                H_INNER_START: begin
                    sha_start <= 1'b1;
                    state <= H_INNER_FEED;
                end

                H_INNER_FEED: begin
                    if (sha_ready) begin
                        sha_byte_valid <= 1'b1;
                        sha_byte_data <= inner_byte(feed_index);
                        if (feed_index == (16'd32 + inner_len - 16'd1)) begin
                            feed_index <= 16'd0;
                            state <= H_INNER_FINISH;
                        end else begin
                            feed_index <= feed_index + 16'd1;
                        end
                    end
                end

                H_INNER_FINISH: begin
                    if (sha_ready) begin
                        sha_finish <= 1'b1;
                        state <= H_INNER_FINISH_WAIT;
                    end
                end

                H_INNER_FINISH_WAIT: begin
                    state <= H_INNER_WAIT;
                end

                H_INNER_WAIT: begin
                    if (sha_digest_valid) begin
                        inner_digest <= sha_digest;
                        feed_index <= 16'd0;
                        state <= H_OUTER_START;
                    end
                end

                H_OUTER_START: begin
                    sha_start <= 1'b1;
                    state <= H_OUTER_FEED;
                end

                H_OUTER_FEED: begin
                    if (sha_ready) begin
                        sha_byte_valid <= 1'b1;
                        sha_byte_data <= outer_byte(feed_index);
                        if (feed_index == 16'd64) begin
                            feed_index <= 16'd0;
                            state <= H_OUTER_FINISH;
                        end else begin
                            feed_index <= feed_index + 16'd1;
                        end
                    end
                end

                H_OUTER_FINISH: begin
                    if (sha_ready) begin
                        sha_finish <= 1'b1;
                        state <= H_OUTER_FINISH_WAIT;
                    end
                end

                H_OUTER_FINISH_WAIT: begin
                    state <= H_OUTER_WAIT;
                end

                H_OUTER_WAIT: begin
                    if (sha_digest_valid) begin
                        tag <= sha_digest[255:128];
                        tag_valid <= 1'b1;
                        state <= H_IDLE;
                    end
                end

                default: state <= H_IDLE;
            endcase
        end
    end

    function [7:0] key_byte;
        input [15:0] index;
        begin
            key_byte = hmac_key[(31 - index[4:0]) * 8 +: 8];
        end
    endfunction

    function [7:0] message_byte;
        input [15:0] index;
        begin
            case (index)
                16'd0: message_byte = cmd;
                16'd1: message_byte = addr[15:8];
                16'd2: message_byte = addr[7:0];
                16'd3: message_byte = len[15:8];
                16'd4: message_byte = len[7:0];
                16'd5: message_byte = packet_counter[15:8];
                16'd6: message_byte = packet_counter[7:0];
                default: message_byte = payload_flat[(index - 16'd7) * 8 +: 8];
            endcase
        end
    endfunction

    function [7:0] inner_byte;
        input [15:0] index;
        begin
            if (index < 16'd32)
                inner_byte = key_byte(index) ^ 8'h36;
            else
                inner_byte = message_byte(index - 16'd32);
        end
    endfunction

    function [7:0] outer_byte;
        input [15:0] index;
        begin
            if (index < 16'd32)
                outer_byte = key_byte(index) ^ 8'h5c;
            else if (index < 16'd64)
                outer_byte = inner_digest[(31 - (index - 16'd32)) * 8 +: 8];
            else
                outer_byte = 8'h00;
        end
    endfunction

endmodule

`default_nettype wire
