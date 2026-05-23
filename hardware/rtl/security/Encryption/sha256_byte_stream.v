`timescale 1ns/1ps
`default_nettype none

module sha256_byte_stream (
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    input  wire         byte_valid,
    input  wire [7:0]   byte_data,
    input  wire         finish,
    output wire         busy,
    output wire         ready,
    output wire [255:0] digest,
    output wire         digest_valid,
    output wire         error
);

    localparam S_IDLE      = 3'd0;
    localparam S_COLLECT   = 3'd1;
    localparam S_SUBMIT    = 3'd2;
    localparam S_WAIT_CORE = 3'd3;

    localparam AFTER_COLLECT    = 2'd0;
    localparam AFTER_DONE       = 2'd1;
    localparam AFTER_SECOND_PAD = 2'd2;

    reg [2:0]   state_reg;
    reg [511:0] block_reg;
    reg [5:0]   byte_index_reg;
    reg [31:0]  total_bits_reg;
    reg         first_block_reg;
    reg         submit_first_reg;
    reg [1:0]   after_core_reg;
    reg         digest_valid_reg;
    reg         error_reg;

    wire         core_ready;
    wire [255:0] core_digest;
    wire         core_digest_valid;

    wire core_init = (state_reg == S_SUBMIT) && core_ready && submit_first_reg;
    wire core_next = (state_reg == S_SUBMIT) && core_ready && !submit_first_reg;

    assign busy         = (state_reg != S_IDLE);
    assign ready        = (state_reg == S_COLLECT);
    assign digest       = core_digest;
    assign digest_valid = digest_valid_reg;
    assign error        = error_reg;

    sha256_core core (
        .clk(clk),
        .reset_n(~rst),
        .init(core_init),
        .next(core_next),
        .mode(1'b1),
        .block(block_reg),
        .ready(core_ready),
        .digest(core_digest),
        .digest_valid(core_digest_valid)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state_reg        <= S_IDLE;
            block_reg        <= 512'b0;
            byte_index_reg   <= 6'd0;
            total_bits_reg   <= 32'd0;
            first_block_reg  <= 1'b1;
            submit_first_reg <= 1'b1;
            after_core_reg   <= AFTER_COLLECT;
            digest_valid_reg <= 1'b0;
            error_reg        <= 1'b0;
        end else begin
            digest_valid_reg <= 1'b0;

            if (start && (state_reg == S_IDLE)) begin
                state_reg        <= S_COLLECT;
                block_reg        <= 512'b0;
                byte_index_reg   <= 6'd0;
                total_bits_reg   <= 32'd0;
                first_block_reg  <= 1'b1;
                submit_first_reg <= 1'b1;
                after_core_reg   <= AFTER_COLLECT;
                error_reg        <= 1'b0;
            end else begin
                case (state_reg)
                    S_IDLE: begin
                        if (byte_valid || finish)
                            error_reg <= 1'b1;
                    end

                    S_COLLECT: begin
                        if (byte_valid && finish) begin
                            error_reg <= 1'b1;
                        end else if (byte_valid) begin
                            block_reg <= block_with_byte(block_reg, byte_index_reg, byte_data);
                            total_bits_reg <= total_bits_reg + 32'd8;

                            if (byte_index_reg == 6'd63) begin
                                submit_first_reg <= first_block_reg;
                                after_core_reg <= AFTER_COLLECT;
                                state_reg <= S_SUBMIT;
                            end else begin
                                byte_index_reg <= byte_index_reg + 6'd1;
                            end
                        end else if (finish) begin
                            block_reg <= final_block(block_reg,
                                                     byte_index_reg,
                                                     {32'b0, total_bits_reg},
                                                     1'b0);
                            submit_first_reg <= first_block_reg;
                            if (byte_index_reg <= 6'd55)
                                after_core_reg <= AFTER_DONE;
                            else
                                after_core_reg <= AFTER_SECOND_PAD;
                            state_reg <= S_SUBMIT;
                        end
                    end

                    S_SUBMIT: begin
                        if (byte_valid || finish)
                            error_reg <= 1'b1;
                        if (core_ready) begin
                            first_block_reg <= 1'b0;
                            state_reg <= S_WAIT_CORE;
                        end
                    end

                    S_WAIT_CORE: begin
                        if (byte_valid || finish)
                            error_reg <= 1'b1;

                        if (core_digest_valid) begin
                            if (after_core_reg == AFTER_COLLECT) begin
                                block_reg <= 512'b0;
                                byte_index_reg <= 6'd0;
                                state_reg <= S_COLLECT;
                            end else if (after_core_reg == AFTER_SECOND_PAD) begin
                                block_reg <= final_block(512'b0,
                                                         6'd0,
                                                         {32'b0, total_bits_reg},
                                                         1'b1);
                                submit_first_reg <= 1'b0;
                                after_core_reg <= AFTER_DONE;
                                state_reg <= S_SUBMIT;
                            end else begin
                                digest_valid_reg <= 1'b1;
                                state_reg <= S_IDLE;
                            end
                        end
                    end

                    default: begin
                        state_reg <= S_IDLE;
                        error_reg <= 1'b1;
                    end
                endcase
            end
        end
    end

    function [511:0] block_with_byte;
        input [511:0] in_block;
        input [5:0]   index;
        input [7:0]   data;
        reg [511:0] tmp;
        integer shift;
        begin
            tmp = in_block;
            shift = (63 - index) * 8;
            tmp[shift +: 8] = data;
            block_with_byte = tmp;
        end
    endfunction

    function [511:0] final_block;
        input [511:0] partial_block;
        input [5:0]   next_byte_index;
        input [63:0]  bit_length;
        input         second_block;
        reg [511:0] tmp;
        integer shift;
        begin
            tmp = 512'b0;
            if (!second_block) begin
                tmp = partial_block;
                shift = (63 - next_byte_index) * 8;
                tmp[shift +: 8] = 8'h80;
            end
            tmp[63:32] = bit_length[63:32];
            tmp[31:0]  = bit_length[31:0];
            final_block = tmp;
        end
    endfunction

endmodule

`default_nettype wire
