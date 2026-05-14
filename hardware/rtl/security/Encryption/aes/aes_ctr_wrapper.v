`timescale 1ns/1ps
`default_nettype none

module aes_ctr_wrapper (
    input  wire          clk,
    input  wire          rst,
    input  wire          enable,
    input  wire [127:0]  key,
    input  wire [95:0]   nonce,
    input  wire          seed,
    input  wire [127:0]  data_in,
    input  wire          load,
    output wire [127:0]  data_out,
    output wire          data_valid,
    input  wire          data_ack
);

    wire reset_n = ~rst;

    localparam KEYLEN = 1'b0;
    localparam ENCDEC = 1'b1;

    localparam [1:0] W_IDLE  = 2'd0;
    localparam [1:0] W_BUSY  = 2'd1;
    localparam [1:0] W_VALID = 2'd2;

    reg [1:0]  state_reg, state_next;
    reg        key_ready_reg;
    reg        key_ready_set;
    reg [31:0] ctr_reg;
    reg        ctr_inc;

    reg core_init;
    reg core_next;

    wire        core_ready;
    wire [127:0] core_result;
    wire        core_result_valid;

    always @(posedge clk) begin
        if (rst) begin
            state_reg     <= W_IDLE;
            key_ready_reg <= 1'b0;
            ctr_reg       <= 32'd0;
        end else begin
            state_reg <= state_next;
            if (seed) begin
                key_ready_reg <= 1'b0;
                ctr_reg       <= 32'd0;
            end else begin
                if (key_ready_set)
                    key_ready_reg <= 1'b1;
                if (ctr_inc)
                    ctr_reg <= ctr_reg + 32'd1;
            end
        end
    end

    always @* begin
        state_next    = state_reg;
        core_init     = 1'b0;
        core_next     = 1'b0;
        key_ready_set = 1'b0;
        ctr_inc       = 1'b0;

        case (state_reg)
            W_IDLE: begin
                if (seed) begin
                    core_init  = 1'b1;
                    state_next = W_BUSY;
                end else if (load && key_ready_reg) begin
                    core_next  = 1'b1;
                    state_next = W_BUSY;
                end
            end

            W_BUSY: begin
                if (core_result_valid) begin
                    state_next = W_VALID;
                    ctr_inc    = 1'b1;
                end else if (core_ready && !core_result_valid) begin
                    state_next    = W_IDLE;
                    key_ready_set = 1'b1;
                end
            end

            W_VALID: begin
                if (seed) begin
                    core_init  = 1'b1;
                    state_next = W_BUSY;
                end else if (data_ack && enable) begin
                    state_next = W_IDLE;
                end
            end

            default: state_next = W_IDLE;
        endcase
    end

    assign data_valid = (state_reg == W_VALID) && enable;
    assign data_out   = core_result ^ data_in;

    aes_core core (
        .clk          (clk),
        .reset_n      (reset_n),
        .encdec       (ENCDEC),
        .init         (core_init),
        .next         (core_next),
        .ready        (core_ready),
        .key          ({key, 128'b0}),
        .keylen       (KEYLEN),
        .block        ({nonce, ctr_reg}),
        .result       (core_result),
        .result_valid (core_result_valid)
    );

endmodule

`default_nettype wire
