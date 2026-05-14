`timescale 1ns/1ps
`default_nettype none

module chacha_wrapper (
    input  wire          clk,
    input  wire          rst,
    input  wire          enable,
    input  wire [255:0]  key,
    input  wire          seed,
    output wire [511:0]  data_out,
    output wire          data_valid,
    input  wire          data_ack
);

    wire reset_n = ~rst;

    localparam         KEYLEN  = 1'b1;
    localparam [4:0]   ROUNDS  = 5'h14;
    localparam [63:0]  IV      = 64'h0;
    localparam [63:0]  CTR     = 64'h0;
    localparam [511:0] DATA_IN = 512'h0;

    localparam [1:0] W_IDLE  = 2'd0;
    localparam [1:0] W_BUSY  = 2'd1;
    localparam [1:0] W_VALID = 2'd2;

    reg [1:0] state_reg, state_next;
    reg       core_init;
    reg       core_next;

    wire        core_ready;
    wire [511:0] core_data_out;
    wire        core_data_out_valid;

    always @(posedge clk) begin
        if (rst)
            state_reg <= W_IDLE;
        else
            state_reg <= state_next;
    end

    always @* begin
        state_next = state_reg;
        core_init  = 1'b0;
        core_next  = 1'b0;

        case (state_reg)
            W_IDLE: begin
                if (seed) begin
                    core_init  = 1'b1;
                    state_next = W_BUSY;
                end
            end

            W_BUSY: begin
                if (core_ready && core_data_out_valid)
                    state_next = W_VALID;
            end

            W_VALID: begin
                if (seed) begin
                    core_init  = 1'b1;
                    state_next = W_BUSY;
                end else if (data_ack && enable) begin
                    core_next  = 1'b1;
                    state_next = W_BUSY;
                end
            end

            default: state_next = W_IDLE;
        endcase
    end

    assign data_valid = (state_reg == W_VALID) && enable;
    assign data_out   = core_data_out;

    chacha_core core (
        .clk            (clk),
        .reset_n        (reset_n),
        .init           (core_init),
        .next           (core_next),
        .key            (key),
        .keylen         (KEYLEN),
        .iv             (IV),
        .ctr            (CTR),
        .rounds         (ROUNDS),
        .data_in        (DATA_IN),
        .ready          (core_ready),
        .data_out       (core_data_out),
        .data_out_valid (core_data_out_valid)
    );

endmodule

`default_nettype wire
