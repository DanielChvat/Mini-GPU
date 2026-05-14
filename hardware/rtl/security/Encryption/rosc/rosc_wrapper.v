`timescale 1ns/1ps
`default_nettype none

module rosc_wrapper (
    input  wire        clk,
    input  wire        rst,
    input  wire        enable,
    output wire [31:0] entropy_data,
    output wire        entropy_valid,
    input  wire        entropy_ack
);

    wire reset_n = ~rst;
    wire [31:0] rosc_en_internal = {32{enable}};
    wire core_entropy_valid;
    wire ack_i = entropy_ack & enable;

    localparam [31:0] ROSC_OPA = 32'haaaaaaaa;
    localparam [31:0] ROSC_OPB = 32'h55555555;

    assign entropy_valid = core_entropy_valid & enable;

    rosc_entropy_core core (
        .clk          (clk),
        .reset_n      (reset_n),
        .opa          (ROSC_OPA),
        .opb          (ROSC_OPB),
        .rosc_en      (rosc_en_internal),
        .rosc_cycles  (8'hFF),
        .raw_entropy  (),
        .rosc_outputs (),
        .entropy_data (entropy_data),
        .entropy_valid(core_entropy_valid),
        .entropy_ack  (ack_i),
        .debug        (),
        .debug_update (1'b0)
    );

endmodule

`default_nettype wire
