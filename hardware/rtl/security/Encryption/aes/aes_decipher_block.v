`default_nettype none

// Stub: AES-CTR mode only uses AES encryption.
// This module exists only to satisfy aes_core elaboration.
// It must never be exercised in CTR-only builds.
module aes_decipher_block(
    input  wire           clk,
    input  wire           reset_n,

    input  wire           next,

    input  wire           keylen,
    output wire [3 : 0]   round,
    input  wire [127 : 0] round_key,

    input  wire [127 : 0] block,
    output wire [127 : 0] new_block,
    output wire           ready
);

    assign round     = 4'h0;
    assign new_block = 128'h0;
    assign ready     = 1'b1;

endmodule

`default_nettype wire