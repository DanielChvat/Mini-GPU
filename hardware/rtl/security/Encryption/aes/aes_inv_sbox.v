`default_nettype none

// Stub: CTR mode only uses encryption, so the inverse S-box is never exercised.
// This satisfies elaboration for aes_decipher_block.
module aes_inv_sbox(
                    input wire [31 : 0]  sboxw,
                    output wire [31 : 0] new_sboxw
                   );

    assign new_sboxw = 32'h0;

endmodule

`default_nettype wire
