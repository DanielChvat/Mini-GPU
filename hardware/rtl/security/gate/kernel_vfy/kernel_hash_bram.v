`timescale 1ns/1ps
`default_nettype none

module kernel_hash_bram #(
    parameter NUM_HASHES = 64,
    parameter ADDR_WIDTH = 6,
    parameter HASH_MEM_FILE = "kernel_golden_hashes.mem"
) (
    input  wire                  clk,
    input  wire [ADDR_WIDTH-1:0] addr,
    output reg  [255:0]          hash_out
);

    (* ram_style = "block" *) reg [255:0] hash_mem [0:NUM_HASHES-1];

    initial begin
        $readmemh(HASH_MEM_FILE, hash_mem);
    end

    always @(posedge clk) begin
        hash_out <= hash_mem[addr];
    end

endmodule

`default_nettype wire
