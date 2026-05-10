`timescale 1ns/1ps

module memory_bank #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 14,
    parameter DEPTH = 8192
) (
    input  wire                  clk,
    input  wire                  en_a,
    input  wire                  we_a,
    input  wire [3:0]            wem_a,
    input  wire [ADDR_WIDTH-1:0] addr_a,
    input  wire [DATA_WIDTH-1:0] din_a,
    output reg  [DATA_WIDTH-1:0] dout_a,
    input  wire                  en_b,
    input  wire                  we_b,
    input  wire [3:0]            wem_b,
    input  wire [ADDR_WIDTH-1:0] addr_b,
    input  wire [DATA_WIDTH-1:0] din_b,
    output reg  [DATA_WIDTH-1:0] dout_b
);
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (en_a) begin
            if (we_a) begin
                if (wem_a[0]) mem[addr_a][7:0] <= din_a[7:0];
                if (wem_a[1]) mem[addr_a][15:8] <= din_a[15:8];
                if (wem_a[2]) mem[addr_a][23:16] <= din_a[23:16];
                if (wem_a[3]) mem[addr_a][31:24] <= din_a[31:24];
            end
            dout_a <= mem[addr_a];
        end
    end

    always @(posedge clk) begin
        if (en_b) begin
            if (we_b) begin
                if (wem_b[0]) mem[addr_b][7:0] <= din_b[7:0];
                if (wem_b[1]) mem[addr_b][15:8] <= din_b[15:8];
                if (wem_b[2]) mem[addr_b][23:16] <= din_b[23:16];
                if (wem_b[3]) mem[addr_b][31:24] <= din_b[31:24];
            end
            dout_b <= mem[addr_b];
        end
    end
endmodule
