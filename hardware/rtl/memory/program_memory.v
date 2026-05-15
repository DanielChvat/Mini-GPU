`timescale 1ns/1ps

module program_memory #(
    parameter ADDR_WIDTH = 12,
    parameter DATA_WIDTH = 32
) (
    input  wire                    clk,

    input  wire                    write_en,
    input  wire [ADDR_WIDTH-1:0]   write_addr,
    input  wire [DATA_WIDTH-1:0]   write_data,

    input  wire [ADDR_WIDTH-1:0]   read_addr,
    output reg  [DATA_WIDTH-1:0]   read_data
);
    localparam DEPTH = (1 << ADDR_WIDTH);

    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (write_en) begin
            mem[write_addr] <= write_data;
        end

        read_data <= mem[read_addr];
    end
endmodule
