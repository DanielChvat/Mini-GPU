`timescale 1ns/1ps

module regfile #(
    parameter WIDTH = 32,
    parameter REG_COUNT = 16
) (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     write_enable,
    input  wire [3:0]               write_addr,
    input  wire [WIDTH-1:0]         write_data,
    input  wire [3:0]               read_addr_a,
    input  wire [3:0]               read_addr_b,
    output wire [WIDTH-1:0]         read_data_a,
    output wire [WIDTH-1:0]         read_data_b
);
    reg [WIDTH-1:0] regs [0:REG_COUNT-1];
    integer i;

    assign read_data_a = regs[read_addr_a];
    assign read_data_b = regs[read_addr_b];

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < REG_COUNT; i = i + 1) begin
                regs[i] <= {WIDTH{1'b0}};
            end
        end else if (write_enable) begin
            regs[write_addr] <= write_data;
        end
    end
endmodule
