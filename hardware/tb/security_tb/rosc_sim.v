module rosc #(parameter WIDTH = 2)
(
    input wire                   clk,
    input wire                   reset_n,

    input wire                   we,

    input wire [(WIDTH - 1) : 0] opa,
    input wire [(WIDTH - 1) : 0] opb,

    output reg                  dout
);

    initial begin
        dout = 1'b0;
    end
    


    always @ (posedge clk or negedge reset_n)
    begin
        if (!reset_n)
        begin
            dout <= 1'b0;
        end
        else
        begin
            if (we)
            begin
                dout <= $random();
                // $display("Dout: %b", dout);
            end
        end
    end

endmodule