`timescale 1ns/1ps

module instruction_decode_tb;
    reg  [31:0] instr;
    wire [5:0]  opcode;
    wire [3:0]  rd;
    wire [3:0]  rs1;
    wire [3:0]  rs2;
    wire [13:0] imm14;
    wire [2:0]  fmt;
    wire [31:0] imm_sext;

    instruction_decode dut (
        .instr(instr),
        .opcode(opcode),
        .rd(rd),
        .rs1(rs1),
        .rs2(rs2),
        .imm14(imm14),
        .fmt(fmt),
        .imm_sext(imm_sext)
    );

    initial begin
        instr = {6'h04, 4'h1, 4'h2, 4'h3, 14'h0000};
        #1;
        if (opcode !== 6'h04 || rd !== 4'h1 || rs1 !== 4'h2 || rs2 !== 4'h3 || imm14 !== 14'h0000) begin
            $display("FAIL decode register form");
            $finish;
        end

        instr = {6'h04, 4'h1, 4'h2, 4'h3, 11'h000, 3'h2};
        #1;
        if (opcode !== 6'h04 || fmt !== 3'h2) begin
            $display("FAIL decode typed ALU format");
            $finish;
        end

        instr = {6'h02, 4'hf, 4'h0, 4'h0, 14'h3fff};
        #1;
        if (opcode !== 6'h02 || rd !== 4'hf || imm_sext !== 32'hffffffff) begin
            $display("FAIL decode immediate sign extension");
            $finish;
        end

        instr = {6'h39, 4'h0, 4'h7, 4'h0, 14'h2000};
        #1;
        if (opcode !== 6'h39 || rs1 !== 4'h7 || imm_sext !== 32'hffffe000) begin
            $display("FAIL decode branch offset sign extension");
            $finish;
        end

        $display("instruction_decode_tb PASS");
        $finish;
    end
endmodule
