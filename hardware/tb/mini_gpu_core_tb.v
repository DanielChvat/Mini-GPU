`timescale 1ns/1ps

`include "minigpu_isa.vh"

module mini_gpu_core_tb;
    localparam WIDTH = 32;
    localparam WARP_SIZE = 4;
    localparam PROG_ADDR_WIDTH = 5;
    localparam ADDR_WIDTH = 16;
    localparam CONST_ADDR_WIDTH = 8;

    reg clk;
    reg rst;
    reg prog_we;
    reg [PROG_ADDR_WIDTH-1:0] prog_addr;
    reg [31:0] prog_wdata;
    reg const_we;
    reg [CONST_ADDR_WIDTH-1:0] const_addr;
    reg [WIDTH-1:0] const_wdata;
    reg launch;
    reg [PROG_ADDR_WIDTH-1:0] base_pc;
    reg [WARP_SIZE-1:0] active_mask;
    reg [WIDTH-1:0] block_dim;
    reg [WIDTH-1:0] grid_dim;

    wire busy;
    wire done;
    wire error;
    wire unsupported;
    wire divide_by_zero;
    wire [PROG_ADDR_WIDTH-1:0] pc;
    wire [31:0] current_instr;
    wire [31:0] last_instr;
    wire [15:0] retired_count;
    wire [WARP_SIZE-1:0] last_writeback_mask;
    wire [3:0] last_writeback_addr;
    wire [(WARP_SIZE*WIDTH)-1:0] last_writeback_data;
    wire [WARP_SIZE-1:0] mem_req_valid;
    wire [WARP_SIZE-1:0] mem_req_write;
    wire [(WARP_SIZE*ADDR_WIDTH)-1:0] mem_req_addr;
    wire [(WARP_SIZE*WIDTH)-1:0] mem_req_wdata;
    wire [WARP_SIZE-1:0] mem_req_ready;
    wire [WARP_SIZE-1:0] mem_resp_valid;
    wire [(WARP_SIZE*WIDTH)-1:0] mem_resp_rdata;

    reg [WIDTH-1:0] lane_regs [0:WARP_SIZE-1][0:15];
    reg [31:0] program_image [0:(1 << PROG_ADDR_WIDTH)-1];
    reg [15:0] observed_retired;
    reg [1023:0] program_hex_path;
    reg allow_program_error;

    mini_gpu_core #(
        .WIDTH(WIDTH),
        .WARP_SIZE(WARP_SIZE),
        .PROG_ADDR_WIDTH(PROG_ADDR_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .CONST_ADDR_WIDTH(CONST_ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .prog_we(prog_we),
        .prog_addr(prog_addr),
        .prog_wdata(prog_wdata),
        .const_we(const_we),
        .const_addr(const_addr),
        .const_wdata(const_wdata),
        .launch(launch),
        .base_pc(base_pc),
        .active_mask(active_mask),
        .base_block_id(32'd0),
        .block_dim(block_dim),
        .grid_dim(grid_dim),
        .mem_req_valid(mem_req_valid),
        .mem_req_write(mem_req_write),
        .mem_req_addr(mem_req_addr),
        .mem_req_wdata(mem_req_wdata),
        .mem_req_ready(mem_req_ready),
        .mem_resp_valid(mem_resp_valid),
        .mem_resp_rdata(mem_resp_rdata),
        .busy(busy),
        .done(done),
        .error(error),
        .unsupported(unsupported),
        .divide_by_zero(divide_by_zero),
        .pc(pc),
        .current_instr(current_instr),
        .last_instr(last_instr),
        .retired_count(retired_count),
        .last_writeback_mask(last_writeback_mask),
        .last_writeback_addr(last_writeback_addr),
        .last_writeback_data(last_writeback_data)
    );

    memory #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(WIDTH),
        .BANK_DEPTH(64)
    ) global_memory (
        .clk(clk),
        .rst(rst),
        .req_valid(mem_req_valid),
        .req_write(mem_req_write),
        .req_addr(mem_req_addr),
        .req_wdata(mem_req_wdata),
        .req_ready(mem_req_ready),
        .resp_valid(mem_resp_valid),
        .resp_rdata(mem_resp_rdata)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task reset_core;
        integer lane;
        integer reg_id;
        begin
            @(negedge clk);
            rst = 1'b1;
            prog_we = 1'b0;
            const_we = 1'b0;
            const_addr = {CONST_ADDR_WIDTH{1'b0}};
            const_wdata = {WIDTH{1'b0}};
            launch = 1'b0;
            base_pc = {PROG_ADDR_WIDTH{1'b0}};
            active_mask = 4'b1111;
            block_dim = 32'd4;
            grid_dim = 32'd1;
            observed_retired = 16'b0;
            clear_program_image();
            for (lane = 0; lane < WARP_SIZE; lane = lane + 1) begin
                for (reg_id = 0; reg_id < 16; reg_id = reg_id + 1) begin
                    lane_regs[lane][reg_id] = {WIDTH{1'b0}};
                end
            end
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst = 1'b0;
        end
    endtask

    task load_instr;
        input [PROG_ADDR_WIDTH-1:0] addr;
        input [31:0] instr;
        begin
            @(negedge clk);
            prog_addr = addr;
            prog_wdata = instr;
            program_image[addr] = instr;
            prog_we = 1'b1;
            $display("  LOAD pc=%0d instr=0x%08h  %s", addr, instr, instr_name(instr[31:26]));
            @(negedge clk);
            prog_we = 1'b0;
        end
    endtask

    task load_const;
        input [CONST_ADDR_WIDTH-1:0] addr;
        input [WIDTH-1:0] data;
        begin
            @(negedge clk);
            const_addr = addr;
            const_wdata = data;
            const_we = 1'b1;
            $display("  CONST id=%0d data=0x%08h", addr, data);
            @(negedge clk);
            const_we = 1'b0;
        end
    endtask

    task clear_program_image;
        integer idx;
        begin
            for (idx = 0; idx < (1 << PROG_ADDR_WIDTH); idx = idx + 1) begin
                program_image[idx[PROG_ADDR_WIDTH-1:0]] = pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0);
            end
        end
    endtask

    task load_program_hex_file;
        input [1023:0] path;
        reg [31:0] file_words [0:(1 << PROG_ADDR_WIDTH)-1];
        integer idx;
        begin
            for (idx = 0; idx < (1 << PROG_ADDR_WIDTH); idx = idx + 1) begin
                file_words[idx[PROG_ADDR_WIDTH-1:0]] = pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0);
            end

            $display("  READ program hex file: %0s", path);
            $readmemh(path, file_words);

            for (idx = 0; idx < (1 << PROG_ADDR_WIDTH); idx = idx + 1) begin
                load_instr(idx[PROG_ADDR_WIDTH-1:0], file_words[idx[PROG_ADDR_WIDTH-1:0]]);
            end
        end
    endtask

    task launch_and_wait;
        input [WARP_SIZE-1:0] mask;
        integer cycles;
        begin
            $display("");
            $display("=== LAUNCH active_mask=%b base_pc=%0d block_dim=%0d grid_dim=%0d ===",
                     mask, {PROG_ADDR_WIDTH{1'b0}}, block_dim, grid_dim);
            print_program_listing();
            @(negedge clk);
            base_pc = {PROG_ADDR_WIDTH{1'b0}};
            active_mask = mask;
            launch = 1'b1;
            @(negedge clk);
            launch = 1'b0;
            cycles = 0;
            while (!done && cycles < 100) begin
                cycles = cycles + 1;
                @(negedge clk);
                sample_retire(mask);
            end
            #1;
            if (!done) begin
                $display("FAIL core did not finish within timeout");
                $finish;
            end
            $display("--- FINAL CORE STATUS ---");
            print_core_status();
            print_lane_summary(mask);
            $display("");
        end
    endtask

    task sample_retire;
        input [WARP_SIZE-1:0] mask;
        begin
            if (retired_count != observed_retired) begin
                observed_retired = retired_count;
                $display("");
                $display("  RETIRE #%0d pc=%0d instr=0x%08h %s",
                         retired_count,
                         pc,
                         last_instr,
                         instr_name(last_instr[31:26]));
                print_core_status();
                if (instr_writes_register(last_instr) && last_writeback_mask != 4'b0000) begin
                    apply_writeback();
                    print_writeback(mask);
                end else begin
                    $display("    writeback: none");
                end
                print_lane_summary(mask);
            end
        end
    endtask

    task apply_writeback;
        integer lane;
        begin
            for (lane = 0; lane < WARP_SIZE; lane = lane + 1) begin
                if (last_writeback_mask[lane]) begin
                    lane_regs[lane][last_writeback_addr] = last_writeback_data[(lane*WIDTH) +: WIDTH];
                end
            end
        end
    endtask

    task print_program_listing;
        integer idx;
        reg done_listing;
        begin
            $display("--- PROGRAM ---");
            done_listing = 1'b0;
            for (idx = 0; idx < (1 << PROG_ADDR_WIDTH); idx = idx + 1) begin
                if (!done_listing) begin
                    $display("  pc=%0d instr=0x%08h  %s rd=r%0d rs1=r%0d rs2=r%0d imm=%0d",
                             idx,
                             program_image[idx[PROG_ADDR_WIDTH-1:0]],
                             instr_name(program_image[idx[PROG_ADDR_WIDTH-1:0]][31:26]),
                             program_image[idx[PROG_ADDR_WIDTH-1:0]][25:22],
                             program_image[idx[PROG_ADDR_WIDTH-1:0]][21:18],
                             program_image[idx[PROG_ADDR_WIDTH-1:0]][17:14],
                             sign_extend_imm14(program_image[idx[PROG_ADDR_WIDTH-1:0]][13:0]));
                    if (program_image[idx[PROG_ADDR_WIDTH-1:0]][31:26] == `MGPU_OP_EXIT) begin
                        done_listing = 1'b1;
                    end
                end
            end
        end
    endtask

    task print_core_status;
        begin
            $display("    status: busy=%0d done=%0d error=%0d unsupported=%0d div0=%0d retired=%0d pc=%0d current=%s",
                     busy,
                     done,
                     error,
                     unsupported,
                     divide_by_zero,
                     retired_count,
                     pc,
                     instr_name(current_instr[31:26]));
        end
    endtask

    task print_writeback;
        input [WARP_SIZE-1:0] mask;
        integer lane;
        begin
            $display("    writeback: mask=%b rd=r%0d", last_writeback_mask, last_writeback_addr);
            for (lane = 0; lane < WARP_SIZE; lane = lane + 1) begin
                if (mask[lane]) begin
                    $display("      lane%0d active=%0d tid=%0d lid=%0d r%0d<=0x%08h (%0d)",
                             lane,
                             mask[lane],
                             lane,
                             lane,
                             last_writeback_addr,
                             last_writeback_data[(lane*WIDTH) +: WIDTH],
                             last_writeback_data[(lane*WIDTH) +: WIDTH]);
                end else begin
                    $display("      lane%0d active=0 tid=%0d lid=%0d no write", lane, lane, lane);
                end
            end
        end
    endtask

    task print_lane_summary;
        input [WARP_SIZE-1:0] mask;
        integer lane;
        begin
            $display("    lane registers:");
            for (lane = 0; lane < WARP_SIZE; lane = lane + 1) begin
                $display("      lane%0d active=%0d tid=%0d lid=%0d | r1=%0d r2=%0d r3=%0d r4=%0d r5=%0d r6=%0d r7=%0d",
                         lane,
                         mask[lane],
                         lane,
                         lane,
                         lane_regs[lane][1],
                         lane_regs[lane][2],
                         lane_regs[lane][3],
                         lane_regs[lane][4],
                         lane_regs[lane][5],
                         lane_regs[lane][6],
                         lane_regs[lane][7]);
            end
        end
    endtask

    task expect_lane_data;
        input integer lane;
        input [WIDTH-1:0] expected;
        begin
            if (last_writeback_data[(lane*WIDTH) +: WIDTH] !== expected) begin
                $display("FAIL lane %0d data got=0x%08h expected=0x%08h",
                         lane,
                         last_writeback_data[(lane*WIDTH) +: WIDTH],
                         expected);
                $finish;
            end
        end
    endtask

    task expect_lane_reg;
        input integer lane;
        input integer reg_id;
        input [WIDTH-1:0] expected;
        begin
            if (lane_regs[lane][reg_id] !== expected) begin
                $display("FAIL lane %0d r%0d got=0x%08h expected=0x%08h",
                         lane,
                         reg_id,
                         lane_regs[lane][reg_id],
                         expected);
                $finish;
            end
        end
    endtask

    task expect_all_lanes_reg;
        input integer reg_id;
        input [WIDTH-1:0] expected;
        integer lane;
        begin
            for (lane = 0; lane < WARP_SIZE; lane = lane + 1) begin
                expect_lane_reg(lane, reg_id, expected);
            end
        end
    endtask

    task seed_global_word;
        input [ADDR_WIDTH-1:0] addr;
        input [WIDTH-1:0] data;
        reg [ADDR_WIDTH-3:0] index;
        begin
            index = addr[ADDR_WIDTH-1:2];
            case (addr[1:0])
                2'd0: global_memory.bank0.mem[index] = data;
                2'd1: global_memory.bank1.mem[index] = data;
                2'd2: global_memory.bank2.mem[index] = data;
                default: global_memory.bank3.mem[index] = data;
            endcase
            $display("  SEED mem[%0d] = 0x%08h", addr, data);
        end
    endtask

    task expect_success;
        input [WARP_SIZE-1:0] expected_mask;
        input [3:0] expected_addr;
        input [WIDTH-1:0] expected_data;
        input [15:0] expected_retired;
        integer lane;
        begin
            if (error || unsupported || divide_by_zero) begin
                $display("FAIL unexpected status error=%0d unsupported=%0d div0=%0d",
                         error, unsupported, divide_by_zero);
                $finish;
            end
            if (last_writeback_mask !== expected_mask) begin
                $display("FAIL wb mask got=%b expected=%b", last_writeback_mask, expected_mask);
                $finish;
            end
            if (last_writeback_addr !== expected_addr) begin
                $display("FAIL wb addr got=%0d expected=%0d", last_writeback_addr, expected_addr);
                $finish;
            end
            if (retired_count !== expected_retired) begin
                $display("FAIL retired got=%0d expected=%0d", retired_count, expected_retired);
                $finish;
            end
            for (lane = 0; lane < WARP_SIZE; lane = lane + 1) begin
                if (expected_mask[lane]) begin
                    expect_lane_data(lane, expected_data);
                end
            end
        end
    endtask

    function [31:0] pack_instr;
        input [5:0] op;
        input [3:0] rd;
        input [3:0] rs1;
        input [3:0] rs2;
        input [13:0] imm14;
        begin
            pack_instr = {op, rd, rs1, rs2, imm14};
        end
    endfunction

    function signed [31:0] sign_extend_imm14;
        input [13:0] value;
        begin
            sign_extend_imm14 = {{18{value[13]}}, value};
        end
    endfunction

    function [8*8-1:0] instr_name;
        input [5:0] opcode;
        begin
            case (opcode)
                `MGPU_OP_NOP:   instr_name = "NOP";
                `MGPU_OP_MOV:   instr_name = "MOV";
                `MGPU_OP_MOVI:  instr_name = "MOVI";
                `MGPU_OP_LDC:   instr_name = "LDC";
                `MGPU_OP_ADD:   instr_name = "ADD";
                `MGPU_OP_ADDI:  instr_name = "ADDI";
                `MGPU_OP_SUB:   instr_name = "SUB";
                `MGPU_OP_SUBI:  instr_name = "SUBI";
                `MGPU_OP_MUL:   instr_name = "MUL";
                `MGPU_OP_MULI:  instr_name = "MULI";
                `MGPU_OP_DIV:   instr_name = "DIV";
                `MGPU_OP_MOD:   instr_name = "MOD";
                `MGPU_OP_AND:   instr_name = "AND";
                `MGPU_OP_ANDI:  instr_name = "ANDI";
                `MGPU_OP_OR:    instr_name = "OR";
                `MGPU_OP_ORI:   instr_name = "ORI";
                `MGPU_OP_XOR:   instr_name = "XOR";
                `MGPU_OP_XORI:  instr_name = "XORI";
                `MGPU_OP_NOT:   instr_name = "NOT";
                `MGPU_OP_SHL:   instr_name = "SHL";
                `MGPU_OP_SHLI:  instr_name = "SHLI";
                `MGPU_OP_SHR:   instr_name = "SHR";
                `MGPU_OP_SHRI:  instr_name = "SHRI";
                `MGPU_OP_SLT:   instr_name = "SLT";
                `MGPU_OP_SLE:   instr_name = "SLE";
                `MGPU_OP_SGT:   instr_name = "SGT";
                `MGPU_OP_SGE:   instr_name = "SGE";
                `MGPU_OP_SEQ:   instr_name = "SEQ";
                `MGPU_OP_SNE:   instr_name = "SNE";
                `MGPU_OP_FADD:  instr_name = "FADD";
                `MGPU_OP_FSUB:  instr_name = "FSUB";
                `MGPU_OP_FMUL:  instr_name = "FMUL";
                `MGPU_OP_LDG:   instr_name = "LDG";
                `MGPU_OP_STG:   instr_name = "STG";
                `MGPU_OP_LDS:   instr_name = "LDS";
                `MGPU_OP_STS:   instr_name = "STS";
                `MGPU_OP_FDIV:  instr_name = "FDIV";
                `MGPU_OP_TID:   instr_name = "TID";
                `MGPU_OP_TIDX:  instr_name = "TIDX";
                `MGPU_OP_BID:   instr_name = "BID";
                `MGPU_OP_BDIM:  instr_name = "BDIM";
                `MGPU_OP_GDIM:  instr_name = "GDIM";
                `MGPU_OP_LID:   instr_name = "LID";
                `MGPU_OP_WID:   instr_name = "WID";
                `MGPU_OP_PUSHM: instr_name = "PUSHM";
                `MGPU_OP_PRED:  instr_name = "PRED";
                `MGPU_OP_POPM:  instr_name = "POPM";
                `MGPU_OP_PREDN: instr_name = "PREDN";
                `MGPU_OP_BRA:   instr_name = "BRA";
                `MGPU_OP_BZ:    instr_name = "BZ";
                `MGPU_OP_BNZ:   instr_name = "BNZ";
                `MGPU_OP_BAR:   instr_name = "BAR";
                `MGPU_OP_EXIT:  instr_name = "EXIT";
                default:        instr_name = "UNKNOWN";
            endcase
        end
    endfunction

    function instr_writes_register;
        input [31:0] instr;
        begin
            case (instr[31:26])
                `MGPU_OP_MOV,
                `MGPU_OP_MOVI,
                `MGPU_OP_LDC,
                `MGPU_OP_ADD,
                `MGPU_OP_ADDI,
                `MGPU_OP_SUB,
                `MGPU_OP_SUBI,
                `MGPU_OP_MUL,
                `MGPU_OP_MULI,
                `MGPU_OP_DIV,
                `MGPU_OP_MOD,
                `MGPU_OP_AND,
                `MGPU_OP_ANDI,
                `MGPU_OP_OR,
                `MGPU_OP_ORI,
                `MGPU_OP_XOR,
                `MGPU_OP_XORI,
                `MGPU_OP_NOT,
                `MGPU_OP_SHL,
                `MGPU_OP_SHLI,
                `MGPU_OP_SHR,
                `MGPU_OP_SHRI,
                `MGPU_OP_SLT,
                `MGPU_OP_SLE,
                `MGPU_OP_SGT,
                `MGPU_OP_SGE,
                `MGPU_OP_SEQ,
                `MGPU_OP_SNE,
                `MGPU_OP_FADD,
                `MGPU_OP_FSUB,
                `MGPU_OP_FMUL,
                `MGPU_OP_FDIV,
                `MGPU_OP_LDG,
                `MGPU_OP_LDS,
                `MGPU_OP_TID,
                `MGPU_OP_TIDX,
                `MGPU_OP_BID,
                `MGPU_OP_BDIM,
                `MGPU_OP_GDIM,
                `MGPU_OP_LID,
                `MGPU_OP_WID: instr_writes_register = 1'b1;
                default: instr_writes_register = 1'b0;
            endcase
        end
    endfunction

    initial begin
        integer init_idx;
        rst = 1'b1;
        prog_we = 1'b0;
        prog_addr = {PROG_ADDR_WIDTH{1'b0}};
        prog_wdata = 32'b0;
        const_we = 1'b0;
        const_addr = {CONST_ADDR_WIDTH{1'b0}};
        const_wdata = {WIDTH{1'b0}};
        launch = 1'b0;
        base_pc = {PROG_ADDR_WIDTH{1'b0}};
        active_mask = 4'b1111;
        block_dim = 32'd4;
        grid_dim = 32'd1;
        observed_retired = 16'b0;
        program_hex_path = 1024'b0;
        allow_program_error = 1'b0;
        clear_program_image();

        $display("");
        $display("mini_gpu_core_tb: verbose SIMT trace enabled");

        if ($value$plusargs("program_hex=%s", program_hex_path) ||
            $value$plusargs("program=%s", program_hex_path)) begin
            allow_program_error = $test$plusargs("allow_error");
            reset_core();
            load_program_hex_file(program_hex_path);
            launch_and_wait(4'b1111);
            if ((error || unsupported || divide_by_zero) && !allow_program_error) begin
                $display("FAIL file program status error=%0d unsupported=%0d div0=%0d",
                         error, unsupported, divide_by_zero);
                $finish;
            end
            $display("mini_gpu_core_tb file program PASS");
            $finish;
        end

        reset_core();
        load_instr(4'd0, pack_instr(`MGPU_OP_MOVI, 4'd1, 4'd0, 4'd0, 14'd42));
        load_instr(4'd1, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        expect_success(4'b1111, 4'd1, 32'd42, 16'd2);

        reset_core();
        load_const(8'd7, 32'h0000cafe);
        load_instr(4'd0, pack_instr(`MGPU_OP_LDC, 4'd1, 4'd0, 4'd0, 14'd7));
        load_instr(4'd1, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        expect_success(4'b1111, 4'd1, 32'h0000cafe, 16'd2);

        reset_core();
        load_instr(4'd0, pack_instr(`MGPU_OP_MOVI, 4'd1, 4'd0, 4'd0, 14'd7));
        load_instr(4'd1, pack_instr(`MGPU_OP_MOVI, 4'd2, 4'd0, 4'd0, 14'd2));
        load_instr(4'd2, pack_instr(`MGPU_OP_ADD, 4'd3, 4'd1, 4'd2, 14'd0));
        load_instr(4'd3, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        expect_success(4'b1111, 4'd3, 32'd9, 16'd4);

        reset_core();
        load_instr(4'd0, pack_instr(`MGPU_OP_MOVI, 4'd4, 4'd0, 4'd0, 14'd5));
        load_instr(4'd1, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1011);
        expect_success(4'b1011, 4'd4, 32'd5, 16'd2);

        reset_core();
        load_instr(4'd0, pack_instr(`MGPU_OP_LID, 4'd5, 4'd0, 4'd0, 14'd0));
        load_instr(4'd1, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        if (error || unsupported || divide_by_zero || last_writeback_mask !== 4'b1111 ||
            last_writeback_addr !== 4'd5 || retired_count !== 16'd2) begin
            $display("FAIL LID status error=%0d unsupported=%0d div0=%0d mask=%b addr=%0d retired=%0d",
                     error,
                     unsupported,
                     divide_by_zero,
                     last_writeback_mask,
                     last_writeback_addr,
                     retired_count);
            $finish;
        end
        expect_lane_data(0, 32'd0);
        expect_lane_data(1, 32'd1);
        expect_lane_data(2, 32'd2);
        expect_lane_data(3, 32'd3);

        reset_core();
        load_instr(4'd0, pack_instr(`MGPU_OP_TID, 4'd6, 4'd0, 4'd0, 14'd0));
        load_instr(4'd1, pack_instr(`MGPU_OP_ADDI, 4'd7, 4'd6, 4'd0, 14'd10));
        load_instr(4'd2, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        if (error || unsupported || divide_by_zero || last_writeback_mask !== 4'b1111 ||
            last_writeback_addr !== 4'd7 || retired_count !== 16'd3) begin
            $display("FAIL TID/ADDI status error=%0d unsupported=%0d div0=%0d mask=%b addr=%0d retired=%0d",
                     error,
                     unsupported,
                     divide_by_zero,
                     last_writeback_mask,
                     last_writeback_addr,
                     retired_count);
            $finish;
        end
        expect_lane_data(0, 32'd10);
        expect_lane_data(1, 32'd11);
        expect_lane_data(2, 32'd12);
        expect_lane_data(3, 32'd13);

        reset_core();
        load_instr(4'd0, pack_instr(`MGPU_OP_MOVI, 4'd1, 4'd0, 4'd0, 14'd1000));
        load_instr(4'd1, pack_instr(`MGPU_OP_MOVI, 4'd2, 4'd0, 4'd0, 14'd234));
        load_instr(4'd2, pack_instr(`MGPU_OP_ADD, 4'd3, 4'd1, 4'd2, {11'b0, `MGPU_FMT_I16}));
        load_instr(4'd3, pack_instr(`MGPU_OP_MOVI, 4'd4, 4'd0, 4'd0, 14'd12));
        load_instr(4'd4, pack_instr(`MGPU_OP_MOVI, 4'd5, 4'd0, 4'd0, 14'h3ffd));
        load_instr(4'd5, pack_instr(`MGPU_OP_MUL, 4'd6, 4'd4, 4'd5, {11'b0, `MGPU_FMT_I8}));
        load_instr(4'd6, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        if (error || unsupported || divide_by_zero || last_writeback_mask !== 4'b1111 ||
            last_writeback_addr !== 4'd6 || retired_count !== 16'd7) begin
            $display("FAIL typed integer status error=%0d unsupported=%0d div0=%0d mask=%b addr=%0d retired=%0d",
                     error,
                     unsupported,
                     divide_by_zero,
                     last_writeback_mask,
                     last_writeback_addr,
                     retired_count);
            $finish;
        end
        expect_all_lanes_reg(3, 32'd1234);
        expect_all_lanes_reg(6, 32'hffffffdc);

        reset_core();
        load_instr(4'd0, pack_instr(`MGPU_OP_MOVI, 4'd1, 4'd0, 4'd0, 14'h0038));
        load_instr(4'd1, pack_instr(`MGPU_OP_MOVI, 4'd2, 4'd0, 4'd0, 14'h0040));
        load_instr(4'd2, pack_instr(`MGPU_OP_FADD, 4'd3, 4'd1, 4'd2, {11'b0, `MGPU_FMT_FP8}));
        load_instr(4'd3, pack_instr(`MGPU_OP_FMUL, 4'd4, 4'd1, 4'd2, {11'b0, `MGPU_FMT_FP8}));
        load_instr(4'd4, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        if (error || unsupported || divide_by_zero || last_writeback_mask !== 4'b1111 ||
            last_writeback_addr !== 4'd4 || retired_count !== 16'd5) begin
            $display("FAIL fp8 status error=%0d unsupported=%0d div0=%0d mask=%b addr=%0d retired=%0d",
                     error,
                     unsupported,
                     divide_by_zero,
                     last_writeback_mask,
                     last_writeback_addr,
                     retired_count);
            $finish;
        end
        expect_all_lanes_reg(3, 32'h00000044);
        expect_all_lanes_reg(4, 32'h00000040);

        reset_core();
        seed_global_word(16'd20, 32'h00003c00);
        seed_global_word(16'd21, 32'h00004000);
        load_instr(4'd0, pack_instr(`MGPU_OP_MOVI, 4'd1, 4'd0, 4'd0, 14'd20));
        load_instr(4'd1, pack_instr(`MGPU_OP_LDG, 4'd2, 4'd1, 4'd0, 14'd0));
        load_instr(4'd2, pack_instr(`MGPU_OP_LDG, 4'd3, 4'd1, 4'd0, 14'd1));
        load_instr(4'd3, pack_instr(`MGPU_OP_FADD, 4'd4, 4'd2, 4'd3, {11'b0, `MGPU_FMT_FP16}));
        load_instr(4'd4, pack_instr(`MGPU_OP_FMUL, 4'd5, 4'd2, 4'd3, {11'b0, `MGPU_FMT_FP16}));
        load_instr(4'd5, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        if (error || unsupported || divide_by_zero || last_writeback_mask !== 4'b1111 ||
            last_writeback_addr !== 4'd5 || retired_count !== 16'd6) begin
            $display("FAIL fp16 status error=%0d unsupported=%0d div0=%0d mask=%b addr=%0d retired=%0d",
                     error,
                     unsupported,
                     divide_by_zero,
                     last_writeback_mask,
                     last_writeback_addr,
                     retired_count);
            $finish;
        end
        expect_all_lanes_reg(4, 32'h00004200);
        expect_all_lanes_reg(5, 32'h00004000);

        reset_core();
        seed_global_word(16'd24, 32'h3fc00000);
        seed_global_word(16'd25, 32'h40100000);
        load_instr(4'd0, pack_instr(`MGPU_OP_MOVI, 4'd1, 4'd0, 4'd0, 14'd24));
        load_instr(4'd1, pack_instr(`MGPU_OP_LDG, 4'd2, 4'd1, 4'd0, 14'd0));
        load_instr(4'd2, pack_instr(`MGPU_OP_LDG, 4'd3, 4'd1, 4'd0, 14'd1));
        load_instr(4'd3, pack_instr(`MGPU_OP_FADD, 4'd4, 4'd2, 4'd3, {11'b0, `MGPU_FMT_FP32}));
        load_instr(4'd4, pack_instr(`MGPU_OP_FMUL, 4'd5, 4'd2, 4'd3, {11'b0, `MGPU_FMT_FP32}));
        load_instr(4'd5, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        if (error || unsupported || divide_by_zero || last_writeback_mask !== 4'b1111 ||
            last_writeback_addr !== 4'd5 || retired_count !== 16'd6) begin
            $display("FAIL fp32 status error=%0d unsupported=%0d div0=%0d mask=%b addr=%0d retired=%0d",
                     error,
                     unsupported,
                     divide_by_zero,
                     last_writeback_mask,
                     last_writeback_addr,
                     retired_count);
            $finish;
        end
        expect_all_lanes_reg(4, 32'h40700000);
        expect_all_lanes_reg(5, 32'h40580000);

        reset_core();
        load_instr(4'd0, pack_instr(`MGPU_OP_TID, 4'd1, 4'd0, 4'd0, 14'd0));
        load_instr(4'd1, pack_instr(`MGPU_OP_ADDI, 4'd2, 4'd1, 4'd0, 14'd20));
        load_instr(4'd2, pack_instr(`MGPU_OP_STG, 4'd0, 4'd1, 4'd2, 14'd0));
        load_instr(4'd3, pack_instr(`MGPU_OP_LDG, 4'd3, 4'd1, 4'd0, 14'd0));
        load_instr(4'd4, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        if (error || unsupported || divide_by_zero || last_writeback_mask !== 4'b1111 ||
            last_writeback_addr !== 4'd3 || retired_count !== 16'd5) begin
            $display("FAIL memory status error=%0d unsupported=%0d div0=%0d mask=%b addr=%0d retired=%0d",
                     error,
                     unsupported,
                     divide_by_zero,
                     last_writeback_mask,
                     last_writeback_addr,
                     retired_count);
            $finish;
        end
        expect_lane_data(0, 32'd20);
        expect_lane_data(1, 32'd21);
        expect_lane_data(2, 32'd22);
        expect_lane_data(3, 32'd23);

        reset_core();
        load_instr(4'd0, pack_instr(`MGPU_OP_TID, 4'd1, 4'd0, 4'd0, 14'd0));
        load_instr(4'd1, pack_instr(`MGPU_OP_SHLI, 4'd1, 4'd1, 4'd0, 14'd2));
        load_instr(4'd2, pack_instr(`MGPU_OP_ADDI, 4'd2, 4'd1, 4'd0, 14'd30));
        load_instr(4'd3, pack_instr(`MGPU_OP_STG, 4'd0, 4'd1, 4'd2, 14'd0));
        load_instr(4'd4, pack_instr(`MGPU_OP_LDG, 4'd3, 4'd1, 4'd0, 14'd0));
        load_instr(4'd5, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        if (error || unsupported || divide_by_zero || last_writeback_mask !== 4'b1111 ||
            last_writeback_addr !== 4'd3 || retired_count !== 16'd6) begin
            $display("FAIL conflict memory status error=%0d unsupported=%0d div0=%0d mask=%b addr=%0d retired=%0d",
                     error,
                     unsupported,
                     divide_by_zero,
                     last_writeback_mask,
                     last_writeback_addr,
                     retired_count);
            $finish;
        end
        expect_lane_data(0, 32'd30);
        expect_lane_data(1, 32'd34);
        expect_lane_data(2, 32'd38);
        expect_lane_data(3, 32'd42);

        reset_core();
        load_instr(4'd0, pack_instr(`MGPU_OP_TID, 4'd1, 4'd0, 4'd0, 14'd0));
        load_instr(4'd1, pack_instr(`MGPU_OP_MOVI, 4'd2, 4'd0, 4'd0, 14'd2));
        load_instr(4'd2, pack_instr(`MGPU_OP_SLT, 4'd3, 4'd1, 4'd2, {11'b0, `MGPU_FMT_I32}));
        load_instr(4'd3, pack_instr(`MGPU_OP_PUSHM, 4'd0, 4'd0, 4'd0, 14'd0));
        load_instr(4'd4, pack_instr(`MGPU_OP_PRED, 4'd0, 4'd3, 4'd0, 14'd0));
        load_instr(4'd5, pack_instr(`MGPU_OP_MOVI, 4'd4, 4'd0, 4'd0, 14'd99));
        load_instr(4'd6, pack_instr(`MGPU_OP_POPM, 4'd0, 4'd0, 4'd0, 14'd0));
        load_instr(4'd7, pack_instr(`MGPU_OP_PUSHM, 4'd0, 4'd0, 4'd0, 14'd0));
        load_instr(4'd8, pack_instr(`MGPU_OP_PREDN, 4'd0, 4'd3, 4'd0, 14'd0));
        load_instr(4'd9, pack_instr(`MGPU_OP_MOVI, 4'd5, 4'd0, 4'd0, 14'd77));
        load_instr(4'd10, pack_instr(`MGPU_OP_POPM, 4'd0, 4'd0, 4'd0, 14'd0));
        load_instr(4'd11, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        if (error || unsupported || divide_by_zero || retired_count !== 16'd12) begin
            $display("FAIL predicate mask status error=%0d unsupported=%0d div0=%0d retired=%0d",
                     error, unsupported, divide_by_zero, retired_count);
            $finish;
        end
        expect_lane_reg(0, 4, 32'd99);
        expect_lane_reg(1, 4, 32'd99);
        expect_lane_reg(2, 4, 32'd0);
        expect_lane_reg(3, 4, 32'd0);
        expect_lane_reg(0, 5, 32'd0);
        expect_lane_reg(1, 5, 32'd0);
        expect_lane_reg(2, 5, 32'd77);
        expect_lane_reg(3, 5, 32'd77);

        reset_core();
        load_instr(4'd0, pack_instr(`MGPU_OP_MOVI, 4'd1, 4'd0, 4'd0, 14'd0));
        load_instr(4'd1, pack_instr(`MGPU_OP_BZ, 4'd0, 4'd1, 4'd0, 14'd1));
        load_instr(4'd2, pack_instr(`MGPU_OP_MOVI, 4'd2, 4'd0, 4'd0, 14'd99));
        load_instr(4'd3, pack_instr(`MGPU_OP_MOVI, 4'd2, 4'd0, 4'd0, 14'd7));
        load_instr(4'd4, pack_instr(`MGPU_OP_BRA, 4'd0, 4'd0, 4'd0, 14'd1));
        load_instr(4'd5, pack_instr(`MGPU_OP_MOVI, 4'd3, 4'd0, 4'd0, 14'd99));
        load_instr(4'd6, pack_instr(`MGPU_OP_MOVI, 4'd3, 4'd0, 4'd0, 14'd8));
        load_instr(4'd7, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        if (error || unsupported || divide_by_zero || retired_count !== 16'd6) begin
            $display("FAIL branch status error=%0d unsupported=%0d div0=%0d retired=%0d",
                     error, unsupported, divide_by_zero, retired_count);
            $finish;
        end
        expect_all_lanes_reg(2, 32'd7);
        expect_all_lanes_reg(3, 32'd8);

        reset_core();
        load_instr(4'd0, pack_instr(`MGPU_OP_BAR, 4'd0, 4'd0, 4'd0, 14'd0));
        load_instr(4'd1, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));
        launch_and_wait(4'b1111);
        if (error || unsupported || divide_by_zero || retired_count !== 16'd2 ||
            last_writeback_mask !== 4'b0000) begin
            $display("FAIL BAR status error=%0d unsupported=%0d div0=%0d retired=%0d wb_mask=%b",
                     error, unsupported, divide_by_zero, retired_count, last_writeback_mask);
            $finish;
        end

        $display("mini_gpu_core_tb PASS");
        $finish;
    end
endmodule
