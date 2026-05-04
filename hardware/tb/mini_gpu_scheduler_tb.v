`timescale 1ns/1ps

`include "minigpu_isa.vh"

module mini_gpu_scheduler_tb;
    localparam WIDTH = 32;
    localparam WARP_SIZE = 4;
    localparam NUM_WARPS = 2;
    localparam NUM_BLOCKS = 2;
    localparam WARP_ID_WIDTH = 1;
    localparam BLOCK_ID_WIDTH = 1;
    localparam PROG_ADDR_WIDTH = 5;
    localparam ADDR_WIDTH = 16;
    localparam CONST_ADDR_WIDTH = 8;

    reg clk;
    reg rst;
    reg prog_we;
    reg [PROG_ADDR_WIDTH-1:0] prog_addr;
    reg [31:0] prog_wdata;
    reg launch;

    wire busy;
    wire done;
    wire error;
    wire unsupported;
    wire divide_by_zero;
    wire [PROG_ADDR_WIDTH-1:0] pc;
    wire [31:0] current_instr;
    wire [31:0] last_instr;
    wire [15:0] retired_count;
    wire [WARP_SIZE-1:0] mem_req_valid;
    wire [WARP_SIZE-1:0] mem_req_write;
    wire [(WARP_SIZE*ADDR_WIDTH)-1:0] mem_req_addr;
    wire [(WARP_SIZE*WIDTH)-1:0] mem_req_wdata;
    wire [WARP_SIZE-1:0] last_writeback_mask;
    wire [3:0] last_writeback_addr;
    wire [(WARP_SIZE*WIDTH)-1:0] last_writeback_data;

    reg [15:0] observed_retired;
    reg [31:0] program_image [0:(1 << PROG_ADDR_WIDTH)-1];

    mini_gpu_core #(
        .WIDTH(WIDTH),
        .WARP_SIZE(WARP_SIZE),
        .PROG_ADDR_WIDTH(PROG_ADDR_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .CONST_ADDR_WIDTH(CONST_ADDR_WIDTH),
        .NUM_WARPS(NUM_WARPS),
        .NUM_BLOCKS(NUM_BLOCKS),
        .WARP_ID_WIDTH(WARP_ID_WIDTH),
        .BLOCK_ID_WIDTH(BLOCK_ID_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .prog_we(prog_we),
        .prog_addr(prog_addr),
        .prog_wdata(prog_wdata),
        .const_we(1'b0),
        .const_addr({CONST_ADDR_WIDTH{1'b0}}),
        .const_wdata({WIDTH{1'b0}}),
        .launch(launch),
        .base_pc({PROG_ADDR_WIDTH{1'b0}}),
        .active_mask({WARP_SIZE{1'b1}}),
        .base_block_id(32'd0),
        .block_dim(32'd8),
        .grid_dim(32'd2),
        .mem_req_valid(mem_req_valid),
        .mem_req_write(mem_req_write),
        .mem_req_addr(mem_req_addr),
        .mem_req_wdata(mem_req_wdata),
        .mem_req_ready({WARP_SIZE{1'b0}}),
        .mem_resp_valid({WARP_SIZE{1'b0}}),
        .mem_resp_rdata({(WARP_SIZE*WIDTH){1'b0}}),
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

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task load_instr;
        input [PROG_ADDR_WIDTH-1:0] addr;
        input [31:0] instr;
        begin
            @(negedge clk);
            prog_addr = addr;
            prog_wdata = instr;
            program_image[addr] = instr;
            prog_we = 1'b1;
            @(negedge clk);
            prog_we = 1'b0;
        end
    endtask

    task sample_retire;
        integer global_warp_id;
        integer block_id;
        integer warp_id;
        begin
            if (retired_count != observed_retired) begin
                observed_retired = retired_count;
                global_warp_id = dut.selected_idx;
                block_id = global_warp_id / NUM_WARPS;
                warp_id = global_warp_id % NUM_WARPS;

                $display("");
                $display("  RETIRE #%0d block=%0d warp=%0d pc=%0d instr=0x%08h %s",
                         retired_count,
                         block_id,
                         warp_id,
                         pc,
                         last_instr,
                         instr_name(last_instr[31:26]));
                print_core_status(global_warp_id);

                if (instr_writes_register(last_instr) &&
                    last_writeback_mask != {WARP_SIZE{1'b0}}) begin
                    print_writeback(block_id, warp_id, global_warp_id);
                end else begin
                    $display("    writeback: none");
                end
                print_lane_summary(block_id, warp_id, global_warp_id);
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
        input integer global_warp_id;
        begin
            $display("    status: busy=%0d done=%0d error=%0d unsupported=%0d div0=%0d retired=%0d pc=%0d current=%s active=%b waiting=%0d warp_done=%0d",
                     busy,
                     done,
                     error,
                     unsupported,
                     divide_by_zero,
                     retired_count,
                     pc,
                     instr_name(current_instr[31:26]),
                     dut.active_mask_r[global_warp_id],
                     dut.barrier_waiting[global_warp_id],
                     dut.warp_done_r[global_warp_id]);
        end
    endtask

    task print_writeback;
        input integer block_id;
        input integer warp_id;
        input integer global_warp_id;
        integer lane_id;
        integer tid;
        begin
            $display("    writeback: mask=%b rd=r%0d", last_writeback_mask, last_writeback_addr);
            for (lane_id = 0; lane_id < WARP_SIZE; lane_id = lane_id + 1) begin
                tid = (block_id * NUM_WARPS * WARP_SIZE) + (warp_id * WARP_SIZE) + lane_id;
                if (last_writeback_mask[lane_id]) begin
                    $display("      block%0d warp%0d lane%0d active=%0d tid=%0d lid=%0d r%0d<=0x%08h (%0d)",
                             block_id,
                             warp_id,
                             lane_id,
                             dut.active_mask_r[global_warp_id][lane_id],
                             tid,
                             lane_id,
                             last_writeback_addr,
                             last_writeback_data[(lane_id*WIDTH) +: WIDTH],
                             last_writeback_data[(lane_id*WIDTH) +: WIDTH]);
                end else begin
                    $display("      block%0d warp%0d lane%0d active=%0d tid=%0d lid=%0d no write",
                             block_id,
                             warp_id,
                             lane_id,
                             dut.active_mask_r[global_warp_id][lane_id],
                             tid,
                             lane_id);
                end
            end
        end
    endtask

    task print_lane_summary;
        input integer block_id;
        input integer warp_id;
        input integer global_warp_id;
        integer lane_id;
        integer tid;
        begin
            $display("    lane registers:");
            for (lane_id = 0; lane_id < WARP_SIZE; lane_id = lane_id + 1) begin
                tid = (block_id * NUM_WARPS * WARP_SIZE) + (warp_id * WARP_SIZE) + lane_id;
                $display("      block%0d warp%0d lane%0d active=%0d tid=%0d lid=%0d | r1=%0d r2=%0d r3=%0d r4=%0d",
                         block_id,
                         warp_id,
                         lane_id,
                         dut.active_mask_r[global_warp_id][lane_id],
                         tid,
                         lane_id,
                         lane_reg(block_id, warp_id, lane_id, 1),
                         lane_reg(block_id, warp_id, lane_id, 2),
                         lane_reg(block_id, warp_id, lane_id, 3),
                         lane_reg(block_id, warp_id, lane_id, 4));
            end
        end
    endtask

    task check_lane;
        input integer block_id;
        input integer warp_id;
        input integer lane_id;
        input [31:0] expected_tid;
        input [31:0] expected_wid;
        input [31:0] expected_bid;
        begin
            if (lane_reg(block_id, warp_id, lane_id, 1) !== expected_tid) begin
                $display("FAIL block=%0d warp=%0d lane=%0d r1/tid got=%0d expected=%0d",
                         block_id, warp_id, lane_id,
                         lane_reg(block_id, warp_id, lane_id, 1), expected_tid);
                $finish;
            end
            if (lane_reg(block_id, warp_id, lane_id, 2) !== expected_wid) begin
                $display("FAIL block=%0d warp=%0d lane=%0d r2/wid got=%0d expected=%0d",
                         block_id, warp_id, lane_id,
                         lane_reg(block_id, warp_id, lane_id, 2), expected_wid);
                $finish;
            end
            if (lane_reg(block_id, warp_id, lane_id, 3) !== expected_bid) begin
                $display("FAIL block=%0d warp=%0d lane=%0d r3/bid got=%0d expected=%0d",
                         block_id, warp_id, lane_id,
                         lane_reg(block_id, warp_id, lane_id, 3), expected_bid);
                $finish;
            end
            if (lane_reg(block_id, warp_id, lane_id, 4) !== (expected_tid + 32'd1)) begin
                $display("FAIL block=%0d warp=%0d lane=%0d r4 got=%0d expected=%0d",
                         block_id, warp_id, lane_id,
                         lane_reg(block_id, warp_id, lane_id, 4), expected_tid + 32'd1);
                $finish;
            end
        end
    endtask

    function [31:0] lane_reg;
        input integer block_id;
        input integer warp_id;
        input integer lane_id;
        input integer reg_id;
        begin
            if (block_id == 0 && warp_id == 0 && lane_id == 0) lane_reg = dut.simt.blocks[0].block_unit.warps[0].warp_unit.lanes[0].thread_lane.registers.regs[reg_id];
            else if (block_id == 0 && warp_id == 0 && lane_id == 1) lane_reg = dut.simt.blocks[0].block_unit.warps[0].warp_unit.lanes[1].thread_lane.registers.regs[reg_id];
            else if (block_id == 0 && warp_id == 0 && lane_id == 2) lane_reg = dut.simt.blocks[0].block_unit.warps[0].warp_unit.lanes[2].thread_lane.registers.regs[reg_id];
            else if (block_id == 0 && warp_id == 0 && lane_id == 3) lane_reg = dut.simt.blocks[0].block_unit.warps[0].warp_unit.lanes[3].thread_lane.registers.regs[reg_id];
            else if (block_id == 0 && warp_id == 1 && lane_id == 0) lane_reg = dut.simt.blocks[0].block_unit.warps[1].warp_unit.lanes[0].thread_lane.registers.regs[reg_id];
            else if (block_id == 0 && warp_id == 1 && lane_id == 1) lane_reg = dut.simt.blocks[0].block_unit.warps[1].warp_unit.lanes[1].thread_lane.registers.regs[reg_id];
            else if (block_id == 0 && warp_id == 1 && lane_id == 2) lane_reg = dut.simt.blocks[0].block_unit.warps[1].warp_unit.lanes[2].thread_lane.registers.regs[reg_id];
            else if (block_id == 0 && warp_id == 1 && lane_id == 3) lane_reg = dut.simt.blocks[0].block_unit.warps[1].warp_unit.lanes[3].thread_lane.registers.regs[reg_id];
            else if (block_id == 1 && warp_id == 0 && lane_id == 0) lane_reg = dut.simt.blocks[1].block_unit.warps[0].warp_unit.lanes[0].thread_lane.registers.regs[reg_id];
            else if (block_id == 1 && warp_id == 0 && lane_id == 1) lane_reg = dut.simt.blocks[1].block_unit.warps[0].warp_unit.lanes[1].thread_lane.registers.regs[reg_id];
            else if (block_id == 1 && warp_id == 0 && lane_id == 2) lane_reg = dut.simt.blocks[1].block_unit.warps[0].warp_unit.lanes[2].thread_lane.registers.regs[reg_id];
            else if (block_id == 1 && warp_id == 0 && lane_id == 3) lane_reg = dut.simt.blocks[1].block_unit.warps[0].warp_unit.lanes[3].thread_lane.registers.regs[reg_id];
            else if (block_id == 1 && warp_id == 1 && lane_id == 0) lane_reg = dut.simt.blocks[1].block_unit.warps[1].warp_unit.lanes[0].thread_lane.registers.regs[reg_id];
            else if (block_id == 1 && warp_id == 1 && lane_id == 1) lane_reg = dut.simt.blocks[1].block_unit.warps[1].warp_unit.lanes[1].thread_lane.registers.regs[reg_id];
            else if (block_id == 1 && warp_id == 1 && lane_id == 2) lane_reg = dut.simt.blocks[1].block_unit.warps[1].warp_unit.lanes[2].thread_lane.registers.regs[reg_id];
            else lane_reg = dut.simt.blocks[1].block_unit.warps[1].warp_unit.lanes[3].thread_lane.registers.regs[reg_id];
        end
    endfunction

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

    function instr_writes_register;
        input [31:0] instr;
        begin
            case (instr[31:26])
                `MGPU_OP_TID,
                `MGPU_OP_WID,
                `MGPU_OP_BID,
                `MGPU_OP_ADDI: instr_writes_register = 1'b1;
                default: instr_writes_register = 1'b0;
            endcase
        end
    endfunction

    function [8*8-1:0] instr_name;
        input [5:0] opcode;
        begin
            case (opcode)
                `MGPU_OP_TID:   instr_name = "TID";
                `MGPU_OP_WID:   instr_name = "WID";
                `MGPU_OP_BID:   instr_name = "BID";
                `MGPU_OP_ADDI:  instr_name = "ADDI";
                `MGPU_OP_BAR:   instr_name = "BAR";
                `MGPU_OP_EXIT:  instr_name = "EXIT";
                default:        instr_name = "OTHER";
            endcase
        end
    endfunction

    initial begin
        integer block_id;
        integer warp_id;
        integer lane_id;
        integer init_idx;
        integer tid;

        rst = 1'b1;
        prog_we = 1'b0;
        prog_addr = {PROG_ADDR_WIDTH{1'b0}};
        prog_wdata = 32'b0;
        launch = 1'b0;
        observed_retired = 16'b0;
        for (init_idx = 0; init_idx < (1 << PROG_ADDR_WIDTH); init_idx = init_idx + 1) begin
            program_image[init_idx[PROG_ADDR_WIDTH-1:0]] =
                pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0);
        end

        repeat (2) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        load_instr(5'd0, pack_instr(`MGPU_OP_TID, 4'd1, 4'd0, 4'd0, 14'd0));
        load_instr(5'd1, pack_instr(`MGPU_OP_WID, 4'd2, 4'd0, 4'd0, 14'd0));
        load_instr(5'd2, pack_instr(`MGPU_OP_BID, 4'd3, 4'd0, 4'd0, 14'd0));
        load_instr(5'd3, pack_instr(`MGPU_OP_BAR, 4'd0, 4'd0, 4'd0, 14'd0));
        load_instr(5'd4, pack_instr(`MGPU_OP_ADDI, 4'd4, 4'd1, 4'd0, 14'd1));
        load_instr(5'd5, pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0));

        @(negedge clk);
        launch = 1'b1;
        @(negedge clk);
        launch = 1'b0;

        $display("");
        $display("=== LAUNCH active_mask=%b base_pc=%0d block_dim=%0d grid_dim=%0d blocks=%0d warps_per_block=%0d ===",
                 {WARP_SIZE{1'b1}}, {PROG_ADDR_WIDTH{1'b0}}, 32'd8, 32'd2, NUM_BLOCKS, NUM_WARPS);
        print_program_listing();

        while (!done) begin
            @(negedge clk);
            sample_retire();
        end
        #1;

        $display("--- FINAL CORE STATUS ---");
        print_core_status(dut.selected_idx);
        $display("");

        if (error || unsupported || divide_by_zero) begin
            $display("FAIL scheduler status error=%0d unsupported=%0d div0=%0d",
                     error, unsupported, divide_by_zero);
            $finish;
        end

        if (retired_count !== 16'd24) begin
            $display("FAIL retired_count got=%0d expected=24", retired_count);
            $finish;
        end

        for (block_id = 0; block_id < NUM_BLOCKS; block_id = block_id + 1) begin
            for (warp_id = 0; warp_id < NUM_WARPS; warp_id = warp_id + 1) begin
                for (lane_id = 0; lane_id < WARP_SIZE; lane_id = lane_id + 1) begin
                    tid = (block_id * 8) + (warp_id * WARP_SIZE) + lane_id;
                    check_lane(block_id, warp_id, lane_id, tid, warp_id, block_id);
                end
            end
        end

        $display("mini_gpu_scheduler_tb PASS");
        $finish;
    end
endmodule
