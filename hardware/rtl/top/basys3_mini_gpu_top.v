`timescale 1ns/1ps

`include "minigpu_isa.vh"

module basys3_mini_gpu_top #(
    parameter NUM_CORES = 1,
    parameter NUM_WARPS_PER_CORE = 1,
    parameter WARP_ID_WIDTH = 1,
    parameter SHARED_FLOAT_UNITS = 2
) (
    input  wire        CLK100MHZ,
    input  wire        btnC,
    input  wire [15:0] sw,
    output wire [15:0] led,
    output wire [6:0]  seg,
    output wire        dp,
    output wire [3:0]  an
);
    localparam WIDTH = 32;
    localparam WARP_SIZE = 4;
    localparam PROG_ADDR_WIDTH = 4;
    localparam ADDR_WIDTH = 16;
    localparam CONST_ADDR_WIDTH = 4;
    localparam MEMORY_BANK_DEPTH = 8192;
    localparam ENABLE_FLOAT_ADD = 1;
    localparam ENABLE_FLOAT_MUL = 1;
    localparam ENABLE_FLOAT_DIV = 0;
    localparam FLOAT_FP32_ONLY = 0;
    localparam USE_SHARED_FLOAT = 1;

    localparam STATE_RESET       = 4'd0;
    localparam STATE_LOAD_CONST0 = 4'd1;
    localparam STATE_LOAD_CONST1 = 4'd2;
    localparam STATE_LOAD_PROG0  = 4'd3;
    localparam STATE_LOAD_PROG1  = 4'd4;
    localparam STATE_LOAD_PROG2  = 4'd5;
    localparam STATE_LOAD_PROG3  = 4'd6;
    localparam STATE_LOAD_PROG4  = 4'd7;
    localparam STATE_LAUNCH      = 4'd8;
    localparam STATE_RUN         = 4'd9;
    localparam STATE_DONE        = 4'd10;

    reg [3:0] state = STATE_RESET;
    reg [15:0] power_on_reset = 16'hffff;
    reg btnC_meta = 1'b0;
    reg btnC_sync = 1'b0;
    reg prog_we = 1'b0;
    reg [PROG_ADDR_WIDTH-1:0] prog_addr = {PROG_ADDR_WIDTH{1'b0}};
    reg [31:0] prog_wdata = 32'b0;
    reg const_we = 1'b0;
    reg [CONST_ADDR_WIDTH-1:0] const_addr = {CONST_ADDR_WIDTH{1'b0}};
    reg [WIDTH-1:0] const_wdata = {WIDTH{1'b0}};
    reg launch = 1'b0;
    reg pass_latched = 1'b0;
    reg fail_latched = 1'b0;
    reg [15:0] run_cycles = 16'b0;

    wire rst = btnC_sync || (power_on_reset != 16'b0);
    wire [NUM_CORES-1:0] core_busy;
    wire [NUM_CORES-1:0] core_done;
    wire [NUM_CORES-1:0] core_error;
    wire [NUM_CORES-1:0] core_unsupported;
    wire [NUM_CORES-1:0] core_divide_by_zero;
    wire [(NUM_CORES*PROG_ADDR_WIDTH)-1:0] core_pc;
    wire [(NUM_CORES*32)-1:0] core_current_instr;
    wire [(NUM_CORES*32)-1:0] core_last_instr;
    wire [(NUM_CORES*16)-1:0] core_retired_count;
    wire [(NUM_CORES*WARP_SIZE)-1:0] core_last_writeback_mask;
    wire [(NUM_CORES*4)-1:0] core_last_writeback_addr;
    wire [(NUM_CORES*WARP_SIZE*WIDTH)-1:0] core_last_writeback_data;
    wire busy;
    wire done;
    wire error;
    wire unsupported;
    wire divide_by_zero;

    wire [15:0] retired_count = core_retired_count[15:0];
    wire [3:0] last_writeback_mask = core_last_writeback_mask[3:0];
    wire [3:0] last_writeback_addr = core_last_writeback_addr[3:0];
    wire [31:0] lane0_data = core_last_writeback_data[(0*WIDTH) +: WIDTH];
    wire [31:0] lane1_data = core_last_writeback_data[(1*WIDTH) +: WIDTH];
    wire [31:0] lane2_data = core_last_writeback_data[(2*WIDTH) +: WIDTH];
    wire [31:0] lane3_data = core_last_writeback_data[(3*WIDTH) +: WIDTH];
    wire [31:0] expected_fp32_result = 32'h4163D70A;
    wire result_matches =
        (last_writeback_mask == 4'b1111) &&
        (last_writeback_addr == 4'd4) &&
        (retired_count == 16'd5) &&
        (lane0_data == expected_fp32_result) &&
        (lane1_data == expected_fp32_result) &&
        (lane2_data == expected_fp32_result) &&
        (lane3_data == expected_fp32_result);

    wire [31:0] selected_lane_data =
        (sw[1:0] == 2'd0) ? lane0_data :
        (sw[1:0] == 2'd1) ? lane1_data :
        (sw[1:0] == 2'd2) ? lane2_data :
                             lane3_data;
    wire [15:0] display_word =
        (sw[3:2] == 2'd0) ? selected_lane_data[15:0] :
        (sw[3:2] == 2'd1) ? selected_lane_data[31:16] :
        (sw[3:2] == 2'd2) ? retired_count :
                             run_cycles;

    mini_gpu #(
        .WIDTH(WIDTH),
        .WARP_SIZE(WARP_SIZE),
        .NUM_CORES(NUM_CORES),
        .NUM_WARPS_PER_CORE(NUM_WARPS_PER_CORE),
        .WARP_ID_WIDTH(WARP_ID_WIDTH),
        .PROG_ADDR_WIDTH(PROG_ADDR_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .CONST_ADDR_WIDTH(CONST_ADDR_WIDTH),
        .MEMORY_BANK_DEPTH(MEMORY_BANK_DEPTH),
        .ENABLE_FLOAT_ADD(ENABLE_FLOAT_ADD),
        .ENABLE_FLOAT_MUL(ENABLE_FLOAT_MUL),
        .ENABLE_FLOAT_DIV(ENABLE_FLOAT_DIV),
        .FLOAT_FP32_ONLY(FLOAT_FP32_ONLY),
        .USE_SHARED_FLOAT(USE_SHARED_FLOAT),
        .SHARED_FLOAT_UNITS(SHARED_FLOAT_UNITS)
    ) dut (
        .clk(CLK100MHZ),
        .rst(rst),
        .prog_we(prog_we),
        .prog_addr(prog_addr),
        .prog_wdata(prog_wdata),
        .const_we(const_we),
        .const_addr(const_addr),
        .const_wdata(const_wdata),
        .launch(launch),
        .base_pc({PROG_ADDR_WIDTH{1'b0}}),
        .active_mask(4'b1111),
        .block_dim(32'd4),
        .grid_dim(32'd1),
        .core_busy(core_busy),
        .core_done(core_done),
        .core_error(core_error),
        .core_unsupported(core_unsupported),
        .core_divide_by_zero(core_divide_by_zero),
        .core_pc(core_pc),
        .core_current_instr(core_current_instr),
        .core_last_instr(core_last_instr),
        .core_retired_count(core_retired_count),
        .core_last_writeback_mask(core_last_writeback_mask),
        .core_last_writeback_addr(core_last_writeback_addr),
        .core_last_writeback_data(core_last_writeback_data),
        .busy(busy),
        .done(done),
        .error(error),
        .unsupported(unsupported),
        .divide_by_zero(divide_by_zero)
    );

    always @(posedge CLK100MHZ) begin
        btnC_meta <= btnC;
        btnC_sync <= btnC_meta;

        if (btnC_sync) begin
            power_on_reset <= 16'hffff;
        end else if (power_on_reset != 16'b0) begin
            power_on_reset <= power_on_reset - 16'd1;
        end
    end

    always @(posedge CLK100MHZ) begin
        if (rst) begin
            state <= STATE_RESET;
            prog_we <= 1'b0;
            prog_addr <= {PROG_ADDR_WIDTH{1'b0}};
            prog_wdata <= 32'b0;
            const_we <= 1'b0;
            const_addr <= {CONST_ADDR_WIDTH{1'b0}};
            const_wdata <= {WIDTH{1'b0}};
            launch <= 1'b0;
            pass_latched <= 1'b0;
            fail_latched <= 1'b0;
            run_cycles <= 16'b0;
        end else begin
            prog_we <= 1'b0;
            const_we <= 1'b0;
            launch <= 1'b0;

            case (state)
                STATE_RESET: begin
                    run_cycles <= 16'b0;
                    state <= STATE_LOAD_CONST0;
                end

                STATE_LOAD_CONST0: begin
                    const_we <= 1'b1;
                    const_addr <= 4'd0;
                    const_wdata <= 32'h40A3D70A; // 5.12f
                    state <= STATE_LOAD_CONST1;
                end

                STATE_LOAD_CONST1: begin
                    const_we <= 1'b1;
                    const_addr <= 4'd1;
                    const_wdata <= 32'h40000000; // 2.0f
                    state <= STATE_LOAD_PROG0;
                end

                STATE_LOAD_PROG0: begin
                    prog_we <= 1'b1;
                    prog_addr <= 4'd0;
                    prog_wdata <= pack_instr(`MGPU_OP_LDC, 4'd1, 4'd0, 4'd0, 14'd0);
                    state <= STATE_LOAD_PROG1;
                end

                STATE_LOAD_PROG1: begin
                    prog_we <= 1'b1;
                    prog_addr <= 4'd1;
                    prog_wdata <= pack_instr(`MGPU_OP_LDC, 4'd2, 4'd0, 4'd0, 14'd1);
                    state <= STATE_LOAD_PROG2;
                end

                STATE_LOAD_PROG2: begin
                    prog_we <= 1'b1;
                    prog_addr <= 4'd2;
                    prog_wdata <= pack_instr(`MGPU_OP_FADD, 4'd3, 4'd1, 4'd2, {11'b0, `MGPU_FMT_FP32});
                    state <= STATE_LOAD_PROG3;
                end

                STATE_LOAD_PROG3: begin
                    prog_we <= 1'b1;
                    prog_addr <= 4'd3;
                    prog_wdata <= pack_instr(`MGPU_OP_FMUL, 4'd4, 4'd3, 4'd2, {11'b0, `MGPU_FMT_FP32});
                    state <= STATE_LOAD_PROG4;
                end

                STATE_LOAD_PROG4: begin
                    prog_we <= 1'b1;
                    prog_addr <= 4'd4;
                    prog_wdata <= pack_instr(`MGPU_OP_EXIT, 4'd0, 4'd0, 4'd0, 14'd0);
                    state <= STATE_LAUNCH;
                end

                STATE_LAUNCH: begin
                    launch <= 1'b1;
                    run_cycles <= 16'b0;
                    state <= STATE_RUN;
                end

                STATE_RUN: begin
                    if (run_cycles != 16'hffff) begin
                        run_cycles <= run_cycles + 16'd1;
                    end

                    if (done) begin
                        pass_latched <= result_matches && !error && !unsupported && !divide_by_zero;
                        fail_latched <= !result_matches || error || unsupported || divide_by_zero;
                        state <= STATE_DONE;
                    end
                end

                default: begin
                    state <= STATE_DONE;
                end
            endcase
        end
    end

    assign led[0] = done;
    assign led[1] = busy;
    assign led[2] = pass_latched;
    assign led[3] = fail_latched;
    assign led[4] = error;
    assign led[5] = unsupported;
    assign led[6] = divide_by_zero;
    assign led[7] = (SHARED_FLOAT_UNITS > 1);
    assign led[11:8] = last_writeback_mask;
    assign led[15:12] = state;

    sevenseg_hex display (
        .clk(CLK100MHZ),
        .value(display_word),
        .seg(seg),
        .dp(dp),
        .an(an)
    );

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
endmodule

module sevenseg_hex (
    input  wire        clk,
    input  wire [15:0] value,
    output wire [6:0]  seg,
    output wire        dp,
    output wire [3:0]  an
);
    reg [15:0] refresh = 16'b0;
    reg [3:0] digit = 4'b0;
    reg [6:0] seg_r = 7'b1111111;
    reg [3:0] an_r = 4'b1111;

    always @(posedge clk) begin
        refresh <= refresh + 16'd1;
    end

    always @* begin
        case (refresh[15:14])
            2'd0: begin
                an_r = 4'b1110;
                digit = value[3:0];
            end
            2'd1: begin
                an_r = 4'b1101;
                digit = value[7:4];
            end
            2'd2: begin
                an_r = 4'b1011;
                digit = value[11:8];
            end
            default: begin
                an_r = 4'b0111;
                digit = value[15:12];
            end
        endcase

        case (digit)
            4'h0: seg_r = 7'b1000000;
            4'h1: seg_r = 7'b1111001;
            4'h2: seg_r = 7'b0100100;
            4'h3: seg_r = 7'b0110000;
            4'h4: seg_r = 7'b0011001;
            4'h5: seg_r = 7'b0010010;
            4'h6: seg_r = 7'b0000010;
            4'h7: seg_r = 7'b1111000;
            4'h8: seg_r = 7'b0000000;
            4'h9: seg_r = 7'b0010000;
            4'ha: seg_r = 7'b0001000;
            4'hb: seg_r = 7'b0000011;
            4'hc: seg_r = 7'b1000110;
            4'hd: seg_r = 7'b0100001;
            4'he: seg_r = 7'b0000110;
            default: seg_r = 7'b0001110;
        endcase
    end

    assign seg = seg_r;
    assign an = an_r;
    assign dp = 1'b1;
endmodule
