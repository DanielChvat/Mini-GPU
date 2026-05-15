`timescale 1ns/1ps
module basys3_comm_top #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD_RATE = 115200,
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32,
    parameter MEMORY_BANK_DEPTH = 8192,
    parameter PROG_ADDR_WIDTH = 12,
    parameter CONST_ADDR_WIDTH = 8,
    parameter WARP_SIZE = 4,
    parameter NUM_CORES = 1,
    parameter NUM_WARPS_PER_CORE = 1,
    parameter WARP_ID_WIDTH = 1
) (
    input  wire        CLK100MHZ,
    input  wire        btnC,
    input  wire [15:0] sw,
    input  wire        RsRx,
    output wire        RsTx,
    output wire [15:0] led,
    output wire [6:0]  seg,
    output wire        dp,
    output wire [3:0]  an
);
    reg [15:0] power_on_reset = 16'hffff;
    reg btnC_meta = 1'b0;
    reg btnC_sync = 1'b0;

    wire rst = btnC_sync || (power_on_reset != 16'b0);

    wire        rsp_valid = 1'b0;
    wire [31:0] rsp_data = 32'b0;
    wire        rsp_ready;

    wire [3:0] data_mem_req_valid;
    wire [3:0] data_mem_req_write;
    wire [(4*ADDR_WIDTH)-1:0] data_mem_req_addr;
    wire [(4*DATA_WIDTH)-1:0] data_mem_req_wdata;
    wire [3:0] data_mem_req_ready;
    wire [3:0] data_mem_resp_valid;
    wire [(4*DATA_WIDTH)-1:0] data_mem_resp_rdata;

    wire [7:0]  dbg_rx_cmd;
    wire [15:0] dbg_rx_addr;
    wire [15:0] dbg_rx_len;
    wire [3:0]  validate_model_id;
    wire        validate_triggered;
    wire security_reset_triggered;
    wire        prog_we_full;
    wire [ADDR_WIDTH-1:0] prog_addr_full;
    wire [31:0] prog_wdata;
    wire        const_we_full;
    wire [ADDR_WIDTH-1:0] const_addr_full;
    wire [31:0] const_wdata;
    wire        launch_valid;
    wire [ADDR_WIDTH-1:0] launch_base_pc_full;
    wire [31:0] launch_grid_dim;
    wire [31:0] launch_block_dim;
    wire [31:0] launch_active_mask_full;
    wire [31:0] status_word;

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
    wire [(NUM_CORES*WARP_SIZE*DATA_WIDTH)-1:0] core_last_writeback_data;
    wire gpu_busy;
    wire gpu_done;
    wire gpu_error;
    wire gpu_unsupported;
    wire gpu_divide_by_zero;

    // GPU to bus controller interface (arbitrated)
    wire [WARP_SIZE-1:0] gpu_mem_req_valid;
    wire [WARP_SIZE-1:0] gpu_mem_req_write;
    wire [(WARP_SIZE*ADDR_WIDTH)-1:0] gpu_mem_req_addr;
    wire [(WARP_SIZE*DATA_WIDTH)-1:0] gpu_mem_req_wdata;
    wire [(WARP_SIZE*4)-1:0] gpu_mem_req_wmask;
    wire [WARP_SIZE-1:0] gpu_mem_req_ready;
    wire [WARP_SIZE-1:0] gpu_mem_resp_valid;
    wire [(WARP_SIZE*DATA_WIDTH)-1:0] gpu_mem_resp_rdata;

    // Bus controller to memory interface
    wire [3:0] merged_mem_req_valid;
    wire [3:0] merged_mem_req_write;
    wire [(4*ADDR_WIDTH)-1:0] merged_mem_req_addr;
    wire [(4*DATA_WIDTH)-1:0] merged_mem_req_wdata;
    wire [(4*4)-1:0] merged_mem_req_wmask;
    wire [3:0] merged_mem_req_ready;
    wire [3:0] merged_mem_resp_valid;
    wire [(4*DATA_WIDTH)-1:0] merged_mem_resp_rdata;
    wire merged_mem_write_done;      
    wire merged_mem_write_fail;      // no memory blocks implemented right now 
    wire merged_memory_status_consumed;
    wire [1:0] memory_request_id;

    wire [DATA_WIDTH-1:0] gpu_grid_dim = {{(DATA_WIDTH-1){1'b0}}, 1'b1};
    wire [DATA_WIDTH-1:0] gpu_block_dim = {{(DATA_WIDTH-3){1'b0}}, 3'd4};

    reg [15:0] last_mem_addr = 16'b0;
    reg [31:0] last_mem_wdata = 32'b0;
    reg [31:0] last_mem_rdata = 32'b0;
    reg [15:0] write_count = 16'b0;
    reg [15:0] read_count = 16'b0;

    wire mem_write_done;
    wire mem_write_fail;
    wire memory_status_consumed;    

    wire mem_write0 = data_mem_req_valid[0] && data_mem_req_write[0] && data_mem_req_ready[0];
    wire mem_read0 = data_mem_req_valid[0] && !data_mem_req_write[0] && data_mem_req_ready[0];

    //== old policy enforcement ==//
    // this prevents communication controller from writing to memory while GPU is busy, preventing conflicts between core and host 
    wire [3:0] gated_data_mem_req_valid = gpu_busy ? 4'b0000 : data_mem_req_valid;
    

    communication_controller #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .MEMORY_BANK_DEPTH(MEMORY_BANK_DEPTH)
    ) comm (
        .clk(CLK100MHZ),
        .rst(rst),
        .rx_line(RsRx),
        .tx_line(RsTx),
        .rsp_valid(rsp_valid),
        .rsp_data(rsp_data),
        .rsp_ready(rsp_ready),
        .data_mem_req_valid(data_mem_req_valid),
        .data_mem_req_write(data_mem_req_write),
        .data_mem_req_addr(data_mem_req_addr),
        .data_mem_req_wdata(data_mem_req_wdata),
        .data_mem_req_ready(data_mem_req_ready),
        .data_mem_resp_valid(data_mem_resp_valid),
        .data_mem_resp_rdata(data_mem_resp_rdata),
        .mem_write_done(mem_write_done),
        .mem_write_fail(mem_write_fail),
        .memory_status_consumed(memory_status_consumed),
        .prog_we(prog_we_full),
        .prog_addr(prog_addr_full),
        .prog_wdata(prog_wdata),
        .const_we(const_we_full),
        .const_addr(const_addr_full),
        .const_wdata(const_wdata),
        .launch_valid(launch_valid),
        .launch_base_pc(launch_base_pc_full),
        .launch_grid_dim(launch_grid_dim),
        .launch_block_dim(launch_block_dim),
        .launch_active_mask(launch_active_mask_full),
        .status_word(status_word),
        .validate_model_id(validate_model_id),
        .validate_triggered(validate_triggered),
        .security_reset_triggered(security_reset_triggered),
        .dbg_rx_cmd(dbg_rx_cmd),
        .dbg_rx_addr(dbg_rx_addr),
        .dbg_rx_len(dbg_rx_len)
    );

    mini_gpu #(
        .WIDTH(DATA_WIDTH),
        .WARP_SIZE(WARP_SIZE),
        .NUM_CORES(NUM_CORES),
        .NUM_WARPS_PER_CORE(NUM_WARPS_PER_CORE),
        .WARP_ID_WIDTH(WARP_ID_WIDTH),
        .PROG_ADDR_WIDTH(PROG_ADDR_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .CONST_ADDR_WIDTH(CONST_ADDR_WIDTH),
        .MEMORY_BANK_DEPTH(MEMORY_BANK_DEPTH),
        .ENABLE_FLOAT_ADD(1),
        .ENABLE_FLOAT_MUL(1),
        .ENABLE_FLOAT_DIV(0),
        .FLOAT_FP32_ONLY(0),
        .USE_SHARED_FLOAT(1),
        .SHARED_FLOAT_UNITS(1)
    ) gpu (
        .clk(CLK100MHZ),
        .rst(rst),
        .prog_we(prog_we_full),
        .prog_addr(prog_addr_full[PROG_ADDR_WIDTH-1:0]),
        .prog_wdata(prog_wdata),
        .const_we(const_we_full),
        .const_addr(const_addr_full[CONST_ADDR_WIDTH-1:0]),
        .const_wdata(const_wdata),
        .launch(launch_valid),
        .base_pc(launch_base_pc_full[PROG_ADDR_WIDTH-1:0]),
        .active_mask(launch_active_mask_full[WARP_SIZE-1:0]),
        .block_dim(gpu_block_dim),
        .grid_dim(gpu_grid_dim),
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
        .mem_req_valid(gpu_mem_req_valid),
        .mem_req_write(gpu_mem_req_write),
        .mem_req_addr(gpu_mem_req_addr),
        .mem_req_wdata(gpu_mem_req_wdata),
        .mem_req_wmask(gpu_mem_req_wmask),
        .mem_req_ready(gpu_mem_req_ready),
        .mem_resp_valid(gpu_mem_resp_valid),
        .mem_resp_rdata(gpu_mem_resp_rdata),
        .busy(gpu_busy),
        .done(gpu_done),
        .error(gpu_error),
        .unsupported(gpu_unsupported),
        .divide_by_zero(gpu_divide_by_zero)
    );

    bus_controller #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) memory_bus (
        .clk(CLK100MHZ),
        .rst(rst),
        .host_mem_req_valid(gated_data_mem_req_valid),   // enforce old policy here by gating the valid signal with gpu_busy
        .host_mem_req_write(data_mem_req_write),
        .host_mem_req_addr(data_mem_req_addr),
        .host_mem_req_wdata(data_mem_req_wdata),
        .host_mem_req_wmask({4{4'b1111}}),
        .host_mem_req_ready(data_mem_req_ready),
        .host_mem_resp_valid(data_mem_resp_valid),
        .host_mem_resp_rdata(data_mem_resp_rdata),
        .host_mem_write_done(mem_write_done),
        .host_mem_write_fail(mem_write_fail),
        .host_memory_status_consumed(memory_status_consumed),
        .core_mem_req_valid(gpu_mem_req_valid),
        .core_mem_req_write(gpu_mem_req_write),
        .core_mem_req_addr(gpu_mem_req_addr),
        .core_mem_req_wdata(gpu_mem_req_wdata),
        .core_mem_req_wmask(gpu_mem_req_wmask),
        .core_mem_req_ready(gpu_mem_req_ready),
        .core_mem_resp_valid(gpu_mem_resp_valid),
        .core_mem_resp_rdata(gpu_mem_resp_rdata),
        .core_mem_write_done(),                         // not used/implemented by the core
        .core_mem_write_fail(),                         // not used/implemented by the core
        .core_memory_status_consumed(1'b1),             // will always consume 
        .merged_mem_req_valid(merged_mem_req_valid),
        .merged_mem_req_write(merged_mem_req_write),
        .merged_mem_req_addr(merged_mem_req_addr),
        .merged_mem_req_wdata(merged_mem_req_wdata),
        .merged_mem_req_wmask(merged_mem_req_wmask),
        .merged_mem_req_ready(merged_mem_req_ready),
        .merged_mem_resp_valid(merged_mem_resp_valid),
        .merged_mem_resp_rdata(merged_mem_resp_rdata),
        .merged_mem_write_done(merged_mem_write_done),
        .merged_mem_write_fail(merged_mem_write_fail),
        .merged_memory_status_consumed(merged_memory_status_consumed),
        .memory_request_id(memory_request_id)
    );

    assign merged_mem_write_done = |(merged_mem_req_valid & merged_mem_req_write & merged_mem_req_ready);
    assign merged_mem_write_fail = 1'b0;      // no memory blocks implemented right now, so we can never fail
    memory #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .BANK_DEPTH(MEMORY_BANK_DEPTH)
    ) global_memory (
        .clk(CLK100MHZ),
        .rst(rst),
        .req_valid(merged_mem_req_valid),
        .req_write(merged_mem_req_write),
        .req_addr(merged_mem_req_addr),
        .req_wdata(merged_mem_req_wdata),
        .req_wmask(merged_mem_req_wmask),
        .req_ready(merged_mem_req_ready),
        .resp_valid(merged_mem_resp_valid),
        .resp_rdata(merged_mem_resp_rdata)
    );

    assign status_word = {
        16'b0,
        core_retired_count[7:0],
        3'b0,
        gpu_divide_by_zero,
        gpu_unsupported,
        gpu_error,
        gpu_busy,
        gpu_done
    };

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
            last_mem_addr <= 16'b0;
            last_mem_wdata <= 32'b0;
            last_mem_rdata <= 32'b0;
            write_count <= 16'b0;
            read_count <= 16'b0;
        end else begin
            if (mem_write0) begin
                last_mem_addr <= data_mem_req_addr[0 +: ADDR_WIDTH];
                last_mem_wdata <= data_mem_req_wdata[0 +: DATA_WIDTH];
                write_count <= write_count + 16'd1;
            end

            if (mem_read0) begin
                last_mem_addr <= data_mem_req_addr[0 +: ADDR_WIDTH];
                read_count <= read_count + 16'd1;
            end

            if (data_mem_resp_valid[0]) begin
                last_mem_rdata <= data_mem_resp_rdata[0 +: DATA_WIDTH];
            end
        end
    end

    wire [15:0] display_word =
        (sw[1:0] == 2'd0) ? {8'b0, dbg_rx_cmd} :
        (sw[1:0] == 2'd1) ? dbg_rx_addr :
        (sw[1:0] == 2'd2) ? dbg_rx_len :
                             last_mem_addr;

    assign led[0] = !rst;
    assign led[1] = RsRx;
    assign led[2] = RsTx;
    assign led[3] = launch_valid;
    assign led[4] = mem_write0;
    assign led[5] = mem_read0;
    assign led[6] = data_mem_resp_valid[0];
    assign led[7] = gpu_busy;
    assign led[15:8] = dbg_rx_cmd;

    basys3_comm_sevenseg display (
        .clk(CLK100MHZ),
        .value(display_word),
        .seg(seg),
        .dp(dp),
        .an(an)
    );
endmodule

module basys3_comm_sevenseg (
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
