`timescale 1ns/1ps

module warp #(
    parameter WIDTH = 32,
    parameter WARP_SIZE = 4,
    parameter ADDR_WIDTH = 16,
    parameter ENABLE_FLOAT_ADD = 1,
    parameter ENABLE_FLOAT_MUL = 1,
    parameter ENABLE_FLOAT_DIV = 1,
    parameter FLOAT_FP32_ONLY = 0,
    parameter USE_SHARED_FLOAT = 0,
    parameter SHARED_FLOAT_UNITS = 1
) (
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         instr_valid,
    input  wire [31:0]                  instr,
    input  wire [WARP_SIZE-1:0]         active_mask,
    input  wire [WIDTH-1:0]             warp_id,
    input  wire [WIDTH-1:0]             block_id,
    input  wire [WIDTH-1:0]             block_dim,
    input  wire [WIDTH-1:0]             grid_dim,
    input  wire [WIDTH-1:0]             const_data,
    output wire [WARP_SIZE-1:0]         mem_req_valid,
    output wire [WARP_SIZE-1:0]         mem_req_write,
    output wire [(WARP_SIZE*ADDR_WIDTH)-1:0] mem_req_addr,
    output wire [(WARP_SIZE*WIDTH)-1:0] mem_req_wdata,
    input  wire [WARP_SIZE-1:0]         mem_req_ready,
    input  wire [WARP_SIZE-1:0]         mem_resp_valid,
    input  wire [(WARP_SIZE*WIDTH)-1:0] mem_resp_rdata,
    output wire [WARP_SIZE-1:0]         supported_mask,
    output wire [WARP_SIZE-1:0]         divide_by_zero_mask,
    output wire [WARP_SIZE-1:0]         busy_mask,
    output wire [WARP_SIZE-1:0]         done_mask,
    output wire [WARP_SIZE-1:0]         writeback_enable_mask,
    output wire [(WARP_SIZE*4)-1:0]     writeback_addr,
    output wire [(WARP_SIZE*WIDTH)-1:0] writeback_data,
    output wire                         warp_busy,
    output wire                         warp_done
);
    wire [WARP_SIZE-1:0] float_req_valid;
    wire [WARP_SIZE-1:0] float_req_ready;
    wire [WARP_SIZE-1:0] float_resp_valid;
    wire [(WARP_SIZE*6)-1:0] float_req_opcode;
    wire [(WARP_SIZE*3)-1:0] float_req_fmt;
    wire [(WARP_SIZE*WIDTH)-1:0] float_req_lhs;
    wire [(WARP_SIZE*WIDTH)-1:0] float_req_rhs;
    reg [(WARP_SIZE*WIDTH)-1:0] float_resp_result;
    reg [WARP_SIZE-1:0] float_resp_supported;
    reg [WARP_SIZE-1:0] float_resp_divide_by_zero;

    reg [SHARED_FLOAT_UNITS-1:0] shared_float_start;
    reg [(SHARED_FLOAT_UNITS*6)-1:0] shared_float_opcode;
    reg [(SHARED_FLOAT_UNITS*3)-1:0] shared_float_fmt;
    reg [(SHARED_FLOAT_UNITS*WIDTH)-1:0] shared_float_lhs;
    reg [(SHARED_FLOAT_UNITS*WIDTH)-1:0] shared_float_rhs;
    wire [(SHARED_FLOAT_UNITS*WIDTH)-1:0] shared_float_result;
    wire [SHARED_FLOAT_UNITS-1:0] shared_float_supported;
    wire [SHARED_FLOAT_UNITS-1:0] shared_float_divide_by_zero;
    wire [SHARED_FLOAT_UNITS-1:0] shared_float_busy;
    wire [SHARED_FLOAT_UNITS-1:0] shared_float_done;
    reg [(SHARED_FLOAT_UNITS*WARP_SIZE)-1:0] shared_float_lane;
    reg [(SHARED_FLOAT_UNITS*WARP_SIZE)-1:0] shared_float_lane_next;
    reg [WARP_SIZE-1:0] float_req_ready_r;
    reg [WARP_SIZE-1:0] float_resp_valid_r;

    genvar lane;
    genvar fpu_slot;
    generate
        for (lane = 0; lane < WARP_SIZE; lane = lane + 1) begin : lanes
            wire [WIDTH-1:0] lane_id_value = lane;
            thread #(
                .WIDTH(WIDTH),
                .ADDR_WIDTH(ADDR_WIDTH),
                .ENABLE_FLOAT_ADD(ENABLE_FLOAT_ADD),
                .ENABLE_FLOAT_MUL(ENABLE_FLOAT_MUL),
                .ENABLE_FLOAT_DIV(ENABLE_FLOAT_DIV),
                .FLOAT_FP32_ONLY(FLOAT_FP32_ONLY),
                .USE_SHARED_FLOAT(USE_SHARED_FLOAT)
            ) thread_lane (
                .clk(clk),
                .rst(rst),
                .instr_valid(instr_valid && active_mask[lane]),
                .instr(instr),
                .thread_id(block_id * block_dim + warp_id * WARP_SIZE + lane_id_value),
                .lane_id(lane_id_value),
                .warp_id(warp_id),
                .block_id(block_id),
                .block_dim(block_dim),
                .grid_dim(grid_dim),
                .const_data(const_data),
                .mem_req_valid(mem_req_valid[lane]),
                .mem_req_write(mem_req_write[lane]),
                .mem_req_addr(mem_req_addr[(lane*ADDR_WIDTH) +: ADDR_WIDTH]),
                .mem_req_wdata(mem_req_wdata[(lane*WIDTH) +: WIDTH]),
                .mem_req_ready(mem_req_ready[lane]),
                .mem_resp_valid(mem_resp_valid[lane]),
                .mem_resp_rdata(mem_resp_rdata[(lane*WIDTH) +: WIDTH]),
                .float_req_valid(float_req_valid[lane]),
                .float_req_opcode(float_req_opcode[(lane*6) +: 6]),
                .float_req_fmt(float_req_fmt[(lane*3) +: 3]),
                .float_req_lhs(float_req_lhs[(lane*WIDTH) +: WIDTH]),
                .float_req_rhs(float_req_rhs[(lane*WIDTH) +: WIDTH]),
                .float_req_ready(float_req_ready[lane]),
                .float_resp_valid(float_resp_valid[lane]),
                .float_resp_result(float_resp_result[(lane*WIDTH) +: WIDTH]),
                .float_resp_supported(float_resp_supported[lane]),
                .float_resp_divide_by_zero(float_resp_divide_by_zero[lane]),
                .supported(supported_mask[lane]),
                .divide_by_zero(divide_by_zero_mask[lane]),
                .execute_busy(busy_mask[lane]),
                .execute_done(done_mask[lane]),
                .writeback_enable(writeback_enable_mask[lane]),
                .writeback_addr(writeback_addr[(lane*4) +: 4]),
                .writeback_data(writeback_data[(lane*WIDTH) +: WIDTH])
            );
        end
    endgenerate

    generate
        if (USE_SHARED_FLOAT) begin : gen_shared_float
            for (fpu_slot = 0; fpu_slot < SHARED_FLOAT_UNITS; fpu_slot = fpu_slot + 1) begin : shared_float_units
                shared_fpu #(
                    .ENABLE_FLOAT_ADD(ENABLE_FLOAT_ADD),
                    .ENABLE_FLOAT_MUL(ENABLE_FLOAT_MUL),
                    .ENABLE_FLOAT_DIV(ENABLE_FLOAT_DIV),
                    .FLOAT_FP32_ONLY(FLOAT_FP32_ONLY),
                    .LATENCY(32)
                ) shared_float_unit (
                    .clk(clk),
                    .rst(rst),
                    .start(shared_float_start[fpu_slot]),
                    .opcode(shared_float_opcode[(fpu_slot*6) +: 6]),
                    .fmt(shared_float_fmt[(fpu_slot*3) +: 3]),
                    .lhs(shared_float_lhs[(fpu_slot*WIDTH) +: 32]),
                    .rhs(shared_float_rhs[(fpu_slot*WIDTH) +: 32]),
                    .result(shared_float_result[(fpu_slot*WIDTH) +: WIDTH]),
                    .supported(shared_float_supported[fpu_slot]),
                    .divide_by_zero(shared_float_divide_by_zero[fpu_slot]),
                    .busy(shared_float_busy[fpu_slot]),
                    .done(shared_float_done[fpu_slot])
                );
            end

            assign float_req_ready = float_req_ready_r;
            assign float_resp_valid = float_resp_valid_r;
        end else begin : gen_no_shared_float
            assign shared_float_result = {(SHARED_FLOAT_UNITS*WIDTH){1'b0}};
            assign shared_float_supported = {SHARED_FLOAT_UNITS{1'b0}};
            assign shared_float_divide_by_zero = {SHARED_FLOAT_UNITS{1'b0}};
            assign shared_float_busy = {SHARED_FLOAT_UNITS{1'b0}};
            assign shared_float_done = {SHARED_FLOAT_UNITS{1'b0}};
            assign float_req_ready = {WARP_SIZE{1'b0}};
            assign float_resp_valid = {WARP_SIZE{1'b0}};
        end
    endgenerate

    integer shared_lane_index;
    integer shared_slot_index;
    integer seq_slot_index;
    integer resp_slot_index;
    integer resp_lane_index;
    reg [WARP_SIZE-1:0] granted_float_lanes;
    always @* begin
        shared_float_start = {SHARED_FLOAT_UNITS{1'b0}};
        shared_float_opcode = {(SHARED_FLOAT_UNITS*6){1'b0}};
        shared_float_fmt = {(SHARED_FLOAT_UNITS*3){1'b0}};
        shared_float_lhs = {(SHARED_FLOAT_UNITS*WIDTH){1'b0}};
        shared_float_rhs = {(SHARED_FLOAT_UNITS*WIDTH){1'b0}};
        shared_float_lane_next = {(SHARED_FLOAT_UNITS*WARP_SIZE){1'b0}};
        float_req_ready_r = {WARP_SIZE{1'b0}};
        granted_float_lanes = {WARP_SIZE{1'b0}};

        if (USE_SHARED_FLOAT) begin
            for (shared_slot_index = 0; shared_slot_index < SHARED_FLOAT_UNITS; shared_slot_index = shared_slot_index + 1) begin
                if (!shared_float_busy[shared_slot_index] && !shared_float_done[shared_slot_index]) begin
                    for (shared_lane_index = 0; shared_lane_index < WARP_SIZE; shared_lane_index = shared_lane_index + 1) begin
                        if (!shared_float_start[shared_slot_index] &&
                            float_req_valid[shared_lane_index] &&
                            !granted_float_lanes[shared_lane_index]) begin
                            shared_float_start[shared_slot_index] = 1'b1;
                            shared_float_lane_next[(shared_slot_index*WARP_SIZE) + shared_lane_index] = 1'b1;
                            shared_float_opcode[(shared_slot_index*6) +: 6] =
                                float_req_opcode[(shared_lane_index*6) +: 6];
                            shared_float_fmt[(shared_slot_index*3) +: 3] =
                                float_req_fmt[(shared_lane_index*3) +: 3];
                            shared_float_lhs[(shared_slot_index*WIDTH) +: WIDTH] =
                                float_req_lhs[(shared_lane_index*WIDTH) +: WIDTH];
                            shared_float_rhs[(shared_slot_index*WIDTH) +: WIDTH] =
                                float_req_rhs[(shared_lane_index*WIDTH) +: WIDTH];
                            granted_float_lanes[shared_lane_index] = 1'b1;
                            float_req_ready_r[shared_lane_index] = 1'b1;
                        end
                    end
                end
            end
        end
    end

    always @* begin
        float_resp_valid_r = {WARP_SIZE{1'b0}};
        float_resp_result = {(WARP_SIZE*WIDTH){1'b0}};
        float_resp_supported = {WARP_SIZE{1'b0}};
        float_resp_divide_by_zero = {WARP_SIZE{1'b0}};

        if (USE_SHARED_FLOAT) begin
            for (resp_slot_index = 0; resp_slot_index < SHARED_FLOAT_UNITS; resp_slot_index = resp_slot_index + 1) begin
                if (shared_float_done[resp_slot_index]) begin
                    for (resp_lane_index = 0; resp_lane_index < WARP_SIZE; resp_lane_index = resp_lane_index + 1) begin
                        if (shared_float_lane[(resp_slot_index*WARP_SIZE) + resp_lane_index]) begin
                            float_resp_valid_r[resp_lane_index] = 1'b1;
                            float_resp_result[(resp_lane_index*WIDTH) +: WIDTH] =
                                shared_float_result[(resp_slot_index*WIDTH) +: WIDTH];
                            float_resp_supported[resp_lane_index] =
                                shared_float_supported[resp_slot_index];
                            float_resp_divide_by_zero[resp_lane_index] =
                                shared_float_divide_by_zero[resp_slot_index];
                        end
                    end
                end
            end
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            shared_float_lane <= {(SHARED_FLOAT_UNITS*WARP_SIZE){1'b0}};
        end else begin
            for (seq_slot_index = 0; seq_slot_index < SHARED_FLOAT_UNITS; seq_slot_index = seq_slot_index + 1) begin
                if (shared_float_start[seq_slot_index]) begin
                    shared_float_lane[(seq_slot_index*WARP_SIZE) +: WARP_SIZE] <=
                        shared_float_lane_next[(seq_slot_index*WARP_SIZE) +: WARP_SIZE];
                end else if (shared_float_done[seq_slot_index]) begin
                    shared_float_lane[(seq_slot_index*WARP_SIZE) +: WARP_SIZE] <= {WARP_SIZE{1'b0}};
                end
            end
        end
    end

    assign warp_busy = |(busy_mask & active_mask);
    assign warp_done = |active_mask && ((done_mask & active_mask) == active_mask);
endmodule
