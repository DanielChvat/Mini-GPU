`timescale 1ns/1ps

`include "minigpu_isa.vh"

module mini_gpu_core #(
    parameter WIDTH = 32,
    parameter WARP_SIZE = 4,
    parameter PROG_ADDR_WIDTH = 8,
    parameter ADDR_WIDTH = 16,
    parameter CONST_ADDR_WIDTH = 8,
    parameter NUM_WARPS = 1,
    parameter NUM_BLOCKS = 1,
    parameter WARP_ID_WIDTH = 1,
    parameter BLOCK_ID_WIDTH = 1,
    parameter ENABLE_FLOAT_ADD = 1,
    parameter ENABLE_FLOAT_MUL = 1,
    parameter ENABLE_FLOAT_DIV = 1,
    parameter FLOAT_FP32_ONLY = 0,
    parameter USE_SHARED_FLOAT = 0,
    parameter SHARED_FLOAT_UNITS = 1
) (
    input  wire                         clk,
    input  wire                         rst,

    input  wire                         prog_we,
    input  wire [PROG_ADDR_WIDTH-1:0]   prog_addr,
    input  wire [31:0]                  prog_wdata,

    input  wire                         const_we,
    input  wire [CONST_ADDR_WIDTH-1:0]  const_addr,
    input  wire [WIDTH-1:0]             const_wdata,

    input  wire                         launch,
    input  wire [PROG_ADDR_WIDTH-1:0]   base_pc,
    input  wire [WARP_SIZE-1:0]         active_mask,
    input  wire [WIDTH-1:0]             base_block_id,
    input  wire [WIDTH-1:0]             block_dim,
    input  wire [WIDTH-1:0]             grid_dim,

    output wire [WARP_SIZE-1:0]         mem_req_valid,
    output wire [WARP_SIZE-1:0]         mem_req_write,
    output wire [(WARP_SIZE*ADDR_WIDTH)-1:0] mem_req_addr,
    output wire [(WARP_SIZE*WIDTH)-1:0] mem_req_wdata,
    input  wire [WARP_SIZE-1:0]         mem_req_ready,
    input  wire [WARP_SIZE-1:0]         mem_resp_valid,
    input  wire [(WARP_SIZE*WIDTH)-1:0] mem_resp_rdata,

    output reg                          busy,
    output reg                          done,
    output reg                          error,
    output reg                          unsupported,
    output reg                          divide_by_zero,
    output reg  [PROG_ADDR_WIDTH-1:0]   pc,
    output reg  [31:0]                  current_instr,
    output reg  [31:0]                  last_instr,
    output reg  [15:0]                  retired_count,

    output reg  [WARP_SIZE-1:0]         last_writeback_mask,
    output reg  [3:0]                   last_writeback_addr,
    output reg  [(WARP_SIZE*WIDTH)-1:0] last_writeback_data
);
    localparam PROG_DEPTH = (1 << PROG_ADDR_WIDTH);
    localparam CONST_DEPTH = (1 << CONST_ADDR_WIDTH);
    localparam WARP_COUNT = NUM_BLOCKS * NUM_WARPS;
    localparam TOTAL_LANES = WARP_COUNT * WARP_SIZE;
    localparam WARP_INDEX_WIDTH = (WARP_COUNT <= 1) ? 1 : clog2(WARP_COUNT);
    localparam STATE_IDLE = 3'd0;
    localparam STATE_CLEAR = 3'd1;
    localparam STATE_SCHEDULE = 3'd2;
    localparam STATE_FETCH = 3'd3;
    localparam STATE_ISSUE = 3'd4;
    localparam STATE_WAIT_DONE = 3'd5;
    localparam STATE_DONE = 3'd6;
    localparam MASK_STACK_DEPTH = 4;
    localparam STACK_ENTRIES = WARP_COUNT * MASK_STACK_DEPTH;

    reg [2:0] state;
    reg [31:0] instr_mem [0:PROG_DEPTH-1];
    reg [WIDTH-1:0] const_mem [0:CONST_DEPTH-1];
    reg [PROG_ADDR_WIDTH-1:0] warp_pc [0:WARP_COUNT-1];
    reg [WARP_SIZE-1:0] active_mask_r [0:WARP_COUNT-1];
    reg [WARP_SIZE-1:0] launch_mask_r [0:WARP_COUNT-1];
    reg [WARP_SIZE-1:0] mask_stack [0:STACK_ENTRIES-1];
    reg [2:0] mask_stack_depth [0:WARP_COUNT-1];
    reg warp_done_r [0:WARP_COUNT-1];
    reg barrier_waiting [0:WARP_COUNT-1];
    reg [WARP_INDEX_WIDTH-1:0] selected_idx;
    reg [WARP_INDEX_WIDTH-1:0] rr_index;
    reg sm_instr_valid;
    reg sm_reset;
    reg [WARP_SIZE-1:0] pending_done_mask;
    reg [WARP_SIZE-1:0] pending_writeback_mask;
    reg [(WARP_SIZE*WIDTH)-1:0] pending_writeback_data;
    reg [3:0] pending_writeback_addr;
    reg [WARP_SIZE-1:0] pending_condition_mask;
    reg pending_unsupported;
    reg pending_divide_by_zero;

    reg [TOTAL_LANES-1:0] active_masks_bus;
    reg [TOTAL_LANES-1:0] sm_mem_req_ready;
    reg [TOTAL_LANES-1:0] sm_mem_resp_valid;
    reg [(TOTAL_LANES*WIDTH)-1:0] sm_mem_resp_rdata;
    reg sched_has_ready;
    reg [WARP_INDEX_WIDTH-1:0] sched_next_idx;
    reg all_warps_finished;

    integer lane_index;
    integer warp_index;
    integer sched_scan;
    integer sched_candidate;
    integer block_loop;
    integer warp_loop;
    integer stack_index;
    integer const_index;

    wire [TOTAL_LANES-1:0] sm_mem_req_valid;
    wire [TOTAL_LANES-1:0] sm_mem_req_write;
    wire [(TOTAL_LANES*ADDR_WIDTH)-1:0] sm_mem_req_addr;
    wire [(TOTAL_LANES*WIDTH)-1:0] sm_mem_req_wdata;
    wire [TOTAL_LANES-1:0] sm_supported_mask;
    wire [TOTAL_LANES-1:0] sm_divide_by_zero_mask;
    wire [TOTAL_LANES-1:0] sm_busy_mask;
    wire [TOTAL_LANES-1:0] sm_done_mask;
    wire [TOTAL_LANES-1:0] sm_writeback_enable_mask;
    wire [(TOTAL_LANES*4)-1:0] sm_writeback_addr;
    wire [(TOTAL_LANES*WIDTH)-1:0] sm_writeback_data;
    wire [NUM_BLOCKS-1:0] block_busy;
    wire [NUM_BLOCKS-1:0] block_done;
    wire sm_busy;
    wire sm_done;

    wire [WARP_SIZE-1:0] selected_active_mask = active_mask_r[selected_idx];
    wire [WARP_SIZE-1:0] selected_done_mask =
        sm_done_mask[(selected_idx*WARP_SIZE) +: WARP_SIZE];
    wire [WARP_SIZE-1:0] selected_supported_mask =
        sm_supported_mask[(selected_idx*WARP_SIZE) +: WARP_SIZE];
    wire [WARP_SIZE-1:0] selected_divide_by_zero_mask =
        sm_divide_by_zero_mask[(selected_idx*WARP_SIZE) +: WARP_SIZE];
    wire [WARP_SIZE-1:0] selected_writeback_enable_mask =
        sm_writeback_enable_mask[(selected_idx*WARP_SIZE) +: WARP_SIZE];
    wire [(WARP_SIZE*4)-1:0] selected_writeback_addr =
        sm_writeback_addr[(selected_idx*WARP_SIZE*4) +: (WARP_SIZE*4)];
    wire [(WARP_SIZE*WIDTH)-1:0] selected_writeback_data =
        sm_writeback_data[(selected_idx*WARP_SIZE*WIDTH) +: (WARP_SIZE*WIDTH)];
    wire [WARP_SIZE-1:0] done_update_mask = selected_done_mask & selected_active_mask;
    wire [WARP_SIZE-1:0] writeback_update_mask =
        selected_writeback_enable_mask & selected_active_mask;
    wire [WARP_SIZE-1:0] done_next_mask = pending_done_mask | done_update_mask;
    wire [WARP_SIZE-1:0] writeback_next_mask =
        pending_writeback_mask | writeback_update_mask;
    wire active_done_next = ((done_next_mask & selected_active_mask) == selected_active_mask);
    wire unsupported_next =
        pending_unsupported | |(done_update_mask & ~selected_supported_mask);
    wire divide_by_zero_next =
        pending_divide_by_zero | |(selected_divide_by_zero_mask & done_update_mask);
    wire [WARP_SIZE-1:0] condition_update_mask =
        condition_result_mask(done_update_mask, selected_writeback_data);
    wire [WARP_SIZE-1:0] condition_next_mask =
        pending_condition_mask | condition_update_mask;
    wire [WIDTH-1:0] current_const_data =
        const_mem[current_instr[CONST_ADDR_WIDTH-1:0]];
    wire [BLOCK_ID_WIDTH-1:0] selected_block_id =
        (selected_idx / NUM_WARPS);
    wire [WARP_ID_WIDTH-1:0] selected_warp_id =
        (selected_idx % NUM_WARPS);

    assign mem_req_valid =
        sm_mem_req_valid[(selected_idx*WARP_SIZE) +: WARP_SIZE];
    assign mem_req_write =
        sm_mem_req_write[(selected_idx*WARP_SIZE) +: WARP_SIZE];
    assign mem_req_addr =
        sm_mem_req_addr[(selected_idx*WARP_SIZE*ADDR_WIDTH) +: (WARP_SIZE*ADDR_WIDTH)];
    assign mem_req_wdata =
        sm_mem_req_wdata[(selected_idx*WARP_SIZE*WIDTH) +: (WARP_SIZE*WIDTH)];

    sm #(
        .WIDTH(WIDTH),
        .WARP_SIZE(WARP_SIZE),
        .NUM_WARPS(NUM_WARPS),
        .NUM_BLOCKS(NUM_BLOCKS),
        .WARP_ID_WIDTH(WARP_ID_WIDTH),
        .BLOCK_ID_WIDTH(BLOCK_ID_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .ENABLE_FLOAT_ADD(ENABLE_FLOAT_ADD),
        .ENABLE_FLOAT_MUL(ENABLE_FLOAT_MUL),
        .ENABLE_FLOAT_DIV(ENABLE_FLOAT_DIV),
        .FLOAT_FP32_ONLY(FLOAT_FP32_ONLY),
        .USE_SHARED_FLOAT(USE_SHARED_FLOAT),
        .SHARED_FLOAT_UNITS(SHARED_FLOAT_UNITS)
    ) simt (
        .clk(clk),
        .rst(rst || sm_reset),
        .instr_valid(sm_instr_valid),
        .issue_block_id(selected_block_id),
        .issue_warp_id(selected_warp_id),
        .instr(current_instr),
        .active_masks(active_masks_bus),
        .base_block_id(base_block_id),
        .block_dim(block_dim),
        .grid_dim(grid_dim),
        .const_data(current_const_data),
        .mem_req_valid(sm_mem_req_valid),
        .mem_req_write(sm_mem_req_write),
        .mem_req_addr(sm_mem_req_addr),
        .mem_req_wdata(sm_mem_req_wdata),
        .mem_req_ready(sm_mem_req_ready),
        .mem_resp_valid(sm_mem_resp_valid),
        .mem_resp_rdata(sm_mem_resp_rdata),
        .supported_mask(sm_supported_mask),
        .divide_by_zero_mask(sm_divide_by_zero_mask),
        .busy_mask(sm_busy_mask),
        .done_mask(sm_done_mask),
        .writeback_enable_mask(sm_writeback_enable_mask),
        .writeback_addr(sm_writeback_addr),
        .writeback_data(sm_writeback_data),
        .block_busy(block_busy),
        .block_done(block_done),
        .sm_busy(sm_busy),
        .sm_done(sm_done)
    );

    always @* begin
        active_masks_bus = {TOTAL_LANES{1'b0}};
        for (warp_index = 0; warp_index < WARP_COUNT; warp_index = warp_index + 1) begin
            active_masks_bus[(warp_index*WARP_SIZE) +: WARP_SIZE] = active_mask_r[warp_index];
        end
    end

    always @* begin
        sm_mem_req_ready = {TOTAL_LANES{1'b0}};
        sm_mem_resp_valid = {TOTAL_LANES{1'b0}};
        sm_mem_resp_rdata = {(TOTAL_LANES*WIDTH){1'b0}};

        sm_mem_req_ready[(selected_idx*WARP_SIZE) +: WARP_SIZE] = mem_req_ready;
        sm_mem_resp_valid[(selected_idx*WARP_SIZE) +: WARP_SIZE] = mem_resp_valid;
        sm_mem_resp_rdata[(selected_idx*WARP_SIZE*WIDTH) +: (WARP_SIZE*WIDTH)] = mem_resp_rdata;
    end

    always @* begin
        all_warps_finished = 1'b1;
        sched_has_ready = 1'b0;
        sched_next_idx = rr_index;

        for (warp_index = 0; warp_index < WARP_COUNT; warp_index = warp_index + 1) begin
            if (!warp_done_r[warp_index]) begin
                all_warps_finished = 1'b0;
            end
        end

        for (sched_scan = 0; sched_scan < WARP_COUNT; sched_scan = sched_scan + 1) begin
            sched_candidate = wrap_index(rr_index, sched_scan);
            if (!sched_has_ready && !warp_done_r[sched_candidate] &&
                !barrier_waiting[sched_candidate]) begin
                sched_next_idx = sched_candidate[WARP_INDEX_WIDTH-1:0];
                sched_has_ready = 1'b1;
            end
        end
    end

    always @(posedge clk) begin
        if (prog_we) begin
            instr_mem[prog_addr] <= prog_wdata;
        end

        if (const_we) begin
            const_mem[const_addr] <= const_wdata;
        end

        if (rst) begin
            state <= STATE_IDLE;
            sm_instr_valid <= 1'b0;
            sm_reset <= 1'b0;
            selected_idx <= {WARP_INDEX_WIDTH{1'b0}};
            rr_index <= {WARP_INDEX_WIDTH{1'b0}};
            busy <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            unsupported <= 1'b0;
            divide_by_zero <= 1'b0;
            pc <= {PROG_ADDR_WIDTH{1'b0}};
            current_instr <= 32'b0;
            last_instr <= 32'b0;
            retired_count <= 16'b0;
            last_writeback_mask <= {WARP_SIZE{1'b0}};
            last_writeback_addr <= 4'b0;
            last_writeback_data <= {(WARP_SIZE*WIDTH){1'b0}};
            pending_done_mask <= {WARP_SIZE{1'b0}};
            pending_writeback_mask <= {WARP_SIZE{1'b0}};
            pending_writeback_data <= {(WARP_SIZE*WIDTH){1'b0}};
            pending_writeback_addr <= 4'b0;
            pending_condition_mask <= {WARP_SIZE{1'b0}};
            pending_unsupported <= 1'b0;
            pending_divide_by_zero <= 1'b0;

            for (warp_index = 0; warp_index < WARP_COUNT; warp_index = warp_index + 1) begin
                warp_pc[warp_index] <= {PROG_ADDR_WIDTH{1'b0}};
                active_mask_r[warp_index] <= {WARP_SIZE{1'b0}};
                launch_mask_r[warp_index] <= {WARP_SIZE{1'b0}};
                mask_stack_depth[warp_index] <= 3'b0;
                warp_done_r[warp_index] <= 1'b1;
                barrier_waiting[warp_index] <= 1'b0;
            end

            for (stack_index = 0; stack_index < STACK_ENTRIES; stack_index = stack_index + 1) begin
                mask_stack[stack_index] <= {WARP_SIZE{1'b0}};
            end

            for (const_index = 0; const_index < CONST_DEPTH; const_index = const_index + 1) begin
                const_mem[const_index] <= {WIDTH{1'b0}};
            end
        end else begin
            sm_instr_valid <= 1'b0;
            sm_reset <= 1'b0;

            case (state)
                STATE_IDLE: begin
                    busy <= 1'b0;
                    if (launch) begin
                        launch_core(base_pc, active_mask, block_dim);
                    end
                end

                STATE_CLEAR: begin
                    state <= STATE_SCHEDULE;
                end

                STATE_SCHEDULE: begin
                    busy <= 1'b1;
                    if (all_warps_finished) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        state <= STATE_DONE;
                    end else if (sched_has_ready) begin
                        selected_idx <= sched_next_idx;
                        pc <= warp_pc[sched_next_idx];
                        state <= STATE_FETCH;
                    end else begin
                        error <= 1'b1;
                        unsupported <= 1'b1;
                        busy <= 1'b0;
                        done <= 1'b1;
                        state <= STATE_DONE;
                    end
                end

                STATE_FETCH: begin
                    pc <= warp_pc[selected_idx];
                    current_instr <= instr_mem[warp_pc[selected_idx]];
                    state <= STATE_ISSUE;
                end

                STATE_ISSUE: begin
                    pc <= warp_pc[selected_idx];
                    if (is_exit(current_instr)) begin
                        retire_no_writeback_keep_last();
                        warp_done_r[selected_idx] <= 1'b1;
                        active_mask_r[selected_idx] <= {WARP_SIZE{1'b0}};
                        barrier_waiting[selected_idx] <= 1'b0;
                        if (barrier_can_release_after_current_done(selected_idx / NUM_WARPS, selected_idx)) begin
                            release_block_barrier(selected_idx / NUM_WARPS, 1'b0);
                        end
                        rr_index <= next_index(selected_idx);
                        state <= STATE_SCHEDULE;
                    end else if (is_bra(current_instr)) begin
                        retire_no_writeback();
                        warp_pc[selected_idx] <= branch_target(warp_pc[selected_idx], current_instr[13:0]);
                        rr_index <= next_index(selected_idx);
                        state <= STATE_SCHEDULE;
                    end else if (is_bar(current_instr)) begin
                        retire_no_writeback();
                        if (barrier_can_release_after_current_bar(selected_idx / NUM_WARPS, selected_idx)) begin
                            release_block_barrier(selected_idx / NUM_WARPS, 1'b1);
                        end else begin
                            barrier_waiting[selected_idx] <= 1'b1;
                        end
                        rr_index <= next_index(selected_idx);
                        state <= STATE_SCHEDULE;
                    end else if (is_pushm(current_instr)) begin
                        retire_no_writeback();
                        if (mask_stack_depth[selected_idx] < MASK_STACK_DEPTH[2:0]) begin
                            mask_stack[(selected_idx*MASK_STACK_DEPTH) + mask_stack_depth[selected_idx]] <=
                                active_mask_r[selected_idx];
                            mask_stack_depth[selected_idx] <= mask_stack_depth[selected_idx] + 3'd1;
                            warp_pc[selected_idx] <= warp_pc[selected_idx] +
                                {{(PROG_ADDR_WIDTH-1){1'b0}}, 1'b1};
                            rr_index <= next_index(selected_idx);
                            state <= STATE_SCHEDULE;
                        end else begin
                            error <= 1'b1;
                            unsupported <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= STATE_DONE;
                        end
                    end else if (is_popm(current_instr)) begin
                        retire_no_writeback();
                        if (mask_stack_depth[selected_idx] != 3'b0) begin
                            active_mask_r[selected_idx] <=
                                mask_stack[(selected_idx*MASK_STACK_DEPTH) +
                                           (mask_stack_depth[selected_idx] - 3'd1)] &
                                launch_mask_r[selected_idx];
                            mask_stack_depth[selected_idx] <= mask_stack_depth[selected_idx] - 3'd1;
                            warp_pc[selected_idx] <= warp_pc[selected_idx] +
                                {{(PROG_ADDR_WIDTH-1){1'b0}}, 1'b1};
                            rr_index <= next_index(selected_idx);
                            state <= STATE_SCHEDULE;
                        end else begin
                            error <= 1'b1;
                            unsupported <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= STATE_DONE;
                        end
                    end else if (active_mask_r[selected_idx] == {WARP_SIZE{1'b0}}) begin
                        retire_no_writeback();
                        warp_pc[selected_idx] <= warp_pc[selected_idx] +
                            {{(PROG_ADDR_WIDTH-1){1'b0}}, 1'b1};
                        rr_index <= next_index(selected_idx);
                        state <= STATE_SCHEDULE;
                    end else if (!sm_busy) begin
                        sm_instr_valid <= 1'b1;
                        pending_done_mask <= {WARP_SIZE{1'b0}};
                        pending_writeback_mask <= {WARP_SIZE{1'b0}};
                        pending_writeback_data <= {(WARP_SIZE*WIDTH){1'b0}};
                        pending_writeback_addr <= 4'b0;
                        pending_condition_mask <= {WARP_SIZE{1'b0}};
                        pending_unsupported <= 1'b0;
                        pending_divide_by_zero <= 1'b0;
                        state <= STATE_WAIT_DONE;
                    end
                end

                STATE_WAIT_DONE: begin
                    pending_done_mask <= done_next_mask;
                    pending_condition_mask <= condition_next_mask;
                    pending_unsupported <= unsupported_next;
                    pending_divide_by_zero <= divide_by_zero_next;

                    if (|writeback_update_mask) begin
                        pending_writeback_mask <= writeback_next_mask;
                        if (pending_writeback_mask == {WARP_SIZE{1'b0}}) begin
                            pending_writeback_addr <=
                                first_writeback_addr(writeback_update_mask, selected_writeback_addr);
                        end
                        for (lane_index = 0; lane_index < WARP_SIZE; lane_index = lane_index + 1) begin
                            if (writeback_update_mask[lane_index]) begin
                                pending_writeback_data[(lane_index*WIDTH) +: WIDTH] <=
                                    selected_writeback_data[(lane_index*WIDTH) +: WIDTH];
                            end
                        end
                    end

                    if (active_done_next) begin
                        last_instr <= current_instr;
                        retired_count <= retired_count + 16'd1;

                        if (|writeback_next_mask) begin
                            last_writeback_mask <= writeback_next_mask;
                            last_writeback_addr <=
                                (pending_writeback_mask == {WARP_SIZE{1'b0}})
                                ? first_writeback_addr(writeback_update_mask, selected_writeback_addr)
                                : pending_writeback_addr;
                            for (lane_index = 0; lane_index < WARP_SIZE; lane_index = lane_index + 1) begin
                                if (writeback_update_mask[lane_index]) begin
                                    last_writeback_data[(lane_index*WIDTH) +: WIDTH] <=
                                        selected_writeback_data[(lane_index*WIDTH) +: WIDTH];
                                end else begin
                                    last_writeback_data[(lane_index*WIDTH) +: WIDTH] <=
                                        pending_writeback_data[(lane_index*WIDTH) +: WIDTH];
                                end
                            end
                        end else begin
                            last_writeback_mask <= {WARP_SIZE{1'b0}};
                            last_writeback_addr <= 4'b0;
                            last_writeback_data <= {(WARP_SIZE*WIDTH){1'b0}};
                        end

                        if (unsupported_next) begin
                            error <= 1'b1;
                            unsupported <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= STATE_DONE;
                        end else if (divide_by_zero_next) begin
                            error <= 1'b1;
                            divide_by_zero <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= STATE_DONE;
                        end else begin
                            if (is_pred(current_instr) || is_predn(current_instr)) begin
                                active_mask_r[selected_idx] <= active_mask_r[selected_idx] & condition_next_mask;
                                warp_pc[selected_idx] <= warp_pc[selected_idx] +
                                    {{(PROG_ADDR_WIDTH-1){1'b0}}, 1'b1};
                            end else if (is_conditional_branch(current_instr)) begin
                                warp_pc[selected_idx] <= (|(condition_next_mask & active_mask_r[selected_idx]))
                                    ? branch_target(warp_pc[selected_idx], current_instr[13:0])
                                    : warp_pc[selected_idx] + {{(PROG_ADDR_WIDTH-1){1'b0}}, 1'b1};
                            end else begin
                                warp_pc[selected_idx] <= warp_pc[selected_idx] +
                                    {{(PROG_ADDR_WIDTH-1){1'b0}}, 1'b1};
                            end
                            rr_index <= next_index(selected_idx);
                            state <= STATE_SCHEDULE;
                        end
                    end
                end

                STATE_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    if (launch) begin
                        launch_core(base_pc, active_mask, block_dim);
                    end
                end

                default: begin
                    state <= STATE_IDLE;
                    busy <= 1'b0;
                    done <= 1'b0;
                end
            endcase
        end
    end

    task launch_core;
        input [PROG_ADDR_WIDTH-1:0] launch_base_pc;
        input [WARP_SIZE-1:0] launch_active_mask;
        input [WIDTH-1:0] launch_block_dim;
        integer launch_block;
        integer launch_warp;
        integer launch_index;
        begin
            selected_idx <= {WARP_INDEX_WIDTH{1'b0}};
            rr_index <= {WARP_INDEX_WIDTH{1'b0}};
            pc <= launch_base_pc;
            current_instr <= 32'b0;
            last_instr <= 32'b0;
            done <= !any_launch_active(launch_active_mask, launch_block_dim);
            busy <= any_launch_active(launch_active_mask, launch_block_dim);
            error <= 1'b0;
            unsupported <= 1'b0;
            divide_by_zero <= 1'b0;
            retired_count <= 16'b0;
            last_writeback_mask <= {WARP_SIZE{1'b0}};
            last_writeback_addr <= 4'b0;
            last_writeback_data <= {(WARP_SIZE*WIDTH){1'b0}};
            pending_done_mask <= {WARP_SIZE{1'b0}};
            pending_writeback_mask <= {WARP_SIZE{1'b0}};
            pending_writeback_data <= {(WARP_SIZE*WIDTH){1'b0}};
            pending_writeback_addr <= 4'b0;
            pending_condition_mask <= {WARP_SIZE{1'b0}};
            pending_unsupported <= 1'b0;
            pending_divide_by_zero <= 1'b0;
            sm_reset <= any_launch_active(launch_active_mask, launch_block_dim);

            for (launch_block = 0; launch_block < NUM_BLOCKS; launch_block = launch_block + 1) begin
                for (launch_warp = 0; launch_warp < NUM_WARPS; launch_warp = launch_warp + 1) begin
                    launch_index = (launch_block * NUM_WARPS) + launch_warp;
                    warp_pc[launch_index] <= launch_base_pc;
                    active_mask_r[launch_index] <=
                        initial_warp_mask(launch_warp, launch_active_mask, launch_block_dim);
                    launch_mask_r[launch_index] <=
                        initial_warp_mask(launch_warp, launch_active_mask, launch_block_dim);
                    mask_stack_depth[launch_index] <= 3'b0;
                    warp_done_r[launch_index] <=
                        (initial_warp_mask(launch_warp, launch_active_mask, launch_block_dim) ==
                         {WARP_SIZE{1'b0}});
                    barrier_waiting[launch_index] <= 1'b0;
                end
            end

            state <= any_launch_active(launch_active_mask, launch_block_dim)
                ? STATE_CLEAR
                : STATE_DONE;
        end
    endtask

    task retire_no_writeback;
        begin
            last_instr <= current_instr;
            retired_count <= retired_count + 16'd1;
            last_writeback_mask <= {WARP_SIZE{1'b0}};
            last_writeback_addr <= 4'b0;
            last_writeback_data <= {(WARP_SIZE*WIDTH){1'b0}};
        end
    endtask

    task retire_no_writeback_keep_last;
        begin
            last_instr <= current_instr;
            retired_count <= retired_count + 16'd1;
        end
    endtask

    task release_block_barrier;
        input integer block_value;
        input include_current;
        integer release_warp;
        integer release_index;
        begin
            for (release_warp = 0; release_warp < NUM_WARPS; release_warp = release_warp + 1) begin
                release_index = (block_value * NUM_WARPS) + release_warp;
                if (!warp_done_r[release_index] &&
                    (barrier_waiting[release_index] ||
                     (include_current && (release_index == selected_idx)))) begin
                    barrier_waiting[release_index] <= 1'b0;
                    warp_pc[release_index] <= warp_pc[release_index] +
                        {{(PROG_ADDR_WIDTH-1){1'b0}}, 1'b1};
                end
            end
        end
    endtask

    function [WARP_SIZE-1:0] initial_warp_mask;
        input integer warp_value;
        input [WARP_SIZE-1:0] launch_active_mask;
        input [WIDTH-1:0] launch_block_dim;
        integer lane;
        integer thread_index;
        begin
            initial_warp_mask = {WARP_SIZE{1'b0}};
            for (lane = 0; lane < WARP_SIZE; lane = lane + 1) begin
                thread_index = (warp_value * WARP_SIZE) + lane;
                if ((thread_index < launch_block_dim) && launch_active_mask[lane]) begin
                    initial_warp_mask[lane] = 1'b1;
                end
            end
        end
    endfunction

    function any_launch_active;
        input [WARP_SIZE-1:0] launch_active_mask;
        input [WIDTH-1:0] launch_block_dim;
        integer any_warp;
        begin
            any_launch_active = 1'b0;
            for (any_warp = 0; any_warp < NUM_WARPS; any_warp = any_warp + 1) begin
                if (initial_warp_mask(any_warp, launch_active_mask, launch_block_dim) !=
                    {WARP_SIZE{1'b0}}) begin
                    any_launch_active = 1'b1;
                end
            end
        end
    endfunction

    function barrier_can_release_after_current_bar;
        input integer block_value;
        input integer current_index;
        integer check_warp;
        integer check_index;
        begin
            barrier_can_release_after_current_bar = 1'b1;
            for (check_warp = 0; check_warp < NUM_WARPS; check_warp = check_warp + 1) begin
                check_index = (block_value * NUM_WARPS) + check_warp;
                if (!warp_done_r[check_index] && (check_index != current_index) &&
                    !barrier_waiting[check_index]) begin
                    barrier_can_release_after_current_bar = 1'b0;
                end
            end
        end
    endfunction

    function barrier_can_release_after_current_done;
        input integer block_value;
        input integer current_index;
        integer check_warp;
        integer check_index;
        begin
            barrier_can_release_after_current_done = 1'b1;
            for (check_warp = 0; check_warp < NUM_WARPS; check_warp = check_warp + 1) begin
                check_index = (block_value * NUM_WARPS) + check_warp;
                if (!warp_done_r[check_index] && (check_index != current_index) &&
                    !barrier_waiting[check_index]) begin
                    barrier_can_release_after_current_done = 1'b0;
                end
            end
        end
    endfunction

    function [WARP_INDEX_WIDTH-1:0] next_index;
        input [WARP_INDEX_WIDTH-1:0] value;
        begin
            if (value == (WARP_COUNT - 1)) begin
                next_index = {WARP_INDEX_WIDTH{1'b0}};
            end else begin
                next_index = value + {{(WARP_INDEX_WIDTH-1){1'b0}}, 1'b1};
            end
        end
    endfunction

    function integer wrap_index;
        input integer base;
        input integer offset;
        integer candidate;
        begin
            candidate = base + offset;
            if (candidate >= WARP_COUNT) begin
                candidate = candidate - WARP_COUNT;
            end
            wrap_index = candidate;
        end
    endfunction

    function is_exit;
        input [31:0] instr;
        begin
            is_exit = (instr[31:26] == `MGPU_OP_EXIT);
        end
    endfunction

    function is_bra;
        input [31:0] instr;
        begin
            is_bra = (instr[31:26] == `MGPU_OP_BRA);
        end
    endfunction

    function is_bar;
        input [31:0] instr;
        begin
            is_bar = (instr[31:26] == `MGPU_OP_BAR);
        end
    endfunction

    function is_pushm;
        input [31:0] instr;
        begin
            is_pushm = (instr[31:26] == `MGPU_OP_PUSHM);
        end
    endfunction

    function is_popm;
        input [31:0] instr;
        begin
            is_popm = (instr[31:26] == `MGPU_OP_POPM);
        end
    endfunction

    function is_pred;
        input [31:0] instr;
        begin
            is_pred = (instr[31:26] == `MGPU_OP_PRED);
        end
    endfunction

    function is_predn;
        input [31:0] instr;
        begin
            is_predn = (instr[31:26] == `MGPU_OP_PREDN);
        end
    endfunction

    function is_conditional_branch;
        input [31:0] instr;
        begin
            is_conditional_branch = (instr[31:26] == `MGPU_OP_BZ) ||
                                    (instr[31:26] == `MGPU_OP_BNZ);
        end
    endfunction

    function [PROG_ADDR_WIDTH-1:0] branch_target;
        input [PROG_ADDR_WIDTH-1:0] current_pc;
        input [13:0] imm14;
        reg signed [14:0] next_pc_signed;
        reg signed [14:0] offset_signed;
        begin
            next_pc_signed = {1'b0, current_pc} + 15'sd1;
            offset_signed = {imm14[13], imm14};
            branch_target = next_pc_signed + offset_signed;
        end
    endfunction

    function [WARP_SIZE-1:0] condition_result_mask;
        input [WARP_SIZE-1:0] valid_mask;
        input [(WARP_SIZE*WIDTH)-1:0] data_bus;
        integer idx;
        begin
            condition_result_mask = {WARP_SIZE{1'b0}};
            for (idx = 0; idx < WARP_SIZE; idx = idx + 1) begin
                if (valid_mask[idx] && data_bus[(idx*WIDTH) +: WIDTH] != {WIDTH{1'b0}}) begin
                    condition_result_mask[idx] = 1'b1;
                end
            end
        end
    endfunction

    function [3:0] first_writeback_addr;
        input [WARP_SIZE-1:0] mask;
        input [(WARP_SIZE*4)-1:0] addr_bus;
        begin
            if (mask[0]) begin
                first_writeback_addr = addr_bus[(0*4) +: 4];
            end else if (mask[1]) begin
                first_writeback_addr = addr_bus[(1*4) +: 4];
            end else if (mask[2]) begin
                first_writeback_addr = addr_bus[(2*4) +: 4];
            end else begin
                first_writeback_addr = addr_bus[(3*4) +: 4];
            end
        end
    endfunction

    function integer clog2;
        input integer value;
        integer shifted;
        begin
            shifted = value - 1;
            for (clog2 = 0; shifted > 0; clog2 = clog2 + 1) begin
                shifted = shifted >> 1;
            end
        end
    endfunction
endmodule
