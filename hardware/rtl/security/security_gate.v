`timescale 1ns/1ps
`default_nettype none

module security_gate #(
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32,
    parameter NUM_GOLDEN_HASHES = 4
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        security_reset,

    // Merged memory bus IN (from bus_controller)
    input  wire [3:0]  merged_mem_req_valid,
    input  wire [3:0]  merged_mem_req_write,
    input  wire [(4*ADDR_WIDTH)-1:0] merged_mem_req_addr,
    input  wire [(4*DATA_WIDTH)-1:0] merged_mem_req_wdata,
    output wire [3:0]  merged_mem_req_ready,
    output wire [3:0]  merged_mem_resp_valid,
    output wire [(4*DATA_WIDTH)-1:0] merged_mem_resp_rdata,

    // Request origin from bus_controller
    input  wire [1:0]  memory_request_id,

    // Memory interface OUT (to memory_sec)
    output wire [3:0]  mem_req_valid,
    output wire [3:0]  mem_req_write,
    output wire [(4*ADDR_WIDTH)-1:0] mem_req_addr,
    output wire [(4*DATA_WIDTH)-1:0] mem_req_wdata,
    output wire [(4*4)-1:0] mem_req_wmask,
    input  wire [3:0]  mem_req_ready,
    input  wire [3:0]  mem_resp_valid,
    input  wire [(4*DATA_WIDTH)-1:0] mem_resp_rdata,
    input  wire [3:0]  write_blocked,

    // Validate interface (from comm_controller, direct)
    input  wire        validate_triggered,
    input  wire [3:0]  validate_model_id,

    // Write status (to bus_controller for demux)
    output reg         mem_write_done,
    output reg         mem_write_fail,
    input  wire        memory_status_consumed,

    // Security control (drives memory_sec)
    output wire        weight_locked,
    output wire [ADDR_WIDTH-1:0] protected_base,
    output wire [ADDR_WIDTH-1:0] protected_limit,

    // Observability
    output wire [2:0]  security_state,
    output wire        verified,
    output wire        security_fault
);

    // =========================================================================
    // FSM States
    // =========================================================================
    localparam S_IDLE     = 3'd0;
    localparam S_LOAD     = 3'd1;
    localparam S_FINALIZE = 3'd2;
    localparam S_VERIFY   = 3'd3;
    localparam S_LOCK     = 3'd4;
    localparam S_EXECUTE  = 3'd5;
    localparam S_ERROR    = 3'd6;

    // Request ID constants (must match bus_controller)
    localparam REQ_HOST  = 2'd0;
    localparam REQ_CORE  = 2'd1;
    localparam REQ_SEC   = 2'd2;
    localparam REQ_DEBUG = 2'd3;

    // =========================================================================
    // Internal Registers
    // =========================================================================
    reg [2:0]              state_reg;
    reg [ADDR_WIDTH-1:0]   weight_count_reg;
    reg [3:0]              model_id_reg;
    reg                    locked_reg;
    reg [ADDR_WIDTH-1:0]   limit_reg;
    reg                    verified_reg;
    reg                    fault_reg;
    reg                    sha_needs_start;
    reg                    write_blocked_latch;

    // =========================================================================
    // Golden Hash Storage
    // =========================================================================
    reg [255:0] golden_hash [0:NUM_GOLDEN_HASHES-1];

    initial begin
        golden_hash[0] = 256'h0000000000000000000000000000000000000000000000000000000000000000;
        golden_hash[1] = 256'h0000000000000000000000000000000000000000000000000000000000000000;
        golden_hash[2] = 256'h0000000000000000000000000000000000000000000000000000000000000000;
        golden_hash[3] = 256'h0000000000000000000000000000000000000000000000000000000000000000;
    end

    // =========================================================================
    // SHA-256 Engine
    // =========================================================================
    reg          sha_start;
    reg          sha_word_valid;
    reg  [31:0]  sha_word_data;
    reg          sha_finish;
    wire         sha_busy;
    wire         sha_ready;
    wire [255:0] sha_digest;
    wire         sha_digest_valid;
    wire         sha_error;

    sha256_weight_stream sha_engine (
        .clk(clk),
        .rst(rst || security_reset),
        .start(sha_start),
        .word_valid(sha_word_valid),
        .word_data(sha_word_data),
        .finish(sha_finish),
        .busy(sha_busy),
        .ready(sha_ready),
        .digest(sha_digest),
        .digest_valid(sha_digest_valid),
        .error(sha_error)
    );

    // =========================================================================
    // Lane-0 Decode
    // =========================================================================
    wire lane0_write = merged_mem_req_valid[0] && merged_mem_req_write[0];

    // =========================================================================
    // Protected Region Read Detection (per lane)
    // =========================================================================
    wire [3:0] lane_is_read;
    wire [3:0] lane_in_protected;
    wire [3:0] block_read;

    genvar gi;
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : read_check
            wire [ADDR_WIDTH-1:0] lane_addr = merged_mem_req_addr[gi*ADDR_WIDTH +: ADDR_WIDTH];

            assign lane_is_read[gi] = merged_mem_req_valid[gi] && !merged_mem_req_write[gi];

            // In LOCK/EXECUTE: region is [0, limit_reg) with locked_reg=1
            // In ERROR: region is [0, weight_count_reg) even though locked_reg=0
            assign lane_in_protected[gi] =
                (locked_reg && (lane_addr < limit_reg)) ||
                (state_reg == S_ERROR && (lane_addr < weight_count_reg));

            // Block host reads to protected region in LOCK/EXECUTE/ERROR
            // Block core reads to protected region only in ERROR
            assign block_read[gi] = lane_is_read[gi] && lane_in_protected[gi] && (
                ((memory_request_id == REQ_HOST) &&
                 (state_reg == S_LOCK || state_reg == S_EXECUTE || state_reg == S_ERROR)) ||
                ((memory_request_id == REQ_CORE) &&
                 (state_reg == S_ERROR))
            );
        end
    endgenerate

    // =========================================================================
    // Data Path: Merged bus → memory_sec with read interception
    // Write protection is handled by memory_sec (locked + protected region).
    // =========================================================================

    // Suppress blocked reads from reaching memory; writes pass through to memory_sec
    assign mem_req_valid = {
        block_read[3] ? 1'b0 : merged_mem_req_valid[3],
        block_read[2] ? 1'b0 : merged_mem_req_valid[2],
        block_read[1] ? 1'b0 : merged_mem_req_valid[1],
        block_read[0] ? 1'b0 : merged_mem_req_valid[0]
    };

    assign mem_req_write  = merged_mem_req_write;
    assign mem_req_addr   = merged_mem_req_addr;
    assign mem_req_wdata  = merged_mem_req_wdata;
    assign mem_req_wmask  = {4{4'b1111}};

    // Intercepted reads appear instantly ready; everything else from memory
    assign merged_mem_req_ready = {
        block_read[3] ? 1'b1 : mem_req_ready[3],
        block_read[2] ? 1'b1 : mem_req_ready[2],
        block_read[1] ? 1'b1 : mem_req_ready[1],
        block_read[0] ? 1'b1 : mem_req_ready[0]
    };

    // Response: intercepted reads return 0xDEADDEAD, otherwise pass through from memory
    assign merged_mem_resp_valid[0] = block_read[0] ? 1'b1 : mem_resp_valid[0];
    assign merged_mem_resp_valid[1] = block_read[1] ? 1'b1 : mem_resp_valid[1];
    assign merged_mem_resp_valid[2] = block_read[2] ? 1'b1 : mem_resp_valid[2];
    assign merged_mem_resp_valid[3] = block_read[3] ? 1'b1 : mem_resp_valid[3];

    assign merged_mem_resp_rdata[0*DATA_WIDTH +: DATA_WIDTH] =
        block_read[0] ? 32'hDEADDEAD : mem_resp_rdata[0*DATA_WIDTH +: DATA_WIDTH];
    assign merged_mem_resp_rdata[1*DATA_WIDTH +: DATA_WIDTH] =
        block_read[1] ? 32'hDEADDEAD : mem_resp_rdata[1*DATA_WIDTH +: DATA_WIDTH];
    assign merged_mem_resp_rdata[2*DATA_WIDTH +: DATA_WIDTH] =
        block_read[2] ? 32'hDEADDEAD : mem_resp_rdata[2*DATA_WIDTH +: DATA_WIDTH];
    assign merged_mem_resp_rdata[3*DATA_WIDTH +: DATA_WIDTH] =
        block_read[3] ? 32'hDEADDEAD : mem_resp_rdata[3*DATA_WIDTH +: DATA_WIDTH];

    // =========================================================================
    // Output Assignments
    // =========================================================================
    assign security_state = state_reg;
    assign weight_locked  = locked_reg;
    assign protected_base = {ADDR_WIDTH{1'b0}};
    assign protected_limit = limit_reg;
    assign verified       = verified_reg;
    assign security_fault = fault_reg;

    // =========================================================================
    // Main FSM
    // =========================================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state_reg        <= S_IDLE;
            weight_count_reg <= {ADDR_WIDTH{1'b0}};
            model_id_reg     <= 4'd0;
            locked_reg       <= 1'b0;
            limit_reg        <= {ADDR_WIDTH{1'b0}};
            verified_reg     <= 1'b0;
            fault_reg        <= 1'b0;
            sha_needs_start  <= 1'b1;
            sha_start           <= 1'b0;
            sha_word_valid      <= 1'b0;
            sha_word_data       <= 32'b0;
            sha_finish          <= 1'b0;
            mem_write_done      <= 1'b0;
            mem_write_fail      <= 1'b0;
            write_blocked_latch <= 1'b0;
        end else if (security_reset) begin
            state_reg           <= S_IDLE;
            weight_count_reg    <= {ADDR_WIDTH{1'b0}};
            model_id_reg        <= 4'd0;
            locked_reg          <= 1'b0;
            limit_reg           <= {ADDR_WIDTH{1'b0}};
            verified_reg        <= 1'b0;
            fault_reg           <= 1'b0;
            sha_needs_start     <= 1'b1;
            sha_start           <= 1'b0;
            sha_word_valid      <= 1'b0;
            sha_word_data       <= 32'b0;
            sha_finish          <= 1'b0;
            mem_write_done      <= 1'b0;
            mem_write_fail      <= 1'b0;
            write_blocked_latch <= 1'b0;
        end else begin

            // Defaults
            sha_start      <= 1'b0;
            sha_word_valid <= 1'b0;
            sha_finish     <= 1'b0;

            // Pre-start SHA on first cycle after reset
            if (sha_needs_start) begin
                sha_start       <= 1'b1;
                sha_needs_start <= 1'b0;
            end

            // Accumulate write-blocked events from memory_sec.
            // Cleared when the comm_controller consumes the status.
            if (|write_blocked) begin
                write_blocked_latch <= 1'b1;
            end
            if (memory_status_consumed) begin
                write_blocked_latch <= 1'b0;
                mem_write_done      <= 1'b0;
                mem_write_fail      <= 1'b0;
            end

            case (state_reg)

                S_IDLE: begin
                    mem_write_done <= 1'b1;
                    mem_write_fail <= 1'b0;

                    if (lane0_write && (memory_request_id == REQ_HOST)) begin
                        sha_word_valid <= 1'b1;
                        sha_word_data  <= merged_mem_req_wdata[0 +: DATA_WIDTH];
                        weight_count_reg <= {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
                        state_reg   <= S_LOAD;
                    end
                end

                S_LOAD: begin
                    mem_write_done <= 1'b1;
                    mem_write_fail <= 1'b0;

                    if (validate_triggered) begin
                        model_id_reg <= validate_model_id;
                        sha_finish   <= 1'b1;
                        state_reg    <= S_FINALIZE;
                    end else if (lane0_write && (memory_request_id == REQ_HOST)) begin
                        sha_word_valid <= 1'b1;
                        sha_word_data  <= merged_mem_req_wdata[0 +: DATA_WIDTH];
                        weight_count_reg <= weight_count_reg + {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
                    end

                    if (sha_error) begin
                        fault_reg <= 1'b1;
                        state_reg <= S_ERROR;
                    end
                end

                S_FINALIZE: begin
                    mem_write_done <= 1'b0;
                    mem_write_fail <= 1'b0;

                    if (sha_digest_valid) begin
                        state_reg <= S_VERIFY;
                    end

                    if (sha_error) begin
                        fault_reg <= 1'b1;
                        state_reg <= S_ERROR;
                    end
                end

                S_VERIFY: begin
                    mem_write_done <= 1'b0;
                    mem_write_fail <= 1'b0;

                    if (sha_digest == golden_hash[model_id_reg]) begin
                        state_reg <= S_LOCK;
                    end else begin
                        fault_reg <= 1'b1;
                        state_reg <= S_ERROR;
                    end
                end

                S_LOCK: begin
                    mem_write_done <= 1'b0;
                    mem_write_fail <= 1'b0;
                    locked_reg     <= 1'b1;
                    limit_reg      <= weight_count_reg;
                    verified_reg   <= 1'b1;
                    state_reg      <= S_EXECUTE;
                end

                S_EXECUTE: begin
                    mem_write_done <= !write_blocked_latch;
                    mem_write_fail <= write_blocked_latch;
                end

                S_ERROR: begin
                    mem_write_done <= 1'b0;
                    mem_write_fail <= 1'b1;
                end

                default: begin
                    state_reg <= S_ERROR;
                    fault_reg <= 1'b1;
                end

            endcase
        end
    end

endmodule

`default_nettype wire
