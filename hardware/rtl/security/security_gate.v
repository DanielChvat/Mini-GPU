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

    // Host memory bus IN (from comm_controller)
    input  wire [3:0]  host_mem_req_valid,
    input  wire [3:0]  host_mem_req_write,
    input  wire [(4*ADDR_WIDTH)-1:0] host_mem_req_addr,
    input  wire [(4*DATA_WIDTH)-1:0] host_mem_req_wdata,
    output wire [3:0]  host_mem_req_ready,
    output wire [3:0]  host_mem_resp_valid,
    output wire [(4*DATA_WIDTH)-1:0] host_mem_resp_rdata,

    // Memory interface OUT (to memory_sec)
    output wire [3:0]  mem_req_valid,
    output wire [3:0]  mem_req_write,
    output wire [(4*ADDR_WIDTH)-1:0] mem_req_addr,
    output wire [(4*DATA_WIDTH)-1:0] mem_req_wdata,
    output wire [(4*4)-1:0] mem_req_wmask,
    input  wire [3:0]  mem_req_ready,
    input  wire [3:0]  mem_resp_valid,
    input  wire [(4*DATA_WIDTH)-1:0] mem_resp_rdata,

    // Validate interface (from comm_controller)
    input  wire        validate_triggered,
    input  wire [3:0]  validate_model_id,

    // Write status (to comm_controller)
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

    // SHA pre-start: pulse sha_start on first cycle after reset so SHA is
    // in S_COLLECT (ready=1) before the first write arrives.
    reg                    sha_needs_start;

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
    wire lane0_write = host_mem_req_valid[0] && host_mem_req_write[0];

    // =========================================================================
    // Data Path: Default Pass-Through with State Overrides
    // =========================================================================
    reg        block_lane0_write;

    assign mem_req_valid  = block_lane0_write ?
        {host_mem_req_valid[3:1], host_mem_req_valid[0] & ~host_mem_req_write[0]} :
        host_mem_req_valid;
    assign mem_req_write  = host_mem_req_write;
    assign mem_req_addr   = host_mem_req_addr;
    assign mem_req_wdata  = host_mem_req_wdata;
    assign mem_req_wmask  = {4{4'b1111}};

    assign host_mem_req_ready[3:1] = mem_req_ready[3:1];
    assign host_mem_req_ready[0]   = block_lane0_write ? 1'b1 : mem_req_ready[0];

    assign host_mem_resp_valid = mem_resp_valid;
    assign host_mem_resp_rdata = mem_resp_rdata;

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
            sha_start        <= 1'b0;
            sha_word_valid   <= 1'b0;
            sha_word_data    <= 32'b0;
            sha_finish       <= 1'b0;
            mem_write_done   <= 1'b0;
            mem_write_fail   <= 1'b0;
            block_lane0_write <= 1'b0;
        end else if (security_reset) begin
            state_reg        <= S_IDLE;
            weight_count_reg <= {ADDR_WIDTH{1'b0}};
            model_id_reg     <= 4'd0;
            locked_reg       <= 1'b0;
            limit_reg        <= {ADDR_WIDTH{1'b0}};
            verified_reg     <= 1'b0;
            fault_reg        <= 1'b0;
            sha_needs_start  <= 1'b1;
            sha_start        <= 1'b0;
            sha_word_valid   <= 1'b0;
            sha_word_data    <= 32'b0;
            sha_finish       <= 1'b0;
            mem_write_done   <= 1'b0;
            mem_write_fail   <= 1'b0;
            block_lane0_write <= 1'b0;
        end else begin

            // Defaults
            sha_start      <= 1'b0;
            sha_word_valid <= 1'b0;
            sha_finish     <= 1'b0;
            mem_write_done <= 1'b0;
            mem_write_fail <= 1'b0;
            block_lane0_write <= 1'b0;

            // Pre-start SHA on first cycle after reset so it is in S_COLLECT
            // (ready=1) before any write arrives. At UART speeds there are
            // thousands of idle cycles between reset and the first write word.
            if (sha_needs_start) begin
                sha_start       <= 1'b1;
                sha_needs_start <= 1'b0;
            end

            case (state_reg)

                // =============================================================
                // S_IDLE: Wait for first weight write
                // =============================================================
                S_IDLE: begin
                    mem_write_done <= 1'b1;

                    if (lane0_write) begin
                        sha_word_valid <= 1'b1;
                        sha_word_data  <= host_mem_req_wdata[0 +: DATA_WIDTH];
                        weight_count_reg <= {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
                        state_reg   <= S_LOAD;
                    end
                end

                // =============================================================
                // S_LOAD: Stream weights to memory + SHA-256
                // =============================================================
                S_LOAD: begin
                    mem_write_done <= 1'b1;

                    if (validate_triggered) begin
                        model_id_reg <= validate_model_id;
                        sha_finish   <= 1'b1;
                        block_lane0_write <= 1'b1;
                        state_reg    <= S_FINALIZE;
                    end else if (lane0_write) begin
                        sha_word_valid <= 1'b1;
                        sha_word_data  <= host_mem_req_wdata[0 +: DATA_WIDTH];
                        weight_count_reg <= weight_count_reg + {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
                    end

                    if (sha_error) begin
                        fault_reg <= 1'b1;
                        block_lane0_write <= 1'b1;
                        state_reg <= S_ERROR;
                    end
                end

                // =============================================================
                // S_FINALIZE: Wait for SHA-256 digest
                // =============================================================
                S_FINALIZE: begin
                    block_lane0_write <= 1'b1;

                    if (sha_digest_valid) begin
                        state_reg <= S_VERIFY;
                    end

                    if (sha_error) begin
                        fault_reg <= 1'b1;
                        state_reg <= S_ERROR;
                    end
                end

                // =============================================================
                // S_VERIFY: Compare digest against golden hash
                // =============================================================
                S_VERIFY: begin
                    block_lane0_write <= 1'b1;

                    if (sha_digest == golden_hash[model_id_reg]) begin
                        state_reg <= S_LOCK;
                    end else begin
                        fault_reg <= 1'b1;
                        state_reg <= S_ERROR;
                    end
                end

                // =============================================================
                // S_LOCK: Apply security policy
                // =============================================================
                S_LOCK: begin
                    locked_reg   <= 1'b1;
                    limit_reg    <= weight_count_reg;
                    verified_reg <= 1'b1;
                    state_reg    <= S_EXECUTE;
                end

                // =============================================================
                // S_EXECUTE: Normal operation with protected weights
                // =============================================================
                S_EXECUTE: begin
                    mem_write_done <= 1'b1;
                end

                // =============================================================
                // S_ERROR: Security violation
                // =============================================================
                S_ERROR: begin
                    block_lane0_write <= 1'b1;
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
