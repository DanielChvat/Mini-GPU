`timescale 1ns/1ps
`default_nettype none

module kernel_vfy #(
    parameter ADDR_WIDTH       = 16,
    parameter PROG_ADDR_WIDTH  = 12,
    parameter DATA_WIDTH       = 32,
    parameter NUM_GOLDEN_HASHES = 128,
    parameter KERNEL_ID_WIDTH  = 7
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        security_reset,
    input  wire        security_error,

    // Program write interface IN (from comm_controller)
    input  wire                        cc_prog_we,
    input  wire [ADDR_WIDTH-1:0]       cc_prog_addr,
    input  wire [31:0]                 cc_prog_wdata,

    // Program write interface OUT (to mini_gpu)
    output wire                        gpu_prog_we,
    output wire [PROG_ADDR_WIDTH-1:0]  gpu_prog_addr,
    output wire [31:0]                 gpu_prog_wdata,

    // Launch gating
    input  wire                        cc_launch_valid,
    output wire                        cc_launch_ready,
    output wire                        gpu_launch_valid,

    // Validate interface (from comm_controller)
    input  wire                        validate_triggered,
    input  wire [KERNEL_ID_WIDTH-1:0]  validate_kernel_id,

    // Core status (from mini_gpu)
    input  wire                        core_busy,

    // Status to comm_controller
    output wire                        prog_write_blocked,
    output reg                         validate_done,
    output reg                         validate_fail,

    // Observability
    output wire [2:0]                  kernel_state,
    output wire                        kernel_verified,
    output wire                        kernel_fault
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
    localparam S_RESET    = 3'd7;

    // =========================================================================
    // Internal Registers
    // =========================================================================
    reg [2:0]                  state_reg;
    reg [KERNEL_ID_WIDTH-1:0]  kernel_id_reg;
    reg                        verified_reg;
    reg                        fault_reg;
    reg                        sha_needs_start;
    reg [2:0]                  verify_idx;
    reg                        verify_mismatch;

    // First-word buffering for EXECUTE→LOAD transition
    reg                        first_word_pending;
    reg [31:0]                 first_word_buf;

    // reset zeroization address tracking register
    reg [PROG_ADDR_WIDTH-1:0]  reset_zeroization_addr;
    reg                        reset_addr_hold;

    // =========================================================================
    // SHA-256 Engine
    // =========================================================================
    reg          sha_start;
    reg          sha_word_valid;
    reg  [31:0]  sha_word_data;
    reg          sha_finish;
    reg          sha_reset_pulse;
    wire         sha_busy;
    wire         sha_ready;
    wire [255:0] sha_digest;
    wire         sha_digest_valid;
    wire         sha_error;

    sha256_wrapper sha_engine (
        .clk(clk),
        .rst(rst || security_reset || sha_reset_pulse),
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
    // Golden Hash Lookup
    // =========================================================================
    wire [255:0] golden_hash;
    reg  [31:0]  verify_digest_word;
    reg  [31:0]  verify_golden_word;
    wire         verify_word_mismatch;

    kernel_hash_bram #(
        .NUM_HASHES(NUM_GOLDEN_HASHES),
        .ADDR_WIDTH(KERNEL_ID_WIDTH)
    ) hash_store (
        .clk(clk),
        .addr(kernel_id_reg),
        .hash_out(golden_hash)
    );

    always @* begin
        case (verify_idx)
            3'd0: begin
                verify_digest_word = sha_digest[31:0];
                verify_golden_word = golden_hash[31:0];
            end
            3'd1: begin
                verify_digest_word = sha_digest[63:32];
                verify_golden_word = golden_hash[63:32];
            end
            3'd2: begin
                verify_digest_word = sha_digest[95:64];
                verify_golden_word = golden_hash[95:64];
            end
            3'd3: begin
                verify_digest_word = sha_digest[127:96];
                verify_golden_word = golden_hash[127:96];
            end
            3'd4: begin
                verify_digest_word = sha_digest[159:128];
                verify_golden_word = golden_hash[159:128];
            end
            3'd5: begin
                verify_digest_word = sha_digest[191:160];
                verify_golden_word = golden_hash[191:160];
            end
            3'd6: begin
                verify_digest_word = sha_digest[223:192];
                verify_golden_word = golden_hash[223:192];
            end
            default: begin
                verify_digest_word = sha_digest[255:224];
                verify_golden_word = golden_hash[255:224];
            end
        endcase
    end

    assign verify_word_mismatch = (verify_digest_word != verify_golden_word);

    // =========================================================================
    // Combinational Data Path
    // =========================================================================

    // Program writes pass through in IDLE or LOAD (not locked)
    // Also passes through for the EXECUTE→LOAD first-word transition
    wire execute_accept = (state_reg == S_EXECUTE) && !core_busy && cc_prog_we;
    wire allow_prog_write = (state_reg == S_IDLE) || (state_reg == S_LOAD) || execute_accept;

    assign gpu_prog_we    = allow_prog_write && cc_prog_we || (state_reg == S_RESET);
    assign gpu_prog_addr  = (state_reg == S_RESET) ? reset_zeroization_addr : cc_prog_addr[PROG_ADDR_WIDTH-1:0];
    assign gpu_prog_wdata = (state_reg == S_RESET) ? {32{1'b0}} : cc_prog_wdata;

    // Launch gated by verification
    assign cc_launch_ready = (state_reg == S_EXECUTE) && verified_reg;
    assign gpu_launch_valid = cc_launch_valid && cc_launch_ready;

    // Write blocked when in EXECUTE with core_busy, or in ERROR, or locked
    assign prog_write_blocked = (state_reg == S_EXECUTE && core_busy)
                              || (state_reg == S_ERROR)
                              || (state_reg == S_LOCK);

    // =========================================================================
    // Output Assignments
    // =========================================================================
    assign kernel_state    = state_reg;
    assign kernel_verified = verified_reg;
    assign kernel_fault    = fault_reg;

    // =========================================================================
    // Main FSM
    // =========================================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state_reg          <= S_RESET;
            reset_zeroization_addr <= {PROG_ADDR_WIDTH{1'b0}};
            reset_addr_hold    <= 1'b0;
            kernel_id_reg      <= {KERNEL_ID_WIDTH{1'b0}};
            verified_reg       <= 1'b0;
            fault_reg          <= 1'b0;
            sha_needs_start    <= 1'b1;
            verify_idx         <= 3'b0;
            verify_mismatch    <= 1'b0;
            sha_start          <= 1'b0;
            sha_word_valid     <= 1'b0;
            sha_word_data      <= 32'b0;
            sha_finish         <= 1'b0;
            sha_reset_pulse    <= 1'b0;
            validate_done      <= 1'b0;
            validate_fail      <= 1'b0;
            first_word_pending <= 1'b0;
            first_word_buf     <= 32'b0;
        end else if (security_reset) begin
            state_reg          <= S_RESET;
            reset_zeroization_addr <= {PROG_ADDR_WIDTH{1'b0}};
            reset_addr_hold    <= 1'b0;
            kernel_id_reg      <= {KERNEL_ID_WIDTH{1'b0}};
            verified_reg       <= 1'b0;
            fault_reg          <= 1'b0;
            sha_needs_start    <= 1'b1;
            verify_idx         <= 3'b0;
            verify_mismatch    <= 1'b0;
            sha_start          <= 1'b0;
            sha_word_valid     <= 1'b0;
            sha_word_data      <= 32'b0;
            sha_finish         <= 1'b0;
            sha_reset_pulse    <= 1'b0;
            validate_done      <= 1'b0;
            validate_fail      <= 1'b0;
            first_word_pending <= 1'b0;
            first_word_buf     <= 32'b0;
        end else begin

            // Defaults: pulse signals clear each cycle
            sha_start      <= 1'b0;
            sha_word_valid <= 1'b0;
            sha_finish     <= 1'b0;
            sha_reset_pulse <= 1'b0;
            validate_done  <= 1'b0;
            validate_fail  <= 1'b0;

            // SHA initialization after reset
            if (sha_needs_start) begin
                sha_start       <= 1'b1;
                sha_needs_start <= 1'b0;
            end

            if (security_error) begin
                fault_reg <= 1'b1;
                state_reg <= S_ERROR;
            end else begin
                if (validate_triggered && (state_reg != S_LOAD)) begin
                    validate_fail <= 1'b1;
                end

                case (state_reg)

                S_IDLE: begin
                    if (cc_prog_we) begin
                        sha_word_valid <= 1'b1;
                        sha_word_data  <= cc_prog_wdata;
                        state_reg      <= S_LOAD;
                    end
                end

                S_LOAD: begin
                    // Feed buffered first word from EXECUTE→LOAD transition
                    if (first_word_pending && sha_ready) begin
                        sha_word_valid     <= 1'b1;
                        sha_word_data      <= first_word_buf;
                        first_word_pending <= 1'b0;
                    end
                    // Normal word streaming (only if no buffered word pending
                    // and SHA is ready to accept)
                    else if (cc_prog_we && !first_word_pending && sha_ready) begin
                        sha_word_valid <= 1'b1;
                        sha_word_data  <= cc_prog_wdata;
                    end
                    // go to error state if sha is not ready to accept data but we have a new word (and no buffered word)
                    else if (cc_prog_we && !first_word_pending && !sha_ready) begin
                        fault_reg <= 1'b1;
                        state_reg <= S_ERROR;
                    end

                    if (validate_triggered) begin
                        kernel_id_reg <= validate_kernel_id;
                        sha_finish    <= 1'b1;
                        state_reg     <= S_FINALIZE;
                    end

                    if (sha_error) begin
                        fault_reg <= 1'b1;
                        state_reg <= S_ERROR;
                    end
                end

                S_FINALIZE: begin
                    if (sha_digest_valid) begin
                        verify_idx      <= 3'b0;
                        verify_mismatch <= 1'b0;
                        state_reg <= S_VERIFY;
                    end

                    if (sha_error) begin
                        fault_reg     <= 1'b1;
                        validate_fail <= 1'b1;
                        state_reg     <= S_ERROR;
                    end
                end

                S_VERIFY: begin
                    //future design note: because the hash address get's checked at least 2 cycles after kernel_id_reg, it should be enough time to get the correct golden_hash, but there is a possibility that the memory read takes too long and we get an invalid golden_hash. In that case, the verification will fail, but it will be a false negative. To mitigate this, we could add a state to wait for the golden_hash to be valid, or we could add a timeout mechanism.
                    if (verify_word_mismatch) begin
                        verify_mismatch <= 1'b1;
                    end

                    if (&verify_idx) begin
                        if (verify_mismatch || verify_word_mismatch) begin
                            fault_reg     <= 1'b1;
                            validate_fail <= 1'b1;
                            state_reg     <= S_ERROR;
                        end else begin
                            validate_done <= 1'b1;
                            state_reg     <= S_LOCK;
                        end
                    end else begin
                        verify_idx <= verify_idx + 1'b1;
                    end
                end

                S_LOCK: begin
                    verified_reg       <= 1'b1;
                    state_reg          <= S_EXECUTE;
                end

                S_EXECUTE: begin

                    if (cc_prog_we) begin
                        if (!core_busy) begin
                            // EXECUTE→LOAD transition
                            sha_reset_pulse    <= 1'b1;
                            sha_needs_start    <= 1'b1;
                            verified_reg       <= 1'b0;
                            first_word_pending <= 1'b1;
                            first_word_buf     <= cc_prog_wdata;
                            state_reg          <= S_LOAD;
                        end
                        // core_busy case: write is blocked combinationally via prog_write_blocked
                    end
                end

                // S_ERROR: begin
                //     if (validate_triggered) begin
                //         validate_fail <= 1'b1;
                //     end
                // end

                S_RESET: begin
                    reset_addr_hold <= !reset_addr_hold;

                    if (reset_addr_hold) begin
                        if (&reset_zeroization_addr) begin
                            reset_zeroization_addr <= {PROG_ADDR_WIDTH{1'b0}};
                            state_reg <= S_IDLE;
                        end else begin
                            reset_zeroization_addr <= reset_zeroization_addr + 1'b1;
                        end
                    end
                end

                default: begin
                    state_reg <= S_ERROR;
                    fault_reg <= 1'b1;
                end

                endcase
            end
        end
    end

endmodule

`default_nettype wire