`timescale 1ns/1ps
`default_nettype none

/*
------------------------------------------------------------------------------
sha256_weight_stream
------------------------------------------------------------------------------
Purpose:
    Streaming SHA-256 wrapper around the secworks SHA-256 core.

High-Level Function:
    This module incrementally accepts 32-bit words representing model weights
    (or any authenticated data stream), packs them into 512-bit SHA-256 blocks,
    and feeds them into the secworks sha256_core.

    The design is intended for a secure accelerator loading pipeline where:
        1. Weights are streamed into protected memory
        2. SHA-256 is computed incrementally during load
        3. Final digest is compared against a trusted reference digest
        4. Execution is only allowed if verification succeeds

Architecture Notes:
    - Uses SHA-256 mode only (mode = 1'b1)
    - Uses a streaming/single-pass load architecture
    - Does NOT support random-access updates or rewrites
    - Handles SHA-256 padding internally
    - Supports multi-block messages
    - Resource-optimized for small FPGA devices

Security Semantics:
    - Any protocol violation raises 'error'
    - Data is only accepted during S_COLLECT
    - Incoming data during hash submission or wait states is illegal
    - Final digest becomes valid only after complete SHA finalization

Message Format:
    Input stream is provided as sequential 32-bit words.

    Internally:
        16 x 32-bit words -> 512-bit SHA block

------------------------------------------------------------------------------
FSM Overview
------------------------------------------------------------------------------

IDLE
    Waiting for 'start'

COLLECT
    Accept incoming 32-bit words
    Build 512-bit SHA block

SUBMIT
    Submit 512-bit block to SHA core

WAIT_CORE
    Wait for SHA core digest_valid

------------------------------------------------------------------------------
*/


module sha256_weight_stream (
    input  wire          clk,
    input  wire          rst,
    input  wire          start,             // Start the hashing 
    input  wire          word_valid,        // Indicates incoming 32-bit word is valid
    input  wire [31:0]   word_data,         // Input word stream
    input  wire          finish,            // Indicates final word has already been sent, and hash should be finalized/padded
    output wire          busy,              // High while hashing session active
    output wire          ready,             // High when module ready to accept new word
    output wire [255:0]  digest,            // Final SHA-256 digest
    output wire          digest_valid,      // Pulses high when digest becomes valid
    output wire          error              // Protocol or sequencing error
);

    //--------------------------------------------------------------------------
    // FSM States
    //--------------------------------------------------------------------------
    localparam S_IDLE      = 3'd0;          // Waiting for start signal
    localparam S_COLLECT   = 3'd1;          // Collecting 32-bit words into 512-bit block
    localparam S_SUBMIT    = 3'd2;          // Submitting block into SHA core
    localparam S_WAIT_CORE = 3'd3;          // Waiting for SHA core to finish processing current block


    //--------------------------------------------------------------------------
    // Post-Core Actions
    //
    // Determines what happens AFTER current SHA block completes
    //--------------------------------------------------------------------------
    localparam AFTER_COLLECT    = 2'd0;     // Normal continuation: collect more words
    localparam AFTER_DONE       = 2'd1;     // Hash operation fully complete
    localparam AFTER_SECOND_PAD = 2'd2;     // Need second SHA padding block


    //--------------------------------------------------------------------------
    // Internal Registers
    //--------------------------------------------------------------------------
    reg [2:0] state_reg;               // Current FSM state
    reg [511:0] block_reg;             // Current partially assembled SHA-256 block
    reg [3:0] word_index_reg;          // Current 32-bit word index within block (0–15)
    reg [31:0] total_bits_reg;         // Total number of bits hashed (SHA padding requirement)
    reg first_block_reg;               // Indicates whether next block uses SHA init (first vs continuation)
    reg submit_first_reg;              // Latched INIT/NEXT control for SHA core submission
    reg [1:0] after_core_reg;          // Determines next action after SHA core completes
    reg digest_valid_reg;              // Pulses high when digest becomes valid
    reg error_reg;                     // FSM/protocol violation indicator


    //--------------------------------------------------------------------------
    // SHA Core Interface Signals
    //--------------------------------------------------------------------------
    wire core_ready;
    wire [255:0] core_digest;
    wire core_digest_valid;

    /*
    --------------------------------------------------------------------------
    secworks SHA Core Control

    init:
        Starts NEW hash operation

    next:
        Continues EXISTING hash operation
    --------------------------------------------------------------------------
    */

    wire core_init =
        (state_reg == S_SUBMIT) &&
        core_ready &&
        submit_first_reg;

    wire core_next =
        (state_reg == S_SUBMIT) &&
        core_ready &&
        !submit_first_reg;


    //--------------------------------------------------------------------------
    // Output Assignments
    //--------------------------------------------------------------------------
    assign ready = (state_reg == S_COLLECT);    // Ready only while collecting words
    assign busy = (state_reg != S_IDLE);        // Busy whenever not idle
    assign digest = core_digest;                // Core digest stays stable after completion
    assign digest_valid = digest_valid_reg;     // Digest valid pulse
    assign error = error_reg;                   // Error flag


    //--------------------------------------------------------------------------
    // SHA-256 Core Instance
    //--------------------------------------------------------------------------

    sha256_core core (
        .clk(clk),
        .reset_n(~rst),
        .init(core_init),                   // Start new hash
        .next(core_next),                   // Continue existing hash
        .mode(1'b1),                        // SHA-256 mode
        .block(block_reg),                  // 512-bit message block
        .ready(core_ready),                 // Core ready for new block
        .digest(core_digest),               // Current digest
        .digest_valid(core_digest_valid)    // Digest valid pulse
    );


    //--------------------------------------------------------------------------
    // Main FSM
    //--------------------------------------------------------------------------

    always @(posedge clk or posedge rst) begin
        if (rst) begin

            //------------------------------------------------------------------
            // Global Reset
            //------------------------------------------------------------------
            state_reg <= S_IDLE;
            block_reg <= 512'b0;
            word_index_reg <= 4'd0;
            total_bits_reg <= 32'd0;
            first_block_reg <= 1'b1;
            submit_first_reg <= 1'b1;
            after_core_reg <= AFTER_COLLECT;
            digest_valid_reg <= 1'b0;
            error_reg <= 1'b0;
        end else begin

            //------------------------------------------------------------------
            // Start New Hashing Session
            //------------------------------------------------------------------

            if (start) begin
                state_reg <= S_COLLECT;

                block_reg <= 512'b0;

                word_index_reg <= 4'd0;

                total_bits_reg <= 32'd0;

                first_block_reg <= 1'b1;
                submit_first_reg <= 1'b1;

                after_core_reg <= AFTER_COLLECT;

                digest_valid_reg <= 1'b0;

                error_reg <= 1'b0;

            end else begin

                case (state_reg)

                    //------------------------------------------------------------------
                    // IDLE
                    //------------------------------------------------------------------

                    S_IDLE: begin

                        // Receiving data while idle is illegal
                        if (word_valid || finish) begin
                            error_reg <= 1'b1;
                        end
                    end

                    //------------------------------------------------------------------
                    // COLLECT
                    //
                    // Assemble incoming 32-bit words into 512-bit block
                    //------------------------------------------------------------------

                    S_COLLECT: begin

                        // Simultaneous word_valid + finish is illegal
                        if (word_valid && finish) begin

                            error_reg <= 1'b1;

                        end else if (word_valid) begin

                            //----------------------------------------------------------
                            // Insert incoming word into current SHA block
                            //----------------------------------------------------------

                            block_reg <=
                                block_with_word(
                                    block_reg,
                                    word_index_reg,
                                    word_data
                                );

                            //----------------------------------------------------------
                            // Track total message length
                            //----------------------------------------------------------

                            total_bits_reg <= total_bits_reg + 32'd32;

                            //----------------------------------------------------------
                            // Full 512-bit block complete
                            //----------------------------------------------------------

                            if (word_index_reg == 4'd15) begin

                                // Determine init vs next
                                submit_first_reg <= first_block_reg;

                                // Continue collecting after completion
                                after_core_reg <= AFTER_COLLECT;

                                // Submit to SHA core
                                state_reg <= S_SUBMIT;

                            end else begin

                                // Continue collecting words
                                word_index_reg <= word_index_reg + 4'd1;
                            end

                        end else if (finish) begin

                            //----------------------------------------------------------
                            // Finalize SHA padding block
                            //----------------------------------------------------------

                            block_reg <=
                                final_block(
                                    block_reg,
                                    word_index_reg,
                                    {32'b0, total_bits_reg},
                                    1'b0
                                );

                            submit_first_reg <= first_block_reg;

                            //----------------------------------------------------------
                            // Determine whether second padding block required
                            //
                            // SHA-256 requires:
                            //   - 1-bit padding
                            //   - zero padding
                            //   - 64-bit length field
                            //----------------------------------------------------------

                            if (word_index_reg <= 4'd13) begin

                                // Enough room for length field
                                after_core_reg <= AFTER_DONE;

                            end else begin

                                // Need second padding block
                                after_core_reg <= AFTER_SECOND_PAD;
                            end

                            state_reg <= S_SUBMIT;
                        end
                    end


                    //------------------------------------------------------------------
                    // SUBMIT
                    //
                    // Wait until SHA core ready to accept block
                    //------------------------------------------------------------------

                    S_SUBMIT: begin

                        // Incoming data illegal during submission
                        if (word_valid || finish) begin
                            error_reg <= 1'b1;
                        end

                        if (core_ready) begin

                            // Future blocks use next instead of init
                            first_block_reg <= 1'b0;

                            // Wait for SHA processing completion
                            state_reg <= S_WAIT_CORE;
                        end
                    end


                    //------------------------------------------------------------------
                    // WAIT_CORE
                    //
                    // Wait for SHA core digest_valid
                    //------------------------------------------------------------------

                    S_WAIT_CORE: begin

                        // Incoming data illegal while core busy
                        if (word_valid || finish) begin
                            error_reg <= 1'b1;
                        end

                        if (core_digest_valid) begin

                            //----------------------------------------------------------
                            // Continue collecting additional message data
                            //----------------------------------------------------------

                            if (after_core_reg == AFTER_COLLECT) begin

                                block_reg <= 512'b0;
                                word_index_reg <= 4'd0;
                                state_reg <= S_COLLECT;

                            //----------------------------------------------------------
                            // Need second SHA padding block
                            //----------------------------------------------------------

                            end else if (after_core_reg == AFTER_SECOND_PAD) begin

                                block_reg <=
                                    final_block(
                                        512'b0,
                                        4'd0,
                                        {32'b0, total_bits_reg},
                                        1'b1
                                    );

                                submit_first_reg <= 1'b0;

                                after_core_reg <= AFTER_DONE;

                                state_reg <= S_SUBMIT;

                            //----------------------------------------------------------
                            // Hash operation complete
                            //----------------------------------------------------------

                            end else begin

                                digest_valid_reg <= 1'b1;

                                state_reg <= S_IDLE;
                            end
                        end
                    end


                    //------------------------------------------------------------------
                    // Illegal State Recovery
                    //------------------------------------------------------------------

                    default: begin

                        state_reg <= S_IDLE;

                        error_reg <= 1'b1;
                    end
                endcase
            end
        end
    end


    //--------------------------------------------------------------------------
    // block_with_word
    //
    // Inserts 32-bit word into SHA block
    //
    // SHA blocks are filled MSB-first:
    //     word0 -> bits [511:480]
    //     word15 -> bits [31:0]
    //--------------------------------------------------------------------------

    function [511:0] block_with_word;

        input [511:0] in_block;
        input [3:0] index;
        input [31:0] word;

        reg [511:0] tmp;

        begin

            tmp = in_block;

            case (index)
                4'd0:  tmp[511:480] = word;
                4'd1:  tmp[479:448] = word;
                4'd2:  tmp[447:416] = word;
                4'd3:  tmp[415:384] = word;
                4'd4:  tmp[383:352] = word;
                4'd5:  tmp[351:320] = word;
                4'd6:  tmp[319:288] = word;
                4'd7:  tmp[287:256] = word;
                4'd8:  tmp[255:224] = word;
                4'd9:  tmp[223:192] = word;
                4'd10: tmp[191:160] = word;
                4'd11: tmp[159:128] = word;
                4'd12: tmp[127:96]  = word;
                4'd13: tmp[95:64]   = word;
                4'd14: tmp[63:32]   = word;
                4'd15: tmp[31:0]    = word;
            endcase

            block_with_word = tmp;
            
        end
    endfunction


    //--------------------------------------------------------------------------
    // block_with_length
    //
    // Inserts final 64-bit message length into SHA padding block
    //--------------------------------------------------------------------------

    function [511:0] block_with_length;

        input [511:0] in_block;
        input [63:0] bit_length;

        reg [511:0] tmp;

        begin

            tmp = in_block;

            tmp[63:32] = bit_length[63:32];
            tmp[31:0]  = bit_length[31:0];

            block_with_length = tmp;
        end
    endfunction


    //--------------------------------------------------------------------------
    // final_block
    //
    // Generates SHA-256 final padded block
    //
    // Handles:
    //   - 0x80000000 padding insertion
    //   - optional second padding block
    //   - insertion of 64-bit message length
    //--------------------------------------------------------------------------

    function [511:0] final_block;

        input [511:0] partial_block;

        input [3:0] next_word_index;

        input [63:0] bit_length;

        // Indicates this is second padding block
        input second_block;

        reg [511:0] tmp;

        begin

            if (second_block) begin

                //--------------------------------------------------------------
                // Second padding block:
                // all zeros except final length field
                //--------------------------------------------------------------

                tmp = block_with_length(512'b0, bit_length);

            end else begin

                //--------------------------------------------------------------
                // First/final message block
                //
                // Insert SHA '1' bit padding:
                // 0x80000000
                //--------------------------------------------------------------

                tmp =
                    block_with_word(
                        partial_block,
                        next_word_index,
                        32'h80000000
                    );

                //--------------------------------------------------------------
                // If enough room remains, insert length field
                //--------------------------------------------------------------

                if (next_word_index <= 4'd13) begin

                    tmp = block_with_length(tmp, bit_length);
                end
            end

            final_block = tmp;
        end
    endfunction

endmodule

`default_nettype wire
