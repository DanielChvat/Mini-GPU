`timescale 1ns/1ps

module mini_gpu_program_tb;
    parameter integer WIDTH = 32;
    parameter integer WARP_SIZE = 4;
    parameter integer NUM_CORES = 2;
    parameter integer NUM_WARPS_PER_CORE = 1;
    parameter integer WARP_ID_WIDTH = 1;
    parameter integer PROG_ADDR_WIDTH = 8;
    parameter integer ADDR_WIDTH = 16;
    parameter integer CONST_ADDR_WIDTH = 8;
    parameter integer MEMORY_BANK_DEPTH = 64;

    localparam integer PROG_DEPTH = (1 << PROG_ADDR_WIDTH);
    localparam integer PATH_WIDTH = 1024;

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

    reg [31:0] program_image [0:PROG_DEPTH-1];
    reg [PATH_WIDTH-1:0] program_path;
    reg [PATH_WIDTH-1:0] const_path;
    reg [PATH_WIDTH-1:0] mem_init_path;
    reg [PATH_WIDTH-1:0] mem_expect_path;
    integer base_pc_arg;
    integer block_dim_arg;
    integer grid_dim_arg;
    integer timeout_cycles;
    integer trace_enabled;

    mini_gpu #(
        .WIDTH(WIDTH),
        .WARP_SIZE(WARP_SIZE),
        .NUM_CORES(NUM_CORES),
        .NUM_WARPS_PER_CORE(NUM_WARPS_PER_CORE),
        .WARP_ID_WIDTH(WARP_ID_WIDTH),
        .PROG_ADDR_WIDTH(PROG_ADDR_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .CONST_ADDR_WIDTH(CONST_ADDR_WIDTH),
        .MEMORY_BANK_DEPTH(MEMORY_BANK_DEPTH)
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
        .block_dim(block_dim),
        .grid_dim(grid_dim),
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

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task clear_program_image;
        integer idx;
        begin
            for (idx = 0; idx < PROG_DEPTH; idx = idx + 1) begin
                program_image[idx[PROG_ADDR_WIDTH-1:0]] = 32'hf0000000;
            end
        end
    endtask

    task load_instr;
        input [PROG_ADDR_WIDTH-1:0] addr;
        input [31:0] instr;
        begin
            @(negedge clk);
            prog_addr = addr;
            prog_wdata = instr;
            prog_we = 1'b1;
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
            @(negedge clk);
            const_we = 1'b0;
        end
    endtask

    task load_program_file;
        input [PATH_WIDTH-1:0] path;
        integer idx;
        begin
            clear_program_image();
            $display("  READ program: %0s", path);
            $readmemh(path, program_image);
            for (idx = 0; idx < PROG_DEPTH; idx = idx + 1) begin
                load_instr(idx[PROG_ADDR_WIDTH-1:0],
                           program_image[idx[PROG_ADDR_WIDTH-1:0]]);
            end
        end
    endtask

    task load_const_file;
        input [PATH_WIDTH-1:0] path;
        integer file;
        integer code;
        reg [31:0] file_addr;
        reg [WIDTH-1:0] file_data;
        begin
            file = $fopen(path, "r");
            if (file == 0) begin
                $display("FAIL could not open const file: %0s", path);
                $finish;
            end

            $display("  READ consts:  %0s", path);
            while (!$feof(file)) begin
                code = $fscanf(file, "%h %h\n", file_addr, file_data);
                if (code == 2) begin
                    load_const(file_addr[CONST_ADDR_WIDTH-1:0], file_data);
                end else if (code != -1) begin
                    $display("FAIL malformed const file entry in %0s", path);
                    $finish;
                end
            end
            $fclose(file);
        end
    endtask

    task seed_memory_word;
        input [ADDR_WIDTH-1:0] addr;
        input [WIDTH-1:0] data;
        reg [ADDR_WIDTH-3:0] index;
        begin
            index = addr[ADDR_WIDTH-1:2];
            case (addr[1:0])
                2'd0: dut.global_memory.bank0.mem[index] = data;
                2'd1: dut.global_memory.bank1.mem[index] = data;
                2'd2: dut.global_memory.bank2.mem[index] = data;
                default: dut.global_memory.bank3.mem[index] = data;
            endcase
            if (trace_enabled) begin
                $display("  SEED mem[%0d] = 0x%08h (%0d)", addr, data, data);
            end
        end
    endtask

    task load_memory_file;
        input [PATH_WIDTH-1:0] path;
        integer file;
        integer code;
        reg [31:0] file_addr;
        reg [WIDTH-1:0] file_data;
        begin
            file = $fopen(path, "r");
            if (file == 0) begin
                $display("FAIL could not open memory init file: %0s", path);
                $finish;
            end

            $display("  READ memory:  %0s", path);
            while (!$feof(file)) begin
                code = $fscanf(file, "%h %h\n", file_addr, file_data);
                if (code == 2) begin
                    seed_memory_word(file_addr[ADDR_WIDTH-1:0], file_data);
                end else if (code != -1) begin
                    $display("FAIL malformed memory init entry in %0s", path);
                    $finish;
                end
            end
            $fclose(file);
        end
    endtask

    function [WIDTH-1:0] read_memory_word;
        input [ADDR_WIDTH-1:0] addr;
        reg [ADDR_WIDTH-3:0] index;
        begin
            index = addr[ADDR_WIDTH-1:2];
            case (addr[1:0])
                2'd0: read_memory_word = dut.global_memory.bank0.mem[index];
                2'd1: read_memory_word = dut.global_memory.bank1.mem[index];
                2'd2: read_memory_word = dut.global_memory.bank2.mem[index];
                default: read_memory_word = dut.global_memory.bank3.mem[index];
            endcase
        end
    endfunction

    task expect_memory_word;
        input [ADDR_WIDTH-1:0] addr;
        input [WIDTH-1:0] expected;
        reg [WIDTH-1:0] got;
        begin
            got = read_memory_word(addr);
            $display("  CHECK mem[%0d] got=0x%08h (%0d) expected=0x%08h (%0d)",
                     addr, got, got, expected, expected);
            if (got !== expected) begin
                $display("FAIL mem[%0d] got=0x%08h expected=0x%08h", addr, got, expected);
                $finish;
            end
        end
    endtask

    task check_memory_file;
        input [PATH_WIDTH-1:0] path;
        integer file;
        integer code;
        reg [31:0] file_addr;
        reg [WIDTH-1:0] file_data;
        begin
            file = $fopen(path, "r");
            if (file == 0) begin
                $display("FAIL could not open memory expected file: %0s", path);
                $finish;
            end

            $display("  READ expect:  %0s", path);
            while (!$feof(file)) begin
                code = $fscanf(file, "%h %h\n", file_addr, file_data);
                if (code == 2) begin
                    expect_memory_word(file_addr[ADDR_WIDTH-1:0], file_data);
                end else if (code != -1) begin
                    $display("FAIL malformed memory expected entry in %0s", path);
                    $finish;
                end
            end
            $fclose(file);
        end
    endtask

    task print_core_status;
        integer core;
        begin
            for (core = 0; core < NUM_CORES; core = core + 1) begin
                $display("  core%0d done=%0d error=%0d unsupported=%0d div0=%0d retired=%0d pc=%0d",
                         core,
                         core_done[core],
                         core_error[core],
                         core_unsupported[core],
                         core_divide_by_zero[core],
                         core_retired_count[(core*16) +: 16],
                         core_pc[(core*PROG_ADDR_WIDTH) +: PROG_ADDR_WIDTH]);
            end
        end
    endtask

    initial begin
        integer cycles;

        rst = 1'b1;
        prog_we = 1'b0;
        prog_addr = {PROG_ADDR_WIDTH{1'b0}};
        prog_wdata = 32'b0;
        const_we = 1'b0;
        const_addr = {CONST_ADDR_WIDTH{1'b0}};
        const_wdata = {WIDTH{1'b0}};
        launch = 1'b0;
        base_pc = {PROG_ADDR_WIDTH{1'b0}};
        active_mask = {WARP_SIZE{1'b1}};
        block_dim = WARP_SIZE * NUM_WARPS_PER_CORE;
        grid_dim = NUM_CORES;
        timeout_cycles = 1000;
        trace_enabled = 0;

        program_path = "hardware/tb/programs/vector_add.hex";
        const_path = "hardware/tb/programs/vector_add.const";
        mem_init_path = "hardware/tb/programs/vector_add.mem";
        mem_expect_path = "hardware/tb/programs/vector_add.expect";

        if ($value$plusargs("program_hex=%s", program_path) ||
            $value$plusargs("program=%s", program_path)) begin
        end
        if ($value$plusargs("const_file=%s", const_path)) begin
        end
        if ($value$plusargs("mem_init=%s", mem_init_path)) begin
        end
        if ($value$plusargs("mem_expect=%s", mem_expect_path)) begin
        end
        if ($value$plusargs("base_pc=%d", base_pc_arg)) begin
            base_pc = base_pc_arg[PROG_ADDR_WIDTH-1:0];
        end
        if ($value$plusargs("block_dim=%d", block_dim_arg)) begin
            block_dim = block_dim_arg[WIDTH-1:0];
        end
        if ($value$plusargs("grid_dim=%d", grid_dim_arg)) begin
            grid_dim = grid_dim_arg[WIDTH-1:0];
        end
        if ($value$plusargs("timeout=%d", timeout_cycles)) begin
        end
        trace_enabled = $test$plusargs("trace");

        repeat (2) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        $display("");
        $display("mini_gpu_program_tb");
        $display("  cores=%0d warps/core=%0d warp_size=%0d block_dim=%0d grid_dim=%0d",
                 NUM_CORES, NUM_WARPS_PER_CORE, WARP_SIZE, block_dim, grid_dim);

        load_program_file(program_path);
        if (!$test$plusargs("no_const_file")) begin
            load_const_file(const_path);
        end
        if (!$test$plusargs("no_mem_init")) begin
            load_memory_file(mem_init_path);
        end

        $display("--- LAUNCH ---");
        @(negedge clk);
        launch = 1'b1;
        @(negedge clk);
        launch = 1'b0;

        cycles = 0;
        while (!done && cycles < timeout_cycles) begin
            cycles = cycles + 1;
            @(negedge clk);
        end

        if (!done) begin
            $display("FAIL program did not finish within %0d cycles", timeout_cycles);
            print_core_status();
            $finish;
        end

        print_core_status();
        if (error || unsupported || divide_by_zero) begin
            $display("FAIL program status error=%0d unsupported=%0d div0=%0d",
                     error, unsupported, divide_by_zero);
            $finish;
        end

        if (!$test$plusargs("no_mem_expect")) begin
            check_memory_file(mem_expect_path);
        end

        $display("mini_gpu_program_tb PASS cycles=%0d", cycles);
        $finish;
    end
endmodule
