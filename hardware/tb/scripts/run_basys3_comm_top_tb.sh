#!/bin/bash
# set -euo pipefail

# Must be called from the project root directory.

# mkdir -p tmp

iverilog -g2012 \
    -Ihardware/rtl/include \
    -o tmp/basys3_comm_top_tb.vvp \
    hardware/rtl/common/serial_packet_rx.v \
    hardware/rtl/common/serial_packet_tx.v \
    hardware/rtl/common/communication_controller.v \
    hardware/rtl/common/instruction_decode.v \
    hardware/rtl/memory/memory_bank.v \
    hardware/rtl/memory/memory.v \
    hardware/rtl/lane/int/div_mod_iterative.v \
    hardware/rtl/lane/int/mul.v \
    hardware/rtl/lane/float/add_sub.v \
    hardware/rtl/lane/float/mul.v \
    hardware/rtl/lane/float/div.v \
    hardware/rtl/lane/float/shared_fpu.v \
    hardware/rtl/lane/regfile.v \
    hardware/rtl/lane/execute.v \
    hardware/rtl/lane/thread.v \
    hardware/rtl/core/warp.v \
    hardware/rtl/core/block.v \
    hardware/rtl/core/sm.v \
    hardware/rtl/core/mini_gpu_core.v \
    hardware/rtl/core/mini_gpu.v \
    hardware/rtl/top/basys3_comm_top.v \
    hardware/tb/top_tb/basys3_comm_top_tb.sv

vvp tmp/basys3_comm_top_tb.vvp
