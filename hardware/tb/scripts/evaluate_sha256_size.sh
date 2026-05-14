#!/bin/bash

yosys -Q -p "read_verilog \
  hardware/rtl/security/gate/sha256_weight_stream.v \
  hardware/rtl/security/gate/sha256/sha256_core.v \
  hardware/rtl/security/gate/sha256/sha256_w_mem.v \
  hardware/rtl/security/gate/sha256/sha256_k_constants.v 
synth_xilinx -family xc7 -top sha256_weight_stream; stat"

# yosys -Q -p "
# read_verilog \
#   hardware/rtl/security/gate/sha256_weight_stream.v \
#   hardware/rtl/security/gate/sha256/sha256_core.v \
#   hardware/rtl/security/gate/sha256/sha256_w_mem.v \
#   hardware/rtl/security/gate/sha256/sha256_k_constants.v

# synth_xilinx -family xc7 -top sha256_weight_stream
# stat
# " 2>/dev/null | grep -E "===|LUT|FDCE|CARRY4|MUXF|cells"

# yosys -p "
# read_verilog \
#   hardware/rtl/security/gate/sha256_weight_stream.v \
#   hardware/rtl/security/gate/sha256/sha256_core.v \
#   hardware/rtl/security/gate/sha256/sha256_w_mem.v \
#   hardware/rtl/security/gate/sha256/sha256_k_constants.v

# synth_xilinx -family xc7 -top sha256_weight_stream
# stat
# " > ./tmp/yosys_full.log 2>&1

# awk '/=== design hierarchy ===/{flag=1} flag' ./tmp/yosys_full.log