#!/bin/bash

# to run this command I had to make changes to the files, where it used `include "minigpu_isa.vh"` as well as making a change to the memory file
# the supported files can be found in the size-evaluation branch

yosys -Q -p "read_verilog \
    hardware/rtl/common/*.v \
    hardware/rtl/include/*.vh \
    hardware/rtl/core/*.v \
    hardware/rtl/lane/*.v \
    hardware/rtl/lane/float/*.v \
    hardware/rtl/lane/int/*.v \
    hardware/rtl/memory/*.v \
    hardware/rtl/security/gate/sha256_weight_stream.v \
    hardware/rtl/security/gate/sha256/*v \
    hardware/rtl/security/gate/kernel_vfy/*.v \
    hardware/rtl/top/basys3_security_top.v; \
synth_xilinx -family xc7 -top basys3_security_top; stat"