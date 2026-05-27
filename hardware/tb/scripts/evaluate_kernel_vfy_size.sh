#!/bin/bash

yosys -Q -p "read_verilog \
    hardware/rtl/security/gate/sha256_wrapper.v \
    hardware/rtl/security/gate/sha256/*v \
    hardware/rtl/security/gate/kernel_vfy/*.v 
synth_xilinx -family xc7 -top kernel_vfy; stat"