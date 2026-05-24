#!/bin/bash

# must be called from project root directory

iverilog -o tmp/kernel_vfy_tb.vvp \
  hardware/tb/security_tb/kernel_vfy_tb.v \
  hardware/rtl/security/gate/kernel_vfy/kernel_vfy.v \
  hardware/rtl/security/gate/kernel_vfy/kernel_hash_bram.v \
  hardware/rtl/security/gate/sha256_weight_stream.v \
  hardware/rtl/security/gate/sha256/sha256_core.v \
  hardware/rtl/security/gate/sha256/sha256_k_constants.v \
  hardware/rtl/security/gate/sha256/sha256_w_mem.v \
  2>&1 
  
vvp tmp/kernel_vfy_tb.vvp 2>&1