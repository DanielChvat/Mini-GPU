#!/bin/bash

# must be called from project root directory

iverilog -o ./tmp/sha256_tb.vvp \
  hardware/tb/security_tb/sha256_wrapper_tb.v \
  hardware/rtl/security/gate/sha256_wrapper.v \
  hardware/rtl/security/gate/sha256/sha256_core.v \
  hardware/rtl/security/gate/sha256/sha256_w_mem.v \
  hardware/rtl/security/gate/sha256/sha256_k_constants.v
  
vvp ./tmp/sha256_tb.vvp