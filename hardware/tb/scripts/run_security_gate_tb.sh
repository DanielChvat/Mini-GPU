#!/bin/bash

# must be called from project root directory

iverilog -o ./tmp/security_gate_tb.vvp \
  -I hardware/rtl/include \
  hardware/tb/security_tb/security_gate_tb.v \
  hardware/rtl/security/security_gate.v \
  hardware/rtl/security/sha256_weight_stream.v \
  hardware/rtl/security/sha256/sha256_core.v \
  hardware/rtl/security/sha256/sha256_w_mem.v \
  hardware/rtl/security/sha256/sha256_k_constants.v \
  hardware/rtl/memory/memory_sec.v \
  hardware/rtl/memory/memory.v \
  hardware/rtl/memory/memory_bank.v 

vvp ./tmp/security_gate_tb.vvp 