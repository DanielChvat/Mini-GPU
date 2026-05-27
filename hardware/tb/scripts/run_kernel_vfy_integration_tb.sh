#!/bin/bash

# must be called from project root directory

iverilog -g2012 -o tmp/kernel_vfy_int_tb.vvp \
  hardware/tb/security_tb/kernel_vfy_integration_tb.sv \
  hardware/rtl/security/gate/kernel_vfy/kernel_vfy.v \
  hardware/rtl/security/gate/kernel_vfy/kernel_hash_bram.v \
  hardware/rtl/security/gate/sha256_wrapper.v \
  hardware/rtl/security/gate/sha256/sha256_core.v \
  hardware/rtl/security/gate/sha256/sha256_k_constants.v \
  hardware/rtl/security/gate/sha256/sha256_w_mem.v \
  hardware/rtl/common/communication_controller.v \
  hardware/rtl/common/serial_packet_rx.v \
  hardware/rtl/common/serial_packet_tx.v \
  2>&1

vvp tmp/kernel_vfy_int_tb.vvp