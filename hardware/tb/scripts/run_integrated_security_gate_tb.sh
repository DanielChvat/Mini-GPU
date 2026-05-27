#!/bin/bash

# must be called from project root directory

iverilog -g2012 -o ./tmp/security_gate_integration_tb.vvp \
  -I hardware/rtl/include \
  hardware/tb/security_tb/security_gate_integration_tb.sv \
  hardware/rtl/security/gate/security_gate.v \
  hardware/rtl/security/gate/sha256_wrapper.v \
  hardware/rtl/security/gate/sha256/sha256_core.v \
  hardware/rtl/security/gate/sha256/sha256_w_mem.v \
  hardware/rtl/security/gate/sha256/sha256_k_constants.v \
  hardware/rtl/security/gate/memory_sec.v \
  hardware/rtl/memory/memory.v \
  hardware/rtl/memory/memory_bank.v \
  hardware/rtl/common/communication_controller.v \
  hardware/rtl/common/bus_controller.v \
  hardware/rtl/common/serial_packet_rx.v \
  hardware/rtl/common/serial_packet_tx.v 

vvp ./tmp/security_gate_integration_tb.vvp