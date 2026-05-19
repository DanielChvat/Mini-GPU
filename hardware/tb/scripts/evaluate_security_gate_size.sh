#!/bin/bash

yosys -Q -p "read_verilog \
  hardware/rtl/security/gate/security_gate.v \
  hardware/rtl/security/gate/sha256_weight_stream.v \
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
synth_xilinx -family xc7 -top security_gate; stat"