#!/bin/bash

# must be called from project root directory

iverilog -g2012 -o tmp/communication_controller.vvp hardware/rtl/common/communication_controller.v hardware/rtl/common/serial_packet_tx.v hardware/rtl/common/serial_packet_rx.v hardware/rtl/memory/*.v  hardware/tb/common_tb/communication_controller_tb.sv

vvp tmp/communication_controller.vvp

