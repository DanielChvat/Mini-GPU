#!/bin/bash

# must be called from project root directory

iverilog -g2012 -o tmp/rosc_wrapper.vvp \
    hardware/rtl/security/Encryption/rosc/rosc_wrapper.v \
    hardware/rtl/security/Encryption/rosc/rosc_entropy_core.v \
    hardware/tb/security_tb/rosc_sim.v \
    hardware/tb/security_tb/rosc_wrapper_tb.v 

vvp tmp/rosc_wrapper.vvp

# for actual use, we compile with: # hardware/rtl/security/Encryption/rosc/rosc.v \