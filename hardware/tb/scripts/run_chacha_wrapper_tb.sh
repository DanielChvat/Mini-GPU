#!/bin/bash

# must be called from project root directory

iverilog -g2012 -o tmp/chacha_wrapper.vvp \
    hardware/rtl/security/Encryption/chacha/chacha_wrapper.v \
    hardware/rtl/security/Encryption/chacha/chacha_core.v \
    hardware/rtl/security/Encryption/chacha/chacha_qr.v \
    hardware/tb/security_tb/chacha_wrapper_tb.v

vvp tmp/chacha_wrapper.vvp
