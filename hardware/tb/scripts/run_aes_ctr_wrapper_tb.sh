#!/bin/bash

# must be called from project root directory

iverilog -g2012 -o tmp/aes_ctr_wrapper.vvp \
    hardware/rtl/security/Encryption/aes/aes_ctr_wrapper.v \
    hardware/rtl/security/Encryption/aes/aes_core.v \
    hardware/rtl/security/Encryption/aes/aes_encipher_block.v \
    hardware/rtl/security/Encryption/aes/aes_decipher_block.v \
    hardware/rtl/security/Encryption/aes/aes_key_mem.v \
    hardware/rtl/security/Encryption/aes/aes_sbox.v \
    hardware/rtl/security/Encryption/aes/aes_inv_sbox.v \
    hardware/tb/security_tb/aes_ctr_wrapper_tb.v

vvp tmp/aes_ctr_wrapper.vvp
