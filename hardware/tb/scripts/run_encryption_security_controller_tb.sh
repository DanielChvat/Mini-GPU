#!/bin/bash
set -e

# must be called from project root directory

iverilog -g2012 -o tmp/encryption_security_controller_tb.vvp \
    hardware/rtl/common/uart_rx_byte.v \
    hardware/rtl/common/uart_tx_byte.v \
    hardware/rtl/security/Encryption/encryption_security_controller.v \
    hardware/rtl/security/Encryption/hmac_sha256_packet.v \
    hardware/rtl/security/Encryption/sha256_byte_stream.v \
    hardware/rtl/security/gate/sha256/sha256_core.v \
    hardware/rtl/security/gate/sha256/sha256_k_constants.v \
    hardware/rtl/security/gate/sha256/sha256_w_mem.v \
    hardware/rtl/security/Encryption/rosc/rosc_wrapper.v \
    hardware/rtl/security/Encryption/rosc/rosc_entropy_core.v \
    hardware/rtl/security/Encryption/rosc/rosc.v \
    hardware/rtl/security/Encryption/chacha/chacha_wrapper.v \
    hardware/rtl/security/Encryption/chacha/chacha_core.v \
    hardware/rtl/security/Encryption/chacha/chacha_qr.v \
    hardware/rtl/security/Encryption/aes/aes_ctr_wrapper.v \
    hardware/rtl/security/Encryption/aes/aes_core.v \
    hardware/rtl/security/Encryption/aes/aes_encipher_block.v \
    hardware/rtl/security/Encryption/aes/aes_decipher_block.v \
    hardware/rtl/security/Encryption/aes/aes_key_mem.v \
    hardware/rtl/security/Encryption/aes/aes_sbox.v \
    hardware/rtl/security/Encryption/aes/aes_inv_sbox.v \
    hardware/tb/security_tb/encryption_security_controller_tb.sv

vvp tmp/encryption_security_controller_tb.vvp
