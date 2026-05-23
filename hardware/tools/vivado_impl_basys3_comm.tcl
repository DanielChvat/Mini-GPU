set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ../..]]

create_project -in_memory -part xc7a35tcpg236-1
set_property target_language Verilog [current_project]
set_property include_dirs [file join $repo_root hardware/rtl/include] [current_fileset]

add_files -norecurse [list \
    [file join $repo_root hardware/rtl/top/basys3_comm_top.v] \
    [file join $repo_root hardware/rtl/common/communication_controller.v] \
    [file join $repo_root hardware/rtl/common/serial_packet_rx.v] \
    [file join $repo_root hardware/rtl/common/serial_packet_tx.v] \
    [file join $repo_root hardware/rtl/common/uart_rx_byte.v] \
    [file join $repo_root hardware/rtl/common/uart_tx_byte.v] \
    [file join $repo_root hardware/rtl/security/Encryption/encryption_security_controller.v] \
    [file join $repo_root hardware/rtl/security/Encryption/hmac_sha256_packet.v] \
    [file join $repo_root hardware/rtl/security/Encryption/sha256_byte_stream.v] \
    [file join $repo_root hardware/rtl/security/gate/sha256/sha256_core.v] \
    [file join $repo_root hardware/rtl/security/gate/sha256/sha256_k_constants.v] \
    [file join $repo_root hardware/rtl/security/gate/sha256/sha256_w_mem.v] \
    [file join $repo_root hardware/rtl/security/Encryption/rosc/rosc_wrapper.v] \
    [file join $repo_root hardware/rtl/security/Encryption/rosc/rosc_entropy_core.v] \
    [file join $repo_root hardware/rtl/security/Encryption/rosc/rosc.v] \
    [file join $repo_root hardware/rtl/security/Encryption/chacha/chacha_wrapper.v] \
    [file join $repo_root hardware/rtl/security/Encryption/chacha/chacha_core.v] \
    [file join $repo_root hardware/rtl/security/Encryption/chacha/chacha_qr.v] \
    [file join $repo_root hardware/rtl/security/Encryption/aes/aes_ctr_wrapper.v] \
    [file join $repo_root hardware/rtl/security/Encryption/aes/aes_core.v] \
    [file join $repo_root hardware/rtl/security/Encryption/aes/aes_encipher_block.v] \
    [file join $repo_root hardware/rtl/security/Encryption/aes/aes_decipher_block.v] \
    [file join $repo_root hardware/rtl/security/Encryption/aes/aes_key_mem.v] \
    [file join $repo_root hardware/rtl/security/Encryption/aes/aes_sbox.v] \
    [file join $repo_root hardware/rtl/security/Encryption/aes/aes_inv_sbox.v] \
    [file join $repo_root hardware/rtl/core/mini_gpu.v] \
    [file join $repo_root hardware/rtl/core/mini_gpu_core.v] \
    [file join $repo_root hardware/rtl/core/sm.v] \
    [file join $repo_root hardware/rtl/core/block.v] \
    [file join $repo_root hardware/rtl/core/warp.v] \
    [file join $repo_root hardware/rtl/common/instruction_decode.v] \
    [file join $repo_root hardware/rtl/lane/thread.v] \
    [file join $repo_root hardware/rtl/lane/regfile.v] \
    [file join $repo_root hardware/rtl/lane/execute.v] \
    [file join $repo_root hardware/rtl/lane/int/mul.v] \
    [file join $repo_root hardware/rtl/lane/int/div_mod_iterative.v] \
    [file join $repo_root hardware/rtl/lane/float/add_sub.v] \
    [file join $repo_root hardware/rtl/lane/float/mul.v] \
    [file join $repo_root hardware/rtl/lane/float/div.v] \
    [file join $repo_root hardware/rtl/lane/float/shared_fpu.v] \
    [file join $repo_root hardware/rtl/memory/memory.v] \
    [file join $repo_root hardware/rtl/memory/memory_bank.v] \
]
set_property file_type SystemVerilog [get_files *.v]

add_files -fileset constrs_1 -norecurse [file join $repo_root hardware/constraints/basys3_comm.xdc]
set_property top basys3_comm_top [current_fileset]
update_compile_order -fileset sources_1

synth_design -top basys3_comm_top -part xc7a35tcpg236-1
opt_design
place_design
route_design
report_utilization
report_timing_summary
