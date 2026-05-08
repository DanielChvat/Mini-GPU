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
    [file join $repo_root hardware/rtl/memory/memory.v] \
    [file join $repo_root hardware/rtl/memory/memory_bank.v] \
]
set_property file_type SystemVerilog [get_files *.v]

add_files -fileset constrs_1 -norecurse [file join $repo_root hardware/constraints/basys3_comm.xdc]
set_property top basys3_comm_top [current_fileset]
update_compile_order -fileset sources_1

synth_design -top basys3_comm_top -part xc7a35tcpg236-1
report_utilization
report_timing_summary
