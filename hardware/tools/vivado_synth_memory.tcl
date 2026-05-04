set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ../..]]

create_project -in_memory -part xc7a35tcpg236-1
set_property target_language Verilog [current_project]

add_files -norecurse [list \
    [file join $repo_root hardware/rtl/memory/memory.v] \
    [file join $repo_root hardware/rtl/memory/memory_bank.v] \
]
set_property file_type SystemVerilog [get_files *.v]
set_property top memory [current_fileset]
update_compile_order -fileset sources_1

synth_design -top memory -part xc7a35tcpg236-1 \
    -generic ADDR_WIDTH=16 \
    -generic DATA_WIDTH=32 \
    -generic BANK_DEPTH=8192
report_utilization
