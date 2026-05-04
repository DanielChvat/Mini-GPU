set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ../..]]

create_project -in_memory -part xc7a35tcpg236-1
set_property target_language Verilog [current_project]
set_property include_dirs [file join $repo_root hardware/rtl/include] [current_fileset]

add_files -norecurse [list \
    [file join $repo_root hardware/rtl/top/basys3_mini_gpu_top.v] \
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

add_files -fileset constrs_1 -norecurse [file join $repo_root hardware/constraints/basys3_mini_gpu.xdc]
set_property top basys3_mini_gpu_top [current_fileset]
update_compile_order -fileset sources_1
synth_design -top basys3_mini_gpu_top -part xc7a35tcpg236-1
report_utilization
report_timing_summary
