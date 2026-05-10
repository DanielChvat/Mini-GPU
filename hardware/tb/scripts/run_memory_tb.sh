#!/bin/bash

# must be called from project root directory

iverilog -g2012 -o tmp/memory_sec.vvp hardware/rtl/memory/memory_bank.v hardware/rtl/memory/memory_sec.v hardware/rtl/memory/memory.v hardware/tb/security_tb/memory_sec_tb.v 

vvp tmp/memory_sec.vvp

