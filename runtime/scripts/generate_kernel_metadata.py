#!/usr/bin/env python3
"""Generate kernel metadata (IDs and hashes) from kernels.yaml."""

import hashlib
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def load_manifest(kernels_yaml_path):
    if yaml is not None:
        with open(kernels_yaml_path) as f:
            return yaml.safe_load(f)

    artifacts = []
    current_artifact = None
    current_kernel = None

    for raw_line in kernels_yaml_path.read_text().splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- file:"):
            current_artifact = {
                "file": stripped.split(":", 1)[1].strip(),
                "kernels": [],
            }
            artifacts.append(current_artifact)
            current_kernel = None
        elif stripped.startswith("map:") and current_artifact is not None:
            current_artifact["map"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- name:") and current_artifact is not None:
            current_kernel = {"name": stripped.split(":", 1)[1].strip()}
            current_artifact["kernels"].append(current_kernel)
        elif current_kernel is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_kernel[key.strip()] = value.strip()

    return {"artifacts": artifacts}


def rtl_program_stream_bytes(raw_bytes):
    if len(raw_bytes) % 4 != 0:
        raise ValueError("Kernel binary size must be a multiple of 4 bytes")

    stream = bytearray()
    for offset in range(0, len(raw_bytes), 4):
        stream.extend(raw_bytes[offset:offset + 4][::-1])
    return bytes(stream)


def generate_kernel_metadata(kernels_yaml_path, output_dir, hardware_mem_path):
    manifest = load_manifest(kernels_yaml_path)
    
    kernel_id_map = {}
    kernel_hashes = {}
    kernel_id = 0
    
    for artifact in manifest['artifacts']:
        bin_file = artifact['file']
        bin_path = kernels_yaml_path.parent / bin_file
        
        # Compute the digest over the word stream observed by kernel_vfy.
        sha256_hash = hashlib.sha256(rtl_program_stream_bytes(bin_path.read_bytes())).hexdigest()
        
        for kernel in artifact['kernels']:
            kernel_name = kernel['name']
            
            # Assign ID (0-127)
            if kernel_id > 127:
                raise ValueError(f"Too many kernels: {kernel_id} > 127")
            
            kernel_id_map[kernel_name] = kernel_id
            kernel_hashes[kernel_name] = sha256_hash
            kernel_id += 1
    
    # Generate kernel_id_map.hpp
    id_map_hpp = output_dir / "kernel_id_map.hpp"
    with open(id_map_hpp, 'w') as f:
        f.write('''#ifndef KERNEL_ID_MAP_HPP
#define KERNEL_ID_MAP_HPP

#include <unordered_map>
#include <string>
#include <cstdint>

namespace minigpu {

static const std::unordered_map<std::string, uint8_t> KERNEL_ID_MAP = {
''')
        for name, kid in sorted(kernel_id_map.items()):
            f.write(f'    {{"{name}", {kid}}},\n')
        f.write('''};

} // namespace minigpu

#endif
''')
    
    hashes_by_id = [None] * 128
    for name, kid in kernel_id_map.items():
        hashes_by_id[kid] = kernel_hashes[name]

    # Generate kernel_hashes.hpp
    hashes_hpp = output_dir / "kernel_hashes.hpp"
    with open(hashes_hpp, 'w') as f:
        f.write('''#ifndef KERNEL_HASHES_HPP
#define KERNEL_HASHES_HPP

#include <cstdint>
#include <array>

namespace minigpu {

// Golden hashes for kernel verification (indexed by kernel_id)
// clang-format off
static const std::array<std::array<uint8_t, 32>, 128> KERNEL_GOLDEN_HASHES = {{
''')
        for i, hash_hex in enumerate(hashes_by_id):
            if hash_hex is None:
                # Empty slot
                f.write(f'    {{{", ".join(["0x00"] * 32)}}},  // Slot {i} (unused)\n')
            else:
                hash_str = ', '.join(f'0x{b:02x}' for b in bytes.fromhex(hash_hex))
                f.write(f'    {{{hash_str}}},  // {i}\n')
        
        f.write('''}};
// clang-format on

} // namespace minigpu

#endif
''')

    # Generate Verilog $readmemh contents for kernel_hash_bram.
    hardware_mem_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hardware_mem_path, 'w') as f:
        for hash_hex in hashes_by_id:
            f.write(f"{hash_hex if hash_hex is not None else '0' * 64}\n")
    
    print(f"Generated {id_map_hpp}")
    print(f"Generated {hashes_hpp}")
    print(f"Generated {hardware_mem_path}")
    print(f"Total kernels: {kernel_id}")

if __name__ == '__main__':
    kernels_dir = Path(__file__).parent.parent / 'kernels'
    output_dir = kernels_dir.parent / 'src' / 'generated'
    repo_root = Path(__file__).resolve().parents[2]
    hardware_mem_path = repo_root / 'hardware' / 'rtl' / 'security' / 'gate' / 'kernel_vfy' / 'kernel_golden_hashes.mem'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_kernel_metadata(kernels_dir / 'kernels.yaml', output_dir, hardware_mem_path)
