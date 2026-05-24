#!/usr/bin/env python3
"""Generate kernel metadata (IDs and hashes) from kernels.yaml"""

import yaml
import hashlib
from pathlib import Path

def generate_kernel_metadata(kernels_yaml_path, output_dir):
    with open(kernels_yaml_path) as f:
        manifest = yaml.safe_load(f)
    
    kernel_id_map = {}
    kernel_hashes = {}
    kernel_id = 0
    
    for artifact in manifest['artifacts']:
        bin_file = artifact['file']
        bin_path = kernels_yaml_path.parent / bin_file
        
        # Compute hash of kernel binary
        sha256_hash = hashlib.sha256(bin_path.read_bytes()).hexdigest()
        
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
        # Create array indexed by kernel_id
        hashes_by_id = [None] * 128
        for name, kid in kernel_id_map.items():
            hash_bytes = bytes.fromhex(kernel_hashes[name])
            hash_str = ', '.join(f'0x{b:02x}' for b in hash_bytes)
            hashes_by_id[kid] = hash_str
        
        for i, hash_str in enumerate(hashes_by_id):
            if hash_str is None:
                # Empty slot
                f.write(f'    {{{", ".join(["0x00"] * 32)}}},  // Slot {i} (unused)\n')
            else:
                f.write(f'    {{{hash_str}}},  // {i}\n')
        
        f.write('''}};
// clang-format on

} // namespace minigpu

#endif
''')
    
    print(f"Generated {id_map_hpp}")
    print(f"Generated {hashes_hpp}")
    print(f"Total kernels: {kernel_id}")

if __name__ == '__main__':
    kernels_dir = Path(__file__).parent.parent / 'kernels'
    output_dir = kernels_dir.parent / 'src' / 'generated'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_kernel_metadata(kernels_dir / 'kernels.yaml', output_dir)