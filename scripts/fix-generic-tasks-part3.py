#!/usr/bin/env python3
"""Fix remaining generic tasks with tool-specific content."""

import sqlite3

DB_PATH = "data/quest-log.db"

SPECIFIC_DETAILS = {
    # ============== Evil-WinRM - Remaining Generic Tasks ==============
    ("Reimplement: Evil-WinRM & C2 Frameworks", "Add configuration"): """## Evil-WinRM Configuration

### Configuration Structure
```yaml
# config.yaml
connection:
  host: "192.168.1.100"
  port: 5985
  ssl: false
  timeout: 30

authentication:
  method: "ntlm"  # ntlm, kerberos, basic
  username: "admin"
  domain: "CORP"
  # For pass-the-hash:
  nt_hash: "aad3b435b51404eeaad3b435b51404ee"
  # For kerberos:
  # ticket_cache: "/tmp/krb5cc_1000"

shell:
  prompt: "Evil-WinRM> "
  history_file: "~/.evil_winrm_history"
  colors: true

logging:
  level: "info"
  file: "evil_winrm.log"
```

### Implementation
```go
package config

type Config struct {
    Connection     ConnectionConfig `yaml:"connection"`
    Authentication AuthConfig       `yaml:"authentication"`
    Shell          ShellConfig      `yaml:"shell"`
}

type ConnectionConfig struct {
    Host    string `yaml:"host"`
    Port    int    `yaml:"port"`
    SSL     bool   `yaml:"ssl"`
    Timeout int    `yaml:"timeout"`
}

type AuthConfig struct {
    Method   string `yaml:"method"`
    Username string `yaml:"username"`
    Domain   string `yaml:"domain"`
    NTHash   string `yaml:"nt_hash"`
}

func Load(path string) (*Config, error) {
    data, _ := os.ReadFile(path)
    var cfg Config
    yaml.Unmarshal(data, &cfg)
    return &cfg, nil
}
```

### Completion Criteria
- [ ] Define YAML configuration schema
- [ ] Support multiple authentication methods
- [ ] Add connection timeout settings
- [ ] Implement config file loading""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Build CLI or API"): """## Evil-WinRM CLI Interface

### Command-Line Interface
```go
package main

import (
    "github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
    Use:   "evil-winrm",
    Short: "WinRM shell for penetration testing",
}

var connectCmd = &cobra.Command{
    Use:   "connect",
    Short: "Connect to target via WinRM",
    Run: func(cmd *cobra.Command, args []string) {
        host, _ := cmd.Flags().GetString("host")
        user, _ := cmd.Flags().GetString("user")
        hash, _ := cmd.Flags().GetString("hash")

        client := winrm.NewClient(host, user, winrm.WithHash(hash))
        shell := interactive.NewShell(client)
        shell.Run()
    },
}

func init() {
    connectCmd.Flags().StringP("host", "H", "", "Target host")
    connectCmd.Flags().StringP("user", "u", "", "Username")
    connectCmd.Flags().StringP("password", "p", "", "Password")
    connectCmd.Flags().StringP("hash", "h", "", "NT hash for PTH")
    connectCmd.Flags().BoolP("ssl", "s", false, "Use HTTPS")

    rootCmd.AddCommand(connectCmd)
}
```

### Interactive Shell Commands
```
Evil-WinRM Commands:
  upload <local> <remote>  - Upload file to target
  download <remote> <local> - Download file from target
  services                  - List Windows services
  Invoke-Binary <path>      - Load and execute .NET assembly
  Bypass-4MSI              - Patch AMSI in current session
  menu                     - Show available commands
  exit                     - Close connection
```

### Completion Criteria
- [ ] Implement CLI with cobra/urfave
- [ ] Add all connection flags
- [ ] Build interactive shell REPL
- [ ] Implement built-in commands""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Add logging"): """## Evil-WinRM Logging

### Structured Logging
```go
package logging

import (
    "log/slog"
    "os"
)

type WinRMLogger struct {
    logger *slog.Logger
    opLog  *os.File  // Operation log for audit
}

func NewLogger(config LogConfig) *WinRMLogger {
    var handler slog.Handler

    if config.Format == "json" {
        handler = slog.NewJSONHandler(os.Stdout, nil)
    } else {
        handler = slog.NewTextHandler(os.Stdout, nil)
    }

    return &WinRMLogger{
        logger: slog.New(handler),
    }
}

func (l *WinRMLogger) LogCommand(cmd string, output string) {
    l.logger.Info("command executed",
        "command", cmd,
        "output_length", len(output),
    )
}

func (l *WinRMLogger) LogAuth(method string, success bool, target string) {
    l.logger.Info("authentication attempt",
        "method", method,
        "success", success,
        "target", target,
    )
}

func (l *WinRMLogger) LogFileTransfer(direction string, local, remote string) {
    l.logger.Info("file transfer",
        "direction", direction,
        "local_path", local,
        "remote_path", remote,
    )
}
```

### Security Considerations
- Log authentication attempts (success/fail)
- Record all commands executed
- Track file transfers for audit trail
- Sanitize sensitive data from logs

### Completion Criteria
- [ ] Implement structured logging
- [ ] Log all authentication attempts
- [ ] Record command execution history
- [ ] Add file transfer logging""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Write tests"): """## Evil-WinRM Testing

### Unit Tests
```go
package winrm

import (
    "testing"
)

func TestSOAPMessageBuilder(t *testing.T) {
    tests := []struct {
        name     string
        action   string
        resource string
        wantErr  bool
    }{
        {"create shell", "Create", "shell/cmd", false},
        {"execute command", "Command", "shell/cmd", false},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            msg := BuildSOAPMessage(tt.action, tt.resource)
            if msg == "" && !tt.wantErr {
                t.Error("expected message, got empty")
            }
        })
    }
}

func TestNTLMAuth(t *testing.T) {
    auth := NewNTLMAuth("user", "aad3b435b51404eeaad3b435b51404ee")

    // Test negotiate message
    nego := auth.GetNegotiateMessage()
    if len(nego) < 32 {
        t.Error("negotiate message too short")
    }

    // Verify NTLMSSP signature
    if string(nego[:8]) != "NTLMSSP\\x00" {
        t.Error("invalid NTLMSSP signature")
    }
}

func TestCommandParsing(t *testing.T) {
    tests := []struct {
        input    string
        wantCmd  string
        wantArgs []string
    }{
        {"upload local.txt C:\\\\remote.txt", "upload", []string{"local.txt", "C:\\\\remote.txt"}},
        {"download C:\\\\file.txt local.txt", "download", []string{"C:\\\\file.txt", "local.txt"}},
    }

    for _, tt := range tests {
        cmd, args := ParseShellCommand(tt.input)
        if cmd != tt.wantCmd {
            t.Errorf("got %s, want %s", cmd, tt.wantCmd)
        }
    }
}
```

### Integration Tests
```go
func TestWinRMConnection(t *testing.T) {
    if testing.Short() {
        t.Skip("skipping integration test")
    }

    client := NewClient("192.168.1.100", "admin", WithPassword("password"))

    err := client.Connect()
    if err != nil {
        t.Fatalf("connection failed: %v", err)
    }

    output, err := client.Execute("whoami")
    if err != nil || output == "" {
        t.Error("command execution failed")
    }
}
```

### Completion Criteria
- [ ] Write SOAP message builder tests
- [ ] Test NTLM authentication flow
- [ ] Test command parsing
- [ ] Add integration tests with mock server""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Optimize performance"): """## Evil-WinRM Performance Optimization

### Connection Pooling
```go
package winrm

import (
    "sync"
    "time"
)

type ConnectionPool struct {
    connections chan *WinRMClient
    maxSize     int
    mu          sync.Mutex
}

func NewConnectionPool(host string, maxSize int) *ConnectionPool {
    pool := &ConnectionPool{
        connections: make(chan *WinRMClient, maxSize),
        maxSize:     maxSize,
    }

    // Pre-warm pool
    for i := 0; i < maxSize; i++ {
        client := NewClient(host)
        client.Connect()
        pool.connections <- client
    }

    return pool
}

func (p *ConnectionPool) Get() *WinRMClient {
    return <-p.connections
}

func (p *ConnectionPool) Put(client *WinRMClient) {
    p.connections <- client
}
```

### Shell Reuse
```go
// Reuse shell instead of creating new one for each command
type PersistentShell struct {
    client  *WinRMClient
    shellID string
}

func (s *PersistentShell) Execute(cmd string) (string, error) {
    if s.shellID == "" {
        s.shellID = s.client.CreateShell()
    }

    // Execute in existing shell
    return s.client.ExecuteInShell(s.shellID, cmd)
}
```

### File Transfer Optimization
```go
// Chunked transfer for large files
func (c *WinRMClient) UploadChunked(localPath, remotePath string, chunkSize int) error {
    file, _ := os.Open(localPath)
    defer file.Close()

    buffer := make([]byte, chunkSize)
    offset := 0

    for {
        n, err := file.Read(buffer)
        if err == io.EOF {
            break
        }

        chunk := base64.StdEncoding.EncodeToString(buffer[:n])
        // Append to remote file
        c.Execute(fmt.Sprintf(
            "[IO.File]::WriteAllBytes('%s', [Convert]::FromBase64String('%s'))",
            remotePath, chunk))
        offset += n
    }
    return nil
}
```

### Completion Criteria
- [ ] Implement connection pooling
- [ ] Reuse shells across commands
- [ ] Optimize file transfers with chunking
- [ ] Add response compression""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Add documentation"): """## Evil-WinRM Documentation

### README Structure
```markdown
# Evil-WinRM Reimplementation

A WinRM shell for penetration testing with Pass-the-Hash support.

## Features

- Interactive PowerShell shell over WinRM
- Pass-the-Hash authentication (NTLM)
- File upload/download
- In-memory .NET assembly execution
- AMSI bypass capability

## Installation

```bash
go install github.com/yourname/evil-winrm@latest
```

## Usage

### Basic Connection
```bash
evil-winrm -H 192.168.1.100 -u administrator -p 'Password123!'
```

### Pass-the-Hash
```bash
evil-winrm -H 192.168.1.100 -u administrator -h aad3b435b51404eeaad3b435b51404ee
```

### With SSL
```bash
evil-winrm -H 192.168.1.100 -u admin -p pass -s
```

## Shell Commands

| Command | Description |
|---------|-------------|
| `upload <local> <remote>` | Upload file to target |
| `download <remote> <local>` | Download file from target |
| `Invoke-Binary <path>` | Execute .NET assembly in memory |
| `Bypass-4MSI` | Bypass AMSI |
| `menu` | Show all commands |
```

### API Documentation
```go
// Package winrm provides WinRM client functionality
package winrm

// Client represents a WinRM connection
type Client struct {
    // Host is the target hostname or IP
    Host string
    // Port is the WinRM port (default 5985/5986)
    Port int
}

// NewClient creates a new WinRM client
func NewClient(host string, opts ...Option) *Client

// Connect establishes the WinRM connection
func (c *Client) Connect() error

// Execute runs a command and returns output
func (c *Client) Execute(cmd string) (string, error)
```

### Completion Criteria
- [ ] Write comprehensive README
- [ ] Document all CLI options
- [ ] Create API documentation
- [ ] Add usage examples""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Handle edge cases"): """## Evil-WinRM Edge Cases

### Connection Handling
```go
package winrm

import (
    "errors"
    "time"
)

var (
    ErrShellClosed    = errors.New("shell was closed unexpectedly")
    ErrOutputTruncated = errors.New("output exceeded buffer limit")
)

func (c *Client) ExecuteWithRetry(cmd string, maxRetries int) (string, error) {
    var lastErr error

    for i := 0; i < maxRetries; i++ {
        output, err := c.Execute(cmd)
        if err == nil {
            return output, nil
        }

        lastErr = err

        // Check if shell died
        if errors.Is(err, ErrShellClosed) {
            // Recreate shell
            c.shellID = ""
            c.createShell()
            continue
        }

        // Connection issue - reconnect
        if isConnectionError(err) {
            c.reconnect()
            continue
        }
    }

    return "", lastErr
}

// Handle long-running commands
func (c *Client) ExecuteAsync(cmd string, timeout time.Duration) (string, error) {
    done := make(chan struct{})
    var output string
    var err error

    go func() {
        output, err = c.Execute(cmd)
        close(done)
    }()

    select {
    case <-done:
        return output, err
    case <-time.After(timeout):
        // Command still running - check status
        return "", ErrTimeout
    }
}

// Handle binary output (non-UTF8)
func (c *Client) ExecuteBinary(cmd string) ([]byte, error) {
    // Wrap command to base64 encode output
    wrapped := fmt.Sprintf(
        "[Convert]::ToBase64String([Text.Encoding]::Default.GetBytes((%s)))",
        cmd)

    output, err := c.Execute(wrapped)
    if err != nil {
        return nil, err
    }

    return base64.StdEncoding.DecodeString(strings.TrimSpace(output))
}
```

### Input Validation
```go
func validateCommand(cmd string) error {
    // Check for command injection in upload paths
    if strings.ContainsAny(cmd, "|;&$`") {
        return errors.New("potentially dangerous characters in command")
    }
    return nil
}
```

### Completion Criteria
- [ ] Handle shell disconnection gracefully
- [ ] Support long-running commands
- [ ] Handle binary/non-UTF8 output
- [ ] Validate user input""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Package for distribution"): """## Evil-WinRM Distribution

### Cross-Platform Builds
```makefile
VERSION := $(shell git describe --tags --always)
LDFLAGS := -X main.version=$(VERSION) -s -w

.PHONY: build-all
build-all: build-linux build-windows build-darwin

build-linux:
	GOOS=linux GOARCH=amd64 go build -ldflags "$(LDFLAGS)" -o dist/evil-winrm-linux-amd64 ./cmd/evil-winrm
	GOOS=linux GOARCH=arm64 go build -ldflags "$(LDFLAGS)" -o dist/evil-winrm-linux-arm64 ./cmd/evil-winrm

build-windows:
	GOOS=windows GOARCH=amd64 go build -ldflags "$(LDFLAGS)" -o dist/evil-winrm-windows-amd64.exe ./cmd/evil-winrm

build-darwin:
	GOOS=darwin GOARCH=amd64 go build -ldflags "$(LDFLAGS)" -o dist/evil-winrm-darwin-amd64 ./cmd/evil-winrm
	GOOS=darwin GOARCH=arm64 go build -ldflags "$(LDFLAGS)" -o dist/evil-winrm-darwin-arm64 ./cmd/evil-winrm

.PHONY: release
release: build-all
	cd dist && sha256sum * > checksums.txt
	gh release create $(VERSION) dist/*
```

### Docker Image
```dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o evil-winrm ./cmd/evil-winrm

FROM alpine:latest
RUN apk --no-cache add ca-certificates
COPY --from=builder /app/evil-winrm /usr/local/bin/
ENTRYPOINT ["evil-winrm"]
```

### GitHub Actions Release
```yaml
name: Release
on:
  push:
    tags: ['v*']

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.21'
      - run: make build-all
      - uses: softprops/action-gh-release@v1
        with:
          files: dist/*
```

### Completion Criteria
- [ ] Create cross-platform build scripts
- [ ] Add Docker image
- [ ] Set up CI/CD for releases
- [ ] Create installation documentation""",

    ("Reimplement: Evil-WinRM & C2 Frameworks", "Final testing"): """## Evil-WinRM Final Testing

### End-to-End Test Plan
```bash
#!/bin/bash
# E2E test against Windows target

TARGET="192.168.1.100"
USER="administrator"
HASH="aad3b435b51404eeaad3b435b51404ee"

echo "[*] Testing connection..."
./evil-winrm -H $TARGET -u $USER -h $HASH -c "whoami"
[ $? -eq 0 ] || exit 1

echo "[*] Testing file upload..."
echo "test content" > /tmp/test.txt
./evil-winrm -H $TARGET -u $USER -h $HASH -c "upload /tmp/test.txt C:\\\\temp\\\\test.txt"
[ $? -eq 0 ] || exit 1

echo "[*] Testing file download..."
./evil-winrm -H $TARGET -u $USER -h $HASH -c "download C:\\\\Windows\\\\System32\\\\drivers\\\\etc\\\\hosts /tmp/hosts"
[ -f /tmp/hosts ] || exit 1

echo "[*] Testing PowerShell execution..."
./evil-winrm -H $TARGET -u $USER -h $HASH -c "Get-Process | Select -First 5"
[ $? -eq 0 ] || exit 1

echo "[+] All tests passed!"
```

### Security Testing
```
Manual Security Checks:
1. [ ] Test against patched vs unpatched Windows
2. [ ] Verify AMSI bypass works
3. [ ] Test with various AV/EDR solutions
4. [ ] Verify no credentials in logs
5. [ ] Test SSL certificate validation
6. [ ] Check for memory leaks in long sessions
```

### Performance Benchmarks
```go
func BenchmarkCommandExecution(b *testing.B) {
    client := setupTestClient()

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        client.Execute("whoami")
    }
}

func BenchmarkFileUpload(b *testing.B) {
    client := setupTestClient()
    data := make([]byte, 1024*1024) // 1MB

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        client.Upload(data, "C:\\\\temp\\\\bench.bin")
    }
}
```

### Completion Criteria
- [ ] Run E2E tests against real Windows target
- [ ] Verify all features work
- [ ] Run security tests
- [ ] Benchmark performance""",

    # ============== Impacket Suite - Remaining ==============
    ("Reimplement: Impacket Suite", "Add configuration"): """## Impacket Configuration

### Configuration Schema
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TargetConfig:
    host: str
    port: int = 445
    domain: str = ""

@dataclass
class AuthConfig:
    username: str
    password: Optional[str] = None
    nt_hash: Optional[str] = None
    aes_key: Optional[str] = None
    kerberos: bool = False
    dc_ip: Optional[str] = None

@dataclass
class SMBConfig:
    signing: bool = True
    encryption: bool = False
    dialect: str = "auto"  # SMB1, SMB2, SMB3, auto

@dataclass
class Config:
    target: TargetConfig
    auth: AuthConfig
    smb: SMBConfig = SMBConfig()
```

### Configuration File
```yaml
# impacket.yaml
target:
  host: "dc01.corp.local"
  domain: "CORP"

auth:
  username: "admin"
  password: "Password123!"
  # Or for PTH:
  # nt_hash: "aad3b435b51404eeaad3b435b51404ee"
  # Or for Kerberos:
  # kerberos: true
  # dc_ip: "192.168.1.10"

smb:
  signing: true
  dialect: "auto"
```

### Completion Criteria
- [ ] Define configuration schema
- [ ] Support multiple auth methods
- [ ] Add SMB settings
- [ ] Implement config file loading""",

    ("Reimplement: Impacket Suite", "Add logging"): """## Impacket Logging

### Structured Logging
```python
import logging
from datetime import datetime

class ImpacketLogger:
    def __init__(self, level=logging.INFO):
        self.logger = logging.getLogger('impacket')
        self.logger.setLevel(level)

        # Console handler with colors
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter())
        self.logger.addHandler(handler)

    def log_auth(self, method: str, target: str, success: bool):
        self.logger.info(f"AUTH {method} -> {target}: {'SUCCESS' if success else 'FAILED'}")

    def log_smb_operation(self, op: str, share: str, path: str):
        self.logger.debug(f"SMB {op}: \\\\\\\\{share}\\\\{path}")

    def log_secret(self, secret_type: str, value: str):
        # Redact in non-verbose mode
        self.logger.info(f"SECRET [{secret_type}]: {value[:20]}...")

class ColorFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\\033[92m',    # Green
        'WARNING': '\\033[93m', # Yellow
        'ERROR': '\\033[91m',   # Red
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        return f"{color}[{record.levelname}] {record.msg}\\033[0m"
```

### Completion Criteria
- [ ] Implement structured logging
- [ ] Add authentication logging
- [ ] Log SMB operations
- [ ] Handle secret redaction""",

    ("Reimplement: Impacket Suite", "Write tests"): """## Impacket Testing

### Unit Tests
```python
import unittest
from impacket.smb import SMBPacket, SMBConnection
from impacket.ntlm import NTLM

class TestSMBPacket(unittest.TestCase):
    def test_negotiate_request(self):
        pkt = SMBPacket.negotiate_request(['NT LM 0.12', 'SMB 2.002'])
        self.assertEqual(pkt[:4], b'\\xffSMB')
        self.assertEqual(pkt[4], 0x72)  # SMB_COM_NEGOTIATE

    def test_session_setup_request(self):
        ntlm = NTLM('user', 'password')
        pkt = SMBPacket.session_setup_request(ntlm.get_negotiate())
        self.assertIn(b'NTLMSSP', pkt)

class TestNTLM(unittest.TestCase):
    def test_negotiate_message(self):
        ntlm = NTLM('user', 'password')
        msg = ntlm.get_negotiate()
        self.assertEqual(msg[:8], b'NTLMSSP\\x00')
        self.assertEqual(msg[8], 1)  # Type 1

    def test_nt_hash(self):
        ntlm = NTLM('user', 'password')
        expected = 'a4f49c406510bdcab6824ee7c30fd852'
        self.assertEqual(ntlm.nt_hash.hex(), expected)

class TestSMBConnection(unittest.TestCase):
    def test_parse_negotiate_response(self):
        # Sample response bytes
        response = b'...'
        conn = SMBConnection('test')
        result = conn._parse_negotiate_response(response)
        self.assertIn('dialect', result)
```

### Integration Tests
```python
@unittest.skipUnless(os.getenv('INTEGRATION_TEST'), 'Integration tests disabled')
class TestSMBIntegration(unittest.TestCase):
    def setUp(self):
        self.conn = SMBConnection(os.getenv('TEST_TARGET'))
        self.conn.login(os.getenv('TEST_USER'), os.getenv('TEST_PASS'))

    def test_list_shares(self):
        shares = self.conn.list_shares()
        self.assertIn('ADMIN$', shares)
        self.assertIn('C$', shares)

    def test_file_operations(self):
        self.conn.put_file('C$', 'temp/test.txt', b'test content')
        content = self.conn.get_file('C$', 'temp/test.txt')
        self.assertEqual(content, b'test content')
```

### Completion Criteria
- [ ] Test SMB packet construction
- [ ] Test NTLM authentication
- [ ] Add integration tests
- [ ] Test error handling""",

    # ============== ntlmrelayx - Remaining ==============
    ("Reimplement: ntlmrelayx", "Add configuration"): """## ntlmrelayx Configuration

### Configuration Schema
```python
@dataclass
class ServerConfig:
    smb_enabled: bool = True
    http_enabled: bool = True
    smb_port: int = 445
    http_port: int = 80

@dataclass
class TargetConfig:
    targets_file: Optional[str] = None
    targets: List[str] = None
    protocol: str = "smb"  # smb, ldap, http, mssql

@dataclass
class AttackConfig:
    attack: str = "default"  # default, ldap-delegate, adcs
    loot_dir: str = "./loot"
    execute: Optional[str] = None  # Command to execute

@dataclass
class RelayConfig:
    server: ServerConfig
    target: TargetConfig
    attack: AttackConfig
```

### Command Line
```
ntlmrelayx.py -tf targets.txt -smb2support
ntlmrelayx.py -t ldap://dc01.corp.local --delegate-access
ntlmrelayx.py -t http://adcs.corp.local/certsrv/certfnsh.asp --adcs
```

### Completion Criteria
- [ ] Define server configuration
- [ ] Support target file loading
- [ ] Configure attack modules
- [ ] Add output directory settings""",

    ("Reimplement: ntlmrelayx", "Add logging"): """## ntlmrelayx Logging

### Relay Event Logging
```python
class RelayLogger:
    def __init__(self):
        self.logger = logging.getLogger('ntlmrelayx')
        self.captured = []

    def log_capture(self, client_ip: str, username: str, domain: str):
        self.logger.info(f"[*] NTLM captured: {domain}\\\\{username} from {client_ip}")
        self.captured.append({
            'time': datetime.now(),
            'client': client_ip,
            'user': f"{domain}\\\\{username}"
        })

    def log_relay_attempt(self, source: str, target: str, success: bool):
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"[{'+'if success else '-'}] Relay {source} -> {target}: {status}")

    def log_attack_result(self, attack: str, target: str, result: str):
        self.logger.info(f"[*] Attack [{attack}] on {target}: {result}")

    def log_signing_required(self, target: str):
        self.logger.warning(f"[-] Skipping {target}: SMB signing required")
```

### Completion Criteria
- [ ] Log all NTLM captures
- [ ] Log relay attempts and results
- [ ] Track attack outcomes
- [ ] Report signing/channel binding issues""",

    ("Reimplement: ntlmrelayx", "Write tests"): """## ntlmrelayx Testing

### Unit Tests
```python
class TestNTLMCapture(unittest.TestCase):
    def test_extract_ntlm_type1(self):
        smb_data = b'...'  # SMB with NTLM Type 1
        ntlm_msg = extract_ntlm_message(smb_data)
        self.assertEqual(ntlm_msg[:8], b'NTLMSSP\\x00')
        self.assertEqual(ntlm_msg[8], 1)  # Type 1

    def test_extract_ntlm_type3(self):
        smb_data = b'...'  # SMB with NTLM Type 3
        ntlm_msg = extract_ntlm_message(smb_data)
        self.assertEqual(ntlm_msg[8], 3)  # Type 3

class TestRelayEngine(unittest.TestCase):
    def test_check_signing(self):
        engine = RelayEngine()
        # Target with signing required
        result = engine.check_target('192.168.1.100', 'smb')
        self.assertFalse(result.vulnerable)

    def test_relay_authentication(self):
        engine = RelayEngine()
        mock_type1 = create_mock_ntlm_type1()
        mock_type3 = create_mock_ntlm_type3()

        # Mock target server
        with mock_smb_server() as target:
            result = engine.relay(mock_type1, mock_type3, target.address)
            self.assertTrue(result.success)
```

### Integration Tests
```python
class TestRelayIntegration(unittest.TestCase):
    @unittest.skipUnless(os.getenv('RELAY_TEST'), 'Relay tests disabled')
    def test_smb_relay(self):
        # Requires two test VMs
        victim = os.getenv('RELAY_VICTIM')
        target = os.getenv('RELAY_TARGET')

        server = SMBRelayServer(('0.0.0.0', 445), [target])
        # Trigger auth from victim...
        # Verify relay success
```

### Completion Criteria
- [ ] Test NTLM message extraction
- [ ] Test signing detection
- [ ] Test relay flow
- [ ] Add integration tests""",

    # ============== Network Tunneling - Remaining ==============
    ("Reimplement: Network Tunneling Tools", "Add configuration"): """## Tunneling Tool Configuration

### Configuration Schema
```yaml
# tunnel.yaml
mode: "client"  # client or server

server:
  listen: "0.0.0.0:8080"
  auth:
    enabled: true
    username: "admin"
    password: "secret"

client:
  server: "server.example.com:8080"
  reconnect_interval: 5s
  auth:
    username: "admin"
    password: "secret"

tunnels:
  - type: "socks"
    listen: "127.0.0.1:1080"

  - type: "local"
    listen: "127.0.0.1:8888"
    remote: "internal.corp:80"

  - type: "remote"
    listen: "0.0.0.0:9000"
    local: "127.0.0.1:22"
```

### Implementation
```go
type Config struct {
    Mode    string        `yaml:"mode"`
    Server  ServerConfig  `yaml:"server"`
    Client  ClientConfig  `yaml:"client"`
    Tunnels []TunnelConfig `yaml:"tunnels"`
}

type TunnelConfig struct {
    Type   string `yaml:"type"`   // socks, local, remote
    Listen string `yaml:"listen"`
    Remote string `yaml:"remote"`
    Local  string `yaml:"local"`
}
```

### Completion Criteria
- [ ] Define YAML configuration schema
- [ ] Support multiple tunnel types
- [ ] Add authentication settings
- [ ] Implement config hot-reload""",

    ("Reimplement: Network Tunneling Tools", "Add logging"): """## Tunneling Tool Logging

### Connection Logging
```go
type TunnelLogger struct {
    logger *slog.Logger
}

func (l *TunnelLogger) LogConnection(clientAddr, tunnelType string) {
    l.logger.Info("new connection",
        "client", clientAddr,
        "type", tunnelType,
    )
}

func (l *TunnelLogger) LogTunnel(id uint32, src, dst string, bytesIn, bytesOut int64) {
    l.logger.Info("tunnel stats",
        "id", id,
        "src", src,
        "dst", dst,
        "bytes_in", bytesIn,
        "bytes_out", bytesOut,
    )
}

func (l *TunnelLogger) LogError(op string, err error) {
    l.logger.Error("operation failed",
        "operation", op,
        "error", err,
    )
}
```

### Completion Criteria
- [ ] Log all connections
- [ ] Track bandwidth per tunnel
- [ ] Log errors with context
- [ ] Add debug logging for protocol""",

    ("Reimplement: Network Tunneling Tools", "Write tests"): """## Tunneling Tool Testing

### Unit Tests
```go
func TestSOCKS5Handshake(t *testing.T) {
    // Test SOCKS5 greeting
    greeting := []byte{0x05, 0x01, 0x00}  // Version 5, 1 method, no auth
    response := handleGreeting(greeting)
    assert.Equal(t, []byte{0x05, 0x00}, response)  // Accept no auth
}

func TestSOCKS5Connect(t *testing.T) {
    // Test SOCKS5 CONNECT request
    request := []byte{
        0x05, 0x01, 0x00,        // Ver, CMD=CONNECT, RSV
        0x01,                     // ATYP = IPv4
        0x7f, 0x00, 0x00, 0x01,  // 127.0.0.1
        0x00, 0x50,              // Port 80
    }
    target, err := parseConnectRequest(request)
    assert.NoError(t, err)
    assert.Equal(t, "127.0.0.1:80", target)
}

func TestMultiplexer(t *testing.T) {
    // Test channel multiplexing
    server, client := net.Pipe()
    mux := NewMultiplexer(client)

    // Open channel
    ch, err := mux.OpenChannel("127.0.0.1:8080")
    assert.NoError(t, err)
    assert.NotNil(t, ch)

    // Send data
    ch.Write([]byte("hello"))
    // Verify on server side...
}
```

### Completion Criteria
- [ ] Test SOCKS5 protocol handling
- [ ] Test channel multiplexing
- [ ] Test reconnection logic
- [ ] Add benchmarks for throughput""",
}

def update_tasks():
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
