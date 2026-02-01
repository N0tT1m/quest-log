import Database from 'better-sqlite3';
import { join } from 'path';

const db = new Database(join(process.cwd(), 'data', 'quest-log.db'));

// Networking Projects
const paths = [
  {
    name: 'Build Your Own Packet Sniffer',
    description: 'Create a network packet analyzer like Wireshark with protocol decoding',
    icon: 'wifi',
    color: 'blue',
    language: 'C, Rust, Python, Go',
    skills: 'Raw sockets, Protocol parsing, Pcap, BPF',
    difficulty: 'intermediate',
    estimated_weeks: 4,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Raw sockets | Socket creation |
| 1 | 2 | Packet capture | Receive packets |
| 1 | 3 | Ethernet parsing | MAC addresses |
| 1 | 4 | IP parsing | IPv4/IPv6 |
| 1 | 5 | ICMP parsing | Ping packets |
| 2 | 1 | TCP parsing | TCP headers |
| 2 | 2 | UDP parsing | UDP headers |
| 2 | 3 | ARP parsing | ARP requests |
| 2 | 4 | DNS parsing | DNS queries |
| 2 | 5 | HTTP parsing | HTTP headers |
| 3 | 1 | BPF filters | Packet filtering |
| 3 | 2 | Pcap format | Save captures |
| 3 | 3 | Pcap reading | Load captures |
| 3 | 4 | Statistics | Packet stats |
| 3 | 5 | Flow tracking | TCP streams |
| 4 | 1 | TUI display | Terminal UI |
| 4 | 2 | Hex dump | Packet bytes |
| 4 | 3 | Follow stream | TCP reassembly |
| 4 | 4 | Export | CSV/JSON |
| 4 | 5 | Integration | Full sniffer |`,
    modules: [
      {
        name: 'Packet Capture',
        description: 'Capture and parse network packets',
        tasks: [
          {
            title: 'Packet Sniffer in Rust',
            description: 'Network packet analyzer with protocol decoding',
            details: `# Packet Sniffer in Rust

\`\`\`rust
use std::net::Ipv4Addr;
use std::io;

// Protocol numbers
const IPPROTO_ICMP: u8 = 1;
const IPPROTO_TCP: u8 = 6;
const IPPROTO_UDP: u8 = 17;

// Ethernet header
#[derive(Debug)]
struct EthernetHeader {
    dst_mac: [u8; 6],
    src_mac: [u8; 6],
    ethertype: u16,
}

impl EthernetHeader {
    fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 14 {
            return None;
        }
        Some(EthernetHeader {
            dst_mac: data[0..6].try_into().ok()?,
            src_mac: data[6..12].try_into().ok()?,
            ethertype: u16::from_be_bytes([data[12], data[13]]),
        })
    }

    fn format_mac(mac: &[u8; 6]) -> String {
        format!("{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5])
    }
}

// IPv4 header
#[derive(Debug)]
struct Ipv4Header {
    version: u8,
    ihl: u8,
    tos: u8,
    total_length: u16,
    identification: u16,
    flags: u8,
    fragment_offset: u16,
    ttl: u8,
    protocol: u8,
    checksum: u16,
    src_addr: Ipv4Addr,
    dst_addr: Ipv4Addr,
}

impl Ipv4Header {
    fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 20 {
            return None;
        }
        Some(Ipv4Header {
            version: data[0] >> 4,
            ihl: data[0] & 0x0f,
            tos: data[1],
            total_length: u16::from_be_bytes([data[2], data[3]]),
            identification: u16::from_be_bytes([data[4], data[5]]),
            flags: data[6] >> 5,
            fragment_offset: u16::from_be_bytes([data[6] & 0x1f, data[7]]),
            ttl: data[8],
            protocol: data[9],
            checksum: u16::from_be_bytes([data[10], data[11]]),
            src_addr: Ipv4Addr::new(data[12], data[13], data[14], data[15]),
            dst_addr: Ipv4Addr::new(data[16], data[17], data[18], data[19]),
        })
    }

    fn header_length(&self) -> usize {
        (self.ihl as usize) * 4
    }
}

// TCP header
#[derive(Debug)]
struct TcpHeader {
    src_port: u16,
    dst_port: u16,
    seq_num: u32,
    ack_num: u32,
    data_offset: u8,
    flags: TcpFlags,
    window: u16,
    checksum: u16,
    urgent_ptr: u16,
}

#[derive(Debug)]
struct TcpFlags {
    fin: bool,
    syn: bool,
    rst: bool,
    psh: bool,
    ack: bool,
    urg: bool,
}

impl TcpHeader {
    fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 20 {
            return None;
        }
        let flags_byte = data[13];
        Some(TcpHeader {
            src_port: u16::from_be_bytes([data[0], data[1]]),
            dst_port: u16::from_be_bytes([data[2], data[3]]),
            seq_num: u32::from_be_bytes([data[4], data[5], data[6], data[7]]),
            ack_num: u32::from_be_bytes([data[8], data[9], data[10], data[11]]),
            data_offset: data[12] >> 4,
            flags: TcpFlags {
                fin: flags_byte & 0x01 != 0,
                syn: flags_byte & 0x02 != 0,
                rst: flags_byte & 0x04 != 0,
                psh: flags_byte & 0x08 != 0,
                ack: flags_byte & 0x10 != 0,
                urg: flags_byte & 0x20 != 0,
            },
            window: u16::from_be_bytes([data[14], data[15]]),
            checksum: u16::from_be_bytes([data[16], data[17]]),
            urgent_ptr: u16::from_be_bytes([data[18], data[19]]),
        })
    }

    fn header_length(&self) -> usize {
        (self.data_offset as usize) * 4
    }

    fn flags_string(&self) -> String {
        let mut s = String::new();
        if self.flags.syn { s.push_str("SYN "); }
        if self.flags.ack { s.push_str("ACK "); }
        if self.flags.fin { s.push_str("FIN "); }
        if self.flags.rst { s.push_str("RST "); }
        if self.flags.psh { s.push_str("PSH "); }
        if self.flags.urg { s.push_str("URG "); }
        s.trim().to_string()
    }
}

// UDP header
#[derive(Debug)]
struct UdpHeader {
    src_port: u16,
    dst_port: u16,
    length: u16,
    checksum: u16,
}

impl UdpHeader {
    fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }
        Some(UdpHeader {
            src_port: u16::from_be_bytes([data[0], data[1]]),
            dst_port: u16::from_be_bytes([data[2], data[3]]),
            length: u16::from_be_bytes([data[4], data[5]]),
            checksum: u16::from_be_bytes([data[6], data[7]]),
        })
    }
}

// DNS header
#[derive(Debug)]
struct DnsHeader {
    id: u16,
    flags: u16,
    questions: u16,
    answers: u16,
    authority: u16,
    additional: u16,
}

impl DnsHeader {
    fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }
        Some(DnsHeader {
            id: u16::from_be_bytes([data[0], data[1]]),
            flags: u16::from_be_bytes([data[2], data[3]]),
            questions: u16::from_be_bytes([data[4], data[5]]),
            answers: u16::from_be_bytes([data[6], data[7]]),
            authority: u16::from_be_bytes([data[8], data[9]]),
            additional: u16::from_be_bytes([data[10], data[11]]),
        })
    }

    fn is_query(&self) -> bool {
        self.flags & 0x8000 == 0
    }
}

// Packet analysis
struct PacketAnalyzer {
    packet_count: u64,
    tcp_count: u64,
    udp_count: u64,
    icmp_count: u64,
}

impl PacketAnalyzer {
    fn new() -> Self {
        PacketAnalyzer {
            packet_count: 0,
            tcp_count: 0,
            udp_count: 0,
            icmp_count: 0,
        }
    }

    fn analyze(&mut self, data: &[u8]) {
        self.packet_count += 1;

        // Parse Ethernet
        let eth = match EthernetHeader::parse(data) {
            Some(e) => e,
            None => return,
        };

        // Only handle IPv4 (0x0800)
        if eth.ethertype != 0x0800 {
            return;
        }

        // Parse IP
        let ip_data = &data[14..];
        let ip = match Ipv4Header::parse(ip_data) {
            Some(i) => i,
            None => return,
        };

        let transport_data = &ip_data[ip.header_length()..];

        match ip.protocol {
            IPPROTO_TCP => {
                self.tcp_count += 1;
                if let Some(tcp) = TcpHeader::parse(transport_data) {
                    println!("TCP {} {}:{} -> {}:{} [{}]",
                        self.packet_count,
                        ip.src_addr, tcp.src_port,
                        ip.dst_addr, tcp.dst_port,
                        tcp.flags_string());

                    // Check for HTTP
                    let payload = &transport_data[tcp.header_length()..];
                    if payload.starts_with(b"GET ") || payload.starts_with(b"POST ") ||
                       payload.starts_with(b"HTTP/") {
                        if let Ok(s) = std::str::from_utf8(&payload[..payload.len().min(100)]) {
                            println!("  HTTP: {}", s.lines().next().unwrap_or(""));
                        }
                    }
                }
            }
            IPPROTO_UDP => {
                self.udp_count += 1;
                if let Some(udp) = UdpHeader::parse(transport_data) {
                    println!("UDP {} {}:{} -> {}:{} len={}",
                        self.packet_count,
                        ip.src_addr, udp.src_port,
                        ip.dst_addr, udp.dst_port,
                        udp.length);

                    // Check for DNS
                    if udp.src_port == 53 || udp.dst_port == 53 {
                        let dns_data = &transport_data[8..];
                        if let Some(dns) = DnsHeader::parse(dns_data) {
                            println!("  DNS: {} questions={} answers={}",
                                if dns.is_query() { "Query" } else { "Response" },
                                dns.questions, dns.answers);
                        }
                    }
                }
            }
            IPPROTO_ICMP => {
                self.icmp_count += 1;
                let icmp_type = transport_data.get(0).unwrap_or(&0);
                let icmp_code = transport_data.get(1).unwrap_or(&0);
                println!("ICMP {} {} -> {} type={} code={}",
                    self.packet_count,
                    ip.src_addr, ip.dst_addr,
                    icmp_type, icmp_code);
            }
            _ => {
                println!("IP {} {} -> {} proto={}",
                    self.packet_count,
                    ip.src_addr, ip.dst_addr,
                    ip.protocol);
            }
        }
    }

    fn print_stats(&self) {
        println!("\\n=== Statistics ===");
        println!("Total packets: {}", self.packet_count);
        println!("TCP: {}", self.tcp_count);
        println!("UDP: {}", self.udp_count);
        println!("ICMP: {}", self.icmp_count);
    }
}

// Hex dump utility
fn hexdump(data: &[u8], offset: usize) {
    for (i, chunk) in data.chunks(16).enumerate() {
        print!("{:08x}  ", offset + i * 16);

        for (j, byte) in chunk.iter().enumerate() {
            print!("{:02x} ", byte);
            if j == 7 { print!(" "); }
        }

        for _ in chunk.len()..16 {
            print!("   ");
            if chunk.len() <= 8 { print!(" "); }
        }

        print!(" |");
        for byte in chunk {
            if *byte >= 0x20 && *byte < 0x7f {
                print!("{}", *byte as char);
            } else {
                print!(".");
            }
        }
        println!("|");
    }
}

// Platform-specific capture (Linux)
#[cfg(target_os = "linux")]
mod capture {
    use std::io;
    use std::os::unix::io::RawFd;

    pub fn create_raw_socket() -> io::Result<RawFd> {
        use libc::{socket, AF_PACKET, SOCK_RAW, ETH_P_ALL};

        let fd = unsafe { socket(AF_PACKET, SOCK_RAW, (ETH_P_ALL as u16).to_be() as i32) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(fd)
    }

    pub fn recv_packet(fd: RawFd, buf: &mut [u8]) -> io::Result<usize> {
        use libc::recv;

        let n = unsafe { recv(fd, buf.as_mut_ptr() as *mut _, buf.len(), 0) };
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(n as usize)
    }
}

fn main() -> io::Result<()> {
    println!("Packet Sniffer - requires root privileges");

    #[cfg(target_os = "linux")]
    {
        let fd = capture::create_raw_socket()?;
        let mut analyzer = PacketAnalyzer::new();
        let mut buf = [0u8; 65535];

        println!("Capturing packets... (Ctrl+C to stop)\\n");

        loop {
            let len = capture::recv_packet(fd, &mut buf)?;
            analyzer.analyze(&buf[..len]);
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        println!("This example requires Linux raw sockets");
        println!("On other platforms, use libpcap");
    }

    Ok(())
}
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Build Your Own DNS Server',
    description: 'Implement a DNS server with recursive resolution, caching, and zone management',
    icon: 'globe',
    color: 'green',
    language: 'Go, Rust, C',
    skills: 'DNS protocol, UDP/TCP, Caching, Zone files',
    difficulty: 'intermediate',
    estimated_weeks: 4,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | DNS basics | Protocol overview |
| 1 | 2 | Message format | Header parsing |
| 1 | 3 | Question section | Query parsing |
| 1 | 4 | Answer section | RR parsing |
| 1 | 5 | Name encoding | Label compression |
| 2 | 1 | UDP server | Listen/respond |
| 2 | 2 | A records | IPv4 addresses |
| 2 | 3 | AAAA records | IPv6 addresses |
| 2 | 4 | CNAME records | Aliases |
| 2 | 5 | MX records | Mail servers |
| 3 | 1 | Zone files | Zone parsing |
| 3 | 2 | Authority | NS records |
| 3 | 3 | Recursive query | Forwarding |
| 3 | 4 | Root hints | Root servers |
| 3 | 5 | Iterative query | Full resolution |
| 4 | 1 | Caching | TTL management |
| 4 | 2 | Negative cache | NXDOMAIN |
| 4 | 3 | TCP fallback | Large responses |
| 4 | 4 | EDNS0 | Extensions |
| 4 | 5 | Integration | Full DNS server |`,
    modules: [
      {
        name: 'DNS Protocol',
        description: 'DNS message parsing and generation',
        tasks: [
          {
            title: 'DNS Server in Go',
            description: 'Recursive DNS resolver with caching',
            details: `# DNS Server in Go

\`\`\`go
package main

import (
	"encoding/binary"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"
)

// DNS constants
const (
	TypeA     = 1
	TypeAAAA  = 28
	TypeCNAME = 5
	TypeMX    = 15
	TypeNS    = 2
	TypeSOA   = 6
	TypeTXT   = 16

	ClassIN = 1

	FlagQR     = 1 << 15
	FlagAA     = 1 << 10
	FlagTC     = 1 << 9
	FlagRD     = 1 << 8
	FlagRA     = 1 << 7
	RcodeOK    = 0
	RcodeNXDomain = 3
)

// DNS header
type Header struct {
	ID      uint16
	Flags   uint16
	QDCount uint16
	ANCount uint16
	NSCount uint16
	ARCount uint16
}

func (h *Header) Pack() []byte {
	buf := make([]byte, 12)
	binary.BigEndian.PutUint16(buf[0:2], h.ID)
	binary.BigEndian.PutUint16(buf[2:4], h.Flags)
	binary.BigEndian.PutUint16(buf[4:6], h.QDCount)
	binary.BigEndian.PutUint16(buf[6:8], h.ANCount)
	binary.BigEndian.PutUint16(buf[8:10], h.NSCount)
	binary.BigEndian.PutUint16(buf[10:12], h.ARCount)
	return buf
}

func ParseHeader(data []byte) Header {
	return Header{
		ID:      binary.BigEndian.Uint16(data[0:2]),
		Flags:   binary.BigEndian.Uint16(data[2:4]),
		QDCount: binary.BigEndian.Uint16(data[4:6]),
		ANCount: binary.BigEndian.Uint16(data[6:8]),
		NSCount: binary.BigEndian.Uint16(data[8:10]),
		ARCount: binary.BigEndian.Uint16(data[10:12]),
	}
}

// DNS question
type Question struct {
	Name  string
	Type  uint16
	Class uint16
}

// DNS resource record
type RR struct {
	Name     string
	Type     uint16
	Class    uint16
	TTL      uint32
	RDLength uint16
	RData    []byte
}

func (rr *RR) Pack(msg []byte) []byte {
	buf := EncodeName(rr.Name)
	buf = append(buf, make([]byte, 10)...)
	offset := len(buf) - 10
	binary.BigEndian.PutUint16(buf[offset:], rr.Type)
	binary.BigEndian.PutUint16(buf[offset+2:], rr.Class)
	binary.BigEndian.PutUint32(buf[offset+4:], rr.TTL)
	binary.BigEndian.PutUint16(buf[offset+8:], uint16(len(rr.RData)))
	buf = append(buf, rr.RData...)
	return buf
}

// Encode domain name
func EncodeName(name string) []byte {
	var buf []byte
	for _, label := range strings.Split(name, ".") {
		if len(label) == 0 {
			continue
		}
		buf = append(buf, byte(len(label)))
		buf = append(buf, []byte(label)...)
	}
	buf = append(buf, 0)
	return buf
}

// Decode domain name
func DecodeName(data []byte, offset int) (string, int) {
	var labels []string
	ptr := offset
	jumped := false
	origOffset := offset

	for {
		if ptr >= len(data) {
			break
		}

		length := int(data[ptr])
		if length == 0 {
			ptr++
			break
		}

		// Check for compression pointer
		if length&0xC0 == 0xC0 {
			if !jumped {
				origOffset = ptr + 2
			}
			ptr = int(binary.BigEndian.Uint16(data[ptr:ptr+2]) & 0x3FFF)
			jumped = true
			continue
		}

		ptr++
		labels = append(labels, string(data[ptr:ptr+length]))
		ptr += length
	}

	if jumped {
		return strings.Join(labels, "."), origOffset
	}
	return strings.Join(labels, "."), ptr
}

// Parse question
func ParseQuestion(data []byte, offset int) (Question, int) {
	name, newOffset := DecodeName(data, offset)
	q := Question{
		Name:  name,
		Type:  binary.BigEndian.Uint16(data[newOffset : newOffset+2]),
		Class: binary.BigEndian.Uint16(data[newOffset+2 : newOffset+4]),
	}
	return q, newOffset + 4
}

// Cache entry
type CacheEntry struct {
	RRs       []RR
	ExpiresAt time.Time
}

// DNS server
type DNSServer struct {
	cache     map[string]CacheEntry
	cacheLock sync.RWMutex
	zones     map[string][]RR
	upstream  string
}

func NewDNSServer(upstream string) *DNSServer {
	return &DNSServer{
		cache:    make(map[string]CacheEntry),
		zones:    make(map[string][]RR),
		upstream: upstream,
	}
}

// Add zone records
func (s *DNSServer) AddZone(name string, rrs []RR) {
	s.zones[strings.ToLower(name)] = rrs
}

// Cache lookup
func (s *DNSServer) CacheLookup(name string, qtype uint16) ([]RR, bool) {
	s.cacheLock.RLock()
	defer s.cacheLock.RUnlock()

	key := fmt.Sprintf("%s:%d", strings.ToLower(name), qtype)
	entry, ok := s.cache[key]
	if !ok {
		return nil, false
	}

	if time.Now().After(entry.ExpiresAt) {
		return nil, false
	}

	return entry.RRs, true
}

// Cache store
func (s *DNSServer) CacheStore(name string, qtype uint16, rrs []RR) {
	if len(rrs) == 0 {
		return
	}

	s.cacheLock.Lock()
	defer s.cacheLock.Unlock()

	key := fmt.Sprintf("%s:%d", strings.ToLower(name), qtype)
	minTTL := rrs[0].TTL
	for _, rr := range rrs {
		if rr.TTL < minTTL {
			minTTL = rr.TTL
		}
	}

	s.cache[key] = CacheEntry{
		RRs:       rrs,
		ExpiresAt: time.Now().Add(time.Duration(minTTL) * time.Second),
	}
}

// Resolve query
func (s *DNSServer) Resolve(q Question) []RR {
	// Check local zones first
	if rrs, ok := s.zones[strings.ToLower(q.Name)]; ok {
		var result []RR
		for _, rr := range rrs {
			if rr.Type == q.Type || q.Type == 255 { // 255 = ANY
				result = append(result, rr)
			}
		}
		if len(result) > 0 {
			return result
		}
	}

	// Check cache
	if rrs, ok := s.CacheLookup(q.Name, q.Type); ok {
		return rrs
	}

	// Forward to upstream
	rrs := s.ForwardQuery(q)
	s.CacheStore(q.Name, q.Type, rrs)
	return rrs
}

// Forward query to upstream
func (s *DNSServer) ForwardQuery(q Question) []RR {
	// Build query
	header := Header{
		ID:      uint16(time.Now().UnixNano() & 0xFFFF),
		Flags:   FlagRD, // Recursion desired
		QDCount: 1,
	}

	msg := header.Pack()
	msg = append(msg, EncodeName(q.Name)...)
	msg = append(msg, make([]byte, 4)...)
	binary.BigEndian.PutUint16(msg[len(msg)-4:], q.Type)
	binary.BigEndian.PutUint16(msg[len(msg)-2:], q.Class)

	// Send query
	conn, err := net.Dial("udp", s.upstream)
	if err != nil {
		return nil
	}
	defer conn.Close()

	conn.SetDeadline(time.Now().Add(5 * time.Second))
	conn.Write(msg)

	// Read response
	buf := make([]byte, 512)
	n, err := conn.Read(buf)
	if err != nil {
		return nil
	}

	return s.ParseResponse(buf[:n])
}

// Parse response
func (s *DNSServer) ParseResponse(data []byte) []RR {
	header := ParseHeader(data)
	offset := 12

	// Skip questions
	for i := 0; i < int(header.QDCount); i++ {
		_, offset = ParseQuestion(data, offset)
	}

	// Parse answers
	var rrs []RR
	for i := 0; i < int(header.ANCount); i++ {
		name, newOffset := DecodeName(data, offset)
		rr := RR{
			Name:     name,
			Type:     binary.BigEndian.Uint16(data[newOffset : newOffset+2]),
			Class:    binary.BigEndian.Uint16(data[newOffset+2 : newOffset+4]),
			TTL:      binary.BigEndian.Uint32(data[newOffset+4 : newOffset+8]),
			RDLength: binary.BigEndian.Uint16(data[newOffset+8 : newOffset+10]),
		}
		rr.RData = data[newOffset+10 : newOffset+10+int(rr.RDLength)]
		rrs = append(rrs, rr)
		offset = newOffset + 10 + int(rr.RDLength)
	}

	return rrs
}

// Handle query
func (s *DNSServer) HandleQuery(data []byte, addr *net.UDPAddr, conn *net.UDPConn) {
	header := ParseHeader(data)
	q, _ := ParseQuestion(data, 12)

	fmt.Printf("Query: %s %d from %s\\n", q.Name, q.Type, addr.String())

	// Resolve
	answers := s.Resolve(q)

	// Build response
	respHeader := Header{
		ID:      header.ID,
		Flags:   FlagQR | FlagAA | FlagRA,
		QDCount: 1,
		ANCount: uint16(len(answers)),
	}

	if len(answers) == 0 {
		respHeader.Flags |= RcodeNXDomain
	}

	resp := respHeader.Pack()

	// Copy question
	resp = append(resp, EncodeName(q.Name)...)
	resp = append(resp, make([]byte, 4)...)
	binary.BigEndian.PutUint16(resp[len(resp)-4:], q.Type)
	binary.BigEndian.PutUint16(resp[len(resp)-2:], q.Class)

	// Add answers
	for _, rr := range answers {
		resp = append(resp, rr.Pack(resp)...)
	}

	conn.WriteToUDP(resp, addr)
}

// Start server
func (s *DNSServer) Start(addr string) error {
	udpAddr, err := net.ResolveUDPAddr("udp", addr)
	if err != nil {
		return err
	}

	conn, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		return err
	}
	defer conn.Close()

	fmt.Printf("DNS server listening on %s\\n", addr)

	buf := make([]byte, 512)
	for {
		n, clientAddr, err := conn.ReadFromUDP(buf)
		if err != nil {
			continue
		}

		go s.HandleQuery(buf[:n], clientAddr, conn)
	}
}

func main() {
	server := NewDNSServer("8.8.8.8:53")

	// Add local zone
	server.AddZone("example.local", []RR{
		{
			Name:  "example.local",
			Type:  TypeA,
			Class: ClassIN,
			TTL:   300,
			RData: net.ParseIP("192.168.1.100").To4(),
		},
	})

	server.Start(":5353")
}
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Build Your Own Load Balancer',
    description: 'Create an L4/L7 load balancer with health checks and multiple algorithms',
    icon: 'sliders',
    color: 'orange',
    language: 'Go, Rust, C',
    skills: 'Networking, Proxying, Health checks, Load balancing algorithms',
    difficulty: 'intermediate',
    estimated_weeks: 4,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Architecture | Design |
| 1 | 2 | TCP proxy | Basic proxying |
| 1 | 3 | Backend pool | Server list |
| 1 | 4 | Round robin | RR algorithm |
| 1 | 5 | Weighted RR | Weights |
| 2 | 1 | Least connections | LC algorithm |
| 2 | 2 | IP hash | Sticky sessions |
| 2 | 3 | Random | Random selection |
| 2 | 4 | Health checks | TCP checks |
| 2 | 5 | HTTP checks | Endpoint checks |
| 3 | 1 | L7 proxy | HTTP parsing |
| 3 | 2 | Host routing | Virtual hosts |
| 3 | 3 | Path routing | URL paths |
| 3 | 4 | Header routing | Custom headers |
| 3 | 5 | URL rewriting | Path rewrite |
| 4 | 1 | Connection pooling | Reuse connections |
| 4 | 2 | Timeouts | Read/write timeouts |
| 4 | 3 | Rate limiting | Request limits |
| 4 | 4 | TLS termination | SSL/TLS |
| 4 | 5 | Integration | Full LB |`,
    modules: [
      {
        name: 'Load Balancer Core',
        description: 'TCP/HTTP load balancer',
        tasks: [
          {
            title: 'Load Balancer in Go',
            description: 'L7 load balancer with multiple algorithms',
            details: `# Load Balancer in Go

\`\`\`go
package main

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
	"sync/atomic"
	"time"
)

// Backend represents a backend server
type Backend struct {
	URL          *url.URL
	Alive        bool
	Weight       int
	Connections  int64
	mutex        sync.RWMutex
	ReverseProxy *httputil.ReverseProxy
}

func (b *Backend) SetAlive(alive bool) {
	b.mutex.Lock()
	b.Alive = alive
	b.mutex.Unlock()
}

func (b *Backend) IsAlive() bool {
	b.mutex.RLock()
	alive := b.Alive
	b.mutex.RUnlock()
	return alive
}

// ServerPool holds all backends
type ServerPool struct {
	backends []*Backend
	current  uint64
	mutex    sync.RWMutex
}

func (s *ServerPool) AddBackend(backend *Backend) {
	s.mutex.Lock()
	s.backends = append(s.backends, backend)
	s.mutex.Unlock()
}

// Round Robin
func (s *ServerPool) NextRoundRobin() *Backend {
	next := atomic.AddUint64(&s.current, 1)
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	for i := 0; i < len(s.backends); i++ {
		idx := int(next+uint64(i)) % len(s.backends)
		if s.backends[idx].IsAlive() {
			return s.backends[idx]
		}
	}
	return nil
}

// Weighted Round Robin
func (s *ServerPool) NextWeightedRR() *Backend {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	totalWeight := 0
	for _, b := range s.backends {
		if b.IsAlive() {
			totalWeight += b.Weight
		}
	}

	if totalWeight == 0 {
		return nil
	}

	next := int(atomic.AddUint64(&s.current, 1)) % totalWeight

	for _, b := range s.backends {
		if !b.IsAlive() {
			continue
		}
		next -= b.Weight
		if next < 0 {
			return b
		}
	}

	return nil
}

// Least Connections
func (s *ServerPool) NextLeastConnections() *Backend {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	var best *Backend
	minConns := int64(^uint64(0) >> 1)

	for _, b := range s.backends {
		if b.IsAlive() {
			conns := atomic.LoadInt64(&b.Connections)
			if conns < minConns {
				minConns = conns
				best = b
			}
		}
	}

	return best
}

// IP Hash for sticky sessions
func (s *ServerPool) NextIPHash(ip string) *Backend {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	var aliveBackends []*Backend
	for _, b := range s.backends {
		if b.IsAlive() {
			aliveBackends = append(aliveBackends, b)
		}
	}

	if len(aliveBackends) == 0 {
		return nil
	}

	hash := uint64(0)
	for _, c := range ip {
		hash = hash*31 + uint64(c)
	}

	return aliveBackends[hash%uint64(len(aliveBackends))]
}

// Health checker
func (s *ServerPool) HealthCheck() {
	for _, b := range s.backends {
		go func(backend *Backend) {
			alive := checkBackendHealth(backend.URL)
			backend.SetAlive(alive)
			status := "up"
			if !alive {
				status = "down"
			}
			fmt.Printf("Backend %s is %s\\n", backend.URL, status)
		}(b)
	}
}

func checkBackendHealth(u *url.URL) bool {
	timeout := 2 * time.Second
	conn, err := net.DialTimeout("tcp", u.Host, timeout)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// HTTP health check
func checkBackendHTTP(u *url.URL, path string) bool {
	client := http.Client{
		Timeout: 2 * time.Second,
	}

	healthURL := fmt.Sprintf("%s://%s%s", u.Scheme, u.Host, path)
	resp, err := client.Get(healthURL)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode >= 200 && resp.StatusCode < 400
}

// Load balancer
type LoadBalancer struct {
	pool       *ServerPool
	algorithm  string
	listenAddr string
}

func NewLoadBalancer(addr, algorithm string) *LoadBalancer {
	return &LoadBalancer{
		pool:       &ServerPool{},
		algorithm:  algorithm,
		listenAddr: addr,
	}
}

func (lb *LoadBalancer) AddBackend(urlStr string, weight int) error {
	u, err := url.Parse(urlStr)
	if err != nil {
		return err
	}

	proxy := httputil.NewSingleHostReverseProxy(u)
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, e error) {
		fmt.Printf("Proxy error: %v\\n", e)
		http.Error(w, "Service Unavailable", http.StatusServiceUnavailable)
	}

	backend := &Backend{
		URL:          u,
		Alive:        true,
		Weight:       weight,
		ReverseProxy: proxy,
	}

	lb.pool.AddBackend(backend)
	return nil
}

func (lb *LoadBalancer) getNextBackend(r *http.Request) *Backend {
	switch lb.algorithm {
	case "round-robin":
		return lb.pool.NextRoundRobin()
	case "weighted-round-robin":
		return lb.pool.NextWeightedRR()
	case "least-connections":
		return lb.pool.NextLeastConnections()
	case "ip-hash":
		ip, _, _ := net.SplitHostPort(r.RemoteAddr)
		return lb.pool.NextIPHash(ip)
	default:
		return lb.pool.NextRoundRobin()
	}
}

func (lb *LoadBalancer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	backend := lb.getNextBackend(r)
	if backend == nil {
		http.Error(w, "Service Unavailable", http.StatusServiceUnavailable)
		return
	}

	atomic.AddInt64(&backend.Connections, 1)
	defer atomic.AddInt64(&backend.Connections, -1)

	fmt.Printf("Forwarding to %s\\n", backend.URL)
	backend.ReverseProxy.ServeHTTP(w, r)
}

func (lb *LoadBalancer) StartHealthCheck(interval time.Duration) {
	ticker := time.NewTicker(interval)
	go func() {
		for range ticker.C {
			lb.pool.HealthCheck()
		}
	}()
}

func (lb *LoadBalancer) Start() error {
	lb.StartHealthCheck(10 * time.Second)
	fmt.Printf("Load balancer listening on %s (algorithm: %s)\\n",
		lb.listenAddr, lb.algorithm)
	return http.ListenAndServe(lb.listenAddr, lb)
}

// L4 (TCP) Load Balancer
type TCPLoadBalancer struct {
	pool       *ServerPool
	listenAddr string
}

func NewTCPLoadBalancer(addr string) *TCPLoadBalancer {
	return &TCPLoadBalancer{
		pool:       &ServerPool{},
		listenAddr: addr,
	}
}

func (lb *TCPLoadBalancer) AddBackend(addr string) {
	u, _ := url.Parse("tcp://" + addr)
	lb.pool.AddBackend(&Backend{URL: u, Alive: true, Weight: 1})
}

func (lb *TCPLoadBalancer) handleConnection(clientConn net.Conn) {
	defer clientConn.Close()

	backend := lb.pool.NextLeastConnections()
	if backend == nil {
		return
	}

	atomic.AddInt64(&backend.Connections, 1)
	defer atomic.AddInt64(&backend.Connections, -1)

	backendConn, err := net.Dial("tcp", backend.URL.Host)
	if err != nil {
		backend.SetAlive(false)
		return
	}
	defer backendConn.Close()

	// Bidirectional copy
	done := make(chan struct{})

	go func() {
		io.Copy(backendConn, clientConn)
		done <- struct{}{}
	}()

	go func() {
		io.Copy(clientConn, backendConn)
		done <- struct{}{}
	}()

	<-done
}

func (lb *TCPLoadBalancer) Start() error {
	listener, err := net.Listen("tcp", lb.listenAddr)
	if err != nil {
		return err
	}
	defer listener.Close()

	fmt.Printf("TCP load balancer listening on %s\\n", lb.listenAddr)

	for {
		conn, err := listener.Accept()
		if err != nil {
			continue
		}
		go lb.handleConnection(conn)
	}
}

func main() {
	// HTTP Load Balancer
	lb := NewLoadBalancer(":8080", "round-robin")
	lb.AddBackend("http://localhost:8081", 1)
	lb.AddBackend("http://localhost:8082", 1)
	lb.AddBackend("http://localhost:8083", 1)

	lb.Start()
}
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Build Your Own HTTP Server',
    description: 'Create an HTTP/1.1 and HTTP/2 server with routing, middleware, and static file serving',
    icon: 'server',
    color: 'purple',
    language: 'C, Rust, Go, C++',
    skills: 'HTTP protocol, Socket programming, Concurrency, TLS',
    difficulty: 'intermediate',
    estimated_weeks: 5,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | TCP server | Accept connections |
| 1 | 2 | Request parsing | Parse HTTP |
| 1 | 3 | Response building | HTTP response |
| 1 | 4 | Status codes | Error handling |
| 1 | 5 | Headers | Header parsing |
| 2 | 1 | GET/POST | Method handling |
| 2 | 2 | Query strings | URL parsing |
| 2 | 3 | Request body | Content-Length |
| 2 | 4 | Chunked transfer | Chunked encoding |
| 2 | 5 | Keep-alive | Persistent conn |
| 3 | 1 | Router | URL routing |
| 3 | 2 | Path params | /users/:id |
| 3 | 3 | Middleware | Request pipeline |
| 3 | 4 | Logging | Access logs |
| 3 | 5 | Compression | gzip |
| 4 | 1 | Static files | File serving |
| 4 | 2 | MIME types | Content-Type |
| 4 | 3 | Caching | ETag, If-Modified |
| 4 | 4 | Range requests | Partial content |
| 4 | 5 | Directory listing | Index pages |
| 5 | 1 | TLS/HTTPS | SSL certificates |
| 5 | 2 | Virtual hosts | Host header |
| 5 | 3 | WebSocket | Upgrade |
| 5 | 4 | HTTP/2 basics | HPACK, streams |
| 5 | 5 | Integration | Full server |`,
    modules: [
      {
        name: 'HTTP Server Core',
        description: 'Complete HTTP server implementation',
        tasks: [
          {
            title: 'HTTP Server in Rust',
            description: 'Build HTTP/1.1 server with routing',
            details: `# HTTP Server in Rust

\`\`\`rust
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::Arc;
use std::thread;

// HTTP Request
#[derive(Debug)]
struct Request {
    method: String,
    path: String,
    version: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
    params: HashMap<String, String>,
    query: HashMap<String, String>,
}

impl Request {
    fn parse(stream: &mut TcpStream) -> Option<Self> {
        let mut reader = BufReader::new(stream.try_clone().ok()?);

        // Read request line
        let mut request_line = String::new();
        reader.read_line(&mut request_line).ok()?;
        let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
        if parts.len() < 3 {
            return None;
        }

        let method = parts[0].to_string();
        let full_path = parts[1].to_string();
        let version = parts[2].to_string();

        // Parse path and query string
        let (path, query) = if let Some(idx) = full_path.find('?') {
            let path = full_path[..idx].to_string();
            let query_str = &full_path[idx + 1..];
            let query = parse_query_string(query_str);
            (path, query)
        } else {
            (full_path, HashMap::new())
        };

        // Read headers
        let mut headers = HashMap::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).ok()?;
            let line = line.trim();
            if line.is_empty() {
                break;
            }
            if let Some(idx) = line.find(':') {
                let key = line[..idx].trim().to_lowercase();
                let value = line[idx + 1..].trim().to_string();
                headers.insert(key, value);
            }
        }

        // Read body if Content-Length present
        let mut body = Vec::new();
        if let Some(len) = headers.get("content-length") {
            if let Ok(len) = len.parse::<usize>() {
                body.resize(len, 0);
                reader.read_exact(&mut body).ok()?;
            }
        }

        Some(Request {
            method,
            path,
            version,
            headers,
            body,
            params: HashMap::new(),
            query,
        })
    }
}

fn parse_query_string(s: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for pair in s.split('&') {
        if let Some(idx) = pair.find('=') {
            let key = urlDecode(&pair[..idx]);
            let value = urlDecode(&pair[idx + 1..]);
            map.insert(key, value);
        }
    }
    map
}

fn urlDecode(s: &str) -> String {
    // Simplified URL decoding
    s.replace("+", " ")
        .replace("%20", " ")
        .replace("%2F", "/")
}

// HTTP Response
struct Response {
    status: u16,
    status_text: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl Response {
    fn new(status: u16) -> Self {
        let status_text = match status {
            200 => "OK",
            201 => "Created",
            204 => "No Content",
            301 => "Moved Permanently",
            302 => "Found",
            304 => "Not Modified",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            500 => "Internal Server Error",
            _ => "Unknown",
        };

        Response {
            status,
            status_text: status_text.to_string(),
            headers: HashMap::new(),
            body: Vec::new(),
        }
    }

    fn header(&mut self, key: &str, value: &str) -> &mut Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    fn body(&mut self, body: Vec<u8>) -> &mut Self {
        self.body = body;
        self
    }

    fn text(&mut self, text: &str) -> &mut Self {
        self.header("Content-Type", "text/plain; charset=utf-8");
        self.body = text.as_bytes().to_vec();
        self
    }

    fn html(&mut self, html: &str) -> &mut Self {
        self.header("Content-Type", "text/html; charset=utf-8");
        self.body = html.as_bytes().to_vec();
        self
    }

    fn json(&mut self, json: &str) -> &mut Self {
        self.header("Content-Type", "application/json");
        self.body = json.as_bytes().to_vec();
        self
    }

    fn send(&mut self, stream: &mut TcpStream) -> std::io::Result<()> {
        self.header("Content-Length", &self.body.len().to_string());
        self.header("Connection", "close");

        let mut response = format!(
            "HTTP/1.1 {} {}\\r\\n",
            self.status, self.status_text
        );

        for (key, value) in &self.headers {
            response.push_str(&format!("{}: {}\\r\\n", key, value));
        }
        response.push_str("\\r\\n");

        stream.write_all(response.as_bytes())?;
        stream.write_all(&self.body)?;
        stream.flush()
    }
}

// Route handler
type Handler = Box<dyn Fn(&Request, &mut Response) + Send + Sync>;

struct Route {
    method: String,
    pattern: String,
    handler: Handler,
    params: Vec<String>,
}

impl Route {
    fn matches(&self, method: &str, path: &str) -> Option<HashMap<String, String>> {
        if self.method != method && self.method != "*" {
            return None;
        }

        let pattern_parts: Vec<&str> = self.pattern.split('/').collect();
        let path_parts: Vec<&str> = path.split('/').collect();

        if pattern_parts.len() != path_parts.len() {
            // Check for wildcard
            if !self.pattern.ends_with("/*") {
                return None;
            }
        }

        let mut params = HashMap::new();

        for (i, pattern) in pattern_parts.iter().enumerate() {
            if pattern.starts_with(':') {
                // Path parameter
                let param_name = &pattern[1..];
                if i < path_parts.len() {
                    params.insert(param_name.to_string(), path_parts[i].to_string());
                }
            } else if *pattern == "*" {
                // Wildcard - match rest of path
                break;
            } else if i >= path_parts.len() || *pattern != path_parts[i] {
                return None;
            }
        }

        Some(params)
    }
}

// Middleware
type Middleware = Box<dyn Fn(&Request, &mut Response) -> bool + Send + Sync>;

// HTTP Server
struct Server {
    routes: Vec<Route>,
    middleware: Vec<Middleware>,
    static_dir: Option<String>,
}

impl Server {
    fn new() -> Self {
        Server {
            routes: Vec::new(),
            middleware: Vec::new(),
            static_dir: None,
        }
    }

    fn get<F>(&mut self, path: &str, handler: F) -> &mut Self
    where
        F: Fn(&Request, &mut Response) + Send + Sync + 'static,
    {
        self.add_route("GET", path, handler)
    }

    fn post<F>(&mut self, path: &str, handler: F) -> &mut Self
    where
        F: Fn(&Request, &mut Response) + Send + Sync + 'static,
    {
        self.add_route("POST", path, handler)
    }

    fn add_route<F>(&mut self, method: &str, path: &str, handler: F) -> &mut Self
    where
        F: Fn(&Request, &mut Response) + Send + Sync + 'static,
    {
        self.routes.push(Route {
            method: method.to_string(),
            pattern: path.to_string(),
            handler: Box::new(handler),
            params: Vec::new(),
        });
        self
    }

    fn use_middleware<F>(&mut self, middleware: F) -> &mut Self
    where
        F: Fn(&Request, &mut Response) -> bool + Send + Sync + 'static,
    {
        self.middleware.push(Box::new(middleware));
        self
    }

    fn static_files(&mut self, dir: &str) -> &mut Self {
        self.static_dir = Some(dir.to_string());
        self
    }

    fn handle_request(&self, mut request: Request, stream: &mut TcpStream) {
        let mut response = Response::new(200);

        // Run middleware
        for mw in &self.middleware {
            if !mw(&request, &mut response) {
                response.send(stream).ok();
                return;
            }
        }

        // Find matching route
        for route in &self.routes {
            if let Some(params) = route.matches(&request.method, &request.path) {
                request.params = params;
                (route.handler)(&request, &mut response);
                response.send(stream).ok();
                return;
            }
        }

        // Try static files
        if let Some(ref dir) = self.static_dir {
            let file_path = format!("{}{}", dir, request.path);
            if let Ok(mut file) = File::open(&file_path) {
                let mut contents = Vec::new();
                file.read_to_end(&mut contents).ok();

                let mime = guess_mime(&file_path);
                response.header("Content-Type", mime);
                response.body(contents);
                response.send(stream).ok();
                return;
            }
        }

        // 404 Not Found
        Response::new(404)
            .text("Not Found")
            .send(stream)
            .ok();
    }

    fn listen(&self, addr: &str) -> std::io::Result<()> {
        let listener = TcpListener::bind(addr)?;
        println!("Server listening on {}", addr);

        let server = Arc::new(self);

        for stream in listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    // In real impl: use thread pool
                    if let Some(request) = Request::parse(&mut stream) {
                        // Clone for thread safety
                        let s = Arc::clone(&server);
                        thread::spawn(move || {
                            // Can't easily share &self, simplified for example
                        });
                        // For simplicity, handle in main thread
                        self.handle_request(request, &mut stream);
                    }
                }
                Err(e) => eprintln!("Connection error: {}", e),
            }
        }

        Ok(())
    }
}

fn guess_mime(path: &str) -> &'static str {
    match Path::new(path).extension().and_then(|e| e.to_str()) {
        Some("html") | Some("htm") => "text/html",
        Some("css") => "text/css",
        Some("js") => "application/javascript",
        Some("json") => "application/json",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("svg") => "image/svg+xml",
        Some("pdf") => "application/pdf",
        _ => "application/octet-stream",
    }
}

// Logging middleware
fn logger(req: &Request, _res: &mut Response) -> bool {
    println!("{} {} {}", req.method, req.path, req.version);
    true // Continue to next middleware/handler
}

fn main() -> std::io::Result<()> {
    let mut server = Server::new();

    server.use_middleware(logger);
    server.static_files("./public");

    server.get("/", |_req, res| {
        res.html("<h1>Hello, World!</h1>");
    });

    server.get("/api/users/:id", |req, res| {
        let id = req.params.get("id").unwrap();
        res.json(&format!(r#"{{"id": {}}}"#, id));
    });

    server.post("/api/users", |req, res| {
        let body = String::from_utf8_lossy(&req.body);
        res.status = 201;
        res.json(&format!(r#"{{"created": true, "data": {}}}"#, body));
    });

    server.listen("127.0.0.1:8080")
}
\`\`\``
          }
        ]
      }
    ]
  }
];

// Insert all data
const insertPath = db.prepare(`
  INSERT INTO paths (name, description, icon, color, language, skills, difficulty, estimated_weeks, schedule)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
`);

const insertModule = db.prepare(`
  INSERT INTO modules (path_id, name, description)
  VALUES (?, ?, ?)
`);

const insertTask = db.prepare(`
  INSERT INTO tasks (module_id, title, description, details)
  VALUES (?, ?, ?, ?)
`);

for (const path of paths) {
  const pathResult = insertPath.run(
    path.name,
    path.description,
    path.icon,
    path.color,
    path.language,
    path.skills,
    path.difficulty,
    path.estimated_weeks,
    path.schedule
  );
  const pathId = pathResult.lastInsertRowid;

  for (const module of path.modules) {
    const moduleResult = insertModule.run(pathId, module.name, module.description);
    const moduleId = moduleResult.lastInsertRowid;

    for (const task of module.tasks) {
      insertTask.run(moduleId, task.title, task.description, task.details);
    }
  }
}

console.log('Seeded: Packet Sniffer, DNS Server, Load Balancer, HTTP Server');
