import Database from 'better-sqlite3';
import { join } from 'path';

const db = new Database(join(process.cwd(), 'data', 'quest-log.db'));

// Database Projects - Redis, SQLite, LSM Tree, B-Tree
const paths = [
  {
    name: 'Build Your Own Redis',
    description: 'Implement Redis from scratch with data structures, persistence, and replication',
    icon: 'database',
    color: 'red',
    language: 'C, Rust, Go',
    skills: 'In-memory databases, Data structures, Networking, Serialization',
    difficulty: 'advanced',
    estimated_weeks: 8,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | RESP protocol | Protocol parsing |
| 1 | 2 | String commands | GET/SET/DEL |
| 1 | 3 | Key expiration | TTL/EXPIRE |
| 1 | 4 | TCP server | Event loop |
| 1 | 5 | Basic server | Working server |
| 2 | 1 | Hash table | Custom hashtable |
| 2 | 2 | Collision handling | Chaining |
| 2 | 3 | Rehashing | Incremental rehash |
| 2 | 4 | Hash commands | HSET/HGET |
| 2 | 5 | Performance | Optimization |
| 3 | 1 | Linked list | Doubly-linked |
| 3 | 2 | List commands | LPUSH/RPOP |
| 3 | 3 | Skip list | Skip list impl |
| 3 | 4 | Sorted set | ZADD/ZRANGE |
| 3 | 5 | Set commands | SADD/SMEMBERS |
| 4 | 1 | RDB basics | Binary format |
| 4 | 2 | RDB save | BGSAVE |
| 4 | 3 | RDB load | Restore data |
| 4 | 4 | AOF basics | Append-only |
| 4 | 5 | AOF rewrite | Compaction |
| 5 | 1 | Pub/Sub | SUBSCRIBE |
| 5 | 2 | Pattern sub | PSUBSCRIBE |
| 5 | 3 | Transactions | MULTI/EXEC |
| 5 | 4 | WATCH | Optimistic lock |
| 5 | 5 | Scripting | Lua eval |
| 6 | 1 | Master setup | Replication |
| 6 | 2 | Slave sync | SYNC command |
| 6 | 3 | Partial resync | PSYNC |
| 6 | 4 | Replication buffer | Backlog |
| 6 | 5 | Failover | Sentinel basics |
| 7 | 1 | Cluster hash | Hash slots |
| 7 | 2 | Node gossip | Cluster bus |
| 7 | 3 | Key routing | MOVED/ASK |
| 7 | 4 | Resharding | Slot migration |
| 7 | 5 | Cluster mgmt | CLUSTER cmds |
| 8 | 1 | Memory mgmt | Eviction |
| 8 | 2 | LRU/LFU | Cache policies |
| 8 | 3 | Streams | XADD/XREAD |
| 8 | 4 | Client tracking | Notifications |
| 8 | 5 | Integration | Full testing |`,
    modules: [
      {
        name: 'Core Implementation',
        description: 'Redis core functionality',
        tasks: [
          {
            title: 'Redis in Rust',
            description: 'Build a Redis clone in Rust',
            details: `# Redis Implementation in Rust

\`\`\`rust
use std::collections::{HashMap, BTreeMap, HashSet, VecDeque};
use std::io::{Read, Write, BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::thread;
use std::fs::{File, OpenOptions};

// ===== RESP Protocol =====

#[derive(Debug, Clone)]
enum RespValue {
    SimpleString(String),
    Error(String),
    Integer(i64),
    BulkString(Option<Vec<u8>>),
    Array(Option<Vec<RespValue>>),
}

impl RespValue {
    fn serialize(&self) -> Vec<u8> {
        match self {
            RespValue::SimpleString(s) => format!("+{}\\r\\n", s).into_bytes(),
            RespValue::Error(s) => format!("-{}\\r\\n", s).into_bytes(),
            RespValue::Integer(i) => format!(":{}\\r\\n", i).into_bytes(),
            RespValue::BulkString(None) => b"\$-1\\r\\n".to_vec(),
            RespValue::BulkString(Some(data)) => {
                let mut result = format!("\${}\\r\\n", data.len()).into_bytes();
                result.extend(data);
                result.extend(b"\\r\\n");
                result
            }
            RespValue::Array(None) => b"*-1\\r\\n".to_vec(),
            RespValue::Array(Some(arr)) => {
                let mut result = format!("*{}\\r\\n", arr.len()).into_bytes();
                for item in arr {
                    result.extend(item.serialize());
                }
                result
            }
        }
    }
}

fn parse_resp(reader: &mut BufReader<&TcpStream>) -> Option<RespValue> {
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 {
        return None;
    }

    let line = line.trim_end_matches("\\r\\n");
    let first = line.chars().next()?;

    match first {
        '+' => Some(RespValue::SimpleString(line[1..].to_string())),
        '-' => Some(RespValue::Error(line[1..].to_string())),
        ':' => Some(RespValue::Integer(line[1..].parse().ok()?)),
        '\$' => {
            let len: i64 = line[1..].parse().ok()?;
            if len < 0 {
                return Some(RespValue::BulkString(None));
            }
            let mut data = vec![0u8; len as usize];
            reader.read_exact(&mut data).ok()?;
            let mut crlf = [0u8; 2];
            reader.read_exact(&mut crlf).ok()?;
            Some(RespValue::BulkString(Some(data)))
        }
        '*' => {
            let count: i64 = line[1..].parse().ok()?;
            if count < 0 {
                return Some(RespValue::Array(None));
            }
            let mut arr = Vec::with_capacity(count as usize);
            for _ in 0..count {
                arr.push(parse_resp(reader)?);
            }
            Some(RespValue::Array(Some(arr)))
        }
        _ => None,
    }
}

// ===== Data Types =====

#[derive(Clone)]
enum RedisValue {
    String(Vec<u8>),
    List(VecDeque<Vec<u8>>),
    Set(HashSet<Vec<u8>>),
    Hash(HashMap<Vec<u8>, Vec<u8>>),
    SortedSet(BTreeMap<Vec<u8>, f64>), // Simplified: member -> score
}

struct Entry {
    value: RedisValue,
    expires_at: Option<Instant>,
}

// ===== Database =====

struct Database {
    data: HashMap<Vec<u8>, Entry>,
    expires: BTreeMap<Instant, Vec<u8>>,
}

impl Database {
    fn new() -> Self {
        Database {
            data: HashMap::new(),
            expires: BTreeMap::new(),
        }
    }

    fn get(&self, key: &[u8]) -> Option<&RedisValue> {
        let entry = self.data.get(key)?;
        if let Some(exp) = entry.expires_at {
            if Instant::now() > exp {
                return None;
            }
        }
        Some(&entry.value)
    }

    fn set(&mut self, key: Vec<u8>, value: RedisValue, ttl: Option<Duration>) {
        let expires_at = ttl.map(|d| Instant::now() + d);
        if let Some(exp) = expires_at {
            self.expires.insert(exp, key.clone());
        }
        self.data.insert(key, Entry { value, expires_at });
    }

    fn del(&mut self, key: &[u8]) -> bool {
        self.data.remove(key).is_some()
    }

    fn expire_keys(&mut self) {
        let now = Instant::now();
        let expired: Vec<_> = self.expires
            .range(..=now)
            .map(|(_, k)| k.clone())
            .collect();

        for key in &expired {
            self.data.remove(key);
        }

        self.expires = self.expires.split_off(&now);
    }
}

// ===== Commands =====

struct RedisServer {
    dbs: Vec<RwLock<Database>>,
    aof: Mutex<Option<File>>,
    subscribers: RwLock<HashMap<Vec<u8>, Vec<Arc<Mutex<TcpStream>>>>>,
}

impl RedisServer {
    fn new(num_dbs: usize) -> Self {
        RedisServer {
            dbs: (0..num_dbs).map(|_| RwLock::new(Database::new())).collect(),
            aof: Mutex::new(None),
            subscribers: RwLock::new(HashMap::new()),
        }
    }

    fn enable_aof(&self, path: &str) {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        *self.aof.lock().unwrap() = Some(file);
    }

    fn execute(&self, db_idx: usize, cmd: Vec<RespValue>) -> RespValue {
        let args: Vec<Vec<u8>> = cmd.into_iter().filter_map(|v| {
            if let RespValue::BulkString(Some(data)) = v {
                Some(data)
            } else {
                None
            }
        }).collect();

        if args.is_empty() {
            return RespValue::Error("ERR no command".to_string());
        }

        let command = String::from_utf8_lossy(&args[0]).to_uppercase();

        match command.as_str() {
            "PING" => RespValue::SimpleString("PONG".to_string()),

            "SET" => {
                if args.len() < 3 {
                    return RespValue::Error("ERR wrong number of arguments".to_string());
                }
                let mut ttl = None;
                let mut i = 3;
                while i < args.len() {
                    let opt = String::from_utf8_lossy(&args[i]).to_uppercase();
                    match opt.as_str() {
                        "EX" => {
                            if i + 1 < args.len() {
                                let secs: u64 = String::from_utf8_lossy(&args[i + 1]).parse().unwrap_or(0);
                                ttl = Some(Duration::from_secs(secs));
                                i += 2;
                            }
                        }
                        "PX" => {
                            if i + 1 < args.len() {
                                let ms: u64 = String::from_utf8_lossy(&args[i + 1]).parse().unwrap_or(0);
                                ttl = Some(Duration::from_millis(ms));
                                i += 2;
                            }
                        }
                        _ => i += 1,
                    }
                }

                let mut db = self.dbs[db_idx].write().unwrap();
                db.set(args[1].clone(), RedisValue::String(args[2].clone()), ttl);

                // Log to AOF
                self.log_aof(&args);

                RespValue::SimpleString("OK".to_string())
            }

            "GET" => {
                if args.len() < 2 {
                    return RespValue::Error("ERR wrong number of arguments".to_string());
                }
                let db = self.dbs[db_idx].read().unwrap();
                match db.get(&args[1]) {
                    Some(RedisValue::String(data)) => RespValue::BulkString(Some(data.clone())),
                    _ => RespValue::BulkString(None),
                }
            }

            "DEL" => {
                let mut db = self.dbs[db_idx].write().unwrap();
                let count: i64 = args[1..].iter()
                    .filter(|k| db.del(k))
                    .count() as i64;
                self.log_aof(&args);
                RespValue::Integer(count)
            }

            "EXPIRE" => {
                if args.len() < 3 {
                    return RespValue::Error("ERR wrong number of arguments".to_string());
                }
                let secs: u64 = String::from_utf8_lossy(&args[2]).parse().unwrap_or(0);
                let mut db = self.dbs[db_idx].write().unwrap();

                if let Some(entry) = db.data.get_mut(&args[1]) {
                    entry.expires_at = Some(Instant::now() + Duration::from_secs(secs));
                    RespValue::Integer(1)
                } else {
                    RespValue::Integer(0)
                }
            }

            "TTL" => {
                if args.len() < 2 {
                    return RespValue::Error("ERR wrong number of arguments".to_string());
                }
                let db = self.dbs[db_idx].read().unwrap();
                match db.data.get(&args[1]) {
                    Some(entry) => {
                        match entry.expires_at {
                            Some(exp) => {
                                let now = Instant::now();
                                if now > exp {
                                    RespValue::Integer(-2)
                                } else {
                                    RespValue::Integer((exp - now).as_secs() as i64)
                                }
                            }
                            None => RespValue::Integer(-1),
                        }
                    }
                    None => RespValue::Integer(-2),
                }
            }

            // List commands
            "LPUSH" => {
                if args.len() < 3 {
                    return RespValue::Error("ERR wrong number of arguments".to_string());
                }
                let mut db = self.dbs[db_idx].write().unwrap();
                let list = match db.data.get_mut(&args[1]) {
                    Some(Entry { value: RedisValue::List(list), .. }) => list,
                    None => {
                        db.set(args[1].clone(), RedisValue::List(VecDeque::new()), None);
                        if let Some(Entry { value: RedisValue::List(list), .. }) = db.data.get_mut(&args[1]) {
                            list
                        } else {
                            return RespValue::Error("ERR internal error".to_string());
                        }
                    }
                    _ => return RespValue::Error("WRONGTYPE".to_string()),
                };

                for item in args[2..].iter().rev() {
                    list.push_front(item.clone());
                }
                let len = list.len() as i64;
                self.log_aof(&args);
                RespValue::Integer(len)
            }

            "RPOP" => {
                if args.len() < 2 {
                    return RespValue::Error("ERR wrong number of arguments".to_string());
                }
                let mut db = self.dbs[db_idx].write().unwrap();
                match db.data.get_mut(&args[1]) {
                    Some(Entry { value: RedisValue::List(list), .. }) => {
                        match list.pop_back() {
                            Some(item) => {
                                self.log_aof(&args);
                                RespValue::BulkString(Some(item))
                            }
                            None => RespValue::BulkString(None),
                        }
                    }
                    _ => RespValue::BulkString(None),
                }
            }

            // Hash commands
            "HSET" => {
                if args.len() < 4 || (args.len() - 2) % 2 != 0 {
                    return RespValue::Error("ERR wrong number of arguments".to_string());
                }
                let mut db = self.dbs[db_idx].write().unwrap();
                let hash = match db.data.get_mut(&args[1]) {
                    Some(Entry { value: RedisValue::Hash(hash), .. }) => hash,
                    None => {
                        db.set(args[1].clone(), RedisValue::Hash(HashMap::new()), None);
                        if let Some(Entry { value: RedisValue::Hash(hash), .. }) = db.data.get_mut(&args[1]) {
                            hash
                        } else {
                            return RespValue::Error("ERR internal error".to_string());
                        }
                    }
                    _ => return RespValue::Error("WRONGTYPE".to_string()),
                };

                let mut added = 0i64;
                for i in (2..args.len()).step_by(2) {
                    if hash.insert(args[i].clone(), args[i + 1].clone()).is_none() {
                        added += 1;
                    }
                }
                self.log_aof(&args);
                RespValue::Integer(added)
            }

            "HGET" => {
                if args.len() < 3 {
                    return RespValue::Error("ERR wrong number of arguments".to_string());
                }
                let db = self.dbs[db_idx].read().unwrap();
                match db.get(&args[1]) {
                    Some(RedisValue::Hash(hash)) => {
                        match hash.get(&args[2]) {
                            Some(value) => RespValue::BulkString(Some(value.clone())),
                            None => RespValue::BulkString(None),
                        }
                    }
                    _ => RespValue::BulkString(None),
                }
            }

            // Pub/Sub
            "PUBLISH" => {
                if args.len() < 3 {
                    return RespValue::Error("ERR wrong number of arguments".to_string());
                }
                let subs = self.subscribers.read().unwrap();
                let mut count = 0i64;
                if let Some(clients) = subs.get(&args[1]) {
                    for client in clients {
                        let msg = RespValue::Array(Some(vec![
                            RespValue::BulkString(Some(b"message".to_vec())),
                            RespValue::BulkString(Some(args[1].clone())),
                            RespValue::BulkString(Some(args[2].clone())),
                        ]));
                        if let Ok(mut stream) = client.lock() {
                            let _ = stream.write_all(&msg.serialize());
                            count += 1;
                        }
                    }
                }
                RespValue::Integer(count)
            }

            "KEYS" => {
                if args.len() < 2 {
                    return RespValue::Error("ERR wrong number of arguments".to_string());
                }
                let pattern = String::from_utf8_lossy(&args[1]);
                let db = self.dbs[db_idx].read().unwrap();

                let keys: Vec<RespValue> = db.data.keys()
                    .filter(|k| {
                        let key = String::from_utf8_lossy(k);
                        if pattern == "*" {
                            true
                        } else {
                            // Simple pattern matching
                            key.contains(&pattern[..pattern.len().saturating_sub(1)])
                        }
                    })
                    .map(|k| RespValue::BulkString(Some(k.clone())))
                    .collect();

                RespValue::Array(Some(keys))
            }

            "INFO" => {
                let info = format!(
                    "# Server\\r\\nredis_version:0.1.0\\r\\n# Keyspace\\r\\ndb0:keys={}\\r\\n",
                    self.dbs[db_idx].read().unwrap().data.len()
                );
                RespValue::BulkString(Some(info.into_bytes()))
            }

            _ => RespValue::Error(format!("ERR unknown command '{}'", command)),
        }
    }

    fn log_aof(&self, args: &[Vec<u8>]) {
        if let Some(ref mut file) = *self.aof.lock().unwrap() {
            let resp = RespValue::Array(Some(
                args.iter().map(|a| RespValue::BulkString(Some(a.clone()))).collect()
            ));
            let _ = file.write_all(&resp.serialize());
        }
    }
}

fn handle_client(stream: TcpStream, server: Arc<RedisServer>) {
    let mut reader = BufReader::new(&stream);
    let mut writer = stream.try_clone().unwrap();
    let db_idx = 0;

    loop {
        match parse_resp(&mut reader) {
            Some(RespValue::Array(Some(cmd))) => {
                let response = server.execute(db_idx, cmd);
                if writer.write_all(&response.serialize()).is_err() {
                    break;
                }
            }
            _ => break,
        }
    }
}

fn main() {
    let server = Arc::new(RedisServer::new(16));

    // Enable AOF persistence
    server.enable_aof("appendonly.aof");

    let listener = TcpListener::bind("127.0.0.1:6379").unwrap();
    println!("Redis server listening on port 6379");

    // Background key expiration
    let server_clone = Arc::clone(&server);
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_millis(100));
            for db in &server_clone.dbs {
                db.write().unwrap().expire_keys();
            }
        }
    });

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let server = Arc::clone(&server);
                thread::spawn(move || handle_client(stream, server));
            }
            Err(e) => eprintln!("Connection failed: {}", e),
        }
    }
}
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Build Your Own SQLite',
    description: 'Implement a SQL database with B-tree storage, query parsing, and ACID transactions',
    icon: 'database',
    color: 'blue',
    language: 'C, Rust',
    skills: 'B-trees, SQL parsing, Query optimization, File I/O, ACID',
    difficulty: 'advanced',
    estimated_weeks: 10,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Architecture | Component design |
| 1 | 2 | Page manager | Fixed-size pages |
| 1 | 3 | File I/O | Page read/write |
| 1 | 4 | Free list | Page allocation |
| 1 | 5 | Buffer pool | Page caching |
| 2 | 1 | B-tree basics | Node structure |
| 2 | 2 | B-tree search | Key lookup |
| 2 | 3 | B-tree insert | Leaf insert |
| 2 | 4 | Node splitting | Internal nodes |
| 2 | 5 | B-tree delete | Key removal |
| 3 | 1 | Row format | Serialization |
| 3 | 2 | Schema storage | sqlite_master |
| 3 | 3 | Data types | INT/TEXT/BLOB |
| 3 | 4 | NULL handling | NULL values |
| 3 | 5 | Table storage | Full tables |
| 4 | 1 | SQL tokenizer | Lexer |
| 4 | 2 | SQL parser | Parser basics |
| 4 | 3 | CREATE TABLE | DDL parsing |
| 4 | 4 | SELECT parsing | Query parsing |
| 4 | 5 | INSERT/UPDATE | DML parsing |
| 5 | 1 | Query planner | Plan tree |
| 5 | 2 | Table scan | Full scan |
| 5 | 3 | Index scan | B-tree scan |
| 5 | 4 | Filter/Project | WHERE clause |
| 5 | 5 | Query executor | Volcano model |
| 6 | 1 | Expression eval | Operators |
| 6 | 2 | Comparisons | =, <, >, etc |
| 6 | 3 | Functions | Built-in funcs |
| 6 | 4 | ORDER BY | Sorting |
| 6 | 5 | LIMIT/OFFSET | Pagination |
| 7 | 1 | Indexes | CREATE INDEX |
| 7 | 2 | Unique indexes | Constraints |
| 7 | 3 | Index selection | Query opt |
| 7 | 4 | Composite keys | Multi-column |
| 7 | 5 | Index-only scan | Covering idx |
| 8 | 1 | WAL basics | Write-ahead log |
| 8 | 2 | Transactions | BEGIN/COMMIT |
| 8 | 3 | ROLLBACK | Undo changes |
| 8 | 4 | Crash recovery | Log replay |
| 8 | 5 | Checkpointing | WAL cleanup |
| 9 | 1 | MVCC basics | Versioning |
| 9 | 2 | Read isolation | Snapshots |
| 9 | 3 | Write conflicts | Detection |
| 9 | 4 | Deadlock detect | Cycle detection |
| 9 | 5 | Lock manager | Row locking |
| 10 | 1 | Aggregation | COUNT/SUM/AVG |
| 10 | 2 | GROUP BY | Grouping |
| 10 | 3 | JOIN basics | Nested loop |
| 10 | 4 | Hash join | Join opt |
| 10 | 5 | Integration | Full database |`,
    modules: [
      {
        name: 'Storage Engine',
        description: 'B-tree based storage',
        tasks: [
          {
            title: 'SQLite Storage in C',
            description: 'Implement B-tree storage engine',
            details: `# SQLite-like Storage Engine in C

\`\`\`c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <fcntl.h>
#include <unistd.h>

// Page configuration
#define PAGE_SIZE 4096
#define MAX_PAGES 1000

// B-tree configuration
#define BTREE_ORDER 50  // Max keys per node

// Node types
typedef enum {
    NODE_INTERNAL,
    NODE_LEAF
} NodeType;

// B-tree node header
typedef struct {
    NodeType type;
    uint32_t num_keys;
    uint32_t parent;
    uint32_t next_leaf;  // For leaf nodes only
} NodeHeader;

// Cell in leaf node: key + value
typedef struct {
    uint32_t key;
    uint32_t value_size;
    char value[];  // Flexible array
} LeafCell;

// Cell in internal node: key + child pointer
typedef struct {
    uint32_t key;
    uint32_t child;
} InternalCell;

// Page structure
typedef struct {
    NodeHeader header;
    char data[PAGE_SIZE - sizeof(NodeHeader)];
} Page;

// Pager manages file I/O
typedef struct {
    int fd;
    uint32_t num_pages;
    Page *pages[MAX_PAGES];
} Pager;

// B-tree cursor for traversal
typedef struct {
    uint32_t page_num;
    uint32_t cell_num;
    bool end_of_table;
} Cursor;

// Table structure
typedef struct {
    Pager *pager;
    uint32_t root_page;
} Table;

// ===== Pager =====

Pager *pager_open(const char *filename) {
    int fd = open(filename, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        perror("open");
        return NULL;
    }

    off_t file_size = lseek(fd, 0, SEEK_END);

    Pager *pager = calloc(1, sizeof(Pager));
    pager->fd = fd;
    pager->num_pages = file_size / PAGE_SIZE;

    return pager;
}

Page *pager_get_page(Pager *pager, uint32_t page_num) {
    if (page_num >= MAX_PAGES) {
        fprintf(stderr, "Page number out of bounds\\n");
        return NULL;
    }

    if (pager->pages[page_num] == NULL) {
        Page *page = aligned_alloc(PAGE_SIZE, PAGE_SIZE);
        memset(page, 0, PAGE_SIZE);

        if (page_num < pager->num_pages) {
            lseek(pager->fd, page_num * PAGE_SIZE, SEEK_SET);
            read(pager->fd, page, PAGE_SIZE);
        }

        pager->pages[page_num] = page;

        if (page_num >= pager->num_pages) {
            pager->num_pages = page_num + 1;
        }
    }

    return pager->pages[page_num];
}

void pager_flush(Pager *pager, uint32_t page_num) {
    if (pager->pages[page_num] == NULL) return;

    lseek(pager->fd, page_num * PAGE_SIZE, SEEK_SET);
    write(pager->fd, pager->pages[page_num], PAGE_SIZE);
}

void pager_close(Pager *pager) {
    for (uint32_t i = 0; i < pager->num_pages; i++) {
        if (pager->pages[i]) {
            pager_flush(pager, i);
            free(pager->pages[i]);
        }
    }
    close(pager->fd);
    free(pager);
}

// ===== B-tree Operations =====

void initialize_leaf_node(Page *page) {
    page->header.type = NODE_LEAF;
    page->header.num_keys = 0;
    page->header.next_leaf = 0;
}

void initialize_internal_node(Page *page) {
    page->header.type = NODE_INTERNAL;
    page->header.num_keys = 0;
}

uint32_t *leaf_node_num_cells(Page *page) {
    return &page->header.num_keys;
}

LeafCell *leaf_node_cell(Page *page, uint32_t cell_num) {
    // Simplified: fixed-size cells
    return (LeafCell *)(page->data + cell_num * 256);
}

uint32_t leaf_node_key(Page *page, uint32_t cell_num) {
    return leaf_node_cell(page, cell_num)->key;
}

char *leaf_node_value(Page *page, uint32_t cell_num) {
    return leaf_node_cell(page, cell_num)->value;
}

InternalCell *internal_node_cell(Page *page, uint32_t cell_num) {
    return (InternalCell *)(page->data + cell_num * sizeof(InternalCell));
}

uint32_t internal_node_child(Page *page, uint32_t child_num) {
    if (child_num > page->header.num_keys) {
        fprintf(stderr, "Child index out of bounds\\n");
        return 0;
    }
    if (child_num == page->header.num_keys) {
        // Rightmost child stored at end
        return *(uint32_t *)(page->data + PAGE_SIZE - sizeof(NodeHeader) - sizeof(uint32_t));
    }
    return internal_node_cell(page, child_num)->child;
}

void set_internal_node_child(Page *page, uint32_t child_num, uint32_t child) {
    if (child_num == page->header.num_keys) {
        *(uint32_t *)(page->data + PAGE_SIZE - sizeof(NodeHeader) - sizeof(uint32_t)) = child;
    } else {
        internal_node_cell(page, child_num)->child = child;
    }
}

// Binary search in leaf node
Cursor *leaf_node_find(Table *table, uint32_t page_num, uint32_t key) {
    Page *page = pager_get_page(table->pager, page_num);
    uint32_t num_cells = *leaf_node_num_cells(page);

    Cursor *cursor = malloc(sizeof(Cursor));
    cursor->page_num = page_num;
    cursor->end_of_table = false;

    // Binary search
    uint32_t lo = 0, hi = num_cells;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        uint32_t mid_key = leaf_node_key(page, mid);
        if (mid_key >= key) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    cursor->cell_num = lo;
    return cursor;
}

// Find leaf for key in tree
Cursor *table_find(Table *table, uint32_t key) {
    Page *root = pager_get_page(table->pager, table->root_page);

    if (root->header.type == NODE_LEAF) {
        return leaf_node_find(table, table->root_page, key);
    }

    // Traverse internal nodes
    uint32_t page_num = table->root_page;
    Page *page = root;

    while (page->header.type == NODE_INTERNAL) {
        uint32_t child_num = 0;
        uint32_t num_keys = page->header.num_keys;

        // Binary search for child
        uint32_t lo = 0, hi = num_keys;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            uint32_t mid_key = internal_node_cell(page, mid)->key;
            if (mid_key >= key) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        child_num = lo;

        page_num = internal_node_child(page, child_num);
        page = pager_get_page(table->pager, page_num);
    }

    return leaf_node_find(table, page_num, key);
}

// Split leaf node
void leaf_node_split_and_insert(Table *table, Cursor *cursor, uint32_t key, const char *value) {
    Page *old_page = pager_get_page(table->pager, cursor->page_num);
    uint32_t new_page_num = table->pager->num_pages;
    Page *new_page = pager_get_page(table->pager, new_page_num);
    initialize_leaf_node(new_page);

    // Set up linked list
    new_page->header.next_leaf = old_page->header.next_leaf;
    old_page->header.next_leaf = new_page_num;

    // Split cells between old and new page
    uint32_t split_point = (BTREE_ORDER + 1) / 2;

    for (uint32_t i = BTREE_ORDER; i >= 0; i--) {
        Page *dest;
        uint32_t index;

        if (i >= split_point) {
            dest = new_page;
            index = i - split_point;
        } else {
            dest = old_page;
            index = i;
        }

        if (i == cursor->cell_num) {
            // Insert new cell
            LeafCell *cell = leaf_node_cell(dest, index);
            cell->key = key;
            strcpy(cell->value, value);
        } else if (i > cursor->cell_num) {
            // Copy from old position
            memcpy(leaf_node_cell(dest, index),
                   leaf_node_cell(old_page, i - 1), 256);
        } else {
            memcpy(leaf_node_cell(dest, index),
                   leaf_node_cell(old_page, i), 256);
        }
    }

    *leaf_node_num_cells(old_page) = split_point;
    *leaf_node_num_cells(new_page) = BTREE_ORDER + 1 - split_point;

    // Update parent (simplified - would need to handle parent splits)
    // ...
}

// Insert key-value pair
void table_insert(Table *table, uint32_t key, const char *value) {
    Cursor *cursor = table_find(table, key);
    Page *page = pager_get_page(table->pager, cursor->page_num);
    uint32_t num_cells = *leaf_node_num_cells(page);

    if (cursor->cell_num < num_cells) {
        uint32_t existing_key = leaf_node_key(page, cursor->cell_num);
        if (existing_key == key) {
            fprintf(stderr, "Duplicate key\\n");
            free(cursor);
            return;
        }
    }

    if (num_cells >= BTREE_ORDER) {
        // Split
        leaf_node_split_and_insert(table, cursor, key, value);
    } else {
        // Insert in place
        if (cursor->cell_num < num_cells) {
            // Shift cells
            for (uint32_t i = num_cells; i > cursor->cell_num; i--) {
                memcpy(leaf_node_cell(page, i),
                       leaf_node_cell(page, i - 1), 256);
            }
        }

        LeafCell *cell = leaf_node_cell(page, cursor->cell_num);
        cell->key = key;
        strcpy(cell->value, value);
        (*leaf_node_num_cells(page))++;
    }

    free(cursor);
}

// ===== Table Operations =====

Table *table_open(const char *filename) {
    Pager *pager = pager_open(filename);
    Table *table = malloc(sizeof(Table));
    table->pager = pager;
    table->root_page = 0;

    if (pager->num_pages == 0) {
        // New database
        Page *root = pager_get_page(pager, 0);
        initialize_leaf_node(root);
    }

    return table;
}

void table_close(Table *table) {
    pager_close(table->pager);
    free(table);
}

// ===== SQL Parser (Simplified) =====

typedef enum {
    STMT_INSERT,
    STMT_SELECT
} StatementType;

typedef struct {
    StatementType type;
    uint32_t key;
    char value[256];
} Statement;

bool parse_statement(const char *input, Statement *stmt) {
    if (strncmp(input, "insert ", 7) == 0) {
        stmt->type = STMT_INSERT;
        int n = sscanf(input + 7, "%u %255s", &stmt->key, stmt->value);
        return n == 2;
    }
    if (strcmp(input, "select") == 0) {
        stmt->type = STMT_SELECT;
        return true;
    }
    return false;
}

void execute_statement(Table *table, Statement *stmt) {
    switch (stmt->type) {
        case STMT_INSERT:
            table_insert(table, stmt->key, stmt->value);
            printf("Inserted\\n");
            break;

        case STMT_SELECT: {
            Cursor cursor = {.page_num = table->root_page, .cell_num = 0, .end_of_table = false};

            while (!cursor.end_of_table) {
                Page *page = pager_get_page(table->pager, cursor.page_num);
                uint32_t num_cells = *leaf_node_num_cells(page);

                if (cursor.cell_num >= num_cells) {
                    if (page->header.next_leaf == 0) {
                        cursor.end_of_table = true;
                    } else {
                        cursor.page_num = page->header.next_leaf;
                        cursor.cell_num = 0;
                    }
                    continue;
                }

                printf("%u: %s\\n",
                       leaf_node_key(page, cursor.cell_num),
                       leaf_node_value(page, cursor.cell_num));
                cursor.cell_num++;
            }
            break;
        }
    }
}

int main(int argc, char **argv) {
    const char *filename = argc > 1 ? argv[1] : "test.db";
    Table *table = table_open(filename);

    char input[256];
    while (1) {
        printf("db > ");
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\\n")] = 0;

        if (strcmp(input, ".exit") == 0) {
            table_close(table);
            return 0;
        }

        Statement stmt;
        if (parse_statement(input, &stmt)) {
            execute_statement(table, &stmt);
        } else {
            printf("Unrecognized command: %s\\n", input);
        }
    }

    return 0;
}
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Build Your Own LSM Tree',
    description: 'Implement a Log-Structured Merge Tree database like LevelDB/RocksDB',
    icon: 'layers',
    color: 'green',
    language: 'Rust, Go, C++',
    skills: 'LSM trees, SSTables, Compaction, Bloom filters',
    difficulty: 'advanced',
    estimated_weeks: 6,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Architecture | LSM overview |
| 1 | 2 | Memtable | Skip list |
| 1 | 3 | Write path | Put/Delete |
| 1 | 4 | WAL basics | Write-ahead log |
| 1 | 5 | Memtable flush | Trigger flush |
| 2 | 1 | SSTable format | File format |
| 2 | 2 | Block encoding | Data blocks |
| 2 | 3 | Index block | Block index |
| 2 | 4 | SSTable write | Flush to disk |
| 2 | 5 | SSTable read | Block cache |
| 3 | 1 | Bloom filter | Filter construction |
| 3 | 2 | Filter block | SSTable filter |
| 3 | 3 | Read path | Point lookup |
| 3 | 4 | Merge iterator | Multi-file scan |
| 3 | 5 | Range scan | Iterator API |
| 4 | 1 | Level structure | L0 to Lmax |
| 4 | 2 | Compaction trigger | When to compact |
| 4 | 3 | Compaction picker | Which files |
| 4 | 4 | Merge compaction | File merging |
| 4 | 5 | Manifest | Version tracking |
| 5 | 1 | Compression | Block compression |
| 5 | 2 | Block cache | LRU cache |
| 5 | 3 | Write buffer | Batched writes |
| 5 | 4 | Concurrent access | Read/write locks |
| 5 | 5 | Recovery | WAL replay |
| 6 | 1 | Tombstones | Delete handling |
| 6 | 2 | TTL | Key expiration |
| 6 | 3 | Snapshots | Read snapshots |
| 6 | 4 | Benchmarking | Performance |
| 6 | 5 | Integration | Full LSM tree |`,
    modules: [
      {
        name: 'LSM Implementation',
        description: 'Complete LSM tree storage engine',
        tasks: [
          {
            title: 'LSM Tree in Rust',
            description: 'Build LevelDB-style storage engine',
            details: `# LSM Tree Implementation in Rust

\`\`\`rust
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock, Mutex};

// ===== Memtable (Skip List simplified as BTreeMap) =====

#[derive(Clone)]
enum Value {
    Put(Vec<u8>),
    Delete,
}

struct MemTable {
    data: BTreeMap<Vec<u8>, Value>,
    size: usize,
}

impl MemTable {
    fn new() -> Self {
        MemTable {
            data: BTreeMap::new(),
            size: 0,
        }
    }

    fn put(&mut self, key: Vec<u8>, value: Vec<u8>) {
        self.size += key.len() + value.len();
        self.data.insert(key, Value::Put(value));
    }

    fn delete(&mut self, key: Vec<u8>) {
        self.size += key.len();
        self.data.insert(key, Value::Delete);
    }

    fn get(&self, key: &[u8]) -> Option<&Value> {
        self.data.get(key)
    }

    fn iter(&self) -> impl Iterator<Item = (&Vec<u8>, &Value)> {
        self.data.iter()
    }
}

// ===== SSTable =====

const BLOCK_SIZE: usize = 4096;

struct BlockHandle {
    offset: u64,
    size: u64,
}

struct SSTable {
    path: PathBuf,
    index: Vec<(Vec<u8>, BlockHandle)>,
    bloom_filter: BloomFilter,
    min_key: Vec<u8>,
    max_key: Vec<u8>,
}

impl SSTable {
    fn create(path: PathBuf, memtable: &MemTable) -> io::Result<Self> {
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        let mut index = Vec::new();
        let mut current_block = Vec::new();
        let mut block_first_key = Vec::new();
        let mut offset = 0u64;

        let mut bloom = BloomFilter::new(memtable.data.len());
        let mut min_key = Vec::new();
        let mut max_key = Vec::new();

        for (key, value) in memtable.iter() {
            bloom.insert(key);

            if min_key.is_empty() {
                min_key = key.clone();
            }
            max_key = key.clone();

            // Encode key-value
            let encoded = Self::encode_entry(key, value);

            if current_block.len() + encoded.len() > BLOCK_SIZE && !current_block.is_empty() {
                // Flush block
                writer.write_all(&current_block)?;
                index.push((block_first_key.clone(), BlockHandle {
                    offset,
                    size: current_block.len() as u64,
                }));
                offset += current_block.len() as u64;
                current_block.clear();
            }

            if current_block.is_empty() {
                block_first_key = key.clone();
            }
            current_block.extend(encoded);
        }

        // Flush last block
        if !current_block.is_empty() {
            writer.write_all(&current_block)?;
            index.push((block_first_key.clone(), BlockHandle {
                offset,
                size: current_block.len() as u64,
            }));
        }

        // Write index block
        let index_offset = writer.stream_position()?;
        for (key, handle) in &index {
            writer.write_all(&(key.len() as u32).to_le_bytes())?;
            writer.write_all(key)?;
            writer.write_all(&handle.offset.to_le_bytes())?;
            writer.write_all(&handle.size.to_le_bytes())?;
        }

        // Write bloom filter
        let bloom_offset = writer.stream_position()?;
        writer.write_all(&bloom.bits)?;

        // Write footer
        writer.write_all(&index_offset.to_le_bytes())?;
        writer.write_all(&bloom_offset.to_le_bytes())?;

        writer.flush()?;

        Ok(SSTable {
            path,
            index,
            bloom_filter: bloom,
            min_key,
            max_key,
        })
    }

    fn encode_entry(key: &[u8], value: &Value) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend(&(key.len() as u32).to_le_bytes());
        buf.extend(key);

        match value {
            Value::Put(v) => {
                buf.push(1); // Put marker
                buf.extend(&(v.len() as u32).to_le_bytes());
                buf.extend(v);
            }
            Value::Delete => {
                buf.push(0); // Delete marker
            }
        }
        buf
    }

    fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>> {
        // Check bloom filter first
        if !self.bloom_filter.may_contain(key) {
            return Ok(None);
        }

        // Binary search index
        let block_idx = self.index.partition_point(|(k, _)| k.as_slice() <= key);
        if block_idx == 0 {
            return Ok(None);
        }

        let (_, handle) = &self.index[block_idx - 1];

        // Read block
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(handle.offset))?;
        let mut block = vec![0u8; handle.size as usize];
        file.read_exact(&mut block)?;

        // Search in block
        let mut pos = 0;
        while pos < block.len() {
            let key_len = u32::from_le_bytes(block[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            let entry_key = &block[pos..pos+key_len];
            pos += key_len;

            let is_put = block[pos] == 1;
            pos += 1;

            if entry_key == key {
                if is_put {
                    let val_len = u32::from_le_bytes(block[pos..pos+4].try_into().unwrap()) as usize;
                    pos += 4;
                    return Ok(Some(block[pos..pos+val_len].to_vec()));
                } else {
                    return Ok(None); // Deleted
                }
            }

            if is_put {
                let val_len = u32::from_le_bytes(block[pos..pos+4].try_into().unwrap()) as usize;
                pos += 4 + val_len;
            }
        }

        Ok(None)
    }
}

// ===== Bloom Filter =====

struct BloomFilter {
    bits: Vec<u8>,
    num_hashes: usize,
}

impl BloomFilter {
    fn new(n: usize) -> Self {
        let bits_per_key = 10;
        let size = (n * bits_per_key + 7) / 8;
        BloomFilter {
            bits: vec![0; size.max(1)],
            num_hashes: 7,
        }
    }

    fn hash(&self, key: &[u8], seed: usize) -> usize {
        let mut h = seed as u64;
        for &b in key {
            h = h.wrapping_mul(31).wrapping_add(b as u64);
        }
        h as usize % (self.bits.len() * 8)
    }

    fn insert(&mut self, key: &[u8]) {
        for i in 0..self.num_hashes {
            let bit = self.hash(key, i);
            self.bits[bit / 8] |= 1 << (bit % 8);
        }
    }

    fn may_contain(&self, key: &[u8]) -> bool {
        for i in 0..self.num_hashes {
            let bit = self.hash(key, i);
            if self.bits[bit / 8] & (1 << (bit % 8)) == 0 {
                return false;
            }
        }
        true
    }
}

// ===== LSM Tree =====

struct Level {
    sstables: Vec<SSTable>,
    max_size: usize,
}

struct LSMTree {
    dir: PathBuf,
    memtable: RwLock<MemTable>,
    immutable: RwLock<Option<MemTable>>,
    levels: RwLock<Vec<Level>>,
    wal: Mutex<File>,
    next_file_id: Mutex<u64>,
    memtable_size_limit: usize,
}

impl LSMTree {
    fn open(dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;

        let wal_path = dir.join("wal");
        let wal = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)?;

        let tree = LSMTree {
            dir,
            memtable: RwLock::new(MemTable::new()),
            immutable: RwLock::new(None),
            levels: RwLock::new(vec![
                Level { sstables: Vec::new(), max_size: 4 },  // L0
                Level { sstables: Vec::new(), max_size: 10 }, // L1
                Level { sstables: Vec::new(), max_size: 100 }, // L2
            ]),
            wal,
            next_file_id: Mutex::new(0),
            memtable_size_limit: 4 * 1024 * 1024, // 4MB
        };

        // TODO: Recover from WAL

        Ok(tree)
    }

    fn put(&self, key: Vec<u8>, value: Vec<u8>) -> io::Result<()> {
        // Write to WAL
        {
            let mut wal = self.wal.lock().unwrap();
            wal.write_all(&[1])?; // Put marker
            wal.write_all(&(key.len() as u32).to_le_bytes())?;
            wal.write_all(&key)?;
            wal.write_all(&(value.len() as u32).to_le_bytes())?;
            wal.write_all(&value)?;
            wal.flush()?;
        }

        // Write to memtable
        let should_flush = {
            let mut memtable = self.memtable.write().unwrap();
            memtable.put(key, value);
            memtable.size >= self.memtable_size_limit
        };

        if should_flush {
            self.flush_memtable()?;
        }

        Ok(())
    }

    fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>> {
        // Check memtable
        {
            let memtable = self.memtable.read().unwrap();
            if let Some(value) = memtable.get(key) {
                return Ok(match value {
                    Value::Put(v) => Some(v.clone()),
                    Value::Delete => None,
                });
            }
        }

        // Check immutable memtable
        {
            let imm = self.immutable.read().unwrap();
            if let Some(ref memtable) = *imm {
                if let Some(value) = memtable.get(key) {
                    return Ok(match value {
                        Value::Put(v) => Some(v.clone()),
                        Value::Delete => None,
                    });
                }
            }
        }

        // Check SSTables (newest first)
        let levels = self.levels.read().unwrap();
        for level in levels.iter() {
            for sst in level.sstables.iter().rev() {
                if let Some(value) = sst.get(key)? {
                    return Ok(Some(value));
                }
            }
        }

        Ok(None)
    }

    fn delete(&self, key: Vec<u8>) -> io::Result<()> {
        // Write tombstone to WAL
        {
            let mut wal = self.wal.lock().unwrap();
            wal.write_all(&[0])?; // Delete marker
            wal.write_all(&(key.len() as u32).to_le_bytes())?;
            wal.write_all(&key)?;
            wal.flush()?;
        }

        // Write to memtable
        let mut memtable = self.memtable.write().unwrap();
        memtable.delete(key);

        Ok(())
    }

    fn flush_memtable(&self) -> io::Result<()> {
        // Move memtable to immutable
        let memtable = {
            let mut mem = self.memtable.write().unwrap();
            let old = std::mem::replace(&mut *mem, MemTable::new());
            old
        };

        *self.immutable.write().unwrap() = Some(memtable);

        // Flush to SSTable
        let file_id = {
            let mut id = self.next_file_id.lock().unwrap();
            let current = *id;
            *id += 1;
            current
        };

        let path = self.dir.join(format!("{:08}.sst", file_id));

        let imm = self.immutable.read().unwrap();
        if let Some(ref memtable) = *imm {
            let sst = SSTable::create(path, memtable)?;

            let mut levels = self.levels.write().unwrap();
            levels[0].sstables.push(sst);

            // Check if compaction needed
            if levels[0].sstables.len() > levels[0].max_size {
                drop(levels);
                self.compact(0)?;
            }
        }

        *self.immutable.write().unwrap() = None;

        Ok(())
    }

    fn compact(&self, level: usize) -> io::Result<()> {
        // Simplified: merge all SSTables in level to next level
        // Real implementation would be more sophisticated
        println!("Compacting level {}", level);
        Ok(())
    }
}

fn main() -> io::Result<()> {
    let tree = LSMTree::open("testdb".into())?;

    // Write some data
    for i in 0..1000 {
        let key = format!("key{:05}", i).into_bytes();
        let value = format!("value{}", i).into_bytes();
        tree.put(key, value)?;
    }

    // Read back
    let key = b"key00042";
    match tree.get(key)? {
        Some(value) => println!("Found: {}", String::from_utf8_lossy(&value)),
        None => println!("Not found"),
    }

    Ok(())
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

console.log('Seeded: Redis, SQLite, LSM Tree');
