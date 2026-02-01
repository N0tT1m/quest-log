import Database from 'better-sqlite3';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import * as schema from '../src/lib/server/schema';

const sqlite = new Database('data/quest-log.db');

// Add new columns if they don't exist
try { sqlite.exec(`ALTER TABLE paths ADD COLUMN language TEXT`); } catch {}
try { sqlite.exec(`ALTER TABLE paths ADD COLUMN skills TEXT`); } catch {}
try { sqlite.exec(`ALTER TABLE paths ADD COLUMN start_hint TEXT`); } catch {}
try { sqlite.exec(`ALTER TABLE paths ADD COLUMN difficulty TEXT DEFAULT 'intermediate'`); } catch {}
try { sqlite.exec(`ALTER TABLE paths ADD COLUMN estimated_weeks INTEGER`); } catch {}
try { sqlite.exec(`ALTER TABLE tasks ADD COLUMN details TEXT`); } catch {}
try { sqlite.exec(`ALTER TABLE paths ADD COLUMN schedule TEXT`); } catch {}

const db = drizzle(sqlite, { schema });

const projects = [
	{
		name: 'Terminal-based Task Scheduler',
		description: 'Build a cron-like scheduler with a TUI using textual or rich.',
		language: 'Python',
		color: 'blue',
		skills: 'async/await, file I/O, state management, TUI design',
		startHint: 'A single job that runs a shell command at an interval',
		difficulty: 'intermediate',
		estimatedWeeks: 2,
		schedule: `## 2-Week Schedule

### Week 1: Core Scheduler
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Setup & Cron Parsing | Set up project, implement cron expression parser |
| Day 2 | Job Storage | Implement SQLite/JSON persistence for jobs |
| Day 3 | Scheduler Loop | Build the main scheduling loop with async |
| Day 4 | Retry Logic | Add exponential backoff for failed jobs |
| Day 5 | Notifications | Implement desktop/webhook notifications |
| Weekend | Buffer | Catch up, refactor, or explore edge cases |

### Week 2: TUI Interface
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | TUI Setup | Initialize textual/rich framework, basic layout |
| Day 2 | Job List View | Display jobs with status indicators |
| Day 3 | CRUD Operations | Add/edit/delete jobs through TUI |
| Day 4 | Log Viewer | Show real-time and historical job output |
| Day 5 | Polish | Error handling, keyboard shortcuts, help screen |
| Weekend | Testing | End-to-end testing, documentation |

### Daily Commitment
- **Minimum**: 1-2 hours focused coding
- **Ideal**: 3-4 hours with breaks
- **Rule**: No AI assistance - use docs and debugger only`,
		modules: [
			{
				name: 'Core Scheduler',
				description: 'Build the foundational scheduling logic',
				tasks: [
					{
						title: 'Define jobs with schedules (cron syntax or intervals)',
						description: 'Parse cron expressions and simple interval syntax',
						details: `## Cron Expression Parsing

### Cron Format
\`\`\`
┌───────────── minute (0-59)
│ ┌───────────── hour (0-23)
│ │ ┌───────────── day of month (1-31)
│ │ │ ┌───────────── month (1-12)
│ │ │ │ ┌───────────── day of week (0-6, Sun=0)
│ │ │ │ │
* * * * *
\`\`\`

### Implementation Steps
1. **Tokenize** the cron string into 5 fields
2. **Parse each field** handling:
   - \`*\` (any value)
   - \`*/n\` (every n units)
   - \`n-m\` (range)
   - \`n,m,o\` (list)
3. **Validate ranges** for each field type
4. **Calculate next run time** from current time

### Example Code Structure
\`\`\`python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CronSchedule:
    minute: set[int]
    hour: set[int]
    day: set[int]
    month: set[int]
    weekday: set[int]

    @classmethod
    def parse(cls, expr: str) -> "CronSchedule":
        fields = expr.split()
        # Parse each field...

    def next_run(self, after: datetime) -> datetime:
        # Find next matching datetime...
\`\`\`

### Simple Interval Alternative
\`\`\`python
@dataclass
class IntervalSchedule:
    seconds: int

    @classmethod
    def parse(cls, expr: str) -> "IntervalSchedule":
        # Parse "every 5m", "every 1h", "every 30s"
        ...
\`\`\`

### Testing Checklist
- [ ] Parse \`* * * * *\` (every minute)
- [ ] Parse \`0 */2 * * *\` (every 2 hours)
- [ ] Parse \`0 9-17 * * 1-5\` (weekdays 9am-5pm)
- [ ] Handle invalid expressions gracefully`
					},
					{
						title: 'Persist jobs to SQLite or JSON',
						description: 'Implement job storage with CRUD operations',
						details: `## Job Persistence Layer

### SQLite Schema
\`\`\`sql
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    command TEXT NOT NULL,
    schedule TEXT NOT NULL,
    enabled BOOLEAN DEFAULT 1,
    last_run TIMESTAMP,
    next_run TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE job_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER REFERENCES jobs(id),
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    exit_code INTEGER,
    stdout TEXT,
    stderr TEXT
);
\`\`\`

### Repository Pattern
\`\`\`python
from abc import ABC, abstractmethod

class JobRepository(ABC):
    @abstractmethod
    def create(self, job: Job) -> Job: ...

    @abstractmethod
    def get(self, job_id: int) -> Job | None: ...

    @abstractmethod
    def list(self, enabled_only: bool = False) -> list[Job]: ...

    @abstractmethod
    def update(self, job: Job) -> Job: ...

    @abstractmethod
    def delete(self, job_id: int) -> bool: ...

class SQLiteJobRepository(JobRepository):
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
\`\`\`

### JSON Alternative (simpler)
\`\`\`python
import json
from pathlib import Path

class JSONJobRepository(JobRepository):
    def __init__(self, path: Path):
        self.path = path
        self._load()

    def _load(self):
        if self.path.exists():
            self.jobs = json.loads(self.path.read_text())
        else:
            self.jobs = {}

    def _save(self):
        self.path.write_text(json.dumps(self.jobs, indent=2))
\`\`\`

### Considerations
- Use transactions for SQLite updates
- Add file locking for JSON to prevent corruption
- Consider WAL mode for SQLite concurrent access`
					},
					{
						title: 'Retry failed jobs with backoff',
						description: 'Add exponential backoff for failed job retries',
						details: `## Exponential Backoff Implementation

### Backoff Strategy
\`\`\`
Attempt 1: immediate
Attempt 2: wait 1s
Attempt 3: wait 2s
Attempt 4: wait 4s
Attempt 5: wait 8s
...with jitter to prevent thundering herd
\`\`\`

### Implementation
\`\`\`python
import random
import asyncio
from dataclasses import dataclass

@dataclass
class RetryConfig:
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.1  # ±10%

def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = min(
        config.base_delay * (2 ** attempt),
        config.max_delay
    )
    # Add jitter
    jitter_range = delay * config.jitter
    delay += random.uniform(-jitter_range, jitter_range)
    return max(0, delay)

async def run_with_retry(
    job: Job,
    config: RetryConfig
) -> JobResult:
    last_error = None

    for attempt in range(config.max_attempts):
        try:
            return await execute_job(job)
        except Exception as e:
            last_error = e
            if attempt < config.max_attempts - 1:
                delay = calculate_delay(attempt, config)
                await asyncio.sleep(delay)

    return JobResult(
        success=False,
        error=str(last_error),
        attempts=config.max_attempts
    )
\`\`\`

### Job Model Updates
\`\`\`python
@dataclass
class Job:
    # ... existing fields
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    current_attempt: int = 0
    last_error: str | None = None
\`\`\`

### Testing Scenarios
- [ ] Job succeeds on first try
- [ ] Job fails then succeeds
- [ ] Job exhausts all retries
- [ ] Verify delays increase exponentially
- [ ] Test jitter adds randomness`
					},
					{
						title: 'Send notifications (desktop/webhook)',
						description: 'Integrate desktop notifications and webhook support',
						details: `## Notification System

### Notification Interface
\`\`\`python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class NotificationLevel(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class Notification:
    title: str
    message: str
    level: NotificationLevel
    job_name: str | None = None

class Notifier(ABC):
    @abstractmethod
    async def send(self, notification: Notification) -> bool: ...
\`\`\`

### Desktop Notifications (plyer)
\`\`\`python
from plyer import notification as desktop_notify

class DesktopNotifier(Notifier):
    async def send(self, notif: Notification) -> bool:
        try:
            desktop_notify.notify(
                title=notif.title,
                message=notif.message,
                app_name="Task Scheduler",
                timeout=10
            )
            return True
        except Exception:
            return False
\`\`\`

### Webhook Notifications
\`\`\`python
import httpx

class WebhookNotifier(Notifier):
    def __init__(self, url: str, secret: str | None = None):
        self.url = url
        self.secret = secret

    async def send(self, notif: Notification) -> bool:
        payload = {
            "title": notif.title,
            "message": notif.message,
            "level": notif.level.value,
            "job": notif.job_name,
            "timestamp": datetime.now().isoformat()
        }

        headers = {}
        if self.secret:
            # Add HMAC signature
            import hmac
            sig = hmac.new(
                self.secret.encode(),
                json.dumps(payload).encode(),
                "sha256"
            ).hexdigest()
            headers["X-Signature"] = sig

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.url,
                json=payload,
                headers=headers
            )
            return resp.is_success
\`\`\`

### Composite Notifier
\`\`\`python
class CompositeNotifier(Notifier):
    def __init__(self, notifiers: list[Notifier]):
        self.notifiers = notifiers

    async def send(self, notif: Notification) -> bool:
        results = await asyncio.gather(
            *[n.send(notif) for n in self.notifiers],
            return_exceptions=True
        )
        return any(r is True for r in results)
\`\`\`

### When to Notify
- Job completed successfully
- Job failed (after all retries)
- Job disabled due to repeated failures
- Scheduler started/stopped`
					}
				]
			},
			{
				name: 'TUI Interface',
				description: 'Build the terminal user interface',
				tasks: [
					{
						title: 'Set up textual or rich framework',
						description: 'Initialize the TUI framework and create base layout',
						details: `## TUI Framework Setup

### Option 1: Textual (Recommended)
\`\`\`python
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Header, Footer, DataTable, Static

class SchedulerApp(App):
    """Terminal UI for the task scheduler."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1 3;
        grid-rows: auto 1fr auto;
    }

    #job-list {
        height: 100%;
        border: solid green;
    }

    #status-bar {
        height: 3;
        background: $surface;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("a", "add_job", "Add Job"),
        ("d", "delete_job", "Delete"),
        ("e", "edit_job", "Edit"),
        ("r", "run_job", "Run Now"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            DataTable(id="job-list"),
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#job-list", DataTable)
        table.add_columns("Name", "Schedule", "Last Run", "Status")
        self.refresh_jobs()

if __name__ == "__main__":
    app = SchedulerApp()
    app.run()
\`\`\`

### Option 2: Rich (Simpler, less interactive)
\`\`\`python
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout

console = Console()

def create_layout() -> Layout:
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    return layout

def create_job_table(jobs: list[Job]) -> Table:
    table = Table(title="Scheduled Jobs")
    table.add_column("Name", style="cyan")
    table.add_column("Schedule", style="green")
    table.add_column("Next Run", style="yellow")
    table.add_column("Status", style="magenta")

    for job in jobs:
        table.add_row(
            job.name,
            job.schedule,
            str(job.next_run),
            "✓ Enabled" if job.enabled else "✗ Disabled"
        )
    return table
\`\`\`

### Project Structure
\`\`\`
scheduler/
├── __init__.py
├── main.py          # Entry point
├── app.py           # TUI application
├── models.py        # Job, Schedule dataclasses
├── repository.py    # Data persistence
├── executor.py      # Job execution
├── scheduler.py     # Scheduling logic
└── widgets/         # Custom TUI widgets
    ├── __init__.py
    ├── job_table.py
    └── job_form.py
\`\`\``
					},
					{
						title: 'Display job list with status',
						description: 'Show all jobs with their current state',
						details: `## Job List Display

### DataTable Widget
\`\`\`python
from textual.widgets import DataTable
from textual.reactive import reactive
from rich.text import Text

class JobTable(DataTable):
    """Custom DataTable for displaying jobs."""

    jobs: reactive[list[Job]] = reactive([])

    def __init__(self):
        super().__init__()
        self.cursor_type = "row"
        self.zebra_stripes = True

    def on_mount(self) -> None:
        self.add_columns(
            "Status", "Name", "Schedule",
            "Last Run", "Next Run", "Runs"
        )

    def watch_jobs(self, jobs: list[Job]) -> None:
        """React to job list changes."""
        self.clear()
        for job in jobs:
            self.add_row(
                self._status_icon(job),
                job.name,
                job.schedule,
                self._format_time(job.last_run),
                self._format_time(job.next_run),
                str(job.run_count),
                key=str(job.id)
            )

    def _status_icon(self, job: Job) -> Text:
        if job.is_running:
            return Text("⟳", style="yellow")
        elif not job.enabled:
            return Text("◯", style="dim")
        elif job.last_error:
            return Text("✗", style="red")
        else:
            return Text("✓", style="green")

    def _format_time(self, dt: datetime | None) -> str:
        if not dt:
            return "Never"
        # Show relative time for recent, absolute for old
        delta = datetime.now() - dt
        if delta.days == 0:
            if delta.seconds < 60:
                return "Just now"
            elif delta.seconds < 3600:
                return f"{delta.seconds // 60}m ago"
            else:
                return f"{delta.seconds // 3600}h ago"
        elif delta.days == 1:
            return "Yesterday"
        else:
            return dt.strftime("%Y-%m-%d")
\`\`\`

### Status Bar
\`\`\`python
class StatusBar(Static):
    """Shows scheduler status and stats."""

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static("", id="scheduler-status"),
            Static("", id="job-stats"),
            Static("", id="next-run"),
        )

    def update_stats(
        self,
        running: bool,
        total: int,
        enabled: int,
        next_job: Job | None
    ):
        status = self.query_one("#scheduler-status")
        status.update(
            "🟢 Running" if running else "🔴 Stopped"
        )

        stats = self.query_one("#job-stats")
        stats.update(f"Jobs: {enabled}/{total} enabled")

        next_run = self.query_one("#next-run")
        if next_job:
            next_run.update(
                f"Next: {next_job.name} at {next_job.next_run}"
            )
\`\`\`

### Auto-Refresh
\`\`\`python
class SchedulerApp(App):
    def on_mount(self) -> None:
        # Refresh job list every second
        self.set_interval(1, self.refresh_jobs)

    def refresh_jobs(self) -> None:
        jobs = self.repository.list()
        table = self.query_one(JobTable)
        table.jobs = jobs
\`\`\``
					},
					{
						title: 'Add/edit/delete jobs via TUI',
						description: 'Implement CRUD operations through the interface',
						details: `## Job CRUD in TUI

### Job Form Modal
\`\`\`python
from textual.screen import ModalScreen
from textual.widgets import Input, Button, Select
from textual.containers import Vertical, Horizontal

class JobFormScreen(ModalScreen[Job | None]):
    """Modal form for creating/editing jobs."""

    def __init__(self, job: Job | None = None):
        super().__init__()
        self.job = job  # None = create, Job = edit

    def compose(self) -> ComposeResult:
        with Vertical(id="form-container"):
            yield Static(
                "Edit Job" if self.job else "New Job",
                id="form-title"
            )
            yield Input(
                placeholder="Job name",
                id="name",
                value=self.job.name if self.job else ""
            )
            yield Input(
                placeholder="Command to run",
                id="command",
                value=self.job.command if self.job else ""
            )
            yield Input(
                placeholder="Schedule (cron or interval)",
                id="schedule",
                value=self.job.schedule if self.job else ""
            )
            with Horizontal(id="form-buttons"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            job = self._build_job()
            if job:
                self.dismiss(job)
        else:
            self.dismiss(None)

    def _build_job(self) -> Job | None:
        name = self.query_one("#name", Input).value
        command = self.query_one("#command", Input).value
        schedule = self.query_one("#schedule", Input).value

        # Validate
        if not all([name, command, schedule]):
            self.notify("All fields required", severity="error")
            return None

        return Job(
            id=self.job.id if self.job else None,
            name=name,
            command=command,
            schedule=schedule
        )
\`\`\`

### Delete Confirmation
\`\`\`python
class ConfirmDialog(ModalScreen[bool]):
    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-container"):
            yield Static(self.message)
            with Horizontal():
                yield Button("Yes", id="yes", variant="error")
                yield Button("No", id="no", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")
\`\`\`

### App Actions
\`\`\`python
class SchedulerApp(App):
    async def action_add_job(self) -> None:
        job = await self.push_screen_wait(JobFormScreen())
        if job:
            self.repository.create(job)
            self.notify(f"Created job: {job.name}")
            self.refresh_jobs()

    async def action_edit_job(self) -> None:
        table = self.query_one(JobTable)
        if table.cursor_row is not None:
            job_id = table.get_row_at(table.cursor_row).key
            job = self.repository.get(int(job_id))
            updated = await self.push_screen_wait(JobFormScreen(job))
            if updated:
                self.repository.update(updated)
                self.notify(f"Updated job: {updated.name}")
                self.refresh_jobs()

    async def action_delete_job(self) -> None:
        table = self.query_one(JobTable)
        if table.cursor_row is not None:
            job_id = table.get_row_at(table.cursor_row).key
            job = self.repository.get(int(job_id))
            confirmed = await self.push_screen_wait(
                ConfirmDialog(f"Delete '{job.name}'?")
            )
            if confirmed:
                self.repository.delete(int(job_id))
                self.notify(f"Deleted job: {job.name}")
                self.refresh_jobs()
\`\`\``
					},
					{
						title: 'Show job execution logs',
						description: 'Display real-time and historical job output',
						details: `## Job Execution Logs

### Log Viewer Widget
\`\`\`python
from textual.widgets import RichLog, Static, TabPane, TabbedContent
from textual.containers import Vertical

class LogViewer(Vertical):
    """Display job execution logs."""

    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Output", id="stdout-tab"):
                yield RichLog(id="stdout-log", highlight=True)
            with TabPane("Errors", id="stderr-tab"):
                yield RichLog(id="stderr-log", highlight=True)
            with TabPane("History", id="history-tab"):
                yield DataTable(id="run-history")

    def on_mount(self) -> None:
        history = self.query_one("#run-history", DataTable)
        history.add_columns(
            "Started", "Duration", "Exit Code", "Status"
        )

    def show_run(self, run: JobRun) -> None:
        stdout_log = self.query_one("#stdout-log", RichLog)
        stderr_log = self.query_one("#stderr-log", RichLog)

        stdout_log.clear()
        stderr_log.clear()

        if run.stdout:
            stdout_log.write(run.stdout)
        if run.stderr:
            stderr_log.write(run.stderr, style="red")

    def load_history(self, runs: list[JobRun]) -> None:
        history = self.query_one("#run-history", DataTable)
        history.clear()
        for run in runs:
            duration = (
                run.finished_at - run.started_at
            ).total_seconds() if run.finished_at else "Running"

            status = "✓" if run.exit_code == 0 else f"✗ ({run.exit_code})"

            history.add_row(
                run.started_at.strftime("%H:%M:%S"),
                f"{duration:.1f}s" if isinstance(duration, float) else duration,
                str(run.exit_code or "-"),
                status
            )
\`\`\`

### Real-time Log Streaming
\`\`\`python
import asyncio
from collections.abc import AsyncIterator

class JobExecutor:
    async def execute_streaming(
        self,
        job: Job
    ) -> AsyncIterator[tuple[str, str]]:
        """Execute job and stream output."""
        proc = await asyncio.create_subprocess_shell(
            job.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def read_stream(stream, name):
            while True:
                line = await stream.readline()
                if not line:
                    break
                yield (name, line.decode())

        # Merge stdout and stderr streams
        async for source, line in merge_streams(
            read_stream(proc.stdout, "stdout"),
            read_stream(proc.stderr, "stderr")
        ):
            yield (source, line)

        await proc.wait()
\`\`\`

### Log Panel in App
\`\`\`python
class SchedulerApp(App):
    CSS = \"\"\"
    #main-container {
        layout: horizontal;
    }
    #job-panel { width: 60%; }
    #log-panel { width: 40%; }
    \"\"\"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="job-panel"):
                yield JobTable()
            with Vertical(id="log-panel"):
                yield LogViewer()
        yield Footer()

    def on_data_table_row_selected(
        self,
        event: DataTable.RowSelected
    ) -> None:
        job_id = int(event.row_key.value)
        runs = self.repository.get_runs(job_id, limit=20)

        log_viewer = self.query_one(LogViewer)
        log_viewer.load_history(runs)

        if runs:
            log_viewer.show_run(runs[0])
\`\`\``
					}
				]
			}
		]
	},
	{
		name: 'Log Aggregator & Analyzer',
		description: 'Parse logs from multiple formats, detect anomalies, generate reports.',
		language: 'Python',
		color: 'purple',
		skills: 'regex, generators, file streaming, data structures',
		startHint: 'Parse a single log format and count error occurrences',
		difficulty: 'intermediate',
		estimatedWeeks: 2,
		schedule: `## 2-Week Schedule

### Week 1: Log Parsing
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Nginx Parser | Implement regex-based nginx access log parser |
| Day 2 | Systemd Parser | Parse journald JSON output |
| Day 3 | Custom Formats | Build configurable parser with YAML config |
| Day 4 | Streaming | Implement generator-based file reading |
| Day 5 | Integration | Combine parsers, handle multiple files |
| Weekend | Buffer | Test with real log files, fix edge cases |

### Week 2: Analysis & Reporting
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Statistics | Implement counting, rates, distributions |
| Day 2 | Anomaly Detection | Build spike detection with z-scores |
| Day 3 | HTML Reports | Create Jinja2 templates with Chart.js |
| Day 4 | Terminal Output | Rich tables and ASCII charts |
| Day 5 | CLI Interface | argparse with subcommands |
| Weekend | Polish | Performance testing, documentation |

### Daily Commitment
- **Minimum**: 1-2 hours focused coding
- **Ideal**: 3-4 hours with breaks`,
		modules: [
			{
				name: 'Log Parsing',
				description: 'Build parsers for various log formats',
				tasks: [
					{
						title: 'Parse nginx log format',
						description: 'Handle common nginx access and error log patterns',
						details: `## Nginx Log Parsing

### Combined Log Format
\`\`\`
$remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent "$http_referer" "$http_user_agent"
\`\`\`

### Example Log Line
\`\`\`
192.168.1.1 - - [10/Oct/2023:13:55:36 -0700] "GET /api/users HTTP/1.1" 200 1234 "https://example.com" "Mozilla/5.0..."
\`\`\`

### Regex Parser
\`\`\`python
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class NginxLogEntry:
    remote_addr: str
    remote_user: str | None
    timestamp: datetime
    method: str
    path: str
    protocol: str
    status: int
    bytes_sent: int
    referer: str | None
    user_agent: str

NGINX_PATTERN = re.compile(
    r'(?P<remote_addr>\\S+) - (?P<remote_user>\\S+) '
    r'\\[(?P<time>[^\\]]+)\\] '
    r'"(?P<method>\\S+) (?P<path>\\S+) (?P<protocol>[^"]+)" '
    r'(?P<status>\\d+) (?P<bytes>\\d+) '
    r'"(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)"'
)

def parse_nginx_line(line: str) -> NginxLogEntry | None:
    match = NGINX_PATTERN.match(line)
    if not match:
        return None

    d = match.groupdict()
    return NginxLogEntry(
        remote_addr=d['remote_addr'],
        remote_user=d['remote_user'] if d['remote_user'] != '-' else None,
        timestamp=datetime.strptime(
            d['time'], '%d/%b/%Y:%H:%M:%S %z'
        ),
        method=d['method'],
        path=d['path'],
        protocol=d['protocol'],
        status=int(d['status']),
        bytes_sent=int(d['bytes']),
        referer=d['referer'] if d['referer'] != '-' else None,
        user_agent=d['user_agent']
    )
\`\`\`

### Error Log Format
\`\`\`python
# Format: YYYY/MM/DD HH:MM:SS [level] pid#tid: *cid message
ERROR_PATTERN = re.compile(
    r'(?P<time>[\\d/]+ [\\d:]+) '
    r'\\[(?P<level>\\w+)\\] '
    r'(?P<pid>\\d+)#(?P<tid>\\d+): '
    r'(?:\\*(?P<cid>\\d+) )?'
    r'(?P<message>.+)'
)
\`\`\``
					},
					{
						title: 'Parse systemd journal format',
						description: 'Extract structured data from journald logs',
						details: `## Systemd Journal Parsing

### Using journalctl JSON Output
\`\`\`bash
journalctl -o json --since "1 hour ago"
\`\`\`

### JSON Structure
\`\`\`json
{
  "__REALTIME_TIMESTAMP": "1697045736000000",
  "__MONOTONIC_TIMESTAMP": "12345678",
  "_BOOT_ID": "abc123...",
  "_HOSTNAME": "server1",
  "_SYSTEMD_UNIT": "nginx.service",
  "PRIORITY": "6",
  "SYSLOG_IDENTIFIER": "nginx",
  "MESSAGE": "Started nginx web server"
}
\`\`\`

### Parser Implementation
\`\`\`python
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

@dataclass
class JournalEntry:
    timestamp: datetime
    hostname: str
    unit: str | None
    priority: int
    identifier: str
    message: str
    pid: int | None

    @property
    def priority_name(self) -> str:
        names = ['emerg', 'alert', 'crit', 'err',
                 'warning', 'notice', 'info', 'debug']
        return names[self.priority] if self.priority < 8 else 'unknown'

def parse_journal_entry(data: dict) -> JournalEntry:
    ts = int(data.get('__REALTIME_TIMESTAMP', 0))
    return JournalEntry(
        timestamp=datetime.fromtimestamp(ts / 1_000_000),
        hostname=data.get('_HOSTNAME', ''),
        unit=data.get('_SYSTEMD_UNIT'),
        priority=int(data.get('PRIORITY', 6)),
        identifier=data.get('SYSLOG_IDENTIFIER', ''),
        message=data.get('MESSAGE', ''),
        pid=int(data['_PID']) if '_PID' in data else None
    )

def read_journal(
    since: str = "1 hour ago",
    unit: str | None = None
) -> Iterator[JournalEntry]:
    cmd = ['journalctl', '-o', 'json', '--since', since]
    if unit:
        cmd.extend(['-u', unit])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        text=True
    )

    for line in proc.stdout:
        if line.strip():
            data = json.loads(line)
            yield parse_journal_entry(data)
\`\`\`

### Filtering by Priority
\`\`\`python
def get_errors(entries: Iterator[JournalEntry]) -> Iterator[JournalEntry]:
    for entry in entries:
        if entry.priority <= 3:  # err, crit, alert, emerg
            yield entry
\`\`\``
					},
					{
						title: 'Parse custom log formats',
						description: 'Create configurable parser for arbitrary formats',
						details: `## Configurable Log Parser

### Format Specification
\`\`\`yaml
# log_formats.yaml
formats:
  apache_common:
    pattern: '{ip} {ident} {user} [{time}] "{request}" {status} {size}'
    fields:
      ip: { type: ip }
      ident: { type: string, nullable: true, null_value: "-" }
      user: { type: string, nullable: true, null_value: "-" }
      time: { type: datetime, format: "%d/%b/%Y:%H:%M:%S %z" }
      request: { type: string }
      status: { type: int }
      size: { type: int }

  custom_app:
    pattern: '{timestamp} [{level}] {logger}: {message}'
    fields:
      timestamp: { type: datetime, format: "%Y-%m-%d %H:%M:%S.%f" }
      level: { type: enum, values: [DEBUG, INFO, WARN, ERROR] }
      logger: { type: string }
      message: { type: string, greedy: true }
\`\`\`

### Parser Generator
\`\`\`python
import re
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class FieldSpec:
    name: str
    type: str
    format: str | None = None
    nullable: bool = False
    null_value: str = "-"
    greedy: bool = False

@dataclass
class LogFormat:
    name: str
    pattern: str
    fields: list[FieldSpec]
    _regex: re.Pattern = field(init=False)

    def __post_init__(self):
        self._regex = self._build_regex()

    def _build_regex(self) -> re.Pattern:
        pattern = self.pattern
        for f in self.fields:
            if f.greedy:
                replacement = f'(?P<{f.name}>.+)'
            else:
                replacement = f'(?P<{f.name}>\\\\S+)'
            pattern = pattern.replace('{' + f.name + '}', replacement)
        return re.compile(pattern)

    def parse(self, line: str) -> dict[str, Any] | None:
        match = self._regex.match(line)
        if not match:
            return None

        result = {}
        for f in self.fields:
            raw = match.group(f.name)
            result[f.name] = self._convert(raw, f)
        return result

    def _convert(self, value: str, spec: FieldSpec) -> Any:
        if spec.nullable and value == spec.null_value:
            return None
        if spec.type == 'int':
            return int(value)
        if spec.type == 'datetime':
            return datetime.strptime(value, spec.format)
        return value

# Usage
format = LogFormat.from_yaml('log_formats.yaml', 'apache_common')
for line in open('access.log'):
    entry = format.parse(line)
\`\`\``
					},
					{
						title: 'Stream large files without loading into memory',
						description: 'Use generators for memory-efficient processing',
						details: `## Memory-Efficient Log Processing

### Generator-Based Reading
\`\`\`python
from pathlib import Path
from typing import Iterator, TypeVar
import gzip
import bz2

T = TypeVar('T')

def read_log_file(path: Path) -> Iterator[str]:
    """Read log file, handling compression automatically."""
    openers = {
        '.gz': gzip.open,
        '.bz2': bz2.open,
    }
    opener = openers.get(path.suffix, open)

    with opener(path, 'rt') as f:
        for line in f:
            yield line.rstrip('\\n')

def read_multiple_logs(paths: list[Path]) -> Iterator[str]:
    """Read multiple log files in order."""
    for path in sorted(paths):
        yield from read_log_file(path)
\`\`\`

### Chunked Processing
\`\`\`python
from itertools import islice

def chunked(iterator: Iterator[T], size: int) -> Iterator[list[T]]:
    """Yield chunks of items from iterator."""
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            break
        yield chunk

# Process in batches of 1000
for batch in chunked(read_log_file(path), 1000):
    process_batch(batch)
\`\`\`

### Parallel Processing with Queues
\`\`\`python
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

def process_logs_parallel(
    paths: list[Path],
    parser: Callable[[str], T],
    processor: Callable[[T], None],
    workers: int = 4
) -> None:
    work_queue: queue.Queue = queue.Queue(maxsize=10000)
    done = threading.Event()

    def reader():
        for path in paths:
            for line in read_log_file(path):
                work_queue.put(line)
        done.set()

    def worker():
        while not (done.is_set() and work_queue.empty()):
            try:
                line = work_queue.get(timeout=0.1)
                entry = parser(line)
                if entry:
                    processor(entry)
            except queue.Empty:
                continue

    reader_thread = threading.Thread(target=reader)
    reader_thread.start()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker) for _ in range(workers)]
        for f in futures:
            f.result()

    reader_thread.join()
\`\`\`

### Memory Monitoring
\`\`\`python
import tracemalloc

tracemalloc.start()

# Process logs...
for entry in parse_all_logs(paths):
    process(entry)

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
\`\`\``
					}
				]
			},
			{
				name: 'Analysis & Reporting',
				description: 'Detect patterns and generate insights',
				tasks: [
					{
						title: 'Detect spikes in error rates',
						description: 'Implement anomaly detection for error frequency',
						details: `## Error Rate Anomaly Detection

### Time-Bucketed Error Counting
\`\`\`python
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class ErrorBucket:
    timestamp: datetime
    count: int
    errors: list[str]

def bucket_errors(
    entries: Iterator[LogEntry],
    bucket_size: timedelta = timedelta(minutes=5)
) -> Iterator[ErrorBucket]:
    """Group errors into time buckets."""
    current_bucket: datetime | None = None
    count = 0
    errors = []

    for entry in entries:
        if entry.status >= 400:  # Error status
            bucket = entry.timestamp.replace(
                minute=entry.timestamp.minute // 5 * 5,
                second=0, microsecond=0
            )

            if bucket != current_bucket:
                if current_bucket:
                    yield ErrorBucket(current_bucket, count, errors)
                current_bucket = bucket
                count = 0
                errors = []

            count += 1
            errors.append(f"{entry.status}: {entry.path}")

    if current_bucket:
        yield ErrorBucket(current_bucket, count, errors)
\`\`\`

### Z-Score Anomaly Detection
\`\`\`python
import statistics

def detect_anomalies(
    buckets: list[ErrorBucket],
    threshold: float = 2.0
) -> list[ErrorBucket]:
    """Find buckets with error counts > threshold std devs from mean."""
    if len(buckets) < 10:
        return []

    counts = [b.count for b in buckets]
    mean = statistics.mean(counts)
    stdev = statistics.stdev(counts)

    if stdev == 0:
        return []

    anomalies = []
    for bucket in buckets:
        z_score = (bucket.count - mean) / stdev
        if z_score > threshold:
            anomalies.append(bucket)

    return anomalies
\`\`\`

### Moving Average Detection
\`\`\`python
from collections import deque

def detect_spikes_moving_avg(
    buckets: Iterator[ErrorBucket],
    window_size: int = 12,
    multiplier: float = 3.0
) -> Iterator[tuple[ErrorBucket, float]]:
    """Detect when error count exceeds moving average * multiplier."""
    window = deque(maxlen=window_size)

    for bucket in buckets:
        if len(window) >= window_size:
            avg = sum(window) / len(window)
            if bucket.count > avg * multiplier:
                yield bucket, bucket.count / avg if avg > 0 else float('inf')

        window.append(bucket.count)
\`\`\``
					},
					{
						title: 'Calculate log statistics',
						description: 'Aggregate counts, rates, and distributions',
						details: `## Log Statistics Aggregation

### Statistics Collector
\`\`\`python
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterator

@dataclass
class LogStats:
    total_requests: int = 0
    total_bytes: int = 0
    status_counts: Counter = field(default_factory=Counter)
    path_counts: Counter = field(default_factory=Counter)
    method_counts: Counter = field(default_factory=Counter)
    hourly_counts: Counter = field(default_factory=Counter)
    ip_counts: Counter = field(default_factory=Counter)
    user_agent_counts: Counter = field(default_factory=Counter)
    response_times: list[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        errors = sum(
            c for s, c in self.status_counts.items()
            if s >= 400
        )
        return errors / self.total_requests if self.total_requests > 0 else 0

    @property
    def avg_response_time(self) -> float:
        return (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0
        )

    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx]

def collect_stats(entries: Iterator[LogEntry]) -> LogStats:
    stats = LogStats()

    for entry in entries:
        stats.total_requests += 1
        stats.total_bytes += entry.bytes_sent
        stats.status_counts[entry.status] += 1
        stats.path_counts[entry.path] += 1
        stats.method_counts[entry.method] += 1
        stats.hourly_counts[entry.timestamp.hour] += 1
        stats.ip_counts[entry.remote_addr] += 1

        if entry.user_agent:
            # Simplify user agent
            ua = entry.user_agent.split('/')[0]
            stats.user_agent_counts[ua] += 1

        if entry.response_time:
            stats.response_times.append(entry.response_time)

    return stats
\`\`\`

### Percentile Calculation
\`\`\`python
def percentiles(values: list[float], pcts: list[int]) -> dict[int, float]:
    """Calculate multiple percentiles efficiently."""
    if not values:
        return {p: 0 for p in pcts}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    return {
        p: sorted_vals[int(n * p / 100)]
        for p in pcts
    }

# Usage
times = stats.response_times
pcts = percentiles(times, [50, 90, 95, 99])
print(f"p50: {pcts[50]:.2f}ms, p99: {pcts[99]:.2f}ms")
\`\`\``
					},
					{
						title: 'Output reports as HTML',
						description: 'Generate formatted HTML reports with charts',
						details: `## HTML Report Generation

### Jinja2 Template
\`\`\`html
<!-- templates/report.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Log Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: system-ui; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .card { background: #f5f5f5; border-radius: 8px; padding: 20px; margin: 20px 0; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
        .stat { text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #333; }
        .stat-label { color: #666; }
        .error { color: #dc3545; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>Log Analysis Report</h1>
    <p>Generated: {{ generated_at }}</p>

    <div class="stats-grid">
        <div class="stat">
            <div class="stat-value">{{ stats.total_requests | format_number }}</div>
            <div class="stat-label">Total Requests</div>
        </div>
        <div class="stat">
            <div class="stat-value {{ 'error' if stats.error_rate > 0.05 }}">
                {{ "%.2f%%" | format(stats.error_rate * 100) }}
            </div>
            <div class="stat-label">Error Rate</div>
        </div>
        <!-- More stats -->
    </div>

    <div class="card">
        <h2>Requests by Hour</h2>
        <canvas id="hourlyChart"></canvas>
    </div>

    <script>
        new Chart(document.getElementById('hourlyChart'), {
            type: 'bar',
            data: {
                labels: {{ hourly_labels | tojson }},
                datasets: [{
                    label: 'Requests',
                    data: {{ hourly_data | tojson }},
                    backgroundColor: '#4CAF50'
                }]
            }
        });
    </script>
</body>
</html>
\`\`\`

### Report Generator
\`\`\`python
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

def generate_html_report(
    stats: LogStats,
    output_path: Path
) -> None:
    env = Environment(loader=FileSystemLoader('templates'))
    env.filters['format_number'] = lambda x: f"{x:,}"

    template = env.get_template('report.html')

    hourly_labels = [f"{h:02d}:00" for h in range(24)]
    hourly_data = [stats.hourly_counts.get(h, 0) for h in range(24)]

    html = template.render(
        stats=stats,
        generated_at=datetime.now().isoformat(),
        hourly_labels=hourly_labels,
        hourly_data=hourly_data,
        top_paths=stats.path_counts.most_common(10),
        top_ips=stats.ip_counts.most_common(10),
        status_distribution=dict(stats.status_counts)
    )

    output_path.write_text(html)
\`\`\``
					},
					{
						title: 'Output reports as terminal tables',
						description: 'Display results in formatted terminal output',
						details: `## Terminal Table Output

### Using Rich Library
\`\`\`python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

console = Console()

def print_stats_summary(stats: LogStats) -> None:
    """Print summary statistics in panels."""
    panels = [
        Panel(
            f"[bold]{stats.total_requests:,}[/bold]",
            title="Total Requests"
        ),
        Panel(
            f"[bold {'red' if stats.error_rate > 0.05 else 'green'}]"
            f"{stats.error_rate:.2%}[/bold]",
            title="Error Rate"
        ),
        Panel(
            f"[bold]{stats.total_bytes / 1024 / 1024:.1f} MB[/bold]",
            title="Data Transferred"
        ),
        Panel(
            f"[bold]{stats.avg_response_time:.0f}ms[/bold]",
            title="Avg Response"
        ),
    ]
    console.print(Columns(panels))

def print_status_table(stats: LogStats) -> None:
    """Print status code breakdown."""
    table = Table(title="Status Codes", box=box.ROUNDED)
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_column("Bar", width=20)

    total = stats.total_requests
    for status, count in sorted(stats.status_counts.items()):
        pct = count / total * 100
        bar_len = int(pct / 5)
        bar = "█" * bar_len

        style = "green" if status < 400 else "red"
        table.add_row(
            str(status),
            f"{count:,}",
            f"{pct:.1f}%",
            f"[{style}]{bar}[/{style}]"
        )

    console.print(table)

def print_top_paths(stats: LogStats, n: int = 10) -> None:
    """Print top requested paths."""
    table = Table(title=f"Top {n} Paths", box=box.ROUNDED)
    table.add_column("Path", style="blue", no_wrap=True)
    table.add_column("Hits", justify="right")
    table.add_column("% of Total", justify="right")

    for path, count in stats.path_counts.most_common(n):
        pct = count / stats.total_requests * 100
        table.add_row(
            path[:50] + "..." if len(path) > 50 else path,
            f"{count:,}",
            f"{pct:.2f}%"
        )

    console.print(table)
\`\`\`

### ASCII Charts
\`\`\`python
def print_hourly_chart(stats: LogStats) -> None:
    """Print ASCII bar chart of hourly traffic."""
    max_count = max(stats.hourly_counts.values()) if stats.hourly_counts else 1
    width = 40

    console.print("\\n[bold]Requests by Hour[/bold]")
    for hour in range(24):
        count = stats.hourly_counts.get(hour, 0)
        bar_len = int(count / max_count * width)
        bar = "▓" * bar_len
        console.print(f"{hour:02d}:00 │{bar} {count:,}")
\`\`\``
					}
				]
			}
		]
	},
	{
		name: 'Git Repository Analyzer',
		description: 'Walk commit history and generate insights about a codebase.',
		language: 'Python',
		color: 'emerald',
		skills: 'tree traversal, data aggregation, working with external libraries',
		startHint: 'List all commits with author and date using gitpython',
		difficulty: 'intermediate',
		estimatedWeeks: 2,
		schedule: `## 2-Week Schedule

### Week 1: Commit Analysis
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Setup | Install gitpython, basic repo walking |
| Day 2 | Author Stats | Commits per author, first/last commit dates |
| Day 3 | File Churn | Track most modified files, identify hotspots |
| Day 4 | Large Commits | Detect oversized commits, binary files |
| Day 5 | Message Analysis | Parse conventional commits, quality scoring |
| Weekend | Buffer | Test on various repos, handle edge cases |

### Week 2: Visualization
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Activity Charts | ASCII contribution grid, weekly bars |
| Day 2 | Contributor Profiles | Detailed per-author statistics |
| Day 3 | Ownership Heatmap | File ownership visualization |
| Day 4 | Markdown Reports | Generate shareable report files |
| Day 5 | CLI Polish | Arguments, progress bars, colored output |
| Weekend | Documentation | README, example outputs |

### Daily Commitment
- **Minimum**: 1-2 hours focused coding
- **Ideal**: 3-4 hours with breaks`,
		modules: [
			{
				name: 'Commit Analysis',
				description: 'Extract and analyze commit data',
				tasks: [
					{
						title: 'Calculate commits per author',
						description: 'Aggregate commit counts by contributor',
						details: `## Author Commit Analysis

### Using GitPython
\`\`\`python
from git import Repo
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AuthorStats:
    name: str
    email: str
    commit_count: int
    first_commit: datetime
    last_commit: datetime
    lines_added: int = 0
    lines_removed: int = 0
    files_touched: set = None

def get_author_stats(repo_path: str) -> dict[str, AuthorStats]:
    repo = Repo(repo_path)
    authors: dict[str, AuthorStats] = {}

    for commit in repo.iter_commits('HEAD'):
        key = commit.author.email

        if key not in authors:
            authors[key] = AuthorStats(
                name=commit.author.name,
                email=commit.author.email,
                commit_count=0,
                first_commit=commit.authored_datetime,
                last_commit=commit.authored_datetime,
                files_touched=set()
            )

        stats = authors[key]
        stats.commit_count += 1
        stats.first_commit = min(stats.first_commit, commit.authored_datetime)
        stats.last_commit = max(stats.last_commit, commit.authored_datetime)

        # Get diff stats
        if commit.parents:
            diff = commit.parents[0].diff(commit)
            for d in diff:
                if d.a_path:
                    stats.files_touched.add(d.a_path)

    return authors
\`\`\`

### Sorting and Display
\`\`\`python
def print_author_leaderboard(authors: dict[str, AuthorStats]) -> None:
    sorted_authors = sorted(
        authors.values(),
        key=lambda a: a.commit_count,
        reverse=True
    )

    print(f"{'Author':<30} {'Commits':>8} {'Files':>8} {'Active Days':>12}")
    print("-" * 60)

    for author in sorted_authors[:20]:
        active_days = (author.last_commit - author.first_commit).days
        print(
            f"{author.name:<30} "
            f"{author.commit_count:>8} "
            f"{len(author.files_touched):>8} "
            f"{active_days:>12}"
        )
\`\`\`

### Grouping by Time Period
\`\`\`python
def commits_by_month(repo_path: str) -> dict[str, Counter]:
    """Get commit counts per author per month."""
    repo = Repo(repo_path)
    monthly: dict[str, Counter] = defaultdict(Counter)

    for commit in repo.iter_commits('HEAD'):
        month = commit.authored_datetime.strftime('%Y-%m')
        author = commit.author.email
        monthly[month][author] += 1

    return monthly
\`\`\``
					},
					{
						title: 'Track file churn (most modified files)',
						description: 'Identify hotspots in the codebase',
						details: `## File Churn Analysis

### What is File Churn?
Files that change frequently often indicate:
- Unstable or evolving requirements
- Technical debt
- Bug-prone areas
- Areas needing refactoring

### Implementation
\`\`\`python
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

@dataclass
class FileChurn:
    path: str
    change_count: int
    authors: set[str]
    total_lines_changed: int
    last_modified: datetime

def analyze_file_churn(repo_path: str, branch: str = 'HEAD') -> list[FileChurn]:
    repo = Repo(repo_path)
    file_stats: dict[str, FileChurn] = {}

    for commit in repo.iter_commits(branch):
        if not commit.parents:
            continue

        diff = commit.parents[0].diff(commit)
        for d in diff:
            path = d.a_path or d.b_path
            if not path:
                continue

            if path not in file_stats:
                file_stats[path] = FileChurn(
                    path=path,
                    change_count=0,
                    authors=set(),
                    total_lines_changed=0,
                    last_modified=commit.authored_datetime
                )

            stats = file_stats[path]
            stats.change_count += 1
            stats.authors.add(commit.author.email)

            # Count lines changed
            if d.diff:
                lines = d.diff.decode('utf-8', errors='ignore').split('\\n')
                stats.total_lines_changed += sum(
                    1 for line in lines
                    if line.startswith('+') or line.startswith('-')
                )

    return sorted(
        file_stats.values(),
        key=lambda f: f.change_count,
        reverse=True
    )
\`\`\`

### Filtering by Time
\`\`\`python
def recent_churn(
    repo_path: str,
    days: int = 90
) -> list[FileChurn]:
    """Only consider commits from the last N days."""
    repo = Repo(repo_path)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # ... same logic but filter by commit date
    for commit in repo.iter_commits('HEAD'):
        if commit.authored_datetime < cutoff:
            break
        # ... process commit
\`\`\`

### Churn vs Complexity
\`\`\`python
# High churn + high complexity = danger zone
def complexity_churn_analysis(repo_path: str):
    churn = analyze_file_churn(repo_path)

    for file in churn[:20]:
        if Path(file.path).suffix == '.py':
            # Use radon for complexity
            complexity = get_complexity(file.path)
            print(f"{file.path}: churn={file.change_count}, complexity={complexity}")
\`\`\``
					},
					{
						title: 'Detect large commits or binary files',
						description: 'Flag commits that may need review',
						details: `## Large Commit Detection

### What to Flag
- Commits with many files changed (> 50)
- Large line count changes (> 1000 lines)
- Binary files added
- Generated files committed
- Large file additions (> 1MB)

### Implementation
\`\`\`python
from dataclasses import dataclass
from typing import Literal

@dataclass
class CommitFlag:
    commit_sha: str
    message: str
    author: str
    reason: str
    severity: Literal['warning', 'critical']
    details: dict

def detect_large_commits(
    repo_path: str,
    max_files: int = 50,
    max_lines: int = 1000
) -> list[CommitFlag]:
    repo = Repo(repo_path)
    flags = []

    for commit in repo.iter_commits('HEAD', max_count=500):
        if not commit.parents:
            continue

        diff = commit.parents[0].diff(commit)

        # Check file count
        if len(diff) > max_files:
            flags.append(CommitFlag(
                commit_sha=commit.hexsha[:8],
                message=commit.message.split('\\n')[0][:50],
                author=commit.author.email,
                reason="Too many files changed",
                severity='warning',
                details={'file_count': len(diff)}
            ))

        # Check line count
        stats = commit.stats.total
        if stats['lines'] > max_lines:
            flags.append(CommitFlag(
                commit_sha=commit.hexsha[:8],
                message=commit.message.split('\\n')[0][:50],
                author=commit.author.email,
                reason="Large line count change",
                severity='warning' if stats['lines'] < 5000 else 'critical',
                details={'lines': stats['lines']}
            ))

    return flags
\`\`\`

### Binary File Detection
\`\`\`python
BINARY_EXTENSIONS = {
    '.exe', '.dll', '.so', '.dylib',
    '.zip', '.tar', '.gz', '.rar',
    '.png', '.jpg', '.gif', '.ico',
    '.pdf', '.doc', '.docx',
    '.pyc', '.class', '.o'
}

def detect_binary_files(repo_path: str) -> list[CommitFlag]:
    repo = Repo(repo_path)
    flags = []

    for commit in repo.iter_commits('HEAD'):
        if not commit.parents:
            continue

        diff = commit.parents[0].diff(commit)
        for d in diff:
            if d.new_file and d.b_path:
                ext = Path(d.b_path).suffix.lower()
                if ext in BINARY_EXTENSIONS:
                    flags.append(CommitFlag(
                        commit_sha=commit.hexsha[:8],
                        message=commit.message.split('\\n')[0][:50],
                        author=commit.author.email,
                        reason="Binary file added",
                        severity='warning',
                        details={'file': d.b_path}
                    ))

    return flags
\`\`\``
					},
					{
						title: 'Analyze commit message patterns',
						description: 'Extract common prefixes, lengths, and styles',
						details: `## Commit Message Analysis

### Conventional Commits Detection
\`\`\`python
import re
from collections import Counter

CONVENTIONAL_PATTERN = re.compile(
    r'^(?P<type>feat|fix|docs|style|refactor|test|chore)'
    r'(?:\\((?P<scope>[^)]+)\\))?'
    r'(?P<breaking>!)?: '
    r'(?P<description>.+)$'
)

@dataclass
class MessageAnalysis:
    total_commits: int
    conventional_count: int
    type_distribution: Counter
    avg_length: float
    has_body_count: int
    prefixes: Counter

def analyze_messages(repo_path: str) -> MessageAnalysis:
    repo = Repo(repo_path)
    messages = [c.message for c in repo.iter_commits('HEAD', max_count=1000)]

    type_dist = Counter()
    conventional = 0
    prefixes = Counter()
    lengths = []
    has_body = 0

    for msg in messages:
        first_line = msg.split('\\n')[0]
        lengths.append(len(first_line))

        if '\\n\\n' in msg:
            has_body += 1

        # Check conventional format
        match = CONVENTIONAL_PATTERN.match(first_line)
        if match:
            conventional += 1
            type_dist[match.group('type')] += 1
        else:
            # Extract prefix (first word or bracket content)
            prefix_match = re.match(r'^\\[([^\\]]+)\\]|^(\\w+):', first_line)
            if prefix_match:
                prefix = prefix_match.group(1) or prefix_match.group(2)
                prefixes[prefix.lower()] += 1

    return MessageAnalysis(
        total_commits=len(messages),
        conventional_count=conventional,
        type_distribution=type_dist,
        avg_length=sum(lengths) / len(lengths) if lengths else 0,
        has_body_count=has_body,
        prefixes=prefixes
    )
\`\`\`

### Quality Scoring
\`\`\`python
def score_commit_message(msg: str) -> tuple[int, list[str]]:
    """Score message quality 0-100 with feedback."""
    score = 100
    feedback = []
    first_line = msg.split('\\n')[0]

    # Length checks
    if len(first_line) > 72:
        score -= 10
        feedback.append("Subject > 72 chars")
    if len(first_line) < 10:
        score -= 20
        feedback.append("Subject too short")

    # Capitalization
    if first_line[0].islower():
        score -= 5
        feedback.append("Should start with capital")

    # Ending
    if first_line.endswith('.'):
        score -= 5
        feedback.append("Don't end with period")

    # Imperative mood (heuristic)
    past_tense = ['added', 'fixed', 'changed', 'updated', 'removed']
    if any(first_line.lower().startswith(w) for w in past_tense):
        score -= 10
        feedback.append("Use imperative mood")

    return max(0, score), feedback
\`\`\``
					}
				]
			},
			{
				name: 'Visualization',
				description: 'Display insights visually',
				tasks: [
					{
						title: 'Visualize activity over time (ASCII charts)',
						description: 'Create terminal-based activity graphs',
						details: `## ASCII Activity Visualization

### GitHub-Style Contribution Grid
\`\`\`python
from datetime import datetime, timedelta
from collections import Counter

def get_commit_dates(repo_path: str, days: int = 365) -> Counter:
    """Get commit counts per day."""
    repo = Repo(repo_path)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    dates = Counter()
    for commit in repo.iter_commits('HEAD'):
        if commit.authored_datetime < cutoff:
            break
        date = commit.authored_datetime.date()
        dates[date] += 1

    return dates

def render_contribution_grid(dates: Counter) -> str:
    """Render GitHub-style contribution grid."""
    levels = ' ░▒▓█'
    today = datetime.now().date()
    start = today - timedelta(days=364)

    # Normalize counts to levels
    max_count = max(dates.values()) if dates else 1

    lines = []
    days = ['Mon', '   ', 'Wed', '   ', 'Fri', '   ', 'Sun']

    for day_of_week in range(7):
        line = [days[day_of_week] + ' ']
        current = start + timedelta(days=(day_of_week - start.weekday()) % 7)

        while current <= today:
            count = dates.get(current, 0)
            level = int(count / max_count * 4) if max_count > 0 else 0
            line.append(levels[level])
            current += timedelta(days=7)

        lines.append(''.join(line))

    return '\\n'.join(lines)
\`\`\`

### Weekly Activity Bar Chart
\`\`\`python
def weekly_activity_chart(dates: Counter, weeks: int = 12) -> str:
    """Horizontal bar chart of weekly commits."""
    today = datetime.now().date()
    weekly = Counter()

    for date, count in dates.items():
        week_num = (today - date).days // 7
        if week_num < weeks:
            weekly[week_num] += count

    max_count = max(weekly.values()) if weekly else 1
    width = 40

    lines = []
    for week in range(weeks - 1, -1, -1):
        count = weekly.get(week, 0)
        bar_len = int(count / max_count * width)
        bar = '█' * bar_len

        week_start = today - timedelta(days=(week + 1) * 7)
        label = week_start.strftime('%b %d')
        lines.append(f"{label} │{bar} {count}")

    return '\\n'.join(lines)
\`\`\`

### Hour-of-Day Histogram
\`\`\`python
def hourly_histogram(repo_path: str) -> str:
    """When do commits happen?"""
    repo = Repo(repo_path)
    hours = Counter()

    for commit in repo.iter_commits('HEAD', max_count=1000):
        hours[commit.authored_datetime.hour] += 1

    max_count = max(hours.values()) if hours else 1
    height = 10

    # Build vertical histogram
    rows = []
    for h in range(height, 0, -1):
        row = []
        for hour in range(24):
            count = hours.get(hour, 0)
            bar_height = int(count / max_count * height)
            row.append('█' if bar_height >= h else ' ')
        rows.append(''.join(row))

    rows.append('─' * 24)
    rows.append('0   4   8   12  16  20  ')

    return '\\n'.join(rows)
\`\`\``
					},
					{
						title: 'Generate contributor statistics',
						description: 'Show detailed per-author metrics',
						details: `## Detailed Contributor Stats

### Comprehensive Author Profile
\`\`\`python
@dataclass
class ContributorProfile:
    name: str
    email: str
    commits: int
    lines_added: int
    lines_removed: int
    files_created: int
    files_deleted: int
    first_commit: datetime
    last_commit: datetime
    active_days: int
    favorite_files: list[tuple[str, int]]
    commit_hours: Counter
    merge_commits: int

def build_contributor_profile(
    repo_path: str,
    email: str
) -> ContributorProfile:
    repo = Repo(repo_path)
    profile = ContributorProfile(
        name='', email=email, commits=0,
        lines_added=0, lines_removed=0,
        files_created=0, files_deleted=0,
        first_commit=None, last_commit=None,
        active_days=0, favorite_files=[],
        commit_hours=Counter(), merge_commits=0
    )

    active_dates = set()
    file_counts = Counter()

    for commit in repo.iter_commits('HEAD'):
        if commit.author.email != email:
            continue

        if not profile.name:
            profile.name = commit.author.name

        profile.commits += 1
        active_dates.add(commit.authored_datetime.date())
        profile.commit_hours[commit.authored_datetime.hour] += 1

        if profile.first_commit is None:
            profile.first_commit = commit.authored_datetime
        profile.last_commit = commit.authored_datetime

        if len(commit.parents) > 1:
            profile.merge_commits += 1

        # Analyze diff
        if commit.parents:
            stats = commit.stats.files
            for path, data in stats.items():
                profile.lines_added += data['insertions']
                profile.lines_removed += data['deletions']
                file_counts[path] += 1

    profile.active_days = len(active_dates)
    profile.favorite_files = file_counts.most_common(10)

    return profile
\`\`\`

### Display Profile
\`\`\`python
def print_contributor_profile(profile: ContributorProfile) -> None:
    print(f"\\n{'=' * 50}")
    print(f"  {profile.name}")
    print(f"  {profile.email}")
    print(f"{'=' * 50}")

    tenure = (profile.last_commit - profile.first_commit).days
    print(f"\\nTenure: {tenure} days")
    print(f"Active days: {profile.active_days} ({profile.active_days/tenure*100:.1f}%)")

    print(f"\\nCommits: {profile.commits}")
    print(f"  - Regular: {profile.commits - profile.merge_commits}")
    print(f"  - Merges: {profile.merge_commits}")

    print(f"\\nLines changed: +{profile.lines_added:,} / -{profile.lines_removed:,}")
    net = profile.lines_added - profile.lines_removed
    print(f"Net contribution: {'+' if net > 0 else ''}{net:,} lines")

    print(f"\\nMost active hours:")
    for hour, count in profile.commit_hours.most_common(3):
        print(f"  {hour:02d}:00 - {count} commits")

    print(f"\\nFavorite files:")
    for path, count in profile.favorite_files[:5]:
        print(f"  {path}: {count} changes")
\`\`\``
					},
					{
						title: 'Create file ownership heatmap',
						description: 'Visualize who owns which parts of code',
						details: `## File Ownership Heatmap

### Calculate Ownership
\`\`\`python
from collections import defaultdict
from pathlib import Path

def calculate_ownership(repo_path: str) -> dict[str, dict[str, float]]:
    """
    Calculate ownership percentage for each file.
    Ownership = proportion of commits touching that file.
    """
    repo = Repo(repo_path)
    file_authors: dict[str, Counter] = defaultdict(Counter)

    for commit in repo.iter_commits('HEAD'):
        if not commit.parents:
            continue

        author = commit.author.email
        diff = commit.parents[0].diff(commit)

        for d in diff:
            path = d.a_path or d.b_path
            if path:
                file_authors[path][author] += 1

    # Convert to percentages
    ownership = {}
    for path, authors in file_authors.items():
        total = sum(authors.values())
        ownership[path] = {
            author: count / total
            for author, count in authors.items()
        }

    return ownership

def get_primary_owner(
    ownership: dict[str, dict[str, float]],
    path: str
) -> tuple[str, float]:
    """Get the primary owner of a file."""
    if path not in ownership:
        return ('unknown', 0.0)

    authors = ownership[path]
    primary = max(authors.items(), key=lambda x: x[1])
    return primary
\`\`\`

### Directory-Level Ownership
\`\`\`python
def directory_ownership(
    ownership: dict[str, dict[str, float]]
) -> dict[str, Counter]:
    """Aggregate ownership by directory."""
    dir_authors: dict[str, Counter] = defaultdict(Counter)

    for path, authors in ownership.items():
        directory = str(Path(path).parent)

        for author, pct in authors.items():
            dir_authors[directory][author] += pct

    return dir_authors
\`\`\`

### ASCII Heatmap
\`\`\`python
def render_ownership_heatmap(
    ownership: dict[str, dict[str, float]],
    directory: str = '.'
) -> str:
    """Render ownership as ASCII heatmap."""
    # Get unique authors
    all_authors = set()
    for authors in ownership.values():
        all_authors.update(authors.keys())

    # Assign colors/symbols to top authors
    author_list = list(all_authors)[:8]
    symbols = '█▓▒░●○◆◇'

    lines = []
    lines.append("Legend: " + " ".join(
        f"{symbols[i]}={a.split('@')[0][:8]}"
        for i, a in enumerate(author_list)
    ))
    lines.append("")

    for path, authors in sorted(ownership.items()):
        if not path.startswith(directory):
            continue

        primary, pct = max(authors.items(), key=lambda x: x[1])
        idx = author_list.index(primary) if primary in author_list else -1
        symbol = symbols[idx] if idx >= 0 else '?'

        bar_len = int(pct * 20)
        bar = symbol * bar_len + '·' * (20 - bar_len)

        lines.append(f"{bar} {pct:>5.1%} {path}")

    return '\\n'.join(lines)
\`\`\``
					},
					{
						title: 'Export report as markdown',
						description: 'Generate shareable markdown reports',
						details: `## Markdown Report Generation

### Report Template
\`\`\`python
from datetime import datetime
from io import StringIO

def generate_markdown_report(
    repo_path: str,
    author_stats: dict[str, AuthorStats],
    churn: list[FileChurn],
    message_analysis: MessageAnalysis
) -> str:
    """Generate comprehensive markdown report."""
    out = StringIO()

    repo = Repo(repo_path)
    repo_name = Path(repo_path).name

    # Header
    out.write(f"# Repository Analysis: {repo_name}\\n\\n")
    out.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\\n\\n")

    # Summary
    out.write("## Summary\\n\\n")
    out.write(f"- **Total commits**: {message_analysis.total_commits:,}\\n")
    out.write(f"- **Contributors**: {len(author_stats)}\\n")
    out.write(f"- **Conventional commits**: {message_analysis.conventional_count} ")
    out.write(f"({message_analysis.conventional_count/message_analysis.total_commits*100:.1f}%)\\n")
    out.write(f"- **Avg message length**: {message_analysis.avg_length:.0f} chars\\n\\n")

    # Top contributors
    out.write("## Top Contributors\\n\\n")
    out.write("| Author | Commits | Files Touched | Active Period |\\n")
    out.write("|--------|---------|---------------|---------------|\\n")

    sorted_authors = sorted(
        author_stats.values(),
        key=lambda a: a.commit_count,
        reverse=True
    )[:10]

    for author in sorted_authors:
        period = f"{author.first_commit.strftime('%Y-%m')} to {author.last_commit.strftime('%Y-%m')}"
        out.write(
            f"| {author.name} | {author.commit_count:,} | "
            f"{len(author.files_touched)} | {period} |\\n"
        )

    out.write("\\n")

    # Hot files
    out.write("## Hotspots (Most Changed Files)\\n\\n")
    out.write("| File | Changes | Authors | Risk |\\n")
    out.write("|------|---------|---------|------|\\n")

    for file in churn[:15]:
        risk = "🔴" if file.change_count > 50 else "🟡" if file.change_count > 20 else "🟢"
        out.write(
            f"| \`{file.path}\` | {file.change_count} | "
            f"{len(file.authors)} | {risk} |\\n"
        )

    out.write("\\n")

    # Commit types
    if message_analysis.type_distribution:
        out.write("## Commit Types\\n\\n")
        out.write("| Type | Count | Percentage |\\n")
        out.write("|------|-------|------------|\\n")

        for type_name, count in message_analysis.type_distribution.most_common():
            pct = count / message_analysis.conventional_count * 100
            out.write(f"| {type_name} | {count} | {pct:.1f}% |\\n")

    return out.getvalue()
\`\`\`

### Save Report
\`\`\`python
def save_report(repo_path: str, output_path: str = None) -> str:
    """Generate and save the full report."""
    if output_path is None:
        output_path = f"report_{datetime.now().strftime('%Y%m%d')}.md"

    # Gather all data
    author_stats = get_author_stats(repo_path)
    churn = analyze_file_churn(repo_path)
    messages = analyze_messages(repo_path)

    # Generate report
    report = generate_markdown_report(
        repo_path, author_stats, churn, messages
    )

    Path(output_path).write_text(report)
    return output_path
\`\`\``
					}
				]
			}
		]
	},
	{
		name: 'Concurrent File Sync Tool',
		description: 'A simplified rsync using goroutines.',
		language: 'Go',
		color: 'amber',
		skills: 'goroutines, channels, mutexes, file system operations',
		startHint: 'Copy all files from source to dest, then add change detection',
		difficulty: 'intermediate',
		estimatedWeeks: 2,
		schedule: `## 2-Week Schedule

### Week 1: Core Sync
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | File Watching | Set up fsnotify, recursive directory watching |
| Day 2 | Hash Diffing | Implement SHA-256 file hashing, manifests |
| Day 3 | Conflict Resolution | Build resolution strategies (newest wins, prompt) |
| Day 4 | Progress Reporting | Real-time progress with transfer speeds |
| Day 5 | Integration | Combine components, end-to-end sync |
| Weekend | Buffer | Test with large directories, edge cases |

### Week 2: Concurrency
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Worker Pool | Implement goroutine pool for parallel copies |
| Day 2 | Rate Limiting | Add bandwidth throttling with token bucket |
| Day 3 | Graceful Shutdown | Signal handling, state persistence |
| Day 4 | Dry-Run Mode | Preview changes without applying |
| Day 5 | CLI Interface | Flags, help text, colored output |
| Weekend | Testing | Stress testing, race condition checking |

### Daily Commitment
- **Minimum**: 1-2 hours focused coding
- **Ideal**: 3-4 hours with breaks`,
		modules: [
			{
				name: 'Core Sync',
				description: 'Implement file synchronization logic',
				tasks: [
					{
						title: 'Watch directories for changes (fsnotify)',
						description: 'Set up file system event monitoring',
						details: `## File System Watching with fsnotify

### Installation
\`\`\`bash
go get github.com/fsnotify/fsnotify
\`\`\`

### Basic Watcher
\`\`\`go
package main

import (
    "log"
    "github.com/fsnotify/fsnotify"
)

type FileWatcher struct {
    watcher *fsnotify.Watcher
    events  chan FileEvent
    errors  chan error
}

type FileEvent struct {
    Path      string
    Operation string // create, write, remove, rename, chmod
}

func NewFileWatcher() (*FileWatcher, error) {
    w, err := fsnotify.NewWatcher()
    if err != nil {
        return nil, err
    }

    fw := &FileWatcher{
        watcher: w,
        events:  make(chan FileEvent, 100),
        errors:  make(chan error, 10),
    }

    go fw.processEvents()
    return fw, nil
}

func (fw *FileWatcher) processEvents() {
    for {
        select {
        case event, ok := <-fw.watcher.Events:
            if !ok {
                return
            }
            fw.events <- FileEvent{
                Path:      event.Name,
                Operation: event.Op.String(),
            }

        case err, ok := <-fw.watcher.Errors:
            if !ok {
                return
            }
            fw.errors <- err
        }
    }
}

func (fw *FileWatcher) AddRecursive(root string) error {
    return filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
        if err != nil {
            return err
        }
        if info.IsDir() {
            return fw.watcher.Add(path)
        }
        return nil
    })
}
\`\`\`

### Debouncing Events
\`\`\`go
// Many editors trigger multiple events for one save
func (fw *FileWatcher) DebouncedEvents(delay time.Duration) <-chan FileEvent {
    out := make(chan FileEvent)
    pending := make(map[string]*time.Timer)
    var mu sync.Mutex

    go func() {
        for event := range fw.events {
            mu.Lock()
            if timer, exists := pending[event.Path]; exists {
                timer.Stop()
            }
            pending[event.Path] = time.AfterFunc(delay, func() {
                mu.Lock()
                delete(pending, event.Path)
                mu.Unlock()
                out <- event
            })
            mu.Unlock()
        }
    }()

    return out
}
\`\`\``
					},
					{
						title: 'Diff files by hash, sync only changed files',
						description: 'Implement efficient change detection',
						details: `## Hash-Based Change Detection

### File Hashing
\`\`\`go
package sync

import (
    "crypto/sha256"
    "encoding/hex"
    "io"
    "os"
)

func HashFile(path string) (string, error) {
    f, err := os.Open(path)
    if err != nil {
        return "", err
    }
    defer f.Close()

    h := sha256.New()
    if _, err := io.Copy(h, f); err != nil {
        return "", err
    }

    return hex.EncodeToString(h.Sum(nil)), nil
}

// For large files, hash in chunks with progress
func HashFileWithProgress(path string, progress func(int64)) (string, error) {
    f, err := os.Open(path)
    if err != nil {
        return "", err
    }
    defer f.Close()

    h := sha256.New()
    buf := make([]byte, 1024*1024) // 1MB chunks
    var total int64

    for {
        n, err := f.Read(buf)
        if n > 0 {
            h.Write(buf[:n])
            total += int64(n)
            if progress != nil {
                progress(total)
            }
        }
        if err == io.EOF {
            break
        }
        if err != nil {
            return "", err
        }
    }

    return hex.EncodeToString(h.Sum(nil)), nil
}
\`\`\`

### File Manifest
\`\`\`go
type FileInfo struct {
    Path    string
    Size    int64
    ModTime time.Time
    Hash    string
}

type Manifest struct {
    Files   map[string]FileInfo
    Created time.Time
}

func BuildManifest(root string) (*Manifest, error) {
    m := &Manifest{
        Files:   make(map[string]FileInfo),
        Created: time.Now(),
    }

    err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
        if err != nil || info.IsDir() {
            return err
        }

        relPath, _ := filepath.Rel(root, path)
        hash, err := HashFile(path)
        if err != nil {
            return err
        }

        m.Files[relPath] = FileInfo{
            Path:    relPath,
            Size:    info.Size(),
            ModTime: info.ModTime(),
            Hash:    hash,
        }
        return nil
    })

    return m, err
}

func (m *Manifest) Diff(other *Manifest) (added, modified, deleted []string) {
    for path, info := range m.Files {
        otherInfo, exists := other.Files[path]
        if !exists {
            added = append(added, path)
        } else if info.Hash != otherInfo.Hash {
            modified = append(modified, path)
        }
    }

    for path := range other.Files {
        if _, exists := m.Files[path]; !exists {
            deleted = append(deleted, path)
        }
    }

    return
}
\`\`\``
					},
					{
						title: 'Handle conflicts (newer wins, or prompt)',
						description: 'Build conflict resolution strategies',
						details: `## Conflict Resolution

### Conflict Types
\`\`\`go
type ConflictType int

const (
    BothModified ConflictType = iota
    DeletedLocally
    DeletedRemotely
)

type Conflict struct {
    Path       string
    Type       ConflictType
    LocalInfo  *FileInfo
    RemoteInfo *FileInfo
}

type Resolution int

const (
    KeepLocal Resolution = iota
    KeepRemote
    KeepBoth
    Skip
)
\`\`\`

### Resolution Strategies
\`\`\`go
type ConflictResolver interface {
    Resolve(conflict Conflict) Resolution
}

// Newest file wins
type NewestWinsResolver struct{}

func (r NewestWinsResolver) Resolve(c Conflict) Resolution {
    if c.LocalInfo == nil {
        return KeepRemote
    }
    if c.RemoteInfo == nil {
        return KeepLocal
    }
    if c.LocalInfo.ModTime.After(c.RemoteInfo.ModTime) {
        return KeepLocal
    }
    return KeepRemote
}

// Interactive prompt
type InteractiveResolver struct {
    reader *bufio.Reader
}

func (r InteractiveResolver) Resolve(c Conflict) Resolution {
    fmt.Printf("\\nConflict: %s\\n", c.Path)
    fmt.Printf("  Local:  %s (%d bytes)\\n",
        c.LocalInfo.ModTime.Format(time.RFC3339),
        c.LocalInfo.Size)
    fmt.Printf("  Remote: %s (%d bytes)\\n",
        c.RemoteInfo.ModTime.Format(time.RFC3339),
        c.RemoteInfo.Size)
    fmt.Print("Resolution [l]ocal/[r]emote/[b]oth/[s]kip: ")

    input, _ := r.reader.ReadString('\\n')
    switch strings.TrimSpace(strings.ToLower(input)) {
    case "l", "local":
        return KeepLocal
    case "r", "remote":
        return KeepRemote
    case "b", "both":
        return KeepBoth
    default:
        return Skip
    }
}
\`\`\`

### Keep Both (Rename)
\`\`\`go
func keepBoth(basePath string, localInfo, remoteInfo *FileInfo) error {
    ext := filepath.Ext(basePath)
    base := strings.TrimSuffix(basePath, ext)

    // Rename local to include timestamp
    localNew := fmt.Sprintf("%s_local_%s%s",
        base,
        localInfo.ModTime.Format("20060102_150405"),
        ext)

    remoteNew := fmt.Sprintf("%s_remote_%s%s",
        base,
        remoteInfo.ModTime.Format("20060102_150405"),
        ext)

    if err := os.Rename(basePath, localNew); err != nil {
        return err
    }

    // Copy remote version
    return copyFile(remoteInfo.Path, remoteNew)
}
\`\`\``
					},
					{
						title: 'Progress reporting with concurrent transfers',
						description: 'Show real-time sync progress',
						details: `## Progress Reporting

### Progress Tracker
\`\`\`go
type Progress struct {
    TotalFiles      int
    CompletedFiles  int
    TotalBytes      int64
    TransferredBytes int64
    CurrentFile     string
    StartTime       time.Time
    mu              sync.RWMutex
}

func NewProgress(totalFiles int, totalBytes int64) *Progress {
    return &Progress{
        TotalFiles: totalFiles,
        TotalBytes: totalBytes,
        StartTime:  time.Now(),
    }
}

func (p *Progress) Update(bytes int64) {
    p.mu.Lock()
    p.TransferredBytes += bytes
    p.mu.Unlock()
}

func (p *Progress) CompleteFile(path string) {
    p.mu.Lock()
    p.CompletedFiles++
    p.CurrentFile = ""
    p.mu.Unlock()
}

func (p *Progress) StartFile(path string) {
    p.mu.Lock()
    p.CurrentFile = path
    p.mu.Unlock()
}

func (p *Progress) Stats() (pct float64, speed float64, eta time.Duration) {
    p.mu.RLock()
    defer p.mu.RUnlock()

    if p.TotalBytes > 0 {
        pct = float64(p.TransferredBytes) / float64(p.TotalBytes) * 100
    }

    elapsed := time.Since(p.StartTime).Seconds()
    if elapsed > 0 {
        speed = float64(p.TransferredBytes) / elapsed // bytes/sec
    }

    if speed > 0 {
        remaining := p.TotalBytes - p.TransferredBytes
        eta = time.Duration(float64(remaining)/speed) * time.Second
    }

    return
}
\`\`\`

### Terminal Progress Bar
\`\`\`go
func (p *Progress) Render() string {
    pct, speed, eta := p.Stats()

    // Progress bar
    barWidth := 30
    filled := int(pct / 100 * float64(barWidth))
    bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)

    // Format speed
    var speedStr string
    switch {
    case speed >= 1024*1024*1024:
        speedStr = fmt.Sprintf("%.1f GB/s", speed/1024/1024/1024)
    case speed >= 1024*1024:
        speedStr = fmt.Sprintf("%.1f MB/s", speed/1024/1024)
    case speed >= 1024:
        speedStr = fmt.Sprintf("%.1f KB/s", speed/1024)
    default:
        speedStr = fmt.Sprintf("%.0f B/s", speed)
    }

    p.mu.RLock()
    current := p.CurrentFile
    files := fmt.Sprintf("%d/%d", p.CompletedFiles, p.TotalFiles)
    p.mu.RUnlock()

    return fmt.Sprintf(
        "\\r[%s] %.1f%% %s %s ETA: %s  %s",
        bar, pct, files, speedStr,
        eta.Round(time.Second),
        truncatePath(current, 30),
    )
}

// Live update in terminal
func (p *Progress) StartDisplay(ctx context.Context) {
    ticker := time.NewTicker(100 * time.Millisecond)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            fmt.Println() // New line after progress
            return
        case <-ticker.C:
            fmt.Print(p.Render())
        }
    }
}
\`\`\``
					}
				]
			},
			{
				name: 'Concurrency',
				description: 'Add parallel processing capabilities',
				tasks: [
					{
						title: 'Implement worker pool for parallel copies',
						description: 'Use goroutines for concurrent file transfers',
						details: `## Worker Pool Pattern

### Basic Worker Pool
\`\`\`go
type CopyJob struct {
    Source string
    Dest   string
    Size   int64
}

type CopyResult struct {
    Job   CopyJob
    Error error
    Duration time.Duration
}

func WorkerPool(
    ctx context.Context,
    jobs <-chan CopyJob,
    results chan<- CopyResult,
    workers int,
    progress *Progress,
) {
    var wg sync.WaitGroup

    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()

            for {
                select {
                case <-ctx.Done():
                    return
                case job, ok := <-jobs:
                    if !ok {
                        return
                    }
                    result := processJob(ctx, job, progress)
                    results <- result
                }
            }
        }(i)
    }

    wg.Wait()
    close(results)
}

func processJob(ctx context.Context, job CopyJob, progress *Progress) CopyResult {
    start := time.Now()
    progress.StartFile(job.Source)

    err := copyFileWithProgress(ctx, job.Source, job.Dest, func(n int64) {
        progress.Update(n)
    })

    progress.CompleteFile(job.Source)

    return CopyResult{
        Job:      job,
        Error:    err,
        Duration: time.Since(start),
    }
}
\`\`\`

### Copy with Progress Callback
\`\`\`go
func copyFileWithProgress(
    ctx context.Context,
    src, dst string,
    onProgress func(int64),
) error {
    source, err := os.Open(src)
    if err != nil {
        return err
    }
    defer source.Close()

    dest, err := os.Create(dst)
    if err != nil {
        return err
    }
    defer dest.Close()

    buf := make([]byte, 32*1024) // 32KB buffer

    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
        }

        n, err := source.Read(buf)
        if n > 0 {
            if _, writeErr := dest.Write(buf[:n]); writeErr != nil {
                return writeErr
            }
            if onProgress != nil {
                onProgress(int64(n))
            }
        }
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }
    }

    // Preserve permissions
    info, _ := source.Stat()
    return dest.Chmod(info.Mode())
}
\`\`\`

### Orchestrator
\`\`\`go
func SyncFiles(ctx context.Context, changes []CopyJob, workers int) error {
    jobs := make(chan CopyJob, len(changes))
    results := make(chan CopyResult, len(changes))

    // Calculate total size
    var totalBytes int64
    for _, job := range changes {
        totalBytes += job.Size
    }
    progress := NewProgress(len(changes), totalBytes)

    // Start progress display
    displayCtx, cancelDisplay := context.WithCancel(ctx)
    go progress.StartDisplay(displayCtx)

    // Start workers
    go WorkerPool(ctx, jobs, results, workers, progress)

    // Send jobs
    for _, job := range changes {
        jobs <- job
    }
    close(jobs)

    // Collect results
    var errors []error
    for result := range results {
        if result.Error != nil {
            errors = append(errors, fmt.Errorf("%s: %w", result.Job.Source, result.Error))
        }
    }

    cancelDisplay()

    if len(errors) > 0 {
        return fmt.Errorf("sync completed with %d errors", len(errors))
    }
    return nil
}
\`\`\``
					},
					{
						title: 'Add rate limiting for bandwidth control',
						description: 'Prevent network saturation',
						details: `## Rate Limiting

### Token Bucket Rate Limiter
\`\`\`go
import "golang.org/x/time/rate"

type RateLimitedReader struct {
    reader  io.Reader
    limiter *rate.Limiter
    ctx     context.Context
}

func NewRateLimitedReader(
    ctx context.Context,
    reader io.Reader,
    bytesPerSecond int64,
) *RateLimitedReader {
    // Burst allows small spikes
    burst := int(bytesPerSecond / 10)
    if burst < 1024 {
        burst = 1024
    }

    return &RateLimitedReader{
        reader:  reader,
        limiter: rate.NewLimiter(rate.Limit(bytesPerSecond), burst),
        ctx:     ctx,
    }
}

func (r *RateLimitedReader) Read(p []byte) (int, error) {
    n, err := r.reader.Read(p)
    if n > 0 {
        if waitErr := r.limiter.WaitN(r.ctx, n); waitErr != nil {
            return n, waitErr
        }
    }
    return n, err
}
\`\`\`

### Configurable Bandwidth
\`\`\`go
type SyncConfig struct {
    Workers         int
    MaxBandwidth    int64 // bytes per second, 0 = unlimited
    ChunkSize       int
}

func copyFileRateLimited(
    ctx context.Context,
    src, dst string,
    config SyncConfig,
    onProgress func(int64),
) error {
    source, err := os.Open(src)
    if err != nil {
        return err
    }
    defer source.Close()

    var reader io.Reader = source
    if config.MaxBandwidth > 0 {
        reader = NewRateLimitedReader(ctx, source, config.MaxBandwidth)
    }

    dest, err := os.Create(dst)
    if err != nil {
        return err
    }
    defer dest.Close()

    buf := make([]byte, config.ChunkSize)
    for {
        n, readErr := reader.Read(buf)
        if n > 0 {
            if _, writeErr := dest.Write(buf[:n]); writeErr != nil {
                return writeErr
            }
            if onProgress != nil {
                onProgress(int64(n))
            }
        }
        if readErr == io.EOF {
            return nil
        }
        if readErr != nil {
            return readErr
        }
    }
}
\`\`\`

### Dynamic Rate Adjustment
\`\`\`go
type AdaptiveRateLimiter struct {
    limiter    *rate.Limiter
    baseRate   float64
    mu         sync.Mutex
}

func (a *AdaptiveRateLimiter) AdjustRate(factor float64) {
    a.mu.Lock()
    defer a.mu.Unlock()

    newRate := a.baseRate * factor
    a.limiter.SetLimit(rate.Limit(newRate))
}

// Reduce rate when errors occur
func (a *AdaptiveRateLimiter) OnError() {
    a.AdjustRate(0.5) // Cut rate in half
}

// Gradually increase when successful
func (a *AdaptiveRateLimiter) OnSuccess() {
    a.AdjustRate(1.1) // Increase by 10%
}
\`\`\``
					},
					{
						title: 'Handle graceful shutdown',
						description: 'Clean up properly on interruption',
						details: `## Graceful Shutdown

### Signal Handling
\`\`\`go
func main() {
    ctx, cancel := context.WithCancel(context.Background())

    // Handle interrupt signals
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

    go func() {
        sig := <-sigChan
        log.Printf("Received signal: %v, initiating graceful shutdown...", sig)
        cancel()

        // Force exit after timeout
        <-time.After(30 * time.Second)
        log.Fatal("Shutdown timeout exceeded, forcing exit")
    }()

    if err := run(ctx); err != nil {
        if errors.Is(err, context.Canceled) {
            log.Println("Sync cancelled by user")
        } else {
            log.Fatalf("Error: %v", err)
        }
    }
}
\`\`\`

### Cleanup Manager
\`\`\`go
type CleanupManager struct {
    tasks []func() error
    mu    sync.Mutex
}

func (c *CleanupManager) Register(cleanup func() error) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.tasks = append(c.tasks, cleanup)
}

func (c *CleanupManager) Run() error {
    c.mu.Lock()
    defer c.mu.Unlock()

    var errs []error
    // Run in reverse order (LIFO)
    for i := len(c.tasks) - 1; i >= 0; i-- {
        if err := c.tasks[i](); err != nil {
            errs = append(errs, err)
        }
    }

    if len(errs) > 0 {
        return fmt.Errorf("cleanup errors: %v", errs)
    }
    return nil
}
\`\`\`

### Sync State Persistence
\`\`\`go
type SyncState struct {
    InProgress   []CopyJob   \`json:"in_progress"\`
    Completed    []string    \`json:"completed"\`
    LastCheckpoint time.Time \`json:"last_checkpoint"\`
}

func (s *SyncState) Save(path string) error {
    data, err := json.MarshalIndent(s, "", "  ")
    if err != nil {
        return err
    }
    return os.WriteFile(path, data, 0644)
}

func (s *SyncState) Load(path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        if os.IsNotExist(err) {
            return nil
        }
        return err
    }
    return json.Unmarshal(data, s)
}

// Resume interrupted sync
func ResumableSync(ctx context.Context, changes []CopyJob, statePath string) error {
    state := &SyncState{}
    state.Load(statePath)

    completed := make(map[string]bool)
    for _, path := range state.Completed {
        completed[path] = true
    }

    // Filter out already completed
    var remaining []CopyJob
    for _, job := range changes {
        if !completed[job.Source] {
            remaining = append(remaining, job)
        }
    }

    // Checkpoint periodically
    // ... sync with state updates
}
\`\`\``
					},
					{
						title: 'Add dry-run mode',
						description: 'Preview changes without applying them',
						details: `## Dry-Run Mode

### Sync Plan
\`\`\`go
type SyncAction int

const (
    ActionCopy SyncAction = iota
    ActionDelete
    ActionUpdate
    ActionSkip
)

type PlannedAction struct {
    Action   SyncAction
    Source   string
    Dest     string
    Size     int64
    Reason   string
}

func (a PlannedAction) String() string {
    var actionStr string
    switch a.Action {
    case ActionCopy:
        actionStr = "[+]"
    case ActionDelete:
        actionStr = "[-]"
    case ActionUpdate:
        actionStr = "[~]"
    case ActionSkip:
        actionStr = "[=]"
    }
    return fmt.Sprintf("%s %s (%s)", actionStr, a.Source, humanizeBytes(a.Size))
}
\`\`\`

### Plan Builder
\`\`\`go
type SyncPlan struct {
    Actions      []PlannedAction
    TotalCopy    int64
    TotalDelete  int
    TotalUpdate  int
    TotalSkip    int
}

func BuildPlan(source, dest *Manifest) *SyncPlan {
    plan := &SyncPlan{}

    // Files to copy (new in source)
    for path, info := range source.Files {
        if _, exists := dest.Files[path]; !exists {
            plan.Actions = append(plan.Actions, PlannedAction{
                Action: ActionCopy,
                Source: path,
                Dest:   path,
                Size:   info.Size,
                Reason: "new file",
            })
            plan.TotalCopy += info.Size
        }
    }

    // Files to update (different hash)
    for path, srcInfo := range source.Files {
        if destInfo, exists := dest.Files[path]; exists {
            if srcInfo.Hash != destInfo.Hash {
                plan.Actions = append(plan.Actions, PlannedAction{
                    Action: ActionUpdate,
                    Source: path,
                    Size:   srcInfo.Size,
                    Reason: "content changed",
                })
                plan.TotalUpdate++
            } else {
                plan.Actions = append(plan.Actions, PlannedAction{
                    Action: ActionSkip,
                    Source: path,
                    Reason: "unchanged",
                })
                plan.TotalSkip++
            }
        }
    }

    // Files to delete (not in source)
    for path := range dest.Files {
        if _, exists := source.Files[path]; !exists {
            plan.Actions = append(plan.Actions, PlannedAction{
                Action: ActionDelete,
                Source: path,
                Reason: "removed from source",
            })
            plan.TotalDelete++
        }
    }

    return plan
}
\`\`\`

### Display Plan
\`\`\`go
func (p *SyncPlan) Print(verbose bool) {
    fmt.Println("Sync Plan:")
    fmt.Println(strings.Repeat("-", 50))

    if verbose {
        for _, action := range p.Actions {
            if action.Action != ActionSkip {
                fmt.Println(action.String())
            }
        }
        fmt.Println(strings.Repeat("-", 50))
    }

    fmt.Printf("  Copy:   %d files (%s)\\n",
        len(filterActions(p.Actions, ActionCopy)),
        humanizeBytes(p.TotalCopy))
    fmt.Printf("  Update: %d files\\n", p.TotalUpdate)
    fmt.Printf("  Delete: %d files\\n", p.TotalDelete)
    fmt.Printf("  Skip:   %d files (unchanged)\\n", p.TotalSkip)
}

func (p *SyncPlan) Execute(ctx context.Context, dryRun bool) error {
    if dryRun {
        fmt.Println("\\n[DRY RUN] No changes will be made")
        return nil
    }

    // Actually perform sync...
}
\`\`\``
					}
				]
			}
		]
	},
	{
		name: 'HTTP Proxy with Caching',
		description: 'A caching proxy that stores responses and logs requests.',
		language: 'Go',
		color: 'blue',
		skills: 'net/http, concurrency, memory management, HTTP protocol',
		startHint: 'Simple pass-through proxy that logs requests',
		difficulty: 'intermediate',
		estimatedWeeks: 2,
		schedule: `## 2-Week Schedule

### Week 1: Proxy Core
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Basic Proxy | Forward requests, copy headers |
| Day 2 | Response Caching | In-memory cache with LRU eviction |
| Day 3 | Cache-Control | Parse headers, calculate expiration |
| Day 4 | Request Logging | Async logger with timing metrics |
| Day 5 | Integration | Connect all pieces, basic testing |
| Weekend | Buffer | Test with real websites, fix bugs |

### Week 2: Advanced Features
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Admin API | Cache clear, stats endpoints |
| Day 2 | Metrics | Hit ratios, Prometheus integration |
| Day 3 | HTTPS Support | CONNECT method handling |
| Day 4 | Header Modification | Request/response transformation rules |
| Day 5 | Configuration | YAML config, CLI flags |
| Weekend | Documentation | README, example configurations |

### Daily Commitment
- **Minimum**: 1-2 hours focused coding
- **Ideal**: 3-4 hours with breaks`,
		modules: [
			{
				name: 'Proxy Core',
				description: 'Build the proxy server foundation',
				tasks: [
					{
						title: 'Forward requests to target servers',
						description: 'Implement basic HTTP proxying',
						details: `## HTTP Proxy Implementation

### Basic Proxy Handler
\`\`\`go
package main

import (
    "io"
    "net/http"
    "net/url"
    "time"
)

type Proxy struct {
    client *http.Client
}

func NewProxy() *Proxy {
    return &Proxy{
        client: &http.Client{
            Timeout: 30 * time.Second,
            // Don't follow redirects - pass them to client
            CheckRedirect: func(req *http.Request, via []*http.Request) error {
                return http.ErrUseLastResponse
            },
        },
    }
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    // Build target URL
    targetURL := r.URL.String()
    if r.URL.Host == "" {
        // Use X-Forwarded-Host or Host header
        targetURL = "http://" + r.Host + r.URL.Path
        if r.URL.RawQuery != "" {
            targetURL += "?" + r.URL.RawQuery
        }
    }

    // Create proxy request
    proxyReq, err := http.NewRequestWithContext(
        r.Context(),
        r.Method,
        targetURL,
        r.Body,
    )
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadGateway)
        return
    }

    // Copy headers
    copyHeaders(proxyReq.Header, r.Header)
    proxyReq.Header.Set("X-Forwarded-For", r.RemoteAddr)

    // Make request
    resp, err := p.client.Do(proxyReq)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadGateway)
        return
    }
    defer resp.Body.Close()

    // Copy response
    copyHeaders(w.Header(), resp.Header)
    w.WriteHeader(resp.StatusCode)
    io.Copy(w, resp.Body)
}

func copyHeaders(dst, src http.Header) {
    for key, values := range src {
        for _, value := range values {
            dst.Add(key, value)
        }
    }
}
\`\`\`

### Running the Proxy
\`\`\`go
func main() {
    proxy := NewProxy()
    server := &http.Server{
        Addr:    ":8080",
        Handler: proxy,
    }
    log.Printf("Proxy listening on %s", server.Addr)
    log.Fatal(server.ListenAndServe())
}
\`\`\``
					},
					{
						title: 'Cache GET responses',
						description: 'Store responses for repeated requests',
						details: `## Response Caching

### Cache Entry
\`\`\`go
type CacheEntry struct {
    StatusCode int
    Headers    http.Header
    Body       []byte
    CreatedAt  time.Time
    ExpiresAt  time.Time
    Hits       int64
}

func (e *CacheEntry) IsExpired() bool {
    return time.Now().After(e.ExpiresAt)
}

type Cache struct {
    entries map[string]*CacheEntry
    mu      sync.RWMutex
    maxSize int64
    size    int64
}

func NewCache(maxSizeMB int) *Cache {
    return &Cache{
        entries: make(map[string]*CacheEntry),
        maxSize: int64(maxSizeMB) * 1024 * 1024,
    }
}

func (c *Cache) Get(key string) (*CacheEntry, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()

    entry, exists := c.entries[key]
    if !exists || entry.IsExpired() {
        return nil, false
    }

    atomic.AddInt64(&entry.Hits, 1)
    return entry, true
}

func (c *Cache) Set(key string, entry *CacheEntry) {
    c.mu.Lock()
    defer c.mu.Unlock()

    // Evict if needed
    entrySize := int64(len(entry.Body))
    for c.size+entrySize > c.maxSize && len(c.entries) > 0 {
        c.evictOldest()
    }

    c.entries[key] = entry
    c.size += entrySize
}

func (c *Cache) evictOldest() {
    var oldestKey string
    var oldestTime time.Time

    for key, entry := range c.entries {
        if oldestKey == "" || entry.CreatedAt.Before(oldestTime) {
            oldestKey = key
            oldestTime = entry.CreatedAt
        }
    }

    if oldestKey != "" {
        c.size -= int64(len(c.entries[oldestKey].Body))
        delete(c.entries, oldestKey)
    }
}
\`\`\`

### Cache Key Generation
\`\`\`go
func cacheKey(r *http.Request) string {
    // Include method, host, path, and query
    return fmt.Sprintf("%s:%s%s?%s",
        r.Method,
        r.Host,
        r.URL.Path,
        r.URL.RawQuery,
    )
}

// Only cache GET requests with cacheable responses
func isCacheable(r *http.Request, resp *http.Response) bool {
    if r.Method != http.MethodGet {
        return false
    }

    // Check status code
    switch resp.StatusCode {
    case 200, 203, 204, 206, 300, 301, 404, 405, 410, 414, 501:
        // Cacheable status codes
    default:
        return false
    }

    // Check Cache-Control
    cc := resp.Header.Get("Cache-Control")
    if strings.Contains(cc, "no-store") || strings.Contains(cc, "private") {
        return false
    }

    return true
}
\`\`\``
					},
					{
						title: 'Respect Cache-Control headers and TTLs',
						description: 'Implement proper cache invalidation',
						details: `## Cache-Control Parsing

### Parse Cache-Control Header
\`\`\`go
type CacheControl struct {
    NoCache     bool
    NoStore     bool
    Private     bool
    Public      bool
    MaxAge      time.Duration
    SMaxAge     time.Duration
    MustRevalidate bool
}

func ParseCacheControl(header string) CacheControl {
    cc := CacheControl{}

    for _, directive := range strings.Split(header, ",") {
        directive = strings.TrimSpace(strings.ToLower(directive))

        switch {
        case directive == "no-cache":
            cc.NoCache = true
        case directive == "no-store":
            cc.NoStore = true
        case directive == "private":
            cc.Private = true
        case directive == "public":
            cc.Public = true
        case directive == "must-revalidate":
            cc.MustRevalidate = true
        case strings.HasPrefix(directive, "max-age="):
            if seconds, err := strconv.Atoi(directive[8:]); err == nil {
                cc.MaxAge = time.Duration(seconds) * time.Second
            }
        case strings.HasPrefix(directive, "s-maxage="):
            if seconds, err := strconv.Atoi(directive[9:]); err == nil {
                cc.SMaxAge = time.Duration(seconds) * time.Second
            }
        }
    }

    return cc
}
\`\`\`

### Calculate Expiration
\`\`\`go
func calculateExpiration(resp *http.Response) time.Time {
    cc := ParseCacheControl(resp.Header.Get("Cache-Control"))

    // s-maxage takes precedence for shared caches
    if cc.SMaxAge > 0 {
        return time.Now().Add(cc.SMaxAge)
    }

    // max-age
    if cc.MaxAge > 0 {
        return time.Now().Add(cc.MaxAge)
    }

    // Expires header
    if expires := resp.Header.Get("Expires"); expires != "" {
        if t, err := http.ParseTime(expires); err == nil {
            return t
        }
    }

    // Default: use heuristic based on Last-Modified
    if lastMod := resp.Header.Get("Last-Modified"); lastMod != "" {
        if t, err := http.ParseTime(lastMod); err == nil {
            age := time.Since(t)
            // Cache for 10% of age, max 1 day
            ttl := age / 10
            if ttl > 24*time.Hour {
                ttl = 24 * time.Hour
            }
            return time.Now().Add(ttl)
        }
    }

    // Default: 5 minutes
    return time.Now().Add(5 * time.Minute)
}
\`\`\`

### Validation with ETags
\`\`\`go
func (p *Proxy) validateCache(
    ctx context.Context,
    entry *CacheEntry,
    originalReq *http.Request,
) (bool, *http.Response) {
    // Create conditional request
    req, _ := http.NewRequestWithContext(ctx, "GET", originalReq.URL.String(), nil)

    if etag := entry.Headers.Get("ETag"); etag != "" {
        req.Header.Set("If-None-Match", etag)
    }
    if lastMod := entry.Headers.Get("Last-Modified"); lastMod != "" {
        req.Header.Set("If-Modified-Since", lastMod)
    }

    resp, err := p.client.Do(req)
    if err != nil {
        return false, nil
    }

    if resp.StatusCode == http.StatusNotModified {
        resp.Body.Close()
        return true, nil // Cache is still valid
    }

    return false, resp // Return new response
}
\`\`\``
					},
					{
						title: 'Log all requests with timing',
						description: 'Track request duration and status',
						details: `## Request Logging

### Log Entry Structure
\`\`\`go
type LogEntry struct {
    Timestamp    time.Time     \`json:"timestamp"\`
    Method       string        \`json:"method"\`
    URL          string        \`json:"url"\`
    Status       int           \`json:"status"\`
    Duration     time.Duration \`json:"duration_ms"\`
    BytesSent    int64         \`json:"bytes_sent"\`
    CacheStatus  string        \`json:"cache"\` // HIT, MISS, BYPASS
    ClientIP     string        \`json:"client_ip"\`
    UserAgent    string        \`json:"user_agent"\`
}

func (e LogEntry) String() string {
    return fmt.Sprintf("%s %s %s %d %dms %s",
        e.Timestamp.Format("15:04:05"),
        e.Method,
        e.URL,
        e.Status,
        e.Duration.Milliseconds(),
        e.CacheStatus,
    )
}
\`\`\`

### Logging Middleware
\`\`\`go
type ResponseRecorder struct {
    http.ResponseWriter
    statusCode int
    bytes      int64
}

func (r *ResponseRecorder) WriteHeader(code int) {
    r.statusCode = code
    r.ResponseWriter.WriteHeader(code)
}

func (r *ResponseRecorder) Write(b []byte) (int, error) {
    n, err := r.ResponseWriter.Write(b)
    r.bytes += int64(n)
    return n, err
}

func LoggingMiddleware(next http.Handler, logger *Logger) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()

        recorder := &ResponseRecorder{
            ResponseWriter: w,
            statusCode:     200,
        }

        next.ServeHTTP(recorder, r)

        entry := LogEntry{
            Timestamp:   start,
            Method:      r.Method,
            URL:         r.URL.String(),
            Status:      recorder.statusCode,
            Duration:    time.Since(start),
            BytesSent:   recorder.bytes,
            CacheStatus: w.Header().Get("X-Cache"),
            ClientIP:    getClientIP(r),
            UserAgent:   r.UserAgent(),
        }

        logger.Log(entry)
    })
}

func getClientIP(r *http.Request) string {
    if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
        return strings.Split(xff, ",")[0]
    }
    return strings.Split(r.RemoteAddr, ":")[0]
}
\`\`\`

### Async Logger
\`\`\`go
type Logger struct {
    entries chan LogEntry
    file    *os.File
    encoder *json.Encoder
}

func NewLogger(path string) (*Logger, error) {
    f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
    if err != nil {
        return nil, err
    }

    l := &Logger{
        entries: make(chan LogEntry, 1000),
        file:    f,
        encoder: json.NewEncoder(f),
    }

    go l.run()
    return l, nil
}

func (l *Logger) Log(entry LogEntry) {
    select {
    case l.entries <- entry:
    default:
        // Buffer full, drop entry
        log.Println("Log buffer full, dropping entry")
    }
}

func (l *Logger) run() {
    for entry := range l.entries {
        l.encoder.Encode(entry)
    }
}
\`\`\``
					}
				]
			},
			{
				name: 'Advanced Features',
				description: 'Add production-ready capabilities',
				tasks: [
					{
						title: 'Admin endpoint to clear cache',
						description: 'Build cache management API',
						details: `## Admin API

### Admin Handler
\`\`\`go
type AdminHandler struct {
    cache  *Cache
    proxy  *Proxy
    apiKey string
}

func NewAdminHandler(cache *Cache, proxy *Proxy, apiKey string) *AdminHandler {
    return &AdminHandler{cache: cache, proxy: proxy, apiKey: apiKey}
}

func (h *AdminHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    // Auth check
    if h.apiKey != "" && r.Header.Get("X-API-Key") != h.apiKey {
        http.Error(w, "Unauthorized", http.StatusUnauthorized)
        return
    }

    switch {
    case r.Method == "DELETE" && r.URL.Path == "/admin/cache":
        h.clearCache(w, r)
    case r.Method == "DELETE" && strings.HasPrefix(r.URL.Path, "/admin/cache/"):
        h.deleteEntry(w, r)
    case r.Method == "GET" && r.URL.Path == "/admin/cache":
        h.listCache(w, r)
    default:
        http.NotFound(w, r)
    }
}

func (h *AdminHandler) clearCache(w http.ResponseWriter, r *http.Request) {
    h.cache.mu.Lock()
    count := len(h.cache.entries)
    h.cache.entries = make(map[string]*CacheEntry)
    h.cache.size = 0
    h.cache.mu.Unlock()

    json.NewEncoder(w).Encode(map[string]interface{}{
        "cleared": count,
        "message": "Cache cleared successfully",
    })
}

func (h *AdminHandler) deleteEntry(w http.ResponseWriter, r *http.Request) {
    key := strings.TrimPrefix(r.URL.Path, "/admin/cache/")
    key, _ = url.PathUnescape(key)

    h.cache.mu.Lock()
    if entry, exists := h.cache.entries[key]; exists {
        h.cache.size -= int64(len(entry.Body))
        delete(h.cache.entries, key)
        h.cache.mu.Unlock()

        json.NewEncoder(w).Encode(map[string]string{
            "deleted": key,
        })
    } else {
        h.cache.mu.Unlock()
        http.NotFound(w, r)
    }
}

func (h *AdminHandler) listCache(w http.ResponseWriter, r *http.Request) {
    h.cache.mu.RLock()
    defer h.cache.mu.RUnlock()

    entries := make([]map[string]interface{}, 0, len(h.cache.entries))
    for key, entry := range h.cache.entries {
        entries = append(entries, map[string]interface{}{
            "key":       key,
            "size":      len(entry.Body),
            "hits":      entry.Hits,
            "created":   entry.CreatedAt,
            "expires":   entry.ExpiresAt,
            "expired":   entry.IsExpired(),
        })
    }

    json.NewEncoder(w).Encode(entries)
}
\`\`\``
					},
					{
						title: 'Cache statistics endpoint',
						description: 'Expose hit/miss ratios and storage usage',
						details: `## Cache Statistics

### Stats Tracking
\`\`\`go
type CacheStats struct {
    Hits        int64
    Misses      int64
    Bypasses    int64
    Evictions   int64
    TotalSize   int64
    EntryCount  int
    mu          sync.RWMutex
}

func (s *CacheStats) RecordHit() {
    atomic.AddInt64(&s.Hits, 1)
}

func (s *CacheStats) RecordMiss() {
    atomic.AddInt64(&s.Misses, 1)
}

func (s *CacheStats) RecordBypass() {
    atomic.AddInt64(&s.Bypasses, 1)
}

func (s *CacheStats) HitRatio() float64 {
    hits := atomic.LoadInt64(&s.Hits)
    misses := atomic.LoadInt64(&s.Misses)
    total := hits + misses
    if total == 0 {
        return 0
    }
    return float64(hits) / float64(total)
}
\`\`\`

### Stats Endpoint
\`\`\`go
func (h *AdminHandler) stats(w http.ResponseWriter, r *http.Request) {
    h.cache.mu.RLock()
    entryCount := len(h.cache.entries)
    totalSize := h.cache.size
    maxSize := h.cache.maxSize

    // Calculate per-host stats
    hostStats := make(map[string]int)
    for key := range h.cache.entries {
        if u, err := url.Parse(key); err == nil {
            hostStats[u.Host]++
        }
    }
    h.cache.mu.RUnlock()

    stats := h.cache.stats

    response := map[string]interface{}{
        "hits":       atomic.LoadInt64(&stats.Hits),
        "misses":     atomic.LoadInt64(&stats.Misses),
        "bypasses":   atomic.LoadInt64(&stats.Bypasses),
        "evictions":  atomic.LoadInt64(&stats.Evictions),
        "hit_ratio":  stats.HitRatio(),
        "entries":    entryCount,
        "size_bytes": totalSize,
        "max_bytes":  maxSize,
        "usage_pct":  float64(totalSize) / float64(maxSize) * 100,
        "by_host":    hostStats,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}
\`\`\`

### Prometheus Metrics
\`\`\`go
import "github.com/prometheus/client_golang/prometheus"

var (
    cacheHits = prometheus.NewCounter(prometheus.CounterOpts{
        Name: "proxy_cache_hits_total",
        Help: "Total cache hits",
    })
    cacheMisses = prometheus.NewCounter(prometheus.CounterOpts{
        Name: "proxy_cache_misses_total",
        Help: "Total cache misses",
    })
    cacheSize = prometheus.NewGauge(prometheus.GaugeOpts{
        Name: "proxy_cache_size_bytes",
        Help: "Current cache size in bytes",
    })
    requestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "proxy_request_duration_seconds",
            Help:    "Request duration in seconds",
            Buckets: []float64{.001, .005, .01, .05, .1, .5, 1, 5},
        },
        []string{"method", "status", "cache"},
    )
)

func init() {
    prometheus.MustRegister(cacheHits, cacheMisses, cacheSize, requestDuration)
}
\`\`\``
					},
					{
						title: 'Support HTTPS proxying',
						description: 'Handle TLS connections properly',
						details: `## HTTPS Proxy Support

### CONNECT Method Handling
\`\`\`go
func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    if r.Method == http.MethodConnect {
        p.handleConnect(w, r)
        return
    }
    // ... regular proxy handling
}

func (p *Proxy) handleConnect(w http.ResponseWriter, r *http.Request) {
    // Connect to destination
    destConn, err := net.DialTimeout("tcp", r.Host, 10*time.Second)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadGateway)
        return
    }

    // Hijack the connection
    hijacker, ok := w.(http.Hijacker)
    if !ok {
        http.Error(w, "Hijacking not supported", http.StatusInternalServerError)
        destConn.Close()
        return
    }

    clientConn, _, err := hijacker.Hijack()
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        destConn.Close()
        return
    }

    // Send 200 Connection Established
    clientConn.Write([]byte("HTTP/1.1 200 Connection Established\\r\\n\\r\\n"))

    // Tunnel traffic bidirectionally
    go func() {
        io.Copy(destConn, clientConn)
        destConn.Close()
    }()
    go func() {
        io.Copy(clientConn, destConn)
        clientConn.Close()
    }()
}
\`\`\`

### TLS Termination (MITM for Caching)
\`\`\`go
// WARNING: Only for debugging/development
// Requires installing proxy's CA cert

func (p *Proxy) handleConnectMITM(w http.ResponseWriter, r *http.Request) {
    // Generate certificate for the target host
    cert, err := p.generateCert(r.Host)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    hijacker, _ := w.(http.Hijacker)
    clientConn, _, _ := hijacker.Hijack()

    clientConn.Write([]byte("HTTP/1.1 200 Connection Established\\r\\n\\r\\n"))

    // Wrap with TLS
    tlsConfig := &tls.Config{Certificates: []tls.Certificate{cert}}
    tlsConn := tls.Server(clientConn, tlsConfig)

    // Handle as regular HTTP
    go func() {
        defer tlsConn.Close()
        reader := bufio.NewReader(tlsConn)
        for {
            req, err := http.ReadRequest(reader)
            if err != nil {
                return
            }
            req.URL.Scheme = "https"
            req.URL.Host = r.Host
            p.handleRequest(tlsConn, req)
        }
    }()
}
\`\`\``
					},
					{
						title: 'Add request/response header modification',
						description: 'Allow header injection and removal',
						details: `## Header Modification

### Rule Configuration
\`\`\`go
type HeaderRule struct {
    Pattern   string // URL pattern (glob or regex)
    Phase     string // "request" or "response"
    Action    string // "set", "add", "remove"
    Header    string
    Value     string
}

type HeaderModifier struct {
    rules []HeaderRule
}

func NewHeaderModifier(rules []HeaderRule) *HeaderModifier {
    return &HeaderModifier{rules: rules}
}

func (m *HeaderModifier) ModifyRequest(r *http.Request) {
    for _, rule := range m.rules {
        if rule.Phase != "request" {
            continue
        }
        if !matchPattern(rule.Pattern, r.URL.String()) {
            continue
        }

        switch rule.Action {
        case "set":
            r.Header.Set(rule.Header, rule.Value)
        case "add":
            r.Header.Add(rule.Header, rule.Value)
        case "remove":
            r.Header.Del(rule.Header)
        }
    }
}

func (m *HeaderModifier) ModifyResponse(resp *http.Response, reqURL string) {
    for _, rule := range m.rules {
        if rule.Phase != "response" {
            continue
        }
        if !matchPattern(rule.Pattern, reqURL) {
            continue
        }

        switch rule.Action {
        case "set":
            resp.Header.Set(rule.Header, rule.Value)
        case "add":
            resp.Header.Add(rule.Header, rule.Value)
        case "remove":
            resp.Header.Del(rule.Header)
        }
    }
}
\`\`\`

### Configuration File
\`\`\`yaml
# headers.yaml
rules:
  - pattern: "*"
    phase: request
    action: set
    header: X-Proxy-Version
    value: "1.0"

  - pattern: "*.example.com/*"
    phase: request
    action: set
    header: Authorization
    value: "Bearer secret-token"

  - pattern: "*"
    phase: response
    action: remove
    header: X-Powered-By

  - pattern: "*"
    phase: response
    action: set
    header: X-Cache-Status
    value: "{{.CacheStatus}}"
\`\`\`

### Template Values
\`\`\`go
func (m *HeaderModifier) expandTemplate(value string, ctx map[string]string) string {
    result := value
    for key, val := range ctx {
        result = strings.ReplaceAll(result, "{{."+key+"}}", val)
    }
    return result
}

// Usage in proxy
ctx := map[string]string{
    "CacheStatus": "HIT",
    "ResponseTime": "45ms",
    "ClientIP": r.RemoteAddr,
}
value := m.expandTemplate(rule.Value, ctx)
\`\`\``
					}
				]
			}
		]
	},
	{
		name: 'Simple Message Queue',
		description: 'In-memory pub/sub queue with persistence.',
		language: 'Go',
		color: 'purple',
		skills: 'data structures, serialization, concurrent access patterns',
		startHint: 'In-memory queue with a single topic, add persistence later',
		difficulty: 'advanced',
		estimatedWeeks: 2,
		schedule: `## 2-Week Schedule

### Week 1: Core Queue
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Data Structures | Topic, Message, Subscriber types |
| Day 2 | Pub/Sub | Fan-out to multiple subscribers |
| Day 3 | Acknowledgments | At-least-once delivery with retries |
| Day 4 | REST API | Publish, subscribe, poll endpoints |
| Day 5 | Consumer Groups | Load balancing across consumers |
| Weekend | Buffer | Testing, race condition fixes |

### Week 2: Persistence
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Write-Ahead Log | Append-only log with checksums |
| Day 2 | Snapshots | Periodic state snapshots |
| Day 3 | Recovery | Rebuild state on startup |
| Day 4 | Message TTL | Automatic expiration, cleanup |
| Day 5 | Log Compaction | Reclaim disk space |
| Weekend | Stress Testing | High-throughput testing |

### Daily Commitment
- **Minimum**: 2-3 hours focused coding
- **Ideal**: 4-5 hours with breaks (advanced project)`,
		modules: [
			{
				name: 'Core Queue',
				description: 'Build the messaging foundation',
				tasks: [
					{
						title: 'Topics with multiple subscribers',
						description: 'Implement pub/sub pattern with fan-out',
						details: `## Pub/Sub Implementation

### Core Data Structures
\`\`\`go
type Message struct {
    ID        string    \`json:"id"\`
    Topic     string    \`json:"topic"\`
    Payload   []byte    \`json:"payload"\`
    Timestamp time.Time \`json:"timestamp"\`
    Metadata  map[string]string \`json:"metadata,omitempty"\`
}

type Topic struct {
    Name        string
    subscribers map[string]*Subscriber
    messages    chan *Message
    mu          sync.RWMutex
}

type Subscriber struct {
    ID       string
    Channel  chan *Message
    Filter   func(*Message) bool
    done     chan struct{}
}

type Broker struct {
    topics map[string]*Topic
    mu     sync.RWMutex
}
\`\`\`

### Broker Implementation
\`\`\`go
func NewBroker() *Broker {
    return &Broker{
        topics: make(map[string]*Topic),
    }
}

func (b *Broker) GetOrCreateTopic(name string) *Topic {
    b.mu.Lock()
    defer b.mu.Unlock()

    if topic, exists := b.topics[name]; exists {
        return topic
    }

    topic := &Topic{
        Name:        name,
        subscribers: make(map[string]*Subscriber),
        messages:    make(chan *Message, 1000),
    }

    go topic.fanOut()
    b.topics[name] = topic
    return topic
}

func (t *Topic) fanOut() {
    for msg := range t.messages {
        t.mu.RLock()
        for _, sub := range t.subscribers {
            if sub.Filter == nil || sub.Filter(msg) {
                select {
                case sub.Channel <- msg:
                default:
                    // Subscriber buffer full, skip
                }
            }
        }
        t.mu.RUnlock()
    }
}

func (t *Topic) Subscribe(id string, bufferSize int) *Subscriber {
    t.mu.Lock()
    defer t.mu.Unlock()

    sub := &Subscriber{
        ID:      id,
        Channel: make(chan *Message, bufferSize),
        done:    make(chan struct{}),
    }
    t.subscribers[id] = sub
    return sub
}

func (t *Topic) Publish(msg *Message) {
    msg.Topic = t.Name
    if msg.ID == "" {
        msg.ID = uuid.New().String()
    }
    msg.Timestamp = time.Now()
    t.messages <- msg
}
\`\`\``
					},
					{
						title: 'Message acknowledgment and retry',
						description: 'Handle delivery guarantees',
						details: `## At-Least-Once Delivery

### Pending Message Tracking
\`\`\`go
type PendingMessage struct {
    Message     *Message
    Subscriber  string
    DeliveredAt time.Time
    Attempts    int
}

type AckManager struct {
    pending   map[string]*PendingMessage // messageID -> pending
    timeout   time.Duration
    maxRetries int
    mu        sync.Mutex
}

func NewAckManager(timeout time.Duration, maxRetries int) *AckManager {
    am := &AckManager{
        pending:    make(map[string]*PendingMessage),
        timeout:    timeout,
        maxRetries: maxRetries,
    }
    go am.checkTimeouts()
    return am
}

func (am *AckManager) Track(msg *Message, subscriberID string) {
    am.mu.Lock()
    defer am.mu.Unlock()

    am.pending[msg.ID] = &PendingMessage{
        Message:     msg,
        Subscriber:  subscriberID,
        DeliveredAt: time.Now(),
        Attempts:    1,
    }
}

func (am *AckManager) Ack(messageID string) bool {
    am.mu.Lock()
    defer am.mu.Unlock()

    if _, exists := am.pending[messageID]; exists {
        delete(am.pending, messageID)
        return true
    }
    return false
}

func (am *AckManager) Nack(messageID string) *Message {
    am.mu.Lock()
    defer am.mu.Unlock()

    if pending, exists := am.pending[messageID]; exists {
        delete(am.pending, messageID)
        return pending.Message
    }
    return nil
}
\`\`\`

### Timeout and Retry
\`\`\`go
func (am *AckManager) checkTimeouts() {
    ticker := time.NewTicker(am.timeout / 2)
    for range ticker.C {
        am.mu.Lock()
        now := time.Now()

        for id, pending := range am.pending {
            if now.Sub(pending.DeliveredAt) > am.timeout {
                if pending.Attempts >= am.maxRetries {
                    // Move to dead letter queue
                    am.deadLetter(pending.Message)
                    delete(am.pending, id)
                } else {
                    // Redeliver
                    pending.Attempts++
                    pending.DeliveredAt = now
                    am.redeliver(pending)
                }
            }
        }

        am.mu.Unlock()
    }
}

func (am *AckManager) redeliver(pending *PendingMessage) {
    // Add to retry queue with exponential backoff
    delay := time.Duration(pending.Attempts*pending.Attempts) * time.Second
    time.AfterFunc(delay, func() {
        // Re-enqueue message
    })
}
\`\`\``
					},
					{
						title: 'REST API for publish/subscribe',
						description: 'Create HTTP interface for queue operations',
						details: `## REST API Design

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | /topics | Create topic |
| GET | /topics | List topics |
| POST | /topics/:name/publish | Publish message |
| POST | /topics/:name/subscribe | Create subscription |
| GET | /subscriptions/:id/poll | Poll for messages |
| POST | /subscriptions/:id/ack | Acknowledge message |

### Handler Implementation
\`\`\`go
type APIHandler struct {
    broker *Broker
    router *http.ServeMux
}

func NewAPIHandler(broker *Broker) *APIHandler {
    h := &APIHandler{broker: broker, router: http.NewServeMux()}
    h.setupRoutes()
    return h
}

func (h *APIHandler) setupRoutes() {
    h.router.HandleFunc("POST /topics", h.createTopic)
    h.router.HandleFunc("GET /topics", h.listTopics)
    h.router.HandleFunc("POST /topics/{name}/publish", h.publish)
    h.router.HandleFunc("POST /topics/{name}/subscribe", h.subscribe)
    h.router.HandleFunc("GET /subscriptions/{id}/poll", h.poll)
    h.router.HandleFunc("POST /subscriptions/{id}/ack", h.ack)
}

func (h *APIHandler) publish(w http.ResponseWriter, r *http.Request) {
    topicName := r.PathValue("name")
    topic := h.broker.GetOrCreateTopic(topicName)

    var body struct {
        Payload  json.RawMessage   \`json:"payload"\`
        Metadata map[string]string \`json:"metadata"\`
    }

    if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    msg := &Message{
        Payload:  body.Payload,
        Metadata: body.Metadata,
    }
    topic.Publish(msg)

    json.NewEncoder(w).Encode(map[string]string{
        "id":      msg.ID,
        "status":  "published",
    })
}

func (h *APIHandler) poll(w http.ResponseWriter, r *http.Request) {
    subID := r.PathValue("id")
    timeout := 30 * time.Second

    if t := r.URL.Query().Get("timeout"); t != "" {
        if d, err := time.ParseDuration(t); err == nil {
            timeout = d
        }
    }

    sub := h.broker.GetSubscription(subID)
    if sub == nil {
        http.NotFound(w, r)
        return
    }

    select {
    case msg := <-sub.Channel:
        json.NewEncoder(w).Encode(msg)
    case <-time.After(timeout):
        w.WriteHeader(http.StatusNoContent)
    case <-r.Context().Done():
        return
    }
}
\`\`\``
					},
					{
						title: 'Consumer groups',
						description: 'Support load balancing across consumers',
						details: `## Consumer Groups

### Group Management
\`\`\`go
type ConsumerGroup struct {
    ID        string
    Topic     *Topic
    consumers []*Consumer
    messages  chan *Message
    offsets   map[int]int64 // partition -> offset
    mu        sync.RWMutex
}

type Consumer struct {
    ID        string
    Group     *ConsumerGroup
    Channel   chan *Message
    active    bool
}

func NewConsumerGroup(id string, topic *Topic) *ConsumerGroup {
    cg := &ConsumerGroup{
        ID:        id,
        Topic:     topic,
        consumers: make([]*Consumer, 0),
        messages:  make(chan *Message, 1000),
        offsets:   make(map[int]int64),
    }

    go cg.distribute()
    return cg
}
\`\`\`

### Round-Robin Distribution
\`\`\`go
func (cg *ConsumerGroup) distribute() {
    var idx int

    for msg := range cg.messages {
        cg.mu.RLock()
        if len(cg.consumers) == 0 {
            cg.mu.RUnlock()
            continue
        }

        // Find next active consumer
        attempts := 0
        for attempts < len(cg.consumers) {
            consumer := cg.consumers[idx%len(cg.consumers)]
            idx++

            if consumer.active {
                select {
                case consumer.Channel <- msg:
                    cg.mu.RUnlock()
                    break
                default:
                    attempts++
                }
            } else {
                attempts++
            }
        }

        if attempts >= len(cg.consumers) {
            // All consumers busy/inactive, requeue
            go func(m *Message) {
                time.Sleep(100 * time.Millisecond)
                cg.messages <- m
            }(msg)
        }

        cg.mu.RUnlock()
    }
}

func (cg *ConsumerGroup) Join(consumerID string) *Consumer {
    cg.mu.Lock()
    defer cg.mu.Unlock()

    consumer := &Consumer{
        ID:      consumerID,
        Group:   cg,
        Channel: make(chan *Message, 100),
        active:  true,
    }

    cg.consumers = append(cg.consumers, consumer)
    cg.rebalance()

    return consumer
}

func (cg *ConsumerGroup) Leave(consumerID string) {
    cg.mu.Lock()
    defer cg.mu.Unlock()

    for i, c := range cg.consumers {
        if c.ID == consumerID {
            cg.consumers = append(cg.consumers[:i], cg.consumers[i+1:]...)
            close(c.Channel)
            break
        }
    }

    cg.rebalance()
}

func (cg *ConsumerGroup) rebalance() {
    // Redistribute partitions among consumers
    // This is simplified - real implementations use
    // cooperative rebalancing protocols
}
\`\`\``
					}
				]
			},
			{
				name: 'Persistence',
				description: 'Add durability to the queue',
				tasks: [
					{
						title: 'Persistence to disk (WAL)',
						description: 'Implement write-ahead logging',
						details: `## Write-Ahead Log

### WAL Entry Format
\`\`\`go
type WALEntry struct {
    Sequence  uint64
    Timestamp int64
    Type      byte // 1=message, 2=ack, 3=checkpoint
    TopicLen  uint16
    Topic     []byte
    DataLen   uint32
    Data      []byte
    Checksum  uint32
}

const (
    EntryTypeMessage    byte = 1
    EntryTypeAck        byte = 2
    EntryTypeCheckpoint byte = 3
)
\`\`\`

### WAL Writer
\`\`\`go
type WAL struct {
    dir       string
    file      *os.File
    sequence  uint64
    mu        sync.Mutex
    syncEvery int
    pending   int
}

func NewWAL(dir string) (*WAL, error) {
    if err := os.MkdirAll(dir, 0755); err != nil {
        return nil, err
    }

    w := &WAL{
        dir:       dir,
        syncEvery: 100, // Sync every 100 entries
    }

    return w, w.openLatest()
}

func (w *WAL) Append(entry *WALEntry) error {
    w.mu.Lock()
    defer w.mu.Unlock()

    w.sequence++
    entry.Sequence = w.sequence
    entry.Timestamp = time.Now().UnixNano()

    // Calculate checksum
    data := w.encode(entry)
    entry.Checksum = crc32.ChecksumIEEE(data)

    // Write length + data
    buf := make([]byte, 4+len(data)+4)
    binary.LittleEndian.PutUint32(buf[0:4], uint32(len(data)))
    copy(buf[4:], data)
    binary.LittleEndian.PutUint32(buf[4+len(data):], entry.Checksum)

    if _, err := w.file.Write(buf); err != nil {
        return err
    }

    w.pending++
    if w.pending >= w.syncEvery {
        w.file.Sync()
        w.pending = 0
    }

    return nil
}

func (w *WAL) encode(entry *WALEntry) []byte {
    buf := new(bytes.Buffer)
    binary.Write(buf, binary.LittleEndian, entry.Sequence)
    binary.Write(buf, binary.LittleEndian, entry.Timestamp)
    buf.WriteByte(entry.Type)
    binary.Write(buf, binary.LittleEndian, uint16(len(entry.Topic)))
    buf.Write(entry.Topic)
    binary.Write(buf, binary.LittleEndian, uint32(len(entry.Data)))
    buf.Write(entry.Data)
    return buf.Bytes()
}
\`\`\`

### WAL Reader for Recovery
\`\`\`go
func (w *WAL) Recover() ([]*WALEntry, error) {
    var entries []*WALEntry

    files, _ := filepath.Glob(filepath.Join(w.dir, "wal-*.log"))
    sort.Strings(files)

    for _, file := range files {
        fileEntries, err := w.readFile(file)
        if err != nil {
            return nil, err
        }
        entries = append(entries, fileEntries...)
    }

    return entries, nil
}
\`\`\``
					},
					{
						title: 'Snapshot and recovery',
						description: 'Create periodic snapshots for fast recovery',
						details: `## Snapshot System

### Snapshot Format
\`\`\`go
type Snapshot struct {
    Version     int       \`json:"version"\`
    Timestamp   time.Time \`json:"timestamp"\`
    WALSequence uint64    \`json:"wal_sequence"\`
    Topics      map[string]*TopicSnapshot \`json:"topics"\`
}

type TopicSnapshot struct {
    Name         string            \`json:"name"\`
    Messages     []*Message        \`json:"messages"\`
    Subscribers  []string          \`json:"subscribers"\`
    Groups       []*GroupSnapshot  \`json:"groups"\`
}

type GroupSnapshot struct {
    ID      string         \`json:"id"\`
    Offsets map[int]int64  \`json:"offsets"\`
}
\`\`\`

### Snapshot Writer
\`\`\`go
type SnapshotManager struct {
    dir       string
    broker    *Broker
    interval  time.Duration
    minWALSize int64
}

func (sm *SnapshotManager) CreateSnapshot() error {
    snapshot := &Snapshot{
        Version:   1,
        Timestamp: time.Now(),
        Topics:    make(map[string]*TopicSnapshot),
    }

    sm.broker.mu.RLock()
    for name, topic := range sm.broker.topics {
        topic.mu.RLock()
        snapshot.Topics[name] = &TopicSnapshot{
            Name:        name,
            Messages:    topic.getPendingMessages(),
            Subscribers: topic.getSubscriberIDs(),
        }
        topic.mu.RUnlock()
    }
    snapshot.WALSequence = sm.broker.wal.sequence
    sm.broker.mu.RUnlock()

    // Write to file
    filename := fmt.Sprintf("snapshot-%d.json", time.Now().Unix())
    path := filepath.Join(sm.dir, filename)

    data, err := json.Marshal(snapshot)
    if err != nil {
        return err
    }

    return os.WriteFile(path, data, 0644)
}

func (sm *SnapshotManager) LoadLatestSnapshot() (*Snapshot, error) {
    files, _ := filepath.Glob(filepath.Join(sm.dir, "snapshot-*.json"))
    if len(files) == 0 {
        return nil, nil
    }

    sort.Strings(files)
    latest := files[len(files)-1]

    data, err := os.ReadFile(latest)
    if err != nil {
        return nil, err
    }

    var snapshot Snapshot
    return &snapshot, json.Unmarshal(data, &snapshot)
}
\`\`\`

### Recovery Process
\`\`\`go
func (b *Broker) Recover() error {
    // 1. Load latest snapshot
    snapshot, err := b.snapshots.LoadLatestSnapshot()
    if err != nil {
        return err
    }

    if snapshot != nil {
        b.restoreFromSnapshot(snapshot)
    }

    // 2. Replay WAL from snapshot sequence
    startSeq := uint64(0)
    if snapshot != nil {
        startSeq = snapshot.WALSequence
    }

    entries, err := b.wal.RecoverFrom(startSeq)
    if err != nil {
        return err
    }

    for _, entry := range entries {
        b.applyWALEntry(entry)
    }

    return nil
}
\`\`\``
					},
					{
						title: 'Message TTL and cleanup',
						description: 'Automatically expire old messages',
						details: `## Message Expiration

### TTL Configuration
\`\`\`go
type TopicConfig struct {
    Name            string
    RetentionTime   time.Duration // Max age of messages
    RetentionSize   int64         // Max size in bytes
    CleanupInterval time.Duration
}

type MessageWithTTL struct {
    *Message
    ExpiresAt time.Time
}

func (m *MessageWithTTL) IsExpired() bool {
    return time.Now().After(m.ExpiresAt)
}
\`\`\`

### Cleanup Worker
\`\`\`go
type Cleaner struct {
    broker   *Broker
    interval time.Duration
    stop     chan struct{}
}

func NewCleaner(broker *Broker, interval time.Duration) *Cleaner {
    c := &Cleaner{
        broker:   broker,
        interval: interval,
        stop:     make(chan struct{}),
    }
    go c.run()
    return c
}

func (c *Cleaner) run() {
    ticker := time.NewTicker(c.interval)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            c.cleanup()
        case <-c.stop:
            return
        }
    }
}

func (c *Cleaner) cleanup() {
    c.broker.mu.RLock()
    topics := make([]*Topic, 0, len(c.broker.topics))
    for _, t := range c.broker.topics {
        topics = append(topics, t)
    }
    c.broker.mu.RUnlock()

    var totalCleaned int
    for _, topic := range topics {
        cleaned := topic.cleanExpired()
        totalCleaned += cleaned
    }

    if totalCleaned > 0 {
        log.Printf("Cleaned %d expired messages", totalCleaned)
    }
}

func (t *Topic) cleanExpired() int {
    t.mu.Lock()
    defer t.mu.Unlock()

    now := time.Now()
    cleaned := 0

    // Clean from message store
    t.messages.RemoveIf(func(msg *MessageWithTTL) bool {
        if msg.IsExpired() {
            cleaned++
            return true
        }
        return false
    })

    return cleaned
}
\`\`\`

### Size-Based Retention
\`\`\`go
func (t *Topic) enforceRetentionSize() {
    t.mu.Lock()
    defer t.mu.Unlock()

    if t.config.RetentionSize <= 0 {
        return
    }

    // Calculate current size
    var totalSize int64
    for _, msg := range t.messageStore {
        totalSize += int64(len(msg.Payload))
    }

    // Remove oldest until under limit
    for totalSize > t.config.RetentionSize && len(t.messageStore) > 0 {
        oldest := t.messageStore[0]
        totalSize -= int64(len(oldest.Payload))
        t.messageStore = t.messageStore[1:]
    }
}
\`\`\``
					},
					{
						title: 'Compaction strategy',
						description: 'Reclaim disk space efficiently',
						details: `## Log Compaction

### Segment-Based Storage
\`\`\`go
type Segment struct {
    ID        uint64
    Path      string
    StartSeq  uint64
    EndSeq    uint64
    Size      int64
    CreatedAt time.Time
    Closed    bool
}

type SegmentManager struct {
    dir          string
    segments     []*Segment
    activeSegment *Segment
    maxSize      int64 // Max segment size
    mu           sync.RWMutex
}

func (sm *SegmentManager) rollSegment() error {
    sm.mu.Lock()
    defer sm.mu.Unlock()

    if sm.activeSegment != nil {
        sm.activeSegment.Closed = true
    }

    newID := uint64(len(sm.segments))
    segment := &Segment{
        ID:        newID,
        Path:      filepath.Join(sm.dir, fmt.Sprintf("segment-%08d.log", newID)),
        CreatedAt: time.Now(),
    }

    sm.segments = append(sm.segments, segment)
    sm.activeSegment = segment
    return nil
}
\`\`\`

### Compaction Process
\`\`\`go
type Compactor struct {
    segments       *SegmentManager
    ackTracker     *AckManager
    minSegments    int
    compactAfter   time.Duration
}

func (c *Compactor) Compact() error {
    // Find segments eligible for compaction
    eligible := c.findEligibleSegments()
    if len(eligible) < 2 {
        return nil
    }

    // Create new compacted segment
    compacted, err := c.mergeSegments(eligible)
    if err != nil {
        return err
    }

    // Atomically swap segments
    c.segments.mu.Lock()
    c.segments.replaceSegments(eligible, compacted)
    c.segments.mu.Unlock()

    // Delete old segment files
    for _, seg := range eligible {
        os.Remove(seg.Path)
    }

    return nil
}

func (c *Compactor) mergeSegments(segments []*Segment) (*Segment, error) {
    newPath := filepath.Join(c.segments.dir, fmt.Sprintf("segment-%08d-compacted.log", time.Now().Unix()))

    out, err := os.Create(newPath)
    if err != nil {
        return nil, err
    }
    defer out.Close()

    var kept int
    for _, seg := range segments {
        entries := c.readSegment(seg)
        for _, entry := range entries {
            // Skip acknowledged messages
            if entry.Type == EntryTypeMessage {
                if c.ackTracker.IsAcked(entry.MessageID()) {
                    continue
                }
            }
            // Write to compacted segment
            out.Write(entry.Encode())
            kept++
        }
    }

    log.Printf("Compacted %d segments, kept %d entries", len(segments), kept)

    return &Segment{
        Path:      newPath,
        CreatedAt: time.Now(),
    }, nil
}
\`\`\``
					}
				]
			}
		]
	},
	{
		name: 'Markdown to HTML Compiler',
		description: 'Parse markdown and emit clean HTML.',
		language: 'Rust',
		color: 'rose',
		skills: 'parsing (nom or hand-rolled), string manipulation, ownership',
		startHint: 'Parse headers and paragraphs only, expand from there',
		difficulty: 'intermediate',
		estimatedWeeks: 2,
		schedule: `## 2-Week Schedule

### Week 1: Block Parsing
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Project Setup | Cargo project, basic types, tests |
| Day 2 | Headers | Parse ATX and Setext headers |
| Day 3 | Paragraphs | Handle text blocks, line breaks |
| Day 4 | Lists | Ordered and unordered, nested |
| Day 5 | Code Blocks | Fenced blocks with language hints |
| Weekend | Buffer | Edge cases, CommonMark compliance |

### Week 2: Inline Elements
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Emphasis | Bold, italic, delimiter stack |
| Day 2 | Inline Code | Backtick spans, escaping |
| Day 3 | Links & Images | URL parsing, titles |
| Day 4 | HTML Output | Escaping, formatting |
| Day 5 | CLI Tool | File input/output, options |
| Weekend | Testing | Compare output with reference |

### Daily Commitment
- **Minimum**: 1-2 hours focused coding
- **Ideal**: 3-4 hours with breaks`,
		modules: [
			{
				name: 'Basic Parsing',
				description: 'Handle fundamental markdown elements',
				tasks: [
					{
						title: 'Parse headers (h1-h6)',
						description: 'Handle # through ###### syntax',
						details: `## Header Parsing

### Header Syntax
\`\`\`markdown
# H1 Header
## H2 Header
### H3 Header
#### H4 Header
##### H5 Header
###### H6 Header
\`\`\`

### Implementation
\`\`\`rust
#[derive(Debug, Clone)]
pub enum Block {
    Heading { level: u8, content: Vec<Inline> },
    Paragraph(Vec<Inline>),
    // ... other blocks
}

pub fn parse_header(line: &str) -> Option<Block> {
    let trimmed = line.trim_start();

    // Count leading #
    let level = trimmed.chars().take_while(|&c| c == '#').count();

    if level == 0 || level > 6 {
        return None;
    }

    // Must have space after #
    let rest = &trimmed[level..];
    if !rest.starts_with(' ') && !rest.is_empty() {
        return None;
    }

    let content = rest.trim();
    // Remove optional trailing #
    let content = content.trim_end_matches(|c| c == '#' || c == ' ');

    Some(Block::Heading {
        level: level as u8,
        content: parse_inline(content),
    })
}
\`\`\`

### Alternative: ATX and Setext Headers
\`\`\`rust
// Setext-style headers
// Header 1
// ========
//
// Header 2
// --------

pub fn parse_setext_header(line: &str, next_line: Option<&str>) -> Option<Block> {
    let next = next_line?;
    let trimmed = next.trim();

    if trimmed.chars().all(|c| c == '=') && trimmed.len() >= 3 {
        return Some(Block::Heading {
            level: 1,
            content: parse_inline(line.trim()),
        });
    }

    if trimmed.chars().all(|c| c == '-') && trimmed.len() >= 3 {
        return Some(Block::Heading {
            level: 2,
            content: parse_inline(line.trim()),
        });
    }

    None
}
\`\`\`

### HTML Output
\`\`\`rust
impl Block {
    pub fn to_html(&self) -> String {
        match self {
            Block::Heading { level, content } => {
                let inner = content.iter()
                    .map(|i| i.to_html())
                    .collect::<String>();
                format!("<h{level}>{inner}</h{level}>")
            }
            // ...
        }
    }
}
\`\`\``
					},
					{
						title: 'Parse paragraphs',
						description: 'Handle text blocks with proper spacing',
						details: `## Paragraph Parsing

### Paragraph Rules
- Consecutive non-blank lines form a paragraph
- Blank lines separate paragraphs
- Some elements interrupt paragraphs (headers, lists, etc.)

### Implementation
\`\`\`rust
pub struct Parser<'a> {
    lines: std::iter::Peekable<std::str::Lines<'a>>,
    blocks: Vec<Block>,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            lines: input.lines().peekable(),
            blocks: Vec::new(),
        }
    }

    pub fn parse(&mut self) -> Vec<Block> {
        while let Some(line) = self.lines.next() {
            if line.trim().is_empty() {
                continue; // Skip blank lines
            }

            // Try special blocks first
            if let Some(block) = self.try_parse_special(line) {
                self.blocks.push(block);
                continue;
            }

            // Otherwise, start a paragraph
            self.parse_paragraph(line);
        }

        std::mem::take(&mut self.blocks)
    }

    fn parse_paragraph(&mut self, first_line: &str) {
        let mut lines = vec![first_line.to_string()];

        // Collect continuation lines
        while let Some(line) = self.lines.peek() {
            if line.trim().is_empty() {
                break;
            }
            if self.is_block_start(line) {
                break;
            }
            lines.push(self.lines.next().unwrap().to_string());
        }

        let content = lines.join(" ");
        self.blocks.push(Block::Paragraph(parse_inline(&content)));
    }

    fn is_block_start(&self, line: &str) -> bool {
        let trimmed = line.trim();
        trimmed.starts_with('#')
            || trimmed.starts_with('>')
            || trimmed.starts_with("\`\`\`")
            || trimmed.starts_with("- ")
            || trimmed.starts_with("* ")
            || trimmed.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
    }
}
\`\`\`

### Line Break Handling
\`\`\`rust
// Hard line breaks: two spaces at end of line
// or backslash before newline
fn process_line_breaks(text: &str) -> String {
    text.lines()
        .map(|line| {
            if line.ends_with("  ") || line.ends_with("\\\\") {
                format!("{}<br>", line.trim_end_matches(|c| c == ' ' || c == '\\\\'))
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
\`\`\``
					},
					{
						title: 'Parse lists (ordered/unordered)',
						description: 'Support nested list structures',
						details: `## List Parsing

### List Syntax
\`\`\`markdown
- Item 1
- Item 2
  - Nested item
  - Another nested
- Item 3

1. First
2. Second
   1. Nested ordered
3. Third
\`\`\`

### Data Structures
\`\`\`rust
#[derive(Debug, Clone)]
pub enum ListType {
    Unordered,
    Ordered(u32), // Starting number
}

#[derive(Debug, Clone)]
pub struct ListItem {
    pub content: Vec<Inline>,
    pub children: Option<Box<Block>>, // Nested list
}

// Block::List variant
Block::List {
    list_type: ListType,
    items: Vec<ListItem>,
}
\`\`\`

### Parsing with Indentation
\`\`\`rust
fn parse_list(&mut self, first_line: &str) -> Block {
    let base_indent = get_indent(first_line);
    let list_type = detect_list_type(first_line);
    let mut items = Vec::new();

    // Parse first item
    items.push(self.parse_list_item(first_line, base_indent));

    // Continue while we have list items at same or greater indent
    while let Some(line) = self.lines.peek() {
        let indent = get_indent(line);

        if line.trim().is_empty() {
            self.lines.next();
            continue;
        }

        if indent < base_indent {
            break; // Dedent ends list
        }

        if indent == base_indent && is_list_item(line) {
            items.push(self.parse_list_item(
                self.lines.next().unwrap(),
                base_indent
            ));
        } else if indent > base_indent {
            // Nested content - add to last item
            if let Some(last) = items.last_mut() {
                let nested = self.parse_nested(base_indent + 2);
                last.children = Some(Box::new(nested));
            }
        } else {
            break;
        }
    }

    Block::List { list_type, items }
}

fn get_indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

fn is_list_item(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
        || trimmed.starts_with("+ ")
        || trimmed.chars().next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
}
\`\`\`

### HTML Output
\`\`\`rust
fn list_to_html(list_type: &ListType, items: &[ListItem]) -> String {
    let tag = match list_type {
        ListType::Unordered => "ul",
        ListType::Ordered(_) => "ol",
    };

    let items_html: String = items.iter()
        .map(|item| {
            let content = item.content.iter()
                .map(|i| i.to_html())
                .collect::<String>();
            let nested = item.children.as_ref()
                .map(|b| b.to_html())
                .unwrap_or_default();
            format!("<li>{content}{nested}</li>")
        })
        .collect();

    format!("<{tag}>{items_html}</{tag}>")
}
\`\`\``
					},
					{
						title: 'Parse code blocks with language hints',
						description: 'Handle fenced code blocks with syntax hints',
						details: `## Code Block Parsing

### Syntax
\`\`\`markdown
\\\`\\\`\\\`rust
fn main() {
    println!("Hello!");
}
\\\`\\\`\\\`
\`\`\`

### Implementation
\`\`\`rust
#[derive(Debug, Clone)]
pub struct CodeBlock {
    pub language: Option<String>,
    pub content: String,
}

fn parse_code_block(&mut self, first_line: &str) -> Option<Block> {
    // Detect fence type (backticks or tildes)
    let trimmed = first_line.trim_start();
    let fence_char = trimmed.chars().next()?;

    if fence_char != '\`' && fence_char != '~' {
        return None;
    }

    let fence_len = trimmed.chars()
        .take_while(|&c| c == fence_char)
        .count();

    if fence_len < 3 {
        return None;
    }

    // Extract language hint
    let info_string = trimmed[fence_len..].trim();
    let language = if info_string.is_empty() {
        None
    } else {
        // Take first word as language
        Some(info_string.split_whitespace()
            .next()
            .unwrap()
            .to_string())
    };

    // Collect code lines
    let mut content = String::new();
    let fence = fence_char.to_string().repeat(fence_len);

    while let Some(line) = self.lines.next() {
        if line.trim().starts_with(&fence) {
            break;
        }
        content.push_str(line);
        content.push('\\n');
    }

    // Remove trailing newline
    if content.ends_with('\\n') {
        content.pop();
    }

    Some(Block::CodeBlock(CodeBlock { language, content }))
}
\`\`\`

### HTML Output with Escaping
\`\`\`rust
impl CodeBlock {
    pub fn to_html(&self) -> String {
        let escaped = html_escape(&self.content);

        let class = self.language.as_ref()
            .map(|lang| format!(" class=\\"language-{lang}\\""))
            .unwrap_or_default();

        format!("<pre><code{class}>{escaped}</code></pre>")
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
}
\`\`\`

### Indented Code Blocks
\`\`\`rust
// 4 spaces or 1 tab = code block
fn parse_indented_code(&mut self, first_line: &str) -> Option<Block> {
    if !first_line.starts_with("    ") && !first_line.starts_with("\\t") {
        return None;
    }

    let mut lines = vec![strip_indent(first_line)];

    while let Some(line) = self.lines.peek() {
        if line.trim().is_empty() {
            lines.push(String::new());
            self.lines.next();
        } else if line.starts_with("    ") || line.starts_with("\\t") {
            lines.push(strip_indent(self.lines.next().unwrap()));
        } else {
            break;
        }
    }

    Some(Block::CodeBlock(CodeBlock {
        language: None,
        content: lines.join("\\n"),
    }))
}
\`\`\``
					}
				]
			},
			{
				name: 'Inline Elements',
				description: 'Handle inline formatting',
				tasks: [
					{
						title: 'Parse bold and italic',
						description: 'Handle **bold** and *italic* syntax',
						details: `## Emphasis Parsing

### Syntax Rules
- \`*text*\` or \`_text_\` = italic (em)
- \`**text**\` or \`__text__\` = bold (strong)
- \`***text***\` = bold italic
- Can be nested: \`**bold and *italic***\`

### Inline Types
\`\`\`rust
#[derive(Debug, Clone)]
pub enum Inline {
    Text(String),
    Emphasis(Vec<Inline>),      // *italic*
    Strong(Vec<Inline>),        // **bold**
    Code(String),               // \`code\`
    Link { text: Vec<Inline>, url: String, title: Option<String> },
    Image { alt: String, url: String, title: Option<String> },
    LineBreak,
}
\`\`\`

### Delimiter Stack Parser
\`\`\`rust
#[derive(Debug)]
struct Delimiter {
    char: char,
    count: usize,
    can_open: bool,
    can_close: bool,
    position: usize,
}

pub fn parse_inline(text: &str) -> Vec<Inline> {
    let mut result = Vec::new();
    let mut delimiters: Vec<Delimiter> = Vec::new();
    let mut current_text = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if c == '*' || c == '_' {
            // Count consecutive delimiters
            let count = chars[i..].iter()
                .take_while(|&&x| x == c)
                .count();

            // Determine if can open/close
            let before = if i > 0 { chars.get(i - 1) } else { None };
            let after = chars.get(i + count);

            let can_open = after.map(|&c| !c.is_whitespace()).unwrap_or(false);
            let can_close = before.map(|&c| !c.is_whitespace()).unwrap_or(false);

            // Save current text
            if !current_text.is_empty() {
                result.push(Inline::Text(std::mem::take(&mut current_text)));
            }

            delimiters.push(Delimiter {
                char: c,
                count,
                can_open,
                can_close,
                position: result.len(),
            });

            i += count;
        } else {
            current_text.push(c);
            i += 1;
        }
    }

    if !current_text.is_empty() {
        result.push(Inline::Text(current_text));
    }

    // Process delimiter pairs
    process_delimiters(&mut result, &delimiters);

    result
}
\`\`\`

### Matching Delimiters
\`\`\`rust
fn process_delimiters(inlines: &mut Vec<Inline>, delimiters: &[Delimiter]) {
    // Find matching pairs from inside out
    let mut i = delimiters.len();
    while i > 0 {
        i -= 1;
        let closer = &delimiters[i];
        if !closer.can_close {
            continue;
        }

        // Find matching opener
        for j in (0..i).rev() {
            let opener = &delimiters[j];
            if opener.char == closer.char
                && opener.can_open
                && opener.count > 0
            {
                // Match found - wrap content
                let count = opener.count.min(closer.count).min(2);
                let wrapper = if count == 2 {
                    Inline::Strong
                } else {
                    Inline::Emphasis
                };

                // ... wrap content between positions
                break;
            }
        }
    }
}
\`\`\``
					},
					{
						title: 'Parse inline code',
						description: 'Handle \`code\` spans',
						details: `## Inline Code Parsing

### Syntax
- Single backtick: \\\`code\\\`
- Double backtick: \\\`\\\`code with \\\` inside\\\`\\\`
- Spaces are trimmed if present on both sides

### Implementation
\`\`\`rust
fn parse_inline_code(text: &str, start: usize) -> Option<(Inline, usize)> {
    let chars: Vec<char> = text[start..].chars().collect();

    // Count opening backticks
    let open_count = chars.iter()
        .take_while(|&&c| c == '\`')
        .count();

    if open_count == 0 {
        return None;
    }

    // Find matching closing backticks
    let content_start = open_count;
    let mut i = content_start;

    while i < chars.len() {
        if chars[i] == '\`' {
            let close_count = chars[i..].iter()
                .take_while(|&&c| c == '\`')
                .count();

            if close_count == open_count {
                // Found match
                let content: String = chars[content_start..i].iter().collect();

                // Trim single space from both ends if present
                let trimmed = if content.starts_with(' ')
                    && content.ends_with(' ')
                    && content.len() > 2
                {
                    content[1..content.len()-1].to_string()
                } else {
                    content
                };

                let consumed = start + open_count + (i - content_start) + close_count;
                return Some((Inline::Code(trimmed), consumed));
            }

            i += close_count;
        } else {
            i += 1;
        }
    }

    // No closing backticks found - treat as literal
    None
}
\`\`\`

### Edge Cases
\`\`\`rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inline_code() {
        // Basic
        assert_eq!(
            parse_inline("\`code\`"),
            vec![Inline::Code("code".into())]
        );

        // With backtick inside
        assert_eq!(
            parse_inline("\`\`code \` here\`\`"),
            vec![Inline::Code("code \` here".into())]
        );

        // Space trimming
        assert_eq!(
            parse_inline("\` code \`"),
            vec![Inline::Code("code".into())]
        );

        // Only spaces - not trimmed
        assert_eq!(
            parse_inline("\`  \`"),
            vec![Inline::Code("  ".into())]
        );
    }
}
\`\`\``
					},
					{
						title: 'Parse links and images',
						description: 'Handle [text](url) and ![alt](src) syntax',
						details: `## Link and Image Parsing

### Syntax
\`\`\`markdown
[link text](url "optional title")
![alt text](image.png "optional title")
[reference link][ref]
[ref]: url "title"
\`\`\`

### Implementation
\`\`\`rust
fn parse_link_or_image(text: &str, start: usize) -> Option<(Inline, usize)> {
    let chars: Vec<char> = text[start..].chars().collect();

    let is_image = chars.first() == Some(&'!');
    let bracket_start = if is_image { 1 } else { 0 };

    if chars.get(bracket_start) != Some(&'[') {
        return None;
    }

    // Find closing bracket
    let mut depth = 1;
    let mut i = bracket_start + 1;
    let mut text_end = None;

    while i < chars.len() {
        match chars[i] {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    text_end = Some(i);
                    break;
                }
            }
            '\\\\' => i += 1, // Skip escaped char
            _ => {}
        }
        i += 1;
    }

    let text_end = text_end?;
    let link_text: String = chars[bracket_start + 1..text_end].iter().collect();

    // Check for inline link (url)
    if chars.get(text_end + 1) == Some(&'(') {
        let (url, title, end) = parse_link_destination(&chars[text_end + 2..])?;

        let inline = if is_image {
            Inline::Image { alt: link_text, url, title }
        } else {
            Inline::Link {
                text: parse_inline(&link_text),
                url,
                title,
            }
        };

        return Some((inline, start + text_end + 2 + end + 1));
    }

    // Check for reference link [text][ref]
    // ... handle reference links

    None
}

fn parse_link_destination(chars: &[char]) -> Option<(String, Option<String>, usize)> {
    let mut i = 0;
    let mut url = String::new();
    let mut title = None;

    // Skip whitespace
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }

    // Parse URL (may be in angle brackets)
    if chars.get(i) == Some(&'<') {
        i += 1;
        while i < chars.len() && chars[i] != '>' {
            url.push(chars[i]);
            i += 1;
        }
        i += 1; // Skip >
    } else {
        let mut parens = 0;
        while i < chars.len() {
            match chars[i] {
                ')' if parens == 0 => break,
                '(' => { parens += 1; url.push('('); }
                ')' => { parens -= 1; url.push(')'); }
                c if c.is_whitespace() => break,
                c => url.push(c),
            }
            i += 1;
        }
    }

    // Parse optional title
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }

    if let Some(&quote) = chars.get(i) {
        if quote == '"' || quote == '\\'' {
            i += 1;
            let mut t = String::new();
            while i < chars.len() && chars[i] != quote {
                t.push(chars[i]);
                i += 1;
            }
            title = Some(t);
            i += 1;
        }
    }

    // Find closing paren
    while i < chars.len() && chars[i] != ')' {
        if !chars[i].is_whitespace() {
            return None; // Invalid
        }
        i += 1;
    }

    Some((url, title, i))
}
\`\`\``
					},
					{
						title: 'Emit clean, valid HTML',
						description: 'Generate properly formatted HTML output',
						details: `## HTML Generation

### HTML Escaping
\`\`\`rust
pub fn escape_html(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => result.push_str("&amp;"),
            '<' => result.push_str("&lt;"),
            '>' => result.push_str("&gt;"),
            '"' => result.push_str("&quot;"),
            '\\'' => result.push_str("&#39;"),
            _ => result.push(c),
        }
    }
    result
}

pub fn escape_url(url: &str) -> String {
    // Percent-encode special characters
    url.chars()
        .map(|c| match c {
            ' ' => "%20".to_string(),
            '"' => "%22".to_string(),
            '<' => "%3C".to_string(),
            '>' => "%3E".to_string(),
            _ => c.to_string(),
        })
        .collect()
}
\`\`\`

### Complete HTML Output
\`\`\`rust
impl Inline {
    pub fn to_html(&self) -> String {
        match self {
            Inline::Text(t) => escape_html(t),
            Inline::Emphasis(content) => {
                let inner: String = content.iter()
                    .map(|i| i.to_html())
                    .collect();
                format!("<em>{inner}</em>")
            }
            Inline::Strong(content) => {
                let inner: String = content.iter()
                    .map(|i| i.to_html())
                    .collect();
                format!("<strong>{inner}</strong>")
            }
            Inline::Code(code) => {
                format!("<code>{}</code>", escape_html(code))
            }
            Inline::Link { text, url, title } => {
                let inner: String = text.iter()
                    .map(|i| i.to_html())
                    .collect();
                let title_attr = title.as_ref()
                    .map(|t| format!(" title=\\"{}\\"", escape_html(t)))
                    .unwrap_or_default();
                format!(
                    "<a href=\\"{}\\"{}>{}</a>",
                    escape_url(url),
                    title_attr,
                    inner
                )
            }
            Inline::Image { alt, url, title } => {
                let title_attr = title.as_ref()
                    .map(|t| format!(" title=\\"{}\\"", escape_html(t)))
                    .unwrap_or_default();
                format!(
                    "<img src=\\"{}\\"{} alt=\\"{}\\">",
                    escape_url(url),
                    title_attr,
                    escape_html(alt)
                )
            }
            Inline::LineBreak => "<br>".to_string(),
        }
    }
}
\`\`\`

### Full Document Output
\`\`\`rust
pub fn render_document(blocks: &[Block]) -> String {
    blocks.iter()
        .map(|b| b.to_html())
        .collect::<Vec<_>>()
        .join("\\n")
}

pub fn render_full_html(blocks: &[Block], title: &str) -> String {
    let body = render_document(blocks);
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body>
{body}
</body>
</html>"#,
        title = escape_html(title),
        body = body
    )
}
\`\`\``
					}
				]
			}
		]
	},
	{
		name: 'Simple Key-Value Store',
		description: 'Persistent key-value store with a network interface.',
		language: 'Rust',
		color: 'amber',
		skills: 'ownership, low-level I/O, data structures, networking',
		startHint: 'In-memory hashmap with a TCP interface, add persistence later',
		difficulty: 'advanced',
		estimatedWeeks: 2,
		schedule: `## 2-Week Schedule

### Week 1: Core Store
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Data Structures | Store with RwLock, basic operations |
| Day 2 | RESP Protocol | Parser and encoder for wire format |
| Day 3 | TCP Server | Async server with tokio |
| Day 4 | Concurrent Clients | Connection handling, limits |
| Day 5 | TTL Support | Key expiration, background cleanup |
| Weekend | Buffer | Client library, testing |

### Week 2: Persistence
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Append-Only Log | WAL with checksums |
| Day 2 | Log Compaction | Merge and remove old entries |
| Day 3 | Startup Recovery | Rebuild state from log |
| Day 4 | Benchmarking | Performance testing, optimization |
| Day 5 | CLI & Config | Server configuration, client CLI |
| Weekend | (Optional) | LSM tree implementation |

### Daily Commitment
- **Minimum**: 2-3 hours focused coding
- **Ideal**: 4-5 hours with breaks (advanced project)`,
		modules: [
			{
				name: 'Core Store',
				description: 'Build the storage engine',
				tasks: [
					{
						title: 'GET, SET, DELETE operations',
						description: 'Implement basic CRUD for key-value pairs',
						details: `## Key-Value Store Core

### Data Structures
\`\`\`rust
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct Value {
    data: Vec<u8>,
    expires_at: Option<std::time::Instant>,
}

pub struct Store {
    data: RwLock<HashMap<String, Value>>,
}

impl Store {
    pub fn new() -> Self {
        Store {
            data: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        let data = self.data.read().unwrap();
        data.get(key).and_then(|v| {
            if let Some(exp) = v.expires_at {
                if std::time::Instant::now() > exp {
                    return None; // Expired
                }
            }
            Some(v.data.clone())
        })
    }

    pub fn set(&self, key: String, value: Vec<u8>, ttl: Option<u64>) {
        let expires_at = ttl.map(|secs| {
            std::time::Instant::now() + std::time::Duration::from_secs(secs)
        });

        let mut data = self.data.write().unwrap();
        data.insert(key, Value { data: value, expires_at });
    }

    pub fn delete(&self, key: &str) -> bool {
        let mut data = self.data.write().unwrap();
        data.remove(key).is_some()
    }

    pub fn exists(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
}
\`\`\`

### Additional Operations
\`\`\`rust
impl Store {
    // Increment numeric value
    pub fn incr(&self, key: &str, delta: i64) -> Result<i64, &'static str> {
        let mut data = self.data.write().unwrap();

        let current = data.get(key)
            .map(|v| String::from_utf8_lossy(&v.data).parse::<i64>())
            .transpose()
            .map_err(|_| "value is not an integer")?
            .unwrap_or(0);

        let new_value = current + delta;
        data.insert(
            key.to_string(),
            Value {
                data: new_value.to_string().into_bytes(),
                expires_at: None,
            }
        );

        Ok(new_value)
    }

    // Get all keys matching pattern
    pub fn keys(&self, pattern: &str) -> Vec<String> {
        let data = self.data.read().unwrap();
        data.keys()
            .filter(|k| matches_pattern(k, pattern))
            .cloned()
            .collect()
    }

    // Atomic get-and-delete
    pub fn getdel(&self, key: &str) -> Option<Vec<u8>> {
        let mut data = self.data.write().unwrap();
        data.remove(key).map(|v| v.data)
    }
}
\`\`\``
					},
					{
						title: 'TCP interface with simple protocol',
						description: 'Create wire protocol for client communication',
						details: `## TCP Protocol

### RESP-like Protocol
\`\`\`
Simple strings: +OK\\r\\n
Errors: -ERR message\\r\\n
Integers: :1000\\r\\n
Bulk strings: $6\\r\\nfoobar\\r\\n
Arrays: *2\\r\\n$3\\r\\nGET\\r\\n$3\\r\\nkey\\r\\n
Null: $-1\\r\\n
\`\`\`

### Protocol Parser
\`\`\`rust
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

#[derive(Debug)]
pub enum RespValue {
    SimpleString(String),
    Error(String),
    Integer(i64),
    BulkString(Option<Vec<u8>>),
    Array(Option<Vec<RespValue>>),
}

pub async fn read_value(reader: &mut BufReader<TcpStream>) -> Result<RespValue, std::io::Error> {
    let mut line = String::new();
    reader.read_line(&mut line).await?;

    let line = line.trim_end();
    let (prefix, rest) = line.split_at(1);

    match prefix {
        "+" => Ok(RespValue::SimpleString(rest.to_string())),
        "-" => Ok(RespValue::Error(rest.to_string())),
        ":" => Ok(RespValue::Integer(rest.parse().unwrap())),
        "$" => {
            let len: i32 = rest.parse().unwrap();
            if len < 0 {
                return Ok(RespValue::BulkString(None));
            }
            let mut buf = vec![0u8; len as usize + 2];
            reader.read_exact(&mut buf).await?;
            buf.truncate(len as usize);
            Ok(RespValue::BulkString(Some(buf)))
        }
        "*" => {
            let len: i32 = rest.parse().unwrap();
            if len < 0 {
                return Ok(RespValue::Array(None));
            }
            let mut items = Vec::with_capacity(len as usize);
            for _ in 0..len {
                items.push(Box::pin(read_value(reader)).await?);
            }
            Ok(RespValue::Array(Some(items)))
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid RESP prefix"
        )),
    }
}
\`\`\`

### Response Writer
\`\`\`rust
impl RespValue {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            RespValue::SimpleString(s) => format!("+{}\\r\\n", s).into_bytes(),
            RespValue::Error(e) => format!("-{}\\r\\n", e).into_bytes(),
            RespValue::Integer(i) => format!(":{}\\r\\n", i).into_bytes(),
            RespValue::BulkString(None) => b"$-1\\r\\n".to_vec(),
            RespValue::BulkString(Some(data)) => {
                let mut buf = format!("\${}\\r\\n", data.len()).into_bytes();
                buf.extend_from_slice(data);
                buf.extend_from_slice(b"\\r\\n");
                buf
            }
            RespValue::Array(None) => b"*-1\\r\\n".to_vec(),
            RespValue::Array(Some(items)) => {
                let mut buf = format!("*{}\\r\\n", items.len()).into_bytes();
                for item in items {
                    buf.extend_from_slice(&item.encode());
                }
                buf
            }
        }
    }
}
\`\`\``
					},
					{
						title: 'Handle concurrent clients',
						description: 'Support multiple simultaneous connections',
						details: `## Concurrent Connection Handling

### Async Server
\`\`\`rust
use tokio::net::TcpListener;
use std::sync::Arc;

pub async fn run_server(addr: &str, store: Arc<Store>) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    println!("Listening on {}", addr);

    loop {
        let (socket, addr) = listener.accept().await?;
        let store = Arc::clone(&store);

        tokio::spawn(async move {
            if let Err(e) = handle_client(socket, store).await {
                eprintln!("Client {} error: {}", addr, e);
            }
        });
    }
}

async fn handle_client(
    socket: TcpStream,
    store: Arc<Store>,
) -> std::io::Result<()> {
    let (reader, mut writer) = socket.into_split();
    let mut reader = BufReader::new(reader);

    loop {
        let value = match read_value(&mut reader).await {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(()); // Client disconnected
            }
            Err(e) => return Err(e),
        };

        let response = process_command(&store, value);
        writer.write_all(&response.encode()).await?;
    }
}
\`\`\`

### Command Processing
\`\`\`rust
fn process_command(store: &Store, value: RespValue) -> RespValue {
    let RespValue::Array(Some(parts)) = value else {
        return RespValue::Error("Expected array".into());
    };

    let mut iter = parts.into_iter();
    let cmd = match iter.next() {
        Some(RespValue::BulkString(Some(b))) => {
            String::from_utf8_lossy(&b).to_uppercase()
        }
        _ => return RespValue::Error("Expected command".into()),
    };

    match cmd.as_str() {
        "GET" => {
            let key = extract_string(&mut iter);
            match store.get(&key) {
                Some(data) => RespValue::BulkString(Some(data)),
                None => RespValue::BulkString(None),
            }
        }
        "SET" => {
            let key = extract_string(&mut iter);
            let value = extract_bytes(&mut iter);
            store.set(key, value, None);
            RespValue::SimpleString("OK".into())
        }
        "DEL" => {
            let key = extract_string(&mut iter);
            let deleted = store.delete(&key);
            RespValue::Integer(if deleted { 1 } else { 0 })
        }
        "PING" => RespValue::SimpleString("PONG".into()),
        _ => RespValue::Error(format!("Unknown command: {}", cmd)),
    }
}
\`\`\`

### Connection Limits
\`\`\`rust
use tokio::sync::Semaphore;

pub async fn run_server_with_limits(
    addr: &str,
    store: Arc<Store>,
    max_connections: usize,
) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    let semaphore = Arc::new(Semaphore::new(max_connections));

    loop {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let (socket, _) = listener.accept().await?;
        let store = Arc::clone(&store);

        tokio::spawn(async move {
            let _permit = permit; // Held until task completes
            handle_client(socket, store).await.ok();
        });
    }
}
\`\`\``
					},
					{
						title: 'Implement expiration (TTL)',
						description: 'Support automatic key expiration',
						details: `## Key Expiration

### TTL Commands
\`\`\`rust
impl Store {
    pub fn set_ex(&self, key: String, value: Vec<u8>, seconds: u64) {
        self.set(key, value, Some(seconds));
    }

    pub fn ttl(&self, key: &str) -> Option<i64> {
        let data = self.data.read().unwrap();
        data.get(key).map(|v| {
            match v.expires_at {
                None => -1, // No expiration
                Some(exp) => {
                    let now = std::time::Instant::now();
                    if now > exp {
                        -2 // Expired
                    } else {
                        (exp - now).as_secs() as i64
                    }
                }
            }
        })
    }

    pub fn expire(&self, key: &str, seconds: u64) -> bool {
        let mut data = self.data.write().unwrap();
        if let Some(value) = data.get_mut(key) {
            value.expires_at = Some(
                std::time::Instant::now() +
                std::time::Duration::from_secs(seconds)
            );
            true
        } else {
            false
        }
    }

    pub fn persist(&self, key: &str) -> bool {
        let mut data = self.data.write().unwrap();
        if let Some(value) = data.get_mut(key) {
            value.expires_at = None;
            true
        } else {
            false
        }
    }
}
\`\`\`

### Background Expiration
\`\`\`rust
pub fn start_expiration_task(store: Arc<Store>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(
            std::time::Duration::from_secs(1)
        );

        loop {
            interval.tick().await;

            let now = std::time::Instant::now();
            let mut data = store.data.write().unwrap();

            // Collect expired keys
            let expired: Vec<String> = data.iter()
                .filter_map(|(k, v)| {
                    v.expires_at.and_then(|exp| {
                        if now > exp { Some(k.clone()) } else { None }
                    })
                })
                .take(100) // Limit per iteration
                .collect();

            for key in expired {
                data.remove(&key);
            }
        }
    });
}
\`\`\`

### Lazy Expiration
\`\`\`rust
// Also check on access (lazy expiration)
pub fn get(&self, key: &str) -> Option<Vec<u8>> {
    // First check without write lock
    {
        let data = self.data.read().unwrap();
        if let Some(v) = data.get(key) {
            if let Some(exp) = v.expires_at {
                if std::time::Instant::now() <= exp {
                    return Some(v.data.clone());
                }
            } else {
                return Some(v.data.clone());
            }
        } else {
            return None;
        }
    }

    // Key expired - remove it
    let mut data = self.data.write().unwrap();
    data.remove(key);
    None
}
\`\`\``
					}
				]
			},
			{
				name: 'Persistence',
				description: 'Add data durability',
				tasks: [
					{
						title: 'Persist to disk (append-only log)',
						description: 'Implement WAL for durability',
						details: `## Append-Only Log (AOF)

### Log Entry Format
\`\`\`rust
use bincode::{serialize, deserialize};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub enum LogEntry {
    Set { key: String, value: Vec<u8>, ttl: Option<u64> },
    Delete { key: String },
}

impl LogEntry {
    pub fn encode(&self) -> Vec<u8> {
        let data = serialize(self).unwrap();
        let len = data.len() as u32;

        let mut buf = Vec::with_capacity(4 + data.len());
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&data);
        buf
    }

    pub fn decode(buf: &[u8]) -> Option<(Self, usize)> {
        if buf.len() < 4 {
            return None;
        }

        let len = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        if buf.len() < 4 + len {
            return None;
        }

        let entry: LogEntry = deserialize(&buf[4..4+len]).ok()?;
        Some((entry, 4 + len))
    }
}
\`\`\`

### AOF Writer
\`\`\`rust
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};

pub struct AOFWriter {
    writer: BufWriter<File>,
    sync_every: usize,
    pending: usize,
}

impl AOFWriter {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        Ok(AOFWriter {
            writer: BufWriter::new(file),
            sync_every: 1, // fsync every write for durability
            pending: 0,
        })
    }

    pub fn append(&mut self, entry: &LogEntry) -> std::io::Result<()> {
        let data = entry.encode();
        self.writer.write_all(&data)?;

        self.pending += 1;
        if self.pending >= self.sync_every {
            self.writer.flush()?;
            self.writer.get_ref().sync_all()?;
            self.pending = 0;
        }

        Ok(())
    }
}
\`\`\`

### Store with AOF
\`\`\`rust
pub struct PersistentStore {
    store: Store,
    aof: Mutex<AOFWriter>,
}

impl PersistentStore {
    pub fn set(&self, key: String, value: Vec<u8>, ttl: Option<u64>) {
        // Write to AOF first (for durability)
        {
            let mut aof = self.aof.lock().unwrap();
            aof.append(&LogEntry::Set {
                key: key.clone(),
                value: value.clone(),
                ttl,
            }).unwrap();
        }

        // Then update in-memory store
        self.store.set(key, value, ttl);
    }
}
\`\`\``
					},
					{
						title: 'Log compaction',
						description: 'Merge and compact old log entries',
						details: `## Log Compaction

### Why Compact?
- AOF grows indefinitely
- Old values are superseded
- Deleted keys still in log

### Compaction Strategy
\`\`\`rust
pub struct Compactor {
    store: Arc<PersistentStore>,
    aof_path: String,
    threshold_mb: u64,
}

impl Compactor {
    pub fn compact(&self) -> std::io::Result<()> {
        let aof_size = std::fs::metadata(&self.aof_path)?.len();
        if aof_size < self.threshold_mb * 1024 * 1024 {
            return Ok(()); // Not large enough
        }

        // Create new AOF with current state
        let temp_path = format!("{}.tmp", self.aof_path);
        let mut writer = AOFWriter::open(&temp_path)?;

        // Snapshot current state
        let data = self.store.store.data.read().unwrap();
        for (key, value) in data.iter() {
            // Skip expired entries
            if let Some(exp) = value.expires_at {
                if std::time::Instant::now() > exp {
                    continue;
                }
            }

            writer.append(&LogEntry::Set {
                key: key.clone(),
                value: value.data.clone(),
                ttl: value.expires_at.map(|exp| {
                    (exp - std::time::Instant::now()).as_secs()
                }),
            })?;
        }
        drop(data);

        // Atomic rename
        std::fs::rename(&temp_path, &self.aof_path)?;

        Ok(())
    }
}
\`\`\`

### Background Compaction
\`\`\`rust
pub fn start_compaction_task(
    store: Arc<PersistentStore>,
    aof_path: String,
    interval_secs: u64,
) {
    tokio::spawn(async move {
        let compactor = Compactor {
            store,
            aof_path,
            threshold_mb: 100,
        };

        let mut interval = tokio::time::interval(
            std::time::Duration::from_secs(interval_secs)
        );

        loop {
            interval.tick().await;
            if let Err(e) = compactor.compact() {
                eprintln!("Compaction error: {}", e);
            }
        }
    });
}
\`\`\``
					},
					{
						title: 'Recovery from log on startup',
						description: 'Rebuild state from persisted data',
						details: `## Startup Recovery

### AOF Reader
\`\`\`rust
use std::io::{BufReader, Read};

pub struct AOFReader {
    reader: BufReader<File>,
}

impl AOFReader {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        Ok(AOFReader {
            reader: BufReader::new(file),
        })
    }

    pub fn read_all(&mut self) -> std::io::Result<Vec<LogEntry>> {
        let mut entries = Vec::new();
        let mut buf = Vec::new();
        self.reader.read_to_end(&mut buf)?;

        let mut offset = 0;
        while offset < buf.len() {
            match LogEntry::decode(&buf[offset..]) {
                Some((entry, consumed)) => {
                    entries.push(entry);
                    offset += consumed;
                }
                None => break, // Incomplete entry (crash during write)
            }
        }

        Ok(entries)
    }
}
\`\`\`

### Recovery Process
\`\`\`rust
impl PersistentStore {
    pub fn open(aof_path: &str) -> std::io::Result<Self> {
        let store = Store::new();

        // Recover from AOF if exists
        if std::path::Path::new(aof_path).exists() {
            let mut reader = AOFReader::open(aof_path)?;
            let entries = reader.read_all()?;

            println!("Recovering {} entries from AOF", entries.len());

            for entry in entries {
                match entry {
                    LogEntry::Set { key, value, ttl } => {
                        store.set(key, value, ttl);
                    }
                    LogEntry::Delete { key } => {
                        store.delete(&key);
                    }
                }
            }
        }

        let aof = AOFWriter::open(aof_path)?;

        Ok(PersistentStore {
            store,
            aof: Mutex::new(aof),
        })
    }
}
\`\`\`

### Checksums for Integrity
\`\`\`rust
use crc32fast::Hasher;

impl LogEntry {
    pub fn encode_with_checksum(&self) -> Vec<u8> {
        let data = serialize(self).unwrap();

        let mut hasher = Hasher::new();
        hasher.update(&data);
        let checksum = hasher.finalize();

        let len = data.len() as u32;
        let mut buf = Vec::with_capacity(8 + data.len());
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf.extend_from_slice(&data);
        buf
    }

    pub fn decode_with_checksum(buf: &[u8]) -> Option<(Self, usize)> {
        if buf.len() < 8 {
            return None;
        }

        let len = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let stored_checksum = u32::from_le_bytes(buf[4..8].try_into().unwrap());

        if buf.len() < 8 + len {
            return None;
        }

        let data = &buf[8..8+len];

        // Verify checksum
        let mut hasher = Hasher::new();
        hasher.update(data);
        if hasher.finalize() != stored_checksum {
            return None; // Corruption detected
        }

        let entry: LogEntry = deserialize(data).ok()?;
        Some((entry, 8 + len))
    }
}
\`\`\``
					},
					{
						title: 'Optional: LSM tree implementation',
						description: 'Advanced storage engine structure',
						details: `## LSM Tree Basics

### LSM Structure
\`\`\`
Writes → MemTable (sorted in memory)
            ↓ (when full)
        Immutable MemTable
            ↓ (flush to disk)
        SSTable Level 0
            ↓ (compaction)
        SSTable Level 1, 2, ...
\`\`\`

### MemTable
\`\`\`rust
use std::collections::BTreeMap;

pub struct MemTable {
    data: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    size: usize,
    max_size: usize,
}

impl MemTable {
    pub fn new(max_size: usize) -> Self {
        MemTable {
            data: BTreeMap::new(),
            size: 0,
            max_size,
        }
    }

    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) -> bool {
        let entry_size = key.len() + value.len();
        self.size += entry_size;
        self.data.insert(key, Some(value));
        self.size >= self.max_size
    }

    pub fn delete(&mut self, key: Vec<u8>) {
        self.data.insert(key, None); // Tombstone
    }

    pub fn get(&self, key: &[u8]) -> Option<Option<&Vec<u8>>> {
        self.data.get(key).map(|v| v.as_ref())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Vec<u8>, &Option<Vec<u8>>)> {
        self.data.iter()
    }
}
\`\`\`

### SSTable Format
\`\`\`rust
pub struct SSTable {
    path: PathBuf,
    index: BTreeMap<Vec<u8>, u64>, // key -> file offset
    bloom_filter: BloomFilter,
}

impl SSTable {
    pub fn create(
        path: &Path,
        entries: impl Iterator<Item = (Vec<u8>, Option<Vec<u8>>)>,
    ) -> std::io::Result<Self> {
        let mut file = File::create(path)?;
        let mut index = BTreeMap::new();
        let mut bloom = BloomFilter::new(10000, 0.01);

        for (key, value) in entries {
            let offset = file.stream_position()?;
            bloom.insert(&key);
            index.insert(key.clone(), offset);

            // Write entry: key_len | key | value_len | value
            file.write_all(&(key.len() as u32).to_le_bytes())?;
            file.write_all(&key)?;

            match value {
                Some(v) => {
                    file.write_all(&(v.len() as u32).to_le_bytes())?;
                    file.write_all(&v)?;
                }
                None => {
                    file.write_all(&u32::MAX.to_le_bytes())?; // Tombstone
                }
            }
        }

        Ok(SSTable {
            path: path.to_path_buf(),
            index,
            bloom_filter: bloom,
        })
    }

    pub fn get(&self, key: &[u8]) -> std::io::Result<Option<Vec<u8>>> {
        // Check bloom filter first
        if !self.bloom_filter.might_contain(key) {
            return Ok(None);
        }

        // Binary search in index
        // ... read from file at offset
        Ok(None)
    }
}
\`\`\`

### Read Path
\`\`\`rust
impl LSMTree {
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        // 1. Check active memtable
        if let Some(result) = self.memtable.get(key) {
            return result.cloned();
        }

        // 2. Check immutable memtables
        for imm in &self.immutable_memtables {
            if let Some(result) = imm.get(key) {
                return result.cloned();
            }
        }

        // 3. Check SSTables (newest first)
        for level in &self.levels {
            for sstable in level.iter().rev() {
                if let Ok(Some(value)) = sstable.get(key) {
                    return Some(value);
                }
            }
        }

        None
    }
}
\`\`\``
					}
				]
			}
		]
	},
	{
		name: 'Process Monitor TUI',
		description: 'Display system process stats in a terminal UI.',
		language: 'Rust',
		color: 'emerald',
		skills: 'system calls, sysinfo crate, ratatui TUI, data formatting',
		startHint: 'Print process list to stdout, then add TUI with ratatui',
		difficulty: 'intermediate',
		estimatedWeeks: 2,
		schedule: `## 2-Week Schedule

### Week 1: Process Data
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Sysinfo Setup | List processes with CPU/memory |
| Day 2 | Sorting | Sort by different columns |
| Day 3 | Filtering | Search by name, resource thresholds |
| Day 4 | Auto-Refresh | Periodic updates, caching |
| Day 5 | System Stats | Overall CPU, memory, load average |
| Weekend | Buffer | Performance optimization |

### Week 2: TUI Interface
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Ratatui Setup | Terminal init, panic handler |
| Day 2 | Process Table | Scrollable table widget |
| Day 3 | Keyboard Nav | Arrow keys, page up/down |
| Day 4 | Details View | Split layout, process details |
| Day 5 | Polish | Help overlay, colors, graphs |
| Weekend | Testing | Cross-platform testing |

### Daily Commitment
- **Minimum**: 1-2 hours focused coding
- **Ideal**: 3-4 hours with breaks`,
		modules: [
			{
				name: 'Process Data',
				description: 'Gather system information',
				tasks: [
					{
						title: 'List processes with CPU/memory usage',
						description: 'Use sysinfo crate to gather process data',
						details: `## Process Information with sysinfo

### Setup
\`\`\`toml
# Cargo.toml
[dependencies]
sysinfo = "0.30"
\`\`\`

### Basic Process Listing
\`\`\`rust
use sysinfo::{System, Process, Pid};

pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub cpu_usage: f32,
    pub memory_mb: u64,
    pub status: String,
    pub user: Option<String>,
    pub command: Vec<String>,
}

pub fn get_processes() -> Vec<ProcessInfo> {
    let mut sys = System::new_all();
    sys.refresh_all();

    sys.processes()
        .iter()
        .map(|(pid, process)| ProcessInfo {
            pid: pid.as_u32(),
            name: process.name().to_string(),
            cpu_usage: process.cpu_usage(),
            memory_mb: process.memory() / 1024 / 1024,
            status: format!("{:?}", process.status()),
            user: process.user_id().map(|u| u.to_string()),
            command: process.cmd().to_vec(),
        })
        .collect()
}
\`\`\`

### System Stats
\`\`\`rust
pub struct SystemStats {
    pub total_memory_gb: f64,
    pub used_memory_gb: f64,
    pub total_swap_gb: f64,
    pub used_swap_gb: f64,
    pub cpu_usage: Vec<f32>,
    pub load_avg: [f64; 3],
    pub uptime_secs: u64,
}

pub fn get_system_stats() -> SystemStats {
    let mut sys = System::new_all();
    sys.refresh_all();

    SystemStats {
        total_memory_gb: sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
        used_memory_gb: sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
        total_swap_gb: sys.total_swap() as f64 / 1024.0 / 1024.0 / 1024.0,
        used_swap_gb: sys.used_swap() as f64 / 1024.0 / 1024.0 / 1024.0,
        cpu_usage: sys.cpus().iter().map(|c| c.cpu_usage()).collect(),
        load_avg: [
            System::load_average().one,
            System::load_average().five,
            System::load_average().fifteen,
        ],
        uptime_secs: System::uptime(),
    }
}
\`\`\`

### Caching for Performance
\`\`\`rust
pub struct ProcessMonitor {
    sys: System,
    last_refresh: std::time::Instant,
    refresh_interval: std::time::Duration,
}

impl ProcessMonitor {
    pub fn new(refresh_interval_ms: u64) -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        ProcessMonitor {
            sys,
            last_refresh: std::time::Instant::now(),
            refresh_interval: std::time::Duration::from_millis(refresh_interval_ms),
        }
    }

    pub fn refresh_if_needed(&mut self) {
        if self.last_refresh.elapsed() >= self.refresh_interval {
            self.sys.refresh_all();
            self.last_refresh = std::time::Instant::now();
        }
    }

    pub fn processes(&self) -> impl Iterator<Item = ProcessInfo> + '_ {
        self.sys.processes().iter().map(|(pid, p)| ProcessInfo {
            pid: pid.as_u32(),
            name: p.name().to_string(),
            cpu_usage: p.cpu_usage(),
            memory_mb: p.memory() / 1024 / 1024,
            status: format!("{:?}", p.status()),
            user: p.user_id().map(|u| u.to_string()),
            command: p.cmd().to_vec(),
        })
    }
}
\`\`\``
					},
					{
						title: 'Sort by different columns',
						description: 'Implement sortable data views',
						details: `## Sortable Process List

### Sort Configuration
\`\`\`rust
#[derive(Clone, Copy, PartialEq)]
pub enum SortColumn {
    Pid,
    Name,
    Cpu,
    Memory,
    Status,
}

#[derive(Clone, Copy, PartialEq)]
pub enum SortOrder {
    Ascending,
    Descending,
}

pub struct SortState {
    pub column: SortColumn,
    pub order: SortOrder,
}

impl SortState {
    pub fn toggle(&mut self, column: SortColumn) {
        if self.column == column {
            self.order = match self.order {
                SortOrder::Ascending => SortOrder::Descending,
                SortOrder::Descending => SortOrder::Ascending,
            };
        } else {
            self.column = column;
            self.order = SortOrder::Descending; // Default to desc for CPU/memory
        }
    }
}
\`\`\`

### Sorting Implementation
\`\`\`rust
impl ProcessInfo {
    pub fn cmp_by(&self, other: &Self, column: SortColumn) -> std::cmp::Ordering {
        match column {
            SortColumn::Pid => self.pid.cmp(&other.pid),
            SortColumn::Name => self.name.to_lowercase().cmp(&other.name.to_lowercase()),
            SortColumn::Cpu => self.cpu_usage.partial_cmp(&other.cpu_usage)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortColumn::Memory => self.memory_mb.cmp(&other.memory_mb),
            SortColumn::Status => self.status.cmp(&other.status),
        }
    }
}

pub fn sort_processes(
    processes: &mut [ProcessInfo],
    state: &SortState,
) {
    processes.sort_by(|a, b| {
        let ord = a.cmp_by(b, state.column);
        match state.order {
            SortOrder::Ascending => ord,
            SortOrder::Descending => ord.reverse(),
        }
    });
}
\`\`\`

### Display Sort Indicator
\`\`\`rust
fn column_header(name: &str, column: SortColumn, state: &SortState) -> String {
    if state.column == column {
        let arrow = match state.order {
            SortOrder::Ascending => "▲",
            SortOrder::Descending => "▼",
        };
        format!("{} {}", name, arrow)
    } else {
        name.to_string()
    }
}
\`\`\``
					},
					{
						title: 'Filter by name',
						description: 'Add search/filter functionality',
						details: `## Process Filtering

### Filter State
\`\`\`rust
pub struct FilterState {
    pub query: String,
    pub show_kernel: bool,
    pub min_cpu: f32,
    pub min_memory_mb: u64,
}

impl Default for FilterState {
    fn default() -> Self {
        FilterState {
            query: String::new(),
            show_kernel: false,
            min_cpu: 0.0,
            min_memory_mb: 0,
        }
    }
}

impl FilterState {
    pub fn matches(&self, process: &ProcessInfo) -> bool {
        // Name filter
        if !self.query.is_empty() {
            let query_lower = self.query.to_lowercase();
            if !process.name.to_lowercase().contains(&query_lower)
                && !process.pid.to_string().contains(&query_lower)
            {
                return false;
            }
        }

        // Kernel process filter
        if !self.show_kernel && process.name.starts_with('[') {
            return false;
        }

        // Resource thresholds
        if process.cpu_usage < self.min_cpu {
            return false;
        }
        if process.memory_mb < self.min_memory_mb {
            return false;
        }

        true
    }
}
\`\`\`

### Interactive Search
\`\`\`rust
pub struct SearchMode {
    pub active: bool,
    pub input: String,
    pub cursor: usize,
}

impl SearchMode {
    pub fn handle_key(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Char(c) => {
                self.input.insert(self.cursor, c);
                self.cursor += 1;
                true
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.input.remove(self.cursor);
                }
                true
            }
            KeyCode::Left => {
                self.cursor = self.cursor.saturating_sub(1);
                true
            }
            KeyCode::Right => {
                self.cursor = (self.cursor + 1).min(self.input.len());
                true
            }
            KeyCode::Escape => {
                self.active = false;
                self.input.clear();
                self.cursor = 0;
                true
            }
            KeyCode::Enter => {
                self.active = false;
                true
            }
            _ => false,
        }
    }
}

// Usage in app
fn handle_key(&mut self, key: KeyEvent) {
    if self.search.active {
        self.search.handle_key(key.code);
        self.filter.query = self.search.input.clone();
        return;
    }

    match key.code {
        KeyCode::Char('/') => {
            self.search.active = true;
        }
        // ... other keys
    }
}
\`\`\``
					},
					{
						title: 'Auto-refresh at intervals',
						description: 'Update display periodically',
						details: `## Auto-Refresh System

### Async Event Loop
\`\`\`rust
use std::time::Duration;
use tokio::sync::mpsc;
use crossterm::event::{self, Event, KeyCode};

pub enum AppEvent {
    Tick,
    Key(KeyEvent),
    Resize(u16, u16),
}

pub async fn event_loop(tx: mpsc::Sender<AppEvent>, tick_rate: Duration) {
    let mut interval = tokio::time::interval(tick_rate);

    loop {
        tokio::select! {
            _ = interval.tick() => {
                if tx.send(AppEvent::Tick).await.is_err() {
                    return;
                }
            }
            _ = tokio::task::spawn_blocking(|| event::poll(Duration::from_millis(50))) => {
                if let Ok(true) = event::poll(Duration::ZERO) {
                    if let Ok(event) = event::read() {
                        let app_event = match event {
                            Event::Key(key) => AppEvent::Key(key),
                            Event::Resize(w, h) => AppEvent::Resize(w, h),
                            _ => continue,
                        };
                        if tx.send(app_event).await.is_err() {
                            return;
                        }
                    }
                }
            }
        }
    }
}
\`\`\`

### Main Loop
\`\`\`rust
pub async fn run_app(
    terminal: &mut Terminal<impl Backend>,
    mut app: App,
    tick_rate: Duration,
) -> std::io::Result<()> {
    let (tx, mut rx) = mpsc::channel(100);

    tokio::spawn(event_loop(tx, tick_rate));

    loop {
        terminal.draw(|f| ui(f, &app))?;

        match rx.recv().await {
            Some(AppEvent::Tick) => {
                app.monitor.refresh_if_needed();
                app.update_processes();
            }
            Some(AppEvent::Key(key)) => {
                if key.code == KeyCode::Char('q') {
                    return Ok(());
                }
                app.handle_key(key);
            }
            Some(AppEvent::Resize(_, _)) => {
                // Terminal handles resize
            }
            None => return Ok(()),
        }
    }
}
\`\`\`

### Configurable Refresh Rate
\`\`\`rust
pub struct AppConfig {
    pub refresh_rate_ms: u64,
    pub show_threads: bool,
    pub color_scheme: ColorScheme,
}

impl App {
    pub fn set_refresh_rate(&mut self, ms: u64) {
        self.config.refresh_rate_ms = ms;
        self.monitor = ProcessMonitor::new(ms);
    }
}

// UI control
fn handle_key(&mut self, key: KeyEvent) {
    match key.code {
        KeyCode::Char('+') => {
            let current = self.config.refresh_rate_ms;
            self.set_refresh_rate((current + 500).min(5000));
        }
        KeyCode::Char('-') => {
            let current = self.config.refresh_rate_ms;
            self.set_refresh_rate((current - 500).max(500));
        }
        // ...
    }
}
\`\`\``
					}
				]
			},
			{
				name: 'TUI Interface',
				description: 'Build the terminal interface',
				tasks: [
					{
						title: 'Set up ratatui framework',
						description: 'Initialize terminal UI with proper setup/teardown',
						details: `## Ratatui Setup

### Dependencies
\`\`\`toml
# Cargo.toml
[dependencies]
ratatui = "0.26"
crossterm = "0.27"
\`\`\`

### Terminal Initialization
\`\`\`rust
use std::io;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

pub fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend)
}

pub fn restore_terminal(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
\`\`\`

### Panic Handler
\`\`\`rust
fn main() -> io::Result<()> {
    // Setup panic hook to restore terminal
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(panic);
    }));

    let mut terminal = setup_terminal()?;

    let result = run_app(&mut terminal);

    restore_terminal(&mut terminal)?;
    result
}
\`\`\`

### Basic App Structure
\`\`\`rust
pub struct App {
    pub monitor: ProcessMonitor,
    pub processes: Vec<ProcessInfo>,
    pub selected: usize,
    pub sort: SortState,
    pub filter: FilterState,
    pub should_quit: bool,
}

impl App {
    pub fn new() -> Self {
        let monitor = ProcessMonitor::new(1000);
        let processes = monitor.processes().collect();

        App {
            monitor,
            processes,
            selected: 0,
            sort: SortState {
                column: SortColumn::Cpu,
                order: SortOrder::Descending,
            },
            filter: FilterState::default(),
            should_quit: false,
        }
    }

    pub fn update_processes(&mut self) {
        self.processes = self.monitor
            .processes()
            .filter(|p| self.filter.matches(p))
            .collect();
        sort_processes(&mut self.processes, &self.sort);
    }
}
\`\`\``
					},
					{
						title: 'Create process table widget',
						description: 'Display processes in scrollable table',
						details: `## Process Table Widget

### Table Layout
\`\`\`rust
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Cell, Row, Table, TableState},
    Frame,
};

pub fn render_process_table(f: &mut Frame, app: &App, area: Rect) {
    let header_cells = ["PID", "Name", "CPU %", "Memory", "Status"]
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let column = match i {
                0 => SortColumn::Pid,
                1 => SortColumn::Name,
                2 => SortColumn::Cpu,
                3 => SortColumn::Memory,
                4 => SortColumn::Status,
                _ => unreachable!(),
            };
            let text = column_header(h, column, &app.sort);
            Cell::from(text).style(Style::default().fg(Color::Yellow))
        });

    let header = Row::new(header_cells)
        .style(Style::default().add_modifier(Modifier::BOLD))
        .height(1);

    let rows = app.processes.iter().map(|p| {
        let cpu_color = if p.cpu_usage > 80.0 {
            Color::Red
        } else if p.cpu_usage > 50.0 {
            Color::Yellow
        } else {
            Color::Green
        };

        Row::new(vec![
            Cell::from(p.pid.to_string()),
            Cell::from(p.name.clone()),
            Cell::from(format!("{:.1}", p.cpu_usage))
                .style(Style::default().fg(cpu_color)),
            Cell::from(format!("{} MB", p.memory_mb)),
            Cell::from(p.status.clone()),
        ])
    });

    let table = Table::new(
        rows,
        [
            Constraint::Length(8),   // PID
            Constraint::Min(20),     // Name
            Constraint::Length(8),   // CPU
            Constraint::Length(12),  // Memory
            Constraint::Length(10),  // Status
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title("Processes"))
    .highlight_style(
        Style::default()
            .bg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    );

    let mut state = TableState::default();
    state.select(Some(app.selected));

    f.render_stateful_widget(table, area, &mut state);
}
\`\`\`

### Scrolling
\`\`\`rust
impl App {
    pub fn scroll_up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn scroll_down(&mut self) {
        if !self.processes.is_empty() {
            self.selected = (self.selected + 1).min(self.processes.len() - 1);
        }
    }

    pub fn page_up(&mut self, page_size: usize) {
        self.selected = self.selected.saturating_sub(page_size);
    }

    pub fn page_down(&mut self, page_size: usize) {
        if !self.processes.is_empty() {
            self.selected = (self.selected + page_size).min(self.processes.len() - 1);
        }
    }

    pub fn selected_process(&self) -> Option<&ProcessInfo> {
        self.processes.get(self.selected)
    }
}
\`\`\``
					},
					{
						title: 'Add keyboard navigation',
						description: 'Handle arrow keys, page up/down, etc.',
						details: `## Keyboard Handling

### Key Bindings
\`\`\`rust
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

impl App {
    pub fn handle_key(&mut self, key: KeyEvent) {
        // Handle search mode first
        if self.search.active {
            self.search.handle_key(key.code);
            self.filter.query = self.search.input.clone();
            self.update_processes();
            return;
        }

        match key.code {
            // Navigation
            KeyCode::Up | KeyCode::Char('k') => self.scroll_up(),
            KeyCode::Down | KeyCode::Char('j') => self.scroll_down(),
            KeyCode::PageUp => self.page_up(20),
            KeyCode::PageDown => self.page_down(20),
            KeyCode::Home | KeyCode::Char('g') => self.selected = 0,
            KeyCode::End | KeyCode::Char('G') => {
                self.selected = self.processes.len().saturating_sub(1);
            }

            // Sorting
            KeyCode::Char('p') => self.sort.toggle(SortColumn::Pid),
            KeyCode::Char('n') => self.sort.toggle(SortColumn::Name),
            KeyCode::Char('c') => self.sort.toggle(SortColumn::Cpu),
            KeyCode::Char('m') => self.sort.toggle(SortColumn::Memory),
            KeyCode::Char('s') => self.sort.toggle(SortColumn::Status),

            // Search
            KeyCode::Char('/') => self.search.active = true,

            // Actions
            KeyCode::Char('K') => self.kill_selected(),
            KeyCode::Enter => self.show_details = true,
            KeyCode::Escape => self.show_details = false,

            // Quit
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }

            _ => {}
        }

        self.update_processes();
    }
}
\`\`\`

### Help Overlay
\`\`\`rust
pub fn render_help(f: &mut Frame) {
    let help_text = vec![
        ("Navigation", vec![
            ("↑/k", "Move up"),
            ("↓/j", "Move down"),
            ("PgUp/PgDn", "Page up/down"),
            ("Home/End", "First/last"),
        ]),
        ("Sorting", vec![
            ("p", "Sort by PID"),
            ("n", "Sort by name"),
            ("c", "Sort by CPU"),
            ("m", "Sort by memory"),
        ]),
        ("Actions", vec![
            ("/", "Search"),
            ("Enter", "Details"),
            ("K", "Kill process"),
            ("q", "Quit"),
        ]),
    ];

    // Render as popup
    // ...
}
\`\`\``
					},
					{
						title: 'Add process details view',
						description: 'Show detailed info for selected process',
						details: `## Process Details Panel

### Detail View
\`\`\`rust
use ratatui::widgets::{Paragraph, Wrap};

pub fn render_details(f: &mut Frame, process: &ProcessInfo, area: Rect) {
    let details = vec![
        format!("PID: {}", process.pid),
        format!("Name: {}", process.name),
        format!("Status: {}", process.status),
        format!("CPU Usage: {:.2}%", process.cpu_usage),
        format!("Memory: {} MB", process.memory_mb),
        format!("User: {}", process.user.as_deref().unwrap_or("unknown")),
        String::new(),
        "Command:".to_string(),
        process.command.join(" "),
    ];

    let text = details.join("\\n");

    let paragraph = Paragraph::new(text)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(format!("Process {} Details", process.pid)))
        .wrap(Wrap { trim: false });

    f.render_widget(paragraph, area);
}
\`\`\`

### Split Layout
\`\`\`rust
pub fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header/stats
            Constraint::Min(10),    // Main content
            Constraint::Length(1),  // Status bar
        ])
        .split(f.size());

    render_header(f, app, chunks[0]);

    if app.show_details {
        let main_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(60),
                Constraint::Percentage(40),
            ])
            .split(chunks[1]);

        render_process_table(f, app, main_chunks[0]);

        if let Some(process) = app.selected_process() {
            render_details(f, process, main_chunks[1]);
        }
    } else {
        render_process_table(f, app, chunks[1]);
    }

    render_status_bar(f, app, chunks[2]);
}
\`\`\`

### CPU/Memory Graphs
\`\`\`rust
use ratatui::widgets::Sparkline;

pub fn render_cpu_graph(f: &mut Frame, history: &[f32], area: Rect) {
    let data: Vec<u64> = history.iter()
        .map(|&v| (v * 100.0) as u64)
        .collect();

    let sparkline = Sparkline::default()
        .block(Block::default().title("CPU History"))
        .data(&data)
        .max(100)
        .style(Style::default().fg(Color::Cyan));

    f.render_widget(sparkline, area);
}

// Track history
impl App {
    pub fn record_stats(&mut self) {
        let avg_cpu: f32 = self.processes.iter()
            .map(|p| p.cpu_usage)
            .sum::<f32>() / self.processes.len() as f32;

        self.cpu_history.push(avg_cpu);
        if self.cpu_history.len() > 60 {
            self.cpu_history.remove(0);
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
		name: 'AI Model Opportunities',
		description: 'Identify, evaluate, and capitalize on AI/ML business opportunities. Learn to spot where AI adds value and build products around it.',
		language: 'Python',
		color: 'purple',
		skills: 'market analysis, product thinking, API integration, rapid prototyping, business modeling',
		startHint: 'Find a repetitive task you do daily and prototype an AI solution',
		difficulty: 'intermediate',
		estimatedWeeks: 4,
		modules: [
			{
				name: 'Opportunity Recognition',
				description: 'Learn to identify where AI creates value',
				tasks: [
					{ title: 'Study 10 successful AI products and document their value proposition', description: 'Analyze products like GitHub Copilot, Midjourney, Jasper - what problem do they solve?' },
					{ title: 'Map the AI landscape: foundation models, fine-tuned models, agents', description: 'Understand the different layers of AI products and where value is captured' },
					{ title: 'Identify 5 workflows in your domain that could be AI-augmented', description: 'Document current pain points, time spent, and potential AI solutions' },
					{ title: 'Research API costs and calculate unit economics for each idea', description: 'Price out OpenAI/Anthropic/local model costs per 1000 users' }
				]
			},
			{
				name: 'Rapid Prototyping',
				description: 'Build quick proofs of concept',
				tasks: [
					{ title: 'Build a CLI tool that uses an LLM API for your top idea', description: 'Focus on the core value prop - no UI needed yet' },
					{ title: 'Add prompt engineering: system prompts, few-shot examples, output parsing', description: 'Iterate on prompts to improve reliability and output quality' },
					{ title: 'Implement structured output with JSON mode or function calling', description: 'Make the AI output parseable and actionable data' },
					{ title: 'Add error handling, retries, and fallbacks', description: 'Handle rate limits, API failures, and malformed responses gracefully' }
				]
			},
			{
				name: 'Product Validation',
				description: 'Test your AI product with real users',
				tasks: [
					{ title: 'Build a simple web UI (Streamlit/Gradio) for your prototype', description: 'Create shareable interface for user testing' },
					{ title: 'Get 5 people to use it and document their feedback', description: 'Watch them use it, note confusion points and feature requests' },
					{ title: 'Measure key metrics: task completion rate, time saved, satisfaction', description: 'Quantify the value your tool provides' },
					{ title: 'Iterate based on feedback - 3 improvement cycles minimum', description: 'Ship improvements, re-test, repeat' }
				]
			},
			{
				name: 'Scaling Considerations',
				description: 'Prepare for growth',
				tasks: [
					{ title: 'Implement caching to reduce API costs', description: 'Cache common queries, use semantic similarity for cache hits' },
					{ title: 'Add usage tracking and analytics', description: 'Track which features are used, monitor costs per user' },
					{ title: 'Evaluate local/fine-tuned models vs API tradeoffs', description: 'When does running your own model make sense?' },
					{ title: 'Document your learnings: what worked, what failed, why', description: 'Create a playbook for your next AI product' }
				]
			}
		]
	},
	{
		name: 'Deep Learning Guide Part 2',
		description: 'Advanced deep learning: architectures beyond basics, optimization techniques, and production considerations.',
		language: 'Python',
		color: 'blue',
		skills: 'PyTorch, advanced architectures, optimization, distributed training, model analysis',
		startHint: 'Implement a ResNet from scratch before moving to advanced architectures',
		difficulty: 'advanced',
		estimatedWeeks: 6,
		modules: [
			{
				name: 'Advanced Architectures',
				description: 'Master modern neural network designs',
				tasks: [
					{ title: 'Implement ResNet with skip connections from scratch', description: 'Understand why skip connections enable deeper networks' },
					{ title: 'Build a U-Net for image segmentation', description: 'Learn encoder-decoder architectures with skip connections' },
					{ title: 'Implement attention mechanisms in CNNs (CBAM, SE-Net)', description: 'Add channel and spatial attention to improve feature learning' },
					{ title: 'Build a Vision Transformer (ViT) from the paper', description: 'Patch embeddings, position encodings, transformer encoder for images' }
				]
			},
			{
				name: 'Optimization Deep Dive',
				description: 'Master training dynamics',
				tasks: [
					{ title: 'Implement and compare optimizers: SGD, Adam, AdamW, LAMB', description: 'Understand momentum, adaptive learning rates, weight decay' },
					{ title: 'Implement learning rate schedules: warmup, cosine annealing, OneCycleLR', description: 'Experiment with different schedules on the same model' },
					{ title: 'Study gradient flow: implement gradient clipping, gradient checkpointing', description: 'Handle exploding gradients and memory constraints' },
					{ title: 'Implement mixed precision training (FP16/BF16)', description: 'Use torch.cuda.amp for faster training with lower memory' }
				]
			},
			{
				name: 'Regularization & Generalization',
				description: 'Prevent overfitting and improve robustness',
				tasks: [
					{ title: 'Implement data augmentation pipelines (albumentations, torchvision)', description: 'Random crops, flips, color jitter, mixup, cutout' },
					{ title: 'Compare dropout, droppath, and stochastic depth', description: 'Implement each and measure impact on generalization' },
					{ title: 'Implement label smoothing and knowledge distillation', description: 'Soft targets for better generalization' },
					{ title: 'Study the lottery ticket hypothesis - implement magnitude pruning', description: 'Find sparse subnetworks that train as well as dense networks' }
				]
			},
			{
				name: 'Distributed Training',
				description: 'Scale to multiple GPUs and machines',
				tasks: [
					{ title: 'Implement DataParallel and understand its limitations', description: 'Simple multi-GPU but with GIL and memory imbalance issues' },
					{ title: 'Convert to DistributedDataParallel (DDP)', description: 'Proper multi-GPU training with gradient synchronization' },
					{ title: 'Implement gradient accumulation for large effective batch sizes', description: 'Simulate larger batches when GPU memory is limited' },
					{ title: 'Profile training: identify bottlenecks in data loading, compute, communication', description: 'Use PyTorch profiler and tensorboard to optimize throughput' }
				]
			},
			{
				name: 'Model Analysis',
				description: 'Understand what your models learn',
				tasks: [
					{ title: 'Implement GradCAM for CNN visualization', description: 'See which image regions drive predictions' },
					{ title: 'Analyze learned representations with t-SNE and UMAP', description: 'Visualize embedding spaces and cluster structure' },
					{ title: 'Implement influence functions to find training examples affecting predictions', description: 'Debug model behavior by tracing to training data' },
					{ title: 'Build a model debugging dashboard with Weights & Biases', description: 'Track metrics, visualizations, and artifacts across experiments' }
				]
			}
		]
	},
	{
		name: 'Ethical Hacking & Security',
		description: 'Learn offensive security to build better defenses. Understand vulnerabilities, exploitation, and secure coding.',
		language: 'Python',
		color: 'rose',
		skills: 'network protocols, reverse engineering, vulnerability analysis, secure coding, CTF skills',
		startHint: 'Set up a vulnerable VM (DVWA or HackTheBox) and complete your first CTF challenge',
		difficulty: 'advanced',
		estimatedWeeks: 8,
		modules: [
			{
				name: 'Fundamentals',
				description: 'Core security concepts and setup',
				tasks: [
					{ title: 'Set up Kali Linux VM and essential tools', description: 'Configure virtualization, networking, and core toolset' },
					{ title: 'Study the OSI model and common protocols (TCP/IP, HTTP, DNS)', description: 'Understand how data flows across networks' },
					{ title: 'Learn to use Wireshark for packet analysis', description: 'Capture and analyze network traffic' },
					{ title: 'Complete 5 beginner CTF challenges on HackTheBox or TryHackMe', description: 'Get hands-on with basic exploitation techniques' }
				]
			},
			{
				name: 'Web Application Security',
				description: 'OWASP Top 10 and beyond',
				tasks: [
					{ title: 'Exploit SQL injection vulnerabilities (UNION, blind, time-based)', description: 'Extract data from databases through injection' },
					{ title: 'Exploit XSS vulnerabilities (reflected, stored, DOM-based)', description: 'Inject and execute JavaScript in browsers' },
					{ title: 'Exploit CSRF, SSRF, and authentication bypasses', description: 'Forge requests and bypass security controls' },
					{ title: 'Use Burp Suite for web app testing', description: 'Intercept, modify, and replay HTTP requests' },
					{ title: 'Build a vulnerable app and then secure it', description: 'Implement proper input validation, parameterized queries, CSP' }
				]
			},
			{
				name: 'Network Security',
				description: 'Network-level attacks and defenses',
				tasks: [
					{ title: 'Perform network reconnaissance with nmap', description: 'Discover hosts, open ports, and running services' },
					{ title: 'Exploit common service misconfigurations (SSH, FTP, SMB)', description: 'Identify and exploit weak configurations' },
					{ title: 'Implement ARP spoofing and understand MITM attacks', description: 'Intercept traffic on local networks' },
					{ title: 'Set up and configure a firewall (iptables/nftables)', description: 'Implement network-level defenses' }
				]
			},
			{
				name: 'Binary Exploitation Basics',
				description: 'Low-level vulnerability exploitation',
				tasks: [
					{ title: 'Understand memory layout: stack, heap, text, data segments', description: 'Learn how programs use memory' },
					{ title: 'Exploit basic buffer overflows on disabled protections', description: 'Overwrite return addresses to control execution' },
					{ title: 'Understand and bypass stack canaries', description: 'Learn how stack protection works' },
					{ title: 'Introduction to ROP (Return Oriented Programming)', description: 'Chain gadgets to bypass NX protection' },
					{ title: 'Use GDB and pwntools for exploit development', description: 'Debug binaries and automate exploitation' }
				]
			},
			{
				name: 'Secure Coding Practices',
				description: 'Build security in from the start',
				tasks: [
					{ title: 'Implement secure authentication (password hashing, MFA)', description: 'bcrypt/argon2, TOTP, WebAuthn basics' },
					{ title: 'Implement secure session management', description: 'Secure cookies, token rotation, session fixation prevention' },
					{ title: 'Implement input validation and output encoding', description: 'Prevent injection attacks at the code level' },
					{ title: 'Set up security headers and CSP', description: 'Configure HTTP headers for defense in depth' },
					{ title: 'Implement security logging and monitoring', description: 'Detect and respond to security events' }
				]
			}
		]
	},
	{
		name: 'Learn Deep Learning Without Courses',
		description: 'Self-directed deep learning education through papers, textbooks, and hands-on implementation.',
		language: 'Python',
		color: 'emerald',
		skills: 'paper reading, mathematical foundations, PyTorch, experiment design, scientific thinking',
		startHint: 'Start with 3Blue1Brown neural network videos, then implement a basic MLP',
		difficulty: 'intermediate',
		estimatedWeeks: 12,
		modules: [
			{
				name: 'Mathematical Foundations',
				description: 'Build the math intuition needed for deep learning',
				tasks: [
					{ title: 'Review linear algebra: vectors, matrices, eigenvalues (3Blue1Brown)', description: 'Watch Essence of Linear Algebra series and do practice problems' },
					{ title: 'Review calculus: derivatives, chain rule, gradients', description: 'Understand how gradients flow through computation graphs' },
					{ title: 'Study probability basics: distributions, Bayes theorem, expectation', description: 'Foundation for understanding loss functions and generative models' },
					{ title: 'Implement backpropagation from scratch in NumPy', description: 'Build a simple neural network without frameworks to understand gradients' }
				]
			},
			{
				name: 'Neural Network Fundamentals',
				description: 'Core concepts through implementation',
				tasks: [
					{ title: 'Read and implement the perceptron algorithm', description: 'Understand the simplest neural unit' },
					{ title: 'Implement MLPs for MNIST classification', description: 'Multi-layer networks, activation functions, softmax' },
					{ title: 'Study and implement different loss functions', description: 'MSE, cross-entropy, and when to use each' },
					{ title: 'Implement batch normalization from the paper', description: 'Read Ioffe & Szegedy 2015, implement, verify on a task' },
					{ title: 'Study the vanishing gradient problem and solutions', description: 'ReLU, initialization schemes, skip connections' }
				]
			},
			{
				name: 'Convolutional Networks',
				description: 'Learn CNNs through classic papers',
				tasks: [
					{ title: 'Read and implement LeNet-5 (LeCun 1998)', description: 'The original CNN architecture for digit recognition' },
					{ title: 'Read AlexNet paper and understand its innovations', description: 'ReLU, dropout, data augmentation, GPU training' },
					{ title: 'Implement VGG and understand depth vs width tradeoffs', description: 'Simple architecture, powerful features' },
					{ title: 'Read and implement ResNet (He et al. 2015)', description: 'Skip connections enabling very deep networks' },
					{ title: 'Train a CNN on CIFAR-10 and achieve >90% accuracy', description: 'Apply everything learned to a real benchmark' }
				]
			},
			{
				name: 'Sequence Models',
				description: 'RNNs, LSTMs, and the path to attention',
				tasks: [
					{ title: 'Implement vanilla RNN and understand its limitations', description: 'Vanishing gradients in long sequences' },
					{ title: 'Read and implement LSTM (Hochreiter & Schmidhuber 1997)', description: 'Gates, cell state, and long-term memory' },
					{ title: 'Implement character-level language model with LSTM', description: 'Generate text character by character' },
					{ title: 'Read "Attention Is All You Need" paper', description: 'Understand self-attention and why it replaced RNNs' },
					{ title: 'Implement multi-head attention from scratch', description: 'Queries, keys, values, and scaled dot-product attention' }
				]
			},
			{
				name: 'Modern Deep Learning',
				description: 'Current techniques and best practices',
				tasks: [
					{ title: 'Study transfer learning and fine-tuning strategies', description: 'When to freeze layers, learning rate schedules for fine-tuning' },
					{ title: 'Implement a training pipeline with proper validation', description: 'Train/val/test splits, early stopping, model selection' },
					{ title: 'Read and summarize 5 papers from your area of interest', description: 'Practice extracting key ideas and contributions' },
					{ title: 'Reproduce results from a recent paper', description: 'Full implementation and verification of claimed results' },
					{ title: 'Write up your learning journey as a blog post or notes', description: 'Teaching others solidifies your understanding' }
				]
			}
		]
	},
	{
		name: 'ML Pipeline Complete Guide',
		description: 'End-to-end machine learning: from raw data to deployed model with monitoring and iteration.',
		language: 'Python',
		color: 'amber',
		skills: 'data engineering, feature engineering, MLOps, deployment, monitoring, experimentation',
		startHint: 'Start with a simple dataset and build the full pipeline before optimizing any part',
		difficulty: 'intermediate',
		estimatedWeeks: 6,
		modules: [
			{
				name: 'Data Engineering',
				description: 'Build reliable data pipelines',
				tasks: [
					{ title: 'Set up data versioning with DVC', description: 'Track data changes alongside code changes' },
					{ title: 'Build data validation checks (great_expectations or pandera)', description: 'Catch data quality issues before they break models' },
					{ title: 'Implement data preprocessing pipeline with clear stages', description: 'Cleaning, transformation, feature extraction as separate steps' },
					{ title: 'Handle data imbalance: undersampling, oversampling, SMOTE', description: 'Strategies for imbalanced classification problems' },
					{ title: 'Create train/val/test splits with proper stratification', description: 'Avoid data leakage and ensure representative splits' }
				]
			},
			{
				name: 'Feature Engineering',
				description: 'Create predictive features',
				tasks: [
					{ title: 'Implement numerical feature transformations', description: 'Scaling, normalization, log transforms, binning' },
					{ title: 'Implement categorical encoding strategies', description: 'One-hot, target encoding, embeddings for high cardinality' },
					{ title: 'Create time-based features for temporal data', description: 'Lags, rolling windows, seasonal decomposition' },
					{ title: 'Implement feature selection methods', description: 'Correlation analysis, mutual information, recursive feature elimination' },
					{ title: 'Build a feature store pattern for reusable features', description: 'Centralize feature computation for training and serving' }
				]
			},
			{
				name: 'Model Development',
				description: 'Train and evaluate models systematically',
				tasks: [
					{ title: 'Set up experiment tracking (MLflow or Weights & Biases)', description: 'Track hyperparameters, metrics, and artifacts' },
					{ title: 'Implement cross-validation with proper handling of time series', description: 'K-fold, stratified, time-series split' },
					{ title: 'Compare multiple model types on your problem', description: 'Linear, tree-based, neural network baselines' },
					{ title: 'Implement hyperparameter tuning (Optuna or Ray Tune)', description: 'Bayesian optimization, early stopping, parallel trials' },
					{ title: 'Analyze model errors and iterate on features/model', description: 'Error analysis, confusion matrices, calibration curves' }
				]
			},
			{
				name: 'Model Deployment',
				description: 'Serve models in production',
				tasks: [
					{ title: 'Package model for deployment (ONNX or TorchScript)', description: 'Export model to portable format' },
					{ title: 'Build REST API for model serving (FastAPI)', description: 'Endpoints for prediction, batch prediction, model info' },
					{ title: 'Containerize with Docker', description: 'Reproducible deployment environment' },
					{ title: 'Set up CI/CD for model deployment', description: 'Automated testing and deployment pipeline' },
					{ title: 'Implement A/B testing infrastructure', description: 'Compare model versions in production' }
				]
			},
			{
				name: 'Monitoring & Maintenance',
				description: 'Keep models healthy in production',
				tasks: [
					{ title: 'Implement prediction logging', description: 'Log inputs, outputs, latency for analysis' },
					{ title: 'Set up data drift detection', description: 'Monitor input distribution changes over time' },
					{ title: 'Implement model performance monitoring', description: 'Track accuracy metrics when ground truth is available' },
					{ title: 'Build alerting for model degradation', description: 'Automated alerts when performance drops' },
					{ title: 'Create retraining pipeline', description: 'Automated or triggered retraining on new data' }
				]
			}
		]
	},
	{
		name: 'Transformers & LLMs Deep Dive',
		description: 'Master transformer architecture and large language models from theory to implementation to deployment.',
		language: 'Python',
		color: 'blue',
		skills: 'attention mechanisms, tokenization, fine-tuning, inference optimization, prompt engineering',
		startHint: 'Implement scaled dot-product attention from scratch before using libraries',
		difficulty: 'advanced',
		estimatedWeeks: 8,
		modules: [
			{
				name: 'Transformer Architecture',
				description: 'Deep understanding of the transformer',
				tasks: [
					{ title: 'Implement scaled dot-product attention from scratch', description: 'Q, K, V matrices, scaling, softmax, weighted sum' },
					{ title: 'Implement multi-head attention', description: 'Parallel attention heads, concatenation, projection' },
					{ title: 'Build position encodings (sinusoidal and learned)', description: 'Understand why position information is needed' },
					{ title: 'Implement a full transformer encoder block', description: 'Attention, layer norm, feed-forward, residual connections' },
					{ title: 'Implement a transformer decoder with causal masking', description: 'Autoregressive generation, masked self-attention' }
				]
			},
			{
				name: 'Tokenization & Embeddings',
				description: 'How text becomes numbers',
				tasks: [
					{ title: 'Implement byte-pair encoding (BPE) from scratch', description: 'Build vocabulary through iterative merging' },
					{ title: 'Compare BPE, WordPiece, and SentencePiece', description: 'Understand tradeoffs between tokenization methods' },
					{ title: 'Study how different models tokenize the same text', description: 'GPT-2, BERT, LLaMA tokenizer differences' },
					{ title: 'Implement token embeddings with learned position encodings', description: 'The embedding layer that starts it all' },
					{ title: 'Understand tokenization edge cases and their impact', description: 'Numbers, code, multilingual, special tokens' }
				]
			},
			{
				name: 'Pre-training Objectives',
				description: 'How LLMs learn from text',
				tasks: [
					{ title: 'Implement causal language modeling (GPT-style)', description: 'Predict next token, cross-entropy loss' },
					{ title: 'Implement masked language modeling (BERT-style)', description: 'Random masking, bidirectional context' },
					{ title: 'Study span corruption (T5-style)', description: 'Understand encoder-decoder pre-training' },
					{ title: 'Implement a small language model and train on a corpus', description: 'Full training loop with proper evaluation' },
					{ title: 'Analyze scaling laws: how performance changes with size', description: 'Read Kaplan et al. and Hoffmann et al. papers' }
				]
			},
			{
				name: 'Fine-tuning Techniques',
				description: 'Adapt LLMs to specific tasks',
				tasks: [
					{ title: 'Implement full fine-tuning on a classification task', description: 'Add classification head, train all parameters' },
					{ title: 'Implement LoRA (Low-Rank Adaptation)', description: 'Add trainable low-rank matrices, freeze base model' },
					{ title: 'Implement QLoRA with quantization', description: 'Fine-tune quantized models efficiently' },
					{ title: 'Study and implement instruction tuning', description: 'Format data as instructions, train to follow them' },
					{ title: 'Implement RLHF concepts: reward modeling and PPO basics', description: 'Understand how ChatGPT-style training works' }
				]
			},
			{
				name: 'Inference Optimization',
				description: 'Make LLMs fast and efficient',
				tasks: [
					{ title: 'Implement KV-cache for efficient autoregressive generation', description: 'Cache key-value pairs to avoid recomputation' },
					{ title: 'Implement and compare sampling strategies', description: 'Greedy, top-k, top-p (nucleus), temperature' },
					{ title: 'Implement quantization (INT8, INT4) with bitsandbytes', description: 'Reduce memory and increase throughput' },
					{ title: 'Study and implement speculative decoding', description: 'Use small model to speed up large model generation' },
					{ title: 'Profile inference and identify bottlenecks', description: 'Memory bandwidth, compute, attention costs' }
				]
			},
			{
				name: 'Building with LLMs',
				description: 'Practical LLM applications',
				tasks: [
					{ title: 'Implement RAG (Retrieval Augmented Generation)', description: 'Vector store, retrieval, context injection' },
					{ title: 'Build a chat interface with conversation memory', description: 'Managing context window, summarization strategies' },
					{ title: 'Implement function calling / tool use', description: 'Parse structured output, execute functions, return results' },
					{ title: 'Build an agent with planning and execution', description: 'ReAct-style reasoning, multi-step task completion' },
					{ title: 'Implement guardrails and safety checks', description: 'Input validation, output filtering, content safety' }
				]
			}
		]
	},
	{
		name: 'Distributed Task Pipeline',
		description: 'A multi-service system where Go coordinates, Rust processes CPU-intensive tasks, and Python handles ML/data work.',
		language: 'Multi-Language',
		color: 'rose',
		skills: 'gRPC/HTTP, protocol design, serialization, cross-language error handling',
		startHint: 'Go coordinator with hardcoded task, one Rust worker that echoes back',
		difficulty: 'advanced',
		estimatedWeeks: 4,
		schedule: `## 4-Week Schedule

### Week 1: Go Coordinator
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Project Setup | Initialize Go module, define job/worker data structures |
| Day 2 | REST API | Implement job submission and status endpoints |
| Day 3 | Job Queue | Build in-memory job queue with priority support |
| Day 4 | Worker Registry | Implement worker registration and heartbeat |
| Day 5 | Job Routing | Route jobs to workers based on task type |
| Weekend | Testing | Integration tests, error scenarios |

### Week 2: Rust Worker
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Setup | Initialize Rust project, HTTP client for coordinator |
| Day 2 | Registration | Worker startup, registration, heartbeat loop |
| Day 3 | Job Fetching | Poll coordinator, claim and process jobs |
| Day 4 | CPU Tasks | Implement image processing or data transformation |
| Day 5 | Error Handling | Graceful failures, job retry logic |
| Weekend | Integration | Test Go + Rust together |

### Week 3: Python Worker + CLI
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Python Worker | Mirror Rust worker structure in Python |
| Day 2 | ML Tasks | Implement ML inference or data processing tasks |
| Day 3 | CLI Tool | Build CLI for job submission |
| Day 4 | Results Viewer | CLI commands to view job status and results |
| Day 5 | Progress Tracking | Real-time progress updates via polling |
| Weekend | Documentation | API docs, usage examples |

### Week 4: Production Features
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Persistence | Add job persistence to coordinator |
| Day 2 | Graceful Shutdown | Handle worker disconnects, reassign tasks |
| Day 3 | Metrics | Add basic metrics and logging |
| Day 4 | Load Testing | Stress test with many concurrent jobs |
| Day 5 | Polish | Error messages, edge cases, cleanup |
| Weekend | Final Testing | End-to-end scenarios |`,
		modules: [
			{
				name: 'Go Coordinator Service',
				description: 'Central service managing job queue and worker registration',
				tasks: [
					{ title: 'Design job and worker data structures', description: 'Define Job, Worker, TaskType structs' },
					{ title: 'Implement REST API for job submission', description: 'POST /jobs endpoint with validation' },
					{ title: 'Build in-memory job queue with priority', description: 'Thread-safe queue with mutex protection' },
					{ title: 'Create worker registration system', description: 'Workers register capabilities on startup' },
					{ title: 'Implement heartbeat monitoring', description: 'Detect dead workers, reassign their jobs' },
					{ title: 'Add job routing based on task type', description: 'Match jobs to workers by capability' },
					{ title: 'Build job status and results endpoints', description: 'GET /jobs/:id for status and results' },
					{ title: 'Add persistence layer', description: 'Save jobs to SQLite for crash recovery' }
				]
			},
			{
				name: 'Rust High-Performance Worker',
				description: 'Worker for CPU-intensive tasks like image processing',
				tasks: [
					{ title: 'Set up Rust project with reqwest and tokio', description: 'HTTP client and async runtime' },
					{ title: 'Implement worker registration on startup', description: 'POST to coordinator with capabilities' },
					{ title: 'Build heartbeat loop', description: 'Periodic pings to coordinator' },
					{ title: 'Create job polling and claiming', description: 'Fetch available jobs, mark as claimed' },
					{ title: 'Implement image processing task', description: 'Resize, convert, or apply filters' },
					{ title: 'Add data transformation task', description: 'Parse, transform, aggregate data files' },
					{ title: 'Handle errors and report failures', description: 'Structured error reporting to coordinator' },
					{ title: 'Implement graceful shutdown', description: 'Finish current job before exit' }
				]
			},
			{
				name: 'Python Worker + CLI',
				description: 'ML worker and command-line interface for users',
				tasks: [
					{ title: 'Create Python worker mirroring Rust structure', description: 'Same registration, heartbeat, polling' },
					{ title: 'Implement ML inference task', description: 'Run model prediction on input data' },
					{ title: 'Add data processing task', description: 'Pandas/NumPy data manipulation' },
					{ title: 'Build CLI for job submission', description: 'Click/Typer CLI with subcommands' },
					{ title: 'Add job status command', description: 'Check job progress and state' },
					{ title: 'Implement results retrieval', description: 'Download and display job outputs' },
					{ title: 'Add watch mode for progress', description: 'Poll and display real-time updates' },
					{ title: 'Create batch job submission', description: 'Submit multiple jobs from file' }
				]
			}
		]
	},
	{
		name: 'Polyglot Monitoring Stack',
		description: 'Build your own observability system with Rust agent, Go aggregation server, and Python dashboard.',
		language: 'Multi-Language',
		color: 'rose',
		skills: 'binary protocols, time-series data, network programming, data visualization',
		startHint: 'Rust agent sending one metric to Go server, Python script that prints it',
		difficulty: 'advanced',
		estimatedWeeks: 4,
		schedule: `## 4-Week Schedule

### Week 1: Rust Metrics Agent
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Setup | Initialize Rust project, sysinfo crate |
| Day 2 | CPU Metrics | Collect CPU usage per core |
| Day 3 | Memory Metrics | RAM usage, swap, buffers |
| Day 4 | Disk Metrics | Disk usage, I/O stats |
| Day 5 | Network Metrics | Bytes sent/received per interface |
| Weekend | Binary Protocol | Design efficient metric serialization |

### Week 2: Go Aggregation Server
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Setup | Initialize Go project, TCP listener |
| Day 2 | Protocol Parsing | Decode binary metrics from agents |
| Day 3 | Storage Design | In-memory time-series structure |
| Day 4 | Retention | Implement data retention policies |
| Day 5 | Query API | REST endpoints for metric queries |
| Weekend | Integration | Test Rust agent + Go server |

### Week 3: Python Dashboard
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | API Client | Query metrics from Go server |
| Day 2 | TUI Setup | Initialize Rich/Textual dashboard |
| Day 3 | Live Charts | ASCII/Unicode charts for metrics |
| Day 4 | Historical View | Query and display time ranges |
| Day 5 | Multi-Host | Display metrics from multiple agents |
| Weekend | Polish | Colors, layouts, responsiveness |

### Week 4: Alerting + Production
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Alert Rules | Define threshold-based alerts |
| Day 2 | Alert Engine | Python engine evaluating rules |
| Day 3 | Notifications | Send alerts via webhook/email |
| Day 4 | Agent Config | Configurable collection intervals |
| Day 5 | Server Persistence | Persist metrics to disk |
| Weekend | Load Testing | Many agents, long retention |`,
		modules: [
			{
				name: 'Rust Metrics Agent',
				description: 'Collect system metrics and send to aggregation server',
				tasks: [
					{ title: 'Set up project with sysinfo crate', description: 'Initialize system info collection' },
					{ title: 'Collect CPU metrics per core', description: 'Usage percentage, frequency, temperature' },
					{ title: 'Collect memory metrics', description: 'Total, used, available, swap usage' },
					{ title: 'Collect disk metrics', description: 'Usage per mount, read/write bytes' },
					{ title: 'Collect network metrics', description: 'Bytes/packets per interface' },
					{ title: 'Design binary protocol for metrics', description: 'Efficient serialization format' },
					{ title: 'Implement TCP client to server', description: 'Connect, send metrics, handle reconnects' },
					{ title: 'Add configurable collection interval', description: 'Config file or CLI args' }
				]
			},
			{
				name: 'Go Aggregation Server',
				description: 'Receive, store, and query time-series metrics',
				tasks: [
					{ title: 'Create TCP listener for agents', description: 'Accept connections, spawn handlers' },
					{ title: 'Parse binary metric protocol', description: 'Decode agent messages' },
					{ title: 'Design time-series storage structure', description: 'Efficient in-memory storage' },
					{ title: 'Implement data retention policies', description: 'Delete old data, downsample' },
					{ title: 'Build REST API for queries', description: 'Query by host, metric, time range' },
					{ title: 'Add aggregation functions', description: 'Min, max, avg, percentiles' },
					{ title: 'Implement metric persistence', description: 'Save to disk, load on startup' },
					{ title: 'Handle multiple agents', description: 'Track hosts, handle disconnects' }
				]
			},
			{
				name: 'Python Dashboard + Alerting',
				description: 'Visualize metrics and trigger alerts',
				tasks: [
					{ title: 'Build API client for Go server', description: 'Query metrics with time ranges' },
					{ title: 'Create TUI dashboard with Rich/Textual', description: 'Layout with panels for each metric type' },
					{ title: 'Implement live ASCII charts', description: 'Sparklines or bar charts for metrics' },
					{ title: 'Add historical data view', description: 'Select time range, display trends' },
					{ title: 'Support multiple hosts in dashboard', description: 'Switch between or compare hosts' },
					{ title: 'Define alerting rules format', description: 'YAML config for thresholds' },
					{ title: 'Build alert evaluation engine', description: 'Check rules against metrics' },
					{ title: 'Implement alert notifications', description: 'Webhook, email, or desktop alerts' }
				]
			}
		]
	},
	{
		name: 'Plugin-Based File Processor',
		description: 'Core Go engine with plugins in Rust (fast ops) and Python (complex transforms) communicating via stdin/stdout.',
		language: 'Multi-Language',
		color: 'rose',
		skills: 'process spawning, IPC, streaming data, plugin architecture',
		startHint: 'Go engine that calls a Rust plugin to hash a file',
		difficulty: 'advanced',
		estimatedWeeks: 4,
		schedule: `## 4-Week Schedule

### Week 1: Go Core Engine
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Setup | Initialize project, define plugin interface |
| Day 2 | Directory Watcher | Watch for new/changed files |
| Day 3 | Plugin Discovery | Scan plugin directory, read manifests |
| Day 4 | Process Spawning | Launch plugins as subprocesses |
| Day 5 | IPC Protocol | Design stdin/stdout JSON protocol |
| Weekend | Error Handling | Plugin crashes, timeouts |

### Week 2: Rust Plugins
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Plugin Template | Stdin/stdout loop, JSON parsing |
| Day 2 | Hash Plugin | Calculate file hashes (MD5, SHA256) |
| Day 3 | Compression Plugin | Gzip/zstd compression |
| Day 4 | Format Converter | Image format conversion |
| Day 5 | Streaming Support | Handle large files in chunks |
| Weekend | Testing | Test each plugin independently |

### Week 3: Python Plugins
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Plugin Template | Mirror Rust structure in Python |
| Day 2 | PDF Parser | Extract text from PDFs |
| Day 3 | Image OCR | Extract text from images |
| Day 4 | Data Extractor | Parse structured data from files |
| Day 5 | ML Classifier | Classify file contents |
| Weekend | Integration | Test with Go engine |

### Week 4: Advanced Features
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Plugin Chaining | Pipe output to next plugin |
| Day 2 | Parallel Processing | Process multiple files concurrently |
| Day 3 | Results Storage | Save processing results |
| Day 4 | CLI Interface | Commands to process, list, configure |
| Day 5 | Logging & Metrics | Track processing stats |
| Weekend | Documentation | Plugin development guide |`,
		modules: [
			{
				name: 'Go Core Engine',
				description: 'Watch directories and dispatch files to plugins',
				tasks: [
					{ title: 'Define plugin interface and manifest format', description: 'JSON manifest with capabilities' },
					{ title: 'Implement directory watcher with fsnotify', description: 'Detect new and changed files' },
					{ title: 'Build plugin discovery system', description: 'Scan directory, parse manifests' },
					{ title: 'Create process spawning for plugins', description: 'exec.Command with pipes' },
					{ title: 'Design IPC protocol over stdin/stdout', description: 'JSON messages with file data' },
					{ title: 'Handle plugin errors and crashes', description: 'Timeouts, restart policies' },
					{ title: 'Implement plugin chaining', description: 'Output of one feeds into next' },
					{ title: 'Add concurrent file processing', description: 'Worker pool for parallel execution' }
				]
			},
			{
				name: 'Rust Fast Plugins',
				description: 'High-performance operations: hashing, compression, conversion',
				tasks: [
					{ title: 'Create plugin template with stdin/stdout', description: 'JSON protocol, main loop' },
					{ title: 'Build file hashing plugin', description: 'MD5, SHA256, SHA512 support' },
					{ title: 'Build compression plugin', description: 'Gzip, zstd compression/decompression' },
					{ title: 'Build image format converter', description: 'PNG, JPEG, WebP conversion' },
					{ title: 'Implement streaming for large files', description: 'Process in chunks, report progress' },
					{ title: 'Add plugin manifest generation', description: 'Output manifest on --manifest flag' },
					{ title: 'Handle errors gracefully', description: 'Structured error responses' },
					{ title: 'Optimize for performance', description: 'Buffer sizes, memory usage' }
				]
			},
			{
				name: 'Python Complex Plugins',
				description: 'ML and data extraction operations',
				tasks: [
					{ title: 'Create plugin template mirroring Rust', description: 'Same protocol, different implementation' },
					{ title: 'Build PDF text extractor', description: 'Extract text using PyPDF2 or pdfplumber' },
					{ title: 'Build image OCR plugin', description: 'Extract text using Tesseract' },
					{ title: 'Build structured data extractor', description: 'Parse tables, forms from documents' },
					{ title: 'Build ML file classifier', description: 'Classify documents by content' },
					{ title: 'Handle large files with streaming', description: 'Process without loading full file' },
					{ title: 'Add detailed logging', description: 'Processing steps, timings' },
					{ title: 'Create plugin tests', description: 'Unit tests for each plugin' }
				]
			}
		]
	}
];

async function seed() {
	console.log('Seeding database with coding practice projects...');

	// Note: Don't clear data here - seed.ts handles that at the start of seed-all

	for (const project of projects) {
		// Insert path
		const pathResult = db.insert(schema.paths).values({
			name: project.name,
			description: project.description,
			color: project.color,
			language: project.language,
			skills: project.skills,
			startHint: project.startHint,
			difficulty: project.difficulty,
			estimatedWeeks: project.estimatedWeeks,
			schedule: (project as any).schedule || null
		}).returning().get();

		console.log(`Created path: ${project.name}`);

		// Insert modules and tasks
		for (let i = 0; i < project.modules.length; i++) {
			const mod = project.modules[i];
			const moduleResult = db.insert(schema.modules).values({
				pathId: pathResult.id,
				name: mod.name,
				description: mod.description,
				orderIndex: i
			}).returning().get();

			// Insert tasks
			for (let j = 0; j < mod.tasks.length; j++) {
				const task = mod.tasks[j] as { title: string; description: string; details?: string };
				db.insert(schema.tasks).values({
					moduleId: moduleResult.id,
					title: task.title,
					description: task.description,
					details: task.details || null,
					orderIndex: j,
					completed: false
				}).run();
			}
		}
	}

	console.log('Seeding complete!');
}

seed().catch(console.error);
