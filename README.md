# Quest Log

A personal learning tracker for coding practice projects and AI/ML learning paths. Track your progress through structured curricula with detailed task breakdowns, daily schedules, notes, and completion tracking.

## Features

- **Learning Paths**: Structured curricula for coding projects and AI/ML topics
- **Progress Tracking**: Visual progress bars and completion statistics
- **Task Details**: Rich markdown content with code examples, diagrams, and implementation guides
- **Daily Schedules**: Day-by-day breakdown of what to work on
- **Personal Notes**: Add your own notes, learnings, and links to each task
- **Activity Heatmap**: GitHub-style contribution graph showing daily progress
- **Multi-Language Support**: Projects in Python, Go, Rust, and multi-language combinations

## Tech Stack

- **Frontend**: SvelteKit 5 with Svelte 5 runes
- **Database**: SQLite with Drizzle ORM
- **Styling**: Tailwind CSS with custom theming
- **Markdown**: Rendered with `marked` library
- **Containerization**: Docker with development and production configurations

## Getting Started

### Prerequisites

- Node.js 18+
- npm or pnpm
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd quest-log

# Install dependencies
npm install

# Initialize the database and seed with learning paths
npm run db:push
npx tsx scripts/seed-projects.ts
npx tsx scripts/seed-ai-paths.ts
```

### Development

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Docker

```bash
# Production build
docker-compose up app

# Development with hot reload
docker-compose --profile dev up dev
```

The app will be available at [http://localhost:3000](http://localhost:3000).

## Project Structure

```
quest-log/
├── src/
│   ├── lib/
│   │   ├── components/      # Reusable Svelte components
│   │   └── server/
│   │       ├── db.ts        # Database connection
│   │       ├── schema.ts    # Drizzle schema
│   │       └── queries.ts   # Database queries
│   └── routes/
│       ├── +page.svelte     # Dashboard with activity heatmap
│       └── paths/
│           ├── +page.svelte         # Path listing
│           └── [id]/+page.svelte    # Path detail with tasks
├── scripts/
│   ├── seed-projects.ts     # Coding practice projects
│   └── seed-ai-paths.ts     # AI/ML learning paths
├── data/
│   └── quest-log.db         # SQLite database
├── Dockerfile               # Production Docker image
├── Dockerfile.dev           # Development Docker image
├── docker-compose.yml       # Docker Compose configuration
└── tailwind.config.js
```

## Learning Paths

### Coding Practice Projects

Hands-on projects to build without AI assistance:

#### Single Language Projects

| Project | Language | Difficulty | Duration |
|---------|----------|------------|----------|
| Terminal Task Scheduler | Python | Intermediate | 2 weeks |
| Log Aggregator & Analyzer | Python | Intermediate | 2 weeks |
| Git Repository Analyzer | Python | Intermediate | 2 weeks |
| Concurrent File Sync | Go | Intermediate | 2 weeks |
| HTTP Proxy with Caching | Go | Intermediate | 2 weeks |
| Simple Message Queue | Go | Advanced | 2 weeks |
| Markdown to HTML Compiler | Rust | Intermediate | 2 weeks |
| Simple Key-Value Store | Rust | Advanced | 2 weeks |
| Process Monitor TUI | Rust | Intermediate | 2 weeks |

#### Multi-Language Projects

| Project | Languages | Difficulty | Duration |
|---------|-----------|------------|----------|
| Distributed Task Pipeline | Go + Rust + Python | Advanced | 4 weeks |
| Polyglot Monitoring Stack | Rust + Go + Python | Advanced | 4 weeks |
| Plugin-Based File Processor | Go + Rust + Python | Advanced | 4 weeks |

### AI/ML Learning Paths

Structured curricula for AI/ML education:

| Path | Focus | Duration |
|------|-------|----------|
| AI Model Opportunities | Business + Prototyping | 4 weeks |
| Transformers & LLMs Deep Dive | Architecture + Implementation | 8 weeks |
| Ethical Hacking & Security | Offensive Security | 12 weeks |
| Learn Deep Learning Without Courses | Self-directed Learning | 12 weeks |
| Deep Learning Advanced Topics | Advanced Architectures | 8 weeks |
| ML Pipeline Complete Guide | MLOps + Production | 6 weeks |

## Database Schema

```sql
-- Learning paths/projects
paths (
  id, name, description, icon, color,
  language, skills, difficulty, estimated_weeks,
  schedule, start_hint, created_at
)

-- Modules within a path
modules (
  id, path_id, name, description,
  order_index, target_date, created_at
)

-- Individual tasks
tasks (
  id, module_id, title, description, details,
  order_index, completed, completed_at, notes, created_at
)

-- Daily activity tracking
activity (id, date, tasks_completed)
```

## Practice Rules

1. **No AI assistance** - Write every line yourself
2. **Debug manually** - Use print statements and debuggers
3. **Read documentation** - Spend time in official docs
4. **Start small** - Get a minimal version working, then iterate
5. **Write tests** - Even simple ones help you understand the code
6. **Track your time** - Note how long features take to build

## License

MIT
