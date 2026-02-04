import Database from 'better-sqlite3';
import { join } from 'path';

const db = new Database(join(process.cwd(), 'data', 'quest-log.db'));

// Systems Programming - Shell, Container Runtime, Debugger
const paths = [
  {
    name: 'Build Your Own Shell',
    description: 'Create a Unix shell from scratch with job control, piping, and scripting support',
    icon: 'terminal',
    color: 'gray',
    language: 'C, Rust, Go',
    skills: 'System calls, Process management, Signal handling, Parsing',
    difficulty: 'intermediate',
    estimated_weeks: 6,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | REPL basics | Read-eval-print loop |
| 1 | 2 | Command parsing | Tokenizer |
| 1 | 3 | Command execution | fork/exec |
| 1 | 4 | Built-in commands | cd, exit, pwd |
| 1 | 5 | Environment vars | export, env |
| 2 | 1 | Pipes | Single pipe |
| 2 | 2 | Pipe chains | Multiple pipes |
| 2 | 3 | Redirection | stdin/stdout |
| 2 | 4 | Append redirect | >> operator |
| 2 | 5 | Here documents | << operator |
| 3 | 1 | Background jobs | & operator |
| 3 | 2 | Job control | jobs, fg, bg |
| 3 | 3 | Signal handling | Ctrl+C, Ctrl+Z |
| 3 | 4 | Process groups | setpgid |
| 3 | 5 | Terminal control | tcsetpgrp |
| 4 | 1 | Globbing | *, ?, [] |
| 4 | 2 | Tilde expansion | ~ |
| 4 | 3 | Variable expansion | \$VAR |
| 4 | 4 | Command substitution | \$(cmd) |
| 4 | 5 | Quoting | Single/double |
| 5 | 1 | Control structures | if/then/else |
| 5 | 2 | Loops | for, while |
| 5 | 3 | Functions | Function definitions |
| 5 | 4 | Script execution | shebang |
| 5 | 5 | Source command | . and source |
| 6 | 1 | History | Command history |
| 6 | 2 | Line editing | Arrow keys |
| 6 | 3 | Tab completion | Filename completion |
| 6 | 4 | Aliases | Alias support |
| 6 | 5 | RC files | ~/.shellrc |`,
    modules: [
      {
        name: 'Core Shell',
        description: 'Basic shell with command execution',
        tasks: [
          {
            title: 'Shell in C',
            description: 'Implement a Unix shell with job control in C',
            details: `# Unix Shell Implementation in C

\`\`\`c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>

#define MAX_LINE 1024
#define MAX_ARGS 128
#define MAX_JOBS 64

// Token types
typedef enum {
    TOK_WORD,
    TOK_PIPE,
    TOK_REDIR_IN,
    TOK_REDIR_OUT,
    TOK_REDIR_APPEND,
    TOK_BACKGROUND,
    TOK_AND,
    TOK_OR,
    TOK_SEMICOLON,
    TOK_EOF
} TokenType;

typedef struct {
    TokenType type;
    char *value;
} Token;

// Command structure
typedef struct Command {
    char **args;
    int argc;
    char *input_file;
    char *output_file;
    int append_output;
    struct Command *next;  // For pipelines
} Command;

// Job structure
typedef enum { RUNNING, STOPPED, DONE } JobState;

typedef struct {
    int id;
    pid_t pgid;
    char *command;
    JobState state;
    int foreground;
} Job;

// Global state
Job jobs[MAX_JOBS];
int job_count = 0;
pid_t shell_pgid;
struct termios shell_tmodes;
int shell_terminal;
int shell_interactive;

// Signal handlers
void sigchld_handler(int sig) {
    int status;
    pid_t pid;

    while ((pid = waitpid(-1, &status, WNOHANG | WUNTRACED)) > 0) {
        // Update job status
        for (int i = 0; i < job_count; i++) {
            if (jobs[i].pgid == pid) {
                if (WIFEXITED(status) || WIFSIGNALED(status)) {
                    jobs[i].state = DONE;
                } else if (WIFSTOPPED(status)) {
                    jobs[i].state = STOPPED;
                }
                break;
            }
        }
    }
}

void init_shell() {
    shell_terminal = STDIN_FILENO;
    shell_interactive = isatty(shell_terminal);

    if (shell_interactive) {
        // Wait until we're in the foreground
        while (tcgetpgrp(shell_terminal) != (shell_pgid = getpgrp())) {
            kill(-shell_pgid, SIGTTIN);
        }

        // Ignore interactive signals
        signal(SIGINT, SIG_IGN);
        signal(SIGQUIT, SIG_IGN);
        signal(SIGTSTP, SIG_IGN);
        signal(SIGTTIN, SIG_IGN);
        signal(SIGTTOU, SIG_IGN);
        signal(SIGCHLD, sigchld_handler);

        // Put ourselves in our own process group
        shell_pgid = getpid();
        if (setpgid(shell_pgid, shell_pgid) < 0) {
            perror("setpgid");
            exit(1);
        }

        // Take control of terminal
        tcsetpgrp(shell_terminal, shell_pgid);
        tcgetattr(shell_terminal, &shell_tmodes);
    }
}

// Tokenizer
Token *tokenize(char *line, int *count) {
    Token *tokens = malloc(sizeof(Token) * MAX_ARGS);
    *count = 0;

    char *p = line;
    while (*p) {
        // Skip whitespace
        while (*p == ' ' || *p == '\\t') p++;
        if (!*p) break;

        Token *tok = &tokens[(*count)++];

        // Check operators
        if (*p == '|') {
            if (*(p+1) == '|') {
                tok->type = TOK_OR;
                tok->value = strdup("||");
                p += 2;
            } else {
                tok->type = TOK_PIPE;
                tok->value = strdup("|");
                p++;
            }
        } else if (*p == '<') {
            tok->type = TOK_REDIR_IN;
            tok->value = strdup("<");
            p++;
        } else if (*p == '>') {
            if (*(p+1) == '>') {
                tok->type = TOK_REDIR_APPEND;
                tok->value = strdup(">>");
                p += 2;
            } else {
                tok->type = TOK_REDIR_OUT;
                tok->value = strdup(">");
                p++;
            }
        } else if (*p == '&') {
            if (*(p+1) == '&') {
                tok->type = TOK_AND;
                tok->value = strdup("&&");
                p += 2;
            } else {
                tok->type = TOK_BACKGROUND;
                tok->value = strdup("&");
                p++;
            }
        } else if (*p == ';') {
            tok->type = TOK_SEMICOLON;
            tok->value = strdup(";");
            p++;
        } else {
            // Word token
            tok->type = TOK_WORD;
            char *start = p;
            int in_quotes = 0;
            char quote_char = 0;

            while (*p && (in_quotes || (*p != ' ' && *p != '\\t' &&
                   *p != '|' && *p != '<' && *p != '>' &&
                   *p != '&' && *p != ';'))) {
                if (*p == '\\'' || *p == '"') {
                    if (!in_quotes) {
                        in_quotes = 1;
                        quote_char = *p;
                    } else if (*p == quote_char) {
                        in_quotes = 0;
                    }
                }
                p++;
            }

            int len = p - start;
            tok->value = malloc(len + 1);
            strncpy(tok->value, start, len);
            tok->value[len] = '\\0';
        }
    }

    tokens[*count].type = TOK_EOF;
    tokens[*count].value = NULL;

    return tokens;
}

// Parser
Command *parse_command(Token *tokens, int *pos, int count) {
    Command *cmd = calloc(1, sizeof(Command));
    cmd->args = malloc(sizeof(char*) * MAX_ARGS);
    cmd->argc = 0;

    while (*pos < count && tokens[*pos].type != TOK_EOF) {
        Token *tok = &tokens[*pos];

        if (tok->type == TOK_WORD) {
            cmd->args[cmd->argc++] = strdup(tok->value);
            (*pos)++;
        } else if (tok->type == TOK_REDIR_IN) {
            (*pos)++;
            if (*pos < count && tokens[*pos].type == TOK_WORD) {
                cmd->input_file = strdup(tokens[*pos].value);
                (*pos)++;
            }
        } else if (tok->type == TOK_REDIR_OUT || tok->type == TOK_REDIR_APPEND) {
            cmd->append_output = (tok->type == TOK_REDIR_APPEND);
            (*pos)++;
            if (*pos < count && tokens[*pos].type == TOK_WORD) {
                cmd->output_file = strdup(tokens[*pos].value);
                (*pos)++;
            }
        } else if (tok->type == TOK_PIPE) {
            (*pos)++;
            cmd->next = parse_command(tokens, pos, count);
            break;
        } else {
            break;
        }
    }

    cmd->args[cmd->argc] = NULL;
    return cmd;
}

// Built-in commands
int builtin_cd(char **args) {
    char *dir = args[1] ? args[1] : getenv("HOME");
    if (chdir(dir) != 0) {
        perror("cd");
        return 1;
    }
    return 0;
}

int builtin_exit(char **args) {
    exit(args[1] ? atoi(args[1]) : 0);
}

int builtin_export(char **args) {
    if (!args[1]) {
        // Print all environment variables
        extern char **environ;
        for (char **env = environ; *env; env++) {
            printf("%s\\n", *env);
        }
        return 0;
    }

    char *eq = strchr(args[1], '=');
    if (eq) {
        *eq = '\\0';
        setenv(args[1], eq + 1, 1);
    }
    return 0;
}

int builtin_jobs(char **args) {
    for (int i = 0; i < job_count; i++) {
        if (jobs[i].state != DONE) {
            const char *state_str = jobs[i].state == RUNNING ? "Running" : "Stopped";
            printf("[%d] %s\\t%s\\n", jobs[i].id, state_str, jobs[i].command);
        }
    }
    return 0;
}

int builtin_fg(char **args) {
    int job_id = args[1] ? atoi(args[1]) : job_count;

    for (int i = 0; i < job_count; i++) {
        if (jobs[i].id == job_id && jobs[i].state != DONE) {
            // Put job in foreground
            tcsetpgrp(shell_terminal, jobs[i].pgid);

            // Continue if stopped
            if (jobs[i].state == STOPPED) {
                kill(-jobs[i].pgid, SIGCONT);
                jobs[i].state = RUNNING;
            }

            // Wait for job
            int status;
            waitpid(-jobs[i].pgid, &status, WUNTRACED);

            // Take back terminal
            tcsetpgrp(shell_terminal, shell_pgid);

            if (WIFSTOPPED(status)) {
                jobs[i].state = STOPPED;
                printf("\\n[%d] Stopped\\t%s\\n", jobs[i].id, jobs[i].command);
            } else {
                jobs[i].state = DONE;
            }

            return 0;
        }
    }

    fprintf(stderr, "fg: job not found\\n");
    return 1;
}

int builtin_bg(char **args) {
    int job_id = args[1] ? atoi(args[1]) : job_count;

    for (int i = 0; i < job_count; i++) {
        if (jobs[i].id == job_id && jobs[i].state == STOPPED) {
            kill(-jobs[i].pgid, SIGCONT);
            jobs[i].state = RUNNING;
            printf("[%d] %s &\\n", jobs[i].id, jobs[i].command);
            return 0;
        }
    }

    fprintf(stderr, "bg: job not found\\n");
    return 1;
}

typedef int (*BuiltinFunc)(char **);
typedef struct {
    const char *name;
    BuiltinFunc func;
} Builtin;

Builtin builtins[] = {
    {"cd", builtin_cd},
    {"exit", builtin_exit},
    {"export", builtin_export},
    {"jobs", builtin_jobs},
    {"fg", builtin_fg},
    {"bg", builtin_bg},
    {NULL, NULL}
};

BuiltinFunc get_builtin(const char *name) {
    for (Builtin *b = builtins; b->name; b++) {
        if (strcmp(b->name, name) == 0) {
            return b->func;
        }
    }
    return NULL;
}

// Execute pipeline
void execute_pipeline(Command *cmd, int foreground) {
    int pipefd[2];
    int prev_fd = -1;
    pid_t pgid = 0;

    Command *current = cmd;
    while (current) {
        int has_next = (current->next != NULL);

        if (has_next) {
            if (pipe(pipefd) < 0) {
                perror("pipe");
                return;
            }
        }

        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return;
        }

        if (pid == 0) {
            // Child process

            // Reset signals
            signal(SIGINT, SIG_DFL);
            signal(SIGQUIT, SIG_DFL);
            signal(SIGTSTP, SIG_DFL);
            signal(SIGTTIN, SIG_DFL);
            signal(SIGTTOU, SIG_DFL);
            signal(SIGCHLD, SIG_DFL);

            // Set process group
            setpgid(0, pgid ? pgid : getpid());

            // Input redirection
            if (prev_fd != -1) {
                dup2(prev_fd, STDIN_FILENO);
                close(prev_fd);
            } else if (current->input_file) {
                int fd = open(current->input_file, O_RDONLY);
                if (fd < 0) {
                    perror(current->input_file);
                    exit(1);
                }
                dup2(fd, STDIN_FILENO);
                close(fd);
            }

            // Output redirection
            if (has_next) {
                dup2(pipefd[1], STDOUT_FILENO);
                close(pipefd[0]);
                close(pipefd[1]);
            } else if (current->output_file) {
                int flags = O_WRONLY | O_CREAT | (current->append_output ? O_APPEND : O_TRUNC);
                int fd = open(current->output_file, flags, 0644);
                if (fd < 0) {
                    perror(current->output_file);
                    exit(1);
                }
                dup2(fd, STDOUT_FILENO);
                close(fd);
            }

            // Execute
            execvp(current->args[0], current->args);
            perror(current->args[0]);
            exit(127);
        }

        // Parent process
        if (pgid == 0) {
            pgid = pid;
        }
        setpgid(pid, pgid);

        if (prev_fd != -1) {
            close(prev_fd);
        }
        if (has_next) {
            close(pipefd[1]);
            prev_fd = pipefd[0];
        }

        current = current->next;
    }

    // Add job
    Job *job = &jobs[job_count];
    job->id = job_count + 1;
    job->pgid = pgid;
    job->command = strdup(cmd->args[0]);  // Simplified
    job->state = RUNNING;
    job->foreground = foreground;
    job_count++;

    if (foreground) {
        // Give terminal to job
        tcsetpgrp(shell_terminal, pgid);

        // Wait for job
        int status;
        waitpid(-pgid, &status, WUNTRACED);

        // Take back terminal
        tcsetpgrp(shell_terminal, shell_pgid);

        if (WIFSTOPPED(status)) {
            job->state = STOPPED;
            printf("\\n[%d] Stopped\\t%s\\n", job->id, job->command);
        } else {
            job->state = DONE;
        }
    } else {
        printf("[%d] %d\\n", job->id, pgid);
    }
}

// Main loop
void shell_loop() {
    char line[MAX_LINE];

    while (1) {
        // Print prompt
        printf("mysh> ");
        fflush(stdout);

        // Read line
        if (!fgets(line, MAX_LINE, stdin)) {
            printf("\\n");
            break;
        }

        // Remove newline
        line[strcspn(line, "\\n")] = '\\0';

        if (strlen(line) == 0) continue;

        // Tokenize
        int token_count;
        Token *tokens = tokenize(line, &token_count);

        if (token_count == 0) continue;

        // Check for background
        int background = 0;
        if (tokens[token_count - 1].type == TOK_BACKGROUND) {
            background = 1;
            token_count--;
        }

        // Parse
        int pos = 0;
        Command *cmd = parse_command(tokens, &pos, token_count);

        if (!cmd || !cmd->args[0]) continue;

        // Check for builtin
        BuiltinFunc builtin = get_builtin(cmd->args[0]);
        if (builtin) {
            builtin(cmd->args);
        } else {
            execute_pipeline(cmd, !background);
        }

        // Cleanup completed jobs
        for (int i = 0; i < job_count; i++) {
            if (jobs[i].state == DONE && !jobs[i].foreground) {
                printf("[%d] Done\\t%s\\n", jobs[i].id, jobs[i].command);
            }
        }
    }
}

int main(int argc, char **argv) {
    init_shell();
    shell_loop();
    return 0;
}
\`\`\``
          },
          {
            title: 'Shell in Rust',
            description: 'Modern shell implementation in Rust',
            details: `# Unix Shell in Rust

\`\`\`rust
use std::collections::HashMap;
use std::env;
use std::io::{self, Write, BufRead};
use std::process::{Command, Stdio, Child};
use std::path::PathBuf;
use nix::sys::signal::{self, Signal};
use nix::sys::wait::{waitpid, WaitStatus, WaitPidFlag};
use nix::unistd::{self, Pid, ForkResult};

#[derive(Debug, Clone)]
enum Token {
    Word(String),
    Pipe,
    RedirectIn,
    RedirectOut,
    RedirectAppend,
    Background,
    And,
    Or,
    Semicolon,
}

#[derive(Debug)]
struct SimpleCommand {
    args: Vec<String>,
    input_file: Option<String>,
    output_file: Option<String>,
    append: bool,
}

#[derive(Debug)]
struct Pipeline {
    commands: Vec<SimpleCommand>,
    background: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum JobState {
    Running,
    Stopped,
    Done,
}

struct Job {
    id: usize,
    pgid: Pid,
    command: String,
    state: JobState,
}

struct Shell {
    jobs: Vec<Job>,
    aliases: HashMap<String, String>,
    history: Vec<String>,
    cwd: PathBuf,
}

impl Shell {
    fn new() -> Self {
        Shell {
            jobs: Vec::new(),
            aliases: HashMap::new(),
            history: Vec::new(),
            cwd: env::current_dir().unwrap_or_default(),
        }
    }

    fn tokenize(&self, input: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut chars = input.chars().peekable();

        while let Some(&c) = chars.peek() {
            match c {
                ' ' | '\\t' => {
                    chars.next();
                }
                '|' => {
                    chars.next();
                    if chars.peek() == Some(&'|') {
                        chars.next();
                        tokens.push(Token::Or);
                    } else {
                        tokens.push(Token::Pipe);
                    }
                }
                '<' => {
                    chars.next();
                    tokens.push(Token::RedirectIn);
                }
                '>' => {
                    chars.next();
                    if chars.peek() == Some(&'>') {
                        chars.next();
                        tokens.push(Token::RedirectAppend);
                    } else {
                        tokens.push(Token::RedirectOut);
                    }
                }
                '&' => {
                    chars.next();
                    if chars.peek() == Some(&'&') {
                        chars.next();
                        tokens.push(Token::And);
                    } else {
                        tokens.push(Token::Background);
                    }
                }
                ';' => {
                    chars.next();
                    tokens.push(Token::Semicolon);
                }
                '"' | '\\'' => {
                    let quote = chars.next().unwrap();
                    let mut word = String::new();
                    while let Some(&c) = chars.peek() {
                        if c == quote {
                            chars.next();
                            break;
                        }
                        word.push(chars.next().unwrap());
                    }
                    tokens.push(Token::Word(word));
                }
                _ => {
                    let mut word = String::new();
                    while let Some(&c) = chars.peek() {
                        if " \\t|<>&;".contains(c) {
                            break;
                        }
                        word.push(chars.next().unwrap());
                    }
                    if !word.is_empty() {
                        // Variable expansion
                        let expanded = self.expand_variables(&word);
                        tokens.push(Token::Word(expanded));
                    }
                }
            }
        }

        tokens
    }

    fn expand_variables(&self, word: &str) -> String {
        let mut result = String::new();
        let mut chars = word.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\$' {
                let mut var_name = String::new();
                if chars.peek() == Some(&'{') {
                    chars.next();
                    while let Some(&c) = chars.peek() {
                        if c == '}' {
                            chars.next();
                            break;
                        }
                        var_name.push(chars.next().unwrap());
                    }
                } else {
                    while let Some(&c) = chars.peek() {
                        if !c.is_alphanumeric() && c != '_' {
                            break;
                        }
                        var_name.push(chars.next().unwrap());
                    }
                }
                if let Ok(value) = env::var(&var_name) {
                    result.push_str(&value);
                }
            } else if c == '~' && result.is_empty() {
                if let Ok(home) = env::var("HOME") {
                    result.push_str(&home);
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    fn parse(&self, tokens: &[Token]) -> Option<Pipeline> {
        let mut commands = Vec::new();
        let mut current = SimpleCommand {
            args: Vec::new(),
            input_file: None,
            output_file: None,
            append: false,
        };
        let mut background = false;
        let mut iter = tokens.iter().peekable();

        while let Some(token) = iter.next() {
            match token {
                Token::Word(word) => {
                    current.args.push(word.clone());
                }
                Token::RedirectIn => {
                    if let Some(Token::Word(file)) = iter.next() {
                        current.input_file = Some(file.clone());
                    }
                }
                Token::RedirectOut => {
                    if let Some(Token::Word(file)) = iter.next() {
                        current.output_file = Some(file.clone());
                        current.append = false;
                    }
                }
                Token::RedirectAppend => {
                    if let Some(Token::Word(file)) = iter.next() {
                        current.output_file = Some(file.clone());
                        current.append = true;
                    }
                }
                Token::Pipe => {
                    if !current.args.is_empty() {
                        commands.push(current);
                        current = SimpleCommand {
                            args: Vec::new(),
                            input_file: None,
                            output_file: None,
                            append: false,
                        };
                    }
                }
                Token::Background => {
                    background = true;
                }
                Token::Semicolon | Token::And | Token::Or => {
                    // Handle command chaining (simplified)
                    break;
                }
            }
        }

        if !current.args.is_empty() {
            commands.push(current);
        }

        if commands.is_empty() {
            None
        } else {
            Some(Pipeline { commands, background })
        }
    }

    fn execute_builtin(&mut self, args: &[String]) -> Option<i32> {
        match args.get(0).map(|s| s.as_str()) {
            Some("cd") => {
                let dir = args.get(1)
                    .map(|s| s.as_str())
                    .or_else(|| env::var("HOME").ok().as_deref().map(|s| s.to_string()).as_deref())
                    .unwrap_or(".");

                if let Err(e) = env::set_current_dir(dir) {
                    eprintln!("cd: {}: {}", dir, e);
                    Some(1)
                } else {
                    self.cwd = env::current_dir().unwrap_or_default();
                    Some(0)
                }
            }
            Some("exit") => {
                let code = args.get(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                std::process::exit(code);
            }
            Some("export") => {
                if let Some(arg) = args.get(1) {
                    if let Some(pos) = arg.find('=') {
                        let (key, value) = arg.split_at(pos);
                        env::set_var(key, &value[1..]);
                    }
                } else {
                    for (key, value) in env::vars() {
                        println!("{}={}", key, value);
                    }
                }
                Some(0)
            }
            Some("jobs") => {
                for job in &self.jobs {
                    if job.state != JobState::Done {
                        let state = match job.state {
                            JobState::Running => "Running",
                            JobState::Stopped => "Stopped",
                            JobState::Done => "Done",
                        };
                        println!("[{}] {}\\t{}", job.id, state, job.command);
                    }
                }
                Some(0)
            }
            Some("alias") => {
                if let Some(arg) = args.get(1) {
                    if let Some(pos) = arg.find('=') {
                        let (name, value) = arg.split_at(pos);
                        self.aliases.insert(name.to_string(), value[1..].to_string());
                    }
                } else {
                    for (name, value) in &self.aliases {
                        println!("alias {}='{}'", name, value);
                    }
                }
                Some(0)
            }
            Some("history") => {
                for (i, cmd) in self.history.iter().enumerate() {
                    println!("{:5} {}", i + 1, cmd);
                }
                Some(0)
            }
            Some("pwd") => {
                println!("{}", self.cwd.display());
                Some(0)
            }
            _ => None,
        }
    }

    fn execute_pipeline(&mut self, pipeline: Pipeline) -> io::Result<i32> {
        if pipeline.commands.is_empty() {
            return Ok(0);
        }

        // Check for builtin (single command only)
        if pipeline.commands.len() == 1 {
            if let Some(code) = self.execute_builtin(&pipeline.commands[0].args) {
                return Ok(code);
            }
        }

        let mut children: Vec<Child> = Vec::new();
        let mut prev_stdout: Option<std::process::ChildStdout> = None;

        for (i, cmd) in pipeline.commands.iter().enumerate() {
            let is_first = i == 0;
            let is_last = i == pipeline.commands.len() - 1;

            let mut command = Command::new(&cmd.args[0]);
            command.args(&cmd.args[1..]);

            // Set up stdin
            if let Some(prev) = prev_stdout.take() {
                command.stdin(prev);
            } else if let Some(ref file) = cmd.input_file {
                let file = std::fs::File::open(file)?;
                command.stdin(file);
            }

            // Set up stdout
            if !is_last {
                command.stdout(Stdio::piped());
            } else if let Some(ref file) = cmd.output_file {
                let file = if cmd.append {
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(file)?
                } else {
                    std::fs::File::create(file)?
                };
                command.stdout(file);
            }

            let mut child = command.spawn()?;
            prev_stdout = child.stdout.take();
            children.push(child);
        }

        // Wait for all children
        let mut last_status = 0;
        for mut child in children {
            let status = child.wait()?;
            last_status = status.code().unwrap_or(1);
        }

        Ok(last_status)
    }

    fn run(&mut self) {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            // Print prompt
            print!("rush> ");
            stdout.flush().unwrap();

            // Read line
            let mut input = String::new();
            match stdin.lock().read_line(&mut input) {
                Ok(0) => break, // EOF
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error reading input: {}", e);
                    continue;
                }
            }

            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            // Add to history
            self.history.push(input.to_string());

            // Tokenize and parse
            let tokens = self.tokenize(input);
            if let Some(pipeline) = self.parse(&tokens) {
                if let Err(e) = self.execute_pipeline(pipeline) {
                    eprintln!("Error: {}", e);
                }
            }
        }
    }
}

fn main() {
    let mut shell = Shell::new();
    shell.run();
}
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Build Your Own Container Runtime',
    description: 'Create a Docker-like container runtime using Linux namespaces, cgroups, and overlay filesystems',
    icon: 'box',
    color: 'blue',
    language: 'Go, Rust, C',
    skills: 'Linux namespaces, cgroups, chroot, overlay fs, OCI spec',
    difficulty: 'advanced',
    estimated_weeks: 8,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Container concepts | Architecture design |
| 1 | 2 | Linux namespaces | Namespace overview |
| 1 | 3 | PID namespace | Process isolation |
| 1 | 4 | Mount namespace | Filesystem isolation |
| 1 | 5 | Network namespace | Network isolation |
| 2 | 1 | UTS namespace | Hostname isolation |
| 2 | 2 | User namespace | UID mapping |
| 2 | 3 | IPC namespace | IPC isolation |
| 2 | 4 | Cgroup namespace | Resource limits |
| 2 | 5 | Combined namespaces | Full isolation |
| 3 | 1 | Cgroups v1 | Memory limits |
| 3 | 2 | CPU limits | CPU cgroups |
| 3 | 3 | I/O limits | Block I/O |
| 3 | 4 | PIDs limit | Process limits |
| 3 | 5 | Cgroups v2 | Unified hierarchy |
| 4 | 1 | Root filesystem | chroot basics |
| 4 | 2 | pivot_root | Better isolation |
| 4 | 3 | OverlayFS | Layered filesystem |
| 4 | 4 | Image layers | Layer management |
| 4 | 5 | Copy-on-write | COW implementation |
| 5 | 1 | OCI image spec | Image format |
| 5 | 2 | Image pulling | Registry client |
| 5 | 3 | Image storage | Local storage |
| 5 | 4 | Layer extraction | Tar handling |
| 5 | 5 | Image manifest | Config parsing |
| 6 | 1 | OCI runtime spec | Runtime format |
| 6 | 2 | Container config | config.json |
| 6 | 3 | Lifecycle hooks | Prestart/poststart |
| 6 | 4 | State management | Container state |
| 6 | 5 | Runtime bundle | Bundle creation |
| 7 | 1 | Networking | veth pairs |
| 7 | 2 | Bridge network | Container bridge |
| 7 | 3 | Port mapping | NAT/iptables |
| 7 | 4 | DNS resolution | Container DNS |
| 7 | 5 | Network plugins | CNI basics |
| 8 | 1 | CLI interface | Command parsing |
| 8 | 2 | run command | Container creation |
| 8 | 3 | exec command | Process exec |
| 8 | 4 | ps/logs | Container management |
| 8 | 5 | Integration | Full testing |`,
    modules: [
      {
        name: 'Namespace Isolation',
        description: 'Linux namespace implementation',
        tasks: [
          {
            title: 'Container Runtime in Go',
            description: 'Build container runtime with namespace isolation',
            details: `# Container Runtime in Go

\`\`\`go
package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"syscall"

	"golang.org/x/sys/unix"
)

// ContainerConfig holds container configuration
type ContainerConfig struct {
	ID         string
	RootFS     string
	Command    []string
	Hostname   string
	Env        []string
	WorkingDir string
	Memory     int64  // Memory limit in bytes
	CPUShares  int64  // CPU shares
	Pids       int64  // Max PIDs
}

// Container represents a running container
type Container struct {
	Config    *ContainerConfig
	Pid       int
	State     string
	CgroupDir string
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: container run <rootfs> <command>")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "run":
		run()
	case "child":
		child()
	default:
		fmt.Printf("Unknown command: %s\\n", os.Args[1])
		os.Exit(1)
	}
}

func run() {
	config := &ContainerConfig{
		ID:       generateID(),
		RootFS:   os.Args[2],
		Command:  os.Args[3:],
		Hostname: "container",
		Env: []string{
			"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
			"TERM=xterm",
		},
		WorkingDir: "/",
		Memory:     256 * 1024 * 1024, // 256MB
		CPUShares:  512,
		Pids:       100,
	}

	container := &Container{
		Config: config,
		State:  "creating",
	}

	// Setup cgroups
	if err := container.setupCgroups(); err != nil {
		fmt.Printf("Failed to setup cgroups: %v\\n", err)
		os.Exit(1)
	}
	defer container.cleanupCgroups()

	// Setup network
	if err := container.setupNetwork(); err != nil {
		fmt.Printf("Failed to setup network: %v\\n", err)
		os.Exit(1)
	}

	// Re-exec ourselves in new namespaces
	cmd := exec.Command("/proc/self/exe", append([]string{"child"}, os.Args[2:]...)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Set up namespaces
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Cloneflags: syscall.CLONE_NEWUTS |  // Hostname
			syscall.CLONE_NEWPID |  // PID
			syscall.CLONE_NEWNS |   // Mount
			syscall.CLONE_NEWNET |  // Network
			syscall.CLONE_NEWIPC |  // IPC
			syscall.CLONE_NEWUSER, // User
		UidMappings: []syscall.SysProcIDMap{
			{ContainerID: 0, HostID: os.Getuid(), Size: 1},
		},
		GidMappings: []syscall.SysProcIDMap{
			{ContainerID: 0, HostID: os.Getgid(), Size: 1},
		},
	}

	// Set environment
	cmd.Env = config.Env

	container.State = "running"

	if err := cmd.Run(); err != nil {
		fmt.Printf("Container exited with error: %v\\n", err)
		os.Exit(1)
	}

	container.State = "stopped"
}

// child runs inside the new namespaces
func child() {
	rootfs := os.Args[2]
	command := os.Args[3:]

	// Set hostname
	if err := syscall.Sethostname([]byte("container")); err != nil {
		fmt.Printf("Failed to set hostname: %v\\n", err)
	}

	// Setup root filesystem
	if err := setupRootFS(rootfs); err != nil {
		fmt.Printf("Failed to setup rootfs: %v\\n", err)
		os.Exit(1)
	}

	// Mount proc
	if err := syscall.Mount("proc", "/proc", "proc", 0, ""); err != nil {
		fmt.Printf("Failed to mount proc: %v\\n", err)
	}

	// Mount sysfs
	if err := syscall.Mount("sysfs", "/sys", "sysfs", 0, ""); err != nil {
		fmt.Printf("Failed to mount sysfs: %v\\n", err)
	}

	// Mount devpts
	os.MkdirAll("/dev/pts", 0755)
	if err := syscall.Mount("devpts", "/dev/pts", "devpts", 0, ""); err != nil {
		// Non-fatal
	}

	// Run the command
	cmd := exec.Command(command[0], command[1:]...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		fmt.Printf("Command failed: %v\\n", err)
		os.Exit(1)
	}
}

func setupRootFS(rootfs string) error {
	// Make rootfs private
	if err := syscall.Mount("", "/", "", syscall.MS_PRIVATE|syscall.MS_REC, ""); err != nil {
		return fmt.Errorf("mount private: %w", err)
	}

	// Bind mount new root
	if err := syscall.Mount(rootfs, rootfs, "", syscall.MS_BIND|syscall.MS_REC, ""); err != nil {
		return fmt.Errorf("bind mount rootfs: %w", err)
	}

	// Create old root directory
	oldRoot := filepath.Join(rootfs, ".old_root")
	if err := os.MkdirAll(oldRoot, 0700); err != nil {
		return fmt.Errorf("mkdir old root: %w", err)
	}

	// pivot_root
	if err := syscall.PivotRoot(rootfs, oldRoot); err != nil {
		return fmt.Errorf("pivot_root: %w", err)
	}

	// Change to new root
	if err := os.Chdir("/"); err != nil {
		return fmt.Errorf("chdir: %w", err)
	}

	// Unmount old root
	oldRoot = "/.old_root"
	if err := syscall.Unmount(oldRoot, syscall.MNT_DETACH); err != nil {
		return fmt.Errorf("unmount old root: %w", err)
	}

	// Remove old root
	if err := os.RemoveAll(oldRoot); err != nil {
		// Non-fatal
	}

	return nil
}

func (c *Container) setupCgroups() error {
	cgroupDir := filepath.Join("/sys/fs/cgroup", "container-"+c.Config.ID)
	c.CgroupDir = cgroupDir

	// Create cgroup directory
	if err := os.MkdirAll(cgroupDir, 0755); err != nil {
		return fmt.Errorf("create cgroup dir: %w", err)
	}

	// Enable controllers for cgroups v2
	controllersPath := filepath.Join(filepath.Dir(cgroupDir), "cgroup.subtree_control")
	if err := ioutil.WriteFile(controllersPath, []byte("+memory +pids +cpu"), 0644); err != nil {
		// May fail if not cgroups v2 or controllers not available
	}

	// Set memory limit
	if c.Config.Memory > 0 {
		memFile := filepath.Join(cgroupDir, "memory.max")
		if err := ioutil.WriteFile(memFile, []byte(strconv.FormatInt(c.Config.Memory, 10)), 0644); err != nil {
			// Try cgroups v1 path
			memV1 := filepath.Join("/sys/fs/cgroup/memory", "container-"+c.Config.ID)
			os.MkdirAll(memV1, 0755)
			ioutil.WriteFile(filepath.Join(memV1, "memory.limit_in_bytes"),
				[]byte(strconv.FormatInt(c.Config.Memory, 10)), 0644)
		}
	}

	// Set PID limit
	if c.Config.Pids > 0 {
		pidsFile := filepath.Join(cgroupDir, "pids.max")
		if err := ioutil.WriteFile(pidsFile, []byte(strconv.FormatInt(c.Config.Pids, 10)), 0644); err != nil {
			// Try cgroups v1 path
		}
	}

	// Set CPU shares
	if c.Config.CPUShares > 0 {
		cpuFile := filepath.Join(cgroupDir, "cpu.weight")
		// Convert shares (1-1024) to weight (1-10000)
		weight := (c.Config.CPUShares * 10000) / 1024
		ioutil.WriteFile(cpuFile, []byte(strconv.FormatInt(weight, 10)), 0644)
	}

	return nil
}

func (c *Container) cleanupCgroups() error {
	if c.CgroupDir != "" {
		return os.RemoveAll(c.CgroupDir)
	}
	return nil
}

func (c *Container) setupNetwork() error {
	// Create veth pair
	// In production, would use netlink
	// For now, rely on pre-configured network
	return nil
}

func generateID() string {
	return fmt.Sprintf("%d", os.Getpid())
}

// OverlayFS support for image layers
type ImageLayer struct {
	ID       string
	Parent   string
	DiffPath string
}

type Image struct {
	ID     string
	Layers []ImageLayer
}

func setupOverlayFS(layers []string, workdir, merged string) error {
	if len(layers) == 0 {
		return fmt.Errorf("no layers provided")
	}

	// Create work and merged directories
	os.MkdirAll(workdir, 0755)
	os.MkdirAll(merged, 0755)

	// Build overlay options
	// lowerdir=layer3:layer2:layer1,upperdir=upper,workdir=work
	lowerDirs := ""
	for i := len(layers) - 1; i >= 1; i-- {
		if lowerDirs != "" {
			lowerDirs += ":"
		}
		lowerDirs += layers[i]
	}

	upperDir := layers[0] // Top layer is writable
	upperWork := workdir + "/work"
	os.MkdirAll(upperWork, 0755)

	opts := fmt.Sprintf("lowerdir=%s,upperdir=%s,workdir=%s", lowerDirs, upperDir, upperWork)

	return syscall.Mount("overlay", merged, "overlay", 0, opts)
}

// OCI Runtime Spec structures
type OCISpec struct {
	Version string      \`json:"ociVersion"\`
	Process OCIProcess  \`json:"process"\`
	Root    OCIRoot     \`json:"root"\`
	Mounts  []OCIMount  \`json:"mounts"\`
	Linux   OCILinux    \`json:"linux"\`
}

type OCIProcess struct {
	Terminal bool     \`json:"terminal"\`
	User     OCIUser  \`json:"user"\`
	Args     []string \`json:"args"\`
	Env      []string \`json:"env"\`
	Cwd      string   \`json:"cwd"\`
}

type OCIUser struct {
	UID uint32 \`json:"uid"\`
	GID uint32 \`json:"gid"\`
}

type OCIRoot struct {
	Path     string \`json:"path"\`
	Readonly bool   \`json:"readonly"\`
}

type OCIMount struct {
	Destination string   \`json:"destination"\`
	Type        string   \`json:"type"\`
	Source      string   \`json:"source"\`
	Options     []string \`json:"options"\`
}

type OCILinux struct {
	Namespaces []OCINamespace   \`json:"namespaces"\`
	Resources  *OCIResources    \`json:"resources"\`
}

type OCINamespace struct {
	Type string \`json:"type"\`
	Path string \`json:"path,omitempty"\`
}

type OCIResources struct {
	Memory *OCIMemory \`json:"memory,omitempty"\`
	CPU    *OCICPU    \`json:"cpu,omitempty"\`
	Pids   *OCIPids   \`json:"pids,omitempty"\`
}

type OCIMemory struct {
	Limit int64 \`json:"limit"\`
}

type OCICPU struct {
	Shares int64 \`json:"shares"\`
}

type OCIPids struct {
	Limit int64 \`json:"limit"\`
}

func parseOCISpec(configPath string) (*OCISpec, error) {
	data, err := ioutil.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	var spec OCISpec
	if err := json.Unmarshal(data, &spec); err != nil {
		return nil, err
	}

	return &spec, nil
}
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Build Your Own Debugger',
    description: 'Create a GDB-like debugger with breakpoints, single-stepping, and memory inspection',
    icon: 'bug',
    color: 'red',
    language: 'C, Rust',
    skills: 'ptrace, ELF parsing, DWARF debugging info, Signal handling',
    difficulty: 'advanced',
    estimated_weeks: 6,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | ptrace basics | Attach/detach |
| 1 | 2 | Process control | Start/stop process |
| 1 | 3 | Single stepping | PTRACE_SINGLESTEP |
| 1 | 4 | Continue execution | PTRACE_CONT |
| 1 | 5 | Register access | PTRACE_GETREGS |
| 2 | 1 | Memory reading | PTRACE_PEEKDATA |
| 2 | 2 | Memory writing | PTRACE_POKEDATA |
| 2 | 3 | Breakpoints | INT3 injection |
| 2 | 4 | Breakpoint management | Enable/disable |
| 2 | 5 | Breakpoint restoration | Original bytes |
| 3 | 1 | ELF parsing | Header parsing |
| 3 | 2 | Section headers | .text, .data |
| 3 | 3 | Symbol table | .symtab, .dynsym |
| 3 | 4 | String table | Symbol names |
| 3 | 5 | Address lookup | Symbol resolution |
| 4 | 1 | DWARF basics | Debug info format |
| 4 | 2 | .debug_info | DIE parsing |
| 4 | 3 | .debug_line | Line number info |
| 4 | 4 | Source mapping | Address to line |
| 4 | 5 | Variable info | Local variables |
| 5 | 1 | Stack traces | Backtrace |
| 5 | 2 | Frame pointer | Stack walking |
| 5 | 3 | Call frame info | .eh_frame |
| 5 | 4 | Unwinding | Stack unwinding |
| 5 | 5 | Frame display | Formatted output |
| 6 | 1 | CLI interface | Command parsing |
| 6 | 2 | break command | Set breakpoints |
| 6 | 3 | print command | Variable printing |
| 6 | 4 | step/next | Stepping commands |
| 6 | 5 | Integration | Full debugger |`,
    modules: [
      {
        name: 'ptrace Foundation',
        description: 'Process tracing with ptrace',
        tasks: [
          {
            title: 'Debugger in C',
            description: 'Build a debugger using ptrace',
            details: `# Debugger Implementation in C

\`\`\`c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/personality.h>
#include <elf.h>
#include <fcntl.h>
#include <errno.h>

#define MAX_BREAKPOINTS 256
#define MAX_SYMBOLS 4096

typedef struct {
    unsigned long address;
    unsigned long original_byte;
    int enabled;
    char *name;
} Breakpoint;

typedef struct {
    char *name;
    unsigned long address;
    unsigned long size;
    int type;
} Symbol;

typedef struct {
    pid_t pid;
    char *program;
    int running;

    Breakpoint breakpoints[MAX_BREAKPOINTS];
    int breakpoint_count;

    Symbol symbols[MAX_SYMBOLS];
    int symbol_count;

    unsigned long load_address;
} Debugger;

// Initialize debugger
Debugger *debugger_create(const char *program) {
    Debugger *dbg = calloc(1, sizeof(Debugger));
    dbg->program = strdup(program);
    return dbg;
}

// Read memory from tracee
unsigned long debugger_read_memory(Debugger *dbg, unsigned long addr) {
    errno = 0;
    unsigned long data = ptrace(PTRACE_PEEKDATA, dbg->pid, addr, NULL);
    if (errno != 0) {
        perror("ptrace PEEKDATA");
    }
    return data;
}

// Write memory to tracee
void debugger_write_memory(Debugger *dbg, unsigned long addr, unsigned long data) {
    if (ptrace(PTRACE_POKEDATA, dbg->pid, addr, data) < 0) {
        perror("ptrace POKEDATA");
    }
}

// Get registers
struct user_regs_struct debugger_get_registers(Debugger *dbg) {
    struct user_regs_struct regs;
    if (ptrace(PTRACE_GETREGS, dbg->pid, NULL, &regs) < 0) {
        perror("ptrace GETREGS");
    }
    return regs;
}

// Set registers
void debugger_set_registers(Debugger *dbg, struct user_regs_struct *regs) {
    if (ptrace(PTRACE_SETREGS, dbg->pid, NULL, regs) < 0) {
        perror("ptrace SETREGS");
    }
}

// Set breakpoint
Breakpoint *debugger_set_breakpoint(Debugger *dbg, unsigned long addr) {
    if (dbg->breakpoint_count >= MAX_BREAKPOINTS) {
        fprintf(stderr, "Max breakpoints reached\\n");
        return NULL;
    }

    Breakpoint *bp = &dbg->breakpoints[dbg->breakpoint_count++];
    bp->address = addr;
    bp->enabled = 1;

    // Save original byte and inject INT3 (0xCC)
    unsigned long data = debugger_read_memory(dbg, addr);
    bp->original_byte = data & 0xFF;

    unsigned long int3 = (data & ~0xFF) | 0xCC;
    debugger_write_memory(dbg, addr, int3);

    printf("Breakpoint %d at 0x%lx\\n", dbg->breakpoint_count, addr);
    return bp;
}

// Remove breakpoint
void debugger_remove_breakpoint(Debugger *dbg, Breakpoint *bp) {
    if (!bp->enabled) return;

    unsigned long data = debugger_read_memory(dbg, bp->address);
    unsigned long restored = (data & ~0xFF) | bp->original_byte;
    debugger_write_memory(dbg, bp->address, restored);

    bp->enabled = 0;
}

// Enable breakpoint
void debugger_enable_breakpoint(Debugger *dbg, Breakpoint *bp) {
    if (bp->enabled) return;

    unsigned long data = debugger_read_memory(dbg, bp->address);
    bp->original_byte = data & 0xFF;

    unsigned long int3 = (data & ~0xFF) | 0xCC;
    debugger_write_memory(dbg, bp->address, int3);

    bp->enabled = 1;
}

// Find breakpoint at address
Breakpoint *debugger_find_breakpoint(Debugger *dbg, unsigned long addr) {
    for (int i = 0; i < dbg->breakpoint_count; i++) {
        if (dbg->breakpoints[i].address == addr && dbg->breakpoints[i].enabled) {
            return &dbg->breakpoints[i];
        }
    }
    return NULL;
}

// Continue execution
void debugger_continue(Debugger *dbg) {
    // Check if we're at a breakpoint
    struct user_regs_struct regs = debugger_get_registers(dbg);
    unsigned long addr = regs.rip - 1;  // INT3 advances RIP

    Breakpoint *bp = debugger_find_breakpoint(dbg, addr);
    if (bp) {
        // Step over breakpoint
        regs.rip = addr;
        debugger_set_registers(dbg, &regs);

        // Temporarily disable breakpoint
        debugger_remove_breakpoint(dbg, bp);

        // Single step
        if (ptrace(PTRACE_SINGLESTEP, dbg->pid, NULL, NULL) < 0) {
            perror("ptrace SINGLESTEP");
            return;
        }

        int status;
        waitpid(dbg->pid, &status, 0);

        // Re-enable breakpoint
        debugger_enable_breakpoint(dbg, bp);
    }

    if (ptrace(PTRACE_CONT, dbg->pid, NULL, NULL) < 0) {
        perror("ptrace CONT");
    }
}

// Single step
void debugger_single_step(Debugger *dbg) {
    if (ptrace(PTRACE_SINGLESTEP, dbg->pid, NULL, NULL) < 0) {
        perror("ptrace SINGLESTEP");
    }
}

// Step over (next)
void debugger_step_over(Debugger *dbg) {
    struct user_regs_struct regs = debugger_get_registers(dbg);
    unsigned char instr[16];

    // Read instruction
    for (int i = 0; i < 16; i += sizeof(long)) {
        unsigned long data = debugger_read_memory(dbg, regs.rip + i);
        memcpy(instr + i, &data, sizeof(long));
    }

    // Check if it's a CALL instruction (0xE8 or 0xFF /2)
    if (instr[0] == 0xE8 || (instr[0] == 0xFF && (instr[1] & 0x38) == 0x10)) {
        // Set temporary breakpoint at next instruction
        unsigned long next_addr;
        if (instr[0] == 0xE8) {
            next_addr = regs.rip + 5;  // CALL rel32
        } else {
            // More complex CALL encoding
            next_addr = regs.rip + 2;  // Simplified
        }

        Breakpoint *bp = debugger_set_breakpoint(dbg, next_addr);
        debugger_continue(dbg);

        // Wait for breakpoint
        int status;
        waitpid(dbg->pid, &status, 0);

        // Remove temp breakpoint
        if (bp) {
            debugger_remove_breakpoint(dbg, bp);
            dbg->breakpoint_count--;
        }
    } else {
        debugger_single_step(dbg);
    }
}

// Parse ELF and load symbols
void debugger_load_symbols(Debugger *dbg) {
    FILE *f = fopen(dbg->program, "rb");
    if (!f) return;

    Elf64_Ehdr ehdr;
    fread(&ehdr, sizeof(ehdr), 1, f);

    if (memcmp(ehdr.e_ident, ELFMAG, SELFMAG) != 0) {
        fclose(f);
        return;
    }

    // Find symbol table section
    Elf64_Shdr *shdrs = malloc(ehdr.e_shentsize * ehdr.e_shnum);
    fseek(f, ehdr.e_shoff, SEEK_SET);
    fread(shdrs, ehdr.e_shentsize, ehdr.e_shnum, f);

    // Get section header string table
    Elf64_Shdr *shstrtab = &shdrs[ehdr.e_shstrndx];
    char *shstrs = malloc(shstrtab->sh_size);
    fseek(f, shstrtab->sh_offset, SEEK_SET);
    fread(shstrs, shstrtab->sh_size, 1, f);

    // Find .symtab and .strtab
    Elf64_Shdr *symtab = NULL, *strtab = NULL;
    for (int i = 0; i < ehdr.e_shnum; i++) {
        const char *name = shstrs + shdrs[i].sh_name;
        if (strcmp(name, ".symtab") == 0) {
            symtab = &shdrs[i];
        } else if (strcmp(name, ".strtab") == 0) {
            strtab = &shdrs[i];
        }
    }

    if (!symtab || !strtab) {
        // Try .dynsym and .dynstr
        for (int i = 0; i < ehdr.e_shnum; i++) {
            const char *name = shstrs + shdrs[i].sh_name;
            if (strcmp(name, ".dynsym") == 0) {
                symtab = &shdrs[i];
            } else if (strcmp(name, ".dynstr") == 0) {
                strtab = &shdrs[i];
            }
        }
    }

    if (symtab && strtab) {
        // Load string table
        char *strs = malloc(strtab->sh_size);
        fseek(f, strtab->sh_offset, SEEK_SET);
        fread(strs, strtab->sh_size, 1, f);

        // Load symbols
        int sym_count = symtab->sh_size / sizeof(Elf64_Sym);
        Elf64_Sym *syms = malloc(symtab->sh_size);
        fseek(f, symtab->sh_offset, SEEK_SET);
        fread(syms, sizeof(Elf64_Sym), sym_count, f);

        for (int i = 0; i < sym_count && dbg->symbol_count < MAX_SYMBOLS; i++) {
            if (syms[i].st_name && syms[i].st_value) {
                Symbol *sym = &dbg->symbols[dbg->symbol_count++];
                sym->name = strdup(strs + syms[i].st_name);
                sym->address = syms[i].st_value;
                sym->size = syms[i].st_size;
                sym->type = ELF64_ST_TYPE(syms[i].st_info);
            }
        }

        free(strs);
        free(syms);
    }

    free(shstrs);
    free(shdrs);
    fclose(f);

    printf("Loaded %d symbols\\n", dbg->symbol_count);
}

// Lookup symbol by name
Symbol *debugger_lookup_symbol(Debugger *dbg, const char *name) {
    for (int i = 0; i < dbg->symbol_count; i++) {
        if (strcmp(dbg->symbols[i].name, name) == 0) {
            return &dbg->symbols[i];
        }
    }
    return NULL;
}

// Print backtrace
void debugger_backtrace(Debugger *dbg) {
    struct user_regs_struct regs = debugger_get_registers(dbg);

    printf("Backtrace:\\n");

    unsigned long rip = regs.rip;
    unsigned long rbp = regs.rbp;
    int frame = 0;

    while (rbp != 0 && frame < 20) {
        // Find symbol for address
        const char *name = "??";
        for (int i = 0; i < dbg->symbol_count; i++) {
            if (rip >= dbg->symbols[i].address &&
                rip < dbg->symbols[i].address + dbg->symbols[i].size) {
                name = dbg->symbols[i].name;
                break;
            }
        }

        printf("#%d  0x%lx in %s\\n", frame++, rip, name);

        // Get return address and previous frame pointer
        rip = debugger_read_memory(dbg, rbp + 8);
        rbp = debugger_read_memory(dbg, rbp);
    }
}

// Wait for signal
void debugger_wait(Debugger *dbg) {
    int status;
    waitpid(dbg->pid, &status, 0);

    if (WIFEXITED(status)) {
        printf("Process exited with code %d\\n", WEXITSTATUS(status));
        dbg->running = 0;
    } else if (WIFSTOPPED(status)) {
        int sig = WSTOPSIG(status);

        if (sig == SIGTRAP) {
            struct user_regs_struct regs = debugger_get_registers(dbg);

            Breakpoint *bp = debugger_find_breakpoint(dbg, regs.rip - 1);
            if (bp) {
                printf("Breakpoint hit at 0x%lx\\n", bp->address);
            } else {
                printf("Stopped at 0x%lx\\n", regs.rip);
            }
        } else {
            printf("Received signal %d\\n", sig);
        }
    }
}

// Run debugger
void debugger_run(Debugger *dbg) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child - tracee
        personality(ADDR_NO_RANDOMIZE);  // Disable ASLR
        ptrace(PTRACE_TRACEME, 0, NULL, NULL);
        execl(dbg->program, dbg->program, NULL);
        perror("execl");
        exit(1);
    } else {
        // Parent - debugger
        dbg->pid = pid;
        dbg->running = 1;

        // Wait for initial stop
        debugger_wait(dbg);

        // Load symbols
        debugger_load_symbols(dbg);

        // Get load address for PIE executables
        char maps_path[64];
        snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", pid);
        FILE *maps = fopen(maps_path, "r");
        if (maps) {
            char line[256];
            if (fgets(line, sizeof(line), maps)) {
                sscanf(line, "%lx-", &dbg->load_address);
            }
            fclose(maps);
        }
    }
}

// Handle commands
void debugger_handle_command(Debugger *dbg, char *line) {
    char cmd[64];
    char arg[256];

    int n = sscanf(line, "%63s %255s", cmd, arg);
    if (n < 1) return;

    if (strcmp(cmd, "break") == 0 || strcmp(cmd, "b") == 0) {
        if (n < 2) {
            printf("Usage: break <address|symbol>\\n");
            return;
        }

        unsigned long addr;
        if (arg[0] == '0' && arg[1] == 'x') {
            addr = strtoul(arg, NULL, 16);
        } else {
            Symbol *sym = debugger_lookup_symbol(dbg, arg);
            if (!sym) {
                printf("Symbol not found: %s\\n", arg);
                return;
            }
            addr = sym->address + dbg->load_address;
        }

        debugger_set_breakpoint(dbg, addr);

    } else if (strcmp(cmd, "continue") == 0 || strcmp(cmd, "c") == 0) {
        debugger_continue(dbg);
        debugger_wait(dbg);

    } else if (strcmp(cmd, "step") == 0 || strcmp(cmd, "s") == 0) {
        debugger_single_step(dbg);
        debugger_wait(dbg);

    } else if (strcmp(cmd, "next") == 0 || strcmp(cmd, "n") == 0) {
        debugger_step_over(dbg);

    } else if (strcmp(cmd, "regs") == 0) {
        struct user_regs_struct regs = debugger_get_registers(dbg);
        printf("rax: 0x%llx\\n", regs.rax);
        printf("rbx: 0x%llx\\n", regs.rbx);
        printf("rcx: 0x%llx\\n", regs.rcx);
        printf("rdx: 0x%llx\\n", regs.rdx);
        printf("rsi: 0x%llx\\n", regs.rsi);
        printf("rdi: 0x%llx\\n", regs.rdi);
        printf("rbp: 0x%llx\\n", regs.rbp);
        printf("rsp: 0x%llx\\n", regs.rsp);
        printf("rip: 0x%llx\\n", regs.rip);

    } else if (strcmp(cmd, "backtrace") == 0 || strcmp(cmd, "bt") == 0) {
        debugger_backtrace(dbg);

    } else if (strcmp(cmd, "quit") == 0 || strcmp(cmd, "q") == 0) {
        exit(0);

    } else {
        printf("Unknown command: %s\\n", cmd);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <program>\\n", argv[0]);
        return 1;
    }

    Debugger *dbg = debugger_create(argv[1]);
    debugger_run(dbg);

    char line[256];
    while (1) {
        printf("(dbg) ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) {
            break;
        }

        line[strcspn(line, "\\n")] = 0;
        if (strlen(line) == 0) continue;

        debugger_handle_command(dbg, line);
    }

    return 0;
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

console.log('Seeded: Shell, Container Runtime, Debugger');
