#!/usr/bin/env npx tsx
/**
 * Seed: Compiler & Language Development
 * Build compilers, interpreters, and your own programming language
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
// BUILD YOUR OWN PROGRAMMING LANGUAGE
// ============================================================================
const langSchedule = `## 8-Week Schedule

### Week 1: Language Design
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Vision | Define language goals, paradigm, syntax philosophy |
| Day 2 | Grammar | Write BNF/EBNF grammar specification |
| Day 3 | Examples | Write sample programs in your language |
| Day 4 | Types | Design type system (static/dynamic, strong/weak) |
| Day 5 | Features | List core features vs nice-to-haves |
| Weekend | Document | Create language specification document |

### Week 2: Lexer (Tokenizer)
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Token Types | Define all token types (keywords, operators, literals) |
| Day 2 | Scanner | Build character-by-character scanner |
| Day 3 | Lexer Core | Implement tokenization loop |
| Day 4 | Literals | Handle strings, numbers, identifiers |
| Day 5 | Edge Cases | Comments, whitespace, error handling |
| Weekend | Testing | Unit tests for lexer |

### Week 3: Parser
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | AST Nodes | Define AST node types |
| Day 2 | Expressions | Parse expressions with precedence |
| Day 3 | Statements | Parse statements (if, while, for) |
| Day 4 | Functions | Parse function definitions and calls |
| Day 5 | Error Recovery | Implement parser error recovery |
| Weekend | Testing | Parser tests with edge cases |

### Week 4: Semantic Analysis
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Symbol Table | Build scoped symbol table |
| Day 2 | Name Resolution | Resolve variable and function names |
| Day 3 | Type Checking | Implement type checker |
| Day 4 | Errors | Semantic error reporting |
| Day 5 | Validation | Validate function returns, unreachable code |
| Weekend | Integration | Connect all analysis phases |

### Week 5-6: Interpreter/Bytecode VM
| Day | Focus | Tasks |
|-----|-------|-------|
| Week 5 | Interpreter | Tree-walking interpreter |
| Week 6 | Bytecode | Optional: bytecode compiler + VM |

### Week 7: Standard Library
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1-2 | I/O | print, input, file operations |
| Day 3-4 | Collections | lists, maps, sets |
| Day 5 | Math/String | Built-in functions |
| Weekend | Documentation | Document standard library |

### Week 8: Polish & REPL
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1-2 | REPL | Interactive shell |
| Day 3-4 | Error Messages | Helpful, pretty error output |
| Day 5 | Examples | Sample programs, tutorials |
| Weekend | Release | Package, documentation, share |

### Daily Commitment
- **Minimum**: 2 hours focused coding
- **Ideal**: 4 hours with study breaks`;

const langPath = insertPath.run(
	'Build Your Own Programming Language',
	'Design and implement your own programming language from scratch. Lexer, parser, AST, type system, interpreter, and standard library.',
	'purple',
	'Rust+Python',
	'advanced',
	8,
	'Lexing, parsing, AST, type systems, interpreters, bytecode, VMs',
	langSchedule,
	now
);

// Module 1: Language Design
const langMod1 = insertModule.run(langPath.lastInsertRowid, 'Language Design', 'Design your language before coding', 0, now);

insertTask.run(langMod1.lastInsertRowid, 'Define Language Philosophy', 'Establish core language principles including paradigm choice (functional, OOP, procedural), type system design (static vs dynamic, inference), memory model, error handling strategy, and target use cases', `## Language Design Philosophy

### Key Decisions

1. **Paradigm**
   - Imperative (C, Python)
   - Functional (Haskell, Lisp)
   - Object-Oriented (Java, Ruby)
   - Multi-paradigm (Scala, Rust)

2. **Type System**
   - Static vs Dynamic typing
   - Strong vs Weak typing
   - Type inference level
   - Generics/Polymorphism

3. **Memory Management**
   - Garbage collection
   - Reference counting
   - Manual (malloc/free)
   - Ownership (Rust-style)

4. **Syntax Philosophy**
   - Minimalist (Lisp, Go)
   - Expressive (Ruby, Kotlin)
   - C-family (familiar)
   - Whitespace-sensitive (Python)

### Example: Designing "Spark" Language

\`\`\`
// Spark Language Design Document

Goals:
- Simple, readable syntax
- Static typing with inference
- First-class functions
- Pattern matching
- No null (Option types)

Syntax Examples:

// Variables (immutable by default)
let name = "Alice"
var counter = 0

// Functions
fn greet(name: String) -> String {
    return "Hello, " + name
}

// Or with expression body
fn double(x: Int) -> Int = x * 2

// Pattern matching
match value {
    Some(x) => print(x),
    None => print("empty"),
}

// Structs
struct Person {
    name: String,
    age: Int,
}

// Methods
impl Person {
    fn birthday(self) -> Person {
        Person { age: self.age + 1, ..self }
    }
}

// Enums with data
enum Result<T, E> {
    Ok(T),
    Err(E),
}

// Closures
let add = |a, b| a + b
let nums = [1, 2, 3].map(|x| x * 2)
\`\`\`

### Write Your Grammar (EBNF)

\`\`\`ebnf
program     = declaration* ;
declaration = varDecl | funDecl | statement ;

varDecl     = ("let" | "var") IDENTIFIER (":" type)? "=" expression ";" ;
funDecl     = "fn" IDENTIFIER "(" params? ")" ("->" type)? block ;

statement   = exprStmt | ifStmt | whileStmt | forStmt | returnStmt | block ;
exprStmt    = expression ";" ;
ifStmt      = "if" expression block ("else" (ifStmt | block))? ;
whileStmt   = "while" expression block ;
returnStmt  = "return" expression? ";" ;
block       = "{" declaration* "}" ;

expression  = assignment ;
assignment  = IDENTIFIER "=" assignment | logic_or ;
logic_or    = logic_and ("||" logic_and)* ;
logic_and   = equality ("&&" equality)* ;
equality    = comparison (("==" | "!=") comparison)* ;
comparison  = term ((">" | ">=" | "<" | "<=") term)* ;
term        = factor (("+" | "-") factor)* ;
factor      = unary (("*" | "/" | "%") unary)* ;
unary       = ("!" | "-") unary | call ;
call        = primary ("(" arguments? ")" | "." IDENTIFIER)* ;
primary     = NUMBER | STRING | "true" | "false" | "nil" | IDENTIFIER | "(" expression ")" ;
\`\`\`
`, 0, now);

// Module 2: Lexer
const langMod2 = insertModule.run(langPath.lastInsertRowid, 'Lexer Implementation', 'Tokenize source code', 1, now);

insertTask.run(langMod2.lastInsertRowid, 'Build the Lexer', 'Implement a lexical analyzer that scans source text character-by-character, recognizes language tokens (keywords, identifiers, literals, operators), handles whitespace and comments, and tracks source locations for error reporting', `## Lexer Implementation

### Python Implementation

\`\`\`python
#!/usr/bin/env python3
"""
Lexer for Spark Language
Converts source code into tokens
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Any


class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()

    # Keywords
    LET = auto()
    VAR = auto()
    FN = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    RETURN = auto()
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    STRUCT = auto()
    ENUM = auto()
    IMPL = auto()
    MATCH = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    BANG = auto()
    EQUAL = auto()
    EQUAL_EQUAL = auto()
    BANG_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    AND = auto()
    OR = auto()
    ARROW = auto()
    FAT_ARROW = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    SEMICOLON = auto()
    PIPE = auto()

    # Special
    EOF = auto()
    NEWLINE = auto()


@dataclass
class Token:
    type: TokenType
    lexeme: str
    literal: Any
    line: int
    column: int

    def __repr__(self):
        return f"Token({self.type.name}, '{self.lexeme}', {self.literal})"


class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Line {line}, Column {column}: {message}")


class Lexer:
    KEYWORDS = {
        'let': TokenType.LET,
        'var': TokenType.VAR,
        'fn': TokenType.FN,
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'while': TokenType.WHILE,
        'for': TokenType.FOR,
        'return': TokenType.RETURN,
        'true': TokenType.TRUE,
        'false': TokenType.FALSE,
        'nil': TokenType.NIL,
        'struct': TokenType.STRUCT,
        'enum': TokenType.ENUM,
        'impl': TokenType.IMPL,
        'match': TokenType.MATCH,
    }

    def __init__(self, source: str):
        self.source = source
        self.tokens: List[Token] = []
        self.start = 0
        self.current = 0
        self.line = 1
        self.column = 1
        self.start_column = 1

    def scan_tokens(self) -> List[Token]:
        while not self.is_at_end():
            self.start = self.current
            self.start_column = self.column
            self.scan_token()

        self.tokens.append(Token(TokenType.EOF, "", None, self.line, self.column))
        return self.tokens

    def scan_token(self):
        c = self.advance()

        match c:
            case '(': self.add_token(TokenType.LPAREN)
            case ')': self.add_token(TokenType.RPAREN)
            case '{': self.add_token(TokenType.LBRACE)
            case '}': self.add_token(TokenType.RBRACE)
            case '[': self.add_token(TokenType.LBRACKET)
            case ']': self.add_token(TokenType.RBRACKET)
            case ',': self.add_token(TokenType.COMMA)
            case '.': self.add_token(TokenType.DOT)
            case ':': self.add_token(TokenType.COLON)
            case ';': self.add_token(TokenType.SEMICOLON)
            case '+': self.add_token(TokenType.PLUS)
            case '*': self.add_token(TokenType.STAR)
            case '%': self.add_token(TokenType.PERCENT)
            case '|':
                if self.match('|'):
                    self.add_token(TokenType.OR)
                else:
                    self.add_token(TokenType.PIPE)
            case '&':
                if self.match('&'):
                    self.add_token(TokenType.AND)
                else:
                    self.error("Unexpected character '&'")
            case '-':
                if self.match('>'):
                    self.add_token(TokenType.ARROW)
                else:
                    self.add_token(TokenType.MINUS)
            case '=':
                if self.match('='):
                    self.add_token(TokenType.EQUAL_EQUAL)
                elif self.match('>'):
                    self.add_token(TokenType.FAT_ARROW)
                else:
                    self.add_token(TokenType.EQUAL)
            case '!':
                if self.match('='):
                    self.add_token(TokenType.BANG_EQUAL)
                else:
                    self.add_token(TokenType.BANG)
            case '<':
                if self.match('='):
                    self.add_token(TokenType.LESS_EQUAL)
                else:
                    self.add_token(TokenType.LESS)
            case '>':
                if self.match('='):
                    self.add_token(TokenType.GREATER_EQUAL)
                else:
                    self.add_token(TokenType.GREATER)
            case '/':
                if self.match('/'):
                    # Single-line comment
                    while self.peek() != '\\n' and not self.is_at_end():
                        self.advance()
                elif self.match('*'):
                    # Multi-line comment
                    self.block_comment()
                else:
                    self.add_token(TokenType.SLASH)
            case ' ' | '\\r' | '\\t':
                pass  # Ignore whitespace
            case '\\n':
                self.line += 1
                self.column = 1
            case '"':
                self.string()
            case _:
                if c.isdigit():
                    self.number()
                elif c.isalpha() or c == '_':
                    self.identifier()
                else:
                    self.error(f"Unexpected character '{c}'")

    def string(self):
        while self.peek() != '"' and not self.is_at_end():
            if self.peek() == '\\n':
                self.line += 1
                self.column = 0
            if self.peek() == '\\\\':
                self.advance()  # Skip escape char
            self.advance()

        if self.is_at_end():
            self.error("Unterminated string")
            return

        self.advance()  # Closing "

        # Get string value without quotes
        value = self.source[self.start + 1:self.current - 1]
        # Process escape sequences
        value = value.replace('\\\\n', '\\n').replace('\\\\t', '\\t').replace('\\\\"', '"')
        self.add_token(TokenType.STRING, value)

    def number(self):
        while self.peek().isdigit():
            self.advance()

        # Look for decimal
        if self.peek() == '.' and self.peek_next().isdigit():
            self.advance()  # Consume .
            while self.peek().isdigit():
                self.advance()

        value = float(self.source[self.start:self.current])
        if value.is_integer():
            value = int(value)
        self.add_token(TokenType.NUMBER, value)

    def identifier(self):
        while self.peek().isalnum() or self.peek() == '_':
            self.advance()

        text = self.source[self.start:self.current]
        token_type = self.KEYWORDS.get(text, TokenType.IDENTIFIER)

        literal = None
        if token_type == TokenType.TRUE:
            literal = True
        elif token_type == TokenType.FALSE:
            literal = False

        self.add_token(token_type, literal)

    def block_comment(self):
        depth = 1
        while depth > 0 and not self.is_at_end():
            if self.peek() == '/' and self.peek_next() == '*':
                self.advance()
                self.advance()
                depth += 1
            elif self.peek() == '*' and self.peek_next() == '/':
                self.advance()
                self.advance()
                depth -= 1
            else:
                if self.peek() == '\\n':
                    self.line += 1
                    self.column = 0
                self.advance()

        if depth > 0:
            self.error("Unterminated block comment")

    def advance(self) -> str:
        c = self.source[self.current]
        self.current += 1
        self.column += 1
        return c

    def peek(self) -> str:
        if self.is_at_end():
            return '\\0'
        return self.source[self.current]

    def peek_next(self) -> str:
        if self.current + 1 >= len(self.source):
            return '\\0'
        return self.source[self.current + 1]

    def match(self, expected: str) -> bool:
        if self.is_at_end():
            return False
        if self.source[self.current] != expected:
            return False
        self.current += 1
        self.column += 1
        return True

    def is_at_end(self) -> bool:
        return self.current >= len(self.source)

    def add_token(self, token_type: TokenType, literal: Any = None):
        lexeme = self.source[self.start:self.current]
        self.tokens.append(Token(token_type, lexeme, literal, self.line, self.start_column))

    def error(self, message: str):
        raise LexerError(message, self.line, self.start_column)


def main():
    source = '''
    fn factorial(n: Int) -> Int {
        if n <= 1 {
            return 1;
        }
        return n * factorial(n - 1);
    }

    let result = factorial(5);
    print(result);  // 120
    '''

    lexer = Lexer(source)
    tokens = lexer.scan_tokens()

    for token in tokens:
        print(token)


if __name__ == '__main__':
    main()
\`\`\`
`, 0, now);

// Module 3: Parser
const langMod3 = insertModule.run(langPath.lastInsertRowid, 'Parser Implementation', 'Parse tokens into AST', 2, now);

insertTask.run(langMod3.lastInsertRowid, 'Build the Parser', 'Implement a recursive descent or Pratt parser that consumes tokens and constructs an Abstract Syntax Tree, handling operator precedence, associativity, error recovery, and syntactic sugar desugaring', `## Parser Implementation

### Recursive Descent Parser

\`\`\`python
#!/usr/bin/env python3
"""
Parser for Spark Language
Converts tokens into AST
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Union
from lexer import Token, TokenType, Lexer


# ============================================================================
# AST Node Definitions
# ============================================================================

@dataclass
class Expr:
    """Base class for expressions"""
    pass

@dataclass
class BinaryExpr(Expr):
    left: Expr
    operator: Token
    right: Expr

@dataclass
class UnaryExpr(Expr):
    operator: Token
    operand: Expr

@dataclass
class LiteralExpr(Expr):
    value: Any

@dataclass
class IdentifierExpr(Expr):
    name: Token

@dataclass
class AssignExpr(Expr):
    name: Token
    value: Expr

@dataclass
class CallExpr(Expr):
    callee: Expr
    paren: Token
    arguments: List[Expr]

@dataclass
class GetExpr(Expr):
    object: Expr
    name: Token

@dataclass
class GroupingExpr(Expr):
    expression: Expr

@dataclass
class LambdaExpr(Expr):
    params: List[Token]
    body: 'Stmt'


# Statements
@dataclass
class Stmt:
    """Base class for statements"""
    pass

@dataclass
class ExprStmt(Stmt):
    expression: Expr

@dataclass
class VarStmt(Stmt):
    name: Token
    type_annotation: Optional[Token]
    initializer: Optional[Expr]
    mutable: bool

@dataclass
class BlockStmt(Stmt):
    statements: List[Stmt]

@dataclass
class IfStmt(Stmt):
    condition: Expr
    then_branch: Stmt
    else_branch: Optional[Stmt]

@dataclass
class WhileStmt(Stmt):
    condition: Expr
    body: Stmt

@dataclass
class FunctionStmt(Stmt):
    name: Token
    params: List[tuple]  # (name, type)
    return_type: Optional[Token]
    body: BlockStmt

@dataclass
class ReturnStmt(Stmt):
    keyword: Token
    value: Optional[Expr]

@dataclass
class StructStmt(Stmt):
    name: Token
    fields: List[tuple]  # (name, type)


# ============================================================================
# Parser
# ============================================================================

class ParseError(Exception):
    def __init__(self, token: Token, message: str):
        self.token = token
        self.message = message
        super().__init__(f"Line {token.line}: {message}")


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0

    def parse(self) -> List[Stmt]:
        statements = []
        while not self.is_at_end():
            stmt = self.declaration()
            if stmt:
                statements.append(stmt)
        return statements

    # ========== Declarations ==========

    def declaration(self) -> Optional[Stmt]:
        try:
            if self.match(TokenType.FN):
                return self.function_declaration()
            if self.match(TokenType.LET, TokenType.VAR):
                return self.var_declaration()
            if self.match(TokenType.STRUCT):
                return self.struct_declaration()
            return self.statement()
        except ParseError as e:
            self.synchronize()
            return None

    def function_declaration(self) -> FunctionStmt:
        name = self.consume(TokenType.IDENTIFIER, "Expected function name")

        self.consume(TokenType.LPAREN, "Expected '(' after function name")
        params = []

        if not self.check(TokenType.RPAREN):
            while True:
                param_name = self.consume(TokenType.IDENTIFIER, "Expected parameter name")
                self.consume(TokenType.COLON, "Expected ':' after parameter name")
                param_type = self.consume(TokenType.IDENTIFIER, "Expected parameter type")
                params.append((param_name, param_type))

                if not self.match(TokenType.COMMA):
                    break

        self.consume(TokenType.RPAREN, "Expected ')' after parameters")

        return_type = None
        if self.match(TokenType.ARROW):
            return_type = self.consume(TokenType.IDENTIFIER, "Expected return type")

        self.consume(TokenType.LBRACE, "Expected '{' before function body")
        body = self.block()

        return FunctionStmt(name, params, return_type, body)

    def var_declaration(self) -> VarStmt:
        mutable = self.previous().type == TokenType.VAR
        name = self.consume(TokenType.IDENTIFIER, "Expected variable name")

        type_annotation = None
        if self.match(TokenType.COLON):
            type_annotation = self.consume(TokenType.IDENTIFIER, "Expected type")

        initializer = None
        if self.match(TokenType.EQUAL):
            initializer = self.expression()

        self.consume(TokenType.SEMICOLON, "Expected ';' after variable declaration")
        return VarStmt(name, type_annotation, initializer, mutable)

    def struct_declaration(self) -> StructStmt:
        name = self.consume(TokenType.IDENTIFIER, "Expected struct name")
        self.consume(TokenType.LBRACE, "Expected '{' after struct name")

        fields = []
        while not self.check(TokenType.RBRACE):
            field_name = self.consume(TokenType.IDENTIFIER, "Expected field name")
            self.consume(TokenType.COLON, "Expected ':' after field name")
            field_type = self.consume(TokenType.IDENTIFIER, "Expected field type")
            fields.append((field_name, field_type))

            if not self.match(TokenType.COMMA):
                break

        self.consume(TokenType.RBRACE, "Expected '}' after struct fields")
        return StructStmt(name, fields)

    # ========== Statements ==========

    def statement(self) -> Stmt:
        if self.match(TokenType.IF):
            return self.if_statement()
        if self.match(TokenType.WHILE):
            return self.while_statement()
        if self.match(TokenType.RETURN):
            return self.return_statement()
        if self.match(TokenType.LBRACE):
            return BlockStmt(self.block().statements)
        return self.expression_statement()

    def if_statement(self) -> IfStmt:
        condition = self.expression()

        self.consume(TokenType.LBRACE, "Expected '{' after if condition")
        then_branch = BlockStmt(self.block().statements)

        else_branch = None
        if self.match(TokenType.ELSE):
            if self.match(TokenType.IF):
                else_branch = self.if_statement()
            else:
                self.consume(TokenType.LBRACE, "Expected '{' after else")
                else_branch = BlockStmt(self.block().statements)

        return IfStmt(condition, then_branch, else_branch)

    def while_statement(self) -> WhileStmt:
        condition = self.expression()
        self.consume(TokenType.LBRACE, "Expected '{' after while condition")
        body = BlockStmt(self.block().statements)
        return WhileStmt(condition, body)

    def return_statement(self) -> ReturnStmt:
        keyword = self.previous()
        value = None
        if not self.check(TokenType.SEMICOLON):
            value = self.expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after return value")
        return ReturnStmt(keyword, value)

    def block(self) -> BlockStmt:
        statements = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            statements.append(self.declaration())
        self.consume(TokenType.RBRACE, "Expected '}' after block")
        return BlockStmt(statements)

    def expression_statement(self) -> ExprStmt:
        expr = self.expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after expression")
        return ExprStmt(expr)

    # ========== Expressions ==========

    def expression(self) -> Expr:
        return self.assignment()

    def assignment(self) -> Expr:
        expr = self.or_expr()

        if self.match(TokenType.EQUAL):
            equals = self.previous()
            value = self.assignment()

            if isinstance(expr, IdentifierExpr):
                return AssignExpr(expr.name, value)

            raise ParseError(equals, "Invalid assignment target")

        return expr

    def or_expr(self) -> Expr:
        expr = self.and_expr()

        while self.match(TokenType.OR):
            operator = self.previous()
            right = self.and_expr()
            expr = BinaryExpr(expr, operator, right)

        return expr

    def and_expr(self) -> Expr:
        expr = self.equality()

        while self.match(TokenType.AND):
            operator = self.previous()
            right = self.equality()
            expr = BinaryExpr(expr, operator, right)

        return expr

    def equality(self) -> Expr:
        expr = self.comparison()

        while self.match(TokenType.EQUAL_EQUAL, TokenType.BANG_EQUAL):
            operator = self.previous()
            right = self.comparison()
            expr = BinaryExpr(expr, operator, right)

        return expr

    def comparison(self) -> Expr:
        expr = self.term()

        while self.match(TokenType.LESS, TokenType.LESS_EQUAL,
                        TokenType.GREATER, TokenType.GREATER_EQUAL):
            operator = self.previous()
            right = self.term()
            expr = BinaryExpr(expr, operator, right)

        return expr

    def term(self) -> Expr:
        expr = self.factor()

        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.previous()
            right = self.factor()
            expr = BinaryExpr(expr, operator, right)

        return expr

    def factor(self) -> Expr:
        expr = self.unary()

        while self.match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            operator = self.previous()
            right = self.unary()
            expr = BinaryExpr(expr, operator, right)

        return expr

    def unary(self) -> Expr:
        if self.match(TokenType.BANG, TokenType.MINUS):
            operator = self.previous()
            right = self.unary()
            return UnaryExpr(operator, right)

        return self.call()

    def call(self) -> Expr:
        expr = self.primary()

        while True:
            if self.match(TokenType.LPAREN):
                expr = self.finish_call(expr)
            elif self.match(TokenType.DOT):
                name = self.consume(TokenType.IDENTIFIER, "Expected property name")
                expr = GetExpr(expr, name)
            else:
                break

        return expr

    def finish_call(self, callee: Expr) -> CallExpr:
        arguments = []

        if not self.check(TokenType.RPAREN):
            while True:
                arguments.append(self.expression())
                if not self.match(TokenType.COMMA):
                    break

        paren = self.consume(TokenType.RPAREN, "Expected ')' after arguments")
        return CallExpr(callee, paren, arguments)

    def primary(self) -> Expr:
        if self.match(TokenType.FALSE):
            return LiteralExpr(False)
        if self.match(TokenType.TRUE):
            return LiteralExpr(True)
        if self.match(TokenType.NIL):
            return LiteralExpr(None)

        if self.match(TokenType.NUMBER, TokenType.STRING):
            return LiteralExpr(self.previous().literal)

        if self.match(TokenType.IDENTIFIER):
            return IdentifierExpr(self.previous())

        if self.match(TokenType.LPAREN):
            expr = self.expression()
            self.consume(TokenType.RPAREN, "Expected ')' after expression")
            return GroupingExpr(expr)

        raise ParseError(self.peek(), "Expected expression")

    # ========== Helpers ==========

    def match(self, *types: TokenType) -> bool:
        for t in types:
            if self.check(t):
                self.advance()
                return True
        return False

    def check(self, token_type: TokenType) -> bool:
        if self.is_at_end():
            return False
        return self.peek().type == token_type

    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
        return self.previous()

    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF

    def peek(self) -> Token:
        return self.tokens[self.current]

    def previous(self) -> Token:
        return self.tokens[self.current - 1]

    def consume(self, token_type: TokenType, message: str) -> Token:
        if self.check(token_type):
            return self.advance()
        raise ParseError(self.peek(), message)

    def synchronize(self):
        self.advance()
        while not self.is_at_end():
            if self.previous().type == TokenType.SEMICOLON:
                return
            if self.peek().type in (TokenType.FN, TokenType.LET, TokenType.VAR,
                                   TokenType.IF, TokenType.WHILE, TokenType.RETURN):
                return
            self.advance()
\`\`\`
`, 0, now);

// Module 4: Interpreter
const langMod4 = insertModule.run(langPath.lastInsertRowid, 'Interpreter', 'Execute the AST', 3, now);

insertTask.run(langMod4.lastInsertRowid, 'Build Tree-Walking Interpreter', 'Implement an interpreter that recursively traverses the Abstract Syntax Tree, evaluating expressions and executing statements directly without compilation to bytecode or machine code, handling scoping, function calls, and control flow', `## Tree-Walking Interpreter

\`\`\`python
#!/usr/bin/env python3
"""
Interpreter for Spark Language
Executes AST directly via tree-walking
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from parser import *


class Environment:
    """Variable scope environment"""

    def __init__(self, parent: Optional['Environment'] = None):
        self.values: Dict[str, Any] = {}
        self.parent = parent

    def define(self, name: str, value: Any):
        self.values[name] = value

    def get(self, name: Token) -> Any:
        if name.lexeme in self.values:
            return self.values[name.lexeme]
        if self.parent:
            return self.parent.get(name)
        raise RuntimeError(f"Undefined variable '{name.lexeme}'")

    def assign(self, name: Token, value: Any):
        if name.lexeme in self.values:
            self.values[name.lexeme] = value
            return
        if self.parent:
            self.parent.assign(name, value)
            return
        raise RuntimeError(f"Undefined variable '{name.lexeme}'")


@dataclass
class SparkFunction:
    declaration: FunctionStmt
    closure: Environment

    def call(self, interpreter: 'Interpreter', arguments: List[Any]) -> Any:
        env = Environment(self.closure)

        for i, (param, _) in enumerate(self.declaration.params):
            env.define(param.lexeme, arguments[i])

        try:
            interpreter.execute_block(self.declaration.body.statements, env)
        except ReturnValue as ret:
            return ret.value

        return None

    def arity(self) -> int:
        return len(self.declaration.params)


class ReturnValue(Exception):
    def __init__(self, value: Any):
        self.value = value


class Interpreter:
    def __init__(self):
        self.globals = Environment()
        self.environment = self.globals

        # Built-in functions
        self.globals.define("print", self._builtin_print)
        self.globals.define("input", self._builtin_input)
        self.globals.define("len", self._builtin_len)
        self.globals.define("str", self._builtin_str)
        self.globals.define("int", self._builtin_int)

    def interpret(self, statements: List[Stmt]):
        try:
            for stmt in statements:
                self.execute(stmt)
        except RuntimeError as e:
            print(f"Runtime Error: {e}")

    def execute(self, stmt: Stmt):
        match stmt:
            case ExprStmt(expression):
                self.evaluate(expression)

            case VarStmt(name, _, initializer, _):
                value = None
                if initializer:
                    value = self.evaluate(initializer)
                self.environment.define(name.lexeme, value)

            case BlockStmt(statements):
                self.execute_block(statements, Environment(self.environment))

            case IfStmt(condition, then_branch, else_branch):
                if self.is_truthy(self.evaluate(condition)):
                    self.execute(then_branch)
                elif else_branch:
                    self.execute(else_branch)

            case WhileStmt(condition, body):
                while self.is_truthy(self.evaluate(condition)):
                    self.execute(body)

            case FunctionStmt() as func:
                function = SparkFunction(func, self.environment)
                self.environment.define(func.name.lexeme, function)

            case ReturnStmt(_, value):
                result = None
                if value:
                    result = self.evaluate(value)
                raise ReturnValue(result)

    def execute_block(self, statements: List[Stmt], environment: Environment):
        previous = self.environment
        try:
            self.environment = environment
            for stmt in statements:
                self.execute(stmt)
        finally:
            self.environment = previous

    def evaluate(self, expr: Expr) -> Any:
        match expr:
            case LiteralExpr(value):
                return value

            case IdentifierExpr(name):
                return self.environment.get(name)

            case GroupingExpr(expression):
                return self.evaluate(expression)

            case UnaryExpr(operator, operand):
                right = self.evaluate(operand)
                match operator.type:
                    case TokenType.MINUS:
                        return -float(right)
                    case TokenType.BANG:
                        return not self.is_truthy(right)

            case BinaryExpr(left, operator, right):
                left_val = self.evaluate(left)
                right_val = self.evaluate(right)

                match operator.type:
                    case TokenType.PLUS:
                        if isinstance(left_val, str) or isinstance(right_val, str):
                            return str(left_val) + str(right_val)
                        return left_val + right_val
                    case TokenType.MINUS:
                        return left_val - right_val
                    case TokenType.STAR:
                        return left_val * right_val
                    case TokenType.SLASH:
                        if right_val == 0:
                            raise RuntimeError("Division by zero")
                        return left_val / right_val
                    case TokenType.PERCENT:
                        return left_val % right_val
                    case TokenType.GREATER:
                        return left_val > right_val
                    case TokenType.GREATER_EQUAL:
                        return left_val >= right_val
                    case TokenType.LESS:
                        return left_val < right_val
                    case TokenType.LESS_EQUAL:
                        return left_val <= right_val
                    case TokenType.EQUAL_EQUAL:
                        return left_val == right_val
                    case TokenType.BANG_EQUAL:
                        return left_val != right_val
                    case TokenType.AND:
                        return left_val and right_val
                    case TokenType.OR:
                        return left_val or right_val

            case AssignExpr(name, value):
                val = self.evaluate(value)
                self.environment.assign(name, val)
                return val

            case CallExpr(callee, _, arguments):
                function = self.evaluate(callee)
                args = [self.evaluate(arg) for arg in arguments]

                if callable(function):
                    return function(*args)

                if isinstance(function, SparkFunction):
                    if len(args) != function.arity():
                        raise RuntimeError(
                            f"Expected {function.arity()} arguments but got {len(args)}"
                        )
                    return function.call(self, args)

                raise RuntimeError("Can only call functions")

        return None

    def is_truthy(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        return True

    # Built-in functions
    def _builtin_print(self, *args):
        print(*args)

    def _builtin_input(self, prompt=""):
        return input(prompt)

    def _builtin_len(self, obj):
        return len(obj)

    def _builtin_str(self, obj):
        return str(obj)

    def _builtin_int(self, obj):
        return int(obj)


def main():
    from lexer import Lexer

    source = '''
    fn fib(n: Int) -> Int {
        if n <= 1 {
            return n;
        }
        return fib(n - 1) + fib(n - 2);
    }

    let i = 0;
    while i < 10 {
        print(fib(i));
        i = i + 1;
    }
    '''

    lexer = Lexer(source)
    tokens = lexer.scan_tokens()

    parser = Parser(tokens)
    statements = parser.parse()

    interpreter = Interpreter()
    interpreter.interpret(statements)


if __name__ == '__main__':
    main()
\`\`\`
`, 0, now);

// ============================================================================
// COMPILER IN RUST
// ============================================================================
const rustCompilerSchedule = `## 6-Week Schedule

### Week 1: Lexer in Rust
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Project Setup | Cargo init, define Token enum |
| Day 2 | Scanner | Implement character iteration |
| Day 3 | Tokenization | Numbers, strings, identifiers |
| Day 4 | Keywords | Keyword recognition with HashMap |
| Day 5 | Testing | Unit tests, error handling |
| Weekend | Polish | Edge cases, documentation |

### Week 2: Parser in Rust
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | AST Types | Define AST with enums |
| Day 2 | Expressions | Pratt parser for expressions |
| Day 3 | Statements | Parse statements |
| Day 4 | Functions | Function declarations |
| Day 5 | Errors | Error recovery, spans |
| Weekend | Testing | Comprehensive parser tests |

### Week 3-4: Type Checker
| Day | Focus | Tasks |
|-----|-------|-------|
| Week 3 | Type System | Inference, checking |
| Week 4 | Generics | Polymorphism, traits |

### Week 5-6: Code Generation
| Day | Focus | Tasks |
|-----|-------|-------|
| Week 5 | IR | Intermediate representation |
| Week 6 | Backend | LLVM or bytecode output |

### Daily Commitment
- **Minimum**: 2 hours
- **Ideal**: 4 hours`;

const rustCompilerPath = insertPath.run(
	'Build a Compiler in Rust',
	'Implement a complete compiler in Rust. Lexer, parser with Pratt parsing, type checker, and code generation targeting LLVM or custom bytecode.',
	'orange',
	'Rust',
	'advanced',
	6,
	'Rust, lexing, Pratt parsing, type inference, LLVM, bytecode',
	rustCompilerSchedule,
	now
);

const rustCompMod1 = insertModule.run(rustCompilerPath.lastInsertRowid, 'Rust Lexer', 'Build a lexer in Rust', 0, now);

insertTask.run(rustCompMod1.lastInsertRowid, 'Build Rust Lexer', 'Build a lexical analyzer in Rust that converts source text into tokens, handling keywords, identifiers, literals, operators, and whitespace with proper error reporting and source location tracking', `## Rust Lexer Implementation

\`\`\`rust
//! Lexer for a simple programming language

use std::iter::Peekable;
use std::str::Chars;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Number(f64),
    String(String),
    Identifier(String),

    // Keywords
    Let, Mut, Fn, If, Else, While, For, Return,
    True, False, Nil, Struct, Enum, Impl, Match,

    // Operators
    Plus, Minus, Star, Slash, Percent,
    Bang, Eq, EqEq, BangEq,
    Lt, LtEq, Gt, GtEq,
    And, Or, Arrow, FatArrow,

    // Delimiters
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Comma, Dot, Colon, Semicolon, Pipe,

    // Special
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub lexeme: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug)]
pub struct LexerError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

pub struct Lexer<'a> {
    source: Peekable<Chars<'a>>,
    current_lexeme: String,
    line: usize,
    column: usize,
    start_column: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source: source.chars().peekable(),
            current_lexeme: String::new(),
            line: 1,
            column: 1,
            start_column: 1,
        }
    }

    pub fn scan_tokens(&mut self) -> Result<Vec<Token>, LexerError> {
        let mut tokens = Vec::new();

        loop {
            self.skip_whitespace();
            self.current_lexeme.clear();
            self.start_column = self.column;

            let token = self.scan_token()?;
            let is_eof = token.kind == TokenKind::Eof;
            tokens.push(token);

            if is_eof {
                break;
            }
        }

        Ok(tokens)
    }

    fn scan_token(&mut self) -> Result<Token, LexerError> {
        let c = match self.advance() {
            Some(c) => c,
            None => return Ok(self.make_token(TokenKind::Eof)),
        };

        match c {
            '(' => Ok(self.make_token(TokenKind::LParen)),
            ')' => Ok(self.make_token(TokenKind::RParen)),
            '{' => Ok(self.make_token(TokenKind::LBrace)),
            '}' => Ok(self.make_token(TokenKind::RBrace)),
            '[' => Ok(self.make_token(TokenKind::LBracket)),
            ']' => Ok(self.make_token(TokenKind::RBracket)),
            ',' => Ok(self.make_token(TokenKind::Comma)),
            '.' => Ok(self.make_token(TokenKind::Dot)),
            ':' => Ok(self.make_token(TokenKind::Colon)),
            ';' => Ok(self.make_token(TokenKind::Semicolon)),
            '+' => Ok(self.make_token(TokenKind::Plus)),
            '*' => Ok(self.make_token(TokenKind::Star)),
            '%' => Ok(self.make_token(TokenKind::Percent)),
            '|' => {
                if self.match_char('|') {
                    Ok(self.make_token(TokenKind::Or))
                } else {
                    Ok(self.make_token(TokenKind::Pipe))
                }
            }
            '&' => {
                if self.match_char('&') {
                    Ok(self.make_token(TokenKind::And))
                } else {
                    Err(self.error("Unexpected '&'"))
                }
            }
            '-' => {
                if self.match_char('>') {
                    Ok(self.make_token(TokenKind::Arrow))
                } else {
                    Ok(self.make_token(TokenKind::Minus))
                }
            }
            '=' => {
                if self.match_char('=') {
                    Ok(self.make_token(TokenKind::EqEq))
                } else if self.match_char('>') {
                    Ok(self.make_token(TokenKind::FatArrow))
                } else {
                    Ok(self.make_token(TokenKind::Eq))
                }
            }
            '!' => {
                if self.match_char('=') {
                    Ok(self.make_token(TokenKind::BangEq))
                } else {
                    Ok(self.make_token(TokenKind::Bang))
                }
            }
            '<' => {
                if self.match_char('=') {
                    Ok(self.make_token(TokenKind::LtEq))
                } else {
                    Ok(self.make_token(TokenKind::Lt))
                }
            }
            '>' => {
                if self.match_char('=') {
                    Ok(self.make_token(TokenKind::GtEq))
                } else {
                    Ok(self.make_token(TokenKind::Gt))
                }
            }
            '/' => {
                if self.match_char('/') {
                    // Line comment
                    while self.peek() != Some('\\n') && self.peek().is_some() {
                        self.advance();
                    }
                    self.scan_token()
                } else if self.match_char('*') {
                    self.block_comment()?;
                    self.scan_token()
                } else {
                    Ok(self.make_token(TokenKind::Slash))
                }
            }
            '"' => self.string(),
            c if c.is_ascii_digit() => self.number(),
            c if c.is_alphabetic() || c == '_' => self.identifier(),
            _ => Err(self.error(&format!("Unexpected character '{}'", c))),
        }
    }

    fn string(&mut self) -> Result<Token, LexerError> {
        let mut value = String::new();

        while let Some(&c) = self.source.peek() {
            if c == '"' {
                break;
            }

            let ch = self.advance().unwrap();

            if ch == '\\\\' {
                match self.advance() {
                    Some('n') => value.push('\\n'),
                    Some('t') => value.push('\\t'),
                    Some('"') => value.push('"'),
                    Some('\\\\') => value.push('\\\\'),
                    Some(c) => value.push(c),
                    None => return Err(self.error("Unterminated string")),
                }
            } else {
                if ch == '\\n' {
                    self.line += 1;
                    self.column = 1;
                }
                value.push(ch);
            }
        }

        if self.advance() != Some('"') {
            return Err(self.error("Unterminated string"));
        }

        Ok(self.make_token(TokenKind::String(value)))
    }

    fn number(&mut self) -> Result<Token, LexerError> {
        while self.peek().map_or(false, |c| c.is_ascii_digit()) {
            self.advance();
        }

        if self.peek() == Some('.') {
            if self.peek_next().map_or(false, |c| c.is_ascii_digit()) {
                self.advance(); // consume '.'
                while self.peek().map_or(false, |c| c.is_ascii_digit()) {
                    self.advance();
                }
            }
        }

        let value: f64 = self.current_lexeme.parse().unwrap();
        Ok(self.make_token(TokenKind::Number(value)))
    }

    fn identifier(&mut self) -> Result<Token, LexerError> {
        while self.peek().map_or(false, |c| c.is_alphanumeric() || c == '_') {
            self.advance();
        }

        let kind = match self.current_lexeme.as_str() {
            "let" => TokenKind::Let,
            "mut" => TokenKind::Mut,
            "fn" => TokenKind::Fn,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "return" => TokenKind::Return,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "nil" => TokenKind::Nil,
            "struct" => TokenKind::Struct,
            "enum" => TokenKind::Enum,
            "impl" => TokenKind::Impl,
            "match" => TokenKind::Match,
            _ => TokenKind::Identifier(self.current_lexeme.clone()),
        };

        Ok(self.make_token(kind))
    }

    fn block_comment(&mut self) -> Result<(), LexerError> {
        let mut depth = 1;

        while depth > 0 {
            match (self.peek(), self.peek_next()) {
                (Some('/'), Some('*')) => {
                    self.advance();
                    self.advance();
                    depth += 1;
                }
                (Some('*'), Some('/')) => {
                    self.advance();
                    self.advance();
                    depth -= 1;
                }
                (Some('\\n'), _) => {
                    self.advance();
                    self.line += 1;
                    self.column = 1;
                }
                (Some(_), _) => {
                    self.advance();
                }
                (None, _) => {
                    return Err(self.error("Unterminated block comment"));
                }
            }
        }

        Ok(())
    }

    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.source.peek() {
            match c {
                ' ' | '\\r' | '\\t' => {
                    self.advance();
                }
                '\\n' => {
                    self.advance();
                    self.line += 1;
                    self.column = 1;
                }
                _ => break,
            }
        }
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.source.next()?;
        self.current_lexeme.push(c);
        self.column += 1;
        Some(c)
    }

    fn peek(&mut self) -> Option<char> {
        self.source.peek().copied()
    }

    fn peek_next(&self) -> Option<char> {
        let mut iter = self.source.clone();
        iter.next();
        iter.next()
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn make_token(&self, kind: TokenKind) -> Token {
        Token {
            kind,
            lexeme: self.current_lexeme.clone(),
            line: self.line,
            column: self.start_column,
        }
    }

    fn error(&self, message: &str) -> LexerError {
        LexerError {
            message: message.to_string(),
            line: self.line,
            column: self.start_column,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokens() {
        let mut lexer = Lexer::new("let x = 42;");
        let tokens = lexer.scan_tokens().unwrap();

        assert_eq!(tokens[0].kind, TokenKind::Let);
        assert!(matches!(tokens[1].kind, TokenKind::Identifier(_)));
        assert_eq!(tokens[2].kind, TokenKind::Eq);
        assert!(matches!(tokens[3].kind, TokenKind::Number(42.0)));
        assert_eq!(tokens[4].kind, TokenKind::Semicolon);
    }
}
\`\`\`
`, 0, now);

// ============================================================================
// COMPILER IN GO
// ============================================================================
const goCompilerSchedule = `## 6-Week Schedule

### Week 1: Go Lexer
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Setup | Go modules, token types |
| Day 2 | Scanner | Rune iteration, position tracking |
| Day 3 | Tokenization | All token types |
| Day 4 | Testing | Table-driven tests |
| Day 5 | Polish | Error messages |
| Weekend | Buffer | Integration tests |

### Week 2: Parser
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | AST | Define AST structs |
| Day 2 | Expressions | Expression parsing |
| Day 3 | Statements | Statement parsing |
| Day 4 | Functions | Functions and calls |
| Day 5 | Errors | Error recovery |
| Weekend | Testing | Parser tests |

### Week 3-4: Semantic Analysis
| Day | Focus | Tasks |
|-----|-------|-------|
| Week 3 | Types | Type system |
| Week 4 | Checking | Semantic checks |

### Week 5-6: Bytecode
| Day | Focus | Tasks |
|-----|-------|-------|
| Week 5 | Compiler | Bytecode emission |
| Week 6 | VM | Stack-based VM |

### Daily Commitment
- **Minimum**: 2 hours
- **Ideal**: 4 hours`;

const goCompilerPath = insertPath.run(
	'Build a Compiler in Go',
	'Implement a compiler in Go with a bytecode VM. Clean Go idioms, interfaces for AST nodes, and a stack-based virtual machine.',
	'cyan',
	'Go',
	'advanced',
	6,
	'Go, lexing, parsing, bytecode, virtual machines',
	goCompilerSchedule,
	now
);

const goCompMod1 = insertModule.run(goCompilerPath.lastInsertRowid, 'Go Compiler Core', 'Build the compiler in Go', 0, now);

insertTask.run(goCompMod1.lastInsertRowid, 'Build Go Lexer and Parser', 'Implement a lexer and recursive descent parser in Go that tokenizes source code and constructs an AST, leveraging Go interfaces for node types and built-in testing for validation', `## Go Compiler Implementation

### Lexer

\`\`\`go
package lexer

import (
    "fmt"
    "unicode"
)

type TokenType int

const (
    // Literals
    NUMBER TokenType = iota
    STRING
    IDENT

    // Keywords
    LET
    FN
    IF
    ELSE
    WHILE
    RETURN
    TRUE
    FALSE

    // Operators
    PLUS
    MINUS
    STAR
    SLASH
    EQ
    EQ_EQ
    BANG
    BANG_EQ
    LT
    LT_EQ
    GT
    GT_EQ

    // Delimiters
    LPAREN
    RPAREN
    LBRACE
    RBRACE
    COMMA
    SEMICOLON
    COLON
    ARROW

    EOF
    ILLEGAL
)

var keywords = map[string]TokenType{
    "let":    LET,
    "fn":     FN,
    "if":     IF,
    "else":   ELSE,
    "while":  WHILE,
    "return": RETURN,
    "true":   TRUE,
    "false":  FALSE,
}

type Token struct {
    Type    TokenType
    Literal string
    Line    int
    Column  int
}

type Lexer struct {
    input   string
    pos     int
    readPos int
    ch      byte
    line    int
    column  int
}

func New(input string) *Lexer {
    l := &Lexer{input: input, line: 1, column: 0}
    l.readChar()
    return l
}

func (l *Lexer) NextToken() Token {
    l.skipWhitespace()

    startCol := l.column

    var tok Token
    tok.Line = l.line
    tok.Column = startCol

    switch l.ch {
    case '(':
        tok = l.newToken(LPAREN)
    case ')':
        tok = l.newToken(RPAREN)
    case '{':
        tok = l.newToken(LBRACE)
    case '}':
        tok = l.newToken(RBRACE)
    case ',':
        tok = l.newToken(COMMA)
    case ';':
        tok = l.newToken(SEMICOLON)
    case ':':
        tok = l.newToken(COLON)
    case '+':
        tok = l.newToken(PLUS)
    case '*':
        tok = l.newToken(STAR)
    case '/':
        if l.peekChar() == '/' {
            l.skipLineComment()
            return l.NextToken()
        }
        tok = l.newToken(SLASH)
    case '-':
        if l.peekChar() == '>' {
            l.readChar()
            tok = Token{Type: ARROW, Literal: "->", Line: l.line, Column: startCol}
        } else {
            tok = l.newToken(MINUS)
        }
    case '=':
        if l.peekChar() == '=' {
            l.readChar()
            tok = Token{Type: EQ_EQ, Literal: "==", Line: l.line, Column: startCol}
        } else {
            tok = l.newToken(EQ)
        }
    case '!':
        if l.peekChar() == '=' {
            l.readChar()
            tok = Token{Type: BANG_EQ, Literal: "!=", Line: l.line, Column: startCol}
        } else {
            tok = l.newToken(BANG)
        }
    case '<':
        if l.peekChar() == '=' {
            l.readChar()
            tok = Token{Type: LT_EQ, Literal: "<=", Line: l.line, Column: startCol}
        } else {
            tok = l.newToken(LT)
        }
    case '>':
        if l.peekChar() == '=' {
            l.readChar()
            tok = Token{Type: GT_EQ, Literal: ">=", Line: l.line, Column: startCol}
        } else {
            tok = l.newToken(GT)
        }
    case '"':
        tok.Type = STRING
        tok.Literal = l.readString()
    case 0:
        tok.Type = EOF
        tok.Literal = ""
    default:
        if isDigit(l.ch) {
            tok.Type = NUMBER
            tok.Literal = l.readNumber()
            return tok
        } else if isLetter(l.ch) {
            tok.Literal = l.readIdentifier()
            tok.Type = lookupIdent(tok.Literal)
            return tok
        } else {
            tok = Token{Type: ILLEGAL, Literal: string(l.ch), Line: l.line, Column: startCol}
        }
    }

    l.readChar()
    return tok
}

func (l *Lexer) readChar() {
    if l.readPos >= len(l.input) {
        l.ch = 0
    } else {
        l.ch = l.input[l.readPos]
    }
    l.pos = l.readPos
    l.readPos++
    l.column++
}

func (l *Lexer) peekChar() byte {
    if l.readPos >= len(l.input) {
        return 0
    }
    return l.input[l.readPos]
}

func (l *Lexer) skipWhitespace() {
    for l.ch == ' ' || l.ch == '\\t' || l.ch == '\\n' || l.ch == '\\r' {
        if l.ch == '\\n' {
            l.line++
            l.column = 0
        }
        l.readChar()
    }
}

func (l *Lexer) skipLineComment() {
    for l.ch != '\\n' && l.ch != 0 {
        l.readChar()
    }
}

func (l *Lexer) readString() string {
    l.readChar() // skip opening "
    start := l.pos
    for l.ch != '"' && l.ch != 0 {
        l.readChar()
    }
    return l.input[start:l.pos]
}

func (l *Lexer) readNumber() string {
    start := l.pos
    for isDigit(l.ch) {
        l.readChar()
    }
    if l.ch == '.' && isDigit(l.peekChar()) {
        l.readChar()
        for isDigit(l.ch) {
            l.readChar()
        }
    }
    return l.input[start:l.pos]
}

func (l *Lexer) readIdentifier() string {
    start := l.pos
    for isLetter(l.ch) || isDigit(l.ch) {
        l.readChar()
    }
    return l.input[start:l.pos]
}

func (l *Lexer) newToken(tokenType TokenType) Token {
    return Token{Type: tokenType, Literal: string(l.ch), Line: l.line, Column: l.column}
}

func lookupIdent(ident string) TokenType {
    if tok, ok := keywords[ident]; ok {
        return tok
    }
    return IDENT
}

func isDigit(ch byte) bool {
    return ch >= '0' && ch <= '9'
}

func isLetter(ch byte) bool {
    return unicode.IsLetter(rune(ch)) || ch == '_'
}
\`\`\`

### Bytecode VM

\`\`\`go
package vm

type OpCode byte

const (
    OpConstant OpCode = iota
    OpAdd
    OpSub
    OpMul
    OpDiv
    OpTrue
    OpFalse
    OpEqual
    OpNotEqual
    OpGreater
    OpLess
    OpMinus
    OpBang
    OpJump
    OpJumpNotTruthy
    OpNull
    OpGetGlobal
    OpSetGlobal
    OpGetLocal
    OpSetLocal
    OpCall
    OpReturn
    OpReturnValue
    OpPop
)

type VM struct {
    constants    []interface{}
    instructions []byte
    stack        []interface{}
    sp           int // stack pointer
    globals      []interface{}
}

func New(instructions []byte, constants []interface{}) *VM {
    return &VM{
        instructions: instructions,
        constants:    constants,
        stack:        make([]interface{}, 1024),
        sp:           0,
        globals:      make([]interface{}, 65536),
    }
}

func (vm *VM) Run() error {
    for ip := 0; ip < len(vm.instructions); ip++ {
        op := OpCode(vm.instructions[ip])

        switch op {
        case OpConstant:
            constIndex := int(vm.instructions[ip+1])<<8 | int(vm.instructions[ip+2])
            ip += 2
            vm.push(vm.constants[constIndex])

        case OpAdd, OpSub, OpMul, OpDiv:
            right := vm.pop()
            left := vm.pop()
            vm.push(vm.binaryOp(op, left, right))

        case OpTrue:
            vm.push(true)

        case OpFalse:
            vm.push(false)

        case OpEqual:
            right := vm.pop()
            left := vm.pop()
            vm.push(left == right)

        case OpGreater:
            right := vm.pop().(float64)
            left := vm.pop().(float64)
            vm.push(left > right)

        case OpMinus:
            operand := vm.pop().(float64)
            vm.push(-operand)

        case OpBang:
            operand := vm.pop()
            vm.push(!vm.isTruthy(operand))

        case OpJump:
            pos := int(vm.instructions[ip+1])<<8 | int(vm.instructions[ip+2])
            ip = pos - 1

        case OpJumpNotTruthy:
            pos := int(vm.instructions[ip+1])<<8 | int(vm.instructions[ip+2])
            ip += 2
            condition := vm.pop()
            if !vm.isTruthy(condition) {
                ip = pos - 1
            }

        case OpPop:
            vm.pop()

        case OpGetGlobal:
            index := int(vm.instructions[ip+1])<<8 | int(vm.instructions[ip+2])
            ip += 2
            vm.push(vm.globals[index])

        case OpSetGlobal:
            index := int(vm.instructions[ip+1])<<8 | int(vm.instructions[ip+2])
            ip += 2
            vm.globals[index] = vm.pop()
        }
    }
    return nil
}

func (vm *VM) push(value interface{}) {
    vm.stack[vm.sp] = value
    vm.sp++
}

func (vm *VM) pop() interface{} {
    vm.sp--
    return vm.stack[vm.sp]
}

func (vm *VM) binaryOp(op OpCode, left, right interface{}) interface{} {
    l := left.(float64)
    r := right.(float64)

    switch op {
    case OpAdd:
        return l + r
    case OpSub:
        return l - r
    case OpMul:
        return l * r
    case OpDiv:
        return l / r
    }
    return nil
}

func (vm *VM) isTruthy(obj interface{}) bool {
    switch v := obj.(type) {
    case bool:
        return v
    case nil:
        return false
    default:
        return true
    }
}

func (vm *VM) StackTop() interface{} {
    if vm.sp == 0 {
        return nil
    }
    return vm.stack[vm.sp-1]
}
\`\`\`
`, 0, now);

// ============================================================================
// C COMPILER
// ============================================================================
const cCompilerPath = insertPath.run(
	'Build a Compiler in C',
	'Implement a compiler in C for maximum control. Manual memory management, hand-written lexer, recursive descent parser, and native code generation.',
	'red',
	'C',
	'expert',
	8,
	'C, memory management, lexing, parsing, x86 assembly, ELF',
	`## 8-Week Schedule

### Week 1-2: Lexer
Build a hand-written lexer in C with proper memory management.

### Week 3-4: Parser
Recursive descent parser producing AST in C structs.

### Week 5-6: Semantic Analysis
Type checking and symbol table management.

### Week 7-8: Code Generation
Generate x86-64 assembly or compile to ELF directly.

### Daily Commitment
- **Minimum**: 3 hours
- **Ideal**: 5 hours`,
	now
);

const cCompMod1 = insertModule.run(cCompilerPath.lastInsertRowid, 'C Compiler Basics', 'Low-level compiler implementation', 0, now);

insertTask.run(cCompMod1.lastInsertRowid, 'Build C Lexer', 'Write a hand-rolled lexer in C using character-by-character processing, managing token buffers, handling escape sequences, and implementing efficient lookahead without external lexer generator dependencies', `## C Lexer Implementation

\`\`\`c
// lexer.h
#ifndef LEXER_H
#define LEXER_H

#include <stddef.h>

typedef enum {
    TOK_EOF = 0,
    TOK_NUMBER,
    TOK_STRING,
    TOK_IDENT,

    // Keywords
    TOK_LET,
    TOK_FN,
    TOK_IF,
    TOK_ELSE,
    TOK_WHILE,
    TOK_RETURN,

    // Operators
    TOK_PLUS,
    TOK_MINUS,
    TOK_STAR,
    TOK_SLASH,
    TOK_EQ,
    TOK_EQ_EQ,
    TOK_BANG,
    TOK_BANG_EQ,
    TOK_LT,
    TOK_GT,

    // Delimiters
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_LBRACE,
    TOK_RBRACE,
    TOK_COMMA,
    TOK_SEMICOLON,
    TOK_COLON,
    TOK_ARROW,

    TOK_ERROR
} TokenType;

typedef struct {
    TokenType type;
    const char* start;
    size_t length;
    int line;
    int column;

    // For literals
    union {
        double number;
        char* string;
    } value;
} Token;

typedef struct {
    const char* source;
    const char* start;
    const char* current;
    int line;
    int column;
} Lexer;

Lexer* lexer_new(const char* source);
void lexer_free(Lexer* lexer);
Token lexer_next_token(Lexer* lexer);
const char* token_type_name(TokenType type);

#endif

// lexer.c
#include "lexer.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

static Token make_token(Lexer* l, TokenType type) {
    Token tok;
    tok.type = type;
    tok.start = l->start;
    tok.length = l->current - l->start;
    tok.line = l->line;
    tok.column = l->column - tok.length;
    return tok;
}

static Token error_token(Lexer* l, const char* msg) {
    Token tok;
    tok.type = TOK_ERROR;
    tok.start = msg;
    tok.length = strlen(msg);
    tok.line = l->line;
    tok.column = l->column;
    return tok;
}

static char advance(Lexer* l) {
    l->column++;
    return *l->current++;
}

static char peek(Lexer* l) {
    return *l->current;
}

static char peek_next(Lexer* l) {
    if (*l->current == '\\0') return '\\0';
    return l->current[1];
}

static int match(Lexer* l, char expected) {
    if (*l->current != expected) return 0;
    l->current++;
    l->column++;
    return 1;
}

static int is_at_end(Lexer* l) {
    return *l->current == '\\0';
}

static void skip_whitespace(Lexer* l) {
    for (;;) {
        char c = peek(l);
        switch (c) {
            case ' ':
            case '\\r':
            case '\\t':
                advance(l);
                break;
            case '\\n':
                l->line++;
                l->column = 0;
                advance(l);
                break;
            case '/':
                if (peek_next(l) == '/') {
                    while (peek(l) != '\\n' && !is_at_end(l))
                        advance(l);
                } else {
                    return;
                }
                break;
            default:
                return;
        }
    }
}

static TokenType check_keyword(Lexer* l, int start, int len,
                               const char* rest, TokenType type) {
    if (l->current - l->start == start + len &&
        memcmp(l->start + start, rest, len) == 0) {
        return type;
    }
    return TOK_IDENT;
}

static TokenType identifier_type(Lexer* l) {
    switch (l->start[0]) {
        case 'e': return check_keyword(l, 1, 3, "lse", TOK_ELSE);
        case 'f': return check_keyword(l, 1, 1, "n", TOK_FN);
        case 'i': return check_keyword(l, 1, 1, "f", TOK_IF);
        case 'l': return check_keyword(l, 1, 2, "et", TOK_LET);
        case 'r': return check_keyword(l, 1, 5, "eturn", TOK_RETURN);
        case 'w': return check_keyword(l, 1, 4, "hile", TOK_WHILE);
    }
    return TOK_IDENT;
}

static Token identifier(Lexer* l) {
    while (isalnum(peek(l)) || peek(l) == '_')
        advance(l);
    return make_token(l, identifier_type(l));
}

static Token number(Lexer* l) {
    while (isdigit(peek(l)))
        advance(l);

    if (peek(l) == '.' && isdigit(peek_next(l))) {
        advance(l);
        while (isdigit(peek(l)))
            advance(l);
    }

    Token tok = make_token(l, TOK_NUMBER);
    tok.value.number = strtod(l->start, NULL);
    return tok;
}

static Token string(Lexer* l) {
    while (peek(l) != '"' && !is_at_end(l)) {
        if (peek(l) == '\\n') {
            l->line++;
            l->column = 0;
        }
        advance(l);
    }

    if (is_at_end(l))
        return error_token(l, "Unterminated string");

    advance(l); // closing "

    Token tok = make_token(l, TOK_STRING);
    size_t len = tok.length - 2;
    tok.value.string = malloc(len + 1);
    memcpy(tok.value.string, tok.start + 1, len);
    tok.value.string[len] = '\\0';
    return tok;
}

Lexer* lexer_new(const char* source) {
    Lexer* l = malloc(sizeof(Lexer));
    l->source = source;
    l->start = source;
    l->current = source;
    l->line = 1;
    l->column = 1;
    return l;
}

void lexer_free(Lexer* lexer) {
    free(lexer);
}

Token lexer_next_token(Lexer* l) {
    skip_whitespace(l);
    l->start = l->current;

    if (is_at_end(l))
        return make_token(l, TOK_EOF);

    char c = advance(l);

    if (isalpha(c) || c == '_') return identifier(l);
    if (isdigit(c)) return number(l);

    switch (c) {
        case '(': return make_token(l, TOK_LPAREN);
        case ')': return make_token(l, TOK_RPAREN);
        case '{': return make_token(l, TOK_LBRACE);
        case '}': return make_token(l, TOK_RBRACE);
        case ',': return make_token(l, TOK_COMMA);
        case ';': return make_token(l, TOK_SEMICOLON);
        case ':': return make_token(l, TOK_COLON);
        case '+': return make_token(l, TOK_PLUS);
        case '*': return make_token(l, TOK_STAR);
        case '/': return make_token(l, TOK_SLASH);
        case '-':
            return make_token(l, match(l, '>') ? TOK_ARROW : TOK_MINUS);
        case '=':
            return make_token(l, match(l, '=') ? TOK_EQ_EQ : TOK_EQ);
        case '!':
            return make_token(l, match(l, '=') ? TOK_BANG_EQ : TOK_BANG);
        case '<': return make_token(l, TOK_LT);
        case '>': return make_token(l, TOK_GT);
        case '"': return string(l);
    }

    return error_token(l, "Unexpected character");
}
\`\`\`
`, 0, now);

console.log('Seeded: Compiler & Language Development');
console.log('  - Build Your Own Programming Language (8 weeks)');
console.log('  - Build a Compiler in Rust (6 weeks)');
console.log('  - Build a Compiler in Go (6 weeks)');
console.log('  - Build a Compiler in C (8 weeks)');
