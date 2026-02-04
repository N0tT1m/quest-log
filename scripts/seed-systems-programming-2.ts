import Database from 'better-sqlite3';
import { join } from 'path';

const db = new Database(join(process.cwd(), 'data', 'quest-log.db'));

// Memory Allocator, Thread Pool, Async Runtime
const paths = [
  {
    name: 'Build Your Own Memory Allocator',
    description: 'Create a malloc implementation with free lists, coalescing, and arena allocation',
    icon: 'cpu',
    color: 'purple',
    language: 'C, C++, Rust',
    skills: 'Memory management, Data structures, Systems programming',
    difficulty: 'advanced',
    estimated_weeks: 4,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Memory basics | sbrk/mmap intro |
| 1 | 2 | Block header | Metadata structure |
| 1 | 3 | First-fit | Basic allocation |
| 1 | 4 | Free implementation | Block freeing |
| 1 | 5 | Block splitting | Size optimization |
| 2 | 1 | Coalescing | Merge free blocks |
| 2 | 2 | Boundary tags | Footer for coalescing |
| 2 | 3 | Best-fit | Search optimization |
| 2 | 4 | Free list | Explicit free list |
| 2 | 5 | Segregated lists | Size classes |
| 3 | 1 | Arena allocator | Region-based |
| 3 | 2 | Pool allocator | Fixed-size blocks |
| 3 | 3 | Slab allocator | Object caching |
| 3 | 4 | Thread safety | Locking |
| 3 | 5 | Thread-local | Per-thread arenas |
| 4 | 1 | realloc | Resize implementation |
| 4 | 2 | calloc | Zero-init allocation |
| 4 | 3 | Debugging | Memory leak detection |
| 4 | 4 | Performance | Benchmarking |
| 4 | 5 | Integration | Full allocator |`,
    modules: [
      {
        name: 'Allocator Implementation',
        description: 'Complete memory allocator',
        tasks: [
          {
            title: 'Memory Allocator in C',
            description: 'Implement malloc/free with free list and coalescing',
            details: `# Custom Memory Allocator in C

\`\`\`c
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>

// Block header structure
typedef struct block_header {
    size_t size;            // Size of the block (including header)
    bool is_free;           // Is this block free?
    struct block_header *next;  // Next block in free list
    struct block_header *prev;  // Previous block in free list
} block_header_t;

// Footer for coalescing (only for allocated blocks)
typedef struct block_footer {
    size_t size;
} block_footer_t;

#define HEADER_SIZE sizeof(block_header_t)
#define FOOTER_SIZE sizeof(block_footer_t)
#define MIN_BLOCK_SIZE (HEADER_SIZE + FOOTER_SIZE + 16)
#define ALIGNMENT 16
#define ALIGN(size) (((size) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

// Free list head
static block_header_t *free_list = NULL;
static void *heap_start = NULL;
static void *heap_end = NULL;
static pthread_mutex_t alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

// Get footer from header
static inline block_footer_t *get_footer(block_header_t *header) {
    return (block_footer_t *)((char *)header + header->size - FOOTER_SIZE);
}

// Get header from footer
static inline block_header_t *get_header_from_footer(block_footer_t *footer) {
    return (block_header_t *)((char *)footer - footer->size + FOOTER_SIZE);
}

// Get next physical block
static inline block_header_t *next_block(block_header_t *header) {
    return (block_header_t *)((char *)header + header->size);
}

// Get previous physical block
static inline block_header_t *prev_block(block_header_t *header) {
    block_footer_t *prev_footer = (block_footer_t *)((char *)header - FOOTER_SIZE);
    if ((void *)prev_footer < heap_start) return NULL;
    return get_header_from_footer(prev_footer);
}

// Add block to free list
static void add_to_free_list(block_header_t *block) {
    block->is_free = true;
    block->next = free_list;
    block->prev = NULL;

    if (free_list) {
        free_list->prev = block;
    }
    free_list = block;
}

// Remove block from free list
static void remove_from_free_list(block_header_t *block) {
    if (block->prev) {
        block->prev->next = block->next;
    } else {
        free_list = block->next;
    }

    if (block->next) {
        block->next->prev = block->prev;
    }

    block->is_free = false;
}

// Request more memory from OS
static block_header_t *request_memory(size_t size) {
    size_t request_size = size < 4096 * 16 ? 4096 * 16 : size;
    request_size = ALIGN(request_size);

    void *mem = mmap(NULL, request_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (mem == MAP_FAILED) {
        return NULL;
    }

    if (!heap_start || mem < heap_start) {
        heap_start = mem;
    }
    heap_end = (char *)mem + request_size;

    block_header_t *block = (block_header_t *)mem;
    block->size = request_size;
    block->is_free = true;

    block_footer_t *footer = get_footer(block);
    footer->size = request_size;

    return block;
}

// Split a block if it's large enough
static void split_block(block_header_t *block, size_t needed_size) {
    size_t remaining = block->size - needed_size;

    if (remaining >= MIN_BLOCK_SIZE) {
        // Create new block from remaining space
        block_header_t *new_block = (block_header_t *)((char *)block + needed_size);
        new_block->size = remaining;
        new_block->is_free = true;

        // Update footers
        block_footer_t *new_footer = get_footer(new_block);
        new_footer->size = remaining;

        block->size = needed_size;
        block_footer_t *footer = get_footer(block);
        footer->size = needed_size;

        // Add new block to free list
        add_to_free_list(new_block);
    }
}

// Coalesce with adjacent free blocks
static block_header_t *coalesce(block_header_t *block) {
    block_header_t *next = next_block(block);
    block_header_t *prev = prev_block(block);

    bool prev_free = prev && prev->is_free;
    bool next_free = next && (void *)next < heap_end && next->is_free;

    if (prev_free && next_free) {
        // Coalesce with both
        remove_from_free_list(prev);
        remove_from_free_list(next);
        remove_from_free_list(block);

        prev->size += block->size + next->size;
        block_footer_t *footer = get_footer(next);
        footer->size = prev->size;

        add_to_free_list(prev);
        return prev;
    } else if (prev_free) {
        // Coalesce with previous
        remove_from_free_list(prev);
        remove_from_free_list(block);

        prev->size += block->size;
        block_footer_t *footer = get_footer(block);
        footer->size = prev->size;

        add_to_free_list(prev);
        return prev;
    } else if (next_free) {
        // Coalesce with next
        remove_from_free_list(next);
        remove_from_free_list(block);

        block->size += next->size;
        block_footer_t *footer = get_footer(next);
        footer->size = block->size;

        add_to_free_list(block);
        return block;
    }

    return block;
}

// Find best-fit free block
static block_header_t *find_best_fit(size_t size) {
    block_header_t *best = NULL;
    block_header_t *current = free_list;

    while (current) {
        if (current->is_free && current->size >= size) {
            if (!best || current->size < best->size) {
                best = current;
                if (current->size == size) break;  // Exact fit
            }
        }
        current = current->next;
    }

    return best;
}

// malloc implementation
void *my_malloc(size_t size) {
    if (size == 0) return NULL;

    pthread_mutex_lock(&alloc_mutex);

    size_t total_size = ALIGN(size + HEADER_SIZE + FOOTER_SIZE);
    if (total_size < MIN_BLOCK_SIZE) {
        total_size = MIN_BLOCK_SIZE;
    }

    // Try to find a free block
    block_header_t *block = find_best_fit(total_size);

    if (block) {
        remove_from_free_list(block);
        split_block(block, total_size);
    } else {
        // Request more memory
        block = request_memory(total_size);
        if (!block) {
            pthread_mutex_unlock(&alloc_mutex);
            return NULL;
        }
        split_block(block, total_size);
    }

    block->is_free = false;

    pthread_mutex_unlock(&alloc_mutex);

    return (void *)((char *)block + HEADER_SIZE);
}

// free implementation
void my_free(void *ptr) {
    if (!ptr) return;

    pthread_mutex_lock(&alloc_mutex);

    block_header_t *block = (block_header_t *)((char *)ptr - HEADER_SIZE);
    add_to_free_list(block);
    coalesce(block);

    pthread_mutex_unlock(&alloc_mutex);
}

// realloc implementation
void *my_realloc(void *ptr, size_t size) {
    if (!ptr) return my_malloc(size);
    if (size == 0) {
        my_free(ptr);
        return NULL;
    }

    block_header_t *block = (block_header_t *)((char *)ptr - HEADER_SIZE);
    size_t old_size = block->size - HEADER_SIZE - FOOTER_SIZE;

    if (size <= old_size) {
        // Shrink: could split but we'll keep it simple
        return ptr;
    }

    // Need to grow
    void *new_ptr = my_malloc(size);
    if (new_ptr) {
        memcpy(new_ptr, ptr, old_size);
        my_free(ptr);
    }

    return new_ptr;
}

// calloc implementation
void *my_calloc(size_t count, size_t size) {
    size_t total = count * size;
    void *ptr = my_malloc(total);
    if (ptr) {
        memset(ptr, 0, total);
    }
    return ptr;
}

// Debug: print heap status
void heap_dump() {
    printf("=== Heap Dump ===\\n");
    printf("Free list:\\n");

    block_header_t *current = free_list;
    while (current) {
        printf("  Block at %p, size=%zu, free=%d\\n",
               (void *)current, current->size, current->is_free);
        current = current->next;
    }
}

// Arena allocator for fast bulk allocations
typedef struct arena {
    void *start;
    void *current;
    size_t size;
    size_t remaining;
} arena_t;

arena_t *arena_create(size_t size) {
    arena_t *arena = my_malloc(sizeof(arena_t));
    if (!arena) return NULL;

    arena->start = mmap(NULL, size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena->start == MAP_FAILED) {
        my_free(arena);
        return NULL;
    }

    arena->current = arena->start;
    arena->size = size;
    arena->remaining = size;

    return arena;
}

void *arena_alloc(arena_t *arena, size_t size) {
    size = ALIGN(size);
    if (size > arena->remaining) {
        return NULL;
    }

    void *ptr = arena->current;
    arena->current = (char *)arena->current + size;
    arena->remaining -= size;

    return ptr;
}

void arena_reset(arena_t *arena) {
    arena->current = arena->start;
    arena->remaining = arena->size;
}

void arena_destroy(arena_t *arena) {
    munmap(arena->start, arena->size);
    my_free(arena);
}

// Pool allocator for fixed-size objects
typedef struct pool {
    void *memory;
    void *free_list;
    size_t block_size;
    size_t num_blocks;
} pool_t;

pool_t *pool_create(size_t block_size, size_t num_blocks) {
    if (block_size < sizeof(void *)) {
        block_size = sizeof(void *);
    }
    block_size = ALIGN(block_size);

    pool_t *pool = my_malloc(sizeof(pool_t));
    if (!pool) return NULL;

    size_t total_size = block_size * num_blocks;
    pool->memory = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pool->memory == MAP_FAILED) {
        my_free(pool);
        return NULL;
    }

    pool->block_size = block_size;
    pool->num_blocks = num_blocks;

    // Initialize free list
    pool->free_list = pool->memory;
    char *current = pool->memory;
    for (size_t i = 0; i < num_blocks - 1; i++) {
        char *next = current + block_size;
        *(void **)current = next;
        current = next;
    }
    *(void **)current = NULL;

    return pool;
}

void *pool_alloc(pool_t *pool) {
    if (!pool->free_list) return NULL;

    void *block = pool->free_list;
    pool->free_list = *(void **)block;

    return block;
}

void pool_free(pool_t *pool, void *ptr) {
    *(void **)ptr = pool->free_list;
    pool->free_list = ptr;
}

void pool_destroy(pool_t *pool) {
    munmap(pool->memory, pool->block_size * pool->num_blocks);
    my_free(pool);
}

// Test
int main() {
    printf("Testing custom allocator...\\n");

    // Basic allocation
    int *arr = my_malloc(sizeof(int) * 100);
    for (int i = 0; i < 100; i++) arr[i] = i;
    printf("Allocated array of 100 ints\\n");

    // Realloc
    arr = my_realloc(arr, sizeof(int) * 200);
    printf("Reallocated to 200 ints\\n");

    my_free(arr);
    printf("Freed array\\n");

    // Arena allocator test
    arena_t *arena = arena_create(1024 * 1024);  // 1MB
    for (int i = 0; i < 1000; i++) {
        void *p = arena_alloc(arena, 100);
    }
    printf("Arena allocated 1000 blocks\\n");
    arena_reset(arena);
    printf("Arena reset\\n");
    arena_destroy(arena);

    // Pool allocator test
    pool_t *pool = pool_create(64, 1000);
    void *blocks[100];
    for (int i = 0; i < 100; i++) {
        blocks[i] = pool_alloc(pool);
    }
    printf("Pool allocated 100 blocks\\n");
    for (int i = 0; i < 100; i++) {
        pool_free(pool, blocks[i]);
    }
    printf("Pool freed 100 blocks\\n");
    pool_destroy(pool);

    heap_dump();

    return 0;
}
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Build Your Own Thread Pool',
    description: 'Implement a work-stealing thread pool with task queues and synchronization primitives',
    icon: 'layers',
    color: 'cyan',
    language: 'C, C++, Rust, Go',
    skills: 'Concurrency, Synchronization, Lock-free data structures',
    difficulty: 'intermediate',
    estimated_weeks: 3,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Thread basics | pthread/std::thread |
| 1 | 2 | Mutex/Condition | Synchronization |
| 1 | 3 | Task queue | Thread-safe queue |
| 1 | 4 | Worker threads | Thread creation |
| 1 | 5 | Basic pool | Simple thread pool |
| 2 | 1 | Task submission | Submit interface |
| 2 | 2 | Future/Promise | Async results |
| 2 | 3 | Wait mechanism | Join tasks |
| 2 | 4 | Dynamic sizing | Grow/shrink pool |
| 2 | 5 | Priority queue | Task priorities |
| 3 | 1 | Work stealing | Steal from others |
| 3 | 2 | Lock-free queue | CAS operations |
| 3 | 3 | Parallel for | High-level API |
| 3 | 4 | Graceful shutdown | Clean termination |
| 3 | 5 | Integration | Full testing |`,
    modules: [
      {
        name: 'Thread Pool Implementation',
        description: 'Complete thread pool with work stealing',
        tasks: [
          {
            title: 'Thread Pool in C++',
            description: 'Modern C++ thread pool with futures',
            details: `# Thread Pool Implementation in C++

\`\`\`cpp
#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <memory>
#include <deque>
#include <random>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency())
        : stop_(false), active_tasks_(0) {

        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back(&ThreadPool::worker_thread, this, i);
            local_queues_.push_back(std::make_unique<WorkQueue>());
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        condition_.notify_all();

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // Submit a task and get a future for the result
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type> {

        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> result = task->get_future();

        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_) {
                throw std::runtime_error("ThreadPool is stopped");
            }

            global_queue_.push([task]() { (*task)(); });
            ++active_tasks_;
        }

        condition_.notify_one();
        return result;
    }

    // Submit task with priority (higher = more important)
    template<typename F>
    void submit_priority(int priority, F&& f) {
        std::unique_lock<std::mutex> lock(mutex_);
        priority_queue_.push({priority, std::function<void()>(std::forward<F>(f))});
        ++active_tasks_;
        condition_.notify_one();
    }

    // Wait for all tasks to complete
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_condition_.wait(lock, [this] {
            return active_tasks_ == 0 && global_queue_.empty();
        });
    }

    // Get number of pending tasks
    size_t pending_tasks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return global_queue_.size();
    }

    // Get number of worker threads
    size_t size() const {
        return workers_.size();
    }

private:
    struct PriorityTask {
        int priority;
        std::function<void()> task;

        bool operator<(const PriorityTask& other) const {
            return priority < other.priority;  // Higher priority first
        }
    };

    // Lock-free work queue for work stealing
    class WorkQueue {
    public:
        void push(std::function<void()> task) {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(std::move(task));
        }

        bool pop(std::function<void()>& task) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) return false;
            task = std::move(queue_.front());
            queue_.pop_front();
            return true;
        }

        bool steal(std::function<void()>& task) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) return false;
            task = std::move(queue_.back());
            queue_.pop_back();
            return true;
        }

        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

    private:
        std::deque<std::function<void()>> queue_;
        mutable std::mutex mutex_;
    };

    void worker_thread(size_t id) {
        thread_local static std::mt19937 rng(std::random_device{}());

        while (true) {
            std::function<void()> task;

            // Try to get task from local queue first
            if (local_queues_[id]->pop(task)) {
                execute_task(task);
                continue;
            }

            // Try to get from global queue
            {
                std::unique_lock<std::mutex> lock(mutex_);

                // Check priority queue first
                if (!priority_queue_.empty()) {
                    task = std::move(const_cast<PriorityTask&>(priority_queue_.top()).task);
                    priority_queue_.pop();
                    lock.unlock();
                    execute_task(task);
                    continue;
                }

                if (!global_queue_.empty()) {
                    task = std::move(global_queue_.front());
                    global_queue_.pop();
                    lock.unlock();
                    execute_task(task);
                    continue;
                }

                // Try to steal from other workers
                lock.unlock();
                bool stolen = false;
                size_t victim = rng() % workers_.size();

                for (size_t i = 0; i < workers_.size(); ++i) {
                    size_t target = (victim + i) % workers_.size();
                    if (target != id && local_queues_[target]->steal(task)) {
                        execute_task(task);
                        stolen = true;
                        break;
                    }
                }

                if (stolen) continue;

                // Wait for new tasks
                lock.lock();
                condition_.wait(lock, [this] {
                    return stop_ || !global_queue_.empty() || !priority_queue_.empty();
                });

                if (stop_ && global_queue_.empty() && priority_queue_.empty()) {
                    return;
                }
            }
        }
    }

    void execute_task(std::function<void()>& task) {
        try {
            task();
        } catch (const std::exception& e) {
            std::cerr << "Task exception: " << e.what() << std::endl;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            --active_tasks_;
        }
        done_condition_.notify_all();
    }

    std::vector<std::thread> workers_;
    std::vector<std::unique_ptr<WorkQueue>> local_queues_;
    std::queue<std::function<void()>> global_queue_;
    std::priority_queue<PriorityTask> priority_queue_;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable done_condition_;

    std::atomic<bool> stop_;
    std::atomic<size_t> active_tasks_;
};

// Parallel for implementation
template<typename Iterator, typename Func>
void parallel_for(ThreadPool& pool, Iterator begin, Iterator end, Func f) {
    size_t n = std::distance(begin, end);
    size_t num_threads = pool.size();
    size_t chunk_size = (n + num_threads - 1) / num_threads;

    std::vector<std::future<void>> futures;

    for (Iterator it = begin; it != end;) {
        Iterator chunk_end = it;
        std::advance(chunk_end, std::min(chunk_size, static_cast<size_t>(std::distance(it, end))));

        futures.push_back(pool.submit([=, &f]() {
            for (Iterator i = it; i != chunk_end; ++i) {
                f(*i);
            }
        }));

        it = chunk_end;
    }

    for (auto& fut : futures) {
        fut.get();
    }
}

// Parallel reduce
template<typename Iterator, typename T, typename BinaryOp>
T parallel_reduce(ThreadPool& pool, Iterator begin, Iterator end, T init, BinaryOp op) {
    size_t n = std::distance(begin, end);
    size_t num_threads = pool.size();
    size_t chunk_size = (n + num_threads - 1) / num_threads;

    std::vector<std::future<T>> futures;

    for (Iterator it = begin; it != end;) {
        Iterator chunk_end = it;
        std::advance(chunk_end, std::min(chunk_size, static_cast<size_t>(std::distance(it, end))));

        futures.push_back(pool.submit([=, &op]() {
            T result = T{};
            for (Iterator i = it; i != chunk_end; ++i) {
                result = op(result, *i);
            }
            return result;
        }));

        it = chunk_end;
    }

    T result = init;
    for (auto& fut : futures) {
        result = op(result, fut.get());
    }

    return result;
}

int main() {
    ThreadPool pool(4);

    // Submit individual tasks
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.submit([i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return i * i;
        }));
    }

    // Get results
    std::cout << "Results: ";
    for (auto& fut : futures) {
        std::cout << fut.get() << " ";
    }
    std::cout << std::endl;

    // Priority tasks
    pool.submit_priority(10, []() { std::cout << "High priority\\n"; });
    pool.submit_priority(1, []() { std::cout << "Low priority\\n"; });
    pool.wait();

    // Parallel for
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    parallel_for(pool, data.begin(), data.end(), [](int& x) {
        x = x * x;
    });

    std::cout << "First 10 squares: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    // Parallel reduce
    long long sum = parallel_reduce(pool, data.begin(), data.end(), 0LL,
        [](long long a, int b) { return a + b; });
    std::cout << "Sum of squares: " << sum << std::endl;

    return 0;
}
\`\`\``
          },
          {
            title: 'Thread Pool in Rust',
            description: 'Rust thread pool with crossbeam channels',
            details: `# Thread Pool in Rust

\`\`\`rust
use std::sync::{Arc, Mutex, Condvar};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::collections::VecDeque;

type Task = Box<dyn FnOnce() + Send + 'static>;

pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: crossbeam_channel::Sender<Message>,
    active_count: Arc<AtomicUsize>,
    done_signal: Arc<(Mutex<bool>, Condvar)>,
}

enum Message {
    NewTask(Task),
    Shutdown,
}

struct Worker {
    id: usize,
    thread: Option<JoinHandle<()>>,
}

impl ThreadPool {
    pub fn new(size: usize) -> ThreadPool {
        assert!(size > 0);

        let (sender, receiver) = crossbeam_channel::unbounded::<Message>();
        let receiver = Arc::new(Mutex::new(receiver));
        let active_count = Arc::new(AtomicUsize::new(0));
        let done_signal = Arc::new((Mutex::new(true), Condvar::new()));

        let mut workers = Vec::with_capacity(size);

        for id in 0..size {
            workers.push(Worker::new(
                id,
                Arc::clone(&receiver),
                Arc::clone(&active_count),
                Arc::clone(&done_signal),
            ));
        }

        ThreadPool {
            workers,
            sender,
            active_count,
            done_signal,
        }
    }

    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let task = Box::new(f);
        self.active_count.fetch_add(1, Ordering::SeqCst);

        {
            let (lock, _) = &*self.done_signal;
            let mut done = lock.lock().unwrap();
            *done = false;
        }

        self.sender.send(Message::NewTask(task)).unwrap();
    }

    pub fn wait(&self) {
        let (lock, cvar) = &*self.done_signal;
        let mut done = lock.lock().unwrap();
        while !*done {
            done = cvar.wait(done).unwrap();
        }
    }

    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::SeqCst)
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for _ in &self.workers {
            self.sender.send(Message::Shutdown).unwrap();
        }

        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

impl Worker {
    fn new(
        id: usize,
        receiver: Arc<Mutex<crossbeam_channel::Receiver<Message>>>,
        active_count: Arc<AtomicUsize>,
        done_signal: Arc<(Mutex<bool>, Condvar)>,
    ) -> Worker {
        let thread = thread::spawn(move || loop {
            let message = {
                let rx = receiver.lock().unwrap();
                rx.recv().unwrap()
            };

            match message {
                Message::NewTask(task) => {
                    task();

                    let count = active_count.fetch_sub(1, Ordering::SeqCst) - 1;
                    if count == 0 {
                        let (lock, cvar) = &*done_signal;
                        let mut done = lock.lock().unwrap();
                        *done = true;
                        cvar.notify_all();
                    }
                }
                Message::Shutdown => {
                    break;
                }
            }
        });

        Worker {
            id,
            thread: Some(thread),
        }
    }
}

// Work-stealing deque
pub struct WorkStealingPool {
    workers: Vec<StealingWorker>,
    global_queue: Arc<Mutex<VecDeque<Task>>>,
    shutdown: Arc<AtomicBool>,
}

struct StealingWorker {
    id: usize,
    thread: Option<JoinHandle<()>>,
    local_queue: Arc<Mutex<VecDeque<Task>>>,
}

impl WorkStealingPool {
    pub fn new(size: usize) -> Self {
        let global_queue = Arc::new(Mutex::new(VecDeque::new()));
        let shutdown = Arc::new(AtomicBool::new(false));

        let mut workers = Vec::with_capacity(size);
        let mut local_queues: Vec<Arc<Mutex<VecDeque<Task>>>> = Vec::with_capacity(size);

        for _ in 0..size {
            local_queues.push(Arc::new(Mutex::new(VecDeque::new())));
        }

        for id in 0..size {
            let global = Arc::clone(&global_queue);
            let local = Arc::clone(&local_queues[id]);
            let others: Vec<_> = local_queues.iter()
                .enumerate()
                .filter(|(i, _)| *i != id)
                .map(|(_, q)| Arc::clone(q))
                .collect();
            let shutdown_flag = Arc::clone(&shutdown);

            let thread = thread::spawn(move || {
                Self::worker_loop(id, global, local.clone(), others, shutdown_flag);
            });

            workers.push(StealingWorker {
                id,
                thread: Some(thread),
                local_queue: local_queues[id].clone(),
            });
        }

        WorkStealingPool {
            workers,
            global_queue,
            shutdown,
        }
    }

    fn worker_loop(
        id: usize,
        global: Arc<Mutex<VecDeque<Task>>>,
        local: Arc<Mutex<VecDeque<Task>>>,
        others: Vec<Arc<Mutex<VecDeque<Task>>>>,
        shutdown: Arc<AtomicBool>,
    ) {
        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Try local queue first
            if let Some(task) = local.lock().unwrap().pop_front() {
                task();
                continue;
            }

            // Try global queue
            if let Some(task) = global.lock().unwrap().pop_front() {
                task();
                continue;
            }

            // Try stealing from others
            let mut stolen = false;
            for other in &others {
                if let Some(task) = other.lock().unwrap().pop_back() {
                    task();
                    stolen = true;
                    break;
                }
            }

            if !stolen {
                thread::yield_now();
            }
        }
    }

    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.global_queue.lock().unwrap().push_back(Box::new(f));
    }
}

impl Drop for WorkStealingPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

// Parallel iterators
pub fn parallel_for<I, F>(pool: &ThreadPool, iter: I, f: F)
where
    I: IntoIterator,
    I::Item: Send + 'static,
    F: Fn(I::Item) + Send + Sync + 'static,
{
    let f = Arc::new(f);
    let items: Vec<_> = iter.into_iter().collect();

    for item in items {
        let f = Arc::clone(&f);
        pool.execute(move || f(item));
    }

    pool.wait();
}

fn main() {
    // Basic thread pool
    let pool = ThreadPool::new(4);

    for i in 0..10 {
        pool.execute(move || {
            println!("Task {} on thread {:?}", i, thread::current().id());
            thread::sleep(std::time::Duration::from_millis(100));
        });
    }

    pool.wait();
    println!("All tasks completed");

    // Work-stealing pool
    let ws_pool = WorkStealingPool::new(4);

    for i in 0..20 {
        ws_pool.execute(move || {
            println!("WS Task {}", i);
        });
    }

    thread::sleep(std::time::Duration::from_secs(1));
}
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Build Your Own Async Runtime',
    description: 'Create an async/await runtime like Tokio with reactor, executor, and timers',
    icon: 'zap',
    color: 'yellow',
    language: 'Rust, C++',
    skills: 'Async programming, Polling, Wakers, Event loops',
    difficulty: 'advanced',
    estimated_weeks: 6,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Future trait | Future basics |
| 1 | 2 | Poll/Ready | Poll implementation |
| 1 | 3 | Waker | Wake mechanism |
| 1 | 4 | Context | Polling context |
| 1 | 5 | Pin | Pin and Unpin |
| 2 | 1 | Executor basics | Task spawning |
| 2 | 2 | Task queue | Run queue |
| 2 | 3 | Single-threaded | ST executor |
| 2 | 4 | Multi-threaded | MT executor |
| 2 | 5 | Work stealing | Load balancing |
| 3 | 1 | epoll/kqueue | OS events |
| 3 | 2 | Reactor | Event reactor |
| 3 | 3 | Registration | FD registration |
| 3 | 4 | Readiness | I/O readiness |
| 3 | 5 | Integration | Reactor + executor |
| 4 | 1 | Timer wheel | Timer data structure |
| 4 | 2 | Timer future | Sleep future |
| 4 | 3 | Timeout | Timeout combinator |
| 4 | 4 | Interval | Interval stream |
| 4 | 5 | Timer integration | Full timers |
| 5 | 1 | Async I/O | TcpStream |
| 5 | 2 | Read/Write | AsyncRead/Write |
| 5 | 3 | TcpListener | Async accept |
| 5 | 4 | Buffered I/O | BufReader/Writer |
| 5 | 5 | Networking | Full async net |
| 6 | 1 | Channels | mpsc channel |
| 6 | 2 | Select | Future selection |
| 6 | 3 | Join | Future joining |
| 6 | 4 | spawn_blocking | Blocking tasks |
| 6 | 5 | Full runtime | Complete runtime |`,
    modules: [
      {
        name: 'Runtime Implementation',
        description: 'Complete async runtime',
        tasks: [
          {
            title: 'Async Runtime in Rust',
            description: 'Build a minimal async runtime from scratch',
            details: `# Async Runtime in Rust

\`\`\`rust
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::collections::{HashMap, VecDeque};
use std::os::unix::io::{AsRawFd, RawFd};
use std::io;
use std::time::{Duration, Instant};
use std::cell::RefCell;
use std::thread;

// ===== Task =====

struct Task {
    future: Mutex<Pin<Box<dyn Future<Output = ()> + Send>>>,
    task_sender: std::sync::mpsc::Sender<Arc<Task>>,
}

impl Task {
    fn wake_by_ref(arc: &Arc<Task>) {
        let _ = arc.task_sender.send(Arc::clone(arc));
    }
}

fn task_waker(task: &Arc<Task>) -> Waker {
    let raw = Arc::into_raw(Arc::clone(task)) as *const ();

    unsafe fn clone(ptr: *const ()) -> RawWaker {
        let arc = Arc::from_raw(ptr as *const Task);
        let waker = task_waker(&arc);
        std::mem::forget(arc);
        waker.into_raw()
    }

    unsafe fn wake(ptr: *const ()) {
        let arc = Arc::from_raw(ptr as *const Task);
        Task::wake_by_ref(&arc);
    }

    unsafe fn wake_by_ref_fn(ptr: *const ()) {
        let arc = std::mem::ManuallyDrop::new(Arc::from_raw(ptr as *const Task));
        Task::wake_by_ref(&arc);
    }

    unsafe fn drop(ptr: *const ()) {
        drop(Arc::from_raw(ptr as *const Task));
    }

    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref_fn, drop);

    unsafe { Waker::from_raw(RawWaker::new(raw, &VTABLE)) }
}

// ===== Reactor (epoll-based) =====

pub struct Reactor {
    epoll_fd: RawFd,
    wakers: Mutex<HashMap<u64, Waker>>,
    next_token: Mutex<u64>,
}

impl Reactor {
    pub fn new() -> io::Result<Self> {
        let epoll_fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        if epoll_fd < 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(Reactor {
            epoll_fd,
            wakers: Mutex::new(HashMap::new()),
            next_token: Mutex::new(0),
        })
    }

    pub fn register(&self, fd: RawFd, interest: Interest, waker: Waker) -> io::Result<u64> {
        let mut next = self.next_token.lock().unwrap();
        let token = *next;
        *next += 1;

        let mut event = libc::epoll_event {
            events: interest.to_epoll() as u32,
            u64: token,
        };

        let res = unsafe {
            libc::epoll_ctl(self.epoll_fd, libc::EPOLL_CTL_ADD, fd, &mut event)
        };

        if res < 0 {
            return Err(io::Error::last_os_error());
        }

        self.wakers.lock().unwrap().insert(token, waker);
        Ok(token)
    }

    pub fn deregister(&self, fd: RawFd, token: u64) -> io::Result<()> {
        let res = unsafe {
            libc::epoll_ctl(self.epoll_fd, libc::EPOLL_CTL_DEL, fd, std::ptr::null_mut())
        };

        if res < 0 {
            return Err(io::Error::last_os_error());
        }

        self.wakers.lock().unwrap().remove(&token);
        Ok(())
    }

    pub fn poll(&self, timeout: Option<Duration>) -> io::Result<usize> {
        let timeout_ms = timeout
            .map(|d| d.as_millis() as i32)
            .unwrap_or(-1);

        let mut events = [libc::epoll_event { events: 0, u64: 0 }; 64];

        let n = unsafe {
            libc::epoll_wait(self.epoll_fd, events.as_mut_ptr(), 64, timeout_ms)
        };

        if n < 0 {
            return Err(io::Error::last_os_error());
        }

        let wakers = self.wakers.lock().unwrap();
        for i in 0..n as usize {
            let token = events[i].u64;
            if let Some(waker) = wakers.get(&token) {
                waker.wake_by_ref();
            }
        }

        Ok(n as usize)
    }
}

impl Drop for Reactor {
    fn drop(&mut self) {
        unsafe { libc::close(self.epoll_fd) };
    }
}

#[derive(Clone, Copy)]
pub struct Interest {
    readable: bool,
    writable: bool,
}

impl Interest {
    pub fn readable() -> Self {
        Interest { readable: true, writable: false }
    }

    pub fn writable() -> Self {
        Interest { readable: false, writable: true }
    }

    fn to_epoll(self) -> i32 {
        let mut flags = libc::EPOLLET; // Edge-triggered
        if self.readable { flags |= libc::EPOLLIN; }
        if self.writable { flags |= libc::EPOLLOUT; }
        flags
    }
}

// ===== Timer Wheel =====

struct TimerWheel {
    slots: Vec<VecDeque<(u64, Waker)>>,
    current_slot: usize,
    tick_duration: Duration,
    next_id: u64,
    start_time: Instant,
}

impl TimerWheel {
    fn new(slots: usize, tick_duration: Duration) -> Self {
        TimerWheel {
            slots: (0..slots).map(|_| VecDeque::new()).collect(),
            current_slot: 0,
            tick_duration,
            next_id: 0,
            start_time: Instant::now(),
        }
    }

    fn add_timer(&mut self, duration: Duration, waker: Waker) -> u64 {
        let ticks = (duration.as_nanos() / self.tick_duration.as_nanos()) as usize;
        let slot = (self.current_slot + ticks) % self.slots.len();

        let id = self.next_id;
        self.next_id += 1;

        self.slots[slot].push_back((id, waker));
        id
    }

    fn advance(&mut self) -> Vec<Waker> {
        let mut expired = Vec::new();

        // Calculate current slot based on elapsed time
        let elapsed = self.start_time.elapsed();
        let target_slot = (elapsed.as_nanos() / self.tick_duration.as_nanos()) as usize
            % self.slots.len();

        while self.current_slot != target_slot {
            while let Some((_, waker)) = self.slots[self.current_slot].pop_front() {
                expired.push(waker);
            }
            self.current_slot = (self.current_slot + 1) % self.slots.len();
        }

        expired
    }
}

// ===== Executor =====

thread_local! {
    static REACTOR: RefCell<Option<Arc<Reactor>>> = RefCell::new(None);
    static TIMERS: RefCell<Option<Arc<Mutex<TimerWheel>>>> = RefCell::new(None);
}

pub struct Runtime {
    executor: Executor,
    reactor: Arc<Reactor>,
    timers: Arc<Mutex<TimerWheel>>,
}

struct Executor {
    task_sender: std::sync::mpsc::Sender<Arc<Task>>,
    task_receiver: std::sync::mpsc::Receiver<Arc<Task>>,
}

impl Runtime {
    pub fn new() -> io::Result<Self> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let reactor = Arc::new(Reactor::new()?);
        let timers = Arc::new(Mutex::new(TimerWheel::new(256, Duration::from_millis(10))));

        Ok(Runtime {
            executor: Executor {
                task_sender: sender,
                task_receiver: receiver,
            },
            reactor,
            timers,
        })
    }

    pub fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let task = Arc::new(Task {
            future: Mutex::new(Box::pin(future)),
            task_sender: self.executor.task_sender.clone(),
        });

        self.executor.task_sender.send(task).unwrap();
    }

    pub fn block_on<F: Future>(&self, future: F) -> F::Output {
        // Set thread-local reactor
        REACTOR.with(|r| *r.borrow_mut() = Some(Arc::clone(&self.reactor)));
        TIMERS.with(|t| *t.borrow_mut() = Some(Arc::clone(&self.timers)));

        // Pin the future
        let mut future = std::pin::pin!(future);

        // Create waker that does nothing (we poll manually)
        static VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(std::ptr::null(), &VTABLE),
            |_| {},
            |_| {},
            |_| {},
        );
        let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
        let mut cx = Context::from_waker(&waker);

        loop {
            // Poll the main future
            if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
                return output;
            }

            // Run spawned tasks
            while let Ok(task) = self.executor.task_receiver.try_recv() {
                let waker = task_waker(&task);
                let mut cx = Context::from_waker(&waker);

                let mut future = task.future.lock().unwrap();
                let _ = future.as_mut().poll(&mut cx);
            }

            // Check timers
            let expired = self.timers.lock().unwrap().advance();
            for waker in expired {
                waker.wake();
            }

            // Poll I/O
            let _ = self.reactor.poll(Some(Duration::from_millis(10)));
        }
    }
}

// ===== Async Primitives =====

pub struct Sleep {
    deadline: Instant,
    registered: bool,
}

impl Sleep {
    pub fn new(duration: Duration) -> Self {
        Sleep {
            deadline: Instant::now() + duration,
            registered: false,
        }
    }
}

impl Future for Sleep {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if Instant::now() >= self.deadline {
            return Poll::Ready(());
        }

        if !self.registered {
            self.registered = true;

            let remaining = self.deadline.duration_since(Instant::now());
            TIMERS.with(|t| {
                if let Some(timers) = &*t.borrow() {
                    timers.lock().unwrap().add_timer(remaining, cx.waker().clone());
                }
            });
        }

        Poll::Pending
    }
}

pub fn sleep(duration: Duration) -> Sleep {
    Sleep::new(duration)
}

// ===== Usage =====

async fn example_task(id: u32) {
    println!("Task {} starting", id);
    sleep(Duration::from_millis(100 * id as u64)).await;
    println!("Task {} done", id);
}

fn main() -> io::Result<()> {
    let runtime = Runtime::new()?;

    runtime.block_on(async {
        for i in 1..=5 {
            // In a real runtime, spawn would be async
            println!("Spawning task {}", i);
        }

        // Simple sequential execution for demo
        example_task(1).await;
        example_task(2).await;
        example_task(3).await;
    });

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

console.log('Seeded: Memory Allocator, Thread Pool, Async Runtime');
