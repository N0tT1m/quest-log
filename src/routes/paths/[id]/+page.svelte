<script lang="ts">
	import { enhance } from '$app/forms';
	import { invalidateAll } from '$app/navigation';
	import {
		ChevronLeft,
		ChevronDown,
		ChevronRight,
		CheckCircle2,
		Circle,
		FileText,
		X,
		Code2,
		Lightbulb,
		Clock,
		Target,
		Zap,
		Calendar
	} from 'lucide-svelte';
	import { marked } from 'marked';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();

	let expandedModules = $state<Set<number>>(new Set());
	let currentPathId = $state<number | null>(null);
	let selectedTask = $state<{
		id: number;
		title: string;
		description: string | null;
		details: string | null;
		notes: string | null;
		completed: boolean;
		completedAt: Date | null;
		moduleName: string;
	} | null>(null);
	let editingNotes = $state('');

	// Configure marked for safe rendering
	marked.setOptions({
		breaks: true,
		gfm: true
	});

	function renderMarkdown(content: string | null): string {
		if (!content) return '';
		return marked(content) as string;
	}

	$effect(() => {
		if (data.path.id !== currentPathId) {
			currentPathId = data.path.id;
			expandedModules = new Set(data.path.modules.map((m) => m.id));
		}
	});

	function toggleModule(moduleId: number) {
		if (expandedModules.has(moduleId)) {
			expandedModules.delete(moduleId);
		} else {
			expandedModules.add(moduleId);
		}
		expandedModules = new Set(expandedModules);
	}

	function openTaskDetails(task: typeof selectedTask extends infer T | null ? NonNullable<T> : never, moduleName: string) {
		selectedTask = { ...task, moduleName };
		editingNotes = task.notes || '';
	}

	const colorClasses: Record<string, string> = {
		emerald: 'bg-emerald-500',
		blue: 'bg-blue-500',
		purple: 'bg-purple-500',
		amber: 'bg-amber-500',
		rose: 'bg-rose-500'
	};

	const bgColorClasses: Record<string, string> = {
		emerald: 'bg-emerald-500/10 border-emerald-500/30',
		blue: 'bg-blue-500/10 border-blue-500/30',
		purple: 'bg-purple-500/10 border-purple-500/30',
		amber: 'bg-amber-500/10 border-amber-500/30',
		rose: 'bg-rose-500/10 border-rose-500/30'
	};

	const textColorClasses: Record<string, string> = {
		emerald: 'text-emerald-500',
		blue: 'text-blue-500',
		purple: 'text-purple-500',
		amber: 'text-amber-500',
		rose: 'text-rose-500'
	};

	const languageColors: Record<string, string> = {
		Python: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
		Go: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
		Rust: 'bg-orange-500/20 text-orange-400 border-orange-500/30'
	};

	const difficultyColors: Record<string, string> = {
		beginner: 'bg-green-500/20 text-green-400 border-green-500/30',
		intermediate: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
		advanced: 'bg-red-500/20 text-red-400 border-red-500/30'
	};

	let skills = $derived(data.path.skills?.split(', ') || []);
</script>

<div class="p-8 max-w-5xl mx-auto">
	<!-- Header -->
	<div class="mb-8">
		<a
			href="/paths"
			class="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground mb-4"
		>
			<ChevronLeft class="w-4 h-4" />
			Back to paths
		</a>

		<div class="flex items-start justify-between gap-6">
			<div class="flex-1">
				<div class="flex items-center gap-3 mb-3">
					<h1 class="text-3xl font-bold">{data.path.name}</h1>
					{#if data.path.language}
						<span class="px-3 py-1 text-sm font-medium rounded-full border {languageColors[data.path.language] || 'bg-muted text-muted-foreground'}">
							{data.path.language}
						</span>
					{/if}
					{#if data.path.difficulty}
						<span class="px-3 py-1 text-sm font-medium rounded-full border capitalize {difficultyColors[data.path.difficulty] || 'bg-muted text-muted-foreground'}">
							{data.path.difficulty}
						</span>
					{/if}
				</div>
				<p class="text-muted-foreground text-lg">{data.path.description}</p>
			</div>

			<div class="text-right shrink-0">
				<div class="text-4xl font-bold {textColorClasses[data.path.color || 'emerald']}">{data.stats.progress}%</div>
				<p class="text-sm text-muted-foreground mt-1">
					{data.stats.completedTasks} of {data.stats.totalTasks} tasks
				</p>
			</div>
		</div>

		<!-- Progress bar -->
		<div class="mt-6 h-3 bg-muted rounded-full overflow-hidden">
			<div
				class="h-full rounded-full transition-all duration-500 {colorClasses[data.path.color || 'emerald']}"
				style="width: {data.stats.progress}%"
			></div>
		</div>
	</div>

	<!-- Project Details Cards -->
	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
		<!-- Skills Card -->
		{#if skills.length > 0}
			<div class="bg-card border border-border rounded-xl p-5">
				<div class="flex items-center gap-2 mb-3">
					<div class="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
						<Zap class="w-4 h-4 text-purple-500" />
					</div>
					<h3 class="font-semibold">Skills Practiced</h3>
				</div>
				<div class="flex flex-wrap gap-2">
					{#each skills as skill}
						<span class="px-2.5 py-1 text-xs font-medium rounded-full bg-muted text-muted-foreground">
							{skill}
						</span>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Start Hint Card -->
		{#if data.path.startHint}
			<div class="bg-card border border-border rounded-xl p-5">
				<div class="flex items-center gap-2 mb-3">
					<div class="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center">
						<Lightbulb class="w-4 h-4 text-amber-500" />
					</div>
					<h3 class="font-semibold">Start With</h3>
				</div>
				<p class="text-sm text-muted-foreground">{data.path.startHint}</p>
			</div>
		{/if}

		<!-- Time Estimate Card -->
		{#if data.path.estimatedWeeks}
			<div class="bg-card border border-border rounded-xl p-5">
				<div class="flex items-center gap-2 mb-3">
					<div class="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center">
						<Clock class="w-4 h-4 text-blue-500" />
					</div>
					<h3 class="font-semibold">Estimated Time</h3>
				</div>
				<p class="text-2xl font-bold">{data.path.estimatedWeeks} <span class="text-sm font-normal text-muted-foreground">weeks</span></p>
			</div>
		{/if}
	</div>

	<!-- Practice Rules Reminder -->
	<div class="bg-gradient-to-r from-primary/10 to-primary/5 border border-primary/20 rounded-xl p-5 mb-8">
		<div class="flex items-start gap-3">
			<div class="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center shrink-0">
				<Target class="w-5 h-5 text-primary" />
			</div>
			<div>
				<h3 class="font-semibold mb-2">Practice Rules</h3>
				<ul class="text-sm text-muted-foreground space-y-1">
					<li>Write every line yourself - no AI assistance</li>
					<li>Debug manually using print statements and debuggers</li>
					<li>Read official documentation first</li>
					<li>Start small and iterate</li>
				</ul>
			</div>
		</div>
	</div>

	<!-- Schedule Section -->
	{#if data.path.schedule}
		<div class="bg-card border border-border rounded-xl p-6 mb-8">
			<div class="flex items-center gap-3 mb-4">
				<div class="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
					<Calendar class="w-5 h-5 text-emerald-500" />
				</div>
				<h2 class="text-xl font-semibold">Schedule</h2>
			</div>
			<div class="prose prose-sm prose-invert max-w-none
				prose-headings:text-foreground prose-headings:font-semibold
				prose-h2:text-lg prose-h2:mt-6 prose-h2:mb-3
				prose-h3:text-base prose-h3:mt-4 prose-h3:mb-2
				prose-h4:text-sm prose-h4:mt-3 prose-h4:mb-1
				prose-p:text-muted-foreground prose-p:leading-relaxed
				prose-strong:text-foreground
				prose-code:text-primary prose-code:bg-muted prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm
				prose-ul:text-muted-foreground prose-ol:text-muted-foreground
				prose-li:marker:text-muted-foreground">
				{@html renderMarkdown(data.path.schedule)}
			</div>
		</div>
	{/if}

	<!-- Modules -->
	<h2 class="text-xl font-semibold mb-4">Core Features</h2>
	<div class="space-y-4">
		{#each data.path.modules as module, moduleIndex}
			{@const moduleTasks = module.tasks}
			{@const completedCount = moduleTasks.filter((t) => t.completed).length}
			{@const isExpanded = expandedModules.has(module.id)}
			{@const moduleProgress = moduleTasks.length > 0 ? Math.round((completedCount / moduleTasks.length) * 100) : 0}

			<div class="bg-card border border-border rounded-xl overflow-hidden">
				<!-- Module header -->
				<button
					onclick={() => toggleModule(module.id)}
					class="w-full flex items-center gap-4 p-5 hover:bg-muted/50 transition-colors"
				>
					<div
						class="w-10 h-10 rounded-xl {bgColorClasses[data.path.color || 'emerald']}
							flex items-center justify-center text-sm font-bold {textColorClasses[data.path.color || 'emerald']}"
					>
						{moduleIndex + 1}
					</div>

					<div class="flex-1 text-left">
						<h3 class="font-semibold text-lg">{module.name}</h3>
						{#if module.description}
							<p class="text-sm text-muted-foreground">{module.description}</p>
						{/if}
					</div>

					<div class="flex items-center gap-4">
						<div class="text-right">
							<span class="text-sm font-medium {textColorClasses[data.path.color || 'emerald']}">
								{moduleProgress}%
							</span>
							<p class="text-xs text-muted-foreground">
								{completedCount}/{moduleTasks.length} tasks
							</p>
						</div>

						<div class="w-6 h-6 flex items-center justify-center">
							{#if isExpanded}
								<ChevronDown class="w-5 h-5 text-muted-foreground" />
							{:else}
								<ChevronRight class="w-5 h-5 text-muted-foreground" />
							{/if}
						</div>
					</div>
				</button>

				<!-- Tasks -->
				{#if isExpanded}
					<div class="border-t border-border">
						{#each moduleTasks as task, taskIndex}
							<div
								class="flex items-start gap-4 p-4 hover:bg-muted/30 transition-colors
									{taskIndex !== moduleTasks.length - 1 ? 'border-b border-border' : ''}"
							>
								<form
									method="POST"
									action="?/toggle"
									use:enhance={() => {
										return async ({ update }) => {
											await update();
											await invalidateAll();
										};
									}}
								>
									<input type="hidden" name="taskId" value={task.id} />
									<button
										type="submit"
										class="mt-0.5 transition-transform hover:scale-110"
									>
										{#if task.completed}
											<CheckCircle2 class="w-6 h-6 text-primary" />
										{:else}
											<Circle class="w-6 h-6 text-muted-foreground hover:text-primary" />
										{/if}
									</button>
								</form>

								<button
								onclick={() => openTaskDetails(task, module.name)}
								class="flex-1 min-w-0 text-left hover:bg-muted/50 -my-2 -ml-2 p-2 rounded-lg transition-colors cursor-pointer"
							>
								<p
									class="font-medium {task.completed ? 'line-through text-muted-foreground' : ''}"
								>
									{task.title}
								</p>
								{#if task.description}
									<p class="text-sm text-muted-foreground mt-1 line-clamp-2">{task.description}</p>
								{/if}
								{#if task.notes}
									<div class="mt-2 p-2 bg-primary/10 rounded-lg border border-primary/20">
										<p class="text-sm text-primary/90 italic line-clamp-1">
											<span class="font-medium not-italic">Note:</span> {task.notes}
										</p>
									</div>
								{/if}
							</button>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		{/each}
	</div>
</div>

<!-- Notes Modal -->
<!-- Task Details Modal -->
{#if selectedTask}
	<div class="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
		<div class="bg-card border border-border rounded-xl w-full max-w-3xl max-h-[90vh] shadow-xl overflow-hidden flex flex-col">
			<!-- Header -->
			<div class="flex items-start justify-between p-6 border-b border-border shrink-0">
				<div class="flex-1 pr-4">
					<p class="text-xs text-muted-foreground mb-1">{selectedTask.moduleName}</p>
					<h3 class="text-lg font-semibold leading-tight">{selectedTask.title}</h3>
				</div>
				<button
					onclick={() => (selectedTask = null)}
					class="p-1 text-muted-foreground hover:text-foreground shrink-0"
				>
					<X class="w-5 h-5" />
				</button>
			</div>

			<!-- Content - Scrollable -->
			<div class="flex-1 overflow-y-auto p-6 space-y-5">
				<!-- Status -->
				<div class="flex items-center gap-3">
					<form
						method="POST"
						action="?/toggle"
						use:enhance={() => {
							return async ({ update }) => {
								await update();
								await invalidateAll();
								// Update local state to reflect the toggle
								if (selectedTask) {
									selectedTask = {
										...selectedTask,
										completed: !selectedTask.completed,
										completedAt: !selectedTask.completed ? new Date() : null
									};
								}
							};
						}}
					>
						<input type="hidden" name="taskId" value={selectedTask.id} />
						<button
							type="submit"
							class="flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-colors
								{selectedTask.completed
									? 'bg-primary/10 border-primary/30 text-primary'
									: 'bg-muted border-border text-muted-foreground hover:border-primary/50'}"
						>
							{#if selectedTask.completed}
								<CheckCircle2 class="w-4 h-4" />
								<span class="text-sm font-medium">Completed</span>
							{:else}
								<Circle class="w-4 h-4" />
								<span class="text-sm font-medium">Mark complete</span>
							{/if}
						</button>
					</form>
					{#if selectedTask.completed && selectedTask.completedAt}
						<span class="text-xs text-muted-foreground">
							{new Date(selectedTask.completedAt).toLocaleDateString()}
						</span>
					{/if}
				</div>

				<!-- Brief Description -->
				{#if selectedTask.description && !selectedTask.details}
					<div>
						<h4 class="text-sm font-medium mb-2">Description</h4>
						<p class="text-sm text-muted-foreground leading-relaxed">{selectedTask.description}</p>
					</div>
				{/if}

				<!-- Detailed Content (Markdown) -->
				{#if selectedTask.details}
					<div class="prose prose-sm prose-invert max-w-none
						prose-headings:text-foreground prose-headings:font-semibold
						prose-h2:text-lg prose-h2:mt-6 prose-h2:mb-3
						prose-h3:text-base prose-h3:mt-4 prose-h3:mb-2
						prose-h4:text-sm prose-h4:mt-3 prose-h4:mb-1
						prose-p:text-muted-foreground prose-p:leading-relaxed
						prose-strong:text-foreground
						prose-code:text-primary prose-code:bg-muted prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none
						prose-pre:bg-muted prose-pre:border prose-pre:border-border prose-pre:rounded-lg
						prose-a:text-primary prose-a:no-underline hover:prose-a:underline
						prose-ul:text-muted-foreground prose-ol:text-muted-foreground
						prose-li:marker:text-muted-foreground
						prose-table:text-sm
						prose-th:text-foreground prose-th:font-medium prose-th:bg-muted/50 prose-th:px-3 prose-th:py-2 prose-th:border prose-th:border-border
						prose-td:px-3 prose-td:py-2 prose-td:border prose-td:border-border prose-td:text-muted-foreground
						prose-hr:border-border">
						{@html renderMarkdown(selectedTask.details)}
					</div>
				{/if}

				<!-- Notes -->
				<div class="border-t border-border pt-5">
					<h4 class="text-sm font-medium mb-2">Your Notes</h4>
					<form
						method="POST"
						action="?/updateNotes"
						use:enhance={() => {
							return async ({ update }) => {
								await update();
								await invalidateAll();
							};
						}}
					>
						<input type="hidden" name="taskId" value={selectedTask.id} />
						<textarea
							name="notes"
							bind:value={editingNotes}
							placeholder="Add your notes, learnings, or links here..."
							class="w-full h-28 bg-background border border-input rounded-lg px-3 py-2 text-sm
								focus:outline-none focus:ring-2 focus:ring-ring resize-none"
						></textarea>
						{#if editingNotes !== (selectedTask.notes || '')}
							<div class="flex justify-end mt-2">
								<button
									type="submit"
									class="px-3 py-1.5 bg-primary text-primary-foreground rounded-lg text-sm font-medium
										hover:bg-primary/90 transition-colors"
								>
									Save Notes
								</button>
							</div>
						{/if}
					</form>
				</div>
			</div>

			<!-- Footer -->
			<div class="flex justify-end gap-2 px-6 py-4 border-t border-border bg-muted/30">
				<button
					onclick={() => (selectedTask = null)}
					class="px-4 py-2 text-sm font-medium hover:bg-muted rounded-lg transition-colors"
				>
					Close
				</button>
			</div>
		</div>
	</div>
{/if}
