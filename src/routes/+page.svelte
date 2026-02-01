<script lang="ts">
	import { Flame, Target, CheckCircle2, TrendingUp, ChevronRight, Calendar } from 'lucide-svelte';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();

	const colorClasses: Record<string, string> = {
		emerald: 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/30',
		blue: 'from-blue-500/20 to-blue-500/5 border-blue-500/30',
		purple: 'from-purple-500/20 to-purple-500/5 border-purple-500/30',
		amber: 'from-amber-500/20 to-amber-500/5 border-amber-500/30',
		rose: 'from-rose-500/20 to-rose-500/5 border-rose-500/30',
		red: 'from-red-500/20 to-red-500/5 border-red-500/30',
		cyan: 'from-cyan-500/20 to-cyan-500/5 border-cyan-500/30',
		green: 'from-green-500/20 to-green-500/5 border-green-500/30',
		indigo: 'from-indigo-500/20 to-indigo-500/5 border-indigo-500/30',
		orange: 'from-orange-500/20 to-orange-500/5 border-orange-500/30',
		pink: 'from-pink-500/20 to-pink-500/5 border-pink-500/30',
		teal: 'from-teal-500/20 to-teal-500/5 border-teal-500/30',
		yellow: 'from-yellow-500/20 to-yellow-500/5 border-yellow-500/30'
	};

	const languageColors: Record<string, string> = {
		Python: 'bg-blue-500/20 text-blue-400',
		Go: 'bg-cyan-500/20 text-cyan-400',
		Rust: 'bg-orange-500/20 text-orange-400',
		'C': 'bg-gray-500/20 text-gray-400',
		'C++': 'bg-pink-500/20 text-pink-400',
		'C#': 'bg-purple-500/20 text-purple-400',
		'C+Python+C#+Rust': 'bg-red-500/20 text-red-400',
		'Go+Rust+Python': 'bg-rose-500/20 text-rose-400',
		'Multi-Language': 'bg-rose-500/20 text-rose-400'
	};
</script>

<div class="p-8">
	<!-- Header -->
	<div class="mb-8">
		<h1 class="text-3xl font-bold mb-2">Dashboard</h1>
		<p class="text-muted-foreground">Track your learning progress</p>
	</div>

	<!-- Stats Grid -->
	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
		<div class="bg-card border border-border rounded-xl p-6">
			<div class="flex items-center justify-between mb-4">
				<div class="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center">
					<Flame class="w-6 h-6 text-primary" />
				</div>
				<span class="text-3xl font-bold">{data.stats.streak}</span>
			</div>
			<p class="text-sm text-muted-foreground">Day Streak</p>
		</div>

		<div class="bg-card border border-border rounded-xl p-6">
			<div class="flex items-center justify-between mb-4">
				<div class="w-12 h-12 rounded-lg bg-blue-500/20 flex items-center justify-center">
					<CheckCircle2 class="w-6 h-6 text-blue-500" />
				</div>
				<span class="text-3xl font-bold">{data.stats.completedTasks}</span>
			</div>
			<p class="text-sm text-muted-foreground">Tasks Completed</p>
		</div>

		<div class="bg-card border border-border rounded-xl p-6">
			<div class="flex items-center justify-between mb-4">
				<div class="w-12 h-12 rounded-lg bg-purple-500/20 flex items-center justify-center">
					<Target class="w-6 h-6 text-purple-500" />
				</div>
				<span class="text-3xl font-bold">{data.stats.totalTasks}</span>
			</div>
			<p class="text-sm text-muted-foreground">Total Tasks</p>
		</div>

		<div class="bg-card border border-border rounded-xl p-6">
			<div class="flex items-center justify-between mb-4">
				<div class="w-12 h-12 rounded-lg bg-amber-500/20 flex items-center justify-center">
					<TrendingUp class="w-6 h-6 text-amber-500" />
				</div>
				<span class="text-3xl font-bold">{data.stats.completionRate}%</span>
			</div>
			<p class="text-sm text-muted-foreground">Completion Rate</p>
		</div>
	</div>

	<div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
		<!-- Active Paths -->
		<div class="lg:col-span-2">
			<div class="flex items-center justify-between mb-4">
				<h2 class="text-xl font-semibold">Active Learning Paths</h2>
				<a href="/paths" class="text-sm text-primary hover:underline flex items-center gap-1">
					View all <ChevronRight class="w-4 h-4" />
				</a>
			</div>

			{#if data.paths.length === 0}
				<div class="bg-card border border-border rounded-xl p-8 text-center">
					<p class="text-muted-foreground mb-4">No learning paths yet</p>
					<a href="/paths" class="text-primary hover:underline">Create your first path</a>
				</div>
			{:else}
				<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
					{#each data.paths.slice(0, 4) as path}
						<a
							href="/paths/{path.id}"
							class="bg-gradient-to-br {colorClasses[path.color || 'emerald']}
								border rounded-xl p-6 hover:scale-[1.02] transition-transform"
						>
							<div class="flex items-start justify-between mb-4">
								<div>
									{#if path.language}
										<span class="inline-block px-2 py-0.5 text-xs font-medium rounded-full mb-2 {languageColors[path.language] || 'bg-muted text-muted-foreground'}">
											{path.language}
										</span>
									{/if}
									<h3 class="font-semibold text-lg mb-1">{path.name}</h3>
									<p class="text-sm text-muted-foreground line-clamp-2">{path.description}</p>
								</div>
							</div>

							<div class="mt-4">
								<div class="flex items-center justify-between text-sm mb-2">
									<span class="text-muted-foreground">Progress</span>
									<span class="font-medium">{path.progress}%</span>
								</div>
								<div class="h-2 bg-background/50 rounded-full overflow-hidden">
									<div
										class="h-full bg-primary rounded-full transition-all duration-500"
										style="width: {path.progress}%"
									></div>
								</div>
								<p class="text-xs text-muted-foreground mt-2">
									{path.completedTasks} of {path.totalTasks} tasks
								</p>
							</div>
						</a>
					{/each}
				</div>
			{/if}
		</div>

		<!-- Upcoming Tasks -->
		<div>
			<h2 class="text-xl font-semibold mb-4">Up Next</h2>

			{#if data.upcomingTasks.length === 0}
				<div class="bg-card border border-border rounded-xl p-6 text-center">
					<CheckCircle2 class="w-12 h-12 text-primary mx-auto mb-3 opacity-50" />
					<p class="text-muted-foreground">All caught up!</p>
				</div>
			{:else}
				<div class="space-y-3">
					{#each data.upcomingTasks as task}
						<a
							href="/paths/{task.pathId}"
							class="block bg-card border border-border rounded-xl p-4 hover:border-primary/50 transition-colors"
						>
							<p class="font-medium mb-1 line-clamp-1">{task.title}</p>
							<p class="text-xs text-muted-foreground">
								{task.pathName} &bull; {task.moduleName}
							</p>
						</a>
					{/each}
				</div>
			{/if}

			<!-- Activity Calendar Preview -->
			<div class="mt-6">
				<h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
					<Calendar class="w-5 h-5" />
					Recent Activity
				</h3>
				<div class="bg-card border border-border rounded-xl p-4">
					<div class="flex gap-1 justify-center">
						{#each Array(7) as _, i}
							{@const date = new Date(Date.now() - (6 - i) * 86400000).toISOString().split('T')[0]}
							{@const activityDay = data.recentActivity.find(a => a.date === date)}
							{@const count = activityDay?.tasksCompleted || 0}
							<div class="text-center">
								<div
									class="w-8 h-8 rounded-md flex items-center justify-center text-xs font-medium
										{count > 0 ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}"
								>
									{count}
								</div>
								<p class="text-xs text-muted-foreground mt-1">
									{new Date(date).toLocaleDateString('en', { weekday: 'short' }).slice(0, 2)}
								</p>
							</div>
						{/each}
					</div>
				</div>
			</div>
		</div>
	</div>
</div>
