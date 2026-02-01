<script lang="ts">
	import { BookOpen, Layers, CheckCircle2, Clock, Zap } from 'lucide-svelte';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();

	const colorClasses: Record<string, string> = {
		emerald: 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/30 hover:border-emerald-500/50',
		blue: 'from-blue-500/20 to-blue-500/5 border-blue-500/30 hover:border-blue-500/50',
		purple: 'from-purple-500/20 to-purple-500/5 border-purple-500/30 hover:border-purple-500/50',
		amber: 'from-amber-500/20 to-amber-500/5 border-amber-500/30 hover:border-amber-500/50',
		rose: 'from-rose-500/20 to-rose-500/5 border-rose-500/30 hover:border-rose-500/50',
		red: 'from-red-500/20 to-red-500/5 border-red-500/30 hover:border-red-500/50',
		cyan: 'from-cyan-500/20 to-cyan-500/5 border-cyan-500/30 hover:border-cyan-500/50',
		green: 'from-green-500/20 to-green-500/5 border-green-500/30 hover:border-green-500/50',
		indigo: 'from-indigo-500/20 to-indigo-500/5 border-indigo-500/30 hover:border-indigo-500/50',
		orange: 'from-orange-500/20 to-orange-500/5 border-orange-500/30 hover:border-orange-500/50',
		pink: 'from-pink-500/20 to-pink-500/5 border-pink-500/30 hover:border-pink-500/50',
		teal: 'from-teal-500/20 to-teal-500/5 border-teal-500/30 hover:border-teal-500/50',
		yellow: 'from-yellow-500/20 to-yellow-500/5 border-yellow-500/30 hover:border-yellow-500/50',
		gray: 'from-gray-500/20 to-gray-500/5 border-gray-500/30 hover:border-gray-500/50'
	};

	const iconColors: Record<string, string> = {
		emerald: 'text-emerald-500',
		blue: 'text-blue-500',
		purple: 'text-purple-500',
		amber: 'text-amber-500',
		rose: 'text-rose-500',
		red: 'text-red-500',
		cyan: 'text-cyan-500',
		green: 'text-green-500',
		indigo: 'text-indigo-500',
		orange: 'text-orange-500',
		pink: 'text-pink-500',
		teal: 'text-teal-500',
		yellow: 'text-yellow-500',
		gray: 'text-gray-500'
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

	const difficultyColors: Record<string, string> = {
		beginner: 'text-green-400',
		intermediate: 'text-yellow-400',
		advanced: 'text-red-400'
	};

	const progressColors: Record<string, string> = {
		emerald: 'bg-emerald-500',
		blue: 'bg-blue-500',
		purple: 'bg-purple-500',
		amber: 'bg-amber-500',
		rose: 'bg-rose-500',
		red: 'bg-red-500',
		cyan: 'bg-cyan-500',
		green: 'bg-green-500',
		indigo: 'bg-indigo-500',
		orange: 'bg-orange-500',
		pink: 'bg-pink-500',
		teal: 'bg-teal-500',
		yellow: 'bg-yellow-500',
		gray: 'bg-gray-500'
	};
</script>

<div class="p-8">
	<div class="mb-8">
		<h1 class="text-3xl font-bold mb-2">Coding Practice Projects</h1>
		<p class="text-muted-foreground">Build your skills with these hands-on projects - no AI assistance allowed</p>
	</div>

	{#if data.paths.length === 0}
		<div class="bg-card border border-border rounded-xl p-12 text-center">
			<BookOpen class="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-50" />
			<h2 class="text-xl font-semibold mb-2">No projects yet</h2>
			<p class="text-muted-foreground mb-6">
				Run the seed script to populate with coding practice projects
			</p>
			<code class="bg-muted px-4 py-2 rounded-lg text-sm">npx tsx scripts/seed-projects.ts</code>
		</div>
	{:else}
		<!-- Language filter summary -->
		<div class="flex flex-wrap gap-3 mb-6">
			{#each [...new Set(data.paths.map(p => p.language).filter(Boolean))] as lang}
				{@const count = data.paths.filter(p => p.language === lang).length}
				<div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-card border border-border">
					<span class="w-2 h-2 rounded-full {lang === 'Python' ? 'bg-blue-500' : lang === 'Go' ? 'bg-cyan-500' : 'bg-orange-500'}"></span>
					<span class="text-sm font-medium">{lang}</span>
					<span class="text-xs text-muted-foreground">{count} projects</span>
				</div>
			{/each}
		</div>

		<div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
			{#each data.paths as path}
				<a
					href="/paths/{path.id}"
					class="group bg-gradient-to-br {colorClasses[path.color || 'emerald']}
						border rounded-xl p-6 transition-all hover:scale-[1.02] hover:shadow-lg"
				>
					<div class="flex items-start gap-4 mb-4">
						<div class="w-12 h-12 rounded-xl bg-background/50 flex items-center justify-center">
							<BookOpen class="w-6 h-6 {iconColors[path.color || 'emerald']}" />
						</div>
						<div class="flex-1 min-w-0">
							<div class="flex items-center gap-2 mb-1">
								{#if path.language}
									<span class="px-2 py-0.5 text-xs font-medium rounded-full {languageColors[path.language] || 'bg-muted text-muted-foreground'}">
										{path.language}
									</span>
								{/if}
								{#if path.difficulty}
									<span class="text-xs capitalize {difficultyColors[path.difficulty] || 'text-muted-foreground'}">
										{path.difficulty}
									</span>
								{/if}
							</div>
							<h3 class="font-semibold text-lg group-hover:text-primary transition-colors line-clamp-1">
								{path.name}
							</h3>
						</div>
					</div>

					<p class="text-sm text-muted-foreground line-clamp-2 mb-4">
						{path.description}
					</p>

					<!-- Skills preview -->
					{#if path.skills}
						<div class="flex items-center gap-2 mb-4">
							<Zap class="w-3.5 h-3.5 text-muted-foreground shrink-0" />
							<p class="text-xs text-muted-foreground line-clamp-1">{path.skills}</p>
						</div>
					{/if}

					<div class="flex items-center gap-4 text-sm text-muted-foreground mb-4">
						<span class="flex items-center gap-1.5">
							<Layers class="w-4 h-4" />
							{path.moduleCount} modules
						</span>
						<span class="flex items-center gap-1.5">
							<CheckCircle2 class="w-4 h-4" />
							{path.completedTasks}/{path.totalTasks} tasks
						</span>
						{#if path.estimatedWeeks}
							<span class="flex items-center gap-1.5">
								<Clock class="w-4 h-4" />
								{path.estimatedWeeks}w
							</span>
						{/if}
					</div>

					<div>
						<div class="flex items-center justify-between text-sm mb-2">
							<span class="text-muted-foreground">Progress</span>
							<span class="font-semibold {iconColors[path.color || 'emerald']}">{path.progress}%</span>
						</div>
						<div class="h-2.5 bg-background/50 rounded-full overflow-hidden">
							<div
								class="h-full rounded-full transition-all duration-500 {progressColors[path.color || 'emerald'] || 'bg-emerald-500'}"
								style="width: {path.progress}%"
							></div>
						</div>
					</div>
				</a>
			{/each}
		</div>
	{/if}
</div>
