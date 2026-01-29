<script lang="ts">
	import { BookOpen, Layers, CheckCircle2, Clock, Zap } from 'lucide-svelte';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();

	const colorClasses: Record<string, string> = {
		emerald: 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/30 hover:border-emerald-500/50',
		blue: 'from-blue-500/20 to-blue-500/5 border-blue-500/30 hover:border-blue-500/50',
		purple: 'from-purple-500/20 to-purple-500/5 border-purple-500/30 hover:border-purple-500/50',
		amber: 'from-amber-500/20 to-amber-500/5 border-amber-500/30 hover:border-amber-500/50',
		rose: 'from-rose-500/20 to-rose-500/5 border-rose-500/30 hover:border-rose-500/50'
	};

	const iconColors: Record<string, string> = {
		emerald: 'text-emerald-500',
		blue: 'text-blue-500',
		purple: 'text-purple-500',
		amber: 'text-amber-500',
		rose: 'text-rose-500'
	};

	const languageColors: Record<string, string> = {
		Python: 'bg-blue-500/20 text-blue-400',
		Go: 'bg-cyan-500/20 text-cyan-400',
		Rust: 'bg-orange-500/20 text-orange-400'
	};

	const difficultyColors: Record<string, string> = {
		beginner: 'text-green-400',
		intermediate: 'text-yellow-400',
		advanced: 'text-red-400'
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
								class="h-full rounded-full transition-all duration-500
									{path.color === 'blue' ? 'bg-blue-500' :
									 path.color === 'purple' ? 'bg-purple-500' :
									 path.color === 'amber' ? 'bg-amber-500' :
									 path.color === 'rose' ? 'bg-rose-500' : 'bg-emerald-500'}"
								style="width: {path.progress}%"
							></div>
						</div>
					</div>
				</a>
			{/each}
		</div>
	{/if}
</div>
