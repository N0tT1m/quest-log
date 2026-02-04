<script lang="ts">
	import '../app.css';
	import { page } from '$app/stores';
	import { BookOpen, LayoutDashboard, Flame } from 'lucide-svelte';

	let { children } = $props();

	const navItems = [
		{ href: '/', label: 'Dashboard', icon: LayoutDashboard },
		{ href: '/paths', label: 'Paths', icon: BookOpen }
	];
</script>

<div class="min-h-screen flex">
	<!-- Sidebar -->
	<aside class="w-64 border-r border-border bg-card/50 flex flex-col">
		<div class="p-6 border-b border-border">
			<div class="flex items-center gap-3">
				<div class="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
					<Flame class="w-6 h-6 text-primary" />
				</div>
				<div>
					<h1 class="font-bold text-lg">Quest Log</h1>
					<p class="text-xs text-muted-foreground">Track your journey</p>
				</div>
			</div>
		</div>

		<nav class="flex-1 p-4">
			<ul class="space-y-2">
				{#each navItems as item}
					{@const isActive = $page.url.pathname === item.href ||
						(item.href !== '/' && $page.url.pathname.startsWith(item.href))}
					<li>
						<a
							href={item.href}
							class="flex items-center gap-3 px-4 py-2.5 rounded-lg transition-colors
								{isActive
									? 'bg-primary/20 text-primary'
									: 'text-muted-foreground hover:text-foreground hover:bg-muted'}"
						>
							<item.icon class="w-5 h-5" />
							<span class="font-medium">{item.label}</span>
						</a>
					</li>
				{/each}
			</ul>
		</nav>

		<div class="p-4 border-t border-border">
			<p class="text-xs text-muted-foreground text-center">
				Keep learning, keep growing
			</p>
		</div>
	</aside>

	<!-- Main content -->
	<main class="flex-1 overflow-auto">
		{@render children()}
	</main>
</div>
