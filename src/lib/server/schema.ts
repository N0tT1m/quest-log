import { sqliteTable, text, integer } from 'drizzle-orm/sqlite-core';

export const paths = sqliteTable('paths', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	name: text('name').notNull(),
	description: text('description'),
	icon: text('icon').default('book'),
	color: text('color').default('emerald'),
	language: text('language'),
	skills: text('skills'),
	startHint: text('start_hint'),
	difficulty: text('difficulty').default('intermediate'),
	estimatedWeeks: integer('estimated_weeks'),
	schedule: text('schedule'),
	createdAt: integer('created_at', { mode: 'timestamp' }).$defaultFn(() => new Date())
});

export const modules = sqliteTable('modules', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	pathId: integer('path_id')
		.notNull()
		.references(() => paths.id, { onDelete: 'cascade' }),
	name: text('name').notNull(),
	description: text('description'),
	orderIndex: integer('order_index').notNull().default(0),
	targetDate: text('target_date'),
	createdAt: integer('created_at', { mode: 'timestamp' }).$defaultFn(() => new Date())
});

export const tasks = sqliteTable('tasks', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	moduleId: integer('module_id')
		.notNull()
		.references(() => modules.id, { onDelete: 'cascade' }),
	title: text('title').notNull(),
	description: text('description'),
	details: text('details'),
	orderIndex: integer('order_index').notNull().default(0),
	completed: integer('completed', { mode: 'boolean' }).notNull().default(false),
	completedAt: integer('completed_at', { mode: 'timestamp' }),
	notes: text('notes'),
	createdAt: integer('created_at', { mode: 'timestamp' }).$defaultFn(() => new Date())
});

export const activity = sqliteTable('activity', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	date: text('date').notNull().unique(),
	tasksCompleted: integer('tasks_completed').notNull().default(0)
});

// Type exports
export type Path = typeof paths.$inferSelect;
export type NewPath = typeof paths.$inferInsert;
export type Module = typeof modules.$inferSelect;
export type NewModule = typeof modules.$inferInsert;
export type Task = typeof tasks.$inferSelect;
export type NewTask = typeof tasks.$inferInsert;
export type Activity = typeof activity.$inferSelect;
