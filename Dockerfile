# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies for better-sqlite3
RUN apk add --no-cache python3 make g++

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies (including devDependencies for tsx)
RUN npm ci

# Copy source
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM node:20-alpine AS runner

WORKDIR /app

# Install runtime dependencies for better-sqlite3
RUN apk add --no-cache python3 make g++

# Copy everything needed to run seeds and the app
COPY --from=builder /app/build ./build
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
COPY --from=builder /app/scripts ./scripts
COPY --from=builder /app/src ./src
COPY --from=builder /app/tsconfig.json ./

# Create data directory
RUN mkdir -p /app/data

# Set environment
ENV NODE_ENV=production
ENV DATABASE_URL=file:/app/data/quest-log.db
ENV PORT=3000
ENV HOST=0.0.0.0

EXPOSE 3000

# Run seeds and start app
CMD sh -c "npx tsx scripts/seed-all.ts && node build"
