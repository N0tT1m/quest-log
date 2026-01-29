# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies for better-sqlite3
RUN apk add --no-cache python3 make g++

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
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

# Copy built app
COPY --from=builder /app/build ./build
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
COPY --from=builder /app/scripts ./scripts

# Create data directory
RUN mkdir -p /app/data

# Set environment
ENV NODE_ENV=production
ENV DATABASE_URL=file:/app/data/quest-log.db
ENV PORT=3000
ENV HOST=0.0.0.0

EXPOSE 3000

# Run the app
CMD ["node", "build"]
