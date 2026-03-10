# SonarQube Java Bug Detection - Preconfigured Setup

> **Zero manual setup required!** Use the Docker-based SonarQube instance that's already running via `docker-compose up -d`.

## Quick Start

```bash
# 1. Ensure SonarQube is running
docker compose ps | grep sonarqube

# 2. Access SonarQube Dashboard
open http://localhost:9000
# OR: http://localhost:9000/projects

# 3. Initial Login (First Time Only)
# - Username: admin
# - Password: admin
# - You'll be prompted to change password on first login
```

## Running Analysis

### Option A: Using Docker CLI (Recommended)

```bash
# Analyze Java code with sonar-scanner
docker run --rm \
  -v $(pwd)/your-java-path:/src:ro \
  --network=code-smell-network \
  sonarsource/sonar-scanner-cli:5.0.1.3822 \
  -Dsonar.projectKey=my-project \
  -Dsonar.sources=/src \
  -Dsonar.host.url=http://sonarqube:9000 \
  -Dsonar.login=admin \
  -Dsonar.password=YOUR_NEW_PASSWORD
```

### Option B: Using Provided Script

```bash
# For a quick test analysis
bash scripts/sonarqube_quick_scan.sh \
  tools/baseline/test \
  code-smell-java-test \
  results/predictions/baseline/sonarqube_findings.json
```

### Option C: Using Python API Wrapper

```bash
python scripts/sonarqube_analyze.py \
  --source-dir tools/baseline/test \
  --project-key code-smell-java \
  --output results/predictions/baseline/sonarqube.json
```

## Initial Setup (First Time After Docker Start)

When SonarQube first starts, you need to change the default password:

1. Open http://localhost:9000 in browser
2. Login with `admin/admin`
3. Go to **Administration → Security → Users** 
4. Change the admin password
5. Update scripts with new password

**OR** - Use environment variable approach in `docker-compose.yml`:

```yaml
environment:
  - SONARQUBE_ADMIN_PASSWORD=your_secure_password
```

Then rebuild:
```bash
docker compose up -d --force-recreate sonarqube
```

## Authentication

### With curl/API calls:

```bash
# After changing password, use:
curl -u "admin:YOUR_NEW_PASSWORD" \
  "http://localhost:9000/api/issues/search" \
  -d "projectKeys=my-project"
```

### Generate Token (Optional):

```bash
# Create a token for API access
curl -u "admin:PASSWORD" \
  -X POST "http://localhost:9000/api/user_tokens/generate" \
  -d "name=my-analyzer-token"

# Use token in requests:
curl -u "TOKEN:" \
  "http://localhost:9000/api/issues/search" \
  -d "projectKeys=my-project"
```

## Common Tasks

### View Findings in Dashboard

```
http://localhost:9000/dashboard?id=code-smell-java-test
```

### Export Results as JSON

```bash
# Get all issues for a project
curl -u "admin:PASSWORD" \
  "http://localhost:9000/api/issues/search?projectKeys=code-smell-java-test" \
  > results.json

# Pretty print
cat results.json | python3 -m json.tool
```

### Reset SonarQube (Clear All Projects)

```bash
# Stop and remove volumes
docker compose down -v

# Restart fresh
docker compose up -d

# Wait for SonarQube to initialize (30 seconds)
sleep 30

# Verify it's running
curl http://localhost:9000/api/system/status
```

## Environment Variables

Set these before running docker-compose:

```bash
export SONAR_ADMIN_PASSWORD="SecurePassword123"
export SONAR_HOST_URL="http://sonarqube:9000"
export SONAR_DB_USER="sonarqube"
export SONAR_DB_PASSWORD="sonarqube"
```

## Container Details

- **Service Name:** sonarqube
- **Container Name:** code-smell-sonarqube
- **Port:** 9000 (accessible at http://localhost:9000)
- **Database:** PostgreSQL 15 (code-smell-sonarqube-db)
- **Data Persistence:** Volumes (sonarqube_data, sonarqube_db, sonarqube_logs, sonarqube_extensions)

## Troubleshooting

### "Connection refused" error

```bash
# Check if container is running
docker compose ps sonarqube

# Check logs
docker compose logs -f sonarqube

# Wait for startup (up to 60 seconds)
sleep 60
```

### "Invalid credentials" error

Change the admin password in UI, then update scripts:

```bash
# Set environment variable for scripts
export SONAR_PASSWORD="your_new_password"
```

### Issues not appearing after analysis

Wait 10-30 seconds for indexing:

```bash
sleep 30
curl -u "admin:PASSWORD" \
  "http://localhost:9000/api/issues/search?projectKeys=your-project"
```

###Port 9000 already in use

```bash
# Find and kill process using port 9000
lsof -i :9000
kill -9 <PID>

# Or use different port in docker-compose.yml
# Change:    "9000:9000"
# To:        "9001:9000"
```

## Integration with Code Smell Pipeline

Add to `scripts/run_baseline_tools_docker.sh`:

```bash
# Run SonarQube analysis
echo "Running SonarQube analysis..."
docker run --rm \
  -v "$(pwd)/$SOURCE_DIR:/src:ro" \
  --network=code-smell-network \
  sonarsource/sonar-scanner-cli:5.0.1.3822 \
  -Dsonar.projectKey="code-smell-$LANGUAGE-$(date +%s)" \
  -Dsonar.sources=/src \
  -Dsonar.host.url=http://sonarqube:9000 \
  -Dsonar.login=admin \
  -Dsonar.password="$SONAR_PASSWORD"
```

## Performance Notes

- First analysis: 30-120 seconds (depends on codebase size)
- Subsequent analyses: 10-60 seconds (cached)
- Memory requirement: 1-2 GB minimum
- Concurrency: Can run multiple sonar-scanner instances in parallel

## References

- [SonarQube API Documentation](https://docs.sonarsource.com/sonarqube/10.4/extension-guide/web-api/)
- [SonarScanner Documentation](https://docs.sonarsource.com/sonarqube/10.4/analyzing-source-code/scanners/sonarscanner/)
- [SonarQube Docker Hub](https://hub.docker.com/_/sonarqube)
