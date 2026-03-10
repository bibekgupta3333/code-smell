# Using Preconfigured SonarQube for Java Bug Detection

> **Requirements:** Docker, Bash, Python 3
> **Setup Time:** ~2 minutes (one-time)
> **Analysis Time:** 10-60 seconds per codebase

## What is this?

This provides **zero-setup SonarQube integration** for detecting bugs in Java code. The SonarQube service is already running via `docker-compose` - just run the analysis scripts.

## Quick Start (30 seconds)

```bash
# 1. One-time setup (generate auth token)
bash scripts/sonarqube_setup.sh

# 2. Run analysis
bash scripts/sonarqube_compile_and_analyze.sh \
  data/datasets/SmellyCodeDataset/Java \
  code-smell-java \
  results/predictions/baseline/sonarqube_java.json

# 3. View results
cat results/predictions/baseline/sonarqube_java.json | python3 -m json.tool
```

## Available Scripts

### 1. `sonarqube_setup.sh` (Run Once)
**Purpose:** Initialize SonarQube and generate authentication token

```bash
bash scripts/sonarqube_setup.sh
```

**What it does:**
- Verifies SonarQube is running
- Changes admin password
- Generates API token (saves to `.sonar-token`)
- Creates default project

**Output:**
```
✓ SonarQube is running
✓ Token generated: squ_cb25041e262441af...
✓ Saved to: .sonar-token
```

### 2. `sonarqube_compile_and_analyze.sh` (Main Analysis)
**Purpose:** Compile Java and analyze with SonarQube

```bash
bash scripts/sonarqube_compile_and_analyze.sh <source-dir> <project-key> [output-file]
```

**Examples:**
```bash
# Analyze dataset
bash scripts/sonarqube_compile_and_analyze.sh \
  data/datasets/SmellyCodeDataset/Java \
  code-smell-java

# Custom output file
bash scripts/sonarqube_compile_and_analyze.sh \
  tools/baseline/test \
  code-smell-sample \
  results/findings.json
```

**What it does:**
1. Compiles Java source files (.java → .class)
2. Runs SonarScanner in Docker container
3. Fetches results from SonarQube API
4. Saves findings to JSON file

**Output Example:**
```json
{
  "total": 286,
  "issues": [
    {
      "key": "AXy7ZvEq5l_VpBYXy000",
      "type": "CODE_SMELL",
      "severity": "MAJOR",
      "message": "Remove this unused private field 'unusedField1'",
      "component": "code-smell-sample:SampleSmelly.java",
      "line": 42,
      "rule": "java:S1068"
    },
    ...
  ]
}
```

### 3. `sonarqube_run.sh` (Simple Analysis - Source Code Only)
**Purpose:** Analyze without compilation (for projects with pre-compiled classes)

```bash
bash scripts/sonarqube_run.sh <source-dir> <project-key> [output-file]
```

## Results Format

SonarQube findings in JSON format:

```json
{
  "total": 5,
  "issues": [
    {
      "key": "issue-id",
      "type": "BUG|VULNERABILITY|CODE_SMELL",
      "severity": "BLOCKER|CRITICAL|MAJOR|MINOR|INFO",
      "message": "Description of the issue",
      "component": "project:File.java",
      "line": 42,
      "column": 8,
      "rule": "java:S1234",
      "resolution": "FIXED|WONTFIX|FALSE_POSITIVE|REMOVED",
      "status": "OPEN|CONFIRMED|REOPENED|RESOLVED|CLOSED"
    }
  ]
}
```

## Severity Levels

| Severity | Code Quality Impact | Priority |
|----------|-------------------|----------|
| BLOCKER  | Code must be fixed immediately | Critical bug |
| CRITICAL | Major issue affecting reliability | Security vulnerability |
| MAJOR    | Significant quality degradation | Architectural issue |
| MINOR    | Code quality could be improved | Minor cleanup |
| INFO     | Informational only | Nice to have |

## Common Tasks

### View Find in Dashboard

```bash
# Open browser to SonarQube dashboard
open "http://localhost:9000/dashboard?id=code-smell-java"

# Or print results
cat results/predictions/baseline/sonarqube_java.json | python3 -m json.tool
```

### Filter Issues by Severity

```bash
# Show only MAJOR and above
cat results/predictions/baseline/sonarqube_java.json | \
  python3 -c "
import json, sys
data = json.load(sys.stdin)
critical = [i for i in data['issues'] if i['severity'] in ['BLOCKER', 'CRITICAL', 'MAJOR']]
print(json.dumps(critical, indent=2))
" | less
```

### Count Issues by Type

```bash
cat results/predictions/baseline/sonarqube_java.json | \
  python3 -c "
import json, sys
from collections import Counter
data = json.load(sys.stdin)
types = Counter(i['type'] for i in data['issues'])
print('Issue Types:')
for type_name, count in types.most_common():
    print(f'  {type_name:20} {count:5}')
"
```

### Export to CSV

```bash
cat results/predictions/baseline/sonarqube_java.json | \
  python3 -c "
import json, sys, csv
data = json.load(sys.stdin)
writer = csv.DictWriter(sys.stdout, fieldnames=['file', 'line', 'rule', 'severity', 'message'])
writer.writeheader()
for issue in data['issues']:
    writer.writerow({
        'file': issue.get('component', '').split(':')[-1],
        'line': issue.get('line', ''),
        'rule': issue.get('rule', ''),
        'severity': issue.get('severity', ''),
        'message': issue.get('message', '')
    })
" > sonarqube_findings.csv
```

## Troubleshooting

### "Token file not found"

```bash
# Run setup to generate token
bash scripts/sonarqube_setup.sh
```

### "Cannot reach SonarQube"

```bash
# Check if containers are running
docker compose ps | grep sonarqube

# Start if not running
docker compose up -d sonarqube sonarqube-db

# Wait for initialization (30 seconds)
sleep 30

# Test connection
curl http://localhost:9000/api/system/status
```

### "No Java files found"

```bash
# Verify source directory has .java files
find <source-dir> -name "*.java" | head -5

# Ensure path is correct
bash scripts/sonarqube_compile_and_analyze.sh ./path/to/java ./project-key
```

### "Analysis returned 0 issues"

This is normal if:
- The code follows SonarQube quality standards
- The code is generated (generated code is often ignored)
- The quality profile is minimal

To verify analysis ran:
```bash
# Check SonarQube logs
docker compose logs sonarqube | grep -i "analysis\|issue\|error"

# View dashboard - might show metrics even with 0 issues
open http://localhost:9000/dashboard?id=<project-key>
```

## Advanced Usage

### Custom Quality Profile

Edit SonarQube quality profile in dashboard:
1. Open http://localhost:9000
2. Quality Profiles → Java
3. Activate/deactivate rules
4. Re-run analysis

### Batch Analysis

```bash
#!/bin/bash
# Analyze multiple directories

PROJECTS=(
  "data/datasets/SmellyCodeDataset/Java:dataset-java"
  "tools/baseline/test:baseline-test"
  "src/main/java:my-app"
)

for project in "${PROJECTS[@]}"; do
  IFS=':' read -r dir key <<< "$project"
  echo "Analyzing: $key"
  bash scripts/sonarqube_compile_and_analyze.sh "$dir" "$key"
done

echo "All analyses complete!"
```

### Pipeline Integration

```bash
#!/bin/bash
# Run full code analysis pipeline

SOURCE_DIR="$1"
PROJECT_KEY="$2"

# Run all baseline tools
echo "Running PMD..."
bash scripts/run_baseline_tools_docker.sh java "$SOURCE_DIR"

echo "Running SonarQube..."
bash scripts/sonarqube_compile_and_analyze.sh "$SOURCE_DIR" "$PROJECT_KEY"

echo "Analysis complete"
```

## Performance Notes

- **First analysis:** 30-60 seconds (includes download of SonarScanner)
- **Subsequent analyses:** 10-30 seconds (cached)
- **Compilation time:** 5-20 seconds (depends on file count)
- **Network bandwidth:** ~100 MB first-time (SonarScanner download)

## Files Generated

```
results/predictions/baseline/
├── sonarqube_<project-key>.json    # Main findings
└── sonarqube_<project-key>.raw.json # Raw API response (if needed)
```

## Architecture

```
┌─────────────────────────────────┐
│   Your Java Code                │
└────────────┬────────────────────┘
             │
             ↓ (docker run)
┌─────────────────────────────────┐
│   SonarScanner CLI              │
│   - Compiles sources            │
│   - Indexes files               │
│   - Sends analysis              │
└────────────┬────────────────────┘
             │
    ┌────────┴────────┐
    ↓                 ↓
 (docker network)
    │
    ↓
┌─────────────────────────────────┐
│   SonarQube 10.4                │
│   - Receives analysis           │
│   - Analyzes code               │
│   - Stores results              │
└────────────┬────────────────────┘
             │
             ↓ (curl API)
┌─────────────────────────────────┐
│   Your JSON Output              │
│   - Issues list                 │
│   - Metrics                     │
│   - Severity breakdown          │
└─────────────────────────────────┘
```

## Related Documentation

- [SonarQube User Guide](https://docs.sonarsource.com/sonarqube/latest/)
- [Java Quality Rules](https://rules.sonarsource.com/java/)
- [SonarQube REST API](https://docs.sonarsource.com/sonarqube/10.4/extension-guide/web-api/)

## Support

**Common issues:**
- Port 9000 in use → Change `docker-compose.yml` port mapping
- Out of memory → Increase Docker memory limit
- Slow analysis → Check Java files for compilation errors

**More help:**
```bash
# Show all available scripts
ls -lh scripts/sonarqube*.sh

# Check SonarQube logs
docker compose logs -f sonarqube

# Verify setup
curl http://localhost:9000/api/system/status | python3 -m json.tool
```
