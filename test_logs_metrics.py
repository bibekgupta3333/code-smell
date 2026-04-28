import asyncio
import sys
import json
from pathlib import Path
sys.path.insert(0, '.')

from src.api.detection_integration import run_code_smell_detection_with_scoring
from src.utils.logger import setup_logging

# Setup logging  
setup_logging(log_dir=Path('results/logs'), log_name='final_test')

sample_code = '''
def process_data(items, config, cache, database, logger, metrics):
    result = []
    for i in range(len(items)):
        if items[i].is_valid():
            processed = items[i].process()
            result.append(processed)
            logger.log(f"Processed {i}")
            cache.set(f"item_{i}", processed)
            database.save(processed)
    return result
'''

print("🔍 Running code smell analysis...")
result = asyncio.run(run_code_smell_detection_with_scoring(
    code=sample_code,
    sample_id='test.py',
    use_rag=False,
    model='llama3:8b'
))

print("\n✅ ANALYSIS RESULTS:")
print(f"  Success: {result.get('success')}")
print(f"  Findings: {len(result.get('findings', []))}")
print(f"  Model Used: {result.get('model_used')}")

print("\n📊 EVALUATION METRICS:")
metrics = result.get('metrics', {})
print(f"  F1 Score: {metrics.get('f1', 'N/A')}")
print(f"  Precision: {metrics.get('precision', 'N/A')}")
print(f"  Recall: {metrics.get('recall', 'N/A')}")
print(f"  Evaluation Mode: {metrics.get('evaluation_mode', 'unknown')}")
print(f"  Has Ground Truth: {metrics.get('has_ground_truth')}")

print("\n📝 DETECTED ISSUES:")
for finding in result.get('findings', [])[:3]:
    print(f"  - {finding['smell_type']} ({finding['severity']})")

print("\n📄 LOGS CHECK:")
log_files = list(Path('results/logs').glob('*.json'))
if log_files:
    latest_log = sorted(log_files, key=lambda x: x.stat().st_mtime)[-1]
    print(f"  ✓ Latest log file: {latest_log.name}")
    with open(latest_log) as f:
        lines = f.readlines()
        print(f"  ✓ Total log entries: {len(lines)}")
        if lines:
            try:
                print(f"  ✓ First log: {json.loads(lines[0]).get('message', 'N/A')}")
                print(f"  ✓ Last log: {json.loads(lines[-1]).get('message', 'N/A')}")
            except:
                 print(f"  ✓ First log: {lines[0][:50]}...")
                 print(f"  ✓ Last log: {lines[-1][:50]}...")
else:
    print("  ✗ No log files found")

print("\n✅ ALL TESTS COMPLETE")
