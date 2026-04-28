import asyncio
import sys
sys.path.insert(0, '.')

from src.api.detection_integration import run_code_smell_detection_with_scoring
from src.utils.logger import setup_logging
from pathlib import Path

# Setup logging
setup_logging(log_dir=Path('results/logs'), log_name='test_workflow')

sample_code = '''
def process_user_data(items, config, cache, database, logger, metrics):
    result = []
    for i in range(len(items)):
        if items[i].is_valid():
            processed = items[i].process()
            result.append(processed)
            logger.log(f"Processed {i}")
    return result
'''

async def main():
    # Run analysis
    result = await run_code_smell_detection_with_scoring(
        code=sample_code,
        sample_id='test.py',
        use_rag=False,
        model='llama3:8b'
    )

    # Check results
    print(f"✓ Analysis completed")
    print(f"  - Findings: {len(result.get('findings', []))}")
    print(f"  - Success: {result.get('success', False)}")
    print(f"  - Metrics keys: {list(result.get('metrics', {}).keys())}")
    print(f"  - Model used: {result.get('model_used')}")
    print(f"  - F1 score: {result.get('metrics', {}).get('f1', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
