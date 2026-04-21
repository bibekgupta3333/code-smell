"""
Background task processing and async queue management.
Handles asynchronous analysis jobs, caching, and timeout management.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque

from src.api.config import config


class TaskStatus(str, Enum):
    """Task execution status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class AnalysisTask:
    """Represents an analysis task in the queue."""
    analysis_id: str
    code: str
    language: str
    file_name: str
    include_rag: bool
    timeout_seconds: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.QUEUED
    result: Optional[Dict] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = config.MAX_RETRIES

    def to_dict(self) -> Dict:
        """Convert task to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["started_at"] = self.started_at.isoformat() if self.started_at else None
        data["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return data

    def is_expired(self) -> bool:
        """Check if task has exceeded timeout."""
        if not self.started_at:
            return False
        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        return elapsed > self.timeout_seconds

    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return self.retry_count < self.max_retries

    def can_process(self) -> bool:
        """Check if task can be processed."""
        return self.status in [TaskStatus.QUEUED, TaskStatus.FAILED]


class AsyncQueue:
    """Thread-safe async task queue with timeout and retry handling."""

    def __init__(self, max_size: int = config.MAX_QUEUE_SIZE):
        """
        Initialize async queue.

        Args:
            max_size: Maximum queue size
        """
        self.queue: deque = deque(maxlen=max_size)
        self.tasks: Dict[str, AnalysisTask] = {}
        self.max_size = max_size
        self.logger = logging.getLogger(__name__)

    def put(self, task: AnalysisTask) -> bool:
        """
        Add task to queue.

        Args:
            task: Task to add

        Returns:
            True if added successfully, False if queue is full
        """
        if len(self.queue) >= self.max_size:
            self.logger.warning(f"Queue full: {len(self.queue)}/{self.max_size}")
            return False

        self.queue.append(task)
        self.tasks[task.analysis_id] = task
        self.logger.info(f"Task {task.analysis_id} queued (queue size: {len(self.queue)})")
        return True

    def get(self) -> Optional[AnalysisTask]:
        """
        Remove and return next task from queue.

        Returns:
            Next task or None if queue is empty
        """
        if not self.queue:
            return None

        task = self.queue.popleft()
        self.logger.info(f"Task {task.analysis_id} dequeued (queue size: {len(self.queue)})")
        return task

    def get_task(self, analysis_id: str) -> Optional[AnalysisTask]:
        """
        Get task by analysis ID.

        Args:
            analysis_id: Analysis identifier

        Returns:
            Task or None if not found
        """
        return self.tasks.get(analysis_id)

    def update_task(self, task: AnalysisTask) -> None:
        """
        Update task state.

        Args:
            task: Updated task
        """
        self.tasks[task.analysis_id] = task

    def remove_task(self, analysis_id: str) -> Optional[AnalysisTask]:
        """
        Remove task from tracking.

        Args:
            analysis_id: Analysis identifier

        Returns:
            Removed task or None
        """
        return self.tasks.pop(analysis_id, None)

    def get_all_tasks(self) -> Dict[str, AnalysisTask]:
        """
        Get all tasks.

        Returns:
            Dictionary of all tasks
        """
        return dict(self.tasks)

    def get_active_tasks(self) -> list:
        """
        Get all active (processing) tasks.

        Returns:
            List of active tasks
        """
        return [t for t in self.tasks.values() if t.status == TaskStatus.PROCESSING]

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self.queue)

    def get_queue_depth(self) -> int:
        """Get total number of tracked tasks."""
        return len(self.tasks)

    def clear_expired_tasks(self) -> int:
        """
        Remove expired tasks.

        Returns:
            Number of tasks removed
        """
        expired = []
        for analysis_id, task in self.tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                age = (datetime.utcnow() - task.completed_at).total_seconds()
                if age > 3600:  # 1 hour
                    expired.append(analysis_id)

        for analysis_id in expired:
            self.remove_task(analysis_id)

        if expired:
            self.logger.info(f"Cleaned up {len(expired)} expired tasks")

        return len(expired)


class ResultCache:
    """In-memory result cache with TTL."""

    def __init__(self, max_size_mb: int = config.MAX_CACHE_SIZE_MB):
        """
        Initialize result cache.

        Args:
            max_size_mb: Maximum cache size in MB
        """
        self.cache: Dict[str, Dict] = {}
        self.max_size_mb = max_size_mb
        self.logger = logging.getLogger(__name__)

    def get(self, cache_key: str) -> Optional[Dict]:
        """
        Get cached result.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None if expired/missing
        """
        if cache_key not in self.cache:
            return None

        cached = self.cache[cache_key]
        age = (datetime.utcnow() - cached["cached_at"]).total_seconds()

        if age > config.CACHE_TTL:
            del self.cache[cache_key]
            self.logger.info(f"Cache expired for key {cache_key}")
            return None

        self.logger.info(f"Cache hit for key {cache_key}")
        return cached["result"]

    def put(self, cache_key: str, result: Dict) -> bool:
        """
        Store result in cache.

        Args:
            cache_key: Cache key
            result: Result to cache

        Returns:
            True if cached, False if cache is full
        """
        # Simple size estimation (in real implementation, use sys.getsizeof)
        if len(self.cache) >= (self.max_size_mb * 100):  # Rough estimate
            self.logger.warning("Cache full, skipping")
            return False

        self.cache[cache_key] = {
            "result": result,
            "cached_at": datetime.utcnow(),
        }
        self.logger.info(f"Result cached with key {cache_key}")
        return True

    def invalidate(self, cache_key: str) -> bool:
        """
        Invalidate cache entry.

        Args:
            cache_key: Cache key

        Returns:
            True if entry was removed
        """
        if cache_key in self.cache:
            del self.cache[cache_key]
            self.logger.info(f"Cache invalidated for key {cache_key}")
            return True
        return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        size = len(self.cache)
        self.cache.clear()
        self.logger.info(f"Cache cleared ({size} entries)")
        return size

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        expired = []
        now = datetime.utcnow()

        for key, data in self.cache.items():
            age = (now - data["cached_at"]).total_seconds()
            if age > config.CACHE_TTL:
                expired.append(key)

        for key in expired:
            del self.cache[key]

        if expired:
            self.logger.info(f"Cleaned up {len(expired)} expired cache entries")

        return len(expired)

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Cache stats
        """
        return {
            "size": len(self.cache),
            "max_size_mb": self.max_size_mb,
            "ttl_seconds": config.CACHE_TTL,
        }


class BackgroundTaskManager:
    """Manages background task execution with queue and caching."""

    def __init__(self):
        """Initialize background task manager."""
        self.queue = AsyncQueue()
        self.cache = ResultCache()
        self.logger = logging.getLogger(__name__)
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_ANALYSES)

    async def submit_task(self, task: AnalysisTask) -> bool:
        """
        Submit task to queue.

        Args:
            task: Task to submit

        Returns:
            True if submitted successfully
        """
        return self.queue.put(task)

    async def get_next_task(self) -> Optional[AnalysisTask]:
        """
        Get next task to process.

        Returns:
            Next task or None
        """
        return self.queue.get()

    async def process_task(
        self,
        task: AnalysisTask,
        processor: Callable,
    ) -> AnalysisTask:
        """
        Process a task with timeout handling.

        Args:
            task: Task to process
            processor: Async function to process task

        Returns:
            Updated task with results
        """
        async with self.semaphore:
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.utcnow()
            self.queue.update_task(task)

            try:
                # Run processor with timeout
                result = await asyncio.wait_for(
                    processor(task),
                    timeout=task.timeout_seconds,
                )
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()

                # Cache result
                if isinstance(result, dict):
                    cache_key = f"{task.analysis_id}:{task.code[:20]}"
                    self.cache.put(cache_key, result)

            except asyncio.TimeoutError:
                self.logger.error(f"Task {task.analysis_id} timeout")
                task.status = TaskStatus.TIMEOUT
                task.error = f"Task exceeded timeout of {task.timeout_seconds}s"
                task.completed_at = datetime.utcnow()

            except Exception as e:
                self.logger.error(f"Task {task.analysis_id} error: {str(e)}")
                if task.should_retry():
                    task.status = TaskStatus.QUEUED
                    task.retry_count += 1
                    self.logger.info(f"Retrying task {task.analysis_id} (attempt {task.retry_count})")
                    await asyncio.sleep(config.RETRY_DELAY * (config.RETRY_BACKOFF ** (task.retry_count - 1)))
                    self.queue.put(task)
                else:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.utcnow()

            finally:
                self.queue.update_task(task)

            return task

    def get_task_status(self, analysis_id: str) -> Optional[Dict]:
        """
        Get task status.

        Args:
            analysis_id: Analysis identifier

        Returns:
            Task status or None
        """
        task = self.queue.get_task(analysis_id)
        if task:
            return task.to_dict()
        return None

    def get_queue_stats(self) -> Dict:
        """Get queue statistics."""
        return {
            "queue_size": self.queue.get_queue_size(),
            "tracked_tasks": self.queue.get_queue_depth(),
            "active_tasks": len(self.queue.get_active_tasks()),
            "max_concurrent": config.MAX_CONCURRENT_ANALYSES,
            "cache_stats": self.cache.get_stats(),
        }

    async def cleanup(self) -> None:
        """Cleanup expired tasks and cache entries."""
        self.queue.clear_expired_tasks()
        self.cache.cleanup_expired()
        self.logger.info("Background task cleanup complete")


# Create singleton instance
background_task_manager = BackgroundTaskManager()
