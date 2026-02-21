from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class Worker:
    host_port: str

    @property
    def http_base_url(self) -> str:
        return f"http://{self.host_port}"

    @property
    def ws_base_url(self) -> str:
        return f"ws://{self.host_port}"


class WorkerPool:
    def __init__(self, worker_hosts: tuple[str, ...]) -> None:
        if not worker_hosts:
            raise ValueError("worker_hosts cannot be empty")
        self._workers: tuple[Worker, ...] = tuple(Worker(host_port=host) for host in worker_hosts)
        self._queue: asyncio.Queue[Worker] = asyncio.Queue(maxsize=len(self._workers))
        for worker in self._workers:
            self._queue.put_nowait(worker)

    @property
    def workers(self) -> tuple[Worker, ...]:
        return self._workers

    @property
    def capacity(self) -> int:
        return len(self._workers)

    @property
    def available(self) -> int:
        return self._queue.qsize()

    @asynccontextmanager
    async def lease(self):
        worker = await self._queue.get()
        try:
            yield worker
        finally:
            self._queue.put_nowait(worker)
