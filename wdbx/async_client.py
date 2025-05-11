"""
async_client.py - Asynchronous WDBX client for the wdbx package.
"""

import asyncio
import inspect
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from functools import wraps
from contextlib import asynccontextmanager

# Underlying WDBX client (ensure the external package providing WDBX is installed)
from wdbx import WDBX as _WDBXClient

class AsyncWDBX:
    """
    Asynchronous interface to the WDBX client.
    """
    def __init__(self, vector_dimension: int, enable_plugins: bool = True, **kwargs):
        self._client = _WDBXClient(
            vector_dimension=vector_dimension,
            enable_plugins=enable_plugins,
            **kwargs
        )

    async def initialize(self) -> None:
        """Initialize the WDBX backend."""
        return await self._client.initialize()

    async def shutdown(self) -> None:
        """Shutdown the WDBX backend and release resources."""
        return await self._client.shutdown()

    async def store(self, vector: List[float], metadata: Dict) -> int:
        """Store a vector and its metadata. Returns the vector ID."""
        return await self._client.vector_store_async(vector, metadata)

    async def search(self, vector: List[float], limit: int = 10) -> List[Dict]:
        """Search for similar vectors. Returns a list of results."""
        return await self._client.vector_search_async(vector, limit=limit)

    async def bulk_store(self, items: List[Tuple[List[float], Dict]]) -> List[int]:
        """Store multiple vectors and metadata in bulk. Returns list of vector IDs."""
        return [await self.store(v, m) for v, m in items]

    async def bulk_search(self, vectors: List[List[float]], limit: int = 10) -> List[List[Dict]]:
        """Search for multiple vectors in bulk. Returns list of result lists."""
        return [await self.search(v, limit) for v in vectors]

    async def delete(self, vector_id: int) -> bool:
        return await self._client.vector_delete_async(vector_id)

    async def update_metadata(self, vector_id: int, metadata: Dict) -> bool:
        return await self._client.vector_update_metadata_async(vector_id, metadata)

    async def get_metadata(self, vector_id: int) -> Dict:
        return await self._client.vector_get_metadata_async(vector_id)

    async def count(self) -> int:
        return await self._client.vector_count_async()

    async def ping(self) -> bool:
        return await self._client.ping()

    async def flush(self) -> None:
        return await self._client.vector_flush_async()

    async def create_block(self, metadata: Dict) -> Dict:
        return await self._client.block_create_async(metadata)

    async def validate_block(self, block_id: int) -> bool:
        return await self._client.block_validate_async(block_id)

    async def begin_transaction(self) -> None:
        return await self._client.transaction_begin_async()

    async def commit_transaction(self) -> None:
        return await self._client.transaction_commit_async()

    async def rollback_transaction(self) -> None:
        return await self._client.transaction_rollback_async()

    async def neural_backtrack(self, pattern: List[float]) -> Dict:
        return await self._client.neural_backtracking_async(pattern)

    async def detect_drift(self, threshold: float = 0.1) -> bool:
        return await self._client.drift_detection_async(threshold)

    async def create_persona(self, name: str, config: Dict) -> int:
        return await self._client.persona_create_async(name, config)

    async def switch_persona(self, name: str) -> None:
        await self._client.persona_switch_async(name)

    async def list_personas(self) -> List[Dict]:
        return await self._client.persona_list_async()

    async def filter_content(self, text: str) -> str:
        return await self._client.content_filter_async(text)

    async def mitigate_bias(self, text: str) -> str:
        return await self._client.bias_mitigation_async(text)

    async def start_http_server(self, host: str = '127.0.0.1', port: int = 8000) -> None:
        return await self._client.http_server_start_async(host, port)

    async def stop_http_server(self) -> None:
        return await self._client.http_server_stop_async()

    async def get_api_endpoints(self) -> List[Dict]:
        return await self._client.api_endpoints_async()

    async def set_api_auth_required(self, required: bool = True) -> None:
        return await self._client.api_auth_required_async(required)

    async def set_api_cors_origins(self, origins: List[str]) -> None:
        return await self._client.api_cors_origins_async(origins)

    async def enable_plugin(self, plugin_name: str) -> bool:
        return await self._client.plugin_enable_async(plugin_name)

    async def disable_plugin(self, plugin_name: str) -> bool:
        return await self._client.plugin_disable_async(plugin_name)

    async def list_plugins(self) -> List[Dict]:
        return await self._client.plugin_list_async()

    async def visualize_vectors(self, vectors: List[List[float]], method: str = 'tsne') -> Dict:
        return await self._client.visualize_vectors_async(vectors, method)

    async def set_access_token(self, token: str) -> None:
        await self._client.set_access_token_async(token)

    async def encrypt_data(self, data: Any) -> str:
        return await self._client.encrypt_data_async(data)

    async def decrypt_data(self, encrypted_data: str) -> Any:
        return await self._client.decrypt_data_async(encrypted_data)

    async def generate_api_key(self, user_id: str) -> str:
        return await self._client.api_key_generate_async(user_id)

    async def revoke_api_key(self, key: str) -> bool:
        return await self._client.api_key_revoke_async(key)

    async def assign_role(self, user_id: str, roles: List[str]) -> bool:
        return await self._client.user_assign_role_async(user_id, roles)

    async def remove_role(self, user_id: str, roles: List[str]) -> bool:
        return await self._client.user_remove_role_async(user_id, roles)

    async def get_audit_logs(self, since: Optional[str] = None) -> List[Dict]:
        return await self._client.audit_logs_async(since)

    async def set_rate_limit(self, limit: int, period: int) -> None:
        return await self._client.rate_limit_async(limit, period)

    async def enable_cache(self) -> None:
        await self._client.cache_enable_async()

    async def disable_cache(self) -> None:
        await self._client.cache_disable_async()

    async def get_cache_stats(self) -> Dict:
        return await self._client.cache_stats_async()

    async def backup_database(self, path: str) -> None:
        await self._client.backup_async(path)

    async def restore_database(self, path: str) -> None:
        await self._client.restore_async(path)

    async def multi_head_attention(
        self,
        queries: List[List[float]],
        keys: List[List[float]],
        values: List[List[float]],
        num_heads: int
    ) -> List[List[float]]:
        return await self._client.multi_head_attention_async(queries, keys, values, num_heads)

    async def health_check(self) -> Dict[str, Any]:
        return {"ping": await self.ping(), "count": await self.count()}

    async def get_backend_version(self) -> str:
        return await self._client.get_version_async()

    async def supports_feature(self, feature: str) -> bool:
        return await self._client.supports_feature_async(feature)


def initialize_async_backend(*args, **kwargs) -> AsyncWDBX:
    """
    Convenience function to create and initialize an AsyncWDBX instance.
    """
    w = AsyncWDBX(*args, **kwargs)
    await w.initialize()
    return w 