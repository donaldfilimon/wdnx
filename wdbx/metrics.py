"""
metrics.py - Prometheus and circuit breaker metrics for the wdbx package.
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Prometheus metrics for RPC calls
RPC_CALL_COUNT = Counter(
    'wdbx_rpc_calls_total', 'Total number of WDBX RPC calls', ['method']
)
RPC_CALL_LATENCY = Histogram(
    'wdbx_rpc_latency_seconds', 'WDBX RPC call latency in seconds', ['method']
)
CB_STATE = Gauge(
    'wdbx_circuit_breaker_state',
    'Circuit breaker state: 0 closed, 1 open'
)
CB_FAILURES = Gauge(
    'wdbx_circuit_breaker_failures',
    'Current consecutive failures in the circuit breaker'
)
CB_TOTAL_FAILURES = Counter(
    'wdbx_circuit_breaker_total_failures_total',
    'Total number of circuit breaker failures'
)

def start_metrics_server(port: int = 8000, addr: str = '0.0.0.0') -> None:
    """
    Start an HTTP server to expose Prometheus metrics on /metrics.
    """
    start_http_server(port, addr=addr) 