"""
Metrics calculation and aggregation for test results.
"""
import numpy as np
from typing import List, Dict, Optional


class MetricsCalculator:
    """Calculate various metrics from test results."""

    @staticmethod
    def calculate_percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        return float(np.percentile(data, percentile))

    @staticmethod
    def calculate_metrics(
        latencies: List[float],
        successful: int,
        total: int,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> Dict:
        """
        Calculate comprehensive metrics from test results.
        
        Args:
            latencies: List of latency values (in seconds)
            successful: Number of successful requests
            total: Total number of requests
            input_tokens: Optional input token count
            output_tokens: Optional output token count
            
        Returns:
            Dictionary with calculated metrics
        """
        if not latencies:
            return {
                "success_rate": 0.0,
                "avg_latency": 0.0,
                "p50_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0,
                "min_latency": 0.0,
                "max_latency": 0.0,
                "std_latency": 0.0,
                "throughput": 0.0,
                "token_throughput": 0.0,
            }

        latencies_array = np.array(latencies)
        
        metrics = {
            "success_rate": (successful / total * 100) if total > 0 else 0.0,
            "avg_latency": float(np.mean(latencies_array)),
            "p50_latency": MetricsCalculator.calculate_percentile(latencies, 50),
            "p95_latency": MetricsCalculator.calculate_percentile(latencies, 95),
            "p99_latency": MetricsCalculator.calculate_percentile(latencies, 99),
            "min_latency": float(np.min(latencies_array)),
            "max_latency": float(np.max(latencies_array)),
            "std_latency": float(np.std(latencies_array)),
        }

        # Calculate throughput (requests per second)
        if metrics["avg_latency"] > 0:
            metrics["throughput"] = 1.0 / metrics["avg_latency"]
        else:
            metrics["throughput"] = 0.0

        # Calculate token throughput (tokens per second)
        if input_tokens and output_tokens and metrics["avg_latency"] > 0:
            total_tokens = input_tokens + output_tokens
            metrics["token_throughput"] = total_tokens / metrics["avg_latency"]
        else:
            metrics["token_throughput"] = 0.0

        return metrics

