#!/usr/bin/env python3
"""
Main test runner for concurrency and token generation testing.
"""
import asyncio
import time
import json
import os
import argparse
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from together import AsyncTogether

from config import TestConfig
from token_generator import TokenGenerator
from metrics import MetricsCalculator


# Base prompt template
BASE_PROMPT = """Solve the following competitive programming problem. Write a complete program that solves it correctly.

A famous story about the mathematicians G.H. Hardy and Srinivasa Ramanujan goes as follows (as told by Hardy): I remember once going to see him (Ramanujan) when he was lying ill at Putney. I had ridden in taxicab No. 1729, and remarked that the number seemed to be rather a dull one, and that I hoped it was not an unfavourable omen. "No", he replied, "it is a very interesting number; it is the smallest number expressible as the sum of two [positive] cubes in two different ways."

It is from this story the taxicab numbers got their name. The $n$'th taxicab numbers is defined to be the smallest number that can be expressed as a sum of two positive cube numbers in $n$ distinct ways.

Your task is to write a program that generates bus numbers; in particular, the largest bus number that is at most equal to some limit $m$."""


async def make_request(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: Optional[int],
    request_id: int,
    temperature: float = 0.7,
) -> Tuple[int, float, bool, str, Optional[int], Optional[int]]:
    """
    Make a single API request using AsyncTogether.
    
    Returns:
        (request_id, latency, success, error_message, input_tokens, output_tokens)
    """
    client = AsyncTogether(api_key=api_key) if api_key else AsyncTogether()
    
    start_time = time.time()
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = time.time() - start_time
        
        if response.choices and response.choices[0].message.content:
            input_tokens = getattr(response.usage, "prompt_tokens", None) if hasattr(response, "usage") else None
            output_tokens = getattr(response.usage, "completion_tokens", None) if hasattr(response, "usage") else None
            return (request_id, latency, True, "", input_tokens, output_tokens)
        else:
            return (request_id, latency, False, "Empty response", None, None)
            
    except Exception as e:
        latency = time.time() - start_time
        error_msg = str(e)
        return (request_id, latency, False, error_msg, None, None)


async def test_concurrency(
    n_concurrent: int,
    input_tokens: int,
    output_tokens: int,
    config: TestConfig,
    token_generator: TokenGenerator,
) -> Dict:
    """
    Test with n_concurrent simultaneous requests.
    
    Args:
        n_concurrent: Number of concurrent requests
        input_tokens: Target input token count
        output_tokens: Target output token count
        config: Test configuration
        token_generator: Token generator instance
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing: {n_concurrent} concurrent requests")
    print(f"Input tokens: {input_tokens:,}, Output tokens: {output_tokens:,}")
    print(f"{'='*60}")
    
    # Generate prompt with target input tokens
    prompt, max_tokens = token_generator.generate_output_prompt(output_tokens, BASE_PROMPT)
    prompt = token_generator.generate_prompt(input_tokens, prompt)
    
    # Verify token count (approximate)
    actual_input_tokens = token_generator.count_tokens(prompt)
    print(f"Generated prompt with ~{actual_input_tokens:,} tokens (target: {input_tokens:,})")
    
    # Create all tasks
    tasks = [
        make_request(
            config.api_key,
            config.model,
            prompt,
            max_tokens or config.max_tokens,
            i,
            config.temperature,
        )
        for i in range(n_concurrent)
    ]
    
    # Run all tasks concurrently
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    # Process results
    successful = 0
    failed = 0
    latencies = []
    errors = []
    individual_requests = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    for result in results:
        if isinstance(result, Exception):
            failed += 1
            errors.append(str(result))
            individual_requests.append({
                "request_id": len(individual_requests),
                "success": False,
                "latency": 0.0,
                "input_tokens": None,
                "output_tokens": None,
                "error": str(result),
            })
        else:
            request_id, latency, success, error_msg, inp_tokens, out_tokens = result
            latencies.append(latency)
            
            if inp_tokens:
                total_input_tokens += inp_tokens
            if out_tokens:
                total_output_tokens += out_tokens
            
            individual_requests.append({
                "request_id": request_id,
                "success": success,
                "latency": latency,
                "input_tokens": inp_tokens,
                "output_tokens": out_tokens,
                "error": error_msg if error_msg else None,
            })
            
            if success:
                successful += 1
            else:
                failed += 1
                if error_msg:
                    errors.append(error_msg)
    
    # Calculate metrics
    avg_input_tokens = total_input_tokens / n_concurrent if n_concurrent > 0 else 0
    avg_output_tokens = total_output_tokens / n_concurrent if n_concurrent > 0 else 0
    
    metrics = MetricsCalculator.calculate_metrics(
        latencies,
        successful,
        n_concurrent,
        int(avg_input_tokens) if avg_input_tokens > 0 else None,
        int(avg_output_tokens) if avg_output_tokens > 0 else None,
    )
    
    # Print summary
    print(f"\nResults:")
    print(f"  Success: {successful}/{n_concurrent} ({metrics['success_rate']:.1f}%)")
    print(f"  Failed: {failed}")
    print(f"  Average latency: {metrics['avg_latency']:.2f}s")
    print(f"  P50/P95/P99: {metrics['p50_latency']:.2f}s / {metrics['p95_latency']:.2f}s / {metrics['p99_latency']:.2f}s")
    print(f"  Min/Max: {metrics['min_latency']:.2f}s / {metrics['max_latency']:.2f}s")
    print(f"  Throughput: {metrics['throughput']:.2f} req/s")
    if metrics['token_throughput'] > 0:
        print(f"  Token throughput: {metrics['token_throughput']:.0f} tokens/s")
    if errors:
        unique_errors = list(set(errors[:5]))[:3]
        print(f"  Sample errors: {unique_errors}")
    
    return {
        "concurrency": n_concurrent,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "actual_input_tokens": int(avg_input_tokens) if avg_input_tokens > 0 else None,
        "actual_output_tokens": int(avg_output_tokens) if avg_output_tokens > 0 else None,
        "metrics": metrics,
        "total_time": total_time,
        "successful": successful,
        "failed": failed,
        "errors": errors[:10],  # Limit stored errors
        "individual_requests": individual_requests,
        "all_latencies": latencies,
    }


async def run_tests(config: TestConfig) -> Dict:
    """
    Run all test combinations.
    
    Args:
        config: Test configuration
        
    Returns:
        Dictionary with all test results
    """
    token_generator = TokenGenerator()
    
    results = []
    total_combinations = (
        len(config.concurrency_levels)
        * len(config.input_token_counts)
        * len(config.output_token_counts)
    )
    current = 0
    
    print(f"\n{'='*60}")
    print(f"Starting concurrency test suite")
    print(f"Model: {config.model}")
    print(f"Concurrency levels: {config.concurrency_levels}")
    print(f"Input token counts: {config.input_token_counts}")
    print(f"Output token counts: {config.output_token_counts}")
    print(f"Total test combinations: {total_combinations}")
    print(f"{'='*60}\n")
    
    for concurrency in config.concurrency_levels:
        for input_tokens in config.input_token_counts:
            for output_tokens in config.output_token_counts:
                current += 1
                print(f"\n[{current}/{total_combinations}] Running test...")
                
                result = await test_concurrency(
                    concurrency,
                    input_tokens,
                    output_tokens,
                    config,
                    token_generator,
                )
                results.append(result)
                
                # Small delay between test combinations
                await asyncio.sleep(1)
    
    return {
        "test_config": {
            "model": config.model,
            "concurrency_levels": config.concurrency_levels,
            "input_token_counts": config.input_token_counts,
            "output_token_counts": config.output_token_counts,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }


def save_results(results: Dict, output_dir: str = "results") -> str:
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"load_benchmark_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    return filepath


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Concurrency test suite for Together AI")
    parser.add_argument(
        "--model",
        type=str,
        default="vdabholkar_together/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF-57a64dac",
        help="Model name",
    )
    parser.add_argument(
        "--concurrency",
        type=str,
        default="32,64,128,256",
        help="Comma-separated concurrency levels (e.g., '32,64,128,256')",
    )
    parser.add_argument(
        "--input-tokens",
        type=str,
        default="1000,10000",
        help="Comma-separated input token counts (e.g., '1000,10000')",
    )
    parser.add_argument(
        "--output-tokens",
        type=str,
        default="1000,10000",
        help="Comma-separated output token counts (e.g., '1000,10000')",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Together API key (defaults to TOGETHER_API_KEY env var)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens parameter (defaults to output token count)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Create config
    config = TestConfig.from_args(
        model=args.model,
        concurrency=args.concurrency,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    # Run tests
    results = await run_tests(config)
    
    # Save results
    filepath = save_results(results, args.output_dir)
    print(f"\n{'='*60}")
    print(f"Results saved to: {filepath}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())

