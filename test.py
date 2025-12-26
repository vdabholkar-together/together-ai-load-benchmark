#!/usr/bin/env python3
"""
Simple, fast concurrency testing - based on test_together_concurrency.py
Just tests concurrency with optional token control. No complex overhead.
"""
import asyncio
import time
import json
import os
import argparse
from typing import List, Dict, Tuple, Optional
from together import AsyncTogether
import numpy as np

# Simple prompt template
PROMPT_TEMPLATE = """Solve the following competitive programming problem. Write a complete program that solves it correctly.

{problem_statement}

Provide your solution as a complete, runnable program. Make sure to handle all edge cases and follow the input/output format exactly as specified."""

# Default problem
DEFAULT_PROBLEM = """A famous story about the mathematicians G.H. Hardy and Srinivasa Ramanujan goes as follows (as told by Hardy): I remember once going to see him (Ramanujan) when he was lying ill at Putney. I had ridden in taxicab No. 1729, and remarked that the number seemed to be rather a dull one, and that I hoped it was not an unfavourable omen. "No", he replied, "it is a very interesting number; it is the smallest number expressible as the sum of two [positive] cubes in two different ways."

It is from this story the taxicab numbers got their name. The $n$'th taxicab numbers is defined to be the smallest number that can be expressed as a sum of two positive cube numbers in $n$ distinct ways.

Your task is to write a program that generates bus numbers; in particular, the largest bus number that is at most equal to some limit $m$."""


async def make_request(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: Optional[int],
    request_id: int,
) -> Tuple[int, float, bool, str, int, int]:
    """Make a single API request - simple and fast."""
    client = AsyncTogether(api_key=api_key) if api_key else AsyncTogether()
    start_time = time.time()
    
    try:
        kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
            
        response = await client.chat.completions.create(**kwargs)
        latency = time.time() - start_time
        
        if response.choices and response.choices[0].message.content:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) if hasattr(response, "usage") else 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) if hasattr(response, "usage") else 0
            return (request_id, latency, True, "", input_tokens, output_tokens)
        else:
            return (request_id, latency, False, "Empty response", 0, 0)
    except Exception as e:
        latency = time.time() - start_time
        return (request_id, latency, False, str(e), 0, 0)


async def test_concurrency(
    n_concurrent: int,
    model: str,
    prompt: str,
    max_tokens: Optional[int],
    api_key: str,
) -> Dict:
    """Test with n_concurrent simultaneous requests - simple like original."""
    print(f"\nTesting {n_concurrent} concurrent requests...")
    
    # Create all tasks
    tasks = [make_request(api_key, model, prompt, max_tokens, i) for i in range(n_concurrent)]
    
    # Run all concurrently - all start at same instant
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    # Process results
    successful = 0
    failed = 0
    generation_times = []
    errors = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    for result in results:
        if isinstance(result, Exception):
            failed += 1
            errors.append(str(result))
            generation_times.append(None)
        else:
            request_id, gen_time, success, error_msg, inp_tokens, out_tokens = result
            generation_times.append(gen_time)
            if success:
                successful += 1
                total_input_tokens += inp_tokens
                total_output_tokens += out_tokens
            else:
                failed += 1
                if error_msg:
                    errors.append(error_msg)
    
    # Filter valid times
    valid_times = [t for t in generation_times if t is not None]
    avg_time = np.mean(valid_times) if valid_times else 0
    completion_rate = (successful / n_concurrent * 100) if n_concurrent > 0 else 0
    
    # Calculate metrics
    p50 = np.percentile(valid_times, 50) if valid_times else 0
    p95 = np.percentile(valid_times, 95) if valid_times else 0
    p99 = np.percentile(valid_times, 99) if valid_times else 0
    min_time = min(valid_times) if valid_times else 0
    max_time = max(valid_times) if valid_times else 0
    
    avg_input_tokens = total_input_tokens / successful if successful > 0 else 0
    avg_output_tokens = total_output_tokens / successful if successful > 0 else 0
    throughput = successful / total_time if total_time > 0 else 0
    token_throughput = (total_input_tokens + total_output_tokens) / total_time if total_time > 0 else 0
    
    # Print results
    print(f"  Completed: {successful}/{n_concurrent} ({completion_rate:.1f}%)")
    print(f"  Average generation time: {avg_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    if valid_times:
        print(f"  Min/Max time: {min_time:.2f}s / {max_time:.2f}s")
        print(f"  P50/P95/P99: {p50:.2f}s / {p95:.2f}s / {p99:.2f}s")
    if avg_input_tokens > 0:
        print(f"  Avg tokens: {avg_input_tokens:.0f} input, {avg_output_tokens:.0f} output")
        print(f"  Throughput: {throughput:.2f} req/s, {token_throughput:.0f} tokens/s")
    if errors:
        unique_errors = list(set(errors[:5]))[:3]
        print(f"  Sample errors: {unique_errors}")
    
    return {
        "n_concurrent": n_concurrent,
        "avg_time": float(avg_time),
        "completion_rate": completion_rate,
        "total_time": total_time,
        "successful": successful,
        "failed": failed,
        "p50_latency": float(p50),
        "p95_latency": float(p95),
        "p99_latency": float(p99),
        "min_latency": float(min_time),
        "max_latency": float(max_time),
        "throughput": throughput,
        "token_throughput": token_throughput,
        "input_tokens": int(avg_input_tokens),
        "output_tokens": int(avg_output_tokens),
        "all_times": valid_times,
        "errors": errors[:10],
    }


async def run_tests(
    concurrency_levels: List[int],
    model: str,
    prompt: str,
    max_tokens: Optional[int],
    api_key: str,
) -> List[Dict]:
    """Run all tests - simple and fast."""
    results = []
    for n_concurrent in concurrency_levels:
        result = await test_concurrency(n_concurrent, model, prompt, max_tokens, api_key)
        results.append(result)
        # Small delay to avoid overwhelming API
        await asyncio.sleep(0.5)
    return results


async def main():
    parser = argparse.ArgumentParser(description="Simple concurrency testing for Together AI")
    parser.add_argument("--model", default="vishwa/Llama-3.3-70B-Instruct-Turbo-256", help="Model name")
    parser.add_argument("--concurrency", default="32,64,128,256", help="Comma-separated concurrency levels")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens (optional)")
    parser.add_argument("--prompt", default=None, help="Custom prompt (optional)")
    parser.add_argument("--output", default="results", help="Output file prefix")
    args = parser.parse_args()
    
    # Get API key
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("Warning: TOGETHER_API_KEY not set")
    
    # Parse concurrency levels
    concurrency_levels = [int(x.strip()) for x in args.concurrency.split(",")]
    
    # Prepare prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = PROMPT_TEMPLATE.format(problem_statement=DEFAULT_PROBLEM)
    
    print(f"\n{'='*60}")
    print(f"Concurrency Test")
    print(f"Model: {args.model}")
    print(f"Concurrency levels: {concurrency_levels}")
    if args.max_tokens:
        print(f"Max tokens: {args.max_tokens}")
    print(f"{'='*60}")
    
    # Run tests
    results = await run_tests(concurrency_levels, args.model, prompt, args.max_tokens, api_key)
    
    # Save results
    output_json = f"{args.output}.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_json}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())

