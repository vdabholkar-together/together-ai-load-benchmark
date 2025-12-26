#!/usr/bin/env python3
"""
Web dashboard for live concurrency test visualization.

This module provides a Flask-based web dashboard with SocketIO for real-time
concurrency testing of Together AI API. It allows users to test API performance
under various concurrency levels and max token configurations.
"""
import os
import sys
import json
import time
import asyncio
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading

# Add src directory to Python path to import test modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import TestConfig
from test_runner import run_tests, save_results

# Also import from test.py for fast mode (simple concurrency testing)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from test import make_request, PROMPT_TEMPLATE, DEFAULT_PROBLEM

# Initialize Flask application
app = Flask(__name__)

# Configure Flask secret key - use environment variable in production
# This key is used for session management and CSRF protection
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'concurrency-test-secret-key-change-in-production')

# Configure CORS (Cross-Origin Resource Sharing) origins
# In production, set CORS_ORIGINS environment variable to restrict access
cors_origins = os.environ.get('CORS_ORIGINS', '*')
socketio = SocketIO(app, cors_allowed_origins=cors_origins)

# Global state to track current test execution
# This stores the current test status, results, and progress information
current_test_state = {
    "running": False,      # Whether a test is currently running
    "results": [],          # List of test results
    "current_test": None,   # Current test configuration
    "progress": {"current": 0, "total": 0},  # Progress tracking
}


def validate_input(config_dict: dict) -> tuple:
    """
    Validate and sanitize user input to prevent security issues.
    
    This function performs comprehensive validation on all user inputs including:
    - API key validation (required)
    - Model name sanitization (prevent path traversal)
    - Concurrency level validation (must be positive, max 256)
    - Max tokens validation (must be positive, max 200k)
    
    Args:
        config_dict: Dictionary containing user input configuration
        
    Returns:
        Tuple of (is_valid: bool, error_message: str or None, validated_config: dict or None)
        - If validation passes: (True, None, validated_config)
        - If validation fails: (False, error_message, None)
    """
    # Step 1: Validate API key
    # Get API key from user input or fallback to environment variable
    api_key = config_dict.get("api_key", "").strip()
    if not api_key:
        # Fallback to environment variable if not provided in UI
        api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        return False, "API key is required. Please provide it in the UI or set TOGETHER_API_KEY environment variable.", None
    
    # Step 2: Validate and sanitize model name
    # Model names can contain forward slashes (e.g., "meta-llama/Meta-Llama-3.1")
    # but we need to prevent path traversal attacks
    model = config_dict.get("model", "").strip()
    if not model:
        return False, "Model name is required.", None
    
    # Prevent path traversal patterns (.. and backslashes)
    # Allow forward slashes as they're common in model names
    if ".." in model or "\\" in model:
        return False, "Invalid model name.", None
    
    # Set reasonable length limit to prevent DoS
    if len(model) > 200:
        return False, "Model name is too long.", None
    
    # Step 3: Validate concurrency levels
    # Concurrency levels must be comma-separated positive integers
    # Maximum allowed concurrency is 256 to prevent resource exhaustion
    try:
        concurrency_str = config_dict.get("concurrency", "").strip()
        if not concurrency_str:
            return False, "Concurrency levels are required.", None
        
        concurrency_levels = []
        # Parse comma-separated concurrency values
        for x in concurrency_str.split(","):
            x = x.strip()
            if not x:  # Skip empty values
                continue
            
            # Convert to integer and validate
            val = int(x)
            if val <= 0:
                return False, f"Concurrency level must be positive, got: {val}", None
            
            # Maximum concurrency limit: 256
            if val > 256:
                return False, f"Concurrency level too high (max 256), got: {val}", None
            
            concurrency_levels.append(val)
        
        # Ensure at least one valid concurrency level
        if not concurrency_levels:
            return False, "At least one concurrency level is required.", None
        
        # Limit total number of concurrency levels to prevent too many tests
        if len(concurrency_levels) > 20:
            return False, "Maximum 20 concurrency levels allowed.", None
        
    except ValueError as e:
        return False, f"Invalid concurrency format: {str(e)}", None
    
    # Step 4: Validate max tokens
    # Max tokens can be a single value or comma-separated list
    # Maximum allowed is 200k (input + output combined)
    max_tokens_list = []
    max_tokens_str = config_dict.get("max_tokens", "").strip()
    
    if max_tokens_str:
        try:
            # Parse comma-separated max token values
            for x in max_tokens_str.split(","):
                x = x.strip()
                if not x:  # Skip empty values
                    continue
                
                # Convert to integer and validate
                val = int(x)
                if val <= 0:
                    return False, f"Max tokens must be positive, got: {val}", None
                
                # Maximum tokens limit: 200,000 (200k)
                # This represents the total token limit (input + output)
                if val > 200000:
                    return False, f"Max tokens too high (max 200000), got: {val}", None
                
                max_tokens_list.append(val)
            
            # Limit number of max token values to prevent too many test combinations
            if len(max_tokens_list) > 10:
                return False, "Maximum 10 max token values allowed.", None
                
        except ValueError as e:
            return False, f"Invalid max tokens format: {str(e)}", None
    
    # If no max tokens specified, use None (will use model default)
    if not max_tokens_list:
        max_tokens_list = [None]
    
    # Return validated configuration dictionary
    validated_config = {
        "api_key": api_key,
        "model": model,
        "concurrency_levels": concurrency_levels,
        "max_tokens_list": max_tokens_list,
    }
    
    return True, None, validated_config


def run_test_async(config_dict: dict):
    """
    Run concurrency test asynchronously in a background thread.
    
    This function:
    1. Validates user input
    2. Sets up test state
    3. Runs the matrix test (all concurrency x max_tokens combinations)
    4. Saves results to file
    5. Emits progress updates via SocketIO
    
    Args:
        config_dict: Dictionary containing test configuration from user input
    """
    global current_test_state
    
    try:
        # Step 1: Validate all user inputs
        is_valid, error_msg, validated_config = validate_input(config_dict)
        if not is_valid:
            # Emit error to client if validation fails
            socketio.emit("test_error", {"error": error_msg})
            return
        
        # Extract validated configuration
        api_key = validated_config["api_key"]
        model = validated_config["model"]
        concurrency_levels = validated_config["concurrency_levels"]
        max_tokens_list = validated_config["max_tokens_list"]
        
        # Step 2: Calculate total number of test combinations
        # This is concurrency_levels × max_tokens_list
        total_combinations = len(concurrency_levels) * len(max_tokens_list)
        
        # Step 3: Initialize test state
        current_test_state["running"] = True
        current_test_state["results"] = []
        current_test_state["progress"] = {"current": 0, "total": total_combinations}
        
        # Step 4: Emit test started event to notify client
        socketio.emit("test_started", {
            "config": {
                "model": model,
                "concurrency_levels": concurrency_levels,
                "max_tokens_list": max_tokens_list,
            },
            "total_combinations": total_combinations,
        })
        
        # Step 5: Run async test in a new event loop
        # We create a new event loop because we're in a background thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(run_matrix_tests(
            model, concurrency_levels, max_tokens_list, socketio, api_key
        ))
        loop.close()
        
        # Step 6: Save results to file
        # Sanitize output directory to prevent path traversal
        output_dir = "results"
        if ".." in output_dir or "/" in output_dir or "\\" in output_dir:
            output_dir = "results"  # Reset to safe default
        
        filepath = save_results(results, output_dir)
        
        # Step 7: Update test state and notify client of completion
        current_test_state["running"] = False
        current_test_state["results"] = results["results"]
        
        socketio.emit("test_completed", {
            "results": results,
            "filepath": filepath,
        })
        
    except Exception as e:
        # Step 8: Handle errors gracefully
        # Don't leak sensitive information in error messages
        current_test_state["running"] = False
        error_msg = "An error occurred during testing. Please check your configuration and try again."
        socketio.emit("test_error", {"error": error_msg})
        # Log full error server-side for debugging (not sent to client)
        print(f"Error in run_test_async: {str(e)}", flush=True)


async def run_matrix_tests(
    model: str, 
    concurrency_levels: list, 
    max_tokens_list: list, 
    socketio_instance, 
    api_key: str
):
    """
    Run matrix tests: test all combinations of concurrency × max_tokens.
    
    This function performs comprehensive testing by running tests for every
    combination of concurrency level and max tokens value. This provides insights
    into how max tokens affects performance at different concurrency levels.
    
    Example: If concurrency_levels=[32, 64] and max_tokens_list=[500, 1000],
    this will run 4 tests: (32,500), (32,1000), (64,500), (64,1000)
    
    Args:
        model: Together AI model name to test
        concurrency_levels: List of concurrency levels to test (e.g., [32, 64, 128])
        max_tokens_list: List of max token values to test (e.g., [500, 1000, 2000])
        socketio_instance: SocketIO instance for emitting progress updates
        api_key: Together AI API key
        
    Returns:
        Dictionary containing test configuration and results
    """
    import numpy as np
    from together import AsyncTogether
    
    # Prepare the prompt template with default problem statement
    # This prompt will be used for all test requests
    prompt = PROMPT_TEMPLATE.format(problem_statement=DEFAULT_PROBLEM)
    
    # Initialize results list to store all test results
    results = []
    test_idx = 0
    total_tests = len(concurrency_levels) * len(max_tokens_list)
    
    # Step 1: Iterate through all combinations of concurrency × max_tokens
    for n_concurrent in concurrency_levels:
        for max_tokens in max_tokens_list:
            test_idx += 1
            
            # Display progress information
            max_tokens_display = max_tokens if max_tokens else "default"
            print(f"\n[{test_idx}/{total_tests}] Testing {n_concurrent} concurrent requests, max_tokens={max_tokens_display}...")
            
            # Step 2: Emit progress update to client via SocketIO
            socketio_instance.emit("test_progress", {
                "current": test_idx,
                "total": total_tests,
                "current_test": {
                    "concurrency": n_concurrent,
                    "max_tokens": max_tokens,
                },
            })
            
            # Step 3: Create concurrent request tasks
            # Each task makes one API request with the specified parameters
            tasks = [make_request(api_key, model, prompt, max_tokens, i) for i in range(n_concurrent)]
            
            # Step 4: Execute all requests concurrently and measure total time
            start_time = time.time()
            test_results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Step 5: Process results and collect metrics
            successful = 0
            failed = 0
            latencies = []
            errors = []
            total_input_tokens = 0
            total_output_tokens = 0
            
            # Iterate through each request result
            for result in test_results:
                if isinstance(result, Exception):
                    # Handle exceptions (network errors, API errors, etc.)
                    failed += 1
                    errors.append(str(result))
                else:
                    # Extract result tuple: (request_id, latency, success, error_msg, input_tokens, output_tokens)
                    request_id, latency, success, error_msg, inp_tokens, out_tokens = result
                    
                    if success:
                        # Count successful requests and collect metrics
                        successful += 1
                        latencies.append(latency)
                        total_input_tokens += inp_tokens
                        total_output_tokens += out_tokens
                    else:
                        # Count failed requests and collect error messages
                        failed += 1
                        if error_msg:
                            errors.append(error_msg)
            
            # Step 6: Calculate performance metrics
            # Average latency across all successful requests
            avg_latency = np.mean(latencies) if latencies else 0
            
            # Latency percentiles (P50, P95, P99) for understanding distribution
            p50 = np.percentile(latencies, 50) if latencies else 0
            p95 = np.percentile(latencies, 95) if latencies else 0
            p99 = np.percentile(latencies, 99) if latencies else 0
            
            # Min and max latencies
            min_latency = min(latencies) if latencies else 0
            max_latency = max(latencies) if latencies else 0
            
            # Success rate as percentage
            completion_rate = (successful / n_concurrent * 100) if n_concurrent > 0 else 0
            
            # Throughput: successful requests per second
            throughput = successful / total_time if total_time > 0 else 0
            
            # Token throughput: total tokens (input + output) processed per second
            token_throughput = (total_input_tokens + total_output_tokens) / total_time if total_time > 0 else 0
            
            # Average tokens per request
            avg_input_tokens = total_input_tokens / successful if successful > 0 else 0
            avg_output_tokens = total_output_tokens / successful if successful > 0 else 0
            
            # Step 7: Print summary to console for server-side monitoring
            print(f"  Completed: {successful}/{n_concurrent} ({completion_rate:.1f}%)")
            print(f"  Average generation time: {avg_latency:.2f}s")
            print(f"  Total time: {total_time:.2f}s")
            if latencies:
                print(f"  Min/Max time: {min_latency:.2f}s / {max_latency:.2f}s")
            if errors:
                unique_errors = list(set(errors[:5]))[:3]
                print(f"  Sample errors: {unique_errors}")
            
            # Step 8: Create result dictionary with all metrics
            result = {
                "concurrency": n_concurrent,
                "max_tokens": max_tokens,
                "input_tokens": int(avg_input_tokens),
                "output_tokens": int(avg_output_tokens),
                "metrics": {
                    "success_rate": completion_rate,
                    "avg_latency": float(avg_latency),
                    "p50_latency": float(p50),
                    "p95_latency": float(p95),
                    "p99_latency": float(p99),
                    "throughput": throughput,
                    "token_throughput": token_throughput,
                },
                "successful": successful,
                "failed": failed,
                "total_time": total_time,
                "all_latencies": latencies,
            }
            results.append(result)
            
            # Step 9: Emit result to client for real-time display
            socketio_instance.emit("test_result", {"result": result})
            
            # Small delay between test combinations to avoid overwhelming the API
            await asyncio.sleep(0.5)
    
    # Step 10: Return complete test results with configuration metadata
    return {
        "test_config": {
            "model": model,
            "concurrency_levels": concurrency_levels,
            "max_tokens_list": max_tokens_list,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }


async def run_tests_with_updates(config: TestConfig, socketio_instance):
    """
    Run tests with token matrix (advanced mode).
    
    This function is for advanced testing with input/output token combinations.
    Currently not used in the simplified UI but kept for future use.
    
    Args:
        config: TestConfig object with full configuration
        socketio_instance: SocketIO instance for emitting updates
        
    Returns:
        Dictionary containing test results
    """
    from token_generator import TokenGenerator
    from test_runner import test_concurrency, save_results
    
    token_generator = TokenGenerator()
    results = []
    
    # Calculate total combinations: concurrency × input_tokens × output_tokens
    total_combinations = (
        len(config.concurrency_levels)
        * len(config.input_token_counts)
        * len(config.output_token_counts)
    )
    current = 0
    
    # Iterate through all combinations
    for concurrency in config.concurrency_levels:
        for input_tokens in config.input_token_counts:
            for output_tokens in config.output_token_counts:
                current += 1
                
                # Emit progress update
                socketio_instance.emit("test_progress", {
                    "current": current,
                    "total": total_combinations,
                    "current_test": {
                        "concurrency": concurrency,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                })
                
                # Run test for this combination
                result = await test_concurrency(
                    concurrency,
                    input_tokens,
                    output_tokens,
                    config,
                    token_generator,
                )
                results.append(result)
                
                # Emit result update
                socketio_instance.emit("test_result", {
                    "result": result,
                })
                
                # Small delay between tests
                await asyncio.sleep(0.5)
    
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


@app.route("/")
def index():
    """
    Serve the main dashboard page.
    
    Returns:
        Rendered HTML template for the dashboard
    """
    return render_template("index.html")


@app.route("/api/status")
def status():
    """
    Get current test status via REST API.
    
    Returns:
        JSON response with current test state
    """
    return jsonify(current_test_state)


@app.route("/api/results")
def get_results():
    """
    Get list of saved test result files.
    
    This endpoint scans the results directory and returns metadata about
    all saved JSON result files. Includes security checks to prevent
    path traversal attacks.
    
    Returns:
        JSON response with list of result files and their metadata
    """
    results_dir = "results"
    results_files = []
    
    # Sanitize directory path to prevent path traversal
    if ".." in results_dir or "/" in results_dir or "\\" in results_dir:
        results_dir = "results"  # Reset to safe default
    
    if os.path.exists(results_dir):
        try:
            # Iterate through files in results directory
            for filename in os.listdir(results_dir):
                # Validate filename to prevent path traversal
                if ".." in filename or "/" in filename or "\\" in filename:
                    continue
                
                # Only process JSON files
                if not filename.endswith(".json"):
                    continue
                
                # Construct file path
                filepath = os.path.join(results_dir, filename)
                
                # Additional safety check: ensure file is within results directory
                if not os.path.abspath(filepath).startswith(os.path.abspath(results_dir)):
                    continue
                
                try:
                    # Read and parse JSON file
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        results_files.append({
                            "filename": filename,
                            "timestamp": data.get("test_config", {}).get("timestamp", ""),
                            "model": data.get("test_config", {}).get("model", ""),
                        })
                except (json.JSONDecodeError, IOError):
                    # Skip invalid or unreadable files
                    pass
        except (OSError, PermissionError):
            # Handle permission errors gracefully
            pass
    
    return jsonify({"results": results_files})


@socketio.on("start_test")
def handle_start_test(data):
    """
    Handle SocketIO event to start a new test.
    
    This function is called when the client emits a "start_test" event.
    It validates that no test is currently running, then starts a new
    test in a background thread.
    
    Args:
        data: Dictionary containing test configuration from client
    """
    # Check if a test is already running
    if current_test_state["running"]:
        emit("error", {"message": "Test already running"})
        return
    
    # Start test in background thread to avoid blocking
    # This allows the server to handle other requests while tests run
    thread = threading.Thread(target=run_test_async, args=(data,))
    thread.daemon = True  # Thread will exit when main program exits
    thread.start()
    
    # Notify client that test is starting
    emit("test_starting", {"message": "Test started"})


@socketio.on("stop_test")
def handle_stop_test():
    """
    Handle SocketIO event to stop the current test.
    
    Note: This is a simple implementation. In production, you'd want
    to properly cancel the async tasks and clean up resources.
    """
    # Set running flag to False
    # The actual cancellation of async tasks would require more complex logic
    current_test_state["running"] = False
    emit("test_stopped", {"message": "Test stopped"})


if __name__ == "__main__":
    # Run the Flask-SocketIO server
    # host="0.0.0.0" makes it accessible from any network interface
    # port=5000 is the default Flask port
    # debug=True enables debug mode (disable in production)
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
