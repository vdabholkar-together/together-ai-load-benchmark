#!/usr/bin/env python3
"""
Web dashboard for live concurrency test visualization.

This module provides a Flask-based web dashboard with SocketIO for real-time
concurrency testing of Together AI API. It allows users to test API performance
under various concurrency levels and max token configurations.
"""
import base64
import os
import sys
import json
import time
import math
import asyncio
import urllib.request
import tempfile
from pathlib import Path
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
from together import AsyncTogether, Together

# Utility placeholders (for future refactor; no behavior change now)
from utils.concurrency import register_concurrency_utils
from utils.images import register_image_utils
from utils.batch import register_batch_utils

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

# Batch API constraints
MAX_BATCH_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_BATCH_REQUESTS = 50000
MAX_BATCH_TOKENS_PER_MODEL = 30_000_000_000

# Placeholder registrations (no-ops today; kept for future refactors)
register_concurrency_utils(app, socketio)
register_image_utils(app)
register_batch_utils(app)


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
def home():
    """Landing page to navigate to concurrency, images, and batch tests."""
    return render_template("home.html")


@app.route("/home.html")
def home_html():
    """Alias for home landing page."""
    return render_template("home.html")


@app.route("/concurrency")
def index():
    """Serve the main concurrency dashboard page."""
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


# -----------------------------
# Images test page + API
# -----------------------------


def _safe_int(value, default, min_value=None, max_value=None):
    """Convert to int with bounds checking."""
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        ivalue = max(min_value, ivalue)
    if max_value is not None:
        ivalue = min(max_value, ivalue)
    return ivalue


def _percentile(values, pct):
    """Lightweight percentile calculation without numpy."""
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(vals[int(k)])
    d = k - f
    return float(vals[f] * (1 - d) + vals[c] * d)


def _make_run_dirs():
    """Create run directory grouped by date; returns (run_id, run_dir Path)."""
    now = datetime.now()
    date_dir = Path("results") / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{now.strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = date_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _save_uploaded_reference(file_storage, run_dir: Path):
    """Save uploaded reference image and return a data URL to use with the API."""
    filename = file_storage.filename or "reference"
    ext = ""
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1]
    safe_name = f"reference.{ext}" if ext else "reference"
    ref_path = run_dir / safe_name
    file_storage.save(ref_path)

    # Encode as data URL so it can be passed to Together as image_url/reference_images
    mime = "image/png" if ext.lower() == "png" else "image/jpeg"
    with open(ref_path, "rb") as rf:
        b64 = base64.b64encode(rf.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    return str(ref_path), data_url


def _save_generated_image(item, idx, run_dir: Path):
    """Persist generated image (url or base64) to disk; return local path."""
    target_path = run_dir / f"image_{idx}.png"
    # URL case
    if hasattr(item, "url") and item.url:
        try:
            urllib.request.urlretrieve(item.url, target_path)
            return str(target_path)
        except Exception:
            return None
    if isinstance(item, dict):
        if item.get("url"):
            try:
                urllib.request.urlretrieve(item["url"], target_path)
                return str(target_path)
            except Exception:
                return None
        if item.get("b64_json"):
            try:
                data = base64.b64decode(item["b64_json"])
                target_path.write_bytes(data)
                return str(target_path)
            except Exception:
                return None
    # b64 on object
    if hasattr(item, "b64_json") and item.b64_json:
        try:
            data = base64.b64decode(item.b64_json)
            target_path.write_bytes(data)
            return str(target_path)
        except Exception:
            return None
    return None


@app.route("/images-test")
def images_test():
    """Serve the images test page."""
    return render_template("images_test.html")


@app.route("/api/images-test", methods=["POST"])
def run_images_test_api():
    """
    Run a rate-limited image generation test against Together's images API.
    Default rate limit is 1 request/second. This endpoint is independent of
    existing concurrency tests and only targets image generation.
    """
    # Accept both JSON and multipart form (for file upload)
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        form = request.form
        payload = {
            "api_key": form.get("api_key", ""),
            "prompt": form.get("prompt", ""),
            "model": form.get("model", ""),
            "steps": form.get("steps"),
            "width": form.get("width"),
            "height": form.get("height"),
            "n": form.get("n"),
            "total_requests": form.get("total_requests"),
            "requests_per_second": form.get("requests_per_second"),
            "seed": form.get("seed"),
            "image_url": form.get("image_url", ""),
            "reference_images": form.get("reference_images", ""),
            "disable_safety_checker": form.get("disable_safety_checker"),
            "response_format": form.get("response_format", "url"),
            "negative_prompt": form.get("negative_prompt", ""),
        }
        reference_file = request.files.get("reference_file")
    else:
        payload = request.get_json(silent=True) or {}
        reference_file = None

    api_key = (payload.get("api_key") or os.environ.get("TOGETHER_API_KEY", "")).strip()
    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    model = (payload.get("model") or "black-forest-labs/FLUX.1-schnell").strip()
    steps = _safe_int(payload.get("steps"), 4, min_value=1, max_value=50)
    width = _safe_int(payload.get("width"), 1024, min_value=64, max_value=2048)
    height = _safe_int(payload.get("height"), 1024, min_value=64, max_value=2048)
    n = _safe_int(payload.get("n"), 1, min_value=1, max_value=4)
    total_requests = _safe_int(payload.get("total_requests"), 5, min_value=1, max_value=50)
    rps = _safe_int(payload.get("requests_per_second"), 1, min_value=1, max_value=10)
    seed = payload.get("seed")
    if seed is not None:
        seed = _safe_int(seed, None)
    image_url_raw = (payload.get("image_url") or "").strip()
    use_image_url = str(payload.get("use_image_url") or "").lower() in {"1", "true", "on", "yes"}
    image_url = image_url_raw if (use_image_url and image_url_raw) else None
    disable_safety = bool(payload.get("disable_safety_checker"))
    response_format = (payload.get("response_format") or "url").strip() or "url"
    negative_prompt = (payload.get("negative_prompt") or "").strip() or None

    # reference_images: comma-separated URLs
    ref_images = []
    ref_images_raw = payload.get("reference_images")
    if ref_images_raw:
        ref_images = [x.strip() for x in ref_images_raw.split(",") if x.strip()]

    run_id, run_dir = _make_run_dirs()
    saved_reference_local = None

    # If a reference file is uploaded, store it and convert to data URL so it can be sent
    if reference_file and reference_file.filename:
        saved_reference_local, data_url = _save_uploaded_reference(reference_file, run_dir)
        ref_images.append(data_url)
        # If requested, also set as image_url (only if user opted in)
        if use_image_url and not image_url:
            image_url = data_url

    client = Together(api_key=api_key)

    results = []
    success_count = 0
    fail_count = 0
    latencies = []
    errors = []

    delay = 1.0 / float(rps)
    saved_files = []

    for i in range(total_requests):
        start = time.time()
        try:
            # Use generate (current Together SDK) – create is not available in this version
            response = client.images.generate(
                prompt=prompt,
                model=model,
                # steps=steps,
                width=width,
                height=height,
                n=n,
                seed=seed,
                image_url=image_url if use_image_url else None,
                reference_images=ref_images or None,
                negative_prompt=negative_prompt,
                response_format=response_format,
                disable_safety_checker=disable_safety,
            )
            latency = time.time() - start

            urls = []
            b64s = []
            try:
                for item in response.data or []:
                    if hasattr(item, "url"):
                        urls.append(item.url)
                    elif isinstance(item, dict) and "url" in item:
                        urls.append(item["url"])
                    if hasattr(item, "b64_json") and item.b64_json:
                        b64s.append(item.b64_json)
                    elif isinstance(item, dict) and item.get("b64_json"):
                        b64s.append(item["b64_json"])
                    # save image to disk
                    saved = _save_generated_image(item, f"{i}_{len(urls)}", run_dir)
                    if saved:
                        saved_files.append(saved)
            except Exception:
                pass

            latencies.append(latency)
            success_count += 1
            results.append(
                {
                    "index": i,
                    "success": True,
                    "latency": latency,
                    "urls": urls,
                    "b64": b64s,
                }
            )
        except Exception as e:
            latency = time.time() - start
            fail_count += 1
            errors.append(str(e))
            results.append(
                {
                    "index": i,
                    "success": False,
                    "latency": latency,
                    "error": str(e),
                    "urls": [],
                    "b64": [],  
                }
            )

        if delay > 0:
            time.sleep(delay)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    summary = {
        "total_requests": total_requests,
        "successful": success_count,
        "failed": fail_count,
        "avg_latency": avg_latency,
        "p50_latency": _percentile(latencies, 50),
        "p95_latency": _percentile(latencies, 95),
        "p99_latency": _percentile(latencies, 99),
        "errors": errors[:5],
        "results": results,
        "run_id": run_id,
        "saved_files": saved_files,
        "run_dir": str(run_dir),
        "saved_reference": saved_reference_local,
    }

    # Save summary JSON for reference
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return jsonify(summary)


# -----------------------------
# Batch test page + API
# -----------------------------


@app.route("/batch-test")
def batch_test():
    """Serve the batch API test page."""
    return render_template("batch_test.html")


def _object_to_dict(obj):
    """Best-effort serialization helper for Together client objects."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return {"repr": repr(obj)}


@app.route("/api/batch/create", methods=["POST"])
def batch_create():
    """
    Create a batch job using Together Batch API.

    Validates:
    - API key provided
    - .jsonl file present
    - File size <= 100MB (Batch API limit)
    """
    api_key = (request.form.get("api_key") or os.environ.get("TOGETHER_API_KEY", "")).strip()
    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    upload = request.files.get("batch_file")
    if not upload or not upload.filename:
        return jsonify({"error": "Batch file (.jsonl) is required"}), 400

    filename = upload.filename
    if not filename.lower().endswith(".jsonl"):
        return jsonify({"error": "Batch file must be .jsonl"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            upload.save(tmp.name)
            tmp_path = tmp.name

        size = os.path.getsize(tmp_path)
        if size > MAX_BATCH_BYTES:
            return jsonify({"error": "Batch file exceeds 100MB limit"}), 400

        client = Together(api_key=api_key)
        file_resp = client.files.upload(file=tmp_path, purpose="batch-api", check=False)
        batch = client.batches.create_batch(file_resp.id, endpoint="/v1/chat/completions")

        file_id = file_resp.get("id") if isinstance(file_resp, dict) else getattr(file_resp, "id", None)
        batch_data = _object_to_dict(batch)

        return jsonify({
            "file_id": file_id,
            "batch": batch_data,
            "limits": {
                "max_requests": MAX_BATCH_REQUESTS,
                "max_tokens_per_model": MAX_BATCH_TOKENS_PER_MODEL,
                "max_file_bytes": MAX_BATCH_BYTES,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.route("/api/batch/status/<batch_id>")
def batch_status(batch_id):
    """Get batch status by ID."""
    api_key = (request.args.get("api_key") or os.environ.get("TOGETHER_API_KEY", "")).strip()
    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    try:
        client = Together(api_key=api_key)
        batch = client.batches.get_batch(batch_id)
        return jsonify(_object_to_dict(batch))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Run the Flask-SocketIO server
    # host="0.0.0.0" makes it accessible from any network interface
    # port=5000 is the default Flask port
    # debug=True enables debug mode (disable in production)
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
