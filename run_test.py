#!/usr/bin/env python
"""
Quick test runner for Metathin
Metathin 快速测试运行器
"""

import subprocess
import sys
import argparse


def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run Metathin tests')
    parser.add_argument('--module', type=str, help='Specific module to test')
    parser.add_argument('--cov', action='store_true', help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    verbose = '-v' if args.verbose else ''
    
    if args.module:
        # Run specific module | 运行特定模块
        cmd = f"python -m pytest tests/{args.module} {verbose}"
        return run_command(cmd, f"Testing module: {args.module}")
    
    # Run all tests | 运行所有测试
    cmd = f"python -m pytest tests/ {verbose}"
    run_command(cmd, "All tests")
    
    if args.cov:
        # Run coverage | 运行覆盖率
        cmd = "python -m pytest tests/ --cov=metathin --cov-report=term-missing"
        run_command(cmd, "Coverage report")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())