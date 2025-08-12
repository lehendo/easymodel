#!/usr/bin/env python3
"""
Test runner for EasyModel API
Run all tests to verify the system is working correctly.
"""

import sys
import os
import subprocess
import time

# Add parent directory to path
sys.path.insert(0, '..')

def run_test(test_file, description):
    """Run a single test file and report results"""
    print(f"\n🧪 Running {description}...")
    print("=" * 50)
    
    try:
        # Change to tests directory and run test
        tests_dir = os.path.dirname(__file__)
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=60,
                              cwd=tests_dir)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 EasyModel API - Test Suite")
    print("=" * 50)
    
    tests = [
        ("test_analytics.py", "Analytics Function"),
        ("test_evaluate.py", "Model Evaluation"),
        ("test_finetune_local.py", "Fine-tuning Infrastructure"),
        ("test_direct.py", "Analytics Endpoint"),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_file, description in tests:
        if run_test(test_file, description):
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! EasyModel API is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
