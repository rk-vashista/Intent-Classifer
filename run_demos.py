#!/usr/bin/env python3
"""
Script Runner - Easy way to run all demonstration scripts
"""

import os
import sys
import subprocess
from pathlib import Path

def run_script(script_name, description):
    """Run a script and handle output"""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, f"scripts/{script_name}"
        ], cwd=Path(__file__).parent, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully!")
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")

def main():
    """Main script runner"""
    print("🎯 ADVANCED INTENT CLASSIFICATION SYSTEM - SCRIPT RUNNER")
    print("=" * 60)
    
    scripts = [
        ("comprehensive_demo.py", "Comprehensive System Demonstration"),
        ("demo_comparison.py", "Rule-based vs Transformer Comparison")
    ]
    
    print("\nAvailable scripts:")
    for i, (script, desc) in enumerate(scripts, 1):
        print(f"  {i}. {script} - {desc}")
    
    print("\nOptions:")
    print("  a. Run all scripts")
    print("  1-2. Run specific script")
    print("  q. Quit")
    
    choice = input("\nEnter your choice: ").strip().lower()
    
    if choice == 'q':
        print("👋 Goodbye!")
        return
    elif choice == 'a':
        for script, desc in scripts:
            run_script(script, desc)
    elif choice in ['1', '2']:
        idx = int(choice) - 1
        if 0 <= idx < len(scripts):
            script, desc = scripts[idx]
            run_script(script, desc)
        else:
            print("❌ Invalid choice!")
    else:
        print("❌ Invalid choice!")
    
    print(f"\n🎯 RESULTS SUMMARY:")
    print(f"📁 All results saved in: results/")
    print(f"📊 Comprehensive analysis: results/comprehensive_analysis.json")
    print(f"📈 Latest predictions: results/modular_results.json")

if __name__ == "__main__":
    main()
