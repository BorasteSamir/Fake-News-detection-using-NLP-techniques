"""
test_project.py
===============
Simple test script to verify the refactored project structure works correctly.

Run this to test imports and basic functionality:
    python test_project.py
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from data_loader import load_data, split_data
        print("[OK] data_loader imports successful")
    except ImportError as e:
        print(f"[FAIL] data_loader import failed: {e}")
        return False
    
    try:
        from preprocess import preprocess_text, preprocess_series
        print("[OK] preprocess imports successful")
    except ImportError as e:
        print(f"[FAIL] preprocess import failed: {e}")
        return False
    
    try:
        from features import get_bow_features, get_tfidf_features
        print("[OK] features imports successful")
    except ImportError as e:
        print(f"[FAIL] features import failed: {e}")
        return False
    
    try:
        from train import train_svm, evaluate_model
        print("[OK] train imports successful")
    except ImportError as e:
        print(f"[FAIL] train import failed: {e}")
        return False
    
    return True

def test_preprocessing():
    """Test basic preprocessing functionality."""
    print("\nTesting preprocessing...")
    
    try:
        from preprocess import preprocess_text
        
        # Test text
        sample_text = "This is a TEST article with URLs http://example.com and numbers 123!"
        cleaned = preprocess_text(sample_text)
        
        print(f"Original: {sample_text}")
        print(f"Cleaned:  {cleaned}")
        
        if cleaned and len(cleaned) > 0:
            print("[OK] Preprocessing works correctly")
            return True
        else:
            print("[FAIL] Preprocessing returned empty result")
            return False
            
    except Exception as e:
        print(f"[FAIL] Preprocessing test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        from data_loader import load_data
        
        # Check if data files exist
        if not os.path.exists("data/Fake.csv") or not os.path.exists("data/Real.csv"):
            print("[SKIP] Data files not found - skipping data loading test")
            print("       Download dataset from Kaggle and place in data/ directory")
            return True
        
        # Try to load a small sample
        df = load_data()
        
        if df is not None and len(df) > 0:
            print(f"[OK] Data loading successful - loaded {len(df)} samples")
            return True
        else:
            print("[FAIL] Data loading returned empty DataFrame")
            return False
            
    except Exception as e:
        print(f"[FAIL] Data loading test failed: {e}")
        return False

def test_project_structure():
    """Test that all required directories and files exist."""
    print("\nTesting project structure...")
    
    required_dirs = ["src", "data", "models", "outputs", "dashboard"]
    required_files = [
        "main.py", "requirements.txt", "README.md",
        "src/__init__.py", "src/data_loader.py", "src/preprocess.py", 
        "src/features.py", "src/train.py",
        "dashboard/dashboard.py", "dashboard/README.md"
    ]
    
    missing_dirs = []
    missing_files = []
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"[FAIL] Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"[FAIL] Missing files: {missing_files}")
        return False
    
    print("[OK] Project structure is correct")
    return True

def main():
    """Run all tests."""
    print("Testing Refactored Fake News Detection Project")
    print("=" * 60)
    
    tests = [
        test_project_structure,
        test_imports,
        test_preprocessing,
        test_data_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("   1. Run: python main.py")
        print("   2. Launch dashboard: cd dashboard && streamlit run dashboard.py")
    else:
        print("Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)