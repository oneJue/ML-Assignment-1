import sys
import time
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib
import requests

if getattr(sys, 'frozen', False):
    current_dir = os.path.dirname(os.path.abspath(sys.executable))
    test_csv_path = os.path.join(sys._MEIPASS, 'test.csv')
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_csv_path = os.path.join(current_dir, 'test.csv')

sys.path.insert(0, current_dir)

LEADERBOARD_URL = "http://172.23.166.133:8000/api/submit"
ASSIGNMENT_ID = "01"

os.environ['STUDENT_ID'] = '2021001234'
os.environ['STUDENT_NAME'] = '张三'
os.environ['STUDENT_NICKNAME'] = '代码小能手'


def compute_file_md5(filepath):
    md5_hash = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except FileNotFoundError:
        return None


def get_student_info():
    student_id = os.environ.get('STUDENT_ID', '').strip()
    student_name = os.environ.get('STUDENT_NAME', '').strip()
    student_nickname = os.environ.get('STUDENT_NICKNAME', '').strip()

    missing = []
    if not student_id:
        missing.append('STUDENT_ID')
    if not student_name:
        missing.append('STUDENT_NAME')
    if not student_nickname:
        missing.append('STUDENT_NICKNAME')

    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    return {
        'student_id': student_id,
        'name': student_name,
        'nickname': student_nickname
    }


if __name__ == "__main__":
    student_info = get_student_info()

    forbidden = ['sklearn', 'scikit_learn', 'tensorflow', 'torch', 'pytorch',
                 'keras', 'xgboost', 'lightgbm', 'catboost', 'statsmodels']

    imported = list(sys.modules.keys())
    violations = []
    for module in imported:
        module_lower = module.lower()
        for pkg in forbidden:
            if module_lower == pkg or module_lower.startswith(pkg + '.'):
                violations.append(module)
                break

    if violations:
        print(f"Error: Forbidden libraries detected: {violations}")
        sys.exit(1)

    try:
        from solution import Solution
    except ImportError as e:
        print(f"Error: Failed to import Solution - {e}")
        sys.exit(1)

    try:
        solution = Solution()
    except Exception as e:
        print(f"Error: Failed to initialize Solution - {e}")
        sys.exit(1)

    try:
        test_df = pd.read_csv(test_csv_path)
    except FileNotFoundError:
        print(f"Error: test.csv not found at {test_csv_path}")
        sys.exit(1)

    original_size = len(test_df)
    np.random.seed(2025)
    sample_indices = np.random.choice(original_size, min(10000, original_size), replace=False)
    test_df = test_df.iloc[sample_indices].reset_index(drop=True)
    y_true = test_df['age'].values

    test_features = test_df.drop('age', axis=1)
    samples = [(idx, row.to_dict()) for idx, row in test_features.iterrows()]
    predictions = [None] * len(samples)


    def process_sample(sample_info):
        idx, sample = sample_info
        result = solution.forward(sample)
        return idx, result['prediction']


    print(f"Evaluating {len(samples)} samples...")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_sample, samples)
        for idx, pred in results:
            predictions[idx] = pred
    prediction_time = time.time() - start_time

    predictions = np.array(predictions)
    mae = np.mean(np.abs(y_true - predictions))
    mse = np.mean((y_true - predictions) ** 2)
    rmse = np.sqrt(mse)

    print(f"\n{'=' * 50}")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Time: {prediction_time:.2f}s")
    print(f"{'=' * 50}\n")

    payload = {
        'student_info': student_info,
        'assignment_id': ASSIGNMENT_ID,
        'metrics': {
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'Prediction_Time': float(prediction_time)
        },
        'checksums': {
            'evaluate.py': "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        }
    }

    print("Submitting to leaderboard...")
    try:
        response = requests.post(
            LEADERBOARD_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print("✓ Submission successful")
            if 'rank' in result:
                print(f"  Rank: {result['rank']}")
            if 'message' in result:
                print(f"  {result['message']}")
        else:
            print(f"✗ Submission failed: HTTP {response.status_code}")

    except requests.exceptions.Timeout:
        print("✗ Submission failed: Request timeout")
    except requests.exceptions.ConnectionError:
        print("✗ Submission failed: Cannot connect to server")
    except Exception as e:
        print(f"✗ Submission failed: {str(e)}")