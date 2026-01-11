# Customer Churn Prediction Project

A machine learning project to predict customer churn using exploratory data analysis and predictive modeling.

## Project Structure

```
Project/
├── .python-version          # Python version specification
├── main.py                  # Main application
├── pyproject.toml           # Project dependencies and metadata
├── README.md                # This file
├── uv.lock                  # Locked dependency versions
├── Dockerfile               # Container definition for API deployment
├── docker-compose.yml       # Local development and deployment setup
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI ML service
├── Data/
│   ├── archive.zip
│   ├── customer_churn_dataset-testing-master.csv
│   └── customer_churn_dataset-training-master.csv
├── Scripts/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── export_best_artifacts.py
│   └── deploy.sh            # Deployment script
├── Notebooks/
│   ├── 1_EDA.ipynb          # Exploratory Data Analysis
│   ├── 2_Data_Prep.ipynb    # Data Preprocessing
│   └── 3_Model_Training.ipynb # Model Training
└── mlruns/                  # MLflow experiment tracking
    └── best_model_artifacts/ # Exported model artifacts
```

## Prerequisites

- Python 3.13+ (see `.python-version`)
- Git (optional)
- `uv` package manager (recommended)
- Visual Studio Code with Python and Jupyter extensions
- Docker (required for deployment)
- Docker Compose (optional, for development)

## Installation & Setup

### Step 1: Install `uv` (Recommended)

**On macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative: Install via pip / pipx:**

```bash
pipx install uv
```

### Step 2: Clone or Navigate to Project

```bash
cd /Users/udaranilupul/Documents/Freelancing/CodeWave/CodeWaveMl101/Project
```

### Step 3: Create Virtual Environment with `uv`

```bash
uv venv
```

**Activate the virtual environment:**

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**

```cmd
.\.venv\Scripts\activate.bat
```

### Step 4: Install Dependencies with `uv`

```bash
uv sync
```

This command reads `pyproject.toml` and `uv.lock` to install all required libraries with exact versions.

## Setting Up Jupyter in VS Code

### Step 1: Install VS Code Extensions

Open VS Code and install these extensions:

1. **Python** — by Microsoft
2. **Jupyter** — by Microsoft
3. **Jupyter Keymap** — by Microsoft (optional)

Search for them in the Extensions sidebar (Ctrl+Shift+X / Cmd+Shift+X).

### Step 2: Select Jupyter Kernel in VS Code

1. Open a `.ipynb` notebook file (e.g., `Notebooks/1_EDA.ipynb`)
2. In the top-right corner, click **Select Kernel**
3. Choose **Python Environments**
4. Select the environment path ending with `.venv` (or search for it)
   - Path will look like: `/Users/udaranilupul/Documents/Freelancing/CodeWave/CodeWaveMl101/Project/.venv/bin/python`
5. VS Code will use this kernel for all notebook cells

### Step 3: Verify Kernel is Connected

- Open `Notebooks/1_EDA.ipynb`
- Look at the top-right corner — you should see the kernel name (e.g., `Python 3.13.x ('./venv')`)
- Run a test cell:

```python
import sys
print(sys.executable)
```

Should output the path to your `.venv` Python executable.

## Running the Project

### Run Jupyter Notebooks in VS Code

1. Open `Notebooks/1_EDA.ipynb`
2. Confirm the correct kernel is selected (top-right)
3. Click **Run All** or run individual cells with Shift+Enter
4. View outputs directly in the editor

### Run Main Application from Terminal

**In VS Code, open the integrated terminal (Ctrl+` / Cmd+`):**

```bash
source .venv/bin/activate  # macOS/Linux
python main.py
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

## Running the Inference API

This starts an API server that loads the exported **best model** from `mlruns/best_model_artifacts/` and uses `Data/preprocessor.pkl` to transform raw inputs.

### 1) Ensure dependencies are installed

```bash
cd /Users/udaranilupul/Documents/Freelancing/CodeWave/CodeWaveMl101/Project
uv sync
```

### 2) Export the best model artifacts (once, or after retraining)

```bash
/opt/homebrew/bin/python3 Scripts/export_best_artifacts.py --experiment-name customer_churn_optimization --metric f1
```

### 3) Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 4) Call the API

- Health: `GET http://127.0.0.1:8000/health`
- Input schema: `GET http://127.0.0.1:8000/schema`
- Predict: `POST http://127.0.0.1:8000/predict`

Example request body:

```json
{
    "features": {
        "Gender": "Male",
        "Contract Length": "Monthly",
        "Subscription Type": "Basic",
        "Age": 35,
        "Tenure": 12
    }
}
```

Notes:

- The API expects **raw** fields that match the columns the preprocessor was fit on. Use `/schema` to see `required_columns`.
- Override paths if needed:
    - `BEST_MODEL_DIR` (folder containing `model.pkl`)
    - `BEST_MODEL_PATH` (direct path to `model.pkl`)
    - `PREPROCESSOR_PATH` (path to `preprocessor.pkl`)

## Dependencies

Key libraries used in this project:

- **pandas** — Data manipulation and analysis
- **numpy** — Numerical computing
- **scikit-learn** — Machine learning models and preprocessing
- **matplotlib** — Data visualization
- **seaborn** — Statistical data visualization
- **ipykernel** — IPython kernel for Jupyter notebooks

All dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

## Dataset

The project uses two customer churn datasets:

- `customer_churn_dataset-training-master.csv` — Training data
- `customer_churn_dataset-testing-master.csv` — Testing data

Located in the `Data/` folder.

## Workflow

1. **Activate the virtual environment** (Step 3 above)
2. **Select kernel in VS Code** (Step 3 of Jupyter setup)
3. **Explore the data** using `Notebooks/1_EDA.ipynb` directly in VS Code
4. **Run the main application** with `python main.py`
5. **Train and evaluate models** for customer churn prediction

## 🚀 CI/CD Pipeline

This project includes automated CI/CD pipelines for model training and deployment.

### CI Pipeline (Model Training)
Automatically triggered on pushes to `main` or `week-5-ci-cd-final` branches:
- Trains ML models using `Scripts/model_training.py`
- Logs experiments and metrics to MLflow
- Stores model artifacts for deployment

### CD Pipeline (Model Deployment)

#### FastAPI Service
The project includes a production-ready FastAPI service for serving ML predictions:
- **API Endpoints:**
  - `/health` - Service health check
  - `/predict` - Make ML predictions
  - `/schema` - Get model input schema
  - `/docs` - Interactive API documentation

#### Deployment Options

**Option 1: Manual Deployment (Recommended for development)**
```bash
# Deploy locally
./Scripts/deploy.sh

# Deploy to staging
./Scripts/deploy.sh -e staging -p 8001

# Deploy to production
./Scripts/deploy.sh -e production -p 80
```

**Option 2: GitHub Actions (Automated)**
1. Go to repository → Actions tab
2. Find "Deploy Model API" workflow
3. Click "Run workflow" → Choose environment → Deploy!

**Option 3: Docker Compose (Development)**
```bash
# Start API service
docker-compose up customer-churn-api

# Start with MLflow server
docker-compose --profile mlflow up
```

#### Deployment Environments
- **Local**: `http://localhost:8000`
- **Staging**: `http://localhost:8001`
- **Production**: Configurable port (default: 80)

#### Features
- ✅ Docker containerization for consistency
- ✅ Multi-environment support (local/staging/production)
- ✅ Automated testing and health checks
- ✅ Model artifact validation
- ✅ Zero-downtime deployments
- ✅ Automatic container restarts

#### Testing the API
```bash
# Health check
curl http://localhost:8000/health

# Get model schema
curl http://localhost:8000/schema

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 35,
      "gender": "Male",
      "tenure": 12,
      "usage_frequency": 15,
      "support_calls": 2,
      "payment_delay": 0,
      "subscription_type": "Standard",
      "contract_length": "Monthly",
      "total_spend": 500,
      "last_interaction": 30
    }
  }'
```

#### Prerequisites for Deployment
- Docker installed and running
- Model artifacts available (run CI pipeline first)
- Ports 8000-8001 available for local/staging deployments

## Troubleshooting

**Kernel not found in VS Code:**

- Ensure the virtual environment is activated in terminal:

```bash
source .venv/bin/activate
```

- Then reload VS Code (Cmd+Shift+P / Ctrl+Shift+P → "Developer: Reload Window")

**Virtual environment not found:**

```bash
uv venv
uv sync
```

**Permission denied on macOS / Linux:**

```bash
chmod +x .venv/bin/activate
source .venv/bin/activate
```

**Kernel still shows old Python version:**

- Click **Select Kernel** again and choose the `.venv` path explicitly
- Restart VS Code



