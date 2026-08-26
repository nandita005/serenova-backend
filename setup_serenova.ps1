# ============================================================
# SERENOVA - AI SYSTEM BACKEND STRUCTURE
# ============================================================

$root = Get-Location

Write-Host ""
Write-Host "Creating Serenova AI System..." -ForegroundColor Cyan
Write-Host "Root: $root"
Write-Host ""

# ------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------

$directories = @(

    # =========================
    # BACKEND
    # =========================
    "backend",
    "backend\app",
    "backend\app\api",
    "backend\app\websocket",
    "backend\app\schemas",
    "backend\app\services",
    "backend\app\database",
    "backend\app\middleware",
    "backend\tests",

    # =========================
    # SENSOR SIMULATOR
    # =========================
    "simulator",
    "simulator\sensors",
    "simulator\scenarios",

    # =========================
    # DATA
    # =========================
    "data",
    "data\raw",
    "data\raw\sensors",
    "data\raw\external",
    "data\processed",
    "data\features",
    "data\training",
    "data\validation",
    "data\scenarios",

    # =========================
    # SIGNAL PROCESSING
    # =========================
    "processing",
    "processing\ppg",
    "processing\ecg",
    "processing\imu",
    "processing\temperature",
    "processing\piezo",
    "processing\feature_engine",

    # =========================
    # MODEL LAYER
    # =========================
    "models",
    "models\base",
    "models\classical",
    "models\classical\xgboost",
    "models\classical\random_forest",
    "models\deep",
    "models\deep\lstm",
    "models\deep\transformer",
    "models\deep\patchtst",
    "models\generative",
    "models\generative\timegan",
    "models\representation",
    "models\representation\contrastive",
    "models\checkpoints",

    # =========================
    # TRAINING
    # =========================
    "training",
    "training\configs",
    "training\trainers",
    "training\datasets",

    # =========================
    # EVALUATION
    # =========================
    "evaluation",
    "evaluation\metrics",
    "evaluation\explainability",
    "evaluation\uncertainty",
    "evaluation\reports",

    # =========================
    # 17 PREGNANCY CONDITIONS
    # =========================
    "conditions",
    "conditions\pregnancy",
    "conditions\postpartum",

    # =========================
    # MULTI-AGENT SYSTEM
    # =========================
    "agents",
    "agents\state",
    "agents\vsa",
    "agents\cfa",
    "agents\nla",
    "agents\rfa",
    "agents\prompts",

    # =========================
    # ORCHESTRATION
    # =========================
    "orchestration",

    # =========================
    # MEMORY
    # =========================
    "memory",
    "memory\short_term",
    "memory\patient",
    "memory\episodic",

    # =========================
    # TOOLS
    # =========================
    "tools",
    "tools\sensor_tools",
    "tools\clinical_tools",
    "tools\analysis_tools",
    "tools\reporting",

    # =========================
    # RAG / SA-RAG
    # =========================
    "rag",
    "rag\documents",
    "rag\ingestion",
    "rag\embeddings",
    "rag\vectorstore",
    "rag\retrieval",
    "rag\safety",

    # =========================
    # SLM / LLM
    # =========================
    "llm",
    "llm\models",
    "llm\datasets",
    "llm\datasets\instruction_data",
    "llm\finetuning",
    "llm\prompts",
    "llm\inference",

    # =========================
    # SAFETY
    # =========================
    "safety",

    # =========================
    # CONFIG
    # =========================
    "config",

    # =========================
    # SCRIPTS
    # =========================
    "scripts",

    # =========================
    # TESTING
    # =========================
    "tests",
    "tests\unit",
    "tests\integration",
    "tests\models",
    "tests\agents",
    "tests\end_to_end",

    # =========================
    # EXPERIMENTS
    # =========================
    "notebooks",
    "notebooks\signal_analysis",
    "notebooks\model_experiments",
    "notebooks\evaluation",

    # =========================
    # DOCUMENTATION
    # =========================
    "docs",
    "docs\architecture",
    "docs\datasets",
    "docs\models",
    "docs\experiments",

    # =========================
    # FRONTEND PLACEHOLDER
    # =========================
    # We keep frontend separate from backend.
    "frontend"
)

# Create directories
foreach ($dir in $directories) {
    $path = Join-Path $root $dir

    if (!(Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

# ------------------------------------------------------------
# FILES
# ------------------------------------------------------------

$files = @(

    # ========================================================
    # BACKEND
    # ========================================================

    "backend\app\main.py",
    "backend\app\config.py",

    "backend\app\api\health.py",
    "backend\app\api\sensors.py",
    "backend\app\api\patients.py",
    "backend\app\api\vitals.py",
    "backend\app\api\risk.py",
    "backend\app\api\alerts.py",
    "backend\app\api\agents.py",
    "backend\app\api\doctor.py",
    "backend\app\api\postpartum.py",

    "backend\app\websocket\sensor_stream.py",

    "backend\app\schemas\sensor.py",
    "backend\app\schemas\patient.py",
    "backend\app\schemas\vitals.py",
    "backend\app\schemas\risk.py",
    "backend\app\schemas\agent.py",
    "backend\app\schemas\alert.py",

    "backend\app\services\ingestion_service.py",
    "backend\app\services\patient_service.py",
    "backend\app\services\risk_service.py",
    "backend\app\services\alert_service.py",

    "backend\app\database\connection.py",
    "backend\app\database\models.py",
    "backend\app\database\repository.py",

    "backend\app\middleware\error_handler.py",

    "backend\requirements.txt",

    # ========================================================
    # SENSOR SIMULATOR
    # ========================================================

    "simulator\sensors\max30102.py",
    "simulator\sensors\ad8232.py",
    "simulator\sensors\mpu6050.py",
    "simulator\sensors\ds18b20.py",
    "simulator\sensors\piezo.py",
    "simulator\sensors\sw420.py",

    "simulator\scenarios\normal_pregnancy.py",
    "simulator\scenarios\deteriorating_pregnancy.py",
    "simulator\scenarios\postpartum.py",

    "simulator\payload_builder.py",
    "simulator\websocket_client.py",
    "simulator\run_simulator.py",

    # ========================================================
    # PROCESSING
    # ========================================================

    "processing\ppg\processor.py",
    "processing\ecg\processor.py",
    "processing\imu\processor.py",
    "processing\temperature\processor.py",
    "processing\piezo\processor.py",

    "processing\feature_engine\maternal_features.py",
    "processing\feature_engine\fetal_features.py",
    "processing\feature_engine\temporal_features.py",

    "processing\pipeline.py",

    # ========================================================
    # MODELS
    # ========================================================

    "models\base\base_model.py",

    "models\classical\xgboost\model.py",
    "models\classical\random_forest\model.py",

    "models\deep\lstm\model.py",
    "models\deep\transformer\model.py",
    "models\deep\patchtst\model.py",

    "models\generative\timegan\model.py",

    "models\representation\contrastive\model.py",

    "models\registry.py",

    # ========================================================
    # TRAINING
    # ========================================================

    "training\train.py",

    "training\datasets\dataset.py",
    "training\datasets\dataloader.py",

    "training\trainers\base_trainer.py",
    "training\trainers\lstm_trainer.py",
    "training\trainers\transformer_trainer.py",
    "training\trainers\patchtst_trainer.py",

    "training\configs\lstm.yaml",
    "training\configs\transformer.yaml",
    "training\configs\patchtst.yaml",

    # ========================================================
    # EVALUATION
    # ========================================================

    "evaluation\evaluate.py",

    "evaluation\metrics\classification.py",
    "evaluation\metrics\forecasting.py",
    "evaluation\metrics\calibration.py",

    "evaluation\explainability\shap_explainer.py",
    "evaluation\uncertainty\mc_dropout.py",

    # ========================================================
    # CONDITIONS
    # ========================================================

    "conditions\registry.py",

    # ========================================================
    # AGENTS
    # ========================================================

    "agents\state\state.py",
    "agents\state\schemas.py",

    "agents\vsa\agent.py",
    "agents\cfa\agent.py",
    "agents\nla\agent.py",
    "agents\rfa\agent.py",

    "agents\prompts\vsa.txt",
    "agents\prompts\cfa.txt",
    "agents\prompts\nla.txt",
    "agents\prompts\rfa.txt",

    # ========================================================
    # ORCHESTRATION
    # ========================================================

    "orchestration\graph.py",
    "orchestration\router.py",
    "orchestration\workflow.py",
    "orchestration\policies.py",

    # ========================================================
    # MEMORY
    # ========================================================

    "memory\short_term\working_memory.py",
    "memory\patient\patient_memory.py",
    "memory\episodic\event_memory.py",
    "memory\manager.py",

    # ========================================================
    # TOOLS
    # ========================================================

    "tools\sensor_tools\sensor_reader.py",

    "tools\clinical_tools\risk_lookup.py",
    "tools\clinical_tools\guideline_lookup.py",

    "tools\analysis_tools\trend_analysis.py",
    "tools\analysis_tools\risk_calculator.py",

    "tools\reporting\pdf_generator.py",

    "tools\tool_registry.py",

    # ========================================================
    # RAG
    # ========================================================

    "rag\ingestion\loader.py",
    "rag\ingestion\chunker.py",

    "rag\embeddings\embedder.py",

    "rag\vectorstore\chroma_store.py",

    "rag\retrieval\retriever.py",

    "rag\safety\safety_filter.py",
    "rag\safety\contraindication.py",

    # ========================================================
    # LLM / SLM
    # ========================================================

    "llm\models\local_model.py",

    "llm\finetuning\lora_config.py",
    "llm\finetuning\train.py",

    "llm\inference\generator.py",

    # ========================================================
    # SAFETY
    # ========================================================

    "safety\clinical_rules.py",
    "safety\risk_thresholds.py",
    "safety\output_guard.py",
    "safety\escalation.py",

    # ========================================================
    # CONFIG
    # ========================================================

    "config\settings.yaml",
    "config\model_config.yaml",
    "config\condition_config.yaml",
    "config\agent_config.yaml",

    # ========================================================
    # SCRIPTS
    # ========================================================

    "scripts\run_backend.ps1",
    "scripts\run_simulator.ps1",
    "scripts\run_pipeline.ps1",

    # ========================================================
    # ROOT
    # ========================================================

    ".env",
    ".gitignore",
    "README.md"
)

# Create files
foreach ($file in $files) {
    $path = Join-Path $root $file

    if (!(Test-Path $path)) {
        New-Item -ItemType File -Path $path -Force | Out-Null
    }
}

# ------------------------------------------------------------
# GITIGNORE
# ------------------------------------------------------------

$gitignore = @"
.venv/
__pycache__/
*.pyc
.env

data/raw/
data/processed/
data/features/
data/training/

models/checkpoints/

*.log
.vscode/
.idea/
"@

Set-Content -Path ".gitignore" -Value $gitignore

# ------------------------------------------------------------
# README
# ------------------------------------------------------------

$readme = @"
# SERENOVA

AI-Orchestrated Pregnancy Monitoring and Decision Support System.

Architecture:

Sensors
    ->
Sensor Simulator / Hardware
    ->
FastAPI WebSocket
    ->
Signal Processing
    ->
Feature Engineering
    ->
ML/DL Models
    ->
Condition Detection
    ->
Multi-Agent System
    ->
Orchestration
    ->
Memory + Tools + SA-RAG
    ->
SLM/LLM
    ->
Safety Layer
    ->
Doctor / Patient Frontend

Major modules:

backend/
simulator/
processing/
models/
training/
evaluation/
conditions/
agents/
orchestration/
memory/
tools/
rag/
llm/
safety/
frontend/

Pregnancy mode and postpartum mode are both first-class system modes.
"@

Set-Content -Path "README.md" -Value $readme

# ------------------------------------------------------------
# BASIC PYTHON PACKAGE FILES
# ------------------------------------------------------------

$pythonDirs = @(
    "backend\app",
    "backend\app\api",
    "backend\app\websocket",
    "backend\app\schemas",
    "backend\app\services",
    "backend\app\database",
    "backend\app\middleware",
    "processing",
    "processing\ppg",
    "processing\ecg",
    "processing\imu",
    "processing\temperature",
    "processing\piezo",
    "processing\feature_engine",
    "models",
    "models\base",
    "training",
    "training\datasets",
    "training\trainers",
    "evaluation",
    "evaluation\metrics",
    "evaluation\explainability",
    "evaluation\uncertainty",
    "conditions",
    "conditions\pregnancy",
    "conditions\postpartum",
    "agents",
    "agents\state",
    "agents\vsa",
    "agents\cfa",
    "agents\nla",
    "agents\rfa",
    "orchestration",
    "memory",
    "memory\short_term",
    "memory\patient",
    "memory\episodic",
    "tools",
    "tools\sensor_tools",
    "tools\clinical_tools",
    "tools\analysis_tools",
    "tools\reporting",
    "rag",
    "rag\ingestion",
    "rag\embeddings",
    "rag\vectorstore",
    "rag\retrieval",
    "rag\safety",
    "llm",
    "llm\models",
    "llm\finetuning",
    "llm\inference",
    "safety"
)

foreach ($dir in $pythonDirs) {
    $init = Join-Path $root "$dir\__init__.py"

    if (!(Test-Path $init)) {
        New-Item -ItemType File -Path $init -Force | Out-Null
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " SERENOVA STRUCTURE CREATED SUCCESSFULLY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

tree /F

Write-Host ""
Write-Host "Next step:" -ForegroundColor Yellow
Write-Host "Create the Python virtual environment:"
Write-Host ""
Write-Host "    python -m venv .venv"
Write-Host ""
Write-Host "Then activate it:"
Write-Host ""
Write-Host "    .venv\Scripts\Activate.ps1"
Write-Host ""