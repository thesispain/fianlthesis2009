# 📜 Project Commit Log & History

**Purpose:** This file tracks every major update ("Commit") to the repository, serving as a manual git history for the project.

| Date | Time | Author | Component | Description |
| :--- | :--- | :--- | :--- | :--- |
| **2026-01-30** | 14:48 | Agent | **Docs** | Initial creation of `project_history_log.md` with full system architecture. |
| **2026-01-30** | 14:55 | Agent | **Docs** | Detailed breakdown of Phase 1, 2, and 3 file locations provided in logs. |
| **2026-01-30** | 15:00 | Agent | **Git** | Established `.gitignore` to exclude 131GB data and temp files. |
| **2026-01-30** | 15:00 | Agent | **Core** | Finalized Phase 3 (Student) Codebase: `final_student/` with pretraining and early exit logic. |
| **2026-01-30** | 15:10 | Agent | **Refactor** | Created `Organized_Final/` directory containing only essential code for Git Push. |
| **2026-01-30** | 15:12 | Agent | **Git** | Added `telegram_config.json` to `.gitignore` to prevent secret leakage. |
| **2026-01-30** | 16:30 | Agent | **Exp** | Phase 4: Distilled Student (Acc: 98.46%, F1: 72.86%) completed. |
| **2026-01-30** | 17:00 | Agent | **Feat** | Phase 5: Implemented 3 Early Exit Strategies types (Baseline, Dynamic, Learned) in `final_student/models/`. |
| **2026-01-30** | 17:15 | Agent | **Exp** | Benchmarking Early Exit mechanisms (Baseline vs Dynamic vs Learned). |

---

## 📦 Repository Structure (What is included)

### 1. `final_phase1/` (Teacher Model)
*   `model.py`: Bi-Mamba Architecture.
*   `run_phase1.py`: Pretraining Script.
*   `config.py`: Configuration.

### 2. `final_phase2/` (Fine-Tuning)
*   `run_fewshot.py`: Fine-tuning logic for Few-Shot Learning.

### 3. `final_student/` (Student Model)
*   `model.py`: Unidirectional Mamba ("Student").
*   `evaluate_anomaly.py`: Unsupervised Anomaly Detection.
*   `run_phase1.py`: Student Pretraining.

### 4. `scripts/`
*   Helper scripts for data processing and visualization.

### 5. `logs/` & Root Docs
*   `project_history_log.md`: Detailed experiment history.
*   `thesis_results.md`: Final metric tables.
