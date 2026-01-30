# Fianlthesis2009 (Final Thesis 2026)

**Project:** Efficient Network Intrusion Detection using Mamba Architectures
**Date:** January 30, 2026

## 📊 Experiment Results (Teacher vs. Student)

This project compares a robust, bidirectional **Teacher** model against a lightweight, unidirectional **Student** model designed for high-speed, real-time intrusion detection.

### 1. Teacher Model (Bidirectional Mamba)
*The "Expert" model. Uses future context to achieve maximum accuracy.*

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Unsupervised Pretraining** | **AUC: 97.02%** | Phase 1 (10 Epochs, Self-Supervised) |
| **Fine-Tuning Accuracy** | **98.90%** | Phase 2 (Fine-tuning on mixed data) |
| **Fine-Tuning F1 Score** | **81.90%** | Higher recall for minority attacks |
| **Fine-Tuning AUC** | **98.44%** | Excellent separation capability |
| **Latency** | **~0.85 ms** | Standard GPU Batch Inference |

---

### 2. Student Model (Unidirectional Mamba)
*The "Real-Time" model. Constrained to forward-only processing for deployment efficiency.*

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Unsupervised Pretraining** | **AUC: 94.35%** | Phase 3 (1 Epoch, Fast Pretraining) |
| **Fine-Tuning Accuracy** | **98.54%** | **Replicated (~99.6% of Teacher Perf)** |
| **Fine-Tuning F1 Score** | **74.88%** | Slightly lower due to unidirectional constraint |
| **Fine-Tuning AUC** | **96.89%** | Strong anomaly detection capability |
| **Latency** | **0.16 ms** | **~5.3x Faster** than Teacher (Batched) |

### 🏆 Conclusion
The Student model successfully distills the knowledge of the Teacher, achieving **98.54% Accuracy** (comparable to Teacher) while offering massive latency reductions (**0.16 ms/flow**). This validates the Unidirectional Mamba architecture for high-speed NIDS deployments.

### 3. Knowledge Distillation (Learned from Scratch)
*Training a Randomly Initialized Student using Teacher Knowledge (No Pretraining).*

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Accuracy** | **98.46%** | Matches Pretrained Student (-0.08%) |
| **F1 Score** | **72.86%** | Slight drop (-2%) vs Pretrained |
| **Conclusion** | **Success** | Teacher Knowledge (Soft Targets) is sufficient to train a Student from scratch without Unsupervised Pretraining. |

---

## 📂 Repository Structure
*   **`final_phase1/`**: Teacher Model (Bi-Mamba) Code & Config.
*   **`final_phase2/`**: Fine-Tuning Logic (Few-Shot).
*   **`final_student/`**: Student Model (Uni-Mamba) Code & Evaluation.
*   **`scripts/`**: Data processing utilities.
*   **`COMMIT_LOG.md`**: Manual history of project changes.

*(Data folder `../data` is excluded from git to save space. See `.gitignore`)*
