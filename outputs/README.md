# 🫀 Digital Health Twin

**AI-Powered Digital Health Twin for Predictive Healthcare with Mental and Physical Health Integration**

A full-stack AI system that combines Machine Learning, Retrieval-Augmented Generation (RAG), and explainable insight generation for cardiovascular risk prediction — integrating both physical biomarkers and mental health indicators.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    🖥️ Presentation Layer                    │
│          Web Dashboard (HTML/CSS/JS) + FastAPI Docs        │
├────────────────────────────────────────────────────────────┤
│                    🧠 Insight Layer                         │
│   ML Prediction + RAG Context → Explainable Health Insight │
├───────────────────────┬────────────────────────────────────┤
│  📊 ML Prediction     │      🔍 RAG Pipeline               │
│  • Logistic Regression│  • sentence-transformers embeddings│
│  • Random Forest      │  • ChromaDB vector store           │
│  • XGBoost            │  • Semantic retrieval (top-k)      │
│  • PyTorch NN         │  • Flan-T5 / template generation   │
│  • SHAP explainability│                                    │
├───────────────────────┴────────────────────────────────────┤
│                    💾 Data Layer                            │
│  UCI Heart Disease (303 samples) + Synthetic Mental Health │
│  18 features: 13 cardiac + 5 mental/lifestyle              │
└────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
digital-health-twin/
├── api/
│   └── main.py              # FastAPI backend (endpoints)
├── data/
│   ├── prepare_data.py       # Dataset download + augmentation
│   └── heart.csv             # Processed dataset
├── models/
│   ├── train.py              # ML training pipeline
│   ├── scaler.pkl            # Feature scaler
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   ├── pytorch_nn.pth        # Neural network weights
│   └── metrics.json          # Evaluation results
├── rag/
│   ├── embedder.py           # Patient → embeddings → ChromaDB
│   ├── retriever.py          # Semantic retrieval + response gen
│   └── chroma_db/            # Persistent vector database
├── insight/
│   └── engine.py             # ML + RAG → explainable insights
├── ui/
│   ├── index.html            # Dashboard frontend
│   ├── style.css             # Premium dark theme
│   └── script.js             # Frontend logic
├── research/
│   └── research_draft.md     # 3-5 page research paper
├── outputs/
│   └── example_outputs.md    # Sample API responses
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/shashiawari/digital-health-twin.git
cd digital-health-twin

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Setup (Run in order)

```bash
# Step 1: Prepare dataset (downloads UCI data + augments with mental health features)
python data/prepare_data.py

# Step 2: Train ML models
python models/train.py

# Step 3: Build RAG vector store
python rag/embedder.py

# Step 4: Start the API server
python api/main.py
```

The API will be available at `http://localhost:8000`

- **Swagger Docs:** http://localhost:8000/docs
- **Demo UI:** http://localhost:8000/ui/index.html

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `GET` | `/model-info` | Model metrics |
| `POST` | `/predict` | ML risk prediction |
| `POST` | `/ask` | RAG-based Q&A |
| `POST` | `/insight` | Combined ML+RAG insight |

### Example: POST /predict

```json
{
    "age": 58, "sex": 1, "cp": 2, "trestbps": 145,
    "chol": 260, "fbs": 1, "restecg": 0, "thalach": 130,
    "exang": 1, "oldpeak": 2.5, "slope": 1, "ca": 2, "thal": 2,
    "stress_level": 8.0, "sleep_quality": 3.5,
    "activity_hours": 1.0, "anxiety_score": 15, "bmi": 32.5
}
```

### Example: POST /ask

```json
{
    "query": "What health risks does a 55-year-old male with high cholesterol have?",
    "top_k": 5
}
```

## 🧪 Methodology

### Data Pipeline
1. **Source:** UCI Heart Disease dataset (Cleveland, 303 samples)
2. **Augmentation:** 5 synthetic mental health features added with controlled correlation to target
3. **Patient Notes:** Natural-language summaries generated for each patient record

### ML Models
- **Baseline:** Logistic Regression (L2), Random Forest (200 trees)
- **Advanced:** XGBoost (300 boosters), PyTorch NN (128→64→32→1)
- **Explainability:** SHAP TreeExplainer for feature importance
- **Preprocessing:** StandardScaler, stratified 80/20 split

### RAG Pipeline
- **Embedding:** `all-MiniLM-L6-v2` (384-dim sentence embeddings)
- **Vector DB:** ChromaDB with cosine similarity
- **Generation:** Flan-T5-base or rule-based template fallback
- **Query Types:** Risk assessment, patient similarity, mental health analysis

### Insight Engine
The core differentiator — combines:
1. ML prediction → risk probability + contributing factors
2. RAG retrieval → similar patient profiles
3. Clinical rules → threshold-based risk factor identification
4. Recommendations → actionable health suggestions

## 📊 System Design

### Design Principles
1. **Modularity:** Each component (ML, RAG, Insight, API) is independently testable
2. **Explainability:** Every prediction includes risk factors and recommendations
3. **Local-first:** No external API keys required — runs entirely locally
4. **Integration:** Mental + physical health considered together

### Data Flow
```
Patient Data → ML Model → Risk Prediction ─┐
                                             ├→ Insight Engine → Explainable Output
Patient Query → Embedder → ChromaDB → RAG ─┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Models | scikit-learn, XGBoost, PyTorch |
| Embeddings | sentence-transformers (MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| LLM | Google Flan-T5-base / Template engine |
| API | FastAPI + Uvicorn |
| Explainability | SHAP |
| Frontend | HTML5 + CSS3 + JavaScript |
| Data | pandas, numpy |

## 📄 Research Paper

See [`research/research_draft.md`](research/research_draft.md) for the full 3–5 page research draft covering:
- Abstract & Problem Definition
- Related Work
- Proposed System Architecture
- Experimental Design & Expected Results
- Limitations

## 📜 License

This project is for educational and research purposes.

---

*Built as part of the HOPn internship evaluation task — Research + Implementation*
