<div align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-1.7%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" />
<img src="https://img.shields.io/badge/Status-Production%20Ready-6366F1?style=for-the-badge" />

<br /><br />

# ⚖️ Legal Clause Classifier

### AI-powered contract clause detection — identify 41 legal clause types instantly

[**Live Demo**](#) · [**Report Bug**](../../issues) · [**Request Feature**](../../issues)

<br />

<img src="https://raw.githubusercontent.com/your-username/legal-clause-classifier/main/assets/demo.gif" alt="App Demo" width="720" />

</div>

---

##  Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Supported Clause Types](#-supported-clause-types)
- [Performance](#-performance)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

##  Overview

**Legal Clause Classifier** is a production-ready NLP web application that automatically identifies the type of any legal contract clause using machine learning. Built on the [CUAD (Contract Understanding Atticus Dataset)](https://www.atticusprojectai.org/cuad), it leverages a **TF-IDF + ComplementNB** pipeline to classify 41 distinct clause categories with high confidence — in milliseconds.

Designed for legal tech teams, contract analysts, and compliance officers who need fast, accurate clause tagging without manual review.

>  **Why ComplementNB?** It outperforms standard Multinomial NB on imbalanced text datasets, making it ideal for legal corpora where some clause types appear far less frequently than others.

---

##  Features

| Feature | Description |
|---|---|
|  **41 Clause Types** | Full CUAD taxonomy coverage — from Governing Law to IP Ownership |
|  **Instant Classification** | Sub-second inference with cached model loading |
|  **Top-5 Predictions** | Confidence scores for top 5 candidate clause types |
|  **8 Built-in Examples** | One-click sample clauses to explore the model |
|  **Glassmorphism UI** | Dark-mode, responsive interface with gradient accents |
|  **Fully Offline** | No API calls — all inference runs locally |

---

##  Tech Stack

```
Frontend     │  Streamlit 1.35+
ML Pipeline  │  scikit-learn (TF-IDF Vectorizer + ComplementNB)
Language     │  Python 3.9+
Dataset      │  CUAD — 510 contracts, 13,000+ annotated clauses
Deployment   │  Streamlit Cloud / Docker / any WSGI host
```

---

##  Project Structure

```
legal-clause-classifier/
│
├── app.py                  # Main Streamlit application
│
├── models/
│   ├── best_model.pkl      # Trained ComplementNB classifier
│   ├── label_encoder.pkl   # LabelEncoder for 41 clause classes
│   ├── tfidf_word.pkl      # Word-level TF-IDF vectorizer (15k vocab)
│   └── tfidf_char.pkl      # Char-level TF-IDF vectorizer (5k vocab)
│
├── assets/
│   └── demo.gif            # Demo animation for README
│
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

##  Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/legal-clause-classifier.git
cd legal-clause-classifier
```

**2. Create and activate a virtual environment** *(recommended)*

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Place model files**

Ensure all four `.pkl` files are inside the `models/` directory:

```
models/best_model.pkl
models/label_encoder.pkl
models/tfidf_word.pkl
models/tfidf_char.pkl
```

**5. Run the app**

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

##  Requirements

```txt
streamlit>=1.35.0
scikit-learn>=1.7.0
joblib>=1.3.0
numpy>=1.24.0
scipy>=1.11.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

##  Usage

### Basic Workflow

1. **Paste** a contract clause into the text area
2. Click **"🔍 Classify Clause"**
3. View the **detected clause type** and **Top 5 confidence scores**

### Example Input

```
This Agreement shall be governed by and construed in accordance with
the laws of the State of California, without regard to conflict of
law principles.
```

### Example Output

```
 Detected: Governing Law   (Confidence: 97.4%)

Top 5:
  #1  Governing Law              97.4%
  #2  Effective Date              1.2%
  #3  Agreement Date              0.8%
  #4  Anti-Assignment             0.4%
  #5  Notice Period To Terminate  0.2%
```

---

##  Model Architecture

```
Raw Clause Text
      │
      ├──► TF-IDF Word Vectorizer  ─► 8,000-dim sparse vector
      │         (15k vocab, top 8k features used)
      │
      └──► ComplementNB Classifier ─► Probability distribution (41 classes)
                                             │
                                             └──► LabelEncoder ─► Clause Name
```

### Why This Pipeline?

| Component | Choice | Reason |
|---|---|---|
| Vectorizer | TF-IDF Word | Captures legal term frequency; robust on formal language |
| Classifier | ComplementNB | Superior on imbalanced classes; fast inference |
| Feature dim | 8,000 | Optimal balance between vocabulary coverage and overfitting |

---

##  Supported Clause Types

<details>
<summary>Click to expand all 41 clause types</summary>

| # | Clause Type | # | Clause Type |
|---|---|---|---|
| 1 | Affiliate License – Licensee | 22 | Most Favored Nation |
| 2 | Affiliate License – Licensor | 23 | No-Solicit of Customers |
| 3 | Agreement Date | 24 | No-Solicit of Employees |
| 4 | Anti-Assignment | 25 | Non-Compete |
| 5 | Audit Rights | 26 | Non-Disparagement |
| 6 | Cap on Liability | 27 | Non-Transferable License |
| 7 | Change of Control | 28 | Notice Period to Terminate Renewal |
| 8 | Competitive Restriction Exception | 29 | Parties |
| 9 | Covenant Not to Sue | 30 | Post-Termination Services |
| 10 | Document Name | 31 | Price Restrictions |
| 11 | Effective Date | 32 | Renewal Term |
| 12 | Exclusivity | 33 | Revenue / Profit Sharing |
| 13 | Expiration Date | 34 | ROFR / ROFO / ROFN |
| 14 | Governing Law | 35 | Source Code Escrow |
| 15 | Insurance | 36 | Termination for Convenience |
| 16 | IP Ownership Assignment | 37 | Third Party Beneficiary |
| 17 | Irrevocable or Perpetual License | 38 | Uncapped Liability |
| 18 | Joint IP Ownership | 39 | Unlimited / All-You-Can-Eat License |
| 19 | License Grant | 40 | Volume Restriction |
| 20 | Liquidated Damages | 41 | Warranty Duration |
| 21 | Minimum Commitment | | |

</details>

---

##  Performance

| Metric | Score |
|---|---|
| Dataset | CUAD (510 contracts) |
| Classes | 41 |
| Vectorizer | TF-IDF Word (15k vocab) |
| Classifier | ComplementNB |
| Inference Speed | < 50ms per clause |

>  **Note:** Model accuracy varies by clause type. Short or ambiguous clauses may yield lower confidence. For production use, consider fine-tuning on domain-specific corpora.

---

##  Roadmap

- [ ] Add BERT / Legal-BERT fine-tuned model option
- [ ] Batch processing — upload a full contract PDF
- [ ] Clause highlighting within document view
- [ ] Export results to CSV / JSON
- [ ] REST API wrapper (FastAPI)
- [ ] Docker containerization
- [ ] Multi-language support

---

##  Contributing

Contributions are welcome! Here's how to get started:

```bash
# 1. Fork the repo and clone it
git clone https://github.com/your-username/legal-clause-classifier.git

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Commit your changes
git commit -m "feat: add your feature"

# 4. Push and open a Pull Request
git push origin feature/your-feature-name
```

Please follow [Conventional Commits](https://www.conventionalcommits.org/) and open an issue before working on major changes.

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgements

- [CUAD Dataset](https://www.atticusprojectai.org/cuad) — The Atticus Project
- [scikit-learn](https://scikit-learn.org/) — ML pipeline
- [Streamlit](https://streamlit.io/) — Web framework

---

<div align="center">

##  Author

#Sarfaraz Ali#

⭐ **Star this repo if you find it useful!**

</div>
