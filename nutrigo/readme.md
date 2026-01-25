# NutriGo - Personal AI Nutritionist v2.0

![NutriGo Banner](static/images/logo.png)

**NutriGo** is a high-performance recipe recommendation engine that combines deep learning with massive nutritional datasets to help users achieve their biological health goals. Version 2.0 introduces hyper-personalized recommendations, a stunning modern UI, and a direct integration of over 3.7 million historical user reviews.

---

## 🚀 Key Upgrades in v2.0

### 📊 Massive Data Enrichment

- **3.7 Million Reviews**: Integrated a colossal dataset of historical user experiences to power social proof in recommendations.
- **Global Recipe Stats**: Every recipe now displays its original `aver_rate` (1-5 stars) and `review_nums`.
- **Bio-Metric Nutrients**: Expanded nutritional profiles to include **Sodium, Sugars, Cholesterol, and Saturated Fat** for every recipe.

### 🎨 Modern UI (Glassmorphism)

- **Fresh Light Theme**: Completely redesigned with a premium, clean aesthetic using glassmorphism and smooth micro-animations.
- **Personalization Guard**: "Your Identity" and "Preferences" are now mandatory, ensuring the AI has sufficient context for perfect matching.
- **Enhanced Recipe View**: Redesigned detail pages with a "Nutritional Summary" grid and historical comment feeds.

### ⚙️ Performance & Stability

- **SQLite WAL Mode**: Enabled Write-Ahead Logging for seamless concurrent access (reads don't block writes).
- **Intelligent Proxy**: Custom Flask route for serving high-quality local images (`data/core-data-images/`) with automatic HTTPS fallback.
- **Resilient Models**: The application now starts in "Degraded Mode" even if heavy ML dependencies or models are missing, ensuring core functionality.

---

## 🎯 Core Features

- **Personalized Recommendations**:
  - Uses Hybrid Engine (SVD + Content Filtering).
  - Constraints based on user goals (e.g., Weight Mitigation, Muscle Growth).
- **AI Health Audit**: Real-time healthiness probability scoring using a deep neural network.
- **Admin Control Center**:
  - Dedicated secure dashboard for user management.
  - One-click model retraining and data integrity audits.
- **Responsive Design**: Optimized for desktop, tablet, and mobile bio-tracking.

---

## 🛠️ Installation

### 1. **Clone & Setup**

```bash
git clone https://github.com/BharadwajMahanthi/upgrad.git
cd upgrad/nutrigo
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. **Environment Configuration**

Create a `.env` file in the `nutrigo/` directory:

```env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=nutrigo-secure-key
SQLALCHEMY_DATABASE_URI=sqlite:///data/nutrigo.db
```

### 3. **Run v2.0**

```bash
python app.py
```

_Note: The app will automatically run `ensure_data_integrity()` and initialize the Hybrid Engine on startup._

---

## 📁 Project Structure (nutrigo/)

```
nutrigo/
├── data/                    # SQLite database & CSV snapshots
│   ├── core-data-images/    # High-res local recipe images
│   └── nutrigo.db           # (WAL Enabled)
├── models/                  # ML Artifacts (SVD, Scalar, TF-IDF)
├── static/
│   ├── css/                 # Modern light-theme stylesheets
│   └── images/
├── templates/               # Glassmorphism templates
│   ├── classification.html  # AI Health Audit Detail
│   ├── explain.html        # Recommendation Match Logic
│   └── admin_dashboard.html # User Management
└── app.py                   # Main Flask server & Route logic
```

---

## 🧰 Powered By

- **Core**: Flask, SQLAlchemy, Jinja2
- **Intelligence**: TensorFlow, Scikit-learn, Scikit-surprise (Optional)
- **Design**: Bootstrap 5, FontAwesome, Google Fonts (Outfit/Inter)
- **Concurrency**: SQLite WAL Mode

---

## 📜 License

Distributed under the [MIT License](LICENSE).

---

## 📬 Contact

**Bharadwaj Mahanthi**

- **GitHub**: [@BharadwajMahanthi](https://github.com/BharadwajMahanthi)
- **Email**: mbpd.1999@gmail.com
- **Project**: [NutriGo AI](https://github.com/BharadwajMahanthi/upgrad)

---

_Developed by Bharadwaj Mahanthi_  

