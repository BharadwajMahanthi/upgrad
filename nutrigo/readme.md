# Nutrigo

![Nutrigo Logo](static/images/logo.png)

**Nutrigo** is a comprehensive recipe recommendation system designed to help users discover healthy and delicious recipes tailored to their nutritional preferences and dietary needs. Built with Flask, Nutrigo leverages machine learning models to provide personalized recommendations, ensuring that every meal aligns with your health goals.

---

## 📖 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Features

- **User Authentication:**
  - Secure user registration and login.
  - Profile management for updating personal preferences.

- **Personalized Recipe Recommendations:**
  - Tailored suggestions based on user dietary preferences and health goals.
  - Dynamic filtering of recipes to match nutritional needs.

- **Recipe Management:**
  - Detailed recipe views with ingredients, instructions, and nutritional information.
  - Admin dashboard for managing recipe data.

- **Health Analytics:**
  - Visualizations like confusion matrices and ROC curves to evaluate model performance.
  - Training history graphs to monitor model improvements.

- **Model Retraining:**
  - Interface for administrators to retrain machine learning models with new data.

- **Responsive Design:**
  - User-friendly interface optimized for both desktop and mobile devices.

---

---

## 🛠️ Installation

Follow these steps to set up Nutrigo on your local machine:

### 1. **Clone the Repository**

```bash
git clone https://github.com/BharadwajMahanthi/upgrad.git
cd upgrad
```

### 2. **Set Up Virtual Environment**

It's recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 4. **Set Up Environment Variables**

Create a `.env` file in the project root directory and add the following:

```env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your_secret_key
SQLALCHEMY_DATABASE_URI=sqlite:///nutrigo.db
```

**Note:** Replace `your_secret_key` with a strong secret key. For production, consider using more secure database systems like PostgreSQL.

### 5. **Initialize the Database**

```bash
flask db init
flask db migrate -m "Initial migration."
flask db upgrade
```

### 6. **Run the Application**

```bash
flask run
```

The application will be accessible at `http://127.0.0.1:5000/`.

---

## 🚀 Usage

### **1. User Registration and Login**

- Navigate to the [Register](#) page to create a new account.
- After registration, log in using your credentials.

### **2. Updating Profile**

- Access the [Update Profile](#) section to modify your dietary preferences and health goals.

### **3. Exploring Recipes**

- Browse through the [Recipe Recommendations](#) tailored to your profile.
- Click on a recipe to view detailed information, including ingredients and nutritional facts.

### **4. Admin Dashboard**

- Admin users can access the [Admin Dashboard](#) to manage recipes and monitor model performance.
- Use the [Retrain Model](#) feature to update the recommendation system with new data.

### **5. Health Analytics**

- View visual analytics like confusion matrices and ROC curves to understand model accuracy.
- Monitor training history to track improvements over time.

---

## 📁 Project Structure

```
nutrigo/
│
├── app.py
├── classifier.ipynb
├── core-data-images/
│   └── ... (image files)
├── core-data-test_rating.csv
├── core-data-train_rating.csv
├── core-data-valid_rating.csv
├── core-data_recipe.csv
├── data/
│   ├── combined_item_features.npz  # Managed by Git LFS
│   ├── core_image_features.npy      # Managed by Git LFS
│   └── raw_image_features.npy       # Managed by Git LFS
├── healthiness_confusion_matrix.png
├── healthiness_roc_curve.png
├── .gitignore
├── instance/
│   └── nutrigo.db                   # Consider excluding if not necessary
├── models/
│   ├── healthiness_model.h5
│   ├── le_recipe.pkl                # Managed by Git LFS
│   ├── le_user.pkl                  # Managed by Git LFS
│   ├── scaler.pkl                   # Managed by Git LFS
│   ├── svd_model.pkl                # Managed by Git LFS
│   ├── tfidf_vectorizer.pkl         # Managed by Git LFS
│   └── training.log
├── nutrigo.code-workspace
├── raw-data-images/
│   └── ... (image files)
├── raw-data_interaction.csv
├── raw-data_recipe.csv
├── requirements.txt
├── settings.json
├── static/
│   ├── css/
│   │   └── styles.css
│   └── images/
│       └── logo.png
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── register.html
│   ├── update_profile.html
│   ├── recommendations.html
│   ├── recipe_detail.html
│   ├── admin_login.html
│   ├── admin_dashboard.html
│   └── retrain_model.html
├── test.py
├── training_history.png
├── venv/                            # Excluded via .gitignore
│   └── ... (virtual environment files)
├── migrations/
│   └── ... (Flask-Migrate migration files)
└── __pycache__/
    └── ... (compiled Python files)
```

---

## 🧰 Technologies Used

- **Backend:**
  - [Flask](https://flask.palletsprojects.com/) - Web framework
  - [Flask-Migrate](https://flask-migrate.readthedocs.io/) - Database migrations
  - [Flask-Login](https://flask-login.readthedocs.io/) - User session management
  - [SQLAlchemy](https://www.sqlalchemy.org/) - ORM for database interactions

- **Frontend:**
  - HTML5 & CSS3
  - [Bootstrap](https://getbootstrap.com/) - CSS framework for responsive design

- **Machine Learning:**
  - [TensorFlow](https://www.tensorflow.org/) - Machine learning library
  - [Scikit-learn](https://scikit-learn.org/) - ML algorithms and tools

- **Data Visualization:**
  - [Matplotlib](https://matplotlib.org/) - Plotting library

- **Version Control:**
  - [Git](https://git-scm.com/) - Version control system
  - [GitHub](https://github.com/) - Repository hosting
  - [Git LFS](https://git-lfs.github.com/) - Large file storage

- **Others:**
  - [Jupyter Notebook](https://jupyter.org/) - Interactive computing
  - [Python-dotenv](https://github.com/theskumar/python-dotenv) - Environment variable management

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

### **Steps to Contribute**

1. **Fork the Project**
   - Click the [Fork](https://github.com/BharadwajMahanthi/upgrad/fork) button at the top right of the repository page.

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/upgrad.git
   cd upgrad
   ```

3. **Create a New Branch**
   ```bash
   git checkout -b feature/YourFeatureName
   ```

4. **Make Your Changes**
   - Implement your feature or bug fix.
   - Ensure code follows the project's coding standards.

5. **Commit Your Changes**
   ```bash
   git commit -m "Add some feature"
   ```

6. **Push to the Branch**
   ```bash
   git push origin feature/YourFeatureName
   ```

7. **Open a Pull Request**
   - Navigate to your fork on GitHub.
   - Click the **New Pull Request** button.
   - Provide a clear description of your changes.

### **Guidelines**

- **Code Quality:** Ensure your code is clean, well-documented, and follows best practices.
- **Testing:** Write tests for new features or bug fixes.
- **Documentation:** Update the README and other relevant documentation as needed.
- **Respect:** Be respectful and considerate in all interactions.

---

## 📜 License

Distributed under the [MIT License](LICENSE).

---

## 📬 Contact

**Bharadwaj Mahanthi**

- **GitHub:** [@BharadwajMahanthi](https://github.com/BharadwajMahanthi)
- **Email:** mbpd.1999@gmail.com

Project Link: [https://github.com/BharadwajMahanthi/upgrad](https://github.com/BharadwajMahanthi/upgrad)

---

## 📝 Acknowledgements

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Bootstrap Documentation](https://getbootstrap.com/docs/)
- [GitHub Guides](https://guides.github.com/)
- [GitHub's Git Large File Storage](https://git-lfs.github.com/)
- [Real Python](https://realpython.com/) for excellent Python tutorials

---

*This README was generated with the help of ChatGPT.*
