# 🧬 Multi-Cancer Detection Using Machine Learning

This project uses Convolutional Neural Networks (CNNs) to classify and detect multiple types of cancers — Breast, Lung, Skin, and Blood — from medical images. It provides a full pipeline including data preprocessing, model training, evaluation, and a web-based prediction interface using Flask.

---

## 📁 Project Directory Structure

```
multi_cancer_detection_using_ML/
├── datasets/
│   ├── benign_malignant/
│   ├── cancer_type/
│
├── models/
│   ├── cancer_type_model.keras
│   ├── breast_cancer_model.keras
│   ├── lung_cancer_model.keras
│   ├── skin_cancer_model.keras
│   └── blood_cancer_model.keras
│
├── results_model/
│   └── evaluation_report.pdf
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── data_split.py
│   └── train.py
│
├── static/
│   ├── uploads/
│   └── styles.css
│
├── templates/
│   ├── index.html
│   ├── detection.html
│   ├── lifestyle.html
│   └── support.html
│
├── uploads/
├── app.py
├── .env
├── README.md
```

---

## ⚙️ Setup Instructions

### 📦 Install Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
tensorflow
opencv-python
numpy
matplotlib
pandas
scikit-learn
flask
python-dotenv
```

---

## 🧼 Step 1: Preprocess and Split Dataset

Split raw images into training, validation, and test sets:

```bash
python scripts/data_split.py
```

---

## 🏗️ Step 2: Train CNN Models

Train the cancer classification models using:

```bash
python scripts/train.py
```

This will:
- Train CNN models using preprocessed data
- Save trained models in the `models/` directory
- Save training history and evaluation report in `results_model/`

---

## 🌐 Step 3: Run Flask Web Application

Start the web interface for image-based prediction:

```bash
python app.py
```

- Visit [http://localhost:5000](http://localhost:5000) in your browser
- Upload an image via the interface
- View the cancer type classification and lifestyle/support recommendations

---

## ✅ Included Functionality

- 🏷️ Cancer Type Classification: Breast, Lung, Skin, Blood
- 📊 Evaluation Report: PDF-based metrics from training
- 🌐 Web UI: Upload and detect using `index.html` and `detection.html`
- 🧬 Lifestyle & Support: Additional resources in `lifestyle.html` and `support.html`

