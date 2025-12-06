
# Deepfake Detection using Vision-KAN + Vision-LSTM  
### DFDC Project — UTA CSE-6367 Computer Vision

> Developed by **Isit Thakkar (1002229820)**  & **Krushna Bhujbal (1002241902)**
> University of Texas at Arlington

---

## 📌 Project Overview

Deepfake generation has become increasingly sophisticated, creating synthetic videos indistinguishable from real human recordings.  
This project presents a **hybrid deepfake detection model** that leverages:

- **Vision-KAN** (Knowledge Attention Network) for spatial feature learning, and  
- **Vision-LSTM** for temporal motion reasoning.

The model is trained and evaluated using the **DeepFake Detection Challenge (DFDC)** dataset.

Our pipeline combines **computer vision preprocessing, feature extraction, sequence modelling and binary classification** to identify manipulated video frames.

# Dataset
We use the DFDCDFDC Kaggle dataset:
🔗 https://www.kaggle.com/datasets/aleksandrpikul222/dfdcdfdc
Downloaded dynamically via:
import kagglehub
path = kagglehub.dataset_download('aleksandrpikul222/dfdcdfdc')

# Model Architecture
Our hybrid classifier contains:
1️⃣ Vision-KAN
Extracts deep spatial structure (texture artifacts, compression noise, feature inconsistencies)
2️⃣ Vision-LSTM
Learns frame–temporal relationships such as:
blink frequency issues
unnatural jaw motion
expression discontinuities
3️⃣ Fusion Classifier
KAN Output  → concat → Linear → ReLU → Linear → Real/Fake Logit
LSTM Output →

# Training Flow (DFDC_Train_Mac.ipynb)
✔ Preprocessing (resize, normalize, tensor conversion)
✔ Custom dataset loader with label assignment
✔ Training loop with BCE loss and Adam optimizer
✔ Validation loop + best model checkpoint saving
✔ Test inference + ROC-AUC scoring
torch.save(model.state_dict(), "as_model_best.pt") 

# Inference Demo (test_mac.ipynb)
To classify an image:
Load test image
Preprocess using same normalization
Reconstruct Vision-KAN + Vision-LSTM model
Load trained weights
Predict real/fake
Results example:
Prediction: REAL
or
Prediction: FAKE

# Evaluation Metric
We report ROC-AUC, preferred for binary detection problems:
auc = roc_auc_score(all_labels, all_probs)
print("Test ROC-AUC:", auc)

# How to Run This Project
1️⃣ Clone repository
git clone https://github.com/isitthakkar11/Deepfake-Detection-using-Vision-KAN-Vision-LSTM.git
cd Deepfake-Detection-using-Vision-KAN-Vision-LSTM
2️⃣ Install dependencies
pip install -r requirements.txt
OR manually install libraries.
3️⃣ Open Notebook for training
jupyter notebook DFDC_Train_Mac.ipynb
4️⃣ Run inference
jupyter notebook test_mac.ipynb

## Authors
Isit Thakkar
Krushna Bhujbal

⭐ Star This Repo
If you find this useful, please ⭐ the repository and share your feedback!
Deepfake detection is not just a software challenge — it is a trust challenge.
Thank you for exploring our work.
