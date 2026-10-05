# Campus Waste Segregation Assistant

A computer-vision-based waste classification system that identifies common waste materials from images and assists users in determining the appropriate waste category.

The system uses a **MobileNetV2 transfer-learning model** trained on a six-class waste image dataset and is deployed through a **Streamlit web application** for image upload and camera-based classification.

The project follows a complete machine-learning workflow:

```text
Image
  ↓
Image preprocessing
  ↓
MobileNetV2
  ↓
Feature extraction
  ↓
Waste classification
  ↓
Prediction probabilities
  ↓
Top-3 results + confidence
  ↓
Waste segregation guidance
```

## 1. Project Objective

The objective is to build an image-based waste segregation assistant capable of classifying waste into six categories:

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash

The system accepts a waste image and provides:

- Predicted waste category
- Prediction confidence
- Top-3 predictions
- Waste classification information
- Segregation/disposal guidance through the Streamlit interface

The project focuses on real-world image inference rather than relying only on training accuracy.

---

## 2. Technical Architecture

The project uses:

- **Python** — machine-learning and application development
- **TensorFlow + Keras** — model training and inference
- **MobileNetV2** — pretrained CNN used for transfer learning
- **ImageNet weights** — pretrained visual feature representation
- **Pillow (PIL)** — image processing
- **NumPy** — numerical processing
- **Scikit-learn** — stratified splitting, class weights and evaluation
- **Matplotlib** — visualization and model analysis
- **Streamlit** — interactive web application
- **JSON** — class mapping and model metadata
- **`.keras`** — trained model serialization

---

## 3. Waste Categories

| Class ID | Category |
|---:|---|
| 0 | Cardboard |
| 1 | Glass |
| 2 | Metal |
| 3 | Paper |
| 4 | Plastic |
| 5 | Trash |

The class ordering is stored in:

```text
class_names.json
```

This ensures that model output indices are mapped consistently to the correct waste categories.

---

## 4. Dataset

The project uses a TrashNet-style dataset containing six waste categories.

| Category | Images |
|---|---:|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |
| **Total** | **2527** |

The dataset is naturally imbalanced, particularly because the `trash` category contains substantially fewer images.

---

## 5. Dataset Split

A fixed **stratified 70/15/15 split** is used:

```text
Total       → 2527
Training    → 1768
Validation  → 379
Test        → 380
```

Stratification preserves approximately the same class distribution across the three subsets.

The exact file-level split is saved in:

```text
splits.csv
```

This makes the experiment reproducible.

---

## 6. Data Leakage Prevention

The training pipeline explicitly checks for overlap between:

```text
Training ∩ Validation
Training ∩ Test
Validation ∩ Test
```

All intersections must contain zero images.

This prevents the same image from unintentionally appearing in multiple subsets.

---

## 7. Image Preprocessing

The preprocessing pipeline is:

```text
Input image
    ↓
Read image
    ↓
RGB conversion
    ↓
Resize to 224 × 224
    ↓
Float32 conversion
    ↓
0–255 pixel values
    ↓
MobileNetV2
```

The final input size is:

```text
224 × 224 × 3
```

The model performs the final input scaling internally.

---

## 8. MobileNetV2 Architecture

MobileNetV2 is used with ImageNet pretrained weights and without its original ImageNet classification head.

The project-specific architecture is:

```text
Input
  ↓
Data Augmentation
  ↓
Rescaling
  ↓
MobileNetV2
  ↓
Global Average Pooling
  ↓
Dropout
  ↓
Dense(6)
  ↓
Softmax
```

The six-output Softmax layer represents the six waste classes.

---

## 9. Internal Input Scaling

The model receives raw image pixels in the `0–255` range.

An internal Keras `Rescaling` layer converts them to approximately `-1 to +1`:

```text
x / 127.5 - 1
```

Therefore, the Streamlit application does **not** apply another `preprocess_input()` transformation.

This prevents accidental double preprocessing during inference.

---

## 10. Data Augmentation

Training images use:

- Horizontal flipping
- Random rotation
- Random zoom
- Random translation
- Random brightness
- Random contrast

Validation and test images are not augmented.

Vertical flipping is intentionally avoided because it can produce unrealistic orientations for waste objects.

---

## 11. Class Imbalance Handling

Balanced class weights are calculated from the training labels.

The resulting weights were:

| Class | Weight |
|---|---:|
| Cardboard | 1.045 |
| Glass | 0.842 |
| Metal | 1.027 |
| Paper | 0.708 |
| Plastic | 0.874 |
| Trash | 3.069 |

The larger weight for `trash` increases the penalty for mistakes involving the underrepresented class.

---

## 12. Two-Stage Training

### Stage 1 — Transfer Learning

Initially:

```text
MobileNetV2 → Frozen
Classification Head → Trainable
```

Configuration:

```text
Optimizer: Adam
Learning rate: 0.001
Loss: Sparse Categorical Crossentropy
Metric: Accuracy
```

Stage 1 was trained for up to 20 epochs.

Best validation accuracy:

```text
≈ 83.11%
```

### Stage 2 — Fine-Tuning

The upper portion of MobileNetV2 is selectively unfrozen.

```text
Total MobileNetV2 layers → 154
Trainable MobileNetV2 layers → 19
Batch Normalization layers → Frozen
```

Fine-tuning learning rate:

```text
0.00003
```

Stage 2 was trained for up to 25 epochs.

Best validation accuracy:

```text
≈ 87.60%
```

---

## 13. Training Controls

The training pipeline uses:

### Early Stopping

Monitors:

```text
val_loss
```

and restores the best weights.

### Model Checkpointing

Best models are saved during training:

```text
best_head_model.keras
best_finetuned_model.keras
```

### Learning Rate Reduction

`ReduceLROnPlateau` reduces the learning rate when validation loss stops improving.

---

## 14. Final Model

The production model is:

```text
waste_segregation_mobilenetv2.keras
```

Model size:

```text
≈ 9.22 MB
```

It contains the MobileNetV2 feature extractor together with the project-specific classification head.

---

## 15. Final Evaluation

The final model was evaluated on the isolated test set containing approximately 380 images.

```text
Test Loss     → 0.4235
Test Accuracy → 85.79%
```

The test set was not used for model training or model selection.

---

## 16. Classification Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Cardboard | 92.73% | 85.00% | 88.70% |
| Glass | 86.49% | 84.21% | 85.53% |
| Metal | 82.09% | 88.71% | 85.27% |
| Paper | 86.02% | 89.89% | 87.91% |
| Plastic | 85.71% | 82.19% | 83.92% |
| Trash | 76.19% | 80.00% | 78.05% |

Overall:

```text
Accuracy        → 85.79%
Macro F1-score  → 84.86%
Weighted F1     → 85.80%
```

The `trash` class remains the most challenging category, consistent with its smaller representation in the dataset.

---

## 17. Confusion Analysis

The major observed confusion patterns include:

```text
Cardboard → Paper
Glass     → Plastic
Plastic   → Glass
Plastic   → Metal
Metal     → Glass
Paper     → Cardboard
```

These errors are understandable because some materials can have visually similar appearance:

```text
Plastic ↔ Glass
Plastic ↔ Metal
Cardboard ↔ Paper
```

depending on lighting, background, shape and image quality.

---

## 18. Streamlit Application

The trained model is integrated into a Streamlit interface.

The application supports:

```text
Image Upload
      +
Camera Input
      ↓
Image Preparation
      ↓
Model Inference
      ↓
Class Probabilities
      ↓
Top-3 Predictions
      ↓
Confidence Display
```

The application is designed to provide an accessible interface instead of exposing the underlying machine-learning workflow directly to the user.

---

## 19. Application Image Processing

The application handles:

- EXIF orientation correction
- Transparent PNG handling
- RGB conversion
- Image resizing
- Numerical conversion for inference

The final image passed to the model is:

```text
224 × 224 × 3
```

with raw pixel values before the model's internal rescaling.

---

## 20. Prediction Output

For an input image, the application can display:

- Predicted class
- Confidence
- Top-3 predictions

Example:

```text
Prediction:
Plastic

Confidence:
99.26%

Top Predictions:
1. Plastic → 99.26%
2. Glass   → 0.57%
3. Trash   → 0.09%
```

---

## 21. Real-World Test

A real-world plastic bottle image was tested independently from the dataset.

The trained model predicted:

```text
Plastic → 99.26%
Glass   → 0.57%
Trash   → 0.09%
```

This is significant because an earlier version of the project incorrectly classified a real-world plastic bottle as metal.

The new training pipeline therefore demonstrated substantially better behavior on this particular real-world test image.

This result is an additional qualitative test and should not be treated as a replacement for the formal test-set accuracy.

---

## 22. Project Structure

```text
Campus_Waste_Segregation_Assistant/
│
├── app.py
├── requirements.txt
│
├── waste_segregation_mobilenetv2.keras
├── class_names.json
├── model_info.json
├── splits.csv
│
├── train_waste_classifier.ipynb
│
└── README.md
```

The local `venv/` directory should normally not be committed to GitHub.

---

## 23. Important Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `waste_segregation_mobilenetv2.keras` | Final trained model |
| `class_names.json` | Class-index mapping |
| `model_info.json` | Model and training metadata |
| `splits.csv` | Exact dataset split |
| `train_waste_classifier.ipynb` | Complete training workflow |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## 24. Local Environment

The tested local environment currently uses:

```text
Python       3.14.2
TensorFlow   2.22.0-rc0
Keras        3.16.0.dev...
Pillow       12.3.0
NumPy        2.5.3
Streamlit    1.65.0
```

The local environment successfully loads:

```text
waste_segregation_mobilenetv2.keras
```

as:

```text
waste_segregation_mobilenetv2
```

---

## 25. Installation

Clone the repository and enter the project directory.

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Verify TensorFlow:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

Verify the trained model:

```powershell
python -c "import keras; model=keras.models.load_model('waste_segregation_mobilenetv2.keras', compile=False); print(model.name)"
```

---

## 26. Run the Application

Activate the virtual environment and run:

```powershell
streamlit run app.py
```

The application is normally available at:

```text
http://localhost:8501
```

Open the URL in a browser and upload a waste image or use the camera input.

---

## 27. Reproducibility

The project stores the important components needed to reproduce the experiment:

```text
Training notebook
        +
Dataset split
        +
Class mapping
        +
Model metadata
        +
Saved trained model
```

The dataset split uses the random seed:

```text
42
```

The exact split is saved in `splits.csv`.

---

## 28. Limitations

### Dataset Size

The model is trained on approximately 2527 images, which limits the diversity of visual examples.

### Class Imbalance

The `trash` category has substantially fewer images than the other categories.

### Real-World Variation

Real images may contain:

- Different lighting
- Cluttered backgrounds
- Multiple objects
- Unusual camera angles
- Partially hidden objects
- Damaged waste
- Transparent materials
- Visually similar materials

### Single-Object Classification

The system is primarily designed for classification of a dominant waste object.

It is not an object-detection or instance-segmentation system.

### Classification Ambiguity

Some waste materials can be visually similar, so predictions should be treated as assistance rather than guaranteed material identification.

---

## 29. Future Improvements

Potential improvements include:

- Expanding the real-world training dataset
- Collecting more examples for the `trash` category
- Creating a dedicated real-world evaluation dataset
- Improving background and lighting augmentation
- Adding confidence-aware rejection thresholds
- Detecting multiple waste objects
- Exploring other lightweight architectures
- Model quantization for faster inference
- Mobile deployment
- Multilingual interface
- Classification history
- Waste-bin recommendations
- Continuous evaluation using new real-world images

---

## 30. Key Technical Highlights

```text
✓ Transfer Learning
✓ MobileNetV2
✓ Image Classification
✓ Data Augmentation
✓ Stratified Dataset Splitting
✓ Data Leakage Checking
✓ Class Weighting
✓ Two-Stage Training
✓ Fine-Tuning
✓ Batch Normalization Freezing
✓ Early Stopping
✓ Learning Rate Scheduling
✓ Model Checkpointing
✓ Confusion Matrix Analysis
✓ Precision / Recall / F1 Evaluation
✓ Reproducible Dataset Splits
✓ Keras Model Serialization
✓ Streamlit Deployment
✓ Image Upload
✓ Camera Input
✓ Top-3 Prediction Analysis
✓ Confidence-Based Output
```

---

## 31. Final Results

```text
────────────────────────────────────────────
              MODEL RESULTS
────────────────────────────────────────────

Dataset Size        2527 images
Classes              6
Input Size           224 × 224 × 3
Architecture         MobileNetV2

Training Strategy    Transfer Learning
                     + Fine-Tuning

Stage 1 Val Acc      ≈ 83.11%
Stage 2 Val Acc      ≈ 87.60%

Test Accuracy        85.79%
Test Loss            0.4235
Macro F1             84.86%
Weighted F1           85.80%

Model Size           ≈ 9.22 MB
────────────────────────────────────────────
```

---

## 32. Complete Workflow

```text
Waste Image Dataset
        ↓
Dataset Inspection
        ↓
Class Distribution Analysis
        ↓
Stratified 70/15/15 Split
        ↓
Leakage Verification
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
MobileNetV2 Transfer Learning
        ↓
Class-Weighted Stage 1 Training
        ↓
Upper-Layer Fine-Tuning
        ↓
Validation-Based Model Selection
        ↓
Untouched Test Evaluation
        ↓
Classification Report
        ↓
Confusion Matrix
        ↓
Final .keras Model
        ↓
Streamlit Integration
        ↓
Real-World Image Testing
```

---

## 33. Disclaimer

This project is an **AI-assisted waste classification system** intended for educational and practical assistance.

Predictions depend on image quality and the visual patterns represented in the training data. The system should not be treated as a guaranteed substitute for human waste-management decisions, especially for ambiguous, contaminated or mixed materials.

---

## License

No license has been specified for this repository yet.
