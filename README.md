# Advanced Masked Autoencoder for Industrial Anomaly Detection
## Sapienza Università degli Studi di Roma - Computer Vision class project - 2024/25

This project introduces an advanced **Masked Autoencoder (MAE)** architecture, built upon a Vision Transformer (ViT), for **unsupervised industrial anomaly detection and localization**. The model is designed to learn the normal appearance of objects from defect-free images and then identify anomalies by detecting deviations in its reconstruction.

The core idea is that the model, once trained as an expert on "normal" data, will fail to accurately reconstruct unseen defects. This reconstruction error serves as a powerful signal for both detecting and localizing anomalies. The model has been validated on the standard **MVTec AD** and **BTAD** datasets.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d2da645-3c47-4d2e-9934-bbfbf65b7f4f" width="90%">
  <br>
  <em>Figure 1: Result achieved on 'Hazelnut' class after fine-tuning</em>
</p>

---

## ✨ Key Features

The vision that guides our work was to develop a fast and lightweight model while ensuring good and comparable performance. This model enhances the standard MAE framework with several innovative features tailored for high-performance anomaly detection (note: don't expect state-of-the-art performance):

* **Multi-Scale Feature Pyramid**: Unlike a standard MAE which only uses the final encoder output, this model extracts features from **intermediate encoder layers**. A `FeatureAggregationModule` then combines them to create a richer, multi-scale representation that captures both low-level textures and high-level semantics.

* **Block-Wise Masking Strategy**: Instead of masking random, sparse patches, this model employs a block-wise masking strategy. This forces the model to learn to reconstruct larger, contiguous regions, thereby improving its understanding of spatial context and long-range dependencies.

* **Hybrid Positional Embeddings**: The model leverages both **learnable** and **sinusoidal** positional embeddings. This hybrid approach combines the data-adaptive nature of learnable embeddings with the strong geometric foundation of fixed sinusoidal embeddings.

* **Dynamic Normalization (DyT)**: A custom `DyT` normalization layer is used in place of the standard `LayerNorm`. It introduces a learnable `alpha` parameter within a `tanh` activation, offering more flexible and dynamic normalization.

* **Lightweight, Attentive Decoder**: Following the MAE philosophy, the decoder is significantly shallower than the encoder (`depth_dec=2`). It is architecturally designed to receive the aggregated features from the encoder pyramid, allowing for a more informed and context-aware reconstruction process.

---

## 🏛️ Model Architecture

The data flows through the model as follows:

1.  **Input & Patching**: The input image is divided into a sequence of non-overlapping patches.
2.  **Embedding**: A hybrid (sinusoidal + learnable) positional embedding is added to each patch token.
3.  **Masking**: A significant portion of the patches is hidden using the block-wise masking strategy (during training only).
4.  **Encoder**: The visible patches are processed by a deep Transformer encoder (`depth_enc=16`). During this pass, outputs from intermediate layers are collected.
5.  **Feature Aggregation**: The `FeatureAggregationModule` takes the collected multi-level features and combines them into a single, rich feature map.
6.  **Decoder**: A shallow Transformer decoder (`depth_dec=2`) receives the encoded patches and is designed to leverage the aggregated features to reconstruct the original full set of image patches.
7.  **Output**: The model outputs the reconstructed image patches.

<p align="center">
  <img src="https://github.com/user-attachments/assets/49f1e3f7-eaa7-4694-85a7-291bc1258ecc" width="80%">
  <br>
  <em>Figure 2: Detailed architecture diagram showing the data flow through our MAE model</em>
</p>

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* PyTorch 1.10+
* torchvision
* scikit-learn
* matplotlib
* Pillow (PIL)

### Installation

* Open and run directly on Kaggle!
  If you are using Google Colab, Jupyter, or another notebook, remember to import the original MVTec AD and BTAD datasets.

### Training & Fine-Tuning

The process is divided into several sections, beginning with data augmentation and then proceeding to pre-training and fine-tuning stages. We observed that after approximately 80 epochs, a plateau is reached and the loss starts to decrease slowly. However, if you have time and want better performance, we suggest trying with more epochs, especially during training (200-300 epochs). 

1.  **Pre-training**:

    In this stage, a total loss is calculated as the weighted sum of the reconstruction loss and SSIM (Structural Similarity Index Measure) loss. 

2.  **Fine-tuning & Validation**:

    Here we define the reconstruction loss as the MSE on images from patch reconstruction. 
    In this stage, we also evaluate the progressing model on the validation set to understand what is happening under the hood: is our model improving with fine-tuning or not? This is the right place for this question. **F1, AUC, ROC** curve, and **AUPRO** are the metrics used to validate the process, with AUC being especially important for deciding when to stop fine-tuning and save our 'best' model.

    The validation set is also fundamental for dynamically defining a threshold to separate 'good' samples from anomalies according to the fine-tuning evolution. We also tried to define a static threshold based on the training good samples distribution, but according to our experiments, a dynamically updated threshold that moves according to validation results provides slightly better discrimination between good and anomalous samples.

   
   
## 📊 Results

The model achieves good performance on the MVTec AD and BTAD datasets. Key evaluation metrics include image-level **AUC** and **F1-Score**, as well as the pixel-level **AUPRO** for localization accuracy. However, we found very contrasting results: some classes are particularly well detected, while others perform less favorably. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/1ee3b2f8-7a5e-4f30-8175-72359ebe4ce6" width="90%">
  <br>
  <em>Figure 3: An example of a good error localization on 'Tile' class </em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8b135c0c-6570-41ca-b3cd-1d0f252980b5" width="90%">
  <br>
  <em>Figure 4: An example of a not so good error localization on 'Transistor' class </em>
</p>

## 🤝 What We Bring Home

This project has provided valuable insights and learning experiences that we plan to build upon in future iterations. Based on our experimental results and analysis, we have identified several key takeaways and areas for improvement:

**Technical Insights:**
- The multi-scale feature pyramid approach shows promise but requires further optimization for consistent performance across all object categories
- Block-wise masking strategy demonstrates improved spatial understanding compared to random masking, though fine-tuning the block size parameters could yield better results
- Dynamic thresholding based on validation performance proves more effective than static threshold approaches, suggesting the importance of adaptive decision boundaries

**Performance Analysis:**
- Class-specific performance variations indicate that certain object types benefit more from our architectural choices than others
- The plateau effect observed around 80 epochs suggests potential for more sophisticated learning rate scheduling or regularization techniques
- The hybrid positional embedding approach shows potential but may need category-specific tuning

**Future Directions:**
- We plan to investigate class-specific architectural modifications (if we have extra time) to address the performance inconsistencies
- Extended training with more sophisticated optimization strategies could potentially push beyond the current performance plateau
- Integration of additional loss functions and regularization techniques may improve generalization across diverse anomaly types

Feel free to reach out to us with questions, ideas, and suggestions as we continue to refine and improve this project based on these learnings.

## 📄 License

This project is distributed under the MIT License.
