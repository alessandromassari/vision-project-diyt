# Advanced Masked Autoencoder for Industrial Anomaly Detection

This project introduces an advanced **Masked Autoencoder (MAE)** architecture, built upon a Vision Transformer (ViT), for **unsupervised industrial anomaly detection and localization**. The model is designed to learn the normal appearance of an object from defect-free images and then identify anomalies by detecting deviations in its reconstruction.

The core idea is that the model, once an expert on "normal" data, will fail to accurately reconstruct unseen defects. This reconstruction error serves as a powerful signal for both detecting and pinpointing anomalies. The model has been validated on the standard **MVTec AD** and **BTAD** datasets.

<p align="center">
  <img src="https://github.com/user-attachments/assets/49f1e3f7-eaa7-4694-85a7-291bc1258ecc" width="80%">
</p>

---

## ✨ Key Features

The vision that guides our work was to have a fast and light model trying to assure at the same time good and comparable performance, sort of 
This model enhances the standard MAE framework with several innovative features tailored for high-performance anomaly detection: (spoiler: don't expect sota performance)

* **Multi-Scale Feature Pyramid**: Unlike a standard MAE which only uses the final encoder output, this model extracts features from **intermediate encoder layers**. A `FeatureAggregationModule` then combines them to create a richer, multi-scale representation that captures both low-level textures and high-level semantics.

* **Block-Wise Masking Strategy**: Instead of masking random, sparse patches, this model employs a block-wise masking strategy. This forces the model to learn to reconstruct larger, contiguous regions, thereby improving its understanding of spatial context and long-range dependencies.

* **Hybrid Positional Embeddings**: The model leverages both **learnable** and **sinusoidal** positional embeddings. This hybrid approach combines the data-adaptive nature of learnable embeddings with the strong geometric foundation of fixed sinusoidal embeddings.

* **Dynamic Normalization (DyT)**: A custom `DyT` normalization layer is used in place of the standard `LayerNorm`. It introduces a learnable `alpha` parameter within a `tanh` activation, offering more flexible and dynamic normalization.

* **Lightweight, Attentive Decoder**: Adhering to the MAE philosophy, the decoder is significantly shallower than the encoder (`depth_dec=2`). It is architecturally prepared to receive the aggregated features from the encoder pyramid, allowing for a more informed and context-aware reconstruction process.

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
  <img src="[INSERT PATH TO YOUR RECONSTRUCTION EXAMPLES HERE]" width="90%">
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
  If you are using Google Colab, Jupyter or another notebook rembember to import the orginal MVtecAD a BTAD dataset.

### Training & Fine-Tuning

The process is divided in many section, first cames the data augmentation and then into a pre-training and a fine-tuning stage. We noted that after about 80 epochs a plateau is reached and loss starts to decrease low. Anyway if you have time and want better performance we suggest to try with more epochs, especially in training (200-300). 

1.  **Pre-training**:

  In this stage is used a total loss calculated as the weighted sum of the reconstruction loss and   ssim - Structural Similarity Index Measure loss. 

2.  **Fine-tuning & Validation**:

   Here we have defined as reconstruction loss the MSE on image from patches reconstruction. 
   In this stage we also evaluate the still progressing model on the validation set in order to
   get what is happening under the wood: is our model improving with fine-tuning or not? This is 
   the right place for this question: **F1, AUC, ROC** curve and **AUPRO** are the metrics used   
   to validate the process. Especially AUC to decide when to stop the FT and save our 'best' model.

   Validation set is also fundamental to define dinamically, according to the fine-tuning evolution, a threshold
   to separete 'good' samples from anomalies. We tried also to define a static threshold based on train good 
   samples distribution but according to our experiments a dinamically updated thersold wich moves led by validation
   results could give a slightly discrimination between good and anomaly samples.  
   
## 📊 Results

The model achieves good performance on the MVTec AD and BTAD datasets. Key evaluation metrics include image-level **AUC** and **F1-Score**, as well as the pixel-level **AUPRO** for localization accuracy. 

<p align="center">
  <img src="[INSERT PATH TO YOUR ROC & HISTOGRAM PLOTS HERE]" width="90%">
</p>

| Class      | Image AUC | F1-Score | AUPRO  |
| :--------- | :-------: | :------: | :----: |
| `screw`    |   0.98    |   0.95   |  0.96  |
| `bottle`   |    ...    |    ...   |   ...  |
| `cable`    |    ...    |    ...   |   ...  |
---

## 🤝 Contributing

Contributions are always welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more details.
