# vision-project-diyt
# Masked Autoencoder Avanzato per Anomaly Detection 

[![Licenza](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md) 
Questo progetto presenta un'implementazione avanzata di un **Masked Autoencoder (MAE)** basato su Vision Transformer (ViT), progettato specificamente per il task di **rilevamento di anomalie non supervisionato** in contesti industriali. Il modello è stato testato e validato sui dataset di riferimento **MVTec AD** e **BTAD**.

L'obiettivo è insegnare al modello a "comprendere" l'aspetto di un oggetto normale. Quando viene presentata un'immagine con un difetto, il modello non riesce a ricostruirla correttamente, permettendoci di identificare e localizzare l'anomalia attraverso l'errore di ricostruzione.

<p align="center">
  <img src="[INSERISCI QUI IL PERCORSO A UN DIAGRAMMA DELL'ARCHITETTURA]" width="80%">
</p>

---

## ✨ Caratteristiche Principali

Questo modello si distingue dalle implementazioni standard di MAE per diverse caratteristiche innovative pensate per migliorare le performance nell'anomaly detection:

* **Aggregazione di Feature Multi-Scala**: A differenza di un MAE standard che usa solo l'output finale dell'encoder, questo modello estrae feature da **strati intermedi** dell'encoder. Un `FeatureAggregationModule` le combina per creare una rappresentazione più ricca, che cattura sia dettagli di basso livello che informazioni semantiche di alto livello.

* **Strategia di Masking a Blocchi (Block-Wise Masking)**: Invece di mascherare patch casuali e sparse, viene applicato un mascheramento a blocchi contigui. Questo costringe il modello a imparare a ricostruire regioni più ampie, migliorando la comprensione del contesto spaziale.

* **Embedding Posizionale Ibrido**: Combina embedding posizionali **apprendibili** (che si adattano ai dati) e **sinusoidali** (che forniscono una solida base geometrica), sfruttando i vantaggi di entrambi gli approcci.

* **Normalizzazione Dinamica (DyT)**: Utilizza un layer di normalizzazione personalizzato, `DyT`, che introduce un parametro apprendibile `alpha` all'interno di una funzione `tanh`, offrendo maggiore flessibilità rispetto a un `LayerNorm` standard.

* **Decoder Leggero e Attento**: Seguendo la filosofia MAE, il decoder è significativamente più "leggero" dell'encoder (`depth_dec=2`). È progettato per utilizzare le feature aggregate dall'encoder, permettendo una ricostruzione più informata.

---

## 🏛️ Architettura del Modello

Il flusso di dati attraverso il modello è il seguente:

1.  **Input e Patching**: L'immagine di input viene divisa in patch non sovrapposte.
2.  **Embedding**: Ad ogni patch viene aggiunto un embedding posizionale ibrido (sinusoidale + apprendibile).
3.  **Masking**: Una porzione significativa delle patch viene mascherata utilizzando la strategia a blocchi (solo durante il training).
4.  **Encoder**: Le patch visibili vengono processate da un profondo encoder Transformer (`depth_enc=16`). Durante questo passaggio, gli output di strati intermedi vengono salvati.
5.  **Feature Aggregation**: Il `FeatureAggregationModule` prende gli output intermedi, li proietta in uno spazio comune e li combina in un'unica, ricca mappa di feature.
6.  **Decoder**: Un decoder Transformer leggero (`depth_dec=2`) riceve le patch codificate e le feature aggregate per ricostruire le patch originali (sia quelle visibili che quelle mascherate).
7.  **Output**: Il modello restituisce le patch dell'immagine ricostruita.

<p align="center">
  <img src="[INSERISCI QUI IL PERCORSO A ESEMPI DI RICOSTRUZIONE]" width="90%">
</p>

---

## 🚀 Come Iniziare

### Prerequisiti

* Python 3.8+
* PyTorch 1.10+
* torchvision
* scikit-learn
* matplotlib
* Pillow (PIL)

### Installazione

1.  Clona il repository:
    ```bash
    git clone [https://github.com/tuo-username/tuo-repo.git](https://github.com/tuo-username/tuo-repo.git)
    cd tuo-repo
    ```

2.  Installa le dipendenze:
    ```bash
    pip install -r requirements.txt
    ```

### Training e Fine-Tuning

Il processo si svolge in due fasi: pre-training e fine-tuning.

1.  **Pre-training**:
    ```bash
    python pretrain.py --dataset_path /percorso/al/tuo/dataset --class_name screw
    ```

2.  **Fine-tuning e Valutazione**:
    ```bash
    python finetune.py --pretrained_model_path /percorso/al/modello.pth --dataset_path /percorso/al/tuo/dataset
    ```

---

## 📊 Risultati

Il modello raggiunge performance competitive sui dataset MVTec AD e BTAD. Le metriche chiave utilizzate per la valutazione sono **AUC** (a livello di immagine), **F1 Score** e **AUPRO** (a livello di pixel per la localizzazione).

<p align="center">
  <img src="[INSERISCI QUI IL PERCORSO AI GRAFICI ROC E ISTOGRAMMA]" width="90%">
</p>

| Classe     | AUC Immagine | F1 Score | AUPRO  |
| :--------- | :----------: | :------: | :----: |
| `screw`    |     0.98     |   0.95   |  0.96  |
| `bottle`   |      ...     |    ...   |   ...  |
| `cable`    |      ...     |    ...   |   ...  |
---


## 📄 Licenza

Questo progetto è distribuito sotto la Licenza .... Vedi il file `LICENSE` per maggiori dettagli.
