# 🤖 Module IA - Détection des Maladies de Tomates

Module d'intelligence artificielle pour la détection automatique des maladies des tomates via analyse d'images. Utilise TensorFlow/Keras avec MobileNetV2 pour classifier 10 conditions différentes.

---

## 📋 Table des Matières

- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Structure du Projet](#-structure-du-projet)
- [Entraînement du Modèle](#-entraînement-du-modèle)
- [API Flask](#-api-flask)
- [Variables d'Environnement](#-variables-denvironnement)
- [Utilisation](#-utilisation)
- [Classes Détectées](#-classes-détectées)
- [Exemples de Code](#-exemples-de-code)
- [Déploiement](#-déploiement)
- [Tests](#-tests)
- [Troubleshooting](#-troubleshooting)
- [Contribution](#-contribution)
- [License](#-license)

---

## ✨ Fonctionnalités

- ✅ Détection de **10 conditions** de tomates (9 maladies + sain)
- 🧠 Modèle basé sur **MobileNetV2** (transfert d'apprentissage)
- 🔄 **Augmentation de données** pour robustesse
- 📡 **API REST Flask** pour intégration ESP32/Backend
- 🎯 Prédictions avec **niveau de confiance** et **sévérité**
- 💡 **Recommandations** de traitement automatiques
- 🌊 Détermination du **besoin d'arrosage**
- 📤 Envoi automatique au **backend Node.js**
- 🗑️ Traitement **sans stockage** (images supprimées après analyse)
- 📊 Mode **DEMO** sans modèle (pour tests)
- 🔁 Support analyse **batch** (plusieurs images)

---

## 🏗️ Architecture
```
ESP32-CAM → [Photo] → Module IA (Flask:5001) → [Analyse] → Backend Node.js (Express:5000) → MongoDB
                            ↓
                    Résultats + Recommandations
```

### Flux de Traitement

1. **Capture** : ESP32-CAM prend une photo
2. **Envoi** : POST `/predict` avec `image` + `capteurId` + `userId`
3. **Prétraitement** : Redimensionnement (224x224), normalisation RGB
4. **Prédiction** : MobileNetV2 → Classe + Confiance
5. **Enrichissement** : Sévérité, recommandations, arrosage
6. **Transmission** : Envoi au backend via API REST
7. **Nettoyage** : Suppression automatique de l'image

---

## 📦 Prérequis

### Système

- **Python** : 3.9 - 3.11 (recommandé : 3.10)
- **RAM** : 4 GB minimum (8 GB recommandé pour entraînement)
- **Stockage** : 2 GB pour modèle + dépendances
- **OS** : Windows / Linux / macOS

### Dataset

- Structure : `data/tomato/{classe1, classe2, ...}`
- Format : JPG/PNG
- Résolution : 224x224 ou supérieure
- Volume : 500-1000 images par classe minimum

---

## 🚀 Installation

### 1. Cloner le Dépôt
```bash
git clone https://github.com/doosr/module_ia_agro.git
cd module_ia
```

### 2. Créer un Environnement Virtuel
```bash
# Créer l'environnement
python -m venv venv

# Activer l'environnement
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Installer les Dépendances

Créer un fichier **`requirements.txt`** :
```txt
Flask==3.0.0
flask-cors==4.0.0
tensorflow==2.15.0
keras==2.15.0
numpy==1.24.3
Pillow==10.1.0
opencv-python==4.8.1.78
scikit-learn==1.3.2
requests==2.31.0
python-dotenv==1.0.0
```

Installer :
```bash
pip install -r requirements.txt
```

### 4. Préparer le Dataset

Structure attendue :
```
module_ia/
├── data/
   └── tomato/
       ├── Tomato_bacterial_spot/
       │   ├── image001.jpg
       │   ├── image002.jpg
       │   └── ...
       ├── Tomato_early_blight/
       ├── Tomato_healthy/
       ├── Tomato_late_blight/
       ├── Tomato_leaf_mold/
       ├── Tomato_septoria_leaf_spot/
       ├──Tomato_spider_mites_two-spotted_spider_m/
       ├── Tomato_target_spot/
       ├── Tomato_mosaic_virus/
       └── Tomato_yellow_leaf_curl_virus/

```

**Sources de dataset** :
- [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)

---

## 📁 Structure du Projet
```
module_ia/
├── app.py                          # API Flask principale
├── train.py                  # Script d'entraînement
├── requirements.txt                # Dépendances Python
├── .env                            # Variables d'environnement
├── .env.example                    # Template de configuration
├── README.md                       #                      
├── data/
│   └── tomato/                     # Dataset (10 classes)
│       ├── Tomato_bacterial_spot/
│       ├── Tomato_early_blight/
│       ├── Tomato_healthy/
│       ├── Tomato_late_blight/
│       ├── Tomato_leaf_mold/
│       ├── Tomato_septoria_leaf_spot/
│       ├── Tomato_spider_mites_two-spotted_spider_mite/
│       ├── Tomato_target_spot/
│       ├── Tomato_mosaic_virus/
│       └── Tomato_yellow_leaf_curl_virus/
├── models/
   └── tomato_disease_model.h5     # Modèle 
```

---

## 🧠 Entraînement du Modèle

### Script `train.py`

Créer le fichier **`train.py`** :
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
DATASET_PATH = './data/tomato'
MODEL_SAVE_PATH = './models/tomato_disease_model.h5'

CLASSES = [
    "Tomato_bacterial_spot",
    "Tomato_early_blight",
    "Tomato_healthy",
    "Tomato_late_blight",
    "Tomato_leaf_mold",
    "Tomato_septoria_leaf_spot",
    "Tomato_spider_mites_two-spotted_spider_mite",
    "Tomato_target_spot",
    "Tomato_mosaic_virus",
    "Tomato_yellow_leaf_curl_virus"
]

# ═══════════════════════════════════════════════════════════
# CRÉATION DU MODÈLE
# ═══════════════════════════════════════════════════════════

def create_model(num_classes):
    """
    Créer le modèle CNN avec MobileNetV2
    Transfer Learning + Fine-tuning
    """
    print("\n🏗️ Création du modèle...")
    
    # Base MobileNetV2 pré-entraînée sur ImageNet
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Geler les couches de base (transfer learning)
    base_model.trainable = False
    
    # Ajouter les couches de classification
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print(f"✅ Modèle créé avec {num_classes} classes")
    return model

# ═══════════════════════════════════════════════════════════
# PRÉPARATION DES DONNÉES
# ═══════════════════════════════════════════════════════════

def prepare_data():
    """
    Préparer les générateurs de données avec augmentation
    """
    print("\n📊 Préparation des données...")
    
    # Augmentation de données pour l'entraînement
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalisation
        rotation_range=20,            # Rotation aléatoire ±20°
        width_shift_range=0.2,        # Translation horizontale
        height_shift_range=0.2,       # Translation verticale
        horizontal_flip=True,         # Flip horizontal
        zoom_range=0.2,               # Zoom aléatoire
        brightness_range=[0.8, 1.2],  # Variation luminosité
        fill_mode='nearest',          # Remplissage pixels
        validation_split=0.2          # 80% train / 20% val
    )
    
    # Validation sans augmentation (seulement normalisation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Générateur d'entraînement
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Générateur de validation
    val_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"✅ Données chargées:")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Classes: {train_generator.num_classes}")
    
    return train_generator, val_generator

# ═══════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════

def train_model():
    """
    Entraîner le modèle complet
    """
    print("\n" + "="*60)
    print("🚀 DÉBUT ENTRAÎNEMENT MODÈLE TOMATE")
    print("="*60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Classes: {len(CLASSES)}")
    print(f"📐 Image size: {IMG_SIZE}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")
    print("="*60)
    
    # Préparer les données
    train_gen, val_gen = prepare_data()
    
    # Créer le modèle
    model = create_model(len(CLASSES))
    
    # Compiler le modèle
    print("\n⚙️ Compilation du modèle...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Afficher l'architecture
    model.summary()
    
    # Callbacks
    callbacks = [
        # Arrêt anticipé si pas d'amélioration
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Réduction du learning rate
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # Sauvegarde du meilleur modèle
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # TensorBoard (optionnel)
        keras.callbacks.TensorBoard(
            log_dir=f'./logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    # Entraînement
    print("\n🏃 Début de l'entraînement...")
    print("-"*60)
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Résultats finaux
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    print(f"📊 Précision finale (train): {history.history['accuracy'][-1]*100:.2f}%")
    print(f"📊 Précision finale (val): {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"📉 Perte finale (train): {history.history['loss'][-1]:.4f}")
    print(f"📉 Perte finale (val): {history.history['val_loss'][-1]:.4f}")
    print(f"💾 Modèle sauvegardé: {MODEL_SAVE_PATH}")
    print("="*60 + "\n")
    
    return model, history

# ═══════════════════════════════════════════════════════════
# ÉVALUATION (OPTIONNEL)
# ═══════════════════════════════════════════════════════════

def evaluate_model(model, val_gen):
    """
    Évaluer le modèle sur l'ensemble de validation
    """
    print("\n📈 Évaluation du modèle...")
    
    loss, accuracy = model.evaluate(val_gen, verbose=1)
    
    print(f"\n📊 Résultats sur validation:")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Loss: {loss:.4f}")
    
    return accuracy, loss

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Vérifier que le dataset existe
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERREUR: Dataset introuvable à {DATASET_PATH}")
        print("📁 Veuillez placer le dataset dans data/tomato/")
        exit(1)
    
    # Créer le dossier models si nécessaire
    os.makedirs('./models', exist_ok=True)
    
    # Entraîner
    try:
        model, history = train_model()
        
        # Évaluation optionnelle
        # _, val_gen = prepare_data()
        # evaluate_model(model, val_gen)
        
    except Exception as e:
        print(f"\n❌ ERREUR pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
```

### Lancer l'Entraînement
```bash
# Activer l'environnement virtuel
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Lancer l'entraînement
python train_model.py
```

**Résultat attendu** :
```
============================================================
🚀 DÉBUT ENTRAÎNEMENT MODÈLE TOMATE
============================================================
📅 Date: 2025-11-14 14:30:00
🎯 Classes: 10
📐 Image size: (224, 224)
📦 Batch size: 32
🔄 Epochs: 50
============================================================

📊 Préparation des données...
Found 8000 images belonging to 10 classes.
Found 2000 images belonging to 10 classes.

🏗️ Création du modèle...
✅ Modèle créé avec 10 classes

⚙️ Compilation du modèle...
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
mobilenetv2 (Functional)    (None, 7, 7, 1280)        2257984   
global_average_pooling2d    (None, 1280)              0         
dropout (Dropout)           (None, 1280)              0         
dense (Dense)               (None, 256)               327936    
dropout_1 (Dropout)         (None, 256)               0         
dense_1 (Dense)             (None, 10)                2570      
=================================================================
Total params: 2,588,490
Trainable params: 330,506
Non-trainable params: 2,257,984
_________________________________________________________________

🏃 Début de l'entraînement...
------------------------------------------------------------
Epoch 1/50
250/250 [==============================] - 120s 480ms/step
loss: 1.2345 - accuracy: 0.6234 - val_loss: 0.8901 - val_accuracy: 0.7456

...

Epoch 35/50
250/250 [==============================] - 115s 460ms/step
loss: 0.1234 - accuracy: 0.9567 - val_loss: 0.2345 - val_accuracy: 0.9234

============================================================
✅ ENTRAÎNEMENT TERMINÉ
============================================================
📊 Précision finale (train): 95.67%
📊 Précision finale (val): 92.34%
📉 Perte finale (train): 0.1234
📉 Perte finale (val): 0.2345
💾 Modèle sauvegardé: ./models/tomato_disease_model.h5
============================================================
```

---

## 🌐 API Flask

### Fichier Principal `app.py`

Le fichier `app.py` fourni dans le document contient l'API complète.

### Démarrage du Serveur
```bash
# Activer l'environnement
source venv/bin/activate

# Lancer le serveur
python app.py
```

**Sortie attendue** :
```
============================================================
🤖 Service IA - Détection Maladies des Tomates
============================================================
📍 URL: http://0.0.0.0:5001
🔗 Backend: http://localhost:5000
🔑 API Key: your-secre...
📦 Modèle: ✅ Chargé
📤 Envoi backend: ✅ Activé
🌱 Classes supportées: 10
💡 Architecture: ESP32 → IA → Backend (sans stockage)
============================================================

📋 Routes disponibles:
   GET  /health           - État du service
   POST /predict          - Analyser une image
   POST /predict-batch    - Analyser plusieurs images
   GET  /stats            - Statistiques
   POST /reload-model     - Recharger le modèle
   GET  /test-backend     - Tester connexion backend

💡 Notes:
   • Les images sont supprimées après analyse
   • Les résultats sont envoyés au backend Node.js
   • Backup local disponible sur ESP32 (carte SD)
============================================================
```

### Routes Disponibles

#### 1. 🏥 GET `/health` - État du Service

Vérification de l'état du service.

**Requête** :
```bash
curl http://localhost:5001/health
```

**Réponse** :
```json
{
  "status": "online",
  "service": "Plant Disease Detection AI",
  "version": "2.0.0",
  "model_loaded": true,
  "model_path": "models/tomato_disease_model.h5",
  "backend_url": "http://localhost:5000",
  "backend_enabled": true,
  "supported_classes": 10,
  "timestamp": "2025-11-19T14:23:45.123456"
}
```

---

#### 2. 📸 POST `/predict` - Analyser une Image

Analyse d'une image unique avec envoi automatique au backend.

**Requête** :
```bash
curl -X POST http://localhost:5001/predict \
  -F "image=@/path/to/tomato_leaf.jpg" \
  -F "capteurId=sensor_001" \
  -F "userId=user_12345"
```

**Parameters** :
- `image` (file, **required**) : Image à analyser (JPG/PNG, max 10MB)
- `capteurId` (string, optional) : Identifiant du capteur ESP32
- `userId` (string, optional) : Identifiant de l'utilisateur

**Réponse** :
```json
{
  "success": true,
  "maladie": "Tomato_early_blight",
  "confiance": 0.9234,
  "recommandations": [
    "Retirer les feuilles touchées",
    "Traiter avec fongicide préventif",
    "Améliorer la circulation d'air",
    "Pailler le sol pour éviter les éclaboussures"
  ],
  "arroser": true,
  "prediction": "Tomato_early_blight",
  "predictionFr": "Mildiou précoce",
  "confidence": 0.9234,
  "diseaseDetected": true,
  "severity": "high",
  "shouldWater": true,
  "timestamp": "2025-11-19T14:23:45.123456",
  "modelUsed": "tomato_disease_model",
  "backend_sent": true
}
```

**Codes de Retour** :
- `200` : Analyse réussie
- `400` : Image manquante ou trop large
- `500` : Erreur serveur

---

#### 3. 📦 POST `/predict-batch` - Analyser Plusieurs Images

Analyse de plusieurs images en une seule requête.

**Requête** :
```bash
curl -X POST http://localhost:5001/predict-batch \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg" \
  -F "capteurId=sensor_001"
```

**Réponse** :
```json
{
  "success": true,
  "total": 3,
  "success_count": 3,
  "results": [
    {
      "success": true,
      "prediction": "Tomato_healthy",
      "confidence": 0.9876,
      "diseaseDetected": false,
      "severity": "none",
      "backend_sent": true
    },
    {
      "success": true,
      "prediction": "Tomato_late_blight",
      "confidence": 0.8765,
      "diseaseDetected": true,
      "severity": "medium",
      "backend_sent": true
    },
    {
      "success": false,
      "image_index": 2,
      "error": "Invalid image format"
    }
  ]
}
```

---

#### 4. 📊 GET `/stats` - Statistiques

Informations sur le modèle et la configuration.

**Requête** :
```bash
curl http://localhost:5001/stats
```

**Réponse** :
```json
{
  "model_loaded": true,
  "model_path": "models/tomato_disease_model.h5",
  "backend_url": "http://localhost:5000",
  "backend_enabled": true,
  "supported_classes": [
    "Tomato_bacterial_spot",
    "Tomato_early_blight",
    "Tomato_healthy",
    "Tomato_late_blight",
    "Tomato_leaf_mold",
    "Tomato_septoria_leaf_spot",
    "Tomato_spider_mites_two-spotted_spider_mite",
    "Tomato_target_spot",
    "Tomato_mosaic_virus",
    "Tomato_yellow_leaf_curl_virus"
  ],
  "total_classes": 10
}
```

---

#### 5. 🔄 POST `/reload-model` - Recharger le Modèle

Recharge le modèle après une mise à jour.

**Requête** :
```bash
curl -X POST http://localhost:5001/reload-model
```

**Réponse** :
```json
{
  "success": true,
  "model_loaded": true,
  "message": "Modèle rechargé avec succès"
}
```

---

#### 6. 🔗 GET `/test-backend` - Tester la Connexion Backend

Vérifie la connectivité avec le backend Node.js.

**Requête** :
```bash
curl http://localhost:5001/test-backend
```

**Réponse** :
```json
{
  "success": true,
  "backend_url": "http://localhost:5000",
  "status_code": 200,
  "response": {
    "status": "online",
    "service": "Smart Agriculture Backend",
    "version": "1.0.0"
  }
}
```

---

## ⚙️ Variables d'Environnement

Créer un fichier **`.env`** à la racine :
```env
# ═══════════════════════════════════════════════════════════
# CONFIGURATION MODULE IA
# ═══════════════════════════════════════════════════════════

# Backend Node.js
BACKEND_URL=http://localhost:5000
BACKEND_API_KEY=your-secret-key-changez-moi
SEND_TO_BACKEND=true

# Modèle IA
MODEL_PATH=models/tomato_disease_model.h5

# Debug mode
DEBUG=false

# Serveur Flask
FLASK_HOST=0.0.0.0
FLASK_PORT=5001
```

### Template `.env.example`
```env
# Backend Configuration
BACKEND_URL=http://localhost:5000
BACKEND_API_KEY=changez-moi-en-production
SEND_TO_BACKEND=true

# Model Configuration
MODEL_PATH=models/tomato_disease_model.h5

# Debug
DEBUG=false
```

---

## 🎯 Classes Détectées

| # | Classe | Nom Français | Sévérité | Arrosage | Description |
|---|--------|--------------|----------|----------|-------------|
| 1 | `Tomato_healthy` | Sain | Aucune | ✅ | Plante en bonne santé |
| 2 | `Tomato_bacterial_spot` | Tache bactérienne | Moyenne-Haute | ✅ | Bactérie *Xanthomonas* |
| 3 | `Tomato_early_blight` | Mildiou précoce | Moyenne | ✅ | Champignon *Alternaria* |
| 4 | `Tomato_late_blight` | Mildiou tardif | Haute | ✅ | Oomycète *Phytophthora* |
| 5 | `Tomato_leaf_mold` | Moisissure feuilles | Moyenne | ❌ | Champignon *Passalora* |
| 6 | `Tomato_septoria_leaf_spot` | Tache septorienne | Moyenne | ❌ | Champignon *Septoria* |
| 7 | `Tomato_spider_mites` | Acariens | Faible-Moyenne | ❌ | Tétranyque tisserand |
| 8 | `Tomato_target_spot` | Tache cible | Moyenne | ❌ | Champignon *Corynespora* |
| 9 | `Tomato_mosaic_virus` | Virus mosaïque | Haute | ❌ | Virus ToMV |
| 10 | `Tomato_yellow_leaf_curl_virus` | Virus enroulement jaune | Haute | ❌ | Virus TYLCV |

### Recommandations par Maladie

#### 🌱 Tomato Healthy (Sain)
```
✅ Plante en bonne santé
- Continuer les soins habituels
- Surveiller régulièrement les feuilles
- Maintenir un bon drainage du sol
```

#### 🦠 Tomato Bacterial Spot (Tache bactérienne)
```
🔴 Sévérité: Moyenne-Haute
- Retirer immédiatement les feuilles infectées
- Appliquer un fongicide à base de cuivre
- Éviter l'arrosage par aspersion
- Nettoyer et désinfecter les outils de taille
- Espacer les plants (circulation d'air)
```

#### 🍂 Tomato Early Blight (Mildiou précoce)
```
🟠 Sévérité: Moyenne

Retirer les feuilles touchées (partir du bas)
Traiter avec fongicide préventif (chlorothalonil)
Améliorer la circulation d'air entre les plants
Pailler le sol pour éviter les éclaboussures
Arroser à la base des plants uniquement
```
#### 🍃 Tomato Late Blight (Mildiou tardif)
```
🟡 Sévérité: Moyenne

Isoler immédiatement la plante infectée
Appliquer un fongicide systémique (mancozèbe)
Détruire les parties gravement infectées
Éviter l'humidité excessive (>90%)
Surveiller les plants voisins quotidiennement
Ne pas composter les résidus infectés
```
#### 🌫️ Tomato Leaf Mold (Moisissure des feuilles)
```
🟠 Sévérité: Moyenne

Supprimer les feuilles malades (brûler)
Traitement fongicide préventif régulier
Éviter de mouiller le feuillage
Rotation des cultures (3-4 ans)
Paillage pour limiter les éclaboussures
```
#### 🕷️ Tomato Spider Mites (Acariens)
```
🔴 Sévérité: Haute - URGENCE
```
Isoler immédiatement la plante infectée
Appliquer un fongicide systémique (mancozèbe)
Détruire les parties gravement infectées
Éviter l'humidité excessive (>90%)
Surveiller les plants voisins quotidiennement
Ne pas composter les résidus infectés
```
#### 🌫️ Tomato Leaf Mold (Moisissure des feuilles)
```
🟡 Sévérité: Moyenne

Améliorer la ventilation (serre/tunnel)
Réduire l'humidité ambiante (<85%)
Espacer davantage les plants
Tailler les feuilles basses pour aérer
Éviter l'arrosage le soir
```	
#### ⚫ Tomato Septoria Leaf Spot (Tache septorienne)

🟠 Sévérité: Moyenne

Supprimer les feuilles malades (brûler)
Traitement fongicide préventif régulier
Éviter de mouiller le feuillage
Rotation des cultures (3-4 ans)
Paillage pour limiter les éclaboussures

#### 🕷️ Tomato Spider Mites (Acariens)

🟡 Sévérité: Faible-Moyenne

Pulvériser insecticide acaricide
Maintenir une humidité élevée (>60%)
Utiliser des acariens prédateurs naturels
Nettoyer régulièrement les feuilles
Isoler les plants infestés

🟠 Sévérité: Moyenne

Enlever les feuilles infectées rapidement
Appliquer fongicide à large spectre
Améliorer le drainage du sol
Espacer les plantations (50-70cm)
Éviter l'irrigation par aspersion
#### 🎯 Tomato Target Spot (Tache cible)
🟠 Sévérité: Moyenne

Enlever les feuilles infectées rapidement
Appliquer fongicide à large spectre
Améliorer le drainage du sol
Espacer les plantations (50-70cm)
Éviter l'irrigation par aspersion

#### 🦠 Tomato Mosaic Virus (Virus de la mosaïque)

🔴 Sévérité: Haute - Viral

Isoler immédiatement la plante
Détruire les plants gravement atteints
Désinfecter tous les outils (eau de javel 10%)
Contrôler les insectes vecteurs (pucerons)
Se laver les mains avant manipulation
Utiliser des variétés résistantes

#### 🟡 Tomato Yellow Leaf Curl Virus (Virus enroulement jaune)

🔴 Sévérité: Haute - Viral

Isoler la plante infectée
Contrôler les aleurodes (mouches blanches)
Utiliser des filets anti-insectes (maille <0.8mm)
Détruire les plants trop atteints
Éliminer les mauvaises herbes hôtes
Planter des variétés résistantes (gène Ty) 
