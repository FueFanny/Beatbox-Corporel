# Système Interactif de Mouvement Musical pour Rééducation en Fauteuil Roulant

## Lancement du projet sur Raspberry Pi

Pour activer le projet final depuis le Raspberry Pi dans son terminal, 3 étapes :

```bash
cd /home/alice/PROJET_FINAL
source tf-env39/bin/activate
python gui.py
````

---

## Présentation du projet

Ce projet est un système interactif basé sur le mouvement, conçu pour encourager l’activité physique chez les personnes en fauteuil roulant.

Le système transforme les mouvements du corps en sons.

L’objectif est d’encourager l’utilisateur à :

* bouger le haut du corps
* se pencher
* tendre les bras
* se rapprocher du sol
* interagir physiquement via la musique pour une fonction de rééducation

---

## Concept global

Le projet combine :

* un Raspberry Pi
* une webcam
* un capteur IMU BNO055
* deux capteurs ultrasoniques
* de la détection de pose avec MoveNet
* de l’audio interactif
* une visualisation vidéo artistique des mouvements
* une génération de graphe d’accélération et d’orientation IMU

---

## Logique d’interaction

Lorsqu’une personne se penche correctement et rapproche sa main du sol, des sons sont joués.

* Plus la main est proche du sol → plus le volume augmente
* Plus le mouvement est intense → plus l’interaction musicale est forte

Les mouvements sont également enregistrés en :

* images
* fichiers CSV
* vidéo générée automatiquement

---

## Matériel nécessaire

### Composants

* Raspberry Pi 4
* Webcam USB
* Capteur IMU Adafruit BNO055
* 2 capteurs ultrasoniques HC-SR04
* Haut-parleurs ou casque audio
* Carte microSD (≥ 16GB)
* Connexion internet

---

## Branchement des capteurs

### IMU BNO055 (I2C)

| BNO055 | Raspberry Pi |
| ------ | ------------ |
| VIN    | 3.3V         |
| GND    | GND          |
| SDA    | SDA          |
| SCL    | SCL          |

Activer I2C :

```bash
sudo raspi-config
```

Interface Options -> I2C -> Enable

---

### HC-SR04 gauche

| HC-SR04 | Raspberry Pi |
| ------- | ------------ |
| VCC     | 5V           |
| GND     | GND          |
| TRIG    | GPIO23       |
| ECHO    | GPIO24       |

---

### HC-SR04 droite

| HC-SR04 | Raspberry Pi |
| ------- | ------------ |
| VCC     | 5V           |
| GND     | GND          |
| TRIG    | GPIO27       |
| ECHO    | GPIO17       |

---

IMPORTANT

Le pin ECHO du HC-SR04 sort du 5V.

Il faut utiliser un diviseur de tension avant de connecter le GPIO au Raspberry Pi.

---

## Structure du projet

```
PROJECT BeatBoxCorp/
│
├── musicam.py
├── video.py
├── graph.py
├── gui.py
├── motion_log.csv
│
├── captures/
│
├── sounds/
│   ├── soft/
│   │   └── kick.wav
│   ├── medium/
│   │   └── kick.wav
│   ├── hard/
│   │   └── kick.wav
│   └── false/
│       └── false.wav
│
├── movenet_lightning.tflite
│
└── outputs/
    └── motion_layers.mp4
```

---

## Installation

### Mise à jour système

```bash
sudo apt update
sudo apt upgrade -y
```

---

### Installation des paquets système

```bash
sudo apt install -y \
python3-pip \
python3-opencv \
python3-pygame \
libatlas-base-dev \
portaudio19-dev
```

---

### Bibliothèques Python

```bash
pip install numpy pandas gpiozero adafruit-circuitpython-bno055
pip install tflite-runtime
```

---

### Si problème TensorFlow Lite (ARM)

```bash
pip install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl
```

---

### Télécharger MoveNet

```bash
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/singlepose/lightning/4.tflite -O movenet_lightning.tflite
```

---

## Système sonore

Les sons doivent être placés dans :

```
sounds/soft/kick.wav
sounds/medium/kick.wav
sounds/hard/kick.wav
sounds/false/false.wav
```

Si les noms changent, modifier `musicam.py` à environ la ligne 80.

---

## Lancement du projet

### Interface graphique

```bash
python3 gui.py
```

Permet :

* lancer / arrêter le système
* générer la vidéo
* ouvrir la vidéo
* fermer le programme

---

### Système interactif seul

```bash
python3 musicam.py
```

Inclut :

* webcam
* MoveNet
* IMU
* capteurs ultrasoniques
* sons interactifs
* capture d’images
* enregistrement CSV

---

### Génération vidéo

```bash
python3 video.py
```

Génère :

```
outputs/motion_layers.mp4
```

Contient :

* couches de mouvements
* effets visuels
* transformation selon accélération
* rotation selon inclinaison
* données de distance
* sons joués

---

### Génération du graphe CSV

```bash
python3 graph.py
```

---

## Fichier CSV

Exemple :

```
time_since_start,image_file,sound_played,sound_volume,distance_cm_droite,distance_cm_gauche
1.52,motion_1.52.jpg,kick_soft,0.72,18.4,50.6
```

Contient :

* timestamp
* image associée
* son joué
* volume
* distances capteurs
* données IMU

---

## Fonctionnement du système

### Détection du corps

* MoveNet détecte les keypoints en temps réel
* analyse du mouvement via soustraction de fond
* capture uniquement les zones en mouvement

---

### IMU BNO055

Mesure :

* inclinaison du corps
* accélération
* orientation

Utilisation :

* roll → gauche/droite
* accélération → intensité musicale

---

### Capteurs ultrasoniques

Mesurent la distance main ↔ sol

Impact :

* distance faible → son plus fort
* mouvement rapide → intensité augmentée

---

## Logique sonore

| Intensité | Son         |
| --------- | ----------- |
| Faible    | Kick soft   |
| Moyenne   | Kick medium |
| Forte     | Kick hard   |

Si mouvement invalide -> son "false". Ici, un mouvement invalide n'est que lorsque la direction de l'inclinaison est inverse au capteur actif.

---

## Dépannage

### Webcam

```bash
ls /dev/video*
libcamera-hello
```

---

### IMU

```bash
i2cdetect -y 1
```

Adresse attendue : 0x28 ou 0x29

---

### GPIO

```bash
sudo usermod -aG gpio $USER
sudo reboot
```

---

## Possibilités futures

* deux capteurs ultrasoniques indépendants main gauche/droite
* intégration MIDI
* connexion Ableton Live
* visualisations génératives en temps réel
* système de score de mouvement
* installation artistique immersive

---

## Technologies utilisées

* Python
* OpenCV
* TensorFlow Lite
* Raspberry Pi
* MoveNet
* GPIO
* IMU BNO055
* audio interactif

---

## Licence

Projet éducatif et de recherche.

```
