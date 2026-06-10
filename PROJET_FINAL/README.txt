# Système Interactif de Mouvement Musical pour Rééducation en Fauteuil Roulant

pour activer le projet final depuis le raspberry pi dans son terminal, 3 étapes : 

$ cd /home/alice/PROJET_FINAL

$ source tf-env39/bin/activate

$ python gui.py

LE PROJET:

Ce projet est un système interactif basé sur le mouvement, conçu pour encourager l’activité physique chez les personnes en fauteuil roulant.
Le système transforme les mouvements du corps en sons.
L’objectif est d’encourager l’utilisateur à :

* bouger le haut du corps,
* se pencher,
* tendre les bras,
* se rapprocher du sol,
* interagir physiquement via de la musique pour une sorte de fonction de réhabilitation.

Le projet combine :

* un Raspberry Pi,
* une webcam,
* un capteur IMU BNO055,
* deux capteurs ultrasonique,
* de la détection de pose avec MoveNet,
* de l’audio interactif,
* une visualisation vidéo artistique des mouvements.
* nouvellement, une création de graphe illustrant l'accélération de l'IMU et son orientation sur le temps.

Lorsqu’une personne se penche correctement et approche sa main du sol, des sons sont joués.
Plus la main est proche du sol, plus le volume augmente, plus le mouvement devient musicalement intense.

Les mouvements sont également enregistrés sous forme d’images, de données CSV, et d’une vidéo générée automatiquement quand le script de création de vidéo est lancé.

--------------------

MATERIEL NECESSAIRE

Composants:

* Raspberry Pi 4
* Webcam USB
* Capteur IMU Adafruit BNO055
* 2 Capteurs ultrasonique HC-SR04
* Haut-parleurs ou casque audio
* Carte microSD pour le Raspberry Pi, au moins 16GB.
* Connexion internet pour l’installation

---

BRANCHEMENT DES CAPTEURS
BNO055(I2C)

| BNO055 | Raspberry Pi |
| ------ | ------------ |
| VIN    | 3.3V         |
| GND    | GND          |
| SDA    | SDA          |
| SCL    | SCL          |

Activer l’I2C :

$ sudo raspi-config

Puis dans l'interface:
Interface Options -> I2C -> Enable

---

HC-SR04 1 gauche

| HC-SR04 | Raspberry Pi |
| ------- | ------------ |
| VCC     | 5V           |
| GND     | GND          |
| TRIG    | GPIO23       |
| ECHO    | GPIO24       |

HC-SR04 2 droite

| HC-SR04 | Raspberry Pi |
| ------- | ------------ |
| VCC     | 5V           |
| GND     | GND          |
| TRIG    | GPIO27       |
| ECHO    | GPIO17       |

IMPORTANT :
Le pin ECHO du HC-SR04 sort du 5V.
Il faut utiliser un diviseur de tension avant de connecter le GPIO au Raspberry Pi.

---
STRUCTURE DU PROJET

PROJECT BeatBoxCorp/
│
├── musicam.py
├── video.py
├── graph.py
├── gui.py
├── motion_log.csv
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

---

INSTALLATION

Mise à jour du système

$ sudo apt update
$ sudo apt upgrade -y

Installation des paquets système

$ sudo apt install -y \
python3-pip \
python3-opencv \
python3-pygame \
libatlas-base-dev \
portaudio19-dev

Installation des bibliothèques Python
    
    $ pip install numpy pandas gpiozero adafruit-circuitpython-bno055

Installation de TensorFlow Lite

    $ pip install tflite-runtime

Si cela ne fonctionne pas :

    $ pip install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl

Télécharger le modèle MoveNet Lightning :

    $ wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/singlepose/lightning/4.tflite -O movenet_lightning.tflite

-----------

SONS

Le projet utilise des fichiers WAV.

Les sons doivent être placés dans le fichier sounds à leurs place respectives.

Exemple, sons de base :

sounds/soft/kick.wav
sounds/medium/kick.wav
sounds/hard/kick.wav
sounds/false/false.wav

Il peuvent être changés manuellement, mais si leurs noms sont différents de "kick.wav", ouvrir musicam.py et allez à la ligne 80 pour changer le nom des fichiés récupérés, voir en ajouter selon les besoins. 

---

LANCER LE PROJET
LANCER INTERFACE GRAPHIQUE 

$ python3 gui.py

L’interface permet :
* de lancer le système,
* d’arrêter le système,
* de générer la vidéo,
* d’ouvrir la vidéo.
* de fermer le programme

Lancer le système interactif seul :

$ python3 musicam.py

Cela lance :
* la webcam,
* la détection de pose,
* le capteur IMU,
* le capteur ultrasonique,
* les sons interactifs,
* les captures d’images quand du mouvement est détecté,
* l’enregistrement CSV.

generer la video d'analyse seule :

$ python3 video.py

Cela génère motion_layers.mp4

La vidéo contient :
* des couches de mouvements,
* des effets de fade,
* une taille dépendante de l’accélération,
* une rotation dépendante de l’inclinaison,
* les informations de distance,
* les sons joués.

generer le graph à partir du fichier csv, non accessible via l'interface utilisateur :

$ python3 graph.py

Le graph devrait s'afficher.

---

FONCTIONNEMENT DU SYSTEME
Détection du corps :

MoveNet détecte les points du corps en temps réel.
Le système détecte ensuite les parties du corps en mouvement grâce à :
* la soustraction de fond,
* les keypoints MoveNet.
Seules les parties réellement en mouvement sont capturées/crop.

--------------
RAPPELS:

Le capteur IMU permet de mesurer :
* l’inclinaison du corps,
* l’accélération,
* l’orientation de l'inclinaison.

Le système utilise notamment le roll pour détecter gauche/droite, et l’accélération linéaire pour mesurer l’intensité de l'instrument.

UTILISATION DES CAPTEURS ULTRASONIQUES
Les capteurs mesurent la distance entre la main associée et le sol.
Plus la main se rapproche du sol, plus le son est fort, plus le movement est rapide, plus l’interaction devient intense.

---

LOGIQUE SONORE
Pour produire un son valide, l’utilisateur doit se pencher dans la bonne direction et rapprocher la main correspondante du sol.
Sinon un faux son est joué.

L’intensité du mouvement//l'accélération de l'IMU change également le type de kick :

| Intensité | Son         |
| --------- | ----------- |
| Faible    | Kick soft   |
| Moyenne   | Kick medium |
| Forte     | Kick hard   |

---

FICHIER CSV
Le système enregistre :

* le timestamp,
* le nom de du screenshot associé,
* toutes les données IMU,
* les sons joués,
* le volume,
* la distance au sol des capteurs ultrasons.

Exemple csv:
time_since_start,image_file,sound_played,sound_volume,distance_cm_droite,distance_cm_gauche
1.52,motion_1.52.jpg,kick_soft,0.72,18.4,50.6

-----------------------

Dépannage :

Si Webcam non détectée
    $ ls /dev/video*
Tester la caméra :
    $ libcamera-hello

---

Si IMU non détecté
    $ i2cdetect -y 1
Le BNO055 apparaît normalement à l’adresse 28 ou 29. Si il n'est pas visible, il est mal cablé.

---
Si problèmes GPIO
    $ sudo usermod -aG gpio $USER
Puis redémarrer le Raspberry Pi.

---
Possibilités futures :

* deux capteurs ultrasoniques,
* interaction main gauche / main droite,
* intégration MIDI,
* connexion Ableton Live,
* visualisations génératives temps réel,
* système de score de mouvement,
* installation artistique immersive.

---

Technologies utilisées

* Python
* OpenCV
* TensorFlow Lite
* Raspberry Pi
* Computer Vision
* Détection de mouvement
* Audio interactif
* IMU
* GPIO

---

LICENSE
Projet éducatif et de recherche.
