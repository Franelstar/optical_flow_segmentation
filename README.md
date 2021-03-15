# Optical Flow Segmentation

#### Lucas_Kanade_Optical_Flow.py
Run `python Lucas_Kanade_Optical_Flow.py -m dataset/slow_traffic_small.mp4`

the parameters are:

1. -m: path to the video

#### Dense_Optical_Flow.py
Run `python Dense_Optical_Flow.py -m dataset/PETS09-S2L1-raw.mp4`

the parameters are:

1. -m: path to the video

# 1. Estimation du flot optique dans une séquence d’images

#### Flow_optique_norme.py
Run `python Flow_optique_norme.py -m dataset/PETS09-S2L1-raw.mp4`

the parameters are:

1. -m: path to the video

#### Flow_optique_norme_direction.py
Run `python Flow_optique_norme_direction.py -m dataset/PETS09-S2L1-raw.mp4`

the parameters are:

1. -m: path to the video

# 2. Segmentation

## 2.1. segmentation des objets se déplaçant le plus vite

### 2.1.1. Avec seuillage

#### segmentation_objets_plus_rapide_seuillage.py
Run `python segmentation_objets_plus_rapide_seuillage.py -m dataset_mini/Venice.avi -t 40 -k 7`

the parameters are:

1. -m: path to the video
1. -t: threshold, default: 40
1. -k: Kernel, default: 7
1. -s: Save result, default: False

### 2.1.2. Avec Kmeans

#### segmentation_objets_plus_rapide_kmeans.py
Run `python segmentation_objets_plus_rapide_kmeans.py -m dataset_mini/Venice.avi -K 3 -k 7
`

the parameters are:

1. -m: path to the video
1. -K: K, default: 3
1. -k: Kernel, default: 7
1. -s: Save result, default: False

## 2.2. segmentation des objets avec différentes vitesse

### 2.2.1. Avec seuillage

#### segmentation_objets_diferrente_vitesse_seuillage.py
Run `python segmentation_objets_diferrente_vitesse_seuillage.py -m dataset_mini/Trafic2.avi -t1 70 -t2 150 -k 7`

the parameters are:

1. -m: path to the video
1. -t1: threshold_1, default: 70
1. -t2: threshold_2, default: 150
1. -k: Kernel, default: 7
1. -s: Save result, default: False

### 2.2.2. Avec Kmeans

#### segmentation_objets_differente_vitesse_kmeans.py
Run `python segmentation_objets_differente_vitesse_kmeans.py -m dataset_mini/Trafic2.avi -k 4 -k 7`

the parameters are:

1. -m: path to the video
1. -K: K, default: 3
1. -k: Kernel, default: 7
1. -s: Save result, default: False

## 2.2. segmentation des en fonction de la norme et de l'angle

#### segmentation_objets_diferrente_vitessse_norme_direction.py
Run `python segmentation_objets_diferrente_vitessse_norme_direction.py -m dataset_mini/Trafic2.avi -K 4 -k 7`

the parameters are:

1. -m: path to the video
1. -K: K, default: 3
1. -k: Kernel, default: 7
1. -s: Save result, default: False