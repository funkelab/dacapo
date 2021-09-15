### Configurables
This is where we keep anything that we might want to provide multiple versions of.
1) Architectures
    - UNet
    - VGG
2) Augments
    - Intensity
    - Simple
3) Data Sources
    - bossdb
    - csv
    - networkx file
    - rasterized graph
    - zarr
4) Executers
    - local
5) Losses
    - cross entropy
    - mse
    - weighted mse
6) optimizers
    - adam
    - radam
7) Predictors
    - affinities
    - lsd
    - one_hot_labels
8) Processing Steps
    - Agglomerate
    - Argmax
    - Create LUTS
    - Fragment
    - Segment