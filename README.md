# Creating Runtime Monitors for ML Components in Autonomous Systems

A toolset for exploring the creation of a model that monitors the performance of ML components to be used in autonomous systems.

### Overview

The aim of this work is to conceptualise and develop a runtime safety monitor for an ML component (here, an image classifier), as part of a wider project on the safety assurance of machine learning for use in autonomous systems.

The problem is tackled sequentially:
1. A representative set of input data is degraded for $n^k$ combinations, with $k$ influencing factors that may impact input data at runtime (e.g., fog, rain), and $n$ different values of  the perturbation value $\epsilon$ (see combineTransform.py).
2. The performance of the ML component is tested on each of the $n^k$ degraded datasets. Each dataset is labelled according to its accuracy class, specified a priori by the developer of the monitor (see labelData.py).
3. Image data is prepared for training by randomly selecting $ntrainind$ and $ntestind$ degraded datasets from each accuracy class. Datasets are then concatenated, randomly shuffled, and pickled under *data.pickle* (see prepareData.py).
4. A deep learning model is trained to classify images based on their accuracy class. When deployed together with the original ML component, the monitor may then trigger a safety assessment upon detecting input data that is likely to result in unacceptable performance (see createMonitor).

### Prerequisites

python == 3.8.1

### Install Dependencies

```bash
git clone <GITHUB_REPO_URL>
cd createMonitors

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Instructions

Python scripts to be executed in this order:
1. combineTransform.py
2. labelData.py
3. prepareData.py
4. createMonitor.py

(moving to jupyter notebook)

### Resources

- [Transferring Assurance for Machine Learning in Autonomous Systems](https://eprints.whiterose.ac.uk/196682/)
- [DeepCERT: contextually-relevant verification of image classification Neural Networks](https://github.com/DeepCert/contextual-robustness/tree/main) 
