# PaX
Repository for 'Position-aware eXplanation: A Model Agnostic Framework for Positional Attributions'

## Repository Structure

```
PaX/
├── README.md
└── src/
    └── pax/
        ├── datasets/
        │   ├── data_utils.py        # Dataset preprocessing and loading utilities
        │   └── image_utils.py       # Image-specific data utilities
        │
        └── explainers/
            ├── explainer_utils.py   # Shared explainer helper functions
            ├── posfullgrad.py       # Position-aware FullGrad
            ├── posintgrad.py        # Position-aware Integrated Gradients
            ├── poslime.py           # Position-aware LIME
            ├── posrise.py           # Position-aware RISE
            ├── posshap.py           # Position-aware SHAP
            └── posshapiq.py         # Position-aware SHAP-IQ
```

### Key Directories

- **datasets/**  
  Utilities for dataset handling and preprocessing.

- **explainers/**  
  Implementation of position-aware attribution methods built on top of existing model-agnostic explainers.
