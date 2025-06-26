# Guided Unconditional and Conditional Generative Models for Super-Resolution and Inference of Quasi-Geostrophic Turbulence

![alt text](img/test_cases.png)  

This repository is associated with our paper, ["Guided Unconditional and Conditional Generative Models for Super-Resolution and Inference of Quasi-Geostrophic Turbulence"](arxiv). It contains trained weights and codes for training and generation with two guided unconditional models ([SDEdit](https://arxiv.org/abs/2108.01073), [Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687)), and two conditional models with and without [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598).  

---
## Installation

Clone this repository. Then, create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
---
## Usage

### Training / Generation

- For training or generation, start from the `Driver` directory. Please change the directory names within according to your system or workflow. 
- The `Config` directory provides example configuration files to train with unconditional and conditional models, and sample (generate) with 4 guided diffusion models.
- The `Weights` directory contains pre-trained weights for super-resolution of forced quasi-geostrophic turbulence in the eddy ($\beta = 0$) and jet ($\beta > 0$) regimes at two Reynolds numbers ($10^3, 10^4$) with coarse-resolution fields and coarse, sparse and gappy (partial) observations. See Datasets (below) for a complete list of data and testcases.

### Datasets

- Quasi-geostrophic simulations and data associated with this code can be downloaded [from the archival repository Zenodo](https://zenodo.org/records/15742146) or [from Kaggle](https://www.kaggle.com/datasets/akhilsadam/quasi-geostrophic-beta-plane-super-resolution).

> Each NumPy file contains all runs at a particular resolution, filtering, and regime, with dimensions `(500, 196, 1, 64, 64)` in the standard `BTCHW` format `(batch, time snapshot, channel, height, width)`.
>
> Files follow `fields_pooled64_ds<res>_<type>_<regime>_re_<reynolds>.npy`: 64x64 filtered fields are down-sampled to resolution `res` in with observation type `type`, regime `regime` and flow Reynolds number `reynolds`.
>
> Resolutions `res` include (down-sampled fields have been interpolated back to the 64x64 grid):
>
> - a default of 64x64
> - `32x32`
> - `16x16`
> - `8x8`
> 
> Types `type` include:
> 
> - a default full field
> - partial (sparse, gappy observations)
> 
> Regimes `regime` include:
> 
> - `eddy` (beta = 0, or north-pole-like flow)
> - `jet` (beta > 0, or upper-latitude-like flow)
> 
> Reynolds numbers `reynolds` include:
> 
> - `1000` (mildly turbulent)
> - `10000` (highly turbulent)

## Citing This Work

If you use find this work interesting, please cite our paper, codes, or dataset:

```bibtex
@Misc{suresh_babu_et_al_2025,
  author =	 {Suresh Babu, Anantha Narayanan and Sadam, Akhil and Lermusiaux, Pierre F. J.},
  title =	 {Guided Unconditional and Conditional Generative Models for Super-Resolution and Inference of Quasi-Geostrophic Turbulence [{D}ataset]},
  month =	 jun,
  year =	 2025,
  doi = {10.5281/zenodo.15742146},
  howpublished = {Zenodo},
}
```
---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
