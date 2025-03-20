# MS-PINN for Horizontal Wind Field Reconstruction

This repository provides a demo version of **MS-PINN (Multi-Scale Physics-Informed Neural Networks)** for reconstructing horizontal wind fields, serving as a reference for interested users. Below is an overview of the included files and setup instructions.

## Repository Contents

- **`train_uv_modify.py`**: The training script for the model. You can run this script directly to start training.
- **`pinn_model.py`**: Defines the core model architecture, as well as data loading and preprocessing methods.
- **`plane_error_cal_zhanbi.py`**: Contains some post-processing methods for reference.
- **`data_sample.rar`**: Contains synthetic LiDAR measurement data used for training. Simply extract this file into the root directory.

All other images or GIFs included in this repository are post-processed results, provided for reference.

## Environment Setup

The script is designed to run in a **PyTorch** environment with the following specifications:
- Python version: **3.9.13**
- PyTorch version: **1.13.0+cu117**
- Additional packages can be installed as needed based on your requirements.

## Usage

1. Extract `data_sample.rar` into the root directory.
2. Run `train_uv_modify.py` to start training the model.
3. Use the provided scripts for post-processing and analysis.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For questions or feedback, please open an issue or contact the repository owner.
