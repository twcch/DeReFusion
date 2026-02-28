# FinTSLib

FinTSLib is a specialized deep learning library dedicated to advanced time series modeling and forecasting, with a primary focus on financial data applications. 

This repository serves as the core codebase for my personal academic research, housing the development and evaluation of custom forecasting architectures.

## Acknowledgments & Base Architecture

This project is built upon the foundational architecture of the excellent open-source repository [Time-Series-Library (TSlib)](https://github.com/thuml/Time-Series-Library) by THUML. 

- **Base Version:** The initial codebase was forked and adapted from the TSlib version as of **February 24, 2026**.
- **Modifications:** While inheriting the robust data processing, training, and evaluation pipelines from the original TSlib, FinTSLib introduces custom modifications, specialized data pipelines, and novel model architectures tailored specifically for financial forecasting tasks.

## Key Features

* **Custom Financial Models:** The primary home for the development and testing of `DyVolFusion` and other proprietary time series models.
* **Robust Pipeline:** Utilizes established and standardized training, validation, and testing loops for deep learning models.
* **Research-Oriented:** Designed specifically for academic experimentation, allowing for rapid prototyping of new algorithms in financial risk and volatility analysis.

## Getting Started

To explore the codebase or run the experiments locally, you can clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/twcch/FinTSLib.git
cd FinTSLib
pip install -r requirements.txt
```

(Note: Please refer to the specific shell scripts in the ./scripts directory for detailed execution commands regarding different models and datasets.)

## Contributing Policy

Please note that FinTSLib is maintained strictly for my personal research and development. Therefore, I do not accept Pull Requests (PRs) or external code contributions. However, this project is completely open-source. The community is more than welcome to fork this repository, explore the models, and adapt the code for your own research needs! For more details, please see the CONTRIBUTING.md file.

## License

This project is licensed under the MIT License.

- Copyright (c) 2026 Chih-Chien Hsieh
- Copyright (c) 2021 THUML @ Tsinghua University

See the [LICENSE](LICENSE) file for the full text.




