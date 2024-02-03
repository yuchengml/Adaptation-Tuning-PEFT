<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Adaptation-Tuning-PEFT</h3>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#to-do-list">To-do List</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
This project involves comparing LLMs based on the fine-tuning methods within the Hugging Face PEFT framework for specified downstream tasks.
This comparison aims to provide relevant experimental data as a reference for other development tasks.

### Built With
[Hugging Face PEFT][peft-url]

[![W&B][wandb-shield]][wandb-url]

<!-- TO DO List -->
## To-do List
### Sequence Classification
- [x] Prepare datasets
- Pre-process data
  - [x] Filtering
- Create models
  - [x] Sequence classification model
- Implement fine-tuning methods
  - [x] P-Tuning
  - [x] Prefix Tuning
  - [ ] LoRA
- [ ] Experiments on W&B


<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/yuchengml/Adaptation-Tuning-PEFT/blob/main/LICENSE
[peft-shield]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[peft-url]: https://github.com/huggingface/peft
[wandb-shield]: https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white
[wandb-url]: https://wandb.ai/site
