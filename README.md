# Large Language Lobotomy: Jailbreaking Mixture-of-Experts via Expert Silencing

[Paper link](https://arxiv.org/abs/2602.08741)

---
## Overview

Step 1: [1_create_lstm_input.py](1_create_lstm_input.py) collects top-k experts selections and creates the LSTM input data.

Step 2: [2_train_lstm.py](2_train_lstm.py) trains the LSTM used to find safety experts.

Step 3: [3_find_safety_experts.py](3_find_safety_experts.py) finds and saves safety experts to silence.

Step 4: [4_prune_safety_experts.py](4_prune_safety_experts.py) lobotomizes (silences) the model and measures ASR.

---

## 📄 Citation

If you find this work helpful, please consider citing our work.

```bibtex
@misc{lintelo2026largelanguagelobotomyjailbreaking,
      title={Large Language Lobotomy: Jailbreaking Mixture-of-Experts via Expert Silencing}, 
      author={Jona te Lintelo and Lichao Wu and Stjepan Picek},
      year={2026},
      eprint={2602.08741},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2602.08741}, 
}
```
---

## 📬 Contact

For questions, please reach out to:
📧 [jona.telintelo@ru.nl](mailto:jona.telintelo@ru.nl)
