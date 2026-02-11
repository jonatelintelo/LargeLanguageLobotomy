# Large Language Lobotomy: Jailbreaking Mixture-of-Experts via Expert Silencing

[Paper link](https://arxiv.org/abs/2602.08741)

---
## Overview

Step 1: [collect_gate_output.py](collect_gate_output.py) collects and saves the expert pattern traces.

Step 2: [process_gate_output.py](process_gate_output.py) loads in the previously saved expert traces and transforms it into a usable dataset for the LSTM.

Step 3: [train_lstm.py](train_lstm.py) trains the LSTM and saves the identified utility experts.

Step 4: [prune_model.py](prune_model.py) prunes the model and measures ASR.

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
