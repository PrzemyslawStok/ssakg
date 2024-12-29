# SSAKG

Sequential Structural Associative Knowledge Graph (SSAKG) is a semantic memory.
It can memorize sequences and then read them using a context. 
The context contains random sequence elements. The elements of the context are not ordered.
## Requirements

- **Python Version:** 3.10-3.12
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ssakg.

```bash
pip install ssakg
```

## Usage

```python
from ssakg import SSAKG

# This basic example creates very simple ssakg, and stores two sequences.
# The program shows how sequences are stored in Associative Knowledge Graph.

ssakg = SSAKG(number_of_symbols=10, sequence_length=3, graphs_to_drawing=True)
ssakg.insert([1, 2, 3])
ssakg.insert([2, 4, 5])
ssakg.show()
```

## Examples
Examples of the use of the program:

[Basics](examples/ssakg_basic.ipynb)\
[Reading sequences](examples/ssakg_reading.ipynb)  
[SSAKG memory](examples/ssakg_tests.ipynb)  

[miRNA example](microrna/mirna_example.ipynb)


## Cite
If you use SSAKG in scientific publication, we would appreciate citation of the following paper:
```bibtex
@misc{stokłosa2024associativeknowledgegraphsefficient,
      title={Associative Knowledge Graphs for Efficient Sequence Storage and Retrieval}, 
      author={Przemysław Stokłosa and Janusz A. Starzyk and Paweł Raif and Adrian Horzyk and Marcin Kowalik},
      year={2024},
      eprint={2411.14480},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2411.14480}, 
}
```

## License

[Apache 2.0](LICENSE)