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
@article{STOKLOSA2025108865,
    title = {Associative knowledge graphs for efficient sequence storage and retrieval},
    journal = {Computer Methods and Programs in Biomedicine},
    volume = {269},
    pages = {108865},
    year = {2025},
    issn = {0169-2607},
    doi = {https://doi.org/10.1016/j.cmpb.2025.108865},
    url = {https://www.sciencedirect.com/science/article/pii/S0169260725002822},
    author = {Przemysław Stokłosa and Janusz A. Starzyk and Paweł Raif and Adrian Horzyk and Marcin Kowalik},
}
```

## License

[Apache 2.0](LICENSE)