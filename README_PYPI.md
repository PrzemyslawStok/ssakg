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

[Basics](https://github.com/PrzemyslawStok/ssakg/blob/main/examples/ssakg_basic.ipynb)\
[Reading sequences](https://github.com/PrzemyslawStok/ssakg/blob/main/examples/ssakg_reading.ipynb)  
[SSAKG memory](https://github.com/PrzemyslawStok/ssakg/blob/main/examples/ssakg_tests.ipynb)  

[miRNA example](https://github.com/PrzemyslawStok/ssakg/blob/main/microrna/mirna_example.ipynb)

## License

[Apache 2.0](LICENSE)