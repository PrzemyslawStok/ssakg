# Copyright 2024 The SSAKG Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from microrna.mirna_encoder import MicroRNAEncoder
from microrna.ssakg_mirna_tester import SSAKG_miRNA_Tester
from ssakg import SSAKG
from fasta_helper import FastaHelper


def mirna_example(sequence_length: int = 20, max_no_sequences=None, symbol_padding_left=2, symbol_padding_right=3,
                  context_range=range(5, 7)):
    dirname = "data"
    fasta_helper = FastaHelper(dirname, "mature.fa")

    sequence_length = sequence_length
    read_sequences = fasta_helper.get_filter_sequences(sequence_length, no_sequences=max_no_sequences)

    print(f"Total sequences no: {len(read_sequences)}")

    micro_rna_encoder = MicroRNAEncoder(symbol_padding_left=symbol_padding_left,
                                        symbol_padding_right=symbol_padding_right)
    enc_sequences = micro_rna_encoder.encode_sequences(read_sequences)

    no_symbols = micro_rna_encoder.get_max_no_symbols()

    print(micro_rna_encoder)

    ssakg = SSAKG(number_of_symbols=no_symbols, sequence_length=sequence_length, graphs_to_drawing=False)

    ssakg.insert(enc_sequences)

    ssakg_tester = SSAKG_miRNA_Tester(ssakg, enc_sequences, padding_left=symbol_padding_left,
                                      padding_right=symbol_padding_right, use_consecutive_context=True)

    for context_length in context_range:
        ssakg_tester.make_test(context_length=context_length, show_progress=True)
        ssakg_tester.plot_agreement_histogram(draw_text=True)
        print(ssakg_tester)

    print(ssakg)


if __name__ == "__main__":
    example_1 = True
    example_2 = False

    if example_1:
        mirna_example(sequence_length=20, max_no_sequences=1000, symbol_padding_left=2, symbol_padding_right=3,
                      context_range=range(5, 6))

    if example_2:
        mirna_example(sequence_length=21, max_no_sequences=None, symbol_padding_left=3, symbol_padding_right=3)
