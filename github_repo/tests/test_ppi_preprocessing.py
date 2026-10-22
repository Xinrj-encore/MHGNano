from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.prepare_ppi_dataset import make_splits, prepare_dataset, read_fasta


class PpiPreprocessingTests(unittest.TestCase):
    def test_fasta_length_filter_and_identifier_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "small.fa"
            fasta.write_text(
                ">p1 description\n" + "A" * 50 + "\n>p2 description\n" + "C" * 51 + "\n",
                encoding="utf-8",
            )
            self.assertEqual(read_fasta(fasta), {"p2": "C" * 51})
            self.assertEqual(read_fasta(fasta, "description"), {"p2 description": "C" * 51})

    def test_split_reproducibility_and_pair_exclusivity(self):
        pairs = [[index, index + 100] for index in range(20)]
        labels = [1] * 10 + [0] * 10
        first = make_splits(pairs, labels, 5, 73, 0.2)
        second = make_splits(pairs, labels, 5, 73, 0.2)
        self.assertEqual(first, second)
        for fold in range(5):
            pair_sets = {
                name: {tuple(sorted(pair)) for pair in first[name][fold][0]}
                for name in ("train", "val", "test")
            }
            self.assertFalse(pair_sets["train"] & pair_sets["val"])
            self.assertFalse(pair_sets["train"] & pair_sets["test"])
            self.assertFalse(pair_sets["val"] & pair_sets["test"])

    def test_dataset_artifacts_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / "proteins.fa"
            fasta.write_text("".join(f">p{i}\n{'A' * 51}\n" for i in range(5)), encoding="utf-8")
            links = root / "links.txt"
            links.write_text(
                "protein1 protein2 combined_score\n"
                "p0 p1 900\np0 p2 900\np0 p3 900\np0 p4 900\np1 p2 900\n"
                "p1 p3 100\np1 p4 100\np2 p3 100\np2 p4 100\np3 p4 100\n",
                encoding="utf-8",
            )
            output = prepare_dataset(
                links, fasta, root / "out", "toy", positive_cap=5, negative_ratio=1,
                n_splits=5, split_seed=7,
            )
            self.assertTrue((output / "train_idx_toy.pkl").is_file())
            self.assertTrue((output / "PPI_cliques_Hyperedge_toy_8M.pkl").is_file())
            with self.assertRaises(FileExistsError):
                prepare_dataset(
                    links, fasta, root / "out", "toy", positive_cap=5, negative_ratio=1,
                    n_splits=5, split_seed=7,
                )


if __name__ == "__main__":
    unittest.main()
