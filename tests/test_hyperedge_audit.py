from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.audit_hyperedge_sizes import SOURCES, summarize


class HyperedgeAuditTests(unittest.TestCase):
    def test_species_summary_counts_clique_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for _, _, relative_path in SOURCES:
                source = root / relative_path
                source.parent.mkdir(parents=True, exist_ok=True)
                with source.open("wb") as handle:
                    pickle.dump(
                        {
                            "num_vertices": 5,
                            "PPI_edge_list": [[0, 1], [1, 2]],
                            "PPI_cliques_list": [[0, 1], [0, 1, 2], [0, 1, 2, 3, 4]],
                        },
                        handle,
                    )

            rows = summarize(root)

        self.assertEqual(len(rows), len(SOURCES))
        self.assertEqual(rows[0]["hyperedges"], 3)
        self.assertEqual(rows[0]["size_2"], 1)
        self.assertEqual(rows[0]["size_3"], 1)
        self.assertEqual(rows[0]["size_ge5"], 1)
        self.assertEqual(rows[0]["size_ge3"], 2)
        self.assertAlmostEqual(rows[0]["pct_size_ge3"], 200 / 3)
        self.assertAlmostEqual(rows[0]["mean_size"], 10 / 3)


if __name__ == "__main__":
    unittest.main()
