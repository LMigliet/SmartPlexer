import itertools
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Combinator:
    """
    A class to generate and store all possible assay combinations.

    Attributes:
    -----------
    combo_count : int
        Stores the number of unique assay combinations generated.
    assay_combos_dict : Dict[str, Dict[str, str]]
        A dictionary storing assay combinations. The keys are multiplex (PM) labels, and the values are dictionaries
        mapping targets to assays.

    Methods:
    --------
    generate_assay_combinations(assay_dict: Dict[str, List[str]], zeros_for_pm_labels: int):
        Generates and stores all possible assay combinations given a dictionary of assays grouped by targets.
    """

    combo_count: int = field(init=False)
    assay_combos_dict: Dict[str, Dict[str, str]] = field(
        init=False, default_factory=dict
    )

    def generate_assay_combinations(
        self,
        assay_dict: Dict[str, List[str]],
        zeros_for_pm_labels: int,
    ):
        """
        Generates and stores all possible assay combinations based on the provided assay data.

        Parameters:
        -----------
        assay_dict : Dict[str, List[str]]
            A dictionary where keys are target names and values are lists of assays associated with each target.

        zeros_for_pm_labels : int
            The number of leading zeros to include in the multiplex (PM) labels. This helps in creating consistent
            label formatting for assay combinations.

        After running this method, `combo_count` will hold the number of unique combinations generated, and
        `assay_combos_dict` will contain the combinations with their respective PM labels.
        """

        # Step 1: Create all possible combinations of assays
        assay_combos = list(itertools.product(*assay_dict.values()))
        self.combo_count = len(assay_combos)

        # Step 2: Generate the assay combinations dictionary
        for i, combo in enumerate(assay_combos):
            pm_label = f"PM{len(assay_dict)}.{str(i+1).zfill(zeros_for_pm_labels)}"
            self.assay_combos_dict[pm_label] = {
                target: assay for target, assay in zip(assay_dict.keys(), combo)
            }
