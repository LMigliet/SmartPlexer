from pathlib import Path
from pprint import pprint

from src.parsing import create_experiment_objects

# path where experiments folders are stored.
data_path = Path.cwd().parents[0] / "00_data"
print(data_path)
# ids of the experiments (make sure your folders are named only with the exp_id)
experiment_ids = ["20210701_01", "20210701_02"]

META_LEN = 5
experiments = create_experiment_objects(experiment_ids, data_path, META_LEN)

print("Data Retrivial complete.\n")

############################################################

from src.combinator import Combinator

# Initialize an empty dictionary for grouping assays by target
assay_dict = {}
for experiment in experiments.values():
    for panel in experiment.panels:
        # Only consider panels with primermix set to "singleplex"
        if panel.primermix == "singleplex":
            # Add the assay to the corresponding target in assay_dict
            if panel.target not in assay_dict:
                assay_dict[panel.target] = set()  # Use a set to avoid duplicates
            assay_dict[panel.target].add(panel.assay)

# Convert sets to sorted lists for consistent ordering
for target in assay_dict:
    assay_dict[target] = sorted(assay_dict[target])

# Generate the assay combinations
combinator = Combinator()
combinator.generate_assay_combinations(assay_dict, 2)

print(f"Number of assay combinations: {combinator.combo_count}")
print("Assay combinations dictionary:")
pprint(combinator.assay_combos_dict)

############################################################
