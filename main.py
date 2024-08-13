from pathlib import Path
from pprint import pprint

from src.parsing import create_experiment_objects

# Path where experiments folders are stored.
data_path = Path.cwd().parents[0] / "00_data"
print(data_path)
# IDs of the experiments (make sure your folders are named only with the exp_id)
experiment_ids = ["20210701_01", "20210701_02"]

META_LEN = 5
experiments = create_experiment_objects(experiment_ids, data_path, META_LEN)

print("Data Retrieval complete.\n")

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
# print("Assay combinations dictionary:")
# pprint(combinator.assay_combos_dict)

############################################################

from src.evaluator import AssayEvaluator

# The actual parameter name you are interested in
parameter_name = "Sc"

# Initialize the evaluator class
evaluator = AssayEvaluator(experiments, combinator.assay_combos_dict)

# Iterate over all PMs and extract parameter values for each
ads_mds_allpm = {}
for pm_name in combinator.assay_combos_dict.keys():
    assay_parameter_values = evaluator.evaluate_median_distances(
        pm_name, parameter_name
    )
    ads_mds_allpm[pm_name] = assay_parameter_values

# Rank the primer mixes
ranked_primer_mixes = AssayEvaluator.rank_primer_mixes(ads_mds_allpm)

print("TOP 5 Primer Mixes :")
print(ranked_primer_mixes.head(5))

# saving if needed.
# ranked_primer_mixes.to_csv('multiplex_ranks.csv')

############################################################

import matplotlib.pyplot as plt

average_distances = []
minimum_distances = []
pm_labels = []

for pm_name, distances in ads_mds_allpm.items():
    average_distances.append(distances["ADS"])
    minimum_distances.append(distances["MDS"])
    pm_labels.append(pm_name)

plt.figure(figsize=(20, 15))
plt.scatter(average_distances, minimum_distances, color="blue")

for i, pm_name in enumerate(pm_labels):
    plt.text(
        average_distances[i], minimum_distances[i], pm_name, fontsize=9, ha="right"
    )

plt.title("Average Distance vs Minimum Distance per Primer Mix")
plt.xlabel("ADS")
plt.ylabel("MDS")
plt.grid(True)
plt.show()
