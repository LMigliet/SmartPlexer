from pathlib import Path

from src.parsing import create_experiment_objects

############################################################

# path where experiments folders are stored.
data_path = Path.cwd().parents[0] / "00_data"
print(data_path)
# ids of the experiments (make sure your folders are named only with the exp_id as follow)
experiment_ids = ["20210701_01", "20210701_02"]

META_LEN = 5
experiments = create_experiment_objects(experiment_ids, data_path, META_LEN)

print("Done")

unique_assays = set()  # Use a set to ensure uniqueness

for id, experiment in experiments.items():
    for panel in experiment.panels:
        unique_assays.add(panel.assay)  # Add the assay to the set

# Convert the set to a list (optional, if you want a list format)
unique_assays_list = list(unique_assays)

# Print or return the list of unique assays
print(unique_assays_list)
