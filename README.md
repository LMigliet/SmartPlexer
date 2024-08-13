# Smart-Plexer: A Framework for Hybrid Development of Multiplex PCR Assays

Please use the branch: "smartplexer_paper_nature" if you want to see the code reported in the [Nature paper](https://www.nature.com/articles/s42003-023-05235-w).

Welcome to the Smart-Plexer repository! This repository contains the code used for the development and validation of the Smart-Plexer framework, as described in our paper. Smart-Plexer is designed to optimize the selection of primer mixes for multiplex PCR assays through a combination of empirical testing and in-silico simulations.

## Introduction

The Smart-Plexer framework is developed to streamline the development of multiplex PCR assays by combining empirical testing with computer simulations. The framework leverages kinetic inter-target distances among amplification curves to optimize the selection of primer sets for accurate multi-pathogen identification. Initially, the ‘c’ parameter was used as the main feature for optimization. However, we have extended the feature set to include additional robust features that enhance the reliability and accuracy of the assay selection process.

## Features

- **Curve Fitting**: Fits the PCR amplification curves using a 5-parameter sigmoid model.
- **Feature Extraction**: Extract features from the amplification curves, including the ‘c’ parameter and additional robust features.
- **Distance Calculation**: Calculates distances between amplification curves using various metrics.
- **Assay Selection**: Ranks and selects optimal multiplex PCR assays based on feature distances.
- **Empirical Validation**: Validates selected assays through wet-lab experiments.

Disclaimer:
There are two patents related to this work at Imperial College London:
- [Patent 1 - AI for PCR data analysis](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=msNVZbcAAAAJ&sortby=pubdate&citation_for_view=msNVZbcAAAAJ:qxL8FJ1GzNcC)
- [Patent 2 - Smart-Plexer](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=msNVZbcAAAAJ&sortby=pubdate&citation_for_view=msNVZbcAAAAJ:Tyk-4Ss8FVUC)


## How It Works

You should first use the [AdaptiveFilter Algorithm](https://github.com/LMigliet/AdaptiveFiltering) to get the processed dataframes and the metadata needed.

### Folder Structure

Your folder structure should look like this:
How your folder should look like: 
```Usr/code/data/20210701_01/```
Inside the folder (if you used [AdaptiveFilter Algorithm](https://github.com/LMigliet/AdaptiveFiltering)), there will be:
- raw_data, 
- processed, 
- plots 
- metadata.csv

#### Metadata Structure

In this example, there are 5 columns in the metadata:

| Panel | PrimerMix | Target | Assay    | Conc      |
|-------|-----------|--------|----------|-----------|
| 1     | PM3.17    | MER    | MER_N_02 | 1.00E+05  |
| 2     | PM3.17    | COC    | COC_N_05 | 1.00E+05  |


### Data Retrieval and Experiment Object Creation

The first step involves retrieving the data and creating experiment objects using the `create_experiment_objects` function. This function loads data from specified directories and structures it into experiment objects for further processing.

```python
# Define the path where experiment folders are stored
data_path = Path.cwd().parents[0] / "00_data"

# Experiment IDs (ensure folders are named only with the experiment ID)
experiment_ids = ["20210701_01", "20210701_02"]

# Define the METADATA LEN as they will be exclueded from the curve/param data.
META_LEN = 5
```

### Assay Combination Generation

Once the experiment data is loaded, the next step is to generate all possible combinations of assays based on their targets. The process involves:

1.	**Grouping assays by target**: Assays are grouped by their target gene.
2.	**Filtering for singleplex panels**: Only panels with primermix set to "singleplex" are considered for combinations.
3.	**Generating combinations**: The Combinator class is used to generate and store all possible combinations of assays forthe given targets.


## Explanation of Key Components

- **Combinator Class**: The Combinator class is responsible for generating all possible combinations of assays based on the provided assay dictionary. It stores the combinations and provides methods to access the generated combinations and the count.

- **create_experiment_objects Function**: This function initializes experiment objects from raw data, facilitating further processing and analysis.


## Running the Code

1. Ensure your experiment data is correctly formatted and stored in the specified directory.
2. Update the `experiment_ids` list with the IDs of your experiments.
3. Run the code to generate assay combinations and RANK the results. It will also plot the combinations

## Conclusion

The Smart-Plexer framework is a powerful tool for the development and validation of multiplex PCR assays. By combining in-silico simulations with empirical testing, it optimizes the selection of primer sets, ensuring accurate and reliable multi-pathogen identification.