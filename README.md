# Smart-Plexer: A Framework for Hybrid Development of Multiplex PCR Assays

Please use the branch: "smartplexer_paper_nature" if you want to see the code reported in the nature paper [https://www.nature.com/articles/s42003-023-05235-w].

Welcome to the Smart-Plexer repository! This repository contains the code used for the development and validation of the Smart-Plexer framework, as described in our paper. Smart-Plexer is designed to optimize the selection of primer mixes for multiplex PCR assays through a combination of empirical testing and in-silico simulations.

## Introduction

The Smart-Plexer framework is developed to streamline the development of multiplex PCR assays by combining empirical testing with computer simulations. The framework leverages kinetic inter-target distances among amplification curves to optimize the selection of primer sets for accurate multi-pathogen identification. Initially, the ‘c’ parameter was used as the main feature for optimization. However, we have extended the feature set to include additional robust features that enhance the reliability and accuracy of the assay selection process.

## Features

- **Curve Fitting**: Fits the PCR amplification curves using a 5-parameter sigmoid model.
- **Feature Extraction**: Extracts features from the amplification curves, including the ‘c’ parameter and additional robust features.
- **Distance Calculation**: Calculates distances between amplification curves using various metrics.
- **Assay Selection**: Ranks and selects optimal multiplex PCR assays based on feature distances.
- **Empirical Validation**: Validates selected assays through wet-lab experiments.

Disclaimer:
There are two patents related to this work at Imperial College London:
- [https://scholar.google.com/citations?view_op=view_citation&hl=en&user=msNVZbcAAAAJ&sortby=pubdate&citation_for_view=msNVZbcAAAAJ:qxL8FJ1GzNcC]
- [https://scholar.google.com/citations?view_op=view_citation&hl=en&user=msNVZbcAAAAJ&sortby=pubdate&citation_for_view=msNVZbcAAAAJ:Tyk-4Ss8FVUC]
