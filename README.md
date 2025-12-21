# DNA Typewriter MOI Inference

This repository deals with analysis of DNA Typewriter sequencing data, as shown in Regalado & Qiu et al, 2025 (https://www.biorxiv.org/content/10.1101/2025.05.23.655664v3).

A major challenge in analyzing DNA Typewriter Tape (DTT) data is that Tape barcodes (TapeBCs) do not uniquely define genomic loci, as most DTTs are duplicated—likely due to the high MOI piggyBac transposase system. This duplication complicates lineage reconstruction, making it difficult to compare edit patterns across cells without knowing the originating DTT locus. To address this, we developed a method to infer the most likely number of integration events per TapeBC into the genome as well as the edit patterns that were generated from each integration locus, capturing the true sequence of events that occur from the starting cell at each locus independently. The key principle on which this method was based was the idea that in a given cell, for a given TapeBC, there should exist only one observed pattern of sequential edits. If multiple patterns of edits with the same TapeBC in the same cell are observed, this is evidence for multiple genomic integration sites for that TapeBC and associated DTT. 

Further details on our algorithmic strategy are presented in a Supplementary Note. This example vignette is for a specific TapeBC from a subset of 32 cells with one major bifurcation into two clades. 

<img width="1358" height="1333" alt="typewriter_figure copy" src="https://github.com/user-attachments/assets/bfeeff2d-d262-4f15-97d8-d7c583c2f625" />


Figure S10. (A) t-SNE depicting clusters of DTT UMIs across all 32 cells. The labels for each cluster represent a unique pattern of edits. Each 3N edit is separated by a “|”. UMIs not assigned to any cluster were discarded as likely noise from PCR or sequencing errors. (B) After identifying consensus edit patterns, we generated a cell-by-pattern matrix with UMI counts, row-normalized and log-transformed to highlight differences between cells. This matrix was then binarized (not shown) and the rows were clustered into two groups of cells. (C) The representative mutual exclusivity graph for this dataset of patterns is shown. An edge is drawn between two patterns of edits if and only if those two patterns of edits are never co-expressed in any cluster of cells in the previous matrix. Edge weights indicate the number of shared sequential edits between mutually exclusive patterns. (D) Graph coloring on the complement of the mutual exclusivity graph reveals the two most likely groups of mutually exclusive patterns. These groupings correspond to unique DTT integration loci of the same TapeBC. 


To run pipeline on example data:

```
python full_pipeline.py -i "tape_data_32cells.csv" \
	-uc 3 \
	-mu 20 \
	-mc 2 \
	-o output.csv
```

For full-sized datasets, keep all optional parameters as default. To include plotting (as above), set `-p True`. This pipeline benefits greatly from parallel processing.


