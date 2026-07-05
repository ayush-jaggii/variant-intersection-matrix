# Variant Intersection Matrix Analyzer (VIM Analyzer)

A professional research analysis tool for identifying variant intersections, systematic research opportunities, and fertility spaces in academic literature datasets.

---

## Overview

The **Variant Intersection Matrix Analyzer (VIM Analyzer)** is designed to assist researchers in performing systematic literature reviews and conceptual mapping. By analyzing collections of academic papers against a predefined set of conceptual variants, the software constructs a **Variant Intersection Matrix**. 

This matrix quantifies how frequently specific conceptual combinations appear together in the literature, enabling researchers to identify:
- **Existing Research Clusters**: Well-explored areas with high intersection counts.
- **Unexplored Combinations**: Pairs of variants that are conceptually relevant but lack supporting literature.
- **Research Opportunity Spaces**: Potential gaps where new research can provide the most significant contributions.

---

## System Architecture

The VIM Analyzer follows a structured analytical pipeline to ensure reproducibility and statistical rigor.

### 1. Paper Processing
The system ingests papers in PDF or TXT format. The processing involves:
- **PDF Text Extraction**: High-fidelity extraction of textual content using PyMuPDF.
- **Keyword & Variant Detection**: Substring matching against a dictionary of variants and their synonyms (alternate names) using case-insensitive regular expressions with word boundary checks.
- **Normalization**: Text is cleaned (Unicode NFKD normalization, lowercase, hyphen/underscore to space replacement, non-alphanumeric stripping) to ensure matching accuracy.

### 2. Paper × Variant Binary Matrix
Each processed paper is converted into a binary representation:
- **Rows**: Individual papers (P1, P2, ...).
- **Columns**: Defined variants.
- **Values**: `1` if the variant is detected in the paper, `0` otherwise.

### 3. Variant Intersection Matrix
The system computes pairwise intersections between variants to identify co-occurrences.
- **Intersection Count**: Computed as the number of papers containing both variants in a pair.
- **Structural Rules**:
    - **Diagonal**: Excluded (self-comparison).
    - **Intra-Dimension Pairs**: Excluded (variants belonging to the same category/dimension are treated as mutually exclusive).
    - **Lower Triangular Matrix**: Valid intersections are displayed in the lower triangle for clarity.

### 4. Heatmap Visualization
The results are presented as an interactive heatmap with the following features:
- **Triangular Display**: Focuses on unique pairwise relationships.
- **Numeric Annotations**: Direct display of intersection counts.
- **Dynamic Scaling**: The color intensity adjusts based on the maximum intersection count.
- **Customization**: Support for multiple color palettes (e.g., Viridis, Blues, Magma, Greys).
- **Image Export**: High-resolution PNG/SVG download for publication.

### 5. CSV Export
Standardized data formats are provided for external analysis:
- **Variant Intersection Matrix (CSV)**: Preserves the full structural data, including excluded markers.
- **Pair Details (CSV)**: A flat file listing every valid pair, its count, and the specific papers (IDs) that form the intersection.

### 6. Manual Validation Layer
Domain experts can refine the analytical results through a manual validation layer for variant pairs:
- **R (Relevant)**: The relationship is conceptually meaningful.
- **N (Not Relevant)**: The relationship is logically or theoretically invalid.
- **? (Unrated / Uncertain)**: Default state or requires further discussion.

### 7. Inter-Rater Reliability
To validate the consistency of manual classifications, the tool computes **Krippendorff’s Alpha** (Nominal) by treating `"?"` as missing data:
- Calculated over active categories (`R` and `N` only) using a 2x2 nominal coincidence matrix.
- Pairs with fewer than 2 active ratings are excluded to prevent distortion.
- **> 0.80**: Strong agreement.
- **0.70 - 0.80**: Acceptable reliability.
- **< 0.70**: Weak reliability.
This ensures that the inter-rater reliability analysis is statistically sound and suitable for peer-reviewed publication.

### 8. Combined Fertility Matrix (G, N, ?, E Outcomes)
A final combined view that merges literature co-occurrence counts and validation consensus ratings. Each pair is classified into one of five outcomes:
- **`G` (Gap / Opportunity)**: Conceptually relevant (`R`) but `0` papers in the literature. Colored dark blue.
- **`N` (Not Relevant)**: Conceptually not relevant (`N`) and `0` papers. Colored white.
- **`?` (Unresolved)**: Unresolved/Tied (`?`) consensus rating and `0` papers. Colored white.
- **`E` (Error / Discrepancy)**: Conceptually not relevant (`N`) but `count > 0` papers exist. Colored white.
- **`Count`**: Conceptually relevant (`R`) or unresolved (`?`) and `count > 0` papers exist. Colored by literature density.

### 9. Research Fertility Ratio
The **Research Fertility Ratio** quantifies the unused opportunity density within the morphological space:
- **Total Possible Pairs**: Total valid cross-dimension variant pairs (excluding same-category combinations).
- **Research Opportunities**: Pairs rated **"R"** (Relevant) by consensus with 0 papers.
- **Fertility Ratio**: `Research_Opportunities / Total_Possible_Pairs`.

---

## Features
- **Automatic Variant Detection**: High-speed processing of large document sets.
- **Inter-Dimension Filtering**: Automatically filters out invalid intra-category pairings.
- **Research Opportunity Identification**: Instantly highlights conceptual gaps (`G`) in relevant spaces.
- **Expert Validation**: Layered manual verification for domain-specific accuracy.
- **Reliability Metrics**: Built-in Krippendorff's Alpha analysis for multi-author studies.
- **Export Ready**: Professional PNG/SVG visualizations and CSV data for publications.

---

## Running the Software

### Prerequisites
Ensure you have Python 3.9+ installed.

### Install Dependencies (pip)
```bash
pip install -r requirements.txt
```

### Optional: Create Conda Environment
```bash
conda env create -f environment.yml
conda activate vim-analyzer
```

### Launch the Application (Recommended)
Use the launcher entrypoint:
```bash
python run.py
```

This starts Streamlit in background mode and opens the app at:

`http://localhost:8501`

### Alternative Manual Launch
```bash
python -m streamlit run interface/app.py --server.headless=true --server.port=8501
```

### Stop the Application
If started via `run.py`, stop the running process from terminal:

```bash
lsof -nP -iTCP:8501 -sTCP:LISTEN
kill <PID>
```

### Troubleshooting
- Port already in use: stop the existing process on port 8501 and relaunch.
- Browser shows stale page: hard refresh the tab after code updates.
- First startup can take a few seconds while Streamlit initializes.

If using the compiled version, run the `VIM_Analyzer` executable.

---

## Intended Use
The VIM Analyzer is designed for:
- **Systematic Literature Reviews (SLR)**.
- **Thematic Analysis** in qualitative and mixed-methods research.
- **Conceptual Framework Development**.
- **Collaborative Research Validation** and peer-verification.

---

## License
This tool is released as an open-research software for academic use and reproducibility.
