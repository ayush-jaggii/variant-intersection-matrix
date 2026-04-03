# Variant Intersection Matrix Analyzer (VIM Analyzer)

A professional research analysis tool for identifying variant intersections, systematic research gaps, and opportunity spaces in academic literature datasets.

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
- **PDF Text Extraction**: High-fidelity extraction of textual content.
- **Keyword & Variant Detection**: Substring matching against a dictionary of variants and their synonyms (alternate names).
- **Normalization**: Text is cleaned (Unicode normalization, lowercase, noise removal) to ensure matching accuracy.

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
    - **Intra-Dimension Pairs**: Excluded (variants belonging to the same category are treated as mutually exclusive).
    - **Lower Triangular Matrix**: Valid intersections are displayed in the lower triangle for clarity.

### 4. Heatmap Visualization
The results are presented as an interactive heatmap with the following features:
- **Triangular Display**: Focuses on unique pairwise relationships.
- **Numeric Annotations**: Direct display of intersection counts.
- **Dynamic Scaling**: The color intensity adjusts based on the maximum intersection count.
- **Customization**: Support for multiple color palettes (e.g., Viridis, Blues, Magma).
- **Image Export**: High-resolution PNG/JPEG download for publication.

### 5. CSV Export
Standardized data formats are provided for external analysis:
- **Variant Intersection Matrix (CSV)**: Preserves the full structural data, including excluded markers.
- **Pair Details (CSV)**: A flat file listing every valid pair, its count, and the specific papers (IDs) that form the intersection.

### 6. Manual Validation Matrix
Domain experts can refine the analytical results through a manual validation layer:
- **R (Relevant)**: The relationship is conceptually meaningful.
- **N (Not Relevant)**: The relationship is logically or theoretically invalid.
- **? (Uncertain)**: Requires further investigation.

### 7. Fertility Analysis
The **Research Fertility Ratio** quantifies the "promise" of a research area by comparing viable opportunities to the total theoretically possible combinations.
- **Total Possible Pairs**: Calculated as `N(N-1) / 2`.
- **Viable Pairs**: Combinations not marked as `N` or `?`.
- **Research Fertility Ratio**: `Viable_Pairs / Total_Possible_Pairs`.

### 8. Multi-Author Validation
For rigorous academic research, the system supports collaborative rating:
- Individual researchers can upload their own validation CSVs.
- The system aggregates multiple rater matrices.
- Majority voting logic is used to determine final consensus ratings.

### 9. Inter-Rater Reliability
To validate the consistency of manual classifications, the tool computes **Krippendorff’s Alpha** (Nominal):
- **> 0.80**: Strong agreement.
- **0.70 - 0.80**: Acceptable reliability.
- **< 0.67**: Weak reliability.
This ensures that the resulting research gap analysis is statistically sound and suitable for peer-reviewed publication.

---

## Features
- **Automatic Variant Detection**: High-speed processing of large document sets.
- **Inter-Dimension Filtering**: Automatically filters out invalid intra-category pairings.
- **Research Gap Identification**: Instantly highlights 0-value intersections in relevant spaces.
- **Expert Validation**: Layered manual verification for domain-specific accuracy.
- **Reliability Metrics**: Built-in statistical analysis for multi-author studies.
- **Export Ready**: Professional visualizations and CSV data for seamless integration into papers.

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
