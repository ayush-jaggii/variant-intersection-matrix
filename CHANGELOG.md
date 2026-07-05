# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-07-05

### Added
- **Combined Fertility Matrix**: Replaced the sparse Opportunity Matrix with a combined Fertility Matrix displaying literature co-occurrence counts alongside concept validation codes for zero-count cells.
- **Academic Classification Mapping**: Implemented custom labeling for zero-paper cells:
  - `G` (Gap / Opportunity): Conceptually relevant (`R`) with 0 papers. Colored dark blue.
  - `N` (Not Relevant): Conceptually not relevant (`N`) with 0 papers. Colored white.
  - `?` (Unresolved): Unresolved/Tied (`?`) consensus and 0 papers. Colored white.
  - `E` (Error / Discrepancy): Conceptually not relevant (`N`) but papers exist ($count > 0$). Colored white.
- **High-Resolution Static Exports for Fertility Matrix**: Ported PNG and SVG image download capability to the Fertility Matrix. Matplotlib's colormap mapping now utilizes a high-resolution segmented lookup table ($N = 4096$ stops) to ensure static images match Plotly's browser colors.
- **Page State Validation Check**: Added ratings validation at startup. Inter-rater reliability and the Fertility Matrix components are hidden from the interface until at least one active rating (`R` or `N`) has been recorded.

### Changed
- **Terminology Standardization**: Renamed "Research Gaps" to "Research Opportunities" and "Opportunity Matrix" to "Fertility Matrix" across the application, metrics panels, and exports.
- **CSV Download Formatting**: Updated exported fertility matrix CSV (`fertility_matrix.csv`) to match screen representations, replacing raw values with `G`, `N`, `?`, `E` or counts.

---

## [1.1.0] - 2026-07-04

### Added
- **Persistent Data Layer**: Separated `BUNDLE_ROOT` (PyInstaller read-only folder) from `PERSISTENT_ROOT` (located next to the compiled executable). Manual validations, overrides, configurations, and uploaded papers now persist safely on disk across app sessions.
- **Pre-Build Verification**: Added directory verification checks in `build_executable.py` to auto-recreate missing output and cache folders before compilation.
- **Streamlit Process Shutdown Handler**: Configured the sidebar "Exit Application" button to execute `os._exit(0)`, preventing ghost background processes on desktops.

### Changed
- **Krippendorff nominal math**: Treating `"?"` as missing data. The nomial agreement is calculated using a 2x2 nominal coincidence matrix (`R` and `N` only), skipping any pairs with fewer than two active ratings.
- **Windows UTF-8 CSV Decoding**: Enforced explicit `utf-8` encoding and replaced decoding fallbacks during download file preparation to resolve Charmap crash errors.

### Fixed
- **requirements.txt syntax**: Removed duplicate blocks, inline concatenations, and comments to resolve dependency installation crashes in Windows environments.
