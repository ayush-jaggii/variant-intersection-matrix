import os
import sys
import shutil
import subprocess
from pathlib import Path


def print_step(step):
    print(f"\n{'='*60}\n⏳ STEP: {step}\n{'='*60}\n")


def clean_build_dirs():
    """Removes previous build artifacts and dist folders safely."""
    paths_to_clean = ["build", "dist", "dist_release"]

    for path_str in paths_to_clean:
        p = Path(path_str)
        if p.exists() and p.is_dir():
            print(f"🧹 Cleaning directory: {p}")
            shutil.rmtree(p, ignore_errors=True)

    print("✨ Clean complete.")


def run_pyinstaller():
    """Runs the PyInstaller compilation command."""
    # Ensure source data directory structure exists so PyInstaller doesn't crash
    for d in ["data/papers", "data/variants", "data/cache", "data/output"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        
    # Ensure default variants.json exists
    var_json = Path("data/variants/variants.json")
    if not var_json.exists():
        with open(var_json, "w", encoding="utf-8") as f:
            f.write('{"dimensions": {}}')

    print("🔨 Starting PyInstaller bundle compilation (this may take a while)...")

    version_info_content = """VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1,0,0,0),
    prodvers=(1,0,0,0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0,0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [
          StringStruct('CompanyName', 'Academic Research Software'),
          StringStruct('FileDescription', 'Variant Intersection Matrix Analyzer'),
          StringStruct('FileVersion', '1.0.0'),
          StringStruct('InternalName', 'VIM_Analyzer'),
          StringStruct('OriginalFilename', 'VIM_Analyzer.exe'),
          StringStruct('ProductName', 'VIM Analyzer'),
          StringStruct('ProductVersion', '1.0.0'),
          StringStruct('LegalCopyright', 'Open Research Tool'),
          StringStruct('Author', 'Academic Multi-Author Research Tool')
        ]
      )
    ]),
    VarFileInfo([
      VarStruct('Translation', [1033,1200])
    ])
  ]
)
"""

    with open("version_info.txt", "w", encoding="utf-8") as f:
        f.write(version_info_content)

    command = [
      sys.executable,
      "-m",
      "PyInstaller",
        "--onefile",
        "--noconsole",

        "--hidden-import", "streamlit",
        "--hidden-import", "pandas",
        "--hidden-import", "numpy",
        "--hidden-import", "pymupdf",
        "--hidden-import", "fitz",
        "--hidden-import", "streamlit.runtime.scriptrunner.magic_funcs",
        "--hidden-import", "seaborn",
        "--hidden-import", "matplotlib",

        "--copy-metadata", "streamlit",

        "--collect-all", "streamlit",
        "--collect-all", "fitz",
        "--collect-all", "plotly",
        "--collect-all", "seaborn",
        "--collect-all", "matplotlib",

        "--add-data", f"data{os.pathsep}data",
        "--add-data", f"interface{os.pathsep}interface",
        "--add-data", f"core{os.pathsep}core",
        "--add-data", f"config{os.pathsep}config",
        "--add-data", f"utils{os.pathsep}utils",

        "--version-file", "version_info.txt",
        "--name", "VIM_Analyzer",

        "launcher.py"
    ]

    try:
        subprocess.run(command, check=True)
        print("✅ PyInstaller build successful.")
    except subprocess.CalledProcessError as e:
        print(f"❌ PyInstaller build failed with error: {e}")
        sys.exit(1)


def release_executable():
    """Moves the compiled file from dist to dist_release."""
    dist_dir = Path("dist")
    release_dir = Path("dist_release")

    release_dir.mkdir(exist_ok=True)

    exe_name = "VIM_Analyzer"
    if os.name == "nt":
        exe_name += ".exe"

    compiled_file = dist_dir / exe_name
    final_file = release_dir / exe_name

    if compiled_file.exists():
        print(f"📦 Moving executable from {compiled_file} to {final_file}...")
        shutil.copy(compiled_file, final_file)
        print(f"🎉 Build Complete! You can find the executable at: {final_file}")
    else:
        print(f"❌ Error: Final executable {compiled_file} not found after build.")


if __name__ == "__main__":
    print("🚀 Starting VIM Analyzer Build Process...")

    print_step("Cleaning Environment")
    clean_build_dirs()

    print_step("Running PyInstaller")
    run_pyinstaller()

    print_step("Finalizing Release")
    release_executable()