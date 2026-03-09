import os
import sys
import shutil
import subprocess
from pathlib import Path

def print_step(step):
    print(f"\n{'='*60}\n⏳ STEP: {step}\n{'='*60}\n")

def clean_build_dirs():
    """Removes previous build artifacts and dist folders."""
    paths_to_clean = ["build", "dist", "dist_release"]
    
    for path_str in paths_to_clean:
        p = Path(path_str)
        if p.exists() and p.is_dir():
            print(f"🧹 Cleaning directory: {p}")
            shutil.rmtree(p)
            
    print("✨ Clean complete.")

def run_pyinstaller():
    """Runs the PyInstaller compilation command."""
    print("🔨 Starting PyInstaller bundle compilation (this may take a while)...")
    
    # Execute the pyinstaller command exactly as requested
    command = [
        "pyinstaller",
        "--onefile",
        "--noconsole",
        "--hidden-import", "streamlit",
        "--hidden-import", "pandas",
        "--hidden-import", "numpy",
        "--hidden-import", "pymupdf",
        "--hidden-import", "fitz",
        "--hidden-import", "pdfplumber",
        "--hidden-import", "streamlit.runtime.scriptrunner.magic_funcs",
        "--copy-metadata", "streamlit",
        "--collect-all", "streamlit",
        "--collect-all", "fitz",
        "--add-data", f"data{os.pathsep}data",
        "--add-data", f"interface{os.pathsep}interface",
        "--add-data", f"core{os.pathsep}core",
        "--add-data", f"config{os.pathsep}config",
        "--add-data", f"utils{os.pathsep}utils",
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
    # Append .exe for Windows logic parity, though on Mac it will be an executable binary
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
