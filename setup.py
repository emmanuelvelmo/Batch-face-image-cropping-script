from cx_Freeze import setup, Executable

setup(name="Batch face image cropping", executables=[Executable("Batch face image cropping script.py")], options={"build_exe": {"excludes": ["tkinter"]}})