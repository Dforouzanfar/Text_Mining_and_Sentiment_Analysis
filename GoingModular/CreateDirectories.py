from pathlib import Path
def CreateDirectories(*args):
  """
  Create multiple directories in the current working directory based on the provided names.
  """
  for directory_name in args:
    directory = Path(directory_name)
    if not directory.exists():
      directory.mkdir(parents=True, exist_ok=True)
      print(f"Directory '{directory}' created successfully!")
    else:
      print(f"Directory '{directory}' already exists.")

# Call the function with multiple directory names
# CreateDirectories("Data", "GoingModular")
