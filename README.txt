README.txt
-----------

Project Name: Project4Group7

Contents of this .zip:
- Group7_Final_Report.pdf : Final project report
- code/  : Working directory containing source code and dataset
  * Group7_Code.ipynb : Jupyter notebook with all code and analysis
  * main.py           : Entry point script to run experiments
  * models.py         : Contains evaluation functions (Case 1 and Case 2)
  * data_utils.py     : Contains dataset loading and grouping functions
- ROFreq/   : Any input data files used
- readme.txt : Instructions for running the code

Instructions to run the code:

1. Requirements:
   - Python 3.9 or later
   - Jupyter Notebook or JupyterLab
   - Required Python packages:
     * numpy
     * pandas
     * matplotlib
     * scikit-learn
     * openpyxl   (needed for reading .xlsx files)
     (install with: `pip install numpy pandas matplotlib scikit-learn openpyxl`)

2. Running the notebook:
   - Navigate to the `code/` directory.
   - Open the notebook: `Group7_Code.ipynb`
   - Run all cells in order (Kernel → Restart & Run All).
   - The notebook will automatically load the dataset from the `dataset/` folder.

3. Running the Python scripts:
   - Navigate to the `code/` directory.
   - Run `python main.py` from the terminal.
   - Results will be printed to the console and saved as `comparative_results.csv`.

4. Output:
   - All figures, tables, and results are generated inside the notebook.
   - Key results are summarized in `Group7_Final_Report.pdf`.
   - CSV results are saved when running `main.py`.

Notes:
- Ensure the dataset files are placed in the `dataset/` folder before running.
- If paths differ, update the file path in the first cell of the notebook or in `main.py`.
- If you encounter errors reading `.xlsx` files, make sure `openpyxl` is installed.