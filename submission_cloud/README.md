# Portable notebook presentation

Share `project_presentation.ipynb` on its own. No other project files from your
computer are needed. Use a **Python 3.11 or newer** kernel:

- **Jupyter Notebook/JupyterLab:** open the notebook and choose **Run All Cells**.
- **VS Code:** open the notebook with Python/Jupyter support, select a Python
  kernel, and choose **Run All**.
- **Google Colab:** choose **Upload notebook**, select the notebook, then choose
  **Runtime → Run all**.

The notebook downloads the public project archive (about 170 MB), verifies its
SHA-256 checksum, and extracts the source, trained models, cached evaluation
features, annotations, and example audio into the system temporary directory.
The same setup works in all three environments without platform-specific
branches. It prints the project folder's location and installs application
dependencies into the selected Python kernel without creating a venv or
registering another kernel. Internet access and permission to install packages
are required. Installation may update packages in the selected environment.
If installation requests a restart, restart the kernel/runtime and run all cells
again.

The presentation reproduces the 15-song evaluation, runs inference on the
selected example, and embeds the synchronized audio/chord player inside the
notebook. Change `song` to another name in the displayed song list to try the
other bundled examples. CPU works; an available CUDA GPU can accelerate CNN
inference. The interactive player requires trusted HTML/JavaScript output;
trust the notebook/workspace in Jupyter or VS Code when prompted.
The supplied features are for evaluation; training needs the full training
feature dataset and is not run by this notebook.

Downloads are pinned to public GitHub revision
`cbbde3575a52a9183feab04ef06df567320f1a20` of
[hanneswesterberg/dt2470-project](https://github.com/hanneswesterberg/dt2470-project).
The notebook can be shared on its own; users do not need access to your Google
Drive, a local checkout, or any uploaded model/data files. The source repository
must remain publicly accessible. A fresh environment requires setup again.
Temporary files may be removed by the host; save generated results elsewhere
if you want to keep them.

The original `submission/` folder is unchanged.
