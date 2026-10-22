# Google Colab presentation

Open [Google Colab](https://colab.research.google.com/), choose **Upload notebook**,
and select `project_presentation.ipynb` from this folder. Then choose
**Runtime → Run all**. No other files from your computer are needed.

The notebook downloads the public project archive (about 170 MB), verifies its
SHA-256 checksum, and extracts the source, trained models, cached evaluation
features, annotations, and example audio into the temporary Colab runtime. It
installs application dependencies directly into Colab without creating a venv
or asking you to select another kernel. If package installation requests a
restart, restart the runtime and run all cells again.

The presentation reproduces the 15-song evaluation, runs inference on the
selected example, and embeds the synchronized audio/chord player inside the
notebook. Change `song` to another name in the displayed song list to try the
other bundled examples. CPU works; a Colab GPU can accelerate CNN inference.
The supplied features are for evaluation; training needs the full training
feature dataset and is not run by this notebook.

Downloads are pinned to public GitHub revision
`cbbde3575a52a9183feab04ef06df567320f1a20` of
[hanneswesterberg/dt2470-project](https://github.com/hanneswesterberg/dt2470-project).
The notebook can be shared on its own; users do not need access to your Google
Drive, a local checkout, or any uploaded model/data files. The source repository
must remain publicly accessible. A fresh Colab runtime requires setup again;
download generated results before the runtime is deleted if you want to keep them.

The original `submission/` folder is unchanged.
