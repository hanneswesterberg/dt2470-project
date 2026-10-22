# Google Colab presentation

`project_presentation.ipynb` is the portable Google Colab counterpart of
`submission/project_presentation.ipynb`. The local notebook remains the
canonical executed presentation; the Colab notebook uses the same evaluation,
plots, inference pipeline, and focused rolling-chord player.

## Run in Colab

1. Upload `project_presentation.ipynb` to Google Colab.
2. Choose **Runtime → Run all**.
3. Allow the setup cell to download and install the submission.

No Google Drive mount, project upload, trained-model upload, or virtual
environment is required. The setup downloads the public `submission_test`
branch into the temporary runtime and installs application dependencies into
the active kernel. Internet access is therefore required.

The notebook reproduces both reported evaluations:

- the CNN acoustic-model diagnostic on 15 annotation-aligned feature sets;
- the fair CNN-only versus CNN+HSMM comparison on identical continuous
  0.2-second frames for all 15 test songs.

It then runs inference on the selected example and embeds the synchronized
audio/chord player directly in the notebook. The generated player contains only
audio controls, current/previous/next chords, and six rolling chord cards.
Available ground truth is displayed beside the prediction. `RWC_P039`, the
best-performing complete-chord example in the 15-song test set, is bundled as a
lossless FLAC file and selected by default.

The project and generated results live in the runtime's temporary directory and
may disappear when the runtime ends. CPU execution is supported; an available
GPU can accelerate CNN inference. The public branch must contain the continuous
posterior caches for the 15-song evaluation to run without downloading the full
RWC audio collection.

The original local notebook is preserved at
`submission/project_presentation.ipynb` and continues to use its project-local
`.venv` kernel plus an external browser window for the player.
