# Vendored e-means source

The implementation files are sourced from Meta FAIR's FAISS repository, pinned to tag `v1.16.0`:
`https://github.com/facebookresearch/faiss/tree/v1.16.0/contrib/e_means`.

The project uses the official PyTorch e-means implementation, with native PyTorch assignment so it
works on Windows without compiling FAISS. `builders/install_emeans.ps1` downloads the pinned files
only when they are missing. The small `__init__.py` in this directory is the project's import shim.
The upstream license is downloaded alongside the implementation files.
