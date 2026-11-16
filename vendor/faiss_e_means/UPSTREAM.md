# Vendored e-means source

The implementation files are sourced from Meta FAIR's FAISS repository, pinned to commit
`6ccfd5f9be2283ebc037564bb45c6ba63b71e96a` (release tag `v1.16.0`):
`https://github.com/facebookresearch/faiss/tree/6ccfd5f9be2283ebc037564bb45c6ba63b71e96a/contrib/e_means`.

The project uses the official PyTorch e-means implementation, with native PyTorch assignment so it
works on Windows without compiling FAISS. `builders/install_emeans.ps1` downloads the pinned files
only when they are missing. The installer verifies SHA256 for all four downloaded source/license
files before copying them. Hash values are pinned in `builders/install_emeans.ps1`. The small
`__init__.py` in this directory is the project's import shim. The upstream license is included.
