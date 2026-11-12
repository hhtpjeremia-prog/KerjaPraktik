"""Generate the NB04b ResNet18 classifier notebook from the canonical trainer."""

from pathlib import Path

import nbformat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notebook_gate import verify_code_cells


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "NB04b_attribute_classifier.ipynb"


def build_notebook() -> nbformat.NotebookNode:
    return nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(
            "# NB04b — Weighted CNN kondisi lampu\n\n"
            "ResNet18 dengan bobot awal ImageNet menggabungkan embedding CNN dengan 35 fitur numerik "
            "HSV/statistik piksel dari crop yang sama untuk membedakan `Standar`, "
            "`TidakStandar`, dan `PantulanCahaya` dari crop bbox/objek. Crop dipertahankan seluruhnya, rasio "
            "aspeknya dijaga, sisi terpanjang diubah ke 224 px, lalu padding mean ImageNet "
            "ditambahkan agar tensor seragam tanpa center-crop atau distorsi. Pixel diubah ke "
            "rentang 0–1 lalu dinormalisasi dengan mean/std ImageNet. Loss weighted `CrossEntropyLoss` "
            "menimbang ketiga kelas berdasarkan distribusi train; angka kelas dan "
            "bobot loss dicetak serta disimpan ke `metrics.json`. Lima epoch awal melatih head "
            "saja sambil membekukan BatchNorm backbone, lalu backbone di-fine-tune dengan "
            "learning rate lebih kecil. Fitur numerik diambil dari crop bersih sebelum augmentasi dan "
            "diskalakan memakai mean/std train saja. ColorJitter hanya diterapkan pada input citra CNN; "
            "fitur numerik saat train dan inference sama-sama berasal dari crop bersih. "
            "Jika NB02c memiliki anotasi test yang sudah direview, metrik test dihitung terpisah. "
            "MSE/RMSE pada label multiclass adalah Brier score dan akar Brier score, bukan metrik bbox. "
            "Kelas PantulanCahaya dipelajari CNN, lalu disaring sebelum masuk ByteTrack/overlay/counting."
        ),
        nbformat.v4.new_code_cell(
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "RISET_ROOT = Path(os.environ.get('RISET_PROJECT_ROOT', Path.cwd())).resolve()\n"
            "sys.path.insert(0, str(RISET_ROOT))\n"
            "sys.path.insert(0, str(RISET_ROOT / 'builders'))\n"
            "from train_attribute_classifier import main\n\n"
            "main()\n"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "from IPython.display import Markdown, display\n\n"
            "history_path = RISET_ROOT / 'output' / 'attribute_classifier' / 'training_history.csv'\n"
            "history_frame = pd.read_csv(history_path)\n"
            "display(history_frame.tail())\n"
            "display(Markdown('Plot lengkap tersimpan di `output/attribute_classifier/training_curves.png`. '\n"
            "'Kolom test hanya ada bila test memiliki ground truth.'))\n"
        ),
    ], metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    })


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.write(notebook, OUTPUT)
    verify_code_cells(OUTPUT, expected_code_cells=2, expected_markdown_cells=1)
    print(f"Generated {OUTPUT}")
