import json
import re
import sys
from pathlib import Path

SOURCE_ROOT = Path(sys.argv[1]).resolve()
DIGEST_ROOT = Path(sys.argv[2]).resolve()
DIGEST_ROOT.mkdir(parents=True, exist_ok=True)

KEYWORDS = [
    "DeepSORT",
    "deepsort",
    "sort(",
    "tracker",
    "CCTV",
    "rtsp",
    "capture",
    "VideoCapture",
    "low light",
    "lowlight",
    "glare",
    "bright",
    "threshold",
    "adaptiveThreshold",
    "inRange",
    "HSV",
    "hsv",
    "histogram",
    "equalizeHist",
    "CLAHE",
    "blur",
    "Laplacian",
    "blob",
    "SimpleBlobDetector",
    "watershed",
    "kmeans",
    "KMeans",
    "dominant",
    "annotat",
    "labelImg",
    "Roboflow",
    "YOLO",
    "yolo",
    "mAP",
    "confusion_matrix",
    "precision",
    "recall",
    "countNonZero",
    "contour",
]

report_lines = []
keyword_hits = {}
total_code_lines = 0
error_files = []

for notebook_path in sorted(SOURCE_ROOT.rglob("*.ipynb")):
    if ".ipynb_checkpoints" in notebook_path.parts:
        continue
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception as parse_error:
        error_files.append(f"{notebook_path.name}: {parse_error}")
        continue
    digest_chunks = []
    for cell in notebook.get("cells", []):
        source_text = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            heading_lines = [line for line in source_text.splitlines() if line.startswith("#")]
            if heading_lines:
                digest_chunks.append("\n".join(heading_lines))
            continue
        if cell.get("cell_type") != "code":
            continue
        cleaned_lines = []
        for line in source_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("%matplotlib", "%pip", "%cd", "!")):
                continue
            if "plt.imshow" in stripped or "plt.show" in stripped or "display(" in stripped:
                continue
            cleaned_lines.append(line.rstrip())
        if not cleaned_lines:
            continue
        digest_chunks.append("\n".join(cleaned_lines))
    digest_text = "\n\n".join(digest_chunks)
    digest_path = DIGEST_ROOT / (notebook_path.stem.replace(" ", "_")[:110] + ".py.txt")
    digest_path.write_text(digest_text, encoding="utf-8")
    line_count = len(digest_text.splitlines())
    total_code_lines += line_count
    lower_text = digest_text.lower()
    hits = [keyword for keyword in KEYWORDS if keyword.lower() in lower_text]
    keyword_hits[notebook_path.name] = (line_count, hits)

print(f"Notebook diproses: {len(keyword_hits)} · total baris kode (setelah buang output/plot): {total_code_lines}")
print(f"Digest ditulis ke: {DIGEST_ROOT}")
if error_files:
    print("GAGAL parse:")
    for error_line in error_files:
        print(f"  {error_line}")
print()

print("=== 25 NOTEBOOK PALING RELEVAN ===")
ranked = sorted(keyword_hits.items(), key=lambda item: (-len(item[1][1]), -item[1][0]))
for notebook_name, (line_count, hits) in ranked[:25]:
    print(f"  {line_count:5d} baris | {len(hits):2d} hit | {notebook_name[:78]:78s} | {', '.join(hits[:9])}")

print()
print("=== HIT PER KATA KUNCI ===")
keyword_frequency = {}
for notebook_name, (line_count, hits) in keyword_hits.items():
    for keyword in hits:
        keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1
for keyword, frequency in sorted(keyword_frequency.items(), key=lambda item: -item[1]):
    print(f"  {keyword:24s} {frequency:3d} notebook")
