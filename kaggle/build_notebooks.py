"""
build_notebooks.py — convert the `# %%`-delimited .py sources in this folder
into Kaggle-ready .ipynb files.

Cell markers:
  # %%             -> code cell
  # %% [markdown]  -> markdown cell (leading '# ' stripped from each line)

Run:  python kaggle/build_notebooks.py
"""
import os
import glob
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["nb7_sentiment_model.py","nb6_architecture_comparison.py","nb5_mitigation_experiments.py","nb4_counterfactual_tone_probe.py","nb1b_ita_validation.py","nb1_dataprep_ita_gate.py",
           "nb2_train_armA_fitzpatrick.py",
           "nb3_train_armB_acne04.py"]


def split_cells(text):
    lines = text.splitlines()
    cells, cur, is_md = [], [], False
    started = False

    def flush():
        if not cur:
            return
        body = "\n".join(cur).strip("\n")
        if body.strip() == "":
            return
        cells.append(("markdown" if is_md else "code", body))

    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("# %%"):
            if started:
                flush()
            cur, started = [], True
            is_md = "[markdown]" in stripped
            continue
        cur.append(ln)
    flush()
    return cells


def to_notebook(cells):
    nb = nbf.v4.new_notebook()
    out = []
    for kind, body in cells:
        if kind == "markdown":
            md = "\n".join(
                (l[2:] if l.startswith("# ") else l[1:] if l == "#" else l)
                for l in body.splitlines())
            out.append(nbf.v4.new_markdown_cell(md))
        else:
            out.append(nbf.v4.new_code_cell(body))
    nb["cells"] = out
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3",
                       "language": "python"},
        "language_info": {"name": "python"},
    }
    return nb


def main():
    for src in SOURCES:
        path = os.path.join(HERE, src)
        if not os.path.exists(path):
            print("skip (missing):", src)
            continue
        with open(path, encoding="utf-8") as fh:
            cells = split_cells(fh.read())
        nb = to_notebook(cells)
        out = path.replace(".py", ".ipynb")
        with open(out, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"built {os.path.basename(out)}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
