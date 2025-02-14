#!/bin/env python3
import pdoc
from pathlib import Path



# Configuration
MODULE_NAME = "pydsphtools"  # Replace with your module name
DOCS_DIR = Path("docs")
HTML_DIR = DOCS_DIR / "html"
MARKDOWN_DIR = DOCS_DIR / "markdown"

HTML_DIR.mkdir(parents=True, exist_ok=True)
MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

pdoc.tpl_lookup.directories.insert(0, "./docs")

context = pdoc.Context()
module = pdoc.Module("pydsphtools", context=context)


def recursive_htmls_mds(mod):
    yield mod.name, mod.html(), mod.text()
    for submod in mod.submodules():
        yield from recursive_htmls_mds(submod)


for mod_name, html, text in recursive_htmls_mds(module):
    filepath_html = HTML_DIR / (
        (
            "index"
            if mod_name == "pydsphtools"
            else mod_name.removeprefix("pydsphtools.")
        )
        + ".html"
    )

    filepath_md = MARKDOWN_DIR / (
        (
            "index"
            if mod_name == "pydsphtools"
            else mod_name.removeprefix("pydsphtools.")
        )
        + ".md"
    )
       
    with open(filepath_html, "w") as file:
        file.write(html)
    with open(filepath_md, "w") as file:
        file.write(text)

# pdoc3 --html -c latex_math=True -f -o docs/html pydsphtools && mv docs/html/pydsphtools/* docs/html
# pdoc3 -c latex_math=True -f -o docs/markdown pydsphtools && mv docs/markdown/pydsphtools/* docs/markdown

# rmdir docs/*/pydsphtools
