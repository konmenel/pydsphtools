#!/bin/bash

pdoc3 --html -c latex_math=True -f -o docs/html pydsphtools && mv docs/html/pydsphtools/* docs/html
pdoc3 -c latex_math=True -f -o docs/markdown pydsphtools && mv docs/markdown/pydsphtools/* docs/markdown

rmdir docs/*/pydsphtools
