#!/bin/sh

python -m zipapp src -p "/usr/bin/env python3" -o NogoodParsing.pyz
chmod +x NogoodParsing.pyz
mv -f NogoodParsing.pyz ../NogoodBinning