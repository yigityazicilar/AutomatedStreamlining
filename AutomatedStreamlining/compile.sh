#!/bin/sh

python -m zipapp src -p "/usr/bin/env python3" -o streamline
chmod +x streamline