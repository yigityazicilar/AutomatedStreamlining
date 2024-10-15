#!/bin/sh

python -m zipapp src -p "/usr/bin/env python3" -o nogoodparser
chmod +x nogoodparser