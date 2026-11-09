"""Export the local API contract without starting a server or reading a study."""

import argparse
import json
from pathlib import Path

from mrax.web.app import create_app

parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
path = Path(__file__).resolve().parents[1] / "openapi.json"
schema = json.dumps(create_app().openapi(), sort_keys=True, indent=2) + "\n"
if args.check:
    if not path.is_file() or path.read_text() != schema:
        parser.exit(1, "API schema changed; run npm run schema\n")
else:
    path.write_text(schema)
