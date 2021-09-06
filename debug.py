import sys
import os
from allennlp.commands import main

config_file = ""
serialization_dir = ""

sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "models_code",
]
if os.path.exists(serialization_dir):
    sys.argv.append("--force")

main()