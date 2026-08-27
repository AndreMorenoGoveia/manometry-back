# Vendored modules

These files are copied verbatim from the two upstream repositories so the API
scores an exam with exactly the same arithmetic that produced the checkpoint.
Editing them here would silently detach inference from training; change them
upstream and re-sync instead.

| file | origin | commit |
|---|---|---|
| `swallow_crop.py` | `manometry-data-unifier` · `unifier/images.py` | `fd6ab6f` |
| `rtf_laudo.py` | `manometry-data-unifier` · `unifier/rtf.py` | `fd6ab6f` |
| `chicago.py` | `manometry-classification` · `experiments/chicago.py` | `fdd46eb` |
| `image_chicago.py` | `manometry-classification` · `experiments/image_chicago.py` | `fdd46eb` |
| `fusion_models.py` | `manometry-classification` · `experiments/fusion_models.py` | `fdd46eb` |

The only edit applied on copy is the import line of `image_chicago.py`
(`from chicago import ...` -> `from .chicago import ...`), because upstream runs
these as flat scripts and here they live inside a package.

Re-sync:

```bash
scripts/sync_vendor.sh /path/to/manometry-classification /path/to/manometry-data-unifier
```

`tests/test_vendor_parity.py` fails if a vendored file drifts from its origin
while both repositories are present on the machine.
