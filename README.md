# pricing-crosscheck

[![ci](https://github.com/abdulazizbalu/pricing-crosscheck/actions/workflows/ci.yml/badge.svg)](https://github.com/abdulazizbalu/pricing-crosscheck/actions/workflows/ci.yml)

A cross-validation harness for derivative pricing models.

This is not another pricing library or VaR engine. The goal is to check
independent implementations of the same model against each other and
against closed forms, measure where and by how much they disagree, and
explain why.

Status: scaffold only. The protocol, reference implementations and
measurements are added in later pull requests; none of the above is
implemented yet.

## Setup

Python 3.10 to 3.12. Run from a clone (editable install):

```
python -m pip install --no-deps -r requirements.lock
python -m pip install --no-deps -e .
python -m pricing_crosscheck verify-env
python -m pytest -m "not slow"
```

`requirements.lock` pins every Python package that is installed (build tools
and the BLAS bundled inside the numpy/scipy wheels are not pinned
separately). Each CSV under `results/` starts with comment lines holding the
content hash, the Python version, the platform and the versions of QuantLib,
numpy, scipy and matplotlib; `manifest.json` lists every installed
distribution. Committed results are generated with CPython 3.12 on
linux-x86_64.
