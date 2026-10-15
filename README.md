# pricing-crosscheck

[![ci](https://github.com/abdulazizbalu/pricing-crosscheck/actions/workflows/ci.yml/badge.svg)](https://github.com/abdulazizbalu/pricing-crosscheck/actions/workflows/ci.yml)

An independent cross-validation harness for derivative pricing models.

This is not another pricing library or VaR engine. It checks independent
implementations of the same model against each other and against closed
forms, measures where and by how much they disagree, and explains why.

Status: scaffold. The protocol, references and measurements are added in
later pull requests.

## Setup

Python 3.10 to 3.12.

```
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -m "not slow"
```

`requirements.lock` pins every package. The exact versions in use are written
into the header of every file under `results/`.
