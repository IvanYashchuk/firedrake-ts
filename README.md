# firedrake-ts &middot; [![Build](https://github.com/ivanyashchuk/firedrake-ts/workflows/CI/badge.svg)](https://github.com/ivanyashchuk/firedrake-ts/actions?query=workflow%3ACI+branch%3Amaster) [![codecov](https://codecov.io/gh/IvanYashchuk/firedrake-ts/branch/master/graph/badge.svg)](https://codecov.io/gh/IvanYashchuk/firedrake-ts)
The firedrake-ts library provides an interface to PETSc TS for the scalable solution of DAEs arising from the discretization of time-dependent PDEs.

## Example
Check `examples/` for the examples.

## Installation
First install [Firedrake](https://firedrakeproject.org/download.html).
Then activate firedrake virtual environment with:

    source firedrake/bin/activate

After that install the firedrake-ts with:

    python -m pip install git+https://github.com/IvanYashchuk/firedrake-ts.git@master

## Reporting bugs

If you found a bug, create an [issue].

[issue]: https://github.com/IvanYashchuk/firedrake-ts/issues/new

## Contributing

Pull requests are welcome from everyone.

Fork, then clone the repository:

    git clone https://github.com/IvanYashchuk/firedrake-ts.git

Make your change. Add tests for your change. Make the tests pass:

    pytest tests/

Check the formatting with `black` and `flake8`. Push to your fork and [submit a pull request][pr].

[pr]: https://github.com/IvanYashchuk/firedrake-ts/pulls