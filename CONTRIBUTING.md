# Contributing to DeepSearcher

We welcome contributions from everyone. This document provides guidelines to make the contribution process straightforward.


## Coding Style

Keeping a consistent style for code, code comments, commit messages, and PR descriptions will greatly accelerate your PR review process.
We highly recommend you run code linter and formatter when you put together your pull requests:

To check the coding styles:

```shell
make lint
```

To fix the coding styles:

```shell
make format
```

## Pull Request Process

1. Fork the repository and create your branch from `master`.
2. Make your changes.
3. Run tests and linting to ensure your code meets the project's standards.
4. Update documentation if necessary.
5. Submit a pull request.

## Developer Certificate of Origin (DCO)

All contributions require a sign-off, acknowledging the [Developer Certificate of Origin](https://developercertificate.org/). 
Add a `Signed-off-by` line to your commit message:

```text
Signed-off-by: Your Name <your.email@example.com>
```