---
id: contributing
title: Contributing
sidebar_label: Contributing
sidebar_position: 6
---

# Contributing to Open Telco

We welcome contributions from the community! Open Telco is an open-source project and we appreciate all forms of contributions.

## Ways to Contribute

<div className="contrib-grid">
  <div className="contrib-card">
    <div className="contrib-icon">🐛</div>
    <h3>Report Issues</h3>
    <p>Found a bug? Open an issue on <a href="https://github.com/gsma-research/open_telco/issues">GitHub</a>.</p>
  </div>
  <div className="contrib-card">
    <div className="contrib-icon">💡</div>
    <h3>Request Features</h3>
    <p>Have an idea? Share it in GitHub Issues or Discussions.</p>
  </div>
  <div className="contrib-card">
    <div className="contrib-icon">📝</div>
    <h3>Improve Docs</h3>
    <p>Fix typos, add examples, or write tutorials.</p>
  </div>
  <div className="contrib-card">
    <div className="contrib-icon">🔬</div>
    <h3>Add Benchmarks</h3>
    <p>Contribute new telecom-specific evaluations.</p>
  </div>
</div>

## Code Contributions

### Quick Start

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/open_telco.git
cd open_telco

# 2. Install dependencies
pip install uv
uv sync --all-extras

# 3. Install pre-commit hooks
pre-commit install

# 4. Create a branch
git checkout -b feature/your-feature

# 5. Make changes and test
uv run pytest

# 6. Submit PR
git push origin feature/your-feature
```

### Code Style

We use these tools for code quality:

| Tool | Purpose |
|------|---------|
| **Ruff** | Linting and formatting |
| **MyPy** | Type checking |
| **Pre-commit** | Automated checks before commit |

Pre-commit hooks run automatically. To run manually:

```bash
pre-commit run --all-files
```

## Adding New Evaluations

Want to contribute a new benchmark? Follow these steps:

1. **Review existing implementations** in `src/open_telco/`
2. **Follow Inspect AI patterns** - see [their docs](https://inspect.aisi.org.uk/)
3. **Include documentation** - README, docstrings, usage examples
4. **Add tests** - ensure your eval works correctly
5. **Update the registry** in `_registry.py`

### Evaluation Requirements

Your benchmark should:

- Target telecommunications-specific skills
- Have clear scoring criteria
- Be reproducible
- Include a dataset (or link to one)

## Documentation

We welcome doc improvements:

- Fix typos or unclear content
- Add tutorials or guides
- Improve code examples
- Translate content

Edit docs in `website/tabs/user-guide/docs/` and submit a PR.

## Community

- **GitHub Discussions:** Ask questions, share ideas
- **GitHub Issues:** Report bugs, request features
- **Email:** [emolero@gsma.com](mailto:emolero@gsma.com)

## License

By contributing to Open Telco, you agree that your contributions will be licensed under the MIT License.
