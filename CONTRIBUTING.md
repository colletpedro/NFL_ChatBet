# Contributing to NFL Game Prediction Model

🎉 Thank you for considering contributing to this project!

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code.

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/nfl-prediction.git
   cd nfl-prediction
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/originalowner/nfl-prediction.git
   ```
4. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Process

### 1. Open an Issue First

Before starting work:
- Check existing issues to avoid duplication
- Open a new issue describing the problem/feature
- Wait for discussion and approval
- Reference the issue in your PR

### 2. Development Workflow

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/issue-123-description

# Make changes and test
# ... edit files ...
pytest tests/

# Commit changes
git add .
git commit -m "feat: add new feature (closes #123)"

# Push to your fork
git push origin feature/issue-123-description
```

### 3. Testing Requirements

- All new features must include tests
- Maintain or improve code coverage (minimum 80%)
- Run the full test suite before submitting:
  ```bash
  pytest --cov=src --cov-report=term-missing
  ```

## Pull Request Process

### PR Checklist

- [ ] Issue created and linked
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] All tests passing
- [ ] Branch is up-to-date with main
- [ ] Commits are atomic and well-described

### PR Template

Use this template for your PR description:

```markdown
### Context
Closes #<issue_number>

### What was done
- Brief description of changes
- List key modifications

### How to validate
- Steps to test the changes
- Expected outcomes

### Performance impact
- Any performance considerations
- Benchmark results if applicable

### Screenshots (if applicable)
- UI changes
- Output examples
```

## Style Guidelines

### Python Code Style

- Follow PEP 8
- Use Black for formatting: `black src/ tests/`
- Use isort for imports: `isort src/ tests/`
- Type hints are required for all functions
- Maximum line length: 88 characters

### Documentation

- Use Google-style docstrings
- Update README.md for user-facing changes
- Add inline comments for complex logic
- Keep documentation up-to-date

### Code Quality Tools

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
pylint src/

# Type checking
mypy src/

# Security checks
bandit -r src/
```

## Commit Messages

### Format

Follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `perf`: Performance improvements
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Other changes

### Examples

```bash
# Good
git commit -m "feat(models): add ensemble voting classifier"
git commit -m "fix(data): handle missing values in weather data"
git commit -m "docs: update API documentation"

# Bad
git commit -m "fixed stuff"
git commit -m "WIP"
git commit -m "update"
```

## Questions?

Feel free to:
- Open an issue for discussion
- Ask in PR comments
- Contact maintainers

Thank you for contributing! 🚀
