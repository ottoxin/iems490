# Contributing to IEMS490

Thank you for your interest in contributing to the IEMS490 course repository! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find bugs, typos, or have suggestions for improvements:

1. Check if the issue already exists in the GitHub Issues
2. If not, create a new issue with:
   - Clear, descriptive title
   - Detailed description of the problem or suggestion
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Your environment (OS, Python version, etc.)

### Submitting Changes

1. **Fork the Repository**
   ```bash
   # Click "Fork" on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/iems490.git
   cd iems490
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make Your Changes**
   - Write clear, commented code
   - Follow existing code style
   - Test your changes thoroughly
   - Update documentation as needed

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Brief description of your changes"
   ```

5. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Provide a clear description of your changes
   - Link any related issues

## Contribution Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Include type hints where appropriate
- Keep functions focused and modular

### Documentation

- Update README files for any new features
- Add comments for complex logic
- Include examples where helpful
- Ensure all links work
- Keep formatting consistent

### Adding Course Materials

#### New Lectures
- Place in `lectures/` directory
- Follow existing naming convention
- Include both slides and code examples
- Update lecture README with description

#### New Assignments
- Place in `assignments/` directory
- Include problem description, starter code, and solution
- Provide clear instructions
- Add to assignments README

#### New Code Examples
- Place in appropriate `code/` subdirectory
- Include clear comments and documentation
- Test thoroughly
- Add example to code README

#### New Resources
- Add to `resources/README.md`
- Include proper citations
- Verify all links
- Categorize appropriately

### Testing

Before submitting a pull request:

1. Test all code changes
   ```bash
   python script_name.py
   # or
   jupyter notebook notebook_name.ipynb
   ```

2. Verify no broken links in documentation

3. Ensure code runs with specified dependencies

4. Check that all cells in notebooks run successfully

### Commit Messages

Write clear commit messages:
- Use present tense ("Add feature" not "Added feature")
- Be descriptive but concise
- Reference issues when applicable (#123)

Good examples:
```
Add transformer implementation example
Fix typo in week 5 lecture notes
Update README with installation instructions
Add dataset preprocessing script
```

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information
- Other unprofessional conduct

## Questions?

If you have questions about contributing:
- Open an issue for discussion
- Contact the course instructor
- Ask in the course forum

## Recognition

Contributors will be acknowledged in the repository and may be mentioned in course materials (with permission).

Thank you for helping improve IEMS490!
