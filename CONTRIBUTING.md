# Contributing

Contributions are welcome, and they are greatly appreciated! Every little helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs to [our issue page][gh-issues]. If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

skfolio could always use more documentation, whether as part of the official docs, in docstrings, or even on the tutorials.

### Submit Feedback

The best way to send feedback is via [our issue page][gh-issues] on GitHub. If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome 😊

## Get Started!

Ready to contribute? Here's how to set yourself up for local development.

1. Fork the repo on GitHub.

2. Clone your fork locally:

   ```shell
   $ git clone git@github.com:your_name_here/skfolio.git
   ```

3. Install the project in development mode with the tests and linting dependencies:

   ```shell
   $ pip install --editable '.[tests]'
   ```

4. Create a branch for local development:

   ```shell
   $ git checkout -b name-of-your-bugfix-or-feature
   ```
   Now you can make your changes locally.

   To name your branch, you can use the convention: 
   `category/reference/description-in-kebab-case`
   with category: `feat`, `fix`, `refactor`, `chore` and reference: 
   `issue-<issue number>` or `no-ref`. For example: `feature/issue-34/factor-model`


5. Add unit tests for your implementation and check that your changes pass all tests:

   ```shell
   $ pytest
   ```

6. Then run linting checks with :

   ```shell
   $ ruff
   ```

7. Commit your changes and push your branch to GitHub:

   ```shell
   $ git add .
   $ git commit -m "feat(something): your detailed description of your changes"
   $ git push origin name-of-your-bugfix-or-feature
   ```

   Note: the commit message should follow [the conventional commits](https://www.conventionalcommits.org).
 
8. Submit a pull request through the GitHub website or using the GitHub CLI:

   ```shell
   $ gh pr create --fill
   ```

## Pull Request Guidelines

We like to have the pull request open as soon as possible, that's a great place to discuss any piece of work, even unfinished. You can use draft pull request if it's still a work in progress. Here are a few guidelines to follow:

1. Include tests for feature or bug fixes.
2. Update the documentation for significant features.
3. Ensure tests are passing on CI.


[gh-issues]: https://github.com/skfolio/skfolio/issues
