Contributing
============

## Issue Tracking

New feature requests, changes, enhancements, non-methodology features, and bug reports can be filed as new issues in the
[GitHub.com issue tracker](https://github.com/brightwind-dev/brightwind/issues) at any time. Please be sure to fully describe the
issue.


## Repository

The brightwind repository is hosted on GitHub, and located here: https://github.com/brightwind-dev/brightwind

This repository is organized using a modified git-flow system. Branches are organized as follows:

- master: Stable release version. Must have good test coverage and may not have all the newest features.
- dev: Development branch which contains the newest features. Tests must pass, but code may be unstable.
- feature/xxx: Branch from dev, should reference a GitHub issue number.

To work on a feature, first create an issue exaplaining the issue or idea and clone the repository locally when you are ready to get to work.

Once you have successfully cloned, create a feature branch from the dev branch. Please name your branch with the issue number you are working from e.g. `iss255_fix_bug`. Work out of this feature branch and push changes to it before submitting a pull request.

Commit messages should contain the issue number, preceded with a #, so these commits can automatically show up in the issue conversation e.g. `git commit -m "iss #255 bug fixed"`.

Be sure to periodically synchronize the upstream dev branch into your feature branch to avoid conflicts in the pull request. You can continue to iterate on your feature branch until you are ready to submit your pull request.

When the feature branch is ready, make a pull request to the dev branch through the GitHub.com UI.

## Pull Request

Pull requests must be made for any changes to be merged into release branches.
They must include updated documentation and pass all unit tests and integration tests.
In addition, code coverage should not be negatively affected by the pull request.

**Scope:**
Encapsulate the changes of ideally one, or potentially a couple, issues.
It is greatly preferable to submit three small pull requests than it is to submit one large pull request.
Write a complete description of these changes in the pull request body.

**Tests:**
The contributor should write a test to cover the new changes.

**Documentation:**
Function docstrings should be fully updated to explain any behaviour changes to the user along with examples. For complicated logic some inline comments are encouraged.

**Changelog:** For pull requests that encapsulate a user-facing feature, or is significant to other contributors for some other reason, please add a line to CHANGELOG.md in the [Unreleased] section.
