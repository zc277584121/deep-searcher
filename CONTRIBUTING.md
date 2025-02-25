# Contributing to DeepSearcher

Contributions to DeepSearcher are welcome from everyone. We strive to make the contribution process simple and straightforward.

The following are a set of guidelines for contributing to DeepSearcher. Following these guidelines makes contributing to this project easy and transparent. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

**Content**

- [Contributing to DeepSearcher](#contributing-to-deepsearcher)  
  - [How can you contribute?](#how-can-you-contribute)  
    - [Contributing code](#contributing-code)  
    - [GitHub workflow](#github-workflow)  
    - [General guidelines](#general-guidelines)  
    - [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)  
  - [Coding Style](#coding-style)  
  - [Commits and PRs](#commits-and-prs)  

## How can you contribute?

### Contributing code

**If you encounter a bug, you can**

- (**Recommended**) File an issue about the bug.
- Provide clear and concrete ways/scripts to reproduce the bug.
- Provide possible solutions for the bug.
- Pull a request to fix the bug.

**If you're interested in existing issues, you can**

- (**Recommended**) Provide answers for issue labeled `question`.
- Provide help for issues labeled `bug`, `improvement`, and `enhancement` by
  - (**Recommended**) Ask questions, reproduce the issue, or provide solutions.
  - Pull a request to fix the issue.

**If you require a new feature or major enhancement, you can**

- (**Recommended**) File an issue about the feature/enhancement with reasons.
- Provide an MEP for the feature/enhancement.
- Pull a request to implement the MEP.

**If you are a reviewer/approver of DeepSearcher, you can**

- Participate in PR review process.
- Instruct newcomers in the community to complete the PR process.

If you want to become a contributor of DeepSearcher, submit your pull requests! For those just getting started, see [GitHub workflow](#github-workflow) below.

All submissions will be reviewed as quickly as possible.
There will be a reviewer to review the codes, and an approver to review everything aside the codes.
If everything is perfect, the reviewer will label `/lgtm`, and the approver will label `/approve`.
Once the 2 labels are on your PR, and all actions pass, your PR will be merged into base branch automatically by our @sre-ci-robot

### GitHub workflow

Generally, we follow the "fork-and-pull" Git workflow.

1.  [Fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) the repository on GitHub.
2.  Clone your fork to your local machine with `git clone git@github.com:<yourname>/deep-searcher.git`.
3.  Create a branch with `git checkout -b my-topic-branch`.
4.  [Commit](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/committing-changes-to-a-pull-request-branch-created-from-a-fork) changes to your own branch, then push to GitHub with `git push origin my-topic-branch`.
5.  Submit a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) so that we can review your changes.

Remember to [sync your forked repository](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo#keep-your-fork-synced) _before_ submitting proposed changes upstream. If you have an existing local repository, please update it before you start, to minimize the chance of merge conflicts.

```shell
git remote add upstream git@github.com:zilliztech/deep-searcher.git
git fetch upstream
git checkout upstream/master -b my-topic-branch
```

### General guidelines

Before submitting your pull requests for review, make sure that your changes are consistent with the coding style.

### Developer Certificate of Origin (DCO)

All contributions to this project must be accompanied by acknowledgment of, and agreement to, the [Developer Certificate of Origin](https://developercertificate.org/). Acknowledgment of and agreement to the Developer Certificate of Origin _must_ be included in the comment section of each contribution and _must_ take the form of `Signed-off-by: {{Full Name}} <{{email address}}>` (without the `{}`). Contributions without this acknowledgment will be required to add it before being accepted. If contributors are unable or unwilling to agree to the Developer Certificate of Origin, their contribution will not be included.

Contributors sign-off that they adhere to DCO by adding the following Signed-off-by line to commit messages:

```text
This is my commit message

Signed-off-by: Random J Developer <random@developer.example.org>
```

Git also has a `-s` command line option to append this automatically to your commit message:

```shell
$ git commit -s -m 'This is my commit message'
```

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

## Commits and PRs

- Commit message and PR description style: refer to [good commit messages](https://chris.beams.io/posts/git-commit)
