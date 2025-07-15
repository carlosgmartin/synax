# Contributing

Instructions:

1. [Fork](https://github.com/carlosgmartin/synax/fork) the repository.

1. Clone the repo:

    ```shell
    git clone https://github.com/YOUR_USERNAME/synax
    cd synax
    ```

1. Perform an editable installation:

    ```shell
    pip install -e .
    ```

1. Add the original repo as an upstream remote:

    ```shell
    git remote add upstream https://github.com/carlosgmartin/synax
    ```

1. Create a branch:

    ```shell
    git checkout -b name-of-change
    ```

1. Implement your change.

1. Lint, typecheck, and run tests:

    ```shell
    ruff check && ruff format && pyright && pytest
    ```

1. Add the files you have modified and create a commit:

    ```shell
    git add file1.py file2.py ...
    git commit -m "Your commit message."
    ```

1. Sync your code with the original repo, resolving any conflicts:

    ```shell
    git fetch upstream
    git rebase upstream/main
    ```

1. Push your commit:

    ```shell
    git push --set-upstream origin name-of-change
    ```

1. Go to [the repo](https://github.com/carlosgmartin/synax) and create a pull request.
