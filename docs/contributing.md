# Contributing

Instructions:

1. [Fork](https://github.com/carlosgmartin/synax/fork) the repository.

2. Clone the repo:

```shell
git clone https://github.com/YOUR_USERNAME/synax
cd synax
```

3. Create an editable install:

```shell
pip install -e .
```

4. Add the original repo as an upstream remote:

```shell
git remote add upstream https://github.com/carlosgmartin/synax
```

5. Create a branch:

```shell
git checkout -b name-of-change
```

6. Implement your changes.

7. Lint, typecheck, and run tests:

```shell
ruff check && ruff format && pyright && pytest
```

8. Add files and create a commit:

```shell
git add file1.py file2.py ...
git commit -m "Your commit message."
```

9. Sync your code with the original repo, resolving any conflicts:

```shell
git fetch upstream
git rebase upstream/main
```

10. Push your commit:

```shell
git push --set-upstream origin name-of-change
```

11. Go to [the repo](https://github.com/carlosgmartin/synax) and create a pull request.
