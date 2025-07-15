# Contributing

Instructions:

1. [Fork](https://github.com/carlosgmartin/synax/fork) the repository.

2. Clone the repo:

```shell
git clone https://github.com/YOUR_USERNAME/synax
cd synax
pip install -e .
```

3. Add the original repo as an upstream remote:

```shell
git remote add upstream https://github.com/carlosgmartin/synax
```

4. Create a branch:

```shell
git checkout -b name-of-change
```

5. Implement your changes.

6. Lint, typecheck, and run tests:

```shell
ruff check && ruff format && pyright && pytest
```

7. Add files and create a commit:

```shell
git add file1.py file2.py ...
git commit -m "Your commit message."
```

8. Sync your code with the original repo, resolving any conflicts:

```shell
git fetch upstream
git rebase upstream/main
```

9. Push your commit:

```shell
git push --set-upstream origin name-of-change
```

10. Go to [the repo](https://github.com/carlosgmartin/synax) and create a pull request.
