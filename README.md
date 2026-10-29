# Fox and Hounds

## How to run

### Install necessary tools

```bash
sudo apt update && sudo apt install python3 pipx
pipx ensurepath
pipx install uv
```

### Sync the environment and run the project

```bash
uv sync
uv run python src/main.py
```
