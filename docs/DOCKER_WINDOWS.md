# Docker Desktop (Windows) — install and test

This document shows step-by-step commands to install Docker Desktop on Windows and run the project's docker-compose setup locally.

1. Install Docker Desktop

- Download Docker Desktop for Windows: https://www.docker.com/get-started
- During install you may be prompted to enable WSL 2. Follow the installer prompts. If you need manual WSL2 install guidance see Microsoft docs: https://docs.microsoft.com/windows/wsl/install
- After installation, start Docker Desktop and sign in if requested.

2. Verify Docker CLI is available

Open PowerShell and run:

```powershell
docker --version
docker compose version
```

If these commands print versions, Docker is available.

3. Prepare requirements

Make sure `requirements.txt` contains required runtime packages. If you maintain a venv and want to export your environment:

```powershell
# in your activated venv
pip freeze > requirements.txt
```

4. Run the project's compose setup

From the project root run the included helper script (recommended):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_docker_compose.ps1
```

Or run the docker compose commands manually:

```powershell
# build and start (detached)
docker compose up --build -d
# or legacy binary if above fails
# docker-compose up --build -d

# check containers
docker compose ps

# tail logs
docker compose logs --tail 200 backend
docker compose logs --tail 200 frontend

# stop & remove
docker compose down
```

5. Verify services

- Backend root: http://localhost:8000/ returns a JSON status.
- Streamlit frontend: http://localhost:8501

6. If you see errors

- "docker: command not found" — install Docker Desktop / ensure `docker` on PATH.
- Missing models: ensure `final/results/<payoff>` contains `model.joblib` and `scaler.joblib`; the compose mounts `./final/results` into the container.
- Port conflicts: change mappings in `docker-compose.yml`.

If the containers fail to start, paste the logs from `docker compose logs backend` and `docker compose logs frontend` here and I'll help debug.
