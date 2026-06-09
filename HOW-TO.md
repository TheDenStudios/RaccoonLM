# 🦝 RaccoonLM — Guide d'Installation & Utilisation

## Prérequis

- **Python 3.10+** (recommandé: 3.12)
- **Ollama** installé et en cours d'exécution
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen3:4b  # ou le modèle de ton choix
  ```
- **Optionnel:** GPU support (Vulkan pour AMD, CUDA pour NVIDIA)
  ```bash
  sudo apt install mesa-vulkan-drivers  # AMD
  # ou: sudo apt install nvidia-cuda-toolkit  # NVIDIA
  ```

## Installation Rapide

```bash
# 1. Cloner le dépôt
git clone <repo-url> raccoonlm
cd raccoonlm

# 2. Créer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer
python main.py
```

Ouvre **http://localhost:5555** dans ton navigateur.

## Installation Détaillée

### 1. Dépendances Python

```bash
pip install fastapi uvicorn ollama httpx beautifulsoup4 huggingface_hub psutil flask flask-cors
```

Ou via le fichier `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Configurer Ollama pour le GPU

```bash
# AMD / Intel (Vulkan)
sudo apt install mesa-vulkan-drivers libvulkan1
sudo systemctl edit ollama
# Ajouter: Environment="OLLAMA_VULKAN=1"
sudo systemctl restart ollama

# NVIDIA (CUDA)
sudo apt install nvidia-cuda-toolkit
# Ollama détecte CUDA automatiquement
```

### 3. Service systemd (auto-démarrage)

```bash
cp raccoonlm.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now raccoonlm
```

## Utilisation

### Chat UI

Ouvre **http://localhost:5555** — interface complète avec:
- Sidebar gauche: conversations
- Centre: chat avec streaming
- Sidebar droite: modèles, paramètres, HF Hub

### API Endpoint (OpenAI-compatible)

1. Dans la sidebar droite, section **🔌 API Endpoint**
2. Clique **Start**
3. Connecte ton outil (OpenClaw, etc.) à:
   ```
   Base URL: http://localhost:5556/v1
   Model: qwen3:4b
   ```

### Chercher des modèles

1. Section **🔍 HuggingFace Hub**
2. Tape "qwen" ou "llama" ou "mistral"
3. Clique sur un résultat pour voir les fichiers
4. Clique 📥 pour télécharger

### System Prompts

- **🦝 Raccoon** — Preset par défaut (assistant direct, sans fluff)
- **✏️ Custom** — Ton propre prompt personnalisé
- **💾 Save** — Sauvegarde ton custom prompt

## API Reference

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/health` | GET | Status du service |
| `/api/models` | GET | Liste des modèles Ollama |
| `/api/models/load` | POST | Charger un modèle |
| `/api/models/unload` | POST | Décharger le modèle |
| `/api/chat` | POST | Chat non-streaming |
| `/api/chat/stream` | POST | Chat streaming SSE |
| `/api/conversations` | GET/POST | Gérer conversations |
| `/api/prompts` | GET/POST | Gérer system prompts |
| `/api/hub/search` | GET | Chercher sur HF |
| `/api/hub/files` | GET | Fichiers d'un repo HF |
| `/api/hub/download` | POST | Télécharger un modèle |
| `/api/hardware` | GET | Infos GPU/CPU |
| `/api/resources` | GET | CPU/RAM/VRAM en direct |
| `/api/endpoint/start` | POST | Démarrer endpoint OpenAI |

## Dépannage

### "Ollama unreachable"
```bash
systemctl status ollama
# Ou: ollama serve
```

### GPU non détecté
```bash
ollama ps  # Vérifier PROCESSOR: GPU ou CPU
systemctl cat ollama | grep OLLAMA_VULKAN
```

### Port déjà utilisé
```bash
# Changer le port dans config.py ou via env:
RACCOONLM_PORT=5556 python main.py
```
