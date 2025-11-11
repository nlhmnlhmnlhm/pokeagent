## Pokéagent 

Fork of [this](https://github.com/sethkarten/pokeagent-speedrun) repo : https://github.com/sethkarten/pokeagent-speedrun

### Setup

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```
uv sync
```
```
source .venv/bin/activate
```
```
pacman -S libmgba
```
```
pokeagent-speedrun/
└── Emerald-GBAdvance/
    └── rom.gba  # Place your Pokémon Emerald ROM file here
```

### Gemini 

```
export GEMINI_API_KEY="your-api-key-here"
```

### Vertex 

```
uv pip install google-genai
```

### Run 

```
python run.py --scaffold simple --agent-auto --backend gemini
```
