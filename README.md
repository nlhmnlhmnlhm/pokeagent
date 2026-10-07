## Pokéagent 

Fork of [Seth Karten's Pokeagent Speedrun repo](https://github.com/sethkarten/pokeagent-speedrun)

### Results

A screenshot verifier that flags ineffective moves, a "threaded" client, per-objective hints, randomized prompts, a scripted clock objective, and a map-formatting fix. Try to implement A* pathfinding via PATHFIND(X,Y) which do not work.

![Agent playing Pokémon Emerald](img/leaderboard_11.png)


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
