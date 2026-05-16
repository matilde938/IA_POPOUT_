# PopOut + MCTS + ID3 — IA 2025/2026

Trabalho prático de Inteligência Artificial 2025/2026.
Implementação do jogo **PopOut** (variante de Connect-4 com pop moves), do algoritmo **Monte Carlo Tree Search** com UCB1, e de árvores de decisão **ID3** aplicadas a Iris e a um dataset gerado pelo próprio MCTS (behavioural cloning).

**Deadline:** 17 de maio de 2026, 23:59:59 (Lisboa).

---

## Sumário


- Implementação completa do jogo PopOut, incluindo drops, pops, regra do duplo-4 por pop e repetição tripla.
- Implementação do algoritmo MCTS com UCB1 (`C = √2`), validado com base nos exemplos trabalhados nas aulas.
- Exploração de várias configurações do MCTS, incluindo diferentes rollout policies (`random`, `heuristic_win`, `heuristic_block`), valores de `N`, `C` e limitação de filhos (`max_children`).
- Adição de um tactical lookahead à raiz (2-ply) para deteção de vitórias forçadas através de pops.
- Implementação do algoritmo ID3 de raiz, sem utilização de `scikit-learn`, com três estratégias de discretização.
- Resultados no dataset Iris com accuracy média de 94.7% ± 4% em validação cruzada 5-fold.
- Geração automática de um dataset de PopOut através de self-play com MCTS.
- Treino de uma árvore de decisão para PopOut, obtendo tempos de decisão significativamente inferiores ao MCTS.
- Desenvolvimento de uma interface gráfica em Pygame com interação por cliques e modo MCTS vs Tree.
## Estrutura

```
IA_WORK/
├── README.md                  (este ficheiro)
├── docs/
│   ├── IA_2526_Project.pdf    (enunciado oficial)
│   └── iris (2).csv           (dataset Iris original)
├── codes/                     (código + dados + notebook)
│   ├── popout.py              (motor do jogo)
│   ├── mcts.py                (MCTS + variações + tactical lookahead)
│   ├── decision_tree_builder.py  (ID3 + discretização + tree_strategy)
│   ├── game.py                (CLI + estratégias)
│   ├── gui.py                 (interface Pygame)
│   ├── generate_dataset.py    (geração do dataset PopOut)
│   ├── train_tree.py          (treino + avaliação da árvore PopOut)
│   ├── evaluation.py          (win-rate matrix + charts)
│   ├── mcts_variations.py     (sweep de variações MCTS)
│   ├── popout_dataset.csv
│   ├── decision_tree.pkl
│   ├── content/               (PNGs das figuras)
│   └── iris/                  (pipeline Iris isolada)
│       ├── iris_test.py
│       ├── iris.csv
│       └── iris_tree.pkl
```

## Instalação

```bash
pip install numpy pandas matplotlib pygame jupyter
```

## Como correr

Tudo a partir de `codes/`:

```bash
cd codes
```

### Jogar — interface gráfica

```bash
python gui.py
```

7 modos: Human vs Human, Human vs MCTS (Easy/Medium/Hard), MCTS vs MCTS, Human vs Tree, MCTS vs Tree.

### Jogar — CLI

```bash
python game.py
```

Atalhos do prompt: `0..6` = drop, `d 3`/`p 0` = explícito, `q` = resign, `?` = ajuda.

### Reproduzir as experiências

```bash
# Variações do MCTS 
python mcts_variations.py --quick

# Iris
python iris_test.py

# Gerar dataset PopOut 
python generate_dataset.py --games 1000 

# Treinar árvore PopOut + matches 
python train_tree.py --sweep --vs-random 10 --vs-mcts 6

# Avaliação experimental + charts 
python evaluation.py --quick
```

## Principais características do trabalho

| Aspeto | Implementação |
|---|---|
| Jogo | Implementação completa de PopOut com drops, pops, regra do duplo-4 e repetição tripla |
| MCTS (UCB1) | Implementado e validado numericamente com exemplos das aulas |
| Melhorias ao MCTS | Constante dinâmica, rollout heurístico, `max_children` e lookahead tático a 2 jogadas |
| ID3 no dataset Iris | Accuracy média de 94.7% ± 4% em validação cruzada 5-fold |
| ID3 no dataset PopOut | Accuracy de 22.7% com profundidade máxima 10 |
| Avaliação experimental | Matriz de win-rate, heatmap, learning curve e gráficos estatísticos |
| Interface | Linha de comandos e GUI interativa em Pygame |