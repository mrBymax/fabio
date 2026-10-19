```
  ______ ___     ____  ____ ____
 / ____//   |   / __ )/  _// __ \
/ /_   / /| |  / __  |/ / / / / /
/ __/  / ___ | / /_/ // / / /_/ /
/_/    /_/  |_|/_____//___/\____/
```

# F.A.B.I.O.

**Fantacalcio Automated Builder & Integer Optimizer**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Algorithm](https://img.shields.io/badge/Algorithm-MILP%20Knapsack-orange.svg)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)

F.A.B.I.O. is a mathematical optimization engine that solves variations of the **Multi-Choice 0/1 Knapsack Problem** applied to Italian Serie A Fantasy Football (_Fantacalcio_).

By leveraging **Mixed-Integer Linear Programming (MILP)**, F.A.B.I.O. determines the globally optimal 25-player roster within budget constraints, and re-computes your optimal drafting strategy in real-time as the auction unfolds.

---

## Features

- **Dynamic Live Re-Optimization:** Re-solves the MILP problem in $<20\text{ ms}$ after every draft pick (`buy`) or rival acquisition (`pass`), dynamically reallocating your remaining budget to maximize remaining expected value (FVM).
- **Dual Engine Architecture:**
  - **Python Live Assistant (`src/auction_cli.py`):** Fast, interactive terminal tool powered by `scipy.optimize.milp` and `rich`.
  - **MATLAB Analytical Suite (`src/phase_1.m`, `src/simulate_auction.m`):** Monte Carlo draft simulations, What-If scenario comparisons, and goalkeeper tiering analysis using `intlinprog`.
- **Fuzzy Player Matching:** Instant player lookup by ID, exact name, substring, or fuzzy spellings (e.g. `buy lautaro 175`).
- **Interactive Slash Commands:** Supports both standard and slash commands (`/targets`, `/buy`, `/pass`, `/roster`, `/undo`, `/clear`, `/help`).
- **Demo Dataset Out-of-the-Box:** Includes a ready-to-run demo dataset (`dataset/demo_quotazioni.csv`) so you can clone and test immediately without scraping.

---

## Maths for this Problem

Let $N$ be the total pool of available Serie A players. Let $x \in \{0, 1\}^N$ be the binary decision vector:

$$x_i = \begin{cases} 1 & \text{if player } i \text{ is drafted} \\ 0 & \text{otherwise} \end{cases}$$

### Objective

Maximize the total expected value (Fantavalore di Mercato, $FVM$) of the drafted roster:

$$\max \sum_{i=1}^{N} FVM_i \cdot x_i \iff \min_{x} (-FVM)^T x$$

### Constraints

1. **Budget Constraint (Linear Inequality: $A \cdot x \le b$):**
   Total cost cannot exceed the remaining auction budget (e.g. 500 crediti):
   $$\sum_{i=1}^{N} \text{Price}_i \cdot x_i \le \text{Budget}_{\text{remaining}}$$

2. **Positional Department Quotas (Linear Equality: $A\_{eq} \cdot x = b\_{eq}$):**
   Exactly 25 players adhering to standard Italian fantasy roster rules:
   $$\sum_{i \in \text{GK}} x_i = 3, \quad \sum_{i \in \text{DEF}} x_i = 8, \quad \sum_{i \in \text{MID}} x_i = 8, \quad \sum_{i \in \text{FWD}} x_i = 6$$

3. **Binary Integrality:**
   $$x_i \in \{0, 1\} \quad \forall i \in \{1, \dots, N\}$$

---

## Quick Start

### 1. Installation

Clone the repository:

```bash
git clone https://github.com/federicobertossi/fabio.git
cd fabio
```

Set up your virtual environment using [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`:

```bash
# Using uv (fastest)
uv sync

# Or using standard python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Launch the Live Auction Assistant

Start the interactive assistant directly:

```bash
python src/auction_cli.py
```

_(If you have `uv` installed, you can also run: `uv run fabio`)_

> [!NOTE]
> If live scraped data is not present, F.A.B.I.O. automatically boots using the included `dataset/demo_quotazioni.csv`.

---

## Commands Reference

Inside the interactive assistant, enter commands or slash commands:

| Command   | Aliases              | Description                                             | Example           |
| :-------- | :------------------- | :------------------------------------------------------ | :---------------- |
| `targets` | `t`, `/targets`      | Solves MILP and displays the updated optimal squad      | `targets`         |
| `buy`     | `b`, `/buy`          | Mark a player as bought by you with price paid          | `buy lautaro 175` |
| `pass`    | `p`, `/pass`         | Mark a player as taken by a rival or skipped            | `pass vlahovic`   |
| `roster`  | `r`, `/roster`       | View your current drafted squad, spend, and differences | `roster`          |
| `search`  | `s`, `/search`       | Inspect player statistics and availability status       | `search barella`  |
| `undo`    | `u`, `/undo`         | Reverts the last purchase or pass action                | `undo`            |
| `status`  | `/status`            | Displays remaining budget and role quotas needed        | `status`          |
| `clear`   | `c`, `/clear`        | Clears screen and re-renders the header banner          | `clear`           |
| `quit`    | `q`, `exit`, `/quit` | Exits the assistant                                     | `quit`            |

---

## Data Scraper

To scrape updated Serie A valuations directly from Fantacalcio's (unofficial?) sources:

```bash
python scraper/scraper.py
```

This generates `dataset/quotazioni_fantacalcio.csv`. When present, F.A.B.I.O. automatically uses this live dataset instead of the demo dataset.

> [!IMPORTANT]
> Live player datasets are intentionally excluded.

---

## MATLAB Simulation Suite

For pre-season strategic modeling, two MATLAB models are available in `src/`:

1. **`src/phase_1.m`**: Solves the macro roster optimization with departmental budget caps and goalkeeper tier constraints (1 Top, 1 Mid, 1 Low reserve).
2. **`src/simulate_auction.m`**: Runs Monte Carlo simulations and What-If strategic comparisons (e.g. Mega-Striker vs Heavy Midfield spine).

Run directly from terminal or MATLAB:

```bash
matlab -batch "run('src/phase_1.m')"
```

---

## Project Structure

```text
fabio/
├── dataset/
│   └── demo_quotazioni.csv   # Public demo dataset (ready-to-use)
├── logo/
│   ├── logo.txt              # ASCII art logo
│   └── logo.jpg              # Graphic emblem
├── scraper/
│   ├── __init__.py
│   └── scraper.py            # Live Fantacalcio quotes scraper
├── src/
│   ├── __init__.py
│   ├── auction_cli.py        # Antigravity-styled live interactive assistant
│   ├── phase_1.m             # Stage 1 Macro MILP optimization (MATLAB)
│   └── simulate_auction.m    # Monte Carlo & What-If scenario engine (MATLAB)
├── .gitignore                # Excludes private data, .env, and caches
├── LICENSE                   # MIT License
├── pyproject.toml            # Packaging & dependencies
└── README.md                 # Project documentation
```

---

## License

This project is licensed under the [MIT License](LICENSE).
