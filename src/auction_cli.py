import difflib
import getpass
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from scipy.optimize import Bounds, LinearConstraint, milp

# Enable readline command history & tab-completion if available
try:
    import readline

    COMMANDS = [
        "buy",
        "pass",
        "targets",
        "roster",
        "search",
        "undo",
        "status",
        "clear",
        "help",
        "quit",
        "exit",
        "/buy",
        "/pass",
        "/targets",
        "/roster",
        "/search",
        "/undo",
        "/status",
        "/clear",
        "/help",
        "/quit",
        "/exit",
    ]

    def _completer(text, state):
        matches = [c for c in COMMANDS if c.startswith(text)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(_completer)
    readline.parse_and_bind("tab: complete")
except ImportError, AttributeError:
    pass

console = Console()

__version__ = "0.1.0"

DEFAULT_ASCII_LOGO = """\
  ______ ___     ____  ____ ____
 / ____//   |   / __ )/  _// __ \\
/ /_   / /| |  / __  |/ / / / / /
/ __/  / ___ | / /_/ // / / /_/ /
/_/    /_/  |_|/_____//___/\\____/ """

ROLE_COLORS = {"P": "yellow", "D": "green", "C": "cyan", "A": "red"}
ROLE_NAMES = {
    "P": "Portiere (GK)",
    "D": "Difensore (DEF)",
    "C": "Centrocampista (MID)",
    "A": "Attaccante (FWD)",
}


# ==============================================================================
# OPTIMIZATION & DATA LOGIC
# ==============================================================================


def solve_optimal_squad(
    df: pd.DataFrame,
    remaining_budget: int,
    quota_needed: dict[str, int],  # e.g. {'P': 2, 'D': 6, 'C': 5, 'A': 4}
    excluded_ids: set[str],
) -> pd.DataFrame:
    """Finds the optimal roster to complete remaining quotas within remaining budget using MILP."""
    # 1. Filter out already drafted / excluded players
    available_mask = ~df["Id"].astype(str).isin(excluded_ids)
    candidates = df[available_mask].reset_index(drop=True)
    n_candidates = len(candidates)

    if n_candidates == 0:
        return pd.DataFrame()

    # 2. Objective: Maximize FVM (minimize -FVM)
    c = -candidates["FVM"].values.astype(float)

    # 3. Decision variables: Binary (0 or 1)
    integrality = np.ones(n_candidates)  # 1 = integer
    bounds = Bounds(lb=np.zeros(n_candidates), ub=np.ones(n_candidates))

    # 4. Constraints
    # Constraint A: Budget limit (sum(EstPrice * x) <= remaining_budget)
    prices = candidates["EstPrice"].values.astype(float)

    # Constraint B: Role quotas (sum(isRole * x) == quota_needed[role])
    is_gk = (candidates["R"] == "P").values.astype(float)
    is_def = (candidates["R"] == "D").values.astype(float)
    is_mid = (candidates["R"] == "C").values.astype(float)
    is_fwd = (candidates["R"] == "A").values.astype(float)

    constraint_matrix = np.vstack([prices, is_gk, is_def, is_mid, is_fwd])

    lhs = np.array(
        [
            0,
            quota_needed["P"],
            quota_needed["D"],
            quota_needed["C"],
            quota_needed["A"],
        ]
    )
    rhs = np.array(
        [
            remaining_budget,
            quota_needed["P"],
            quota_needed["D"],
            quota_needed["C"],
            quota_needed["A"],
        ]
    )

    constraints = LinearConstraint(constraint_matrix, lhs, rhs)

    # 5. Solve MILP
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

    if res.success:
        selected_idx = np.where(res.x > 0.5)[0]
        return candidates.iloc[selected_idx].copy()
    else:
        return pd.DataFrame()


def find_player(query: str, df: pd.DataFrame) -> pd.Series | None:
    """Finds a player by ID, exact name match, substring, or fuzzy match."""
    q_clean = str(query).strip().lower()

    # 1. Match by exact ID
    id_match = df[df["Id"].astype(str) == q_clean]
    if not id_match.empty:
        return id_match.iloc[0]

    # 2. Case-insensitive Substring match in Nome
    sub_matches = df[df["Nome"].str.lower().str.contains(q_clean, regex=False)]
    if len(sub_matches) == 1:
        return sub_matches.iloc[0]
    elif len(sub_matches) > 1:
        # If multiple, prefer exact start or highest FVM
        exact_prefix = sub_matches[
            sub_matches["Nome"].str.lower().str.startswith(q_clean)
        ]
        if not exact_prefix.empty:
            return exact_prefix.sort_values(by="FVM", ascending=False).iloc[0]
        return sub_matches.sort_values(by="FVM", ascending=False).iloc[0]

    # 3. Fuzzy matching with difflib
    all_names = df["Nome"].tolist()
    close = difflib.get_close_matches(
        q_clean, [n.lower() for n in all_names], n=1, cutoff=0.5
    )
    if close:
        matched_idx = [n.lower() for n in all_names].index(close[0])
        return df.iloc[matched_idx]

    return None


@dataclass
class Transaction:
    action: str  # 'buy' or 'pass'
    player: dict
    price_paid: int = 0


class AuctionState:
    def __init__(self, df: pd.DataFrame, total_budget: int = 500):
        self.df = df.copy()
        self.total_budget = total_budget
        self.my_squad: list[dict] = []
        self.excluded_ids: set[str] = set()
        self.history: list[Transaction] = []

    @property
    def remaining_budget(self) -> int:
        spent = sum(p["PricePaid"] for p in self.my_squad)
        return self.total_budget - spent

    @property
    def spent_budget(self) -> int:
        return self.total_budget - self.remaining_budget

    @property
    def quotas_needed(self) -> dict[str, int]:
        total_quotas = {"P": 3, "D": 8, "C": 8, "A": 6}
        my_counts = {"P": 0, "D": 0, "C": 0, "A": 0}
        for p in self.my_squad:
            my_counts[p["R"]] += 1
        return {r: max(0, total_quotas[r] - my_counts[r]) for r in total_quotas}

    def buy_player(self, player_row: pd.Series, price_paid: int):
        player_dict = player_row.to_dict()
        player_dict["PricePaid"] = price_paid
        self.my_squad.append(player_dict)
        self.excluded_ids.add(str(player_row["Id"]))
        self.history.append(Transaction("buy", player_dict, price_paid))

    def pass_player(self, player_row: pd.Series):
        player_dict = player_row.to_dict()
        self.excluded_ids.add(str(player_row["Id"]))
        self.history.append(Transaction("pass", player_dict))

    def undo(self) -> tuple[bool, str]:
        if not self.history:
            return False, "Nothing to undo!"
        last = self.history.pop()
        player_id = str(last.player["Id"])
        self.excluded_ids.discard(player_id)
        if last.action == "buy":
            self.my_squad = [p for p in self.my_squad if str(p["Id"]) != player_id]
            return (
                True,
                f"Undid purchase of [bold white]{last.player['Nome']}[/bold white] (+{last.price_paid} crediti)",
            )
        else:
            return (
                True,
                f"Undid pass of [bold white]{last.player['Nome']}[/bold white] (returned to draft pool)",
            )


def load_logo() -> str:
    """Loads an ASCII logo from logo/ directory if available, or returns default."""
    logo_dir = Path(__file__).resolve().parent.parent / "logo"
    if logo_dir.exists():
        for txt_file in logo_dir.glob("*.txt"):
            try:
                content = txt_file.read_text(encoding="utf-8").rstrip()
                if content:
                    return content
            except Exception:
                pass
    return DEFAULT_ASCII_LOGO


def format_path(cwd: str, max_len: int = 42) -> str:
    """Formats current path with ~ and truncates gracefully if needed."""
    home = str(Path.home())
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home) :]
    if len(cwd) > max_len:
        return "..." + cwd[-(max_len - 3) :]
    return cwd


def print_banner():
    """Renders the CLI-style header panel (logo left, metadata right)."""
    raw_logo = load_logo()
    logo_text = Text(raw_logo, style="bold cyan")

    cwd = os.getcwd()
    term_width = console.width
    # Dynamic path length based on terminal width to prevent truncation ellipsis
    max_path_len = max(28, term_width - 48) if term_width else 42
    display_path = format_path(cwd, max_path_len)
    user = getpass.getuser()

    meta_table = Table.grid(padding=(0, 1))
    meta_table.add_column(style="dim", justify="right", no_wrap=True)
    meta_table.add_column(justify="left", no_wrap=True)
    meta_table.add_row("version:", f"[bold white]{__version__}[/bold white]")
    meta_table.add_row("user:", f"[bold white]{user}[/bold white]")
    meta_table.add_row("path:", f"[bold white]{display_path}[/bold white]")
    meta_table.add_row("engine:", "[cyan]MILP Knapsack (scipy.optimize)[/cyan]")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row(logo_text, meta_table)

    banner_panel = Panel(
        grid,
        box=box.ROUNDED,
        border_style="dim cyan",
        padding=(0, 1),
    )
    console.print(banner_panel)


def print_status_bar(state: AuctionState):
    """Prints a sleek status bar showing budget and squad quota breakdown."""
    quotas = state.quotas_needed
    remaining = state.remaining_budget
    total = state.total_budget
    spent = state.spent_budget
    count = len(state.my_squad)

    # Budget styling based on percentage remaining
    budget_pct = remaining / total if total > 0 else 0
    if budget_pct > 0.4:
        budget_style = "bold green"
    elif budget_pct > 0.15:
        budget_style = "bold yellow"
    else:
        budget_style = "bold red"

    status_table = Table.grid(expand=True, padding=(0, 1))
    status_table.add_column(justify="left", no_wrap=True)
    status_table.add_column(justify="right", no_wrap=True)

    left_text = (
        f"💰 [dim]Budget:[/] [{budget_style}]{remaining}[/{budget_style}] / {total} crediti "
        f"[dim]({spent} spent)[/]"
    )
    right_text = (
        f"📋 [dim]Squad:[/] [bold white]{count}[/bold white]/25  "
        f"[dim]│[/]  "
        f"[yellow]P: {quotas['P']}[/yellow]  "
        f"[green]D: {quotas['D']}[/green]  "
        f"[cyan]C: {quotas['C']}[/cyan]  "
        f"[red]A: {quotas['A']}[/red] [dim]needed[/]"
    )

    status_table.add_row(left_text, right_text)

    console.print(
        Panel(status_table, box=box.ROUNDED, border_style="dim", padding=(0, 1))
    )


def render_targets_table(optimal: pd.DataFrame):
    """Renders the optimal squad MILP solution in a clean Rich table."""
    if optimal.empty:
        console.print(
            Panel(
                "[bold red]Infeasible combination![/bold red] No valid squad found within remaining budget.\n"
                "[dim]Tip: Check passed players or spend less on remaining roles.[/dim]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        return

    tot_price = int(optimal["EstPrice"].sum())
    tot_fvm = int(optimal["FVM"].sum())

    table = Table(
        title="🎯 Optimal Target Squad (MILP Solution)",
        box=box.ROUNDED,
        border_style="dim",
        header_style="bold cyan",
        show_footer=True,
        expand=True,
    )

    table.add_column("Role", justify="center", width=5)
    table.add_column("Player", justify="left", ratio=1)
    table.add_column("Team", justify="center", width=6, style="dim")
    table.add_column(
        "Est. Price", justify="right", width=11, footer=f"[bold]{tot_price} cr[/bold]"
    )
    table.add_column("FVM", justify="right", width=8, footer=f"[bold]{tot_fvm}[/bold]")

    for role in ["P", "D", "C", "A"]:
        sub = optimal[optimal["R"] == role].sort_values(by="FVM", ascending=False)
        if not sub.empty:
            table.add_section()
            for _, row in sub.iterrows():
                c = ROLE_COLORS[role]
                table.add_row(
                    f"[{c}]{role}[/{c}]",
                    f"[bold white]{row['Nome']}[/bold white]",
                    row["Squadra"],
                    f"{row['EstPrice']} cr",
                    f"{row['FVM']}",
                )

    console.print(table)


def render_roster_table(state: AuctionState):
    """Renders the user's current drafted squad in a clean Rich table."""
    if not state.my_squad:
        console.print(
            Panel(
                "[dim]Your roster is currently empty.[/dim]\n"
                "Run [bold cyan]targets[/bold cyan] to view recommended players, or [bold cyan]buy <name> <price>[/bold cyan] to draft your first player.",
                title="📋 Current Roster",
                box=box.ROUNDED,
                border_style="dim",
            )
        )
        return

    tot_spent = state.spent_budget
    tot_fvm = sum(p["FVM"] for p in state.my_squad)
    tot_est = sum(p.get("EstPrice", 0) for p in state.my_squad)
    diff = tot_spent - tot_est
    diff_sign = f"+{diff}" if diff > 0 else str(diff)

    table = Table(
        title=f"📋 Your Current Squad ({len(state.my_squad)}/25 Players)",
        box=box.ROUNDED,
        border_style="dim",
        header_style="bold cyan",
        show_footer=True,
        expand=True,
    )

    table.add_column("Role", justify="center", width=5)
    table.add_column("Player", justify="left", ratio=1)
    table.add_column("Team", justify="center", width=6, style="dim")
    table.add_column(
        "Paid", justify="right", width=9, footer=f"[bold]{tot_spent} cr[/bold]"
    )
    table.add_column(
        "Est. Price", justify="right", width=10, footer=f"[dim]{tot_est} cr[/dim]"
    )
    table.add_column("Diff", justify="right", width=8, footer=f"[dim]{diff_sign}[/dim]")
    table.add_column("FVM", justify="right", width=8, footer=f"[bold]{tot_fvm}[/bold]")

    for p in state.my_squad:
        role = p["R"]
        c = ROLE_COLORS.get(role, "white")
        paid = p["PricePaid"]
        est = p.get("EstPrice", 0)
        p_diff = paid - est
        diff_str = (
            f"[red]+{p_diff}[/red]"
            if p_diff > 0
            else f"[green]{p_diff}[/green]"
            if p_diff < 0
            else "[dim]0[/dim]"
        )

        table.add_row(
            f"[{c}]{role}[/{c}]",
            f"[bold white]{p['Nome']}[/bold white]",
            p.get("Squadra", ""),
            f"{paid} cr",
            f"{est} cr",
            diff_str,
            str(p.get("FVM", 0)),
        )

    console.print(table)


def render_player_card(player: pd.Series, is_taken: bool):
    """Renders a sleek card for player search results."""
    role = player["R"]
    role_color = ROLE_COLORS.get(role, "white")
    role_full = ROLE_NAMES.get(role, role)
    status_badge = (
        "[bold red]TAKEN / DRAFTED[/bold red]"
        if is_taken
        else "[bold green]AVAILABLE[/bold green]"
    )

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", justify="right")
    grid.add_column(style="bold white")

    grid.add_row("Player:", f"{player['Nome']} ({player['Squadra']})")
    grid.add_row("Role:", f"[{role_color}]{role}[/{role_color}] - {role_full}")
    grid.add_row("Est. Price:", f"{player['EstPrice']} crediti")
    grid.add_row("FVM:", str(player["FVM"]))
    grid.add_row("Status:", status_badge)

    panel = Panel(
        grid,
        title=f"🔍 Player Details: [bold white]{player['Nome']}[/bold white]",
        box=box.ROUNDED,
        border_style="dim cyan",
        expand=False,
    )
    console.print(panel)


def render_help():
    """Renders a styled commands table."""
    table = Table(
        title="🛠️ Commands",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Command", style="bold cyan")
    table.add_column("Alias / Slash", style="dim")
    table.add_column("Description")
    table.add_column("Example", style="green")

    table.add_row(
        "targets",
        "t, /targets",
        "Re-solve MILP & display optimal targets squad",
        "targets",
    )
    table.add_row(
        "buy",
        "b, /buy",
        "Mark a player as drafted by you with price paid",
        "buy lautaro 175",
    )
    table.add_row(
        "pass",
        "p, /pass",
        "Mark a player as taken by rival or skipped",
        "pass vlahovic",
    )
    table.add_row(
        "roster",
        "r, /roster",
        "Display your current drafted squad and stats",
        "roster",
    )
    table.add_row(
        "search",
        "s, /search",
        "Look up a player by name or ID",
        "search barella",
    )
    table.add_row(
        "undo",
        "u, /undo",
        "Undo the last buy or pass action",
        "undo",
    )
    table.add_row(
        "status",
        "/status",
        "Display current budget and remaining role quotas",
        "status",
    )
    table.add_row(
        "clear",
        "c, /clear",
        "Clear terminal screen & refresh UI banner",
        "clear",
    )
    table.add_row(
        "quit",
        "q, /exit",
        "Exit the assistant",
        "quit",
    )

    console.print(table)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="F.A.B.I.O. - Live Auction Assistant")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="Path to quotazioni CSV dataset (defaults to live or demo dataset in dataset/)",
    )
    cli_args, _ = parser.parse_known_args()

    dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
    live_dataset = dataset_dir / "quotazioni_fantacalcio.csv"
    demo_dataset = dataset_dir / "demo_quotazioni.csv"

    if cli_args.dataset:
        dataset_path = Path(cli_args.dataset)
        using_demo = False
    elif live_dataset.exists():
        dataset_path = live_dataset
        using_demo = False
    elif demo_dataset.exists():
        dataset_path = demo_dataset
        using_demo = True
    else:
        dataset_path = live_dataset
        using_demo = False

    if not dataset_path.exists():
        console.print(
            f"[bold red]Error:[/] Dataset file not found at [bold]{dataset_path}[/bold].\n"
            "Please run the scraper first: [cyan]python scraper/scraper.py[/cyan] or provide a valid CSV."
        )
        return

    df = pd.read_csv(dataset_path)
    df["EstPrice"] = np.maximum(1, np.round(df["FVM"] / 1000 * 500)).astype(int)

    state = AuctionState(df)

    # Initial Screen Render
    print_banner()
    if using_demo:
        console.print(
            "[dim yellow]ℹ Running in demo mode with [bold]dataset/demo_quotazioni.csv[/bold]. "
            "Run [bold cyan]python scraper/scraper.py[/bold cyan] to scrape live Serie A data.[/dim yellow]\n"
        )
    print_status_bar(state)
    console.print(
        "[dim]Type [bold cyan]targets[/bold cyan] to view optimal squad, "
        "[bold cyan]help[/bold cyan] for commands, or [bold cyan]quit[/bold cyan] to exit.[/dim]\n"
    )

    prompt_str = "\033[1;36mfabio\033[0m ❯ "

    while True:
        try:
            cmd_input = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting... Good luck at the auction! 🏆[/dim]")
            break

        if not cmd_input:
            continue

        parts = cmd_input.split()
        cmd = parts[0].lower().lstrip("/")  # Support slash commands like /targets
        args = parts[1:]

        if cmd in ("q", "quit", "exit"):
            console.print("[dim]Exiting... Good luck at the auction! 🏆[/dim]")
            break

        elif cmd in ("c", "clear"):
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            print_status_bar(state)

        elif cmd in ("status",):
            print_status_bar(state)

        elif cmd in ("b", "buy"):
            if len(args) < 2:
                console.print(
                    "[bold red]Usage:[/] [cyan]buy <player_name> <price_paid>[/cyan] (e.g. [green]'buy lautaro 175'[/green])"
                )
                continue

            name_query, price_str = " ".join(args[:-1]), args[-1]
            try:
                price = int(price_str)
            except ValueError:
                console.print("[bold red]Error:[/] Price must be an integer.")
                continue

            player = find_player(name_query, df)
            if player is None:
                console.print(
                    f"[bold red]Not Found:[/] No player found matching '[bold]{name_query}[/bold]'"
                )
                continue

            if state.quotas_needed[player["R"]] == 0:
                role_label = ROLE_NAMES.get(player["R"], player["R"])
                console.print(
                    f"[bold yellow]⚠️ Quota Full:[/] Your department for role [bold]{player['R']}[/bold] ({role_label}) is already full!"
                )
                continue

            if price > state.remaining_budget:
                console.print(
                    f"[bold yellow]⚠️ Budget Exceeded:[/] Price ({price} cr) exceeds remaining budget ({state.remaining_budget} cr)!"
                )

            state.buy_player(player, price)
            role_c = ROLE_COLORS.get(player["R"], "white")
            console.print(
                f"[bold green]✓ BOUGHT:[/] [bold white]{player['Nome']}[/bold white] "
                f"([{role_c}]{player['R']}[/{role_c}] - {player['Squadra']}) for [bold cyan]{price}[/bold cyan] crediti! "
                f"[dim](Remaining: {state.remaining_budget} cr)[/dim]"
            )
            print_status_bar(state)

        elif cmd in ("p", "pass"):
            if not args:
                console.print(
                    "[bold red]Usage:[/] [cyan]pass <player_name>[/cyan] (e.g. [green]'pass thuram'[/green])"
                )
                continue

            name_query = " ".join(args)
            player = find_player(name_query, df)
            if player is None:
                console.print(
                    f"[bold red]Not Found:[/] No player found matching '[bold]{name_query}[/bold]'"
                )
                continue

            state.pass_player(player)
            role_c = ROLE_COLORS.get(player["R"], "white")
            console.print(
                f"[dim red]✕ PASSED / TAKEN:[/] [bold white]{player['Nome']}[/bold white] "
                f"([{role_c}]{player['R']}[/{role_c}] - {player['Squadra']}) [dim](removed from draft pool)[/dim]"
            )

        elif cmd in ("t", "targets"):
            optimal = solve_optimal_squad(
                df,
                state.remaining_budget,
                state.quotas_needed,
                state.excluded_ids,
            )
            render_targets_table(optimal)

        elif cmd in ("r", "roster"):
            render_roster_table(state)

        elif cmd in ("u", "undo"):
            success, msg = state.undo()
            if success:
                console.print(f"[bold yellow]↺ UNDO:[/] {msg}")
                print_status_bar(state)
            else:
                console.print(f"[dim]{msg}[/dim]")

        elif cmd in ("s", "search"):
            if not args:
                console.print("[bold red]Usage:[/] [cyan]search <query>[/cyan]")
                continue
            player = find_player(" ".join(args), df)
            if player is not None:
                is_taken = str(player["Id"]) in state.excluded_ids
                render_player_card(player, is_taken)
            else:
                console.print(
                    f"[bold red]Not Found:[/] No player found matching '[bold]{' '.join(args)}[/bold]'"
                )

        elif cmd in ("help", "h", "?"):
            render_help()

        else:
            console.print(
                f"[bold red]Unknown command '[bold]{cmd}[/bold]'.[/bold red] "
                f"Type [bold cyan]help[/bold cyan] for available commands."
            )


if __name__ == "__main__":
    main()
