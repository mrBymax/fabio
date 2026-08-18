import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, Bounds, milp


def solve_optimal_squad(
        df: pd.DataFrame,
        remaining_budget: int,
        quota_needed: dict[str, int],  # e.g. {'P': 2, 'D': 6, 'C': 5, 'A': 4}
        excluded_ids: set[str],
) -> pd.DataFrame:
    """Finds the optimal roster to complete remaining quotas within remaining budget."""

    # 1. Filter out already drafted / excluded players
    available_mask = ~df['Id'].astype(str).isin(excluded_ids)
    candidates = df[available_mask].reset_index(drop=True)
    N = len(candidates)

    if N == 0:
        return pd.DataFrame()

    # 2. Objective: Maximize FVM (minimize -FVM)
    c = -candidates['FVM'].values.astype(float)

    # 3. Decision variables: Binary (0 or 1)
    integrality = np.ones(N)  # 1 = integer
    bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))

    # 4. Constraints
    # Constraint A: Budget limit (sum(EstPrice * x) <= remaining_budget)
    prices = candidates['EstPrice'].values.astype(float)

    # Constraint B: Role quotas (sum(isRole * x) == quota_needed[role])
    is_gk = (candidates['R'] == 'P').values.astype(float)
    is_def = (candidates['R'] == 'D').values.astype(float)
    is_mid = (candidates['R'] == 'C').values.astype(float)
    is_fwd = (candidates['R'] == 'A').values.astype(float)

    A = np.vstack([prices, is_gk, is_def, is_mid, is_fwd])

    # Lower and upper bounds for constraints:
    # Budget: 0 <= spend <= remaining_budget
    # Roles: exact equality
    lhs = np.array([0, quota_needed['P'], quota_needed['D'], quota_needed['C'], quota_needed['A']])
    rhs = np.array([remaining_budget, quota_needed['P'], quota_needed['D'], quota_needed['C'], quota_needed['A']])

    constraints = LinearConstraint(A, lhs, rhs)

    # 5. Solve
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

    if res.success:
        selected_idx = np.where(res.x > 0.5)[0]
        return candidates.iloc[selected_idx].copy()
    else:
        return pd.DataFrame()


import difflib


def find_player(query: str, df: pd.DataFrame) -> pd.Series | None:
    """Finds a player by ID, exact name match, substring, or fuzzy match."""
    query = str(query).strip().lower()

    # 1. Match by exact ID
    id_match = df[df['Id'].astype(str) == query]
    if not id_match.empty:
        return id_match.iloc[0]

    # 2. Case-insensitive Substring match in Nome
    sub_matches = df[df['Nome'].str.lower().str.contains(query, regex=False)]
    if len(sub_matches) == 1:
        return sub_matches.iloc[0]
    elif len(sub_matches) > 1:
        # If multiple, prefer exact start or highest FVM
        return sub_matches.sort_values(by='FVM', ascending=False).iloc[0]

    # 3. Fuzzy matching with difflib
    all_names = df['Nome'].tolist()
    close = difflib.get_close_matches(query, [n.lower() for n in all_names], n=1, cutoff=0.5)
    if close:
        matched_idx = [n.lower() for n in all_names].index(close[0])
        return df.iloc[matched_idx]

    return None


from dataclasses import dataclass


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
        spent = sum(p['PricePaid'] for p in self.my_squad)
        return self.total_budget - spent

    @property
    def quotas_needed(self) -> dict[str, int]:
        total_quotas = {'P': 3, 'D': 8, 'C': 8, 'A': 6}
        my_counts = {'P': 0, 'D': 0, 'C': 0, 'A': 0}
        for p in self.my_squad:
            my_counts[p['R']] += 1
        return {r: max(0, total_quotas[r] - my_counts[r]) for r in total_quotas}

    def buy_player(self, player_row: pd.Series, price_paid: int):
        player_dict = player_row.to_dict()
        player_dict['PricePaid'] = price_paid
        self.my_squad.append(player_dict)
        self.excluded_ids.add(str(player_row['Id']))
        self.history.append(Transaction('buy', player_dict, price_paid))

    def pass_player(self, player_row: pd.Series):
        player_dict = player_row.to_dict()
        self.excluded_ids.add(str(player_row['Id']))
        self.history.append(Transaction('pass', player_dict))

    def undo(self) -> str:
        if not self.history:
            return "Nothing to undo!"
        last = self.history.pop()
        player_id = str(last.player['Id'])
        self.excluded_ids.discard(player_id)
        if last.action == 'buy':
            self.my_squad = [p for p in self.my_squad if str(p['Id']) != player_id]
            return f"Undid purchase of {last.player['Nome']} ({last.price_paid} crediti)"
        else:
            return f"Undid pass of {last.player['Nome']} (returned to draft pool)"


from pathlib import Path

def print_header(state: AuctionState):
    quotas = state.quotas_needed
    total_needed = sum(quotas.values())
    spent = state.total_budget - state.remaining_budget
    print("\n" + "=" * 65)
    print(f"💰 BUDGET: {state.remaining_budget} / {state.total_budget} crediti left (Spent: {spent})")
    print(
        f"📋 SQUAD:  {len(state.my_squad)} / 25 players (Needs: {quotas['P']} P | {quotas['D']} D | {quotas['C']} C | {quotas['A']} A)")
    print("=" * 65)


def main():
    dataset_path = Path(__file__).resolve().parent.parent / "dataset" / "quotazioni_fantacalcio.csv"
    df = pd.read_csv(dataset_path)
    df['EstPrice'] = np.maximum(1, np.round(df['FVM'] / 1000 * 500)).astype(int)

    state = AuctionState(df)

    print("\n⚽ Welcome to fabio Live Auction Assistant! Type 'help' for commands.")

    while True:
        print_header(state)
        cmd_input = input("\n[fabio] > ").strip()
        if not cmd_input:
            continue

        parts = cmd_input.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ('q', 'quit', 'exit'):
            print("Exiting. Good luck at the auction! 🏆")
            break

        elif cmd in ('b', 'buy'):
            if len(args) < 2:
                print("Usage: buy <player_name> <price_paid> (e.g. 'buy lautaro 175')")
                continue
            name_query, price_str = " ".join(args[:-1]), args[-1]
            try:
                price = int(price_str)
            except ValueError:
                print("Error: Price must be an integer.")
                continue

            player = find_player(name_query, df)
            if player is None:
                print(f"Player not found for '{name_query}'")
                continue

            if state.quotas_needed[player['R']] == 0:
                print(f"⚠️ Warning: Your department for role '{player['R']}' is already full!")
                continue

            state.buy_player(player, price)
            print(f"✅ BOUGHT: {player['Nome']} ({player['R']} - {player['Squadra']}) for {price} crediti!")

        elif cmd in ('p', 'pass'):
            if not args:
                print("Usage: pass <player_name> (e.g. 'pass thuram')")
                continue
            name_query = " ".join(args)
            player = find_player(name_query, df)
            if player is None:
                print(f"Player not found for '{name_query}'")
                continue

            state.pass_player(player)
            print(f"❌ PASSED / TAKEN BY RIVAL: {player['Nome']} ({player['R']} - {player['Squadra']})")

        elif cmd in ('t', 'targets'):
            print("\nCalculating updated optimal roster...")
            optimal = solve_optimal_squad(df, state.remaining_budget, state.quotas_needed, state.excluded_ids)
            if optimal.empty:
                print("No feasible combination found within remaining budget! Try spending less.")
            else:
                print("\n🎯 UPDATED OPTIMAL TARGET SQUAD:")
                for r, role_name in [('P', 'GK'), ('D', 'DEF'), ('C', 'MID'), ('A', 'FWD')]:
                    sub = optimal[optimal['R'] == r].sort_values(by='FVM', ascending=False)
                    if not sub.empty:
                        print(f"\n--- {role_name} ---")
                        for _, row in sub.iterrows():
                            print(
                                f"  [{row['R']}] {row['Nome']:<18} ({row['Squadra']:<3}) | EstPrice: {row['EstPrice']:<3} | FVM: {row['FVM']}")

        elif cmd in ('r', 'roster'):
            if not state.my_squad:
                print("Roster is empty.")
            else:
                print("\n📋 YOUR CURRENT SQUAD:")
                for p in state.my_squad:
                    print(
                        f"  [{p['R']}] {p['Nome']:<18} ({p['Squadra']:<3}) | Paid: {p['PricePaid']:<3} crediti | FVM: {p['FVM']}")

        elif cmd in ('u', 'undo'):
            msg = state.undo()
            print(f"🔄 {msg}")

        elif cmd in ('s', 'search'):
            if not args:
                print("Usage: search <query>")
                continue
            player = find_player(" ".join(args), df)
            if player is not None:
                status = "TAKEN" if str(player['Id']) in state.excluded_ids else "AVAILABLE"
                print(
                    f"🔍 {player['Nome']} ({player['R']} - {player['Squadra']}) | EstPrice: {player['EstPrice']} | FVM: {player['FVM']} | [{status}]")
            else:
                print("No player found.")

        elif cmd == 'help':
            print("""
    Available Commands:
      b / buy <name> <price>  -> Mark a player as bought by you
      p / pass <name>         -> Mark a player as taken by rival
      t / targets             -> Re-solve & show updated optimal targets
      r / roster              -> Show your current roster & spend
      s / search <name>       -> Lookup player stats
      u / undo                -> Undo the last action
      q / quit                -> Exit the assistant
                """)
        else:
            print(f"Unknown command '{cmd}'. Type 'help' for available commands.")


if __name__ == "__main__":
    main()
