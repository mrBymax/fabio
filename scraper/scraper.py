import csv
import datetime
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URL = "https://www.fantacalcio.it/quotazioni-fantacalcio"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
}

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
METADATA_FILE = DATASET_DIR / "metadata.json"
CSV_FILE = DATASET_DIR / "quotazioni_fantacalcio.csv"


def fetch_and_parse_players() -> list[dict]:
    """Scrapes all Serie A players from the public Fantacalcio quotazioni table."""
    print(f"Fetching data from {URL} ...")
    response = requests.get(URL, headers=HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.select("tr.player-row")
    print(f"Found {len(rows)} player entries on the page.")

    players = []
    for row in rows:
        link = row.select_one("a.player-name")
        href = link["href"] if link and "href" in link.attrs else ""
        player_id = href.rstrip("/").split("/")[-1] if href else ""
        name = link.get_text(strip=True) if link else ""

        role_classic = row.get("data-filter-role-classic", "").upper()
        role_mantra = row.get("data-filter-role-mantra", "").upper()

        team_el = row.select_one("td.player-team")
        team = team_el.get_text(strip=True) if team_el else ""

        # Classic stats
        qi_el = row.select_one("td.player-classic-initial-price")
        qi = int(qi_el.get_text(strip=True)) if qi_el and qi_el.get_text(strip=True).isdigit() else 0

        qa_el = row.select_one("td.player-classic-current-price")
        qa = int(qa_el.get_text(strip=True)) if qa_el and qa_el.get_text(strip=True).isdigit() else 0

        fvm_el = row.select_one("td.player-classic-fvm")
        fvm = int(fvm_el.get_text(strip=True)) if fvm_el and fvm_el.get_text(strip=True).isdigit() else 0

        # Mantra stats
        m_qi_el = row.select_one("td.player-mantra-initial-price")
        m_qi = int(m_qi_el.get_text(strip=True)) if m_qi_el and m_qi_el.get_text(strip=True).isdigit() else 0

        m_qa_el = row.select_one("td.player-mantra-current-price")
        m_qa = int(m_qa_el.get_text(strip=True)) if m_qa_el and m_qa_el.get_text(strip=True).isdigit() else 0

        m_fvm_el = row.select_one("td.player-mantra-fvm")
        m_fvm = int(m_fvm_el.get_text(strip=True)) if m_fvm_el and m_fvm_el.get_text(strip=True).isdigit() else 0

        players.append({
            "Id": player_id,
            "R": role_classic,
            "RM": role_mantra,
            "Nome": name,
            "Squadra": team,
            "Qt.A": qa,
            "Qt.I": qi,
            "Diff.": qa - qi,
            "Qt.A M": m_qa,
            "Qt.I M": m_qi,
            "Diff.M": m_qa - m_qi,
            "FVM": fvm,
            "FVM M": m_fvm,
        })

    return players


def save_dataset(players: list[dict], output_csv: Path) -> None:
    """Saves the parsed players to CSV and updates metadata.json."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Id", "R", "RM", "Nome", "Squadra",
        "Qt.A", "Qt.I", "Diff.", "Qt.A M", "Qt.I M", "Diff.M",
        "FVM", "FVM M",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(players)

    print(f"Saved {len(players)} players to: {output_csv}")

    metadata = {
        "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_players": len(players),
        "filename": output_csv.name,
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Updated metadata in: {METADATA_FILE}")


def main():
    players = fetch_and_parse_players()
    if not players:
        print("Warning: No player data extracted!")
        return

    save_dataset(players, CSV_FILE)
    print("Scraping completed successfully.")


if __name__ == "__main__":
    main()
