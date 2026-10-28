import math
import sqlite3
import urllib.request
from datetime import datetime
from io import StringIO

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FootPredict Pro (IA, Auto-Data & Backtest)", page_icon="🧪", layout="wide")

DB_FILE = "footpredict.db"
REQUIRED = ["date", "competition", "home_team", "away_team", "home_goals", "away_goals"]

AVAILABLE_LEAGUES = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Angleterre - Premier League": ("E0", "Premier League"),
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Angleterre - Championship": ("E1", "Championship"),
    "🇫🇷 France - Ligue 1": ("F1", "Ligue 1"),
    "🇪🇸 Espagne - La Liga": ("SP1", "La Liga"),
    "🇮🇹 Italie - Serie A": ("I1", "Serie A"),
    "🇩🇪 Allemagne - Bundesliga": ("D1", "Bundesliga"),
    "🇵🇹 Portugal - Liga Portugal": ("P1", "Liga Portugal"),
    "🇳🇱 Pays-Bas - Eredivisie": ("N1", "Eredivisie"),
    "🇧🇪 Belgique - Jupiler League": ("B1", "Jupiler League"),
    "🇹🇷 Turquie - Süper Lig": ("T1", "Super Lig"),
}

# ================= GESTION SQLITE =================
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                competition TEXT,
                home_team TEXT,
                away_team TEXT,
                bet_type TEXT,
                bet_name TEXT,
                probability REAL,
                min_odds REAL,
                taken_odds REAL,
                status TEXT DEFAULT 'En attente'
            )
        """)
        conn.commit()

init_db()

def save_prediction(competition, home, away, bet_type, bet_name, prob, min_odds, taken_odds):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (created_at, competition, home_team, away_team, bet_type, bet_name, probability, min_odds, taken_odds, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'En attente')
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            competition, home, away, bet_type, bet_name, round(prob * 100, 1), min_odds, taken_odds
        ))
        conn.commit()

def get_all_predictions():
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)

def update_status(pred_id, new_status):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE predictions SET status = ? WHERE id = ?", (new_status, pred_id))
        conn.commit()

def delete_prediction(pred_id):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
        conn.commit()


# ================= MODULE DE COLLECTE AUTOMATIQUE =================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_league_data(league_code, league_name, season_code="2425"):
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            df = pd.read_csv(response)

        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
        df["competition"] = league_name
        df["home_team"] = df["HomeTeam"].astype(str).str.strip()
        df["away_team"] = df["AwayTeam"].astype(str).str.strip()
        df["home_goals"] = df["FTHG"].astype(int)
        df["away_goals"] = df["FTAG"].astype(int)

        return df[REQUIRED]
    except Exception:
        return pd.DataFrame(columns=REQUIRED)


# ================= MOTEUR STATISTIQUE =================
def poisson_pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def dixon_coles_factor(hg, ag, lambda_home, lambda_away, rho=-0.11):
    if hg == 0 and ag == 0: return 1 - (lambda_home * lambda_away * rho)
    elif hg == 0 and ag == 1: return 1 + (lambda_home * rho)
    elif hg == 1 and ag == 0: return 1 + (lambda_away * rho)
    elif hg == 1 and ag == 1: return 1 - rho
    return 1.0

def normalise_matches(df):
    df = df.copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(missing))
    df["date"] = df["date"].astype(str)
    for col in ["competition", "home_team", "away_team"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])
    df = df[(df["home_goals"] >= 0) & (df["away_goals"] >= 0)]
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df[(df["home_team"] != "") & (df["away_team"] != "")]
    return df

def team_rates(df, team, venue="all", n=10):
    rows = []
    data = df.copy()
    data["_sort_date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values("_sort_date", ascending=False, na_position="last")
    for _, r in data.iterrows():
        h, a = r["home_team"], r["away_team"]
        if venue == "home" and h.casefold() == team.casefold():
            rows.append((int(r["home_goals"]), int(r["away_goals"])))
        elif venue == "away" and a.casefold() == team.casefold():
            rows.append((int(r["away_goals"]), int(r["home_goals"])))
        elif venue == "all":
            if h.casefold() == team.casefold():
                rows.append((int(r["home_goals"]), int(r["away_goals"])))
            elif a.casefold() == team.casefold():
                rows.append((int(r["away_goals"]), int(r["home_goals"])))
        if len(rows) >= n:
            break
    if not rows: return None
    return {
        "played": len(rows),
        "scored": sum(x for x, _ in rows) / len(rows),
        "conceded": sum(y for _, y in rows) / len(rows),
    }

def calculate_prediction(df, home, away, n=10, competition_filter=None):
    data = df.copy()
    if competition_filter and competition_filter != "Toutes les compétitions":
        data = data[data["competition"].str.casefold() == competition_filter.casefold()]

    h_all = team_rates(data, home, "all", n)
    a_all = team_rates(data, away, "all", n)
    h_home = team_rates(data, home, "home", n)
    a_away = team_rates(data, away, "away", n)

    if h_all is None or a_all is None:
        missing = [t for t, v in [(home, h_all), (away, a_all)] if v is None]
        raise ValueError("Historique insuffisant pour : " + ", ".join(missing))

    avg_home = max(data["home_goals"].mean(), 0.30) if len(data) else 1.35
    avg_away = max(data["away_goals"].mean(), 0.30) if len(data) else 1.10

    def blend(venue_rate, overall_rate):
        if venue_rate is None: return overall_rate
        weight = min(venue_rate["played"] / 5.0, 1.0)
        return weight * venue_rate["scored"] + (1 - weight) * overall_rate["scored"], \
               weight * venue_rate["conceded"] + (1 - weight) * overall_rate["conceded"]

    h_scored, h_conceded = blend(h_home, h_all) if h_home else (h_all["scored"], h_all["conceded"])
    a_scored, a_conceded = blend(a_away, a_all) if a_away else (a_all["scored"], a_all["conceded"])

    lambda_home = avg_home * (h_scored / avg_home) * (a_conceded / avg_home)
    lambda_away = avg_away * (a_scored / avg_away) * (h_conceded / avg_away)
    lambda_home = min(max(lambda_home, 0.10), 4.50)
    lambda_away = min(max(lambda_away, 0.10), 4.00)
    total_lambda = lambda_home + lambda_away

    matrix = {}
    home_win = draw = away_win = 0.0
    for hg in range(10):
        for ag in range(10):
            base_p = poisson_pmf(hg, lambda_home) * poisson_pmf(ag, lambda_away)
            adj_p = base_p * dixon_coles_factor(hg, ag, lambda_home, lambda_away)
            matrix[(hg, ag)] = adj_p
            if hg > ag: home_win += adj_p
            elif hg == ag: draw += adj_p
            else: away_win += adj_p

    total_p = sum(matrix.values())
    home_win /= total_p
    draw /= total_p
    away_win /= total_p
    for k in matrix: matrix[k] /= total_p

    prob_goals = [sum(p for (h, a), p in matrix.items() if h + a == g) for g in range(20)]
    
    unders = {
        "Moins de 1,5 buts": sum(prob_goals[:2]),
        "Moins de 2,5 buts": sum(prob_goals[:3]),
        "Moins de 3,5 buts": sum(prob_goals[:4]),
        "Moins de 4,5 buts": sum(prob_goals[:5]),
        "Moins de 5,5 buts": sum(prob_goals[:6]),
    }
    overs = {
        "Plus de 0,5 but": 1 - prob_goals[0],
        "Plus de 1,5 buts": 1 - sum(prob_goals[:2]),
        "Plus de 2,5 buts": 1 - sum(prob_goals[:3]),
    }
    double_chance = {
        f"1X ({home} ou Nul)": home_win + draw,
        f"X2 (Nul ou {away})": draw + away_win,
        f"12 ({home} ou {away})": home_win + away_win,
    }

    all_markets = []
    all_markets.append({"Type": "1N2", "Pari": f"Victoire {home}", "Probabilité": home_win})
    all_markets.append({"Type": "1N2", "Pari": f"Victoire {away}", "Probabilité": away_win})
    all_markets.append({"Type": "1N2", "Pari": "Match Nul", "Probabilité": draw})

    for lbl, p in double_chance.items(): all_markets.append({"Type": "Double Chance", "Pari": lbl, "Probabilité": p})
    for lbl, p in unders.items(): all_markets.append({"Type": "Under/Moins", "Pari": lbl, "Probabilité": p})
    for lbl, p in overs.items(): all_markets.append({"Type": "Over/Plus", "Pari": lbl, "Probabilité": p})

    for m in all_markets:
        p = m["Probabilité"]
        m["Cote Min. Conseillée"] = round(1.0 / p, 2) if p > 0 else 99.0

    safe_picks = [m for m in all_markets if m["Probabilité"] >= 0.78]
    safe_picks.sort(key=lambda x: x["Probabilité"], reverse=True)

    return {
        "lambda_home": lambda_home, "lambda_away": lambda_away, "total": total_lambda,
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "double_chance": double_chance, "unders": unders, "overs": overs,
        "safe_picks": safe_picks,
        "all_markets": all_markets,
        "h_all": h_all, "a_all": a_all, "h_home": h_home, "a_away": a_away
    }


# ================= FONCTION DE VALIDATION DU RÉSULTAT RÉEL =================
def is_bet_won(bet_type, bet_name, hg, ag, home_team, away_team):
    total = hg + ag
    if bet_type == "Under/Moins":
        if "1,5" in bet_name: return total < 2
        if "2,5" in bet_name: return total < 3
        if "3,5" in bet_name: return total < 4
        if "4,5" in bet_name: return total < 5
        if "5,5" in bet_name: return total < 6
    elif bet_type == "Over/Plus":
        if "0,5" in bet_name: return total >= 1
        if "1,5" in bet_name: return total >= 2
        if "2,5" in bet_name: return total >= 3
    elif bet_type == "Double Chance":
        if "1X" in bet_name: return hg >= ag
        if "X2" in bet_name: return ag >= hg
        if "12" in bet_name: return hg != ag
    elif bet_type == "1N2":
        if f"Victoire {home_team}" in bet_name: return hg > ag
        if f"Victoire {away_team}" in bet_name: return ag > hg
        if "Match Nul" in bet_name: return hg == ag
    return False


# ================= INTERFACE UTILISATEUR =================
st.title("🎯 FootPredict Pro — Analytics, Base SQLite & Backtest")

tab_analyser, tab_backtest, tab_historique = st.tabs([
    "⚽ Analyse & Pronostics",
    "🧪 Backtesting & Validation Historique",
    "📊 Bilan & Historique SQLite"
])

# GESTION DES SOURCES DE DONNÉES EN BARRE LATÉRALE
with st.sidebar:
    st.header("📥 Source des données")
    mode = st.radio("Mode d'alimentation", ["🌐 Téléchargement Automatique", "📂 Import Manuel (CSV)"])

    matches = pd.DataFrame(columns=REQUIRED)

    if mode == "🌐 Téléchargement Automatique":
        st.caption("Données Football-Data.co.uk")
        selected_leagues = st.multiselect(
            "Championnats à charger :",
            list(AVAILABLE_LEAGUES.keys()),
            default=["🏴󠁧󠁢󠁥󠁮󠁧󠁿 Angleterre - Premier League", "🇫🇷 France - Ligue 1", "🇪🇸 Espagne - La Liga"]
        )
        season = st.selectbox("Saison", ["2526 (2025-2026)", "2425 (2024-2025)", "2324 (2023-2024)"], index=0)
        season_code = season[:4]

        if st.button("🔄 Télécharger / Actualiser", type="primary", use_container_width=True):
            all_dfs = []
            with st.spinner("Téléchargement des championnats..."):
                for l_name in selected_leagues:
                    code, clean_name = AVAILABLE_LEAGUES[l_name]
                    df_l = fetch_league_data(code, clean_name, season_code)
                    if not df_l.empty:
                        all_dfs.append(df_l)
                
                if all_dfs:
                    st.session_state["auto_matches"] = pd.concat(all_dfs, ignore_index=True)
                    st.success(f"{len(st.session_state['auto_matches'])} matchs chargés !")
                else:
                    st.error("Téléchargement impossible.")

        if "auto_matches" in st.session_state:
            matches = st.session_state["auto_matches"]

    else:
        uploaded = st.file_uploader("Importer votre propre fichier CSV", type=["csv"])
        if uploaded:
            try:
                matches = normalise_matches(pd.read_csv(uploaded))
                st.success(f"{len(matches)} matchs chargés.")
            except Exception as e:
                st.error(f"Erreur : {e}")


# ----------------- ONGLET 1 : ANALYSE & PRONOSTICS -----------------
with tab_analyser:
    if matches.empty:
        st.info("👈 Chargez des championnats dans le menu de gauche pour commencer.")
    else:
        teams = sorted(set(matches["home_team"].tolist() + matches["away_team"].tolist()))
        comps = ["Toutes les compétitions"] + sorted([x for x in matches["competition"].dropna().unique().tolist() if x])
        
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1: home = st.selectbox("Équipe Domicile", teams, index=0)
        with c2: away = st.selectbox("Équipe Extérieur", teams, index=1 if len(teams) > 1 else 0)
        with c3: competition = st.selectbox("Compétition", comps, index=0)
        n_matches = st.slider("Matchs récents analysés", min_value=5, max_value=20, value=10)

        if home.casefold() == away.casefold():
            st.error("Sélectionnez deux équipes distinctes.")
        else:
            if st.button("Calculer le match", type="primary", use_container_width=True):
                try:
                    st.session_state["res"] = calculate_prediction(matches, home, away, n_matches, competition)
                    st.session_state["teams"] = (home, away, competition)
                except Exception as e:
                    st.error(str(e))

        if "res" in st.session_state:
            res = st.session_state["res"]
            hm, aw, comp_selected = st.session_state["teams"]
            
            st.divider()
            st.subheader(f"⭐ Paris Détectés à Haute Probabilité (≥ 80 %) : {hm} vs {aw}")
            
            if res["safe_picks"]:
                st.success(f"{len(res['safe_picks'])} marchés identifiés avec une probabilité ≥ 80% !")
                
                display_picks = []
                for pick in res["safe_picks"]:
                    display_picks.append({
                        "Catégorie": pick["Type"],
                        "Pronostic": pick["Pari"],
                        "Probabilité": f"{pick['Probabilité']*100:.1f} %",
                        "Cote Min.": f"{pick['Cote Min. Conseillée']:.2f}"
                    })
                st.dataframe(pd.DataFrame(display_picks), hide_index=True, use_container_width=True)

                st.markdown("#### 💾 Enregistrer un pronostic dans la base SQLite")
                with st.form("save_bet_form"):
                    f1, f2, f3 = st.columns([2, 1, 1])
                    with f1:
                        chosen_pick_str = st.selectbox("Choisir le pari à sauvegarder", [f"{p['Type']} - {p['Pari']} ({p['Probabilité']*100:.1f}%)" for p in res["safe_picks"]])
                    with f2:
                        idx = [f"{p['Type']} - {p['Pari']} ({p['Probabilité']*100:.1f}%)" for p in res["safe_picks"]].index(chosen_pick_str)
                        target_pick = res["safe_picks"][idx]
                        taken_odds = st.number_input("Cote du Bookmaker", min_value=1.01, max_value=20.0, value=float(target_pick["Cote Min. Conseillée"]), step=0.05)
                    with f3:
                        st.write(""); st.write("")
                        btn_save = st.form_submit_button("Enregistrer le pari 💾", use_container_width=True)

                    if btn_save:
                        save_prediction(
                            competition=comp_selected, home=hm, away=aw,
                            bet_type=target_pick["Type"], bet_name=target_pick["Pari"],
                            prob=target_pick["Probabilité"], min_odds=target_pick["Cote Min. Conseillée"],
                            taken_odds=taken_odds
                        )
                        st.success(f"Pari '{target_pick['Pari']}' enregistré dans SQLite !")
            else:
                st.warning("Aucun pari à 80%+ identifié pour cette rencontre.")

            st.divider()
            colA, colB, colC = st.columns(3)
            colA.metric(f"Buts Attendus ({hm})", f"{res['lambda_home']:.2f}")
            colB.metric(f"Buts Attendus ({aw})", f"{res['lambda_away']:.2f}")
            colC.metric("Total Buts Espéré", f"{res['total']:.2f}")


# ----------------- ONGLET 2 : BACKTESTING & VALIDATION HISTORIQUE -----------------
with tab_backtest:
    st.subheader("🧪 Simulation & Validation Historique (Walk-Forward)")
    st.caption("Cette simulation applique le modèle chronologiquement sur chaque match passé sans connaître le score à l'avance.")

    if matches.empty:
        st.info("Chargez d'abord des matchs depuis la barre latérale pour lancer le backtest.")
    else:
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            backtest_comp = st.selectbox("Compétition à tester", ["Toutes"] + sorted(matches["competition"].unique().tolist()), key="bt_comp")
        with bc2:
            min_prob_threshold = st.slider("Seuil minimum de probabilité", min_value=70, max_value=90, value=80, step=1, format="%d%%") / 100.0
        with bc3:
            max_sim_matches = st.selectbox("Nombre de matchs à tester (les plus récents)", [100, 200, 300, 500, "Tous"], index=1)

        if st.button("🚀 Lancer le Backtesting", type="primary", use_container_width=True):
            # Préparation des données chronologiques
            df_bt = matches.copy()
            if backtest_comp != "Toutes":
                df_bt = df_bt[df_bt["competition"] == backtest_comp]
            
            df_bt["_dt"] = pd.to_datetime(df_bt["date"], errors="coerce")
            df_bt = df_bt.sort_values("_dt", ascending=True).reset_index(drop=True)

            if len(df_bt) < 60:
                st.error("Échantillon insuffisant (au moins 60 matchs nécessaires pour un test représentatif).")
            else:
                # Nombre de matchs à itérer
                total_len = len(df_bt)
                limit = total_len - 50 if max_sim_matches == "Tous" else min(int(max_sim_matches), total_len - 50)
                start_idx = total_len - limit

                sim_results = []
                brier_errors = []
                bankroll = 1000.0  # Capital initial 1000 €
                stake = 10.0      # Mise fixe 10 €
                bankroll_history = [bankroll]

                prog_bar = st.progress(0)
                status_text = st.empty()

                for i_step, i in enumerate(range(start_idx, total_len)):
                    past_slice = df_bt.iloc[:i]
                    target_row = df_bt.iloc[i]
                    h_tm, a_tm = target_row["home_team"], target_row["away_team"]
                    real_hg, real_ag = target_row["home_goals"], target_row["away_goals"]

                    try:
                        pred = calculate_prediction(past_slice, h_tm, a_tm, n=10, competition_filter=target_row["competition"])
                        
                        # Rechercher tous les marchés qui dépassent le seuil
                        candidates = [m for m in pred["all_markets"] if m["Probabilité"] >= min_prob_threshold]
                        if candidates:
                            # On prend le pari le plus solide
                            best_pick = max(candidates, key=lambda x: x["Probabilité"])
                            p = best_pick["Probabilité"]
                            cote = best_pick["Cote Min. Conseillée"]

                            won = is_bet_won(best_pick["Type"], best_pick["Pari"], real_hg, real_ag, h_tm, a_tm)
                            y_true = 1.0 if won else 0.0

                            # Calcul Brier Score: (p - y)^2
                            brier_errors.append((p - y_true) ** 2)

                            # P&L
                            if won:
                                profit = (cote * stake) - stake
                                bankroll += profit
                            else:
                                profit = -stake
                                bankroll += profit

                            bankroll_history.append(round(bankroll, 2))

                            sim_results.append({
                                "Date": target_row["date"],
                                "Match": f"{h_tm} {real_hg}-{real_ag} {a_tm}",
                                "Pronostic": best_pick["Pari"],
                                "Prob. Estimée": f"{p*100:.1f}%",
                                "Cote": cote,
                                "Résultat": "✅ Gagné" if won else "❌ Perdu",
                                "Gain/Perte (€)": round(profit, 2),
                                "Capital (€)": round(bankroll, 2)
                            })
                    except Exception:
                        pass

                    prog_bar.progress((i_step + 1) / limit)
                    status_text.text(f"Évaluation du match {i_step + 1} / {limit}...")

                prog_bar.empty()
                status_text.empty()

                if not sim_results:
                    st.warning("Aucun pari n'a atteint le seuil exigé sur cette sélection de matchs.")
                else:
                    res_df = pd.DataFrame(sim_results)
                    total_sim = len(res_df)
                    total_won = len(res_df[res_df["Résultat"] == "✅ Gagné"])
                    total_lost = total_sim - total_won
                    win_rate_bt = (total_won / total_sim) * 100
                    brier_score = sum(brier_errors) / len(brier_errors) if brier_errors else 0.0
                    net_profit = bankroll - 1000.0
                    roi = (net_profit / (total_sim * stake)) * 100

                    st.success("Simulation terminée avec succès !")

                    # MÉTRIQUES CLÉS
                    bm1, bm2, bm3, bm4, bm5 = st.columns(5)
                    bm1.metric("Paris Déclenchés", total_sim)
                    bm2.metric("Gagnés / Perdus", f"{total_won} / {total_lost}")
                    bm3.metric("Taux de Réussite Réel", f"{win_rate_bt:.1f} %")
                    bm4.metric("Brier Score (Précision)", f"{brier_score:.3f}")
                    bm5.metric("Bénéfice Net (Mise 10€)", f"{net_profit:+.2f} €", delta=f"{roi:+.1f}% ROI")

                    st.info("""
                    📘 **Interprétation scientifique des résultats :**
                    * **Taux de réussite :** Indique le pourcentage réel de paris gagnés.
                    * **Brier Score :** Plus il est proche de 0, plus les probabilités calculées sont scientifiquement exactes (un score < 0.15 est excellent).
                    """)

                    # GRAPHIQUE DE L'ÉVOLUTION DE LA BANKROLL
                    st.markdown("#### 📈 Courbe d'évolution du capital (Simulation)")
                    chart_df = pd.DataFrame({"Capital (€)": bankroll_history})
                    st.line_chart(chart_df, use_container_width=True)

                    # DÉTAIL DES MATCHS
                    with st.expander("Consulter le détail de chaque match testé"):
                        st.dataframe(res_df, hide_index=True, use_container_width=True)


# ----------------- ONGLET 3 : BILAN SQLITE -----------------
with tab_historique:
    st.subheader("📚 Historique permanent des pronostics")
    db_data = get_all_predictions()

    if db_data.empty:
        st.info("Aucun pronostic n'a encore été enregistré.")
    else:
        total_bets = len(db_data)
        won_bets = len(db_data[db_data["status"] == "Gagné"])
        lost_bets = len(db_data[db_data["status"] == "Perdu"])
        pending_bets = len(db_data[db_data["status"] == "En attente"])
        finished_bets = won_bets + lost_bets
        win_rate = (won_bets / finished_bets * 100) if finished_bets > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Pronostics", total_bets)
        m2.metric("En Attente", pending_bets)
        m3.metric("Gagnés / Perdus", f"{won_bets} / {lost_bets}")
        m4.metric("Taux de Réussite Réel", f"{win_rate:.1f} %")

        st.divider()
        st.markdown("#### Liste de vos pronostics enregistrés")
        st.dataframe(
            db_data[["id", "created_at", "home_team", "away_team", "bet_name", "probability", "min_odds", "taken_odds", "status"]],
            hide_index=True,
            use_container_width=True
        )

        st.markdown("#### ⚙️ Valider ou modifier le résultat d'un match")
        col_id, col_action, col_valider, col_del = st.columns([1, 1, 1, 1])
        with col_id: sel_id = st.selectbox("ID du pronostic", db_data["id"].tolist())
        with col_action: new_st = st.selectbox("Nouveau statut", ["Gagné", "Perdu", "Annulé", "En attente"])
        with col_valider:
            st.write(""); st.write("")
            if st.button("Mettre à jour 🔄", use_container_width=True):
                update_status(sel_id, new_st)
                st.success(f"Pari #{sel_id} mis à jour : {new_st}")
                st.rerun()
        with col_del:
            st.write(""); st.write("")
            if st.button("Supprimer 🗑️", use_container_width=True):
                delete_prediction(sel_id)
                st.warning(f"Pari #{sel_id} supprimé.")
                st.rerun()