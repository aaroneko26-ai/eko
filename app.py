import json
import math
import sqlite3
import urllib.request
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FootPredict Pro (1N2, Vainqueurs & Buts)", page_icon="🏆", layout="wide")

DB_FILE = "footpredict.db"

# ================= GESTION DE LA BASE SQLITE =================
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                match_datetime TEXT,
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

def save_prediction(match_dt, competition, home, away, bet_type, bet_name, prob, min_odds, taken_odds):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (created_at, match_datetime, competition, home_team, away_team, bet_type, bet_name, probability, min_odds, taken_odds, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'En attente')
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            match_dt, competition, home, away, bet_type, bet_name, round(prob * 100, 1), min_odds, taken_odds
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


# ================= MODULE API-FOOTBALL (MATCHS, COMPOSITIONS & BLESSURES) =================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_live_fixtures(api_key, target_date):
    url = f"https://v3.football.api-sports.io/fixtures?date={target_date}"
    req = urllib.request.Request(url, headers={"x-apisports-key": api_key, "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", [])
    except Exception as e:
        st.error(f"Erreur API lors de la récupération des matchs : {e}")
        return []

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_team_history(api_key, team_id, n=10):
    url = f"https://v3.football.api-sports.io/fixtures?team={team_id}&last={n}&status=FT"
    req = urllib.request.Request(url, headers={"x-apisports-key": api_key, "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            fixtures = data.get("response", [])
            scored, conceded = [], []
            for f in fixtures:
                is_home = (f["teams"]["home"]["id"] == team_id)
                hg = f["goals"]["home"] or 0
                ag = f["goals"]["away"] or 0
                scored.append(hg if is_home else ag)
                conceded.append(ag if is_home else hg)
            
            if not scored: return {"scored": 1.2, "conceded": 1.2, "played": 0}
            return {
                "scored": sum(scored) / len(scored),
                "conceded": sum(conceded) / len(conceded),
                "played": len(scored)
            }
    except Exception:
        return {"scored": 1.2, "conceded": 1.2, "played": 0}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_fixture_lineups(api_key, fixture_id):
    url = f"https://v3.football.api-sports.io/fixtures/lineups?fixture={fixture_id}"
    req = urllib.request.Request(url, headers={"x-apisports-key": api_key, "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", [])
    except Exception:
        return []

@st.cache_data(ttl=600, show_spinner=False)
def fetch_fixture_injuries(api_key, fixture_id):
    url = f"https://v3.football.api-sports.io/injuries?fixture={fixture_id}"
    req = urllib.request.Request(url, headers={"x-apisports-key": api_key, "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", [])
    except Exception:
        return []

def evaluate_lineup_factors(lineups, injuries, home_id, away_id):
    factor_home_att = 1.0
    factor_away_att = 1.0
    notes = []

    home_injuries = [inj for inj in injuries if inj["team"]["id"] == home_id]
    away_injuries = [inj for inj in injuries if inj["team"]["id"] == away_id]

    if home_injuries:
        notes.append(f"⚠️ **Domicile :** {len(home_injuries)} absence(s) : " + 
                     ", ".join([f"{i['player']['name']} ({i['player']['reason']})" for i in home_injuries[:3]]))
        factor_home_att *= max(1.0 - (0.04 * len(home_injuries)), 0.82)
        factor_away_att *= min(1.0 + (0.04 * len(home_injuries)), 1.18)

    if away_injuries:
        notes.append(f"⚠️ **Extérieur :** {len(away_injuries)} absence(s) : " + 
                     ", ".join([f"{i['player']['name']} ({i['player']['reason']})" for i in away_injuries[:3]]))
        factor_away_att *= max(1.0 - (0.04 * len(away_injuries)), 0.82)
        factor_home_att *= min(1.0 + (0.04 * len(away_injuries)), 1.18)

    is_lineup_official = (len(lineups) >= 2)
    home_formation, away_formation = "Non communiquée", "Non communiquée"
    home_xi, away_xi = [], []

    if is_lineup_official:
        for team_data in lineups:
            t_id = team_data["team"]["id"]
            form = team_data.get("formation", "N/A")
            xi = [p["player"]["name"] for p in team_data.get("startXI", [])]
            if t_id == home_id:
                home_formation = form
                home_xi = xi
            else:
                away_formation = form
                away_xi = xi
        notes.append(f"✅ **Schémas officiels confirmés :** Domicile en `{home_formation}` | Extérieur en `{away_formation}`")

    return {
        "factor_home_att": factor_home_att,
        "factor_away_att": factor_away_att,
        "is_official": is_lineup_official,
        "home_formation": home_formation,
        "away_formation": away_formation,
        "home_xi": home_xi,
        "away_xi": away_xi,
        "notes": notes
    }


# ================= MOTEUR STATISTIQUE (1N2, DIXON-COLES & BUTS) =================
def poisson_pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def dixon_coles_factor(hg, ag, lambda_home, lambda_away, rho=-0.11):
    if hg == 0 and ag == 0: return 1 - (lambda_home * lambda_away * rho)
    elif hg == 0 and ag == 1: return 1 + (lambda_home * rho)
    elif hg == 1 and ag == 0: return 1 + (lambda_away * rho)
    elif hg == 1 and ag == 1: return 1 - rho
    return 1.0

def calculate_prediction_with_lineups(h_rates, a_rates, home_name, away_name, lineup_analysis):
    avg_goals = 1.35
    lambda_home = avg_goals * (h_rates["scored"] / avg_goals) * (a_rates["conceded"] / avg_goals)
    lambda_away = avg_goals * (a_rates["scored"] / avg_goals) * (h_rates["conceded"] / avg_goals)
    
    # Avantage du terrain (+15%)
    lambda_home *= 1.15

    # Impact blessures / compos
    lambda_home *= lineup_analysis["factor_home_att"]
    lambda_away *= lineup_analysis["factor_away_att"]

    lambda_home = min(max(lambda_home, 0.20), 4.50)
    lambda_away = min(max(lambda_away, 0.15), 4.00)
    total_lambda = lambda_home + lambda_away

    # Matrice des scores 0 à 9
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

    # Draw No Bet (Remboursé si nul)
    denom = home_win + away_win
    dnb_home = (home_win / denom) if denom > 0 else 0.50
    dnb_away = (away_win / denom) if denom > 0 else 0.50

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
        f"1X ({home_name} ou Nul)": home_win + draw,
        f"X2 (Nul ou {away_name})": draw + away_win,
        f"12 ({home_name} ou {away_name})": home_win + away_win,
    }
    dnb = {
        f"DNB {home_name} (Remboursé si nul)": dnb_home,
        f"DNB {away_name} (Remboursé si nul)": dnb_away,
    }

    # Liste consolidée de tous les marchés disponibles
    all_markets = []
    all_markets.append({"Type": "1N2 (Vainqueur)", "Pari": f"Victoire {home_name}", "Probabilité": home_win})
    all_markets.append({"Type": "1N2 (Vainqueur)", "Pari": "Match Nul", "Probabilité": draw})
    all_markets.append({"Type": "1N2 (Vainqueur)", "Pari": f"Victoire {away_name}", "Probabilité": away_win})

    for lbl, p in double_chance.items(): all_markets.append({"Type": "Double Chance", "Pari": lbl, "Probabilité": p})
    for lbl, p in dnb.items(): all_markets.append({"Type": "Remboursé si Nul (DNB)", "Pari": lbl, "Probabilité": p})
    for lbl, p in unders.items(): all_markets.append({"Type": "Under/Moins de buts", "Pari": lbl, "Probabilité": p})
    for lbl, p in overs.items(): all_markets.append({"Type": "Over/Plus de buts", "Pari": lbl, "Probabilité": p})

    for m in all_markets:
        p = m["Probabilité"]
        m["Cote Min. Conseillée"] = round(1.0 / p, 2) if p > 0 else 99.0

    # Paris haute probabilité (>= 75%)
    safe_picks = [m for m in all_markets if m["Probabilité"] >= 0.75]
    safe_picks.sort(key=lambda x: x["Probabilité"], reverse=True)

    scores_sorted = sorted([(p, h, a) for (h, a), p in matrix.items()], reverse=True)[:5]

    return {
        "lambda_home": lambda_home, "lambda_away": lambda_away, "total": total_lambda,
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "double_chance": double_chance, "dnb": dnb,
        "unders": unders, "overs": overs,
        "safe_picks": safe_picks,
        "all_markets": all_markets,
        "scores": scores_sorted
    }


# ================= INTERFACE PRINCIPALE STREAMLIT =================
st.title("🌍 FootPredict Pro — Vainqueurs (1N2), Compos & Sync PMUG")

tab_api, tab_sqlite = st.tabs(["⚡ Matchs du Jour & Pronostics Complets", "📊 Bilan SQLite"])

with st.sidebar:
    st.header("🔑 Connexion API-Football")
    api_key = st.text_input("Collez votre clé API ici :", type="password", help="Obtenez votre clé gratuite sur dashboard.api-football.com")
    st.caption("Couvre 900+ ligues (Brésil, MLS, Argentine, etc.) avec compositions et blessures en direct.")
    
    st.divider()
    chosen_date = st.date_input("Date des matchs :", value=datetime.today())
    target_date_str = chosen_date.strftime("%Y-%m-%d")

with tab_api:
    if not api_key:
        st.warning("👈 Veuillez renseigner votre clé API dans le menu à gauche pour afficher les matchs.")
        st.info("💡 **Où trouver la clé ?** Inscrivez-vous gratuitement sur [dashboard.api-football.com](https://dashboard.api-football.com/register), allez dans l'onglet **API-Key** et collez-la à gauche.")
    else:
        with st.spinner(f"Récupération des matchs du {target_date_str}..."):
            fixtures = fetch_live_fixtures(api_key, target_date_str)

        if not fixtures:
            st.info("Aucun match trouvé pour cette date ou quota journalier atteint.")
        else:
            st.success(f"✅ {len(fixtures)} matchs trouvés dans le monde pour la date du {target_date_str} !")

            countries = sorted(list(set(f["league"]["country"] for f in fixtures if f.get("league"))))
            col_search1, col_search2 = st.columns([1, 2])
            with col_search1:
                sel_country = st.selectbox("Filtrer par Pays :", ["Tous les pays"] + countries, index=0)
            
            filtered_fixtures = fixtures
            if sel_country != "Tous les pays":
                filtered_fixtures = [f for f in fixtures if f["league"]["country"] == sel_country]

            match_options = {}
            for f in filtered_fixtures:
                dt_raw = f["fixture"]["date"]
                heure = dt_raw[11:16] if len(dt_raw) >= 16 else "N/A"
                label = f"⏰ {heure} | [{f['league']['country']} - {f['league']['name']}] {f['teams']['home']['name']} vs {f['teams']['away']['name']}"
                match_options[label] = f

            with col_search2:
                selected_match_label = st.selectbox("Sélectionnez le match PMUG :", list(match_options.keys()))

            target_fixture = match_options[selected_match_label]
            fixture_id = target_fixture["fixture"]["id"]
            h_id = target_fixture["teams"]["home"]["id"]
            a_id = target_fixture["teams"]["away"]["id"]
            h_name = target_fixture["teams"]["home"]["name"]
            a_name = target_fixture["teams"]["away"]["name"]
            league_name = f"{target_fixture['league']['country']} - {target_fixture['league']['name']}"
            match_time_str = target_fixture["fixture"]["date"][:16].replace("T", " ")

            if st.button(f"Calculer Vainqueur & Buts : {h_name} vs {a_name} 🚀", type="primary", use_container_width=True):
                with st.spinner("Calcul des probabilités 1N2, analyse de la forme, compositions et blessures..."):
                    h_rates = fetch_team_history(api_key, h_id, n=10)
                    a_rates = fetch_team_history(api_key, a_id, n=10)
                    lineups = fetch_fixture_lineups(api_key, fixture_id)
                    injuries = fetch_fixture_injuries(api_key, fixture_id)
                    lineup_analysis = evaluate_lineup_factors(lineups, injuries, h_id, a_id)

                    res = calculate_prediction_with_lineups(h_rates, a_rates, h_name, a_name, lineup_analysis)

                st.session_state["api_res"] = res
                st.session_state["api_match"] = (h_name, a_name, league_name, match_time_str)
                st.session_state["lineup_info"] = lineup_analysis

        if "api_res" in st.session_state:
            res = st.session_state["api_res"]
            hm, aw, comp, dt_str = st.session_state["api_match"]
            lin_info = st.session_state["lineup_info"]

            st.divider()
            st.subheader(f"📊 Analyse Complète : {hm} vs {aw}")
            st.caption(f"🏆 {comp} | 🗓️ Coup d'envoi : {dt_str}")

            # SECTION 1 : RÉSULTAT DU MATCH & VAINQUEUR (1N2)
            st.markdown("### 🏆 Probabilités du Vainqueur (1N2)")
            
            c_hw, c_dr, c_aw = st.columns(3)
            p_hw, p_dr, p_aw = res["home_win"] * 100, res["draw"] * 100, res["away_win"] * 100
            odds_hw = round(1.0 / res["home_win"], 2) if res["home_win"] > 0 else 99
            odds_dr = round(1.0 / res["draw"], 2) if res["draw"] > 0 else 99
            odds_aw = round(1.0 / res["away_win"], 2) if res["away_win"] > 0 else 99

            c_hw.metric(f"Victoire {hm}", f"{p_hw:.1f} %", f"Cote juste : {odds_hw}")
            c_dr.metric("Match Nul", f"{p_dr:.1f} %", f"Cote juste : {odds_dr}")
            c_aw.metric(f"Victoire {aw}", f"{p_aw:.1f} %", f"Cote juste : {odds_aw}")

            # Barre visuelle de répartition
            st.write(f"**Répartition des chances :** {hm} ({p_hw:.0f}%) | Nul ({p_dr:.0f}%) | {aw} ({p_aw:.0f}%)")
            st.progress(int(p_hw))

            # SECTION 2 : MARCHÉS DE SÉCURITÉ DU VAINQUEUR (DOUBLES CHANCES & DNB)
            col_dc, col_dnb = st.columns(2)
            with col_dc:
                st.markdown("#### 🛡️ Doubles Chances (1X, X2, 12)")
                for lbl, p in res["double_chance"].items():
                    st.write(f"• **{lbl}** : **{p*100:.1f} %** (Cote min : `{1.0/p:.2f}`)")
            with col_dnb:
                st.markdown("#### 🔄 Remboursé si Nul (Draw No Bet)")
                for lbl, p in res["dnb"].items():
                    st.write(f"• **{lbl}** : **{p*100:.1f} %** (Cote min : `{1.0/p:.2f}`)")

            st.divider()

            # SECTION 3 : BUTS ATTENDUS & MARCHÉS OVER / UNDER
            st.markdown("### ⚽ Marchés des Buts")
            colA, colB, colC = st.columns(3)
            colA.metric(f"Buts Attendus ({hm})", f"{res['lambda_home']:.2f}")
            colB.metric(f"Buts Attendus ({aw})", f"{res['lambda_away']:.2f}")
            colC.metric("Total Buts Espéré", f"{res['total']:.2f}")

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("##### Moins de buts (Under)")
                for k, v in res["unders"].items():
                    st.write(f"• {k} : **{v*100:.1f}%** (Cote min : `{1.0/v:.2f}`)")
            with g2:
                st.markdown("##### Plus de buts (Over)")
                for k, v in res["overs"].items():
                    st.write(f"• {k} : **{v*100:.1f}%** (Cote min : `{1.0/v:.2f}`)")

            # RAPPORT BLESSURES ET COMPOSITIONS
            with st.expander("📋 Rapport Tactique : Compositions & Blessures prises en compte"):
                if lin_info["is_official"]:
                    st.success("✅ **Compositions officielles publiées et intégrées au calcul !**")
                    c_h, c_a = st.columns(2)
                    with c_h:
                        st.write(f"**{hm} ({lin_info['home_formation']})**")
                        st.caption(", ".join(lin_info["home_xi"]))
                    with c_a:
                        st.write(f"**{aw} ({lin_info['away_formation']})**")
                        st.caption(", ".join(lin_info["away_xi"]))
                else:
                    st.info("ℹ️ Compositions officielles non publiées (disponibles ~45-60 min avant le coup d'envoi). Les blessures et suspensions connues ont été intégrées.")

                if lin_info["notes"]:
                    for n in lin_info["notes"]:
                        st.write(n)

            # TOP 5 SCORES EXACTS
            with st.expander("🎲 Top 5 des scores exacts les plus probables"):
                score_df = pd.DataFrame([{"Score": f"{h}-{a}", "Probabilité": f"{p*100:.2f}%"} for p, h, a in res["scores"]])
                st.dataframe(score_df, hide_index=True)

            st.divider()

            # SECTION 4 : FORMULAIRE DE SAUVEGARDE SQLITE POUR TOUS LES MARCHÉS
            st.markdown("### 💾 Enregistrer un pronostic dans la base SQLite")
            
            # Liste complète de tous les paris possibles
            all_options = [f"[{m['Type']}] {m['Pari']} — {m['Probabilité']*100:.1f}% (Cote min : {m['Cote Min. Conseillée']})" for m in res["all_markets"]]

            with st.form("api_save_form"):
                f1, f2, f3 = st.columns([3, 1, 1])
                with f1:
                    chosen_pick_str = st.selectbox("Choisir le pronostic à enregistrer (1N2, Double Chance, DNB ou Buts) :", all_options)
                with f2:
                    idx = all_options.index(chosen_pick_str)
                    target_pick = res["all_markets"][idx]
                    taken_odds = st.number_input("Cote proposée sur PMUG", min_value=1.01, max_value=25.0, value=float(target_pick["Cote Min. Conseillée"]), step=0.05)
                with f3:
                    st.write(""); st.write("")
                    btn_save = st.form_submit_button("Enregistrer le pari 💾", use_container_width=True)

                if btn_save:
                    save_prediction(
                        match_dt=dt_str, competition=comp, home=hm, away=aw,
                        bet_type=target_pick["Type"], bet_name=target_pick["Pari"],
                        prob=target_pick["Probabilité"], min_odds=target_pick["Cote Min. Conseillée"],
                        taken_odds=taken_odds
                    )
                    st.success(f"Pari '{target_pick['Pari']}' enregistré dans SQLite !")


# ----------------- ONGLET 2 : BILAN SQLITE -----------------
with tab_sqlite:
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
        st.dataframe(
            db_data[["id", "created_at", "match_datetime", "home_team", "away_team", "bet_type", "bet_name", "probability", "min_odds", "taken_odds", "status"]],
            hide_index=True,
            use_container_width=True
        )

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