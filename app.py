import math
from io import StringIO
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FootPredict AI", page_icon="⚽", layout="wide")

REQUIRED = ["date", "competition", "home_team", "away_team", "home_goals", "away_goals"]

def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def under_probability(total_lambda, max_goals_exclusive):
    return sum(poisson_pmf(k, total_lambda) for k in range(max_goals_exclusive))

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
    # ISO-like date sorting where possible; non-date labels are placed last.
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
    if not rows:
        return None
    return {
        "played": len(rows),
        "scored": sum(x for x, _ in rows) / len(rows),
        "conceded": sum(y for _, y in rows) / len(rows),
        "goals_for": sum(x for x, _ in rows),
        "goals_against": sum(y for _, y in rows)
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
        raise ValueError("Historique insuffisant pour : " + ", ".join(missing) +
                         ". Importez des résultats contenant ces équipes.")

    # Use the chosen competition's home/away scoring averages as baselines.
    if len(data):
        avg_home = max(data["home_goals"].mean(), 0.25)
        avg_away = max(data["away_goals"].mean(), 0.25)
    else:
        avg_home, avg_away = 1.35, 1.10

    def blend(venue_rate, overall_rate):
        if venue_rate is None:
            return overall_rate
        # Shrink venue rate toward overall form unless at least 5 venue matches exist.
        weight = min(venue_rate["played"] / 5.0, 1.0)
        return weight * venue_rate["scored"] + (1 - weight) * overall_rate["scored"], \
               weight * venue_rate["conceded"] + (1 - weight) * overall_rate["conceded"]

    if h_home:
        h_scored, h_conceded = blend(h_home, h_all)
    else:
        h_scored, h_conceded = h_all["scored"], h_all["conceded"]
    if a_away:
        a_scored, a_conceded = blend(a_away, a_all)
    else:
        a_scored, a_conceded = a_all["scored"], a_all["conceded"]

    # Attack/defence factors against the dataset's scoring baseline.
    lambda_home = avg_home * (h_scored / avg_home) * (a_conceded / avg_home)
    lambda_away = avg_away * (a_scored / avg_away) * (h_conceded / avg_away)
    lambda_home = min(max(lambda_home, 0.10), 4.50)
    lambda_away = min(max(lambda_away, 0.10), 4.00)
    total = lambda_home + lambda_away

    unders = {
        "Moins de 1,5": under_probability(total, 2),
        "Moins de 2,5": under_probability(total, 3),
        "Moins de 3,5": under_probability(total, 4),
        "Moins de 4,5": under_probability(total, 5),
        "Moins de 5,5": under_probability(total, 6),
    }
    scores = []
    home_win = draw = away_win = 0.0
    for hg in range(11):
        for ag in range(11):
            p = poisson_pmf(hg, lambda_home) * poisson_pmf(ag, lambda_away)
            scores.append((p, hg, ag))
            if hg > ag: home_win += p
            elif hg == ag: draw += p
            else: away_win += p
    scores.sort(reverse=True)
    notes = []
    if h_all["played"] < 5 or a_all["played"] < 5:
        notes.append("Échantillon faible : moins de 5 matchs pour au moins une équipe.")
    if not h_home or h_home["played"] < 3:
        notes.append("Peu de données à domicile pour l'équipe recevante.")
    if not a_away or a_away["played"] < 3:
        notes.append("Peu de données à l'extérieur pour l'équipe visiteuse.")
    notes.append("Modèle de base Poisson indépendant : les blessures, compositions, météo, motivation et niveau des compétitions ne sont pas encore intégrés.")
    notes.append("Les probabilités sont des estimations mathématiques, pas des garanties. Leur fiabilité dépend de la qualité et de la représentativité des données importées.")
    return {
        "lambda_home": lambda_home, "lambda_away": lambda_away, "total": total,
        "unders": unders, "overs": {k: 1-v for k, v in unders.items()},
        "scores": scores[:10], "home_win": home_win, "draw": draw, "away_win": away_win,
        "h_all": h_all, "a_all": a_all, "h_home": h_home, "a_away": a_away, "notes": notes,
        "n_data": len(data)
    }

st.title("⚽ FootPredict AI")
st.caption("Prototype d'analyse statistique — calcul local dans le navigateur du service Streamlit, sans abonnement API requis.")
st.warning("Important : aucune base de données mondiale n'est incluse. Importez des résultats historiques fiables. Le fichier d'exemple est fictif et ne doit pas servir à pronostiquer de vrais matchs.")

with st.sidebar:
    st.header("Données")
    uploaded = st.file_uploader("Importer l'historique des matchs (CSV)", type=["csv"])
    st.markdown("**Colonnes obligatoires**")
    st.code("date, competition, home_team, away_team, home_goals, away_goals", language="text")
    st.markdown("Dates conseillées : `YYYY-MM-DD`.")
    st.markdown("Le fichier reste utilisé pour cette session ; le prototype ne constitue pas un stockage permanent.")

if uploaded is not None:
    try:
        matches = normalise_matches(pd.read_csv(uploaded))
        st.sidebar.success(f"{len(matches)} lignes de match valides chargées.")
    except Exception as e:
        st.error(f"Import CSV impossible : {e}")
        st.stop()
else:
    sample_path = "data/exemple_fictif.csv"
    try:
        matches = normalise_matches(pd.read_csv(sample_path))
    except Exception:
        matches = pd.DataFrame(columns=REQUIRED)

if matches.empty:
    st.info("Importez un CSV contenant les résultats historiques pour commencer.")
    st.stop()

st.subheader("1. Sélection du match")
teams = sorted(set(matches["home_team"].tolist() + matches["away_team"].tolist()))
comps = ["Toutes les compétitions"] + sorted([x for x in matches["competition"].dropna().unique().tolist() if x])
c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    home = st.selectbox("Équipe à domicile", teams, index=0)
with c2:
    away_default = 1 if len(teams) > 1 else 0
    away = st.selectbox("Équipe à l'extérieur", teams, index=away_default)
with c3:
    competition = st.selectbox("Filtrer par compétition", comps, index=0)
n = st.slider("Nombre de matchs récents utilisés par équipe", min_value=5, max_value=15, value=10)

if home.casefold() == away.casefold():
    st.error("Choisissez deux équipes différentes.")
else:
    if st.button("Calculer les probabilités", type="primary", use_container_width=True):
        try:
            result = calculate_prediction(matches, home, away, n, competition)
            st.session_state["result"] = result
            st.session_state["last_match"] = (home, away, competition, n)
        except Exception as e:
            st.error(str(e))

if "result" in st.session_state:
    r = st.session_state["result"]
    hm, aw, comp, n_used = st.session_state["last_match"]
    st.divider()
    st.subheader(f"2. Résultats : {hm} — {aw}")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Buts attendus — {hm}", f"{r['lambda_home']:.2f}")
    m2.metric(f"Buts attendus — {aw}", f"{r['lambda_away']:.2f}")
    m3.metric("Total de buts attendu", f"{r['total']:.2f}")
    st.markdown("### Probabilités des marchés de buts")
    rows = []
    for label, p in r["unders"].items():
        rows.append({"Marché": label, "Probabilité de moins": f"{p*100:.1f}%", "Probabilité de plus ou égal": f"{(1-p)*100:.1f}%"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Pour « moins de 2,5 », moins signifie 0, 1 ou 2 buts ; plus signifie 3 buts ou davantage.")
    st.markdown("### Résultat 1N2 estimé")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Victoire {hm}", f"{r['home_win']*100:.1f}%")
    c2.metric("Match nul", f"{r['draw']*100:.1f}%")
    c3.metric(f"Victoire {aw}", f"{r['away_win']*100:.1f}%")
    st.markdown("### 10 scores exacts les plus probables")
    score_df = pd.DataFrame([{"Score": f"{h}-{a}", "Probabilité": f"{p*100:.2f}%"} for p,h,a in r["scores"]])
    st.dataframe(score_df, hide_index=True, use_container_width=True)
    st.markdown("### Données utilisées")
    d1, d2 = st.columns(2)
    with d1:
        st.write(f"**{hm} — tous terrains :** {r['h_all']['played']} matchs")
        st.write(f"Buts marqués/match : {r['h_all']['scored']:.2f}")
        st.write(f"Buts encaissés/match : {r['h_all']['conceded']:.2f}")
        if r["h_home"]:
            st.write(f"À domicile : {r['h_home']['played']} matchs")
    with d2:
        st.write(f"**{aw} — tous terrains :** {r['a_all']['played']} matchs")
        st.write(f"Buts marqués/match : {r['a_all']['scored']:.2f}")
        st.write(f"Buts encaissés/match : {r['a_all']['conceded']:.2f}")
        if r["a_away"]:
            st.write(f"À l'extérieur : {r['a_away']['played']} matchs")
    st.markdown("### Limites et avertissements")
    for note in r["notes"]:
        st.write("• " + note)

with st.expander("Voir les données chargées"):
    st.write(f"Nombre total de matchs chargés : {len(matches)}")
    st.dataframe(matches.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_bytes = matches.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Télécharger les données nettoyées", data=csv_bytes, file_name="matchs_nettoyes.csv", mime="text/csv")

st.divider()
st.caption("FootPredict AI — prototype gratuit. Pour un usage sérieux, tester le modèle sur des saisons historiques et vérifier sa calibration avant d'utiliser les estimations pour prendre des décisions.")
