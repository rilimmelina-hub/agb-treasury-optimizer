# -*- coding: utf-8 -*-
"""AGB — Salle des marches — Optimiseur quantitatif de portefeuille v2
Moteur QP rendement / risque de taux avec 6 rubriques, graphes avances et export."""
import io
import json
import os
import re
import unicodedata
import numpy as np
import pandas as pd
import cvxpy as cp
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ==================================================================
# PARAMETRES DE GESTION (valeurs par defaut — configurables en sidebar)
# ==================================================================
FREQ_CPN = 1
# semaine ouvree AGB Alger : dimanche a jeudi, weekend vendredi-samedi (pas lundi-vendredi)
JOURS_OUVRES_WEEKMASK = "Sun Mon Tue Wed Thu"
JOUR_OUVRE_CBD = pd.offsets.CustomBusinessDay(weekmask=JOURS_OUVRES_WEEKMASK)
TAUX_IMPOT = 0.10
SEUIL_EXONERATION_ANS = 5
DEFAULT_PART_PLACEMENT = 0.90
DEFAULT_PART_TRADING = 0.10
DEFAULT_SEUIL_COURT_ANS = 2.0
DEFAULT_SEUIL_LONG_ANS = 5.0
# repartition cible de la poche placement par tranche de maturite (independante des seuils
# ci-dessus, qui ne font que definir les bornes court/moyen/long en annees)
DEFAULT_PART_COURT = 0.20
DEFAULT_PART_MOYEN = 0.50
DEFAULT_PART_LONG = 0.30
COMMISSION_OAT_ANNUELLE = 0.00025
COMMISSION_FIXE_OPERATION = 170_000
CONTREPARTIE_SANS_LIMITE = {"TRS"}
TYPES_NON_CESSIBLES = ("SSI",)
NOUVEAUX_TITRES = [
    ("BTC 2Y", 2.33, 0.030, 0.048, "TRS"),
    ("BTA 4Y", 4.58, 0.040, 0.052, "TRS"),
    ("OAT 7Y", 6.83, 0.050, 0.066, "TRS"),
]
RISK_DEFAULTS = dict(
    vol_court_bp=90.0, vol_moyen_bp=65.0, vol_long_bp=45.0,
    rho_cm=0.85, rho_ml=0.85, rho_cl=0.55,
    kappa_conv=0.02, kappa_cp=0.02, lambda_risk=60.0, coupon_cible_kda=200.0,
)
COLONNES_REQUISES = ["Type", "Date d\'echeance", "Tx Nominal", "Tx de Rendement", "Valeur Nominal", "SVT/Client"]


# ==================================================================
# MATHEMATIQUES OBLIGATAIRES
# ==================================================================
def duration_modifiee(c, ytm, t_res, freq=1):
    if t_res <= 0:
        return 0.0
    k = int(np.ceil(t_res * freq))
    times = t_res - np.arange(k)[::-1] / freq
    times = times[times > 1e-9]
    cf = np.full(len(times), c / freq)
    cf[-1] += 1.0
    pv = cf * (1 + ytm) ** (-times)
    return float((times * pv).sum() / pv.sum() / (1 + ytm))


def convexite_modifiee(c, ytm, t_res, freq=1):
    if t_res <= 0:
        return 0.0
    k = int(np.ceil(t_res * freq))
    times = t_res - np.arange(k)[::-1] / freq
    times = times[times > 1e-9]
    cf = np.full(len(times), c / freq)
    cf[-1] += 1.0
    pv = cf * (1 + ytm) ** (-times)
    return float((pv * times * (times + 1)).sum() / pv.sum() / (1 + ytm) ** 2)


def matrice_risque(vol_court_bp, vol_moyen_bp, vol_long_bp, rho_cm, rho_ml, rho_cl):
    """Matrice de covariance 3 facteurs (court/moyen/long), en vol annuelle de rendement."""
    vol = np.array([vol_court_bp, vol_moyen_bp, vol_long_bp]) / 10000.0
    rho = np.array([[1.0, rho_cm, rho_cl], [rho_cm, 1.0, rho_ml], [rho_cl, rho_ml, 1.0]])
    Sigma = np.outer(vol, vol) * rho
    eigval, eigvec = np.linalg.eigh(Sigma)
    eigval = np.clip(eigval, 1e-12, None)
    return eigvec @ np.diag(eigval) @ eigvec.T


# ==================================================================
# UNIVERS DE MARCHE (titres en circulation + historique de ventes)
# ==================================================================
_DUREE_RE = re.compile(r"(\d+)\s*(AN|ANS|SEMAINE|SEMAINES|MOIS)", re.IGNORECASE)
_TAUX_RE = re.compile(r"(\d+[,.]\d+)\s*%")
_ECHEANCE_RE = re.compile(r"(\d{2}/\d{2}/\d{4})")


def _norm_col(c):
    c = unicodedata.normalize("NFKD", str(c).strip()).encode("ascii", "ignore").decode("ascii")
    return c.lower()


def _duree_en_annees(texte):
    if not isinstance(texte, str):
        return None
    m = _DUREE_RE.search(texte.upper())
    if not m:
        return None
    n, unite = float(m.group(1)), m.group(2).upper()
    if unite.startswith("AN"):
        return n
    if unite.startswith("SEMAINE"):
        return n * 7 / 365.25
    return n / 12


def _taux_depuis_texte(texte):
    if not isinstance(texte, str):
        return None
    m = _TAUX_RE.search(texte)
    return float(m.group(1).replace(",", ".")) / 100 if m else None


def _label_type(prefixe, duree_ans):
    if duree_ans < 1:
        return f"{prefixe} {round(duree_ans * 12)}M"
    return f"{prefixe} {int(round(duree_ans))}Y"


def _parse_nombre_fr(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in ("", "-"):
        return None
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def charger_courbe_taux(_fichier_courbe):
    """Lit une courbe de taux (2 colonnes : tenor, taux)."""
    df = pd.read_excel(_fichier_courbe, header=None).dropna()
    if df.empty:
        return None
    tenors_txt = df.iloc[:, 0].astype(str)
    tenors = np.where(
        tenors_txt.str.contains("m", case=False),
        tenors_txt.str.extract(r"(\d+[.,]?\d*)")[0].str.replace(",", ".").astype(float) / 12,
        tenors_txt.str.extract(r"(\d+[.,]?\d*)")[0].str.replace(",", ".").astype(float),
    )
    taux = df.iloc[:, 1].astype(float).values
    taux = np.where(taux > 1, taux / 100.0, taux)
    ordre = np.argsort(tenors)
    return tenors[ordre], taux[ordre]


def _rendement_depuis_courbe(duree_ans, courbe):
    tenors, taux = courbe
    return float(np.interp(duree_ans, tenors, taux))


def charger_univers_marche(_fichier_marche, _courbe=None):
    """Construit l\'univers des titres pouvant faire l\'objet d\'une nouvelle adjudication."""
    xls = pd.ExcelFile(_fichier_marche)
    lignes = []
    nb_ecartes = 0

    def _rendement(duree, taux_coupon):
        return _rendement_depuis_courbe(duree, _courbe) if _courbe is not None else taux_coupon

    if "BTA" in xls.sheet_names:
        bta = pd.read_excel(_fichier_marche, sheet_name="BTA")
        bta.columns = [_norm_col(c) for c in bta.columns]
        bta = bta.dropna(subset=["code isin"])
        for _, r in bta.iterrows():
            duree = _duree_en_annees(r.get("caracteristiques"))
            taux = _taux_depuis_texte(r.get("caracteristiques"))
            ech = pd.to_datetime(r.get("date d\'echeance"), errors="coerce")
            if duree is None or taux is None or pd.isna(ech):
                nb_ecartes += 1
                continue
            vn_unit = _parse_nombre_fr(r.get("valeur nominale"))
            qte = _parse_nombre_fr(r.get("quantite en cours"))
            encours = vn_unit * qte if (vn_unit is not None and qte is not None) else None
            lignes.append((_label_type("BTA", duree), ech, taux, _rendement(duree, taux), "TRS", encours))

    if "BTC" in xls.sheet_names:
        btc = pd.read_excel(_fichier_marche, sheet_name="BTC")
        btc.columns = [_norm_col(c) for c in btc.columns]
        btc = btc.dropna(subset=["code isin"])
        for _, r in btc.iterrows():
            duree = _duree_en_annees(r.get("caracteristiques"))
            taux = _taux_depuis_texte(r.get("caracteristiques"))
            ech = pd.to_datetime(r.get("date d\'echeance"), errors="coerce")
            if duree is None or pd.isna(ech):
                nb_ecartes += 1
                continue
            if taux is None:
                if _courbe is None:
                    nb_ecartes += 1
                    continue
                taux = 0.0
            lignes.append((_label_type("BTC", duree), ech, taux, _rendement(duree, taux), "TRS", None))

    if "OAT" in xls.sheet_names:
        oat = pd.read_excel(_fichier_marche, sheet_name="OAT")
        oat.columns = [_norm_col(c) for c in oat.columns]
        oat = oat.dropna(subset=["code isin"])
        for _, r in oat.iterrows():
            lib = r.get("libelle valeur")
            duree = _duree_en_annees(lib)
            taux = _taux_depuis_texte(lib)
            m = _ECHEANCE_RE.search(lib) if isinstance(lib, str) else None
            ech = pd.to_datetime(m.group(1), format="%d/%m/%Y", errors="coerce") if m else pd.NaT
            if duree is None or taux is None or pd.isna(ech):
                nb_ecartes += 1
                continue
            encours = _parse_nombre_fr(r.get("montant (da)"))
            lignes.append((_label_type("OAT", duree), ech, taux, _rendement(duree, taux), "TRS", encours))

    if not lignes:
        return None, nb_ecartes
    df = pd.DataFrame(lignes, columns=["Type", "Echeance", "Coupon", "Rendement", "Contrepartie", "Encours_max_DA"])
    df = df.drop_duplicates(subset=["Type", "Echeance"]).reset_index(drop=True)
    return df, nb_ecartes


def charger_historique_ventes(_fichier_marche):
    """Lit la feuille d\'historique de ventes si presente."""
    xls = pd.ExcelFile(_fichier_marche)
    nom_feuille = next((s for s in xls.sheet_names if _norm_col(s).startswith("sld")), None)
    if nom_feuille is None:
        return None
    df = pd.read_excel(_fichier_marche, sheet_name=nom_feuille, header=1)
    noms = [
        "Type", "Date_valeur", "Date_rachat", "Date_echeance_admin", "Date_echeance_paiement",
        "Duree_residuelle_j", "SVT", "Tx_nominal", "Tx_rendement", "Valeur_nominale",
        "Prix_pied_coupon", "Duree_restante", "Contrepartie", "Date_vente", "Prix_achat",
        "Prix_vente", "Plus_value",
    ]
    df = df.iloc[:, : len(noms)]
    df.columns = noms
    df = df.dropna(subset=["Type"]).reset_index(drop=True)
    df["Date_vente"] = pd.to_datetime(df["Date_vente"], errors="coerce")
    df["Date_valeur"] = pd.to_datetime(df["Date_valeur"], errors="coerce")
    maturite_ans = df["Type"].str.extract(r"(\d+)\s*Y")[0].astype(float)
    exonere = maturite_ans.fillna(0) >= SEUIL_EXONERATION_ANS
    is_OAT = df["Type"].str.startswith("OAT")
    duree_detenue_ans = ((df["Date_vente"] - df["Date_valeur"]).dt.days / 365.25).clip(lower=0).fillna(0)
    impot_DA = np.where(exonere, 0.0, df["Plus_value"] * TAUX_IMPOT)
    commission_oat_DA = np.where(is_OAT, df["Valeur_nominale"] * COMMISSION_OAT_ANNUELLE * duree_detenue_ans, 0.0)
    df["Exonere"] = np.where(exonere, "Oui", "Non")
    df["Plus_value_nette"] = df["Plus_value"] - impot_DA - commission_oat_DA - COMMISSION_FIXE_OPERATION
    return df


# ==================================================================
# VALIDATION DES FICHIERS
# ==================================================================
def valider_fichier_position(fichier):
    """Verifie que le fichier de position contient les colonnes requises.
    Retourne (df_brut, liste_erreurs)."""
    erreurs = []
    try:
        df = pd.read_excel(fichier)
    except Exception as e:
        return None, [f"Impossible de lire le fichier Excel : {e}"]
    if df.empty:
        return None, ["Le fichier est vide."]
    colonnes = [str(c).strip() for c in df.columns]
    mapping = {
        "Type": "Type",
        "Date d'echeance": ["Date d'echeance", "Date d'\u00e9ch\u00e9ance", "Echeance"],
        "Tx Nominal": ["Tx Nominal", "Tx nominal", "Coupon"],
        "Tx de Rendement": ["Tx de Rendement", "Tx de rendement", "Rendement"],
        "Valeur Nominal": ["Valeur Nominal", "Valeur nominal", "VN", "Valeur Nominale"],
        "SVT/Client": ["SVT/Client", "SVT", "Contrepartie"],
    }
    for cle, variants in mapping.items():
        if isinstance(variants, str):
            variants = [variants]
        if not any(v in colonnes for v in variants):
            erreurs.append(f"Colonne manquante : '{cle}' (recherchee parmi {variants}).")
    if erreurs:
        return df, erreurs
    vn_col = next((c for c in colonnes if c in mapping["Valeur Nominal"]), None)
    if vn_col:
        vn_serie = pd.to_numeric(df[vn_col], errors="coerce")
        if (vn_serie <= 0).any():
            nb_neg = (vn_serie <= 0).sum()
            erreurs.append(f"{nb_neg} ligne(s) avec Valeur Nominal <= 0 — ces lignes seront ignorees.")
    return df, erreurs


# ==================================================================
# PREPARATION DES DONNEES
# ==================================================================
def preparer_donnees(fichier, marche_df=None, date_eval=None,
                     seuil_court=DEFAULT_SEUIL_COURT_ANS, seuil_long=DEFAULT_SEUIL_LONG_ANS,
                     part_placement=DEFAULT_PART_PLACEMENT, part_trading=DEFAULT_PART_TRADING,
                     part_court=DEFAULT_PART_COURT, part_moyen=DEFAULT_PART_MOYEN, part_long=DEFAULT_PART_LONG):
    # normalise pour garantir que les 3 parts somment exactement a 1 (la contrainte QP l'exige)
    _somme_parts = part_court + part_moyen + part_long
    if _somme_parts <= 0:
        part_court, part_moyen, part_long = DEFAULT_PART_COURT, DEFAULT_PART_MOYEN, DEFAULT_PART_LONG
    else:
        part_court, part_moyen, part_long = part_court / _somme_parts, part_moyen / _somme_parts, part_long / _somme_parts
    df = pd.read_excel(fichier)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Date d'echeance": "Echeance", "Tx Nominal": "Coupon",
                             "Tx de Rendement": "Rendement", "Valeur Nominal": "VN",
                             "SVT/Client": "Contrepartie"})
    df = df.dropna(subset=["Type", "Echeance"]).reset_index(drop=True)
    df["Echeance"] = pd.to_datetime(df["Echeance"])
    df = df[df["Echeance"] > date_eval].sort_values("Echeance").reset_index(drop=True)
    df.insert(0, "ID", [f"L{i+1:03d}" for i in range(len(df))])
    nb_existant = len(df)

    if marche_df is not None and len(marche_df):
        lignes_nv = marche_df[marche_df["Echeance"] > date_eval].copy().reset_index(drop=True)
        lignes_nv.insert(0, "ID", [f"NV{i+1:03d}" for i in range(len(lignes_nv))])
        lignes_nv["VN"] = 0.0
    else:
        lignes = []
        for i, (typ, delai, cpn_h, rdt_h, ctp) in enumerate(NOUVEAUX_TITRES):
            lignes.append({"ID": f"NV{i+1:03d}", "Type": typ,
                            "Echeance": date_eval + pd.Timedelta(days=int(delai * 365.25)),
                            "Coupon": cpn_h, "Rendement": rdt_h, "VN": 0.0, "Contrepartie": ctp,
                            "Encours_max_DA": None})
        lignes_nv = pd.DataFrame(lignes)
    df = pd.concat([df, lignes_nv], ignore_index=True)
    df["Nouveau_titre"] = np.where(np.arange(len(df)) >= nb_existant, "Oui", "Non")
    for c in ["Coupon", "Rendement"]:
        if df[c].max() > 1:
            df[c] = df[c] / 100.0
    n = len(df)

    col_exo = next((c for c in df.columns if str(c).strip().lower() in ("exonere", "exon\u00e9r\u00e9", "exoneree", "exon\u00e9r\u00e9e")), None)
    if col_exo is not None:
        exonere = (df[col_exo].astype(str).str.strip().str.lower().isin(("oui", "o", "yes", "y", "1", "true", "vrai"))).values
    else:
        mat_emission = df["Type"].str.extract(r"(\d+)\s*Y")[0].astype(float)
        exonere = (mat_emission.fillna(0) >= SEUIL_EXONERATION_ANS).values
    df["Exonere"] = np.where(exonere, "Oui", "Non")
    df["Rendement_net"] = df["Rendement"] * np.where(exonere, 1.0, 1.0 - TAUX_IMPOT)
    is_OAT = df["Type"].str.startswith("OAT").values
    df["Rendement_net"] = df["Rendement_net"] - COMMISSION_OAT_ANNUELLE * is_OAT
    y = df["Rendement_net"].values

    cpn = df["Coupon"].values
    VN = df["VN"].values.astype(float)
    BUDGET = VN.sum()
    if BUDGET <= 0:
        raise ValueError("La somme des valeurs nominales est <= 0. Verifiez vos donnees.")

    encours_max = pd.to_numeric(df.get("Encours_max_DA"), errors="coerce").values if "Encours_max_DA" in df.columns else np.full(len(df), np.nan)
    plafond_poids = np.where(np.isfinite(encours_max), np.minimum(encours_max / max(BUDGET, 1.0), 1.0), 1.0)

    T = (df["Echeance"] - date_eval).dt.days.values / 365.25
    df["Delai_annees"] = np.round(T, 2)
    D = np.array([duration_modifiee(cpn[i], y[i], T[i], FREQ_CPN) for i in range(n)])
    Conv = np.array([convexite_modifiee(cpn[i], y[i], T[i], FREQ_CPN) for i in range(n)])
    df["Duration_mod"] = np.round(D, 3)
    df["Convexite"] = np.round(Conv, 3)

    w_act = VN / BUDGET
    m_court = (T <= seuil_court).astype(float)
    m_long = (T > seuil_long).astype(float)
    m_moyen = 1.0 - m_court - m_long

    rep_ctp = pd.Series(w_act, index=df["Contrepartie"]).groupby(level=0).sum().sort_values(ascending=False)
    rep_type = pd.Series(w_act, index=df["Type"].str.split().str[0]).groupby(level=0).sum()
    ctps = [c for c in rep_ctp.index if c not in CONTREPARTIE_SANS_LIMITE]

    mois_coupon = df["Echeance"].dt.month.values
    CouponMonth = np.zeros((12, n))
    for i in range(n):
        CouponMonth[mois_coupon[i] - 1, i] = cpn[i]
    df["Mois_coupon"] = mois_coupon

    return dict(
        df=df, n=n, y=y, D=D, Conv=Conv, w_act=w_act, BUDGET=BUDGET,
        m_court=m_court, m_moyen=m_moyen, m_long=m_long, ctps=ctps,
        rep_ctp=rep_ctp, rep_type=rep_type, T=T, CouponMonth=CouponMonth,
        plafond_poids=plafond_poids,
        seuil_court=seuil_court, seuil_long=seuil_long,
        part_placement=part_placement, part_trading=part_trading,
        part_court=part_court, part_moyen=part_moyen, part_long=part_long,
    )


# ==================================================================
# MOTEUR D'OPTIMISATION (programmation quadratique convexe)
# ==================================================================
def resoudre_qp(data, rdt_cible, d_max, vente_max, risk_params, mode="cible"):
    n, y, D, Conv, w_act = data["n"], data["y"], data["D"], data["Conv"], data["w_act"]
    m_court, m_moyen, m_long, ctps, df = data["m_court"], data["m_moyen"], data["m_long"], data["ctps"], data["df"]
    CouponMonth = data["CouponMonth"]
    K = len(ctps)
    part_P = data["part_placement"]
    part_T = data["part_trading"]

    Sigma = matrice_risque(risk_params["vol_court_bp"], risk_params["vol_moyen_bp"], risk_params["vol_long_bp"],
                            risk_params["rho_cm"], risk_params["rho_ml"], risk_params["rho_cl"])
    BucketExp = np.vstack([m_court, m_moyen, m_long]) * D[None, :]

    if K:
        ind_mat = np.array([(df["Contrepartie"] == c).astype(float).values for c in ctps])
        delta = ind_mat - ind_mat.mean(axis=0, keepdims=True)
    else:
        delta = np.zeros((0, n))

    wP = cp.Variable(n, nonneg=True)
    wT = cp.Variable(n, nonneg=True)
    w = wP + wT
    z = cp.Variable(K, nonneg=True) if K else None

    non_cessible = df["Type"].str.startswith(TYPES_NON_CESSIBLES).values
    plancher_vente = np.where(non_cessible, w_act, (1.0 - vente_max) * w_act)
    plafond_poids = data["plafond_poids"]

    cons = [
        cp.sum(w) == 1.0,
        cp.sum(wT) == part_T,
        m_court @ wP == data["part_court"] * part_P,
        m_moyen @ wP == data["part_moyen"] * part_P,
        m_long @ wP == data["part_long"] * part_P,
        w >= plancher_vente,
        w <= plafond_poids,
        D @ w <= d_max,
    ]
    if K:
        cons += [delta @ w <= z, -delta @ w <= z]

    b = BucketExp @ w
    risque = cp.quad_form(b, cp.psd_wrap(Sigma))
    conv_term = risk_params["kappa_conv"] * (Conv @ w)
    cp_penalty = risk_params["kappa_cp"] * cp.sum(z) if K else 0.0
    flux_mensuel = CouponMonth @ w
    # Contrainte dure : coupon annuel cible, en moyenne, avec une bande de tolerance (abattement)
    # car viser le meme montant exact chaque mois n'est pas realiste. flux_mensuel est en taux
    # (fraction du portefeuille, pas en DA) : il faut le mettre a l'echelle du budget pour le
    # comparer a une cible en DA, sans quoi la contrainte est toujours infaisable (un taux ~0.05
    # ne peut jamais depasser des centaines de milliers/millions de DA).
    BUDGET_QP = float(data["BUDGET"])
    coupon_cible_mensuel_DA = risk_params.get("coupon_cible_kda", 0.0) * 1000.0  # kDA -> DA
    if coupon_cible_mensuel_DA > 0:
        coupon_annuel_cible_DA = coupon_cible_mensuel_DA * 12.0
        abattement = risk_params.get("abattement_coupon_pct", 0.0) / 100.0
        borne_basse_DA = coupon_annuel_cible_DA * (1.0 - abattement)
        cons += [cp.sum(flux_mensuel) * BUDGET_QP >= borne_basse_DA]
    if mode == "cible":
        cons = cons + [y @ w >= rdt_cible]
        objectif = cp.Minimize(risk_params["lambda_risk"] * risque - conv_term + cp_penalty)
    else:
        objectif = cp.Minimize(risk_params["lambda_risk"] * risque - (y @ w) - conv_term + cp_penalty)

    prob = cp.Problem(objectif, cons)
    last_error = ""
    for solver in (cp.CLARABEL, cp.OSQP, cp.SCS):
        try:
            prob.solve(solver=solver)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                return np.clip(w.value, 0, None), np.clip(wP.value, 0, None), np.clip(wT.value, 0, None), prob.status
        except Exception as e:
            last_error = f"{solver}: {e}"
            continue
    detail = f" Dernier solveur : {last_error}" if last_error else f" Status : {prob.status}"
    raise RuntimeError(f"Aucune solution trouvee.{detail} Assouplissez la duration max, la vente max ou le rendement cible.")


def run_optimization(fichier, rdt_cible, d_max, vente_max, risk_params, marche_df=None, date_eval=None,
                     seuil_court=DEFAULT_SEUIL_COURT_ANS, seuil_long=DEFAULT_SEUIL_LONG_ANS,
                     part_placement=DEFAULT_PART_PLACEMENT, part_trading=DEFAULT_PART_TRADING,
                     part_court=DEFAULT_PART_COURT, part_moyen=DEFAULT_PART_MOYEN, part_long=DEFAULT_PART_LONG):
    data = preparer_donnees(fichier, marche_df, date_eval, seuil_court, seuil_long, part_placement, part_trading,
                            part_court, part_moyen, part_long)
    df, n, y, D, Conv = data["df"], data["n"], data["y"], data["D"], data["Conv"]
    w_act, BUDGET = data["w_act"], data["BUDGET"]
    m_court, m_moyen, m_long, ctps = data["m_court"], data["m_moyen"], data["m_long"], data["ctps"]

    coupon_actif = risk_params.get("coupon_cible_kda", 0.0) > 0
    if coupon_actif:
        w_cib, wP_cib, wT_cib, status = resoudre_qp(data, rdt_cible, d_max, vente_max, risk_params, mode="maximiser")
        mode_utilise = "maximiser"
    else:
        w_cib, wP_cib, wT_cib, status = resoudre_qp(data, rdt_cible, d_max, vente_max, risk_params, mode="cible")
        mode_utilise = "cible"
        if w_cib is None:
            w_cib, wP_cib, wT_cib, status = resoudre_qp(data, rdt_cible, d_max, vente_max, risk_params, mode="maximiser")
            mode_utilise = "maximiser"
    if w_cib is None:
        raise RuntimeError(f"Aucune solution trouvee ({status}).")

    Sigma = matrice_risque(risk_params["vol_court_bp"], risk_params["vol_moyen_bp"], risk_params["vol_long_bp"],
                            risk_params["rho_cm"], risk_params["rho_ml"], risk_params["rho_cl"])
    BucketExp = np.vstack([m_court, m_moyen, m_long]) * D[None, :]

    rdt_act, dur_act = float(w_act @ y), float(w_act @ D)
    rdt_cib_res, dur_cib = float(w_cib @ y), float(w_cib @ D)
    b_act, b_cib = BucketExp @ w_act, BucketExp @ w_cib
    risque_act = float(b_act @ Sigma @ b_act)
    risque_cib = float(b_cib @ Sigma @ b_cib)

    gain_pb = (rdt_cib_res - rdt_act) * 10000
    gain_da = (rdt_cib_res - rdt_act) * BUDGET
    rep_ctp_cib = pd.Series(w_cib, index=df["Contrepartie"]).groupby(level=0).sum()
    rep_type_cib = pd.Series(w_cib, index=df["Type"].str.split().str[0]).groupby(level=0).sum()
    nb_operations = int((np.abs(w_cib - w_act) * BUDGET > 1).sum())
    commissions_fixes_DA = nb_operations * COMMISSION_FIXE_OPERATION
    pl_projete_DA = rdt_cib_res * BUDGET

    seuil_c, seuil_l = data["seuil_court"], data["seuil_long"]
    b_court = float(m_court @ wP_cib) / (data["part_court"] * data["part_placement"]) if data["part_court"] > 0 else 0.0
    b_moyen = float(m_moyen @ wP_cib) / (data["part_moyen"] * data["part_placement"]) if data["part_moyen"] > 0 else 0.0
    b_long = float(m_long @ wP_cib) / (data["part_long"] * data["part_placement"]) if data["part_long"] > 0 else 0.0
    rdt_P = float(wP_cib @ y) / data["part_placement"] if data["part_placement"] > 0 else 0.0
    rdt_T = float(wT_cib @ y) / data["part_trading"] if data["part_trading"] > 0 else 0.0

    df["Poids_actuel"] = w_act
    df["Poids_cible"] = w_cib
    df["Ecart"] = w_cib - w_act
    df["Mouvement_DA"] = np.round(df["Ecart"] * BUDGET, 0)

    CouponMonth = data["CouponMonth"]
    flux_coupon_act = CouponMonth @ w_act * BUDGET
    flux_coupon_cib = CouponMonth @ w_cib * BUDGET

    return {
        "df": df, "mouvements": df[df["Ecart"].abs() > 1e-4].sort_values("Ecart"),
        "rdt_act": rdt_act, "dur_act": dur_act, "rdt_cib": rdt_cib_res, "dur_cib": dur_cib,
        "risque_act": risque_act, "risque_cib": risque_cib,
        "gain_pb": gain_pb, "gain_da": gain_da, "pl_projete_DA": pl_projete_DA,
        "commissions_fixes_DA": commissions_fixes_DA, "nb_operations": nb_operations,
        "b_court": b_court, "b_moyen": b_moyen, "b_long": b_long,
        "rdt_P": rdt_P, "rdt_T": rdt_T, "BUDGET": BUDGET, "mode_utilise": mode_utilise, "coupon_actif": coupon_actif,
        "flux_coupon_act": flux_coupon_act, "flux_coupon_cib": flux_coupon_cib,
        "rep_ctp": data["rep_ctp"], "rep_ctp_cib": rep_ctp_cib, "rep_type": data["rep_type"], "rep_type_cib": rep_type_cib,
        "w_act": w_act, "w_cib": w_cib, "y": y, "D": D, "Conv": Conv,
        "Sigma": Sigma, "BucketExp": BucketExp, "ctps": ctps,
        "seuil_court": seuil_c, "seuil_long": seuil_l,
        "part_placement": data["part_placement"], "part_trading": data["part_trading"],
        "part_court": data["part_court"], "part_moyen": data["part_moyen"], "part_long": data["part_long"],
    }


# ==================================================================
# EXPORT EXCEL
# ==================================================================
def exporter_resultats_excel(d, date_eval):
    """Genere un fichier Excel avec mouvements, position cible et KPIs."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        mov = d["mouvements"].copy()
        mov["Echeance"] = mov["Echeance"].dt.strftime("%d/%m/%Y")
        mov.to_excel(writer, sheet_name="Mouvements", index=False)
        pos = d["df"][["ID", "Type", "Echeance", "Coupon", "Rendement", "Rendement_net",
                        "Duration_mod", "Convexite", "Contrepartie", "Poids_actuel", "Poids_cible",
                        "Mouvement_DA", "Nouveau_titre"]].copy()
        pos["Echeance"] = pos["Echeance"].dt.strftime("%d/%m/%Y")
        pos.to_excel(writer, sheet_name="Position", index=False)
        kpis = pd.DataFrame({
            "Indicateur": ["Rendement actuel", "Rendement cible", "Duration actuelle", "Duration cible",
                           "Risque actuel (bp)", "Risque cible (bp)", "Gain vs actuel (bp)",
                           "Nb operations", "Commissions (DA)", "Budget total (DA)"],
            "Valeur": [f"{d['rdt_act']:.4%}", f"{d['rdt_cib']:.4%}", f"{d['dur_act']:.3f}", f"{d['dur_cib']:.3f}",
                      f"{np.sqrt(max(d['risque_act'],0))*10000:.1f}", f"{np.sqrt(max(d['risque_cib'],0))*10000:.1f}",
                      f"{d['gain_pb']:.1f}", d["nb_operations"], f"{d['commissions_fixes_DA']:,.0f}", f"{d['BUDGET']:,.0f}"],
        })
        kpis.to_excel(writer, sheet_name="KPIs", index=False)
    buf.seek(0)
    return buf


# ==================================================================
# PERSISTANCE TRESORERIE (solde, MM, annexes — survit aux redemarrages)
# ==================================================================
TRESORERIE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tresorerie_data.json")


def charger_tresorerie():
    """Recharge le solde d'ouverture, les operations MM et les annexes depuis le disque."""
    if not os.path.exists(TRESORERIE_FILE):
        return 0.0, [], []
    try:
        with open(TRESORERIE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        mm = data.get("mm_operations", [])
        for op in mm:
            op["Date valeur"] = pd.Timestamp(op["Date valeur"])
            op["Date echeance"] = pd.Timestamp(op["Date echeance"])
        an = data.get("annexe_operations", [])
        for op in an:
            op["Date valeur"] = pd.Timestamp(op["Date valeur"])
        return data.get("solde_ouverture", 0.0), mm, an
    except Exception:
        return 0.0, [], []


def sauvegarder_tresorerie():
    """Ecrit le solde d'ouverture, les operations MM et les annexes sur le disque."""
    data = {
        "solde_ouverture": st.session_state.get("solde_ouverture", 0.0),
        "mm_operations": [
            {**op, "Date valeur": op["Date valeur"].strftime("%Y-%m-%d"),
             "Date echeance": op["Date echeance"].strftime("%Y-%m-%d")}
            for op in st.session_state.get("mm_operations", [])
        ],
        "annexe_operations": [
            {**op, "Date valeur": op["Date valeur"].strftime("%Y-%m-%d")}
            for op in st.session_state.get("annexe_operations", [])
        ],
    }
    with open(TRESORERIE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def prochaine_date_coupon(echeance, date_eval):
    """Prochaine date anniversaire du coupon (meme jour/mois que l'echeance) a partir de date_eval,
    sans jamais depasser l'echeance elle-meme (dernier coupon + capital)."""
    try:
        candidate = pd.Timestamp(year=date_eval.year, month=echeance.month, day=echeance.day)
    except ValueError:
        candidate = pd.Timestamp(year=date_eval.year, month=echeance.month, day=28)
    if candidate < date_eval:
        try:
            candidate = pd.Timestamp(year=date_eval.year + 1, month=echeance.month, day=echeance.day)
        except ValueError:
            candidate = pd.Timestamp(year=date_eval.year + 1, month=echeance.month, day=28)
    return min(candidate, echeance)


# ==================================================================
# IDENTITE VISUELLE
# ==================================================================
st.set_page_config(page_title="AGB Quant Desk v2 — Optimisation QP", page_icon=None, layout="wide")

st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
    --bg:#F4F5F7; --bg-2:#FFFFFF; --panel:#FFFFFF; --line:#E3E6EC;
    --teal:#0E7C6B; --teal-soft:#E3F3EF; --amber:#C08A3E; --amber-soft:#F6ECDD;
    --red:#B23A52; --red-soft:#FBEAEE;
    --text:#101826; --text-muted:#5B6472;
}
html, body, [class*="css"]{ font-family:'Inter', sans-serif; }
.stApp{ background:var(--bg); color:var(--text); }
h1,h2,h3,h4{ font-family:'Space Grotesk', sans-serif !important; color:var(--text) !important; letter-spacing:-0.01em; font-weight:600 !important; }
p, span, div, label{ color:var(--text); }
.mono{ font-family:'JetBrains Mono', monospace; }

.td-hero{
    display:flex; align-items:baseline; justify-content:space-between;
    padding-bottom:18px; margin-bottom:28px; border-bottom:1px solid var(--line);
}
.td-hero .eyebrow{
    font-family:'JetBrains Mono', monospace; font-size:11px; letter-spacing:.16em;
    text-transform:uppercase; color:var(--teal); font-weight:600;
}
.td-hero h1{ font-size:34px; margin:4px 0 0 0; }
.td-hero .sub{ color:var(--text-muted); font-size:14.5px; margin-top:6px; max-width:600px; }
.td-hero .stamp{
    font-family:'JetBrains Mono', monospace; font-size:11px; color:var(--text-muted);
    text-align:right; line-height:1.7; border-left:1px solid var(--line); padding-left:18px;
}

.kpi{
    background:var(--panel); border:1px solid var(--line); border-radius:16px;
    padding:22px 22px 18px 22px; position:relative; overflow:hidden;
    box-shadow:0 1px 3px rgba(16,24,38,.05);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.kpi:hover{ box-shadow:0 4px 12px rgba(16,24,38,.10); transform:translateY(-1px); }
.kpi::before{
    content:""; position:absolute; top:0; left:22px; width:34px; height:3px;
    background:var(--teal); border-radius:0 0 3px 3px;
}
.kpi .label{
    font-family:'JetBrains Mono', monospace; font-size:10.5px; letter-spacing:.14em;
    text-transform:uppercase; color:var(--text-muted); font-weight:600; margin:6px 0 10px 0;
}
.kpi .value{ font-family:'JetBrains Mono', monospace; font-size:28px; font-weight:600; color:var(--text); line-height:1; }
.kpi .delta{ font-family:'JetBrains Mono', monospace; font-size:12.5px; margin-top:8px; font-weight:500; }
.kpi .delta.pos{ color:var(--teal); } .kpi .delta.neg{ color:var(--red); } .kpi .delta.neu{ color:var(--text-muted); }

.gauge-track{ height:6px; background:var(--line); border-radius:99px; margin-top:12px; overflow:hidden; }
.gauge-fill{ height:100%; background:var(--teal); border-radius:99px; transition: width 0.6s ease; }
.gauge-fill.ok{ background:var(--teal); }
.gauge-fill.warn{ background:var(--red); }

.badge{
    display:inline-block; font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:600;
    letter-spacing:.06em; text-transform:uppercase; padding:5px 14px; border-radius:99px;
}
.badge.ok{ background:var(--teal-soft); color:var(--teal); }
.badge.warn{ background:var(--red-soft); color:var(--red); }

.panel{
    background:var(--panel); border:1px solid var(--line); border-radius:18px;
    padding:26px 28px; margin-bottom:22px; box-shadow:0 1px 3px rgba(16,24,38,.05);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
.panel:hover{ box-shadow:0 4px 12px rgba(16,24,38,.08); }
.panel h3{ font-size:19px; margin:0 0 4px 0; }
.panel .panel-sub{ color:var(--text-muted); font-size:13px; margin-bottom:18px; }

div[class*="st-key-panel"]{
    background:var(--panel) !important; border:1px solid var(--line) !important; border-radius:18px !important;
    padding:24px 26px !important; box-shadow:0 1px 3px rgba(16,24,38,.05) !important; margin-bottom:4px;
    transition: box-shadow 0.2s ease !important;
}
div[class*="st-key-panel"]:hover{ box-shadow:0 4px 12px rgba(16,24,38,.08) !important; }
div[class*="st-key-panel_ventes"]{ border-left:4px solid var(--red) !important; }
div[class*="st-key-panel_achats"]{ border-left:4px solid var(--teal) !important; }

/* --- SIDEBAR --- */
[data-testid="stSidebar"]{ background:var(--bg-2); border-right:1px solid var(--line); }
[data-testid="stSidebar"] *{ color:var(--text) !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label{ font-family:'Inter', sans-serif; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3{
    font-family:'Space Grotesk', sans-serif !important;
}
.sidebar-eyebrow{
    font-family:'JetBrains Mono', monospace; font-size:10.5px; letter-spacing:.16em;
    text-transform:uppercase; color:var(--teal); font-weight:600; margin:26px 0 10px 0;
}
[data-testid="stSidebar"] .stNumberInput input{
    background:var(--bg) !important; color:var(--text) !important; border:1px solid var(--line) !important;
}
[data-testid="stSidebar"] .stNumberInput input:disabled,
[data-testid="stSidebar"] .stNumberInput input[disabled]{
    opacity:0.45 !important; cursor:not-allowed !important;
}

/* --- BOUTONS --- */
.stButton>button{
    background:var(--teal); color:#FFFFFF; border-radius:99px; font-weight:600;
    border:none; padding:.5rem 1.3rem; font-family:'Inter',sans-serif; letter-spacing:.01em;
    transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease;
}
.stButton>button:hover{
    background:#0B6558; color:#FFFFFF; box-shadow:0 2px 8px rgba(14,124,107,.25); transform:translateY(-1px);
}
.stButton>button:active{ transform:translateY(0); }
.stButton>button:disabled{ background:var(--line); cursor:not-allowed; box-shadow:none; transform:none; }
[data-testid="stSidebar"] .stButton>button{ background:transparent; color:var(--text) !important; border:1px solid var(--line); }
[data-testid="stSidebar"] .stButton>button:hover{ border-color:var(--teal); color:var(--teal) !important; }

/* --- TABS --- */
[data-testid="stTabs"] [role="tablist"]{ gap:22px; border-bottom:1px solid var(--line); }
[data-testid="stTab"]{
    font-family:'Inter', sans-serif; font-weight:600; font-size:14.5px; color:var(--text-muted) !important;
    padding:10px 2px; background:transparent !important; transition: color 0.15s ease;
}
[data-testid="stTab"] p{ color:inherit !important; }
[data-testid="stTab"]:hover{ color:var(--text) !important; }
[data-testid="stTab"][aria-selected="true"]{ color:var(--teal) !important; }
[data-testid="stTab"] .react-aria-SelectionIndicator{ background:var(--teal) !important; height:2px; transition: background 0.15s ease; }

/* --- FILE UPLOADERS --- */
[data-testid="stFileUploaderDropzone"]{
    background:var(--panel); border:1.5px dashed var(--line); border-radius:16px;
    transition: border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{
    border-color:var(--teal); background:var(--teal-soft); box-shadow:0 2px 8px rgba(14,124,107,.08);
}
[data-testid="stFileUploaderDropzoneInstructions"]{
    font-size:11px !important; color:var(--text-muted) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span{
    font-family:'JetBrains Mono', monospace !important; font-size:10px !important; letter-spacing:.04em;
}

/* --- EMPTY STATE --- */
.td-empty{ text-align:center; padding:60px 40px 50px; }
.td-empty-icon{
    width:80px; height:80px; margin:0 auto 24px; border-radius:20px;
    background:var(--teal-soft); display:flex; align-items:center; justify-content:center;
}
.td-empty-icon svg{
    width:36px; height:36px; stroke:var(--teal); fill:none; stroke-width:1.5;
    stroke-linecap:round; stroke-linejoin:round;
}
.td-empty h3{ font-size:20px; margin:0 0 10px 0; color:var(--text); }
.td-empty p{ color:var(--text-muted); font-size:14px; line-height:1.7; max-width:480px; margin:0 auto 28px; }
.td-empty-features{ display:flex; gap:32px; justify-content:center; flex-wrap:wrap; margin-bottom:32px; }
.td-empty-feat{ text-align:center; max-width:160px; }
.td-empty-feat-icon{
    width:44px; height:44px; margin:0 auto 10px; border-radius:12px;
    background:var(--bg); display:flex; align-items:center; justify-content:center;
}
.td-empty-feat-icon svg{
    width:20px; height:20px; stroke:var(--text-muted); fill:none; stroke-width:1.8;
    stroke-linecap:round; stroke-linejoin:round;
}
.td-empty-feat-label{
    font-family:'JetBrains Mono', monospace; font-size:10.5px; letter-spacing:.1em;
    text-transform:uppercase; color:var(--text-muted); font-weight:600;
}

/* --- MISC --- */
[data-testid="stDataFrame"]{ border-radius:14px; overflow:hidden; border:1px solid var(--line); }
[data-testid="stMetricValue"]{ font-family:'JetBrains Mono', monospace; color:var(--text); }
[data-testid="stMetricLabel"]{ font-family:'JetBrains Mono', monospace; font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--text-muted) !important; }

/* --- RESPONSIVE --- */
@media (max-width: 768px){
    .td-hero{ flex-direction:column; gap:12px; }
    .td-hero .stamp{ border-left:none; padding-left:0; text-align:left; border-top:1px solid var(--line); padding-top:12px; }
    .td-hero h1{ font-size:26px; }
    .td-empty{ padding:40px 20px 30px; }
    .td-empty-features{ flex-direction:column; align-items:center; gap:20px; }
    .kpi .value{ font-size:22px; }
}

.td-file-ok{
    display:inline-flex; align-items:center; gap:6px;
    font-family:'JetBrains Mono', monospace; font-size:11px; color:var(--teal);
    background:var(--teal-soft); padding:4px 12px; border-radius:99px; font-weight:600;
    margin-top:8px;
}
.td-file-ok svg{ width:14px; height:14px; stroke:var(--teal); fill:none; stroke-width:2.5; }
</style>
""")

PLOT_FONT = dict(family="Inter, sans-serif", size=12, color="#101826")
PALETTE = {"BTA": "#6B7280", "OAT": "#B23A52", "SSI": "#0E7C6B", "BTC": "#C08A3E",
            "actuel": "#9AA3B2", "cible": "#0E7C6B", "gold": "#C08A3E"}
PLOTLY_BG = "#FFFFFF"
PLOTLY_CONFIG = {"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def kpi_card(label, value, delta=None, delta_kind="neu", gauge_pct=None, gauge_kind="ok"):
    delta_html = f'<div class="delta {delta_kind}">{delta}</div>' if delta else ""
    gauge_html = ""
    if gauge_pct is not None:
        pct = max(0, min(100, gauge_pct))
        gauge_html = f'<div class="gauge-track"><div class="gauge-fill {gauge_kind}" style="width:{pct}%"></div></div>'
    return f"""<div class="kpi"><div class="label">{label}</div><div class="value mono">{value}</div>{delta_html}{gauge_html}</div>"""


def badge(text, kind):
    return f'<span class="badge {kind}">{text}</span>'


def style_plot(fig):
    fig.update_layout(
        plot_bgcolor=PLOTLY_BG, paper_bgcolor=PLOTLY_BG, font=PLOT_FONT,
        margin=dict(t=16, l=55, r=20, b=48),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E3E6EC', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E3E6EC', zeroline=False)
    return fig


def fmt_mda(v, dec=1):
    """Formate une valeur en M DA ou Md DA selon la grandeur."""
    av = abs(v)
    if av >= 1e9:
        return f"{v/1e9:,.{dec}f} Md"
    if av >= 1e6:
        return f"{v/1e6:,.{dec}f} M"
    return f"{v:,.0f}"


# ==================================================================
# ETAT DE SESSION
# ==================================================================
if 'data' not in st.session_state:
    st.session_state.data = None
if 'historique_ventes' not in st.session_state:
    st.session_state.historique_ventes = None

# ==================================================================
# SIDEBAR — date d'evaluation (toujours visible, avant tout le reste)
# ==================================================================
st.sidebar.markdown('<div class="sidebar-eyebrow">Date d\'evaluation</div>', unsafe_allow_html=True)
DATE_EVAL = st.sidebar.date_input("Date d'evaluation", value=pd.Timestamp.now().normalize(),
                                   min_value=pd.Timestamp("2020-01-01"), max_value=pd.Timestamp("2040-12-31"),
                                   format="DD.MM.YYYY")
DATE_EVAL = pd.Timestamp(DATE_EVAL)

# ==================================================================
# ENTETE
# ==================================================================
st.markdown(f"""
<div class="td-hero">
    <div>
        <div class="eyebrow">AGB · Salle des marches</div>
        <h1>Quant Desk — Optimisation QP</h1>
        <div class="sub">Allocation optimale du portefeuille de titres du Tresor par programmation quadratique convexe.</div>
    </div>
    <div class="stamp">Evaluation au {DATE_EVAL.strftime('%d.%m.%Y')}<br>BTA · OAT · SSI · BTC</div>
</div>
""", unsafe_allow_html=True)

# ==================================================================
# ZONE D'UPLOAD
# (execute avant les parametres de la sidebar pour que 'fichier_charge' reflete
# l'upload de ce meme run, sans decalage d'un cycle)
# ==================================================================
st.markdown('<div class="sidebar-eyebrow" style="margin-top:0; margin-bottom:14px;">Fichiers d\'entree</div>', unsafe_allow_html=True)

fichier = st.file_uploader("Charger le fichier de position (Excel)", type=["xlsx"], label_visibility="visible", key="up_principal")
if fichier is not None:
    st.session_state['_fichier'] = fichier
    # Validation a l'upload
    _df_check, _erreurs = valider_fichier_position(fichier)
    if _erreurs:
        for err in _erreurs:
            if "Colonne manquante" in err:
                st.error(err)
            else:
                st.warning(err)
    else:
        st.markdown(f'<div class="td-file-ok"><svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"></polyline></svg>{fichier.name}</div>', unsafe_allow_html=True)

if st.session_state.get('_fichier') is not None:
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        fichier_marche = st.file_uploader(
            "Titres en circulation (optionnel)", type=["xlsx"], label_visibility="visible", key="up_marche",
            help="Alimente les nouvelles adjudications avec l'offre reelle de marche et le P&L realise.")
        if fichier_marche is not None:
            st.session_state['_fichier_marche'] = fichier_marche
    with opt_col2:
        fichier_courbe = st.file_uploader(
            "Courbe de taux (optionnel)", type=["xlsx"], label_visibility="visible", key="up_courbe",
            help="2 colonnes tenor/taux (ex. '1y', 4.65). Sert a interpoler le rendement par duree.")
        if fichier_courbe is not None:
            st.session_state['_fichier_courbe'] = fichier_courbe

# ==================================================================
# SIDEBAR — parametres (actives une fois un fichier de position charge)
# ==================================================================
st.sidebar.markdown("<hr style='border-color:#E3E6EC; margin:18px 0;'>", unsafe_allow_html=True)

fichier_charge = st.session_state.get('_fichier') is not None
_input_kw = dict(disabled=not fichier_charge)

with st.sidebar.expander("Parametres d'optimisation", expanded=False):
    input_rdt = st.number_input("Rendement net vise (%)", value=5.20, min_value=0.0, max_value=20.0, step=0.05, format="%0.2f", **_input_kw)
    input_dur = st.number_input("Duration maximale (annees)", value=3.50, min_value=0.0, max_value=10.0, step=0.10, format="%0.2f", **_input_kw)
    input_vente = st.number_input("Vente maximale par operation (%)", value=50.0, min_value=0.0, max_value=100.0, step=5.0, format="%0.1f", **_input_kw)
    input_pl_cible = st.number_input("Objectif P&L annuel (Md DA)", value=8.0, min_value=0.0, step=0.5, format="%0.1f", **_input_kw)

with st.sidebar.expander("Gestion ALM", expanded=False):
    input_placement = st.number_input("Part placement (%)", value=DEFAULT_PART_PLACEMENT*100, min_value=50.0, max_value=100.0, step=5.0, format="%0.0f", **_input_kw)
    input_trading = st.number_input("Part trading (%)", value=DEFAULT_PART_TRADING*100, min_value=0.0, max_value=50.0, step=5.0, format="%0.0f", **_input_kw)
    input_seuil_court = st.number_input("Seuil court terme (ans)", value=DEFAULT_SEUIL_COURT_ANS, min_value=0.5, max_value=10.0, step=0.5, format="%0.1f", **_input_kw)
    input_seuil_long = st.number_input("Seuil long terme (ans)", value=DEFAULT_SEUIL_LONG_ANS, min_value=1.0, max_value=30.0, step=1.0, format="%0.0f", **_input_kw)
    st.caption("Repartition cible de la poche placement (doit sommer a 100%) :")
    input_part_court = st.number_input("Part court terme (%)", value=DEFAULT_PART_COURT*100, min_value=0.0, max_value=100.0, step=5.0, format="%0.0f", **_input_kw)
    input_part_moyen = st.number_input("Part moyen terme (%)", value=DEFAULT_PART_MOYEN*100, min_value=0.0, max_value=100.0, step=5.0, format="%0.0f", **_input_kw)
    input_part_long = st.number_input("Part long terme (%)", value=DEFAULT_PART_LONG*100, min_value=0.0, max_value=100.0, step=5.0, format="%0.0f", **_input_kw)
    if fichier_charge and abs(input_part_court + input_part_moyen + input_part_long - 100.0) > 0.5:
        st.warning(f"Somme = {input_part_court + input_part_moyen + input_part_long:.0f}% (normalisee automatiquement a 100%).")

AGENTS = {"BTA": "Sofia Derardja", "OAT": "Sofia Derardja", "SSI": "Sofia Derardja", "BTC": "Sofia Derardja"}

# --- Courbe de taux manuelle dans le sidebar ---
DEFAULT_COURBE = {"1A": 4.65, "2A": 5.05, "3A": 5.15, "5A": 5.25, "7A": 6.60, "10A": 6.80}
TENORS_ANS = {"1A": 1.0, "2A": 2.0, "3A": 3.0, "5A": 5.0, "7A": 7.0, "10A": 10.0}

with st.sidebar.expander("Courbe de taux", expanded=False):
    st.caption("Modifiez les taux directement. Sert a interpoler le rendement par duree.")
    _courbe_kw = dict(step=0.01, format="%0.2f")
    courbe_taux = {}
    _c1, _c2 = st.columns(2)
    with _c1:
        courbe_taux["1A"] = st.number_input("1 an (%)", value=DEFAULT_COURBE["1A"], **_courbe_kw) / 100
        courbe_taux["2A"] = st.number_input("2 ans (%)", value=DEFAULT_COURBE["2A"], **_courbe_kw) / 100
        courbe_taux["3A"] = st.number_input("3 ans (%)", value=DEFAULT_COURBE["3A"], **_courbe_kw) / 100
    with _c2:
        courbe_taux["5A"] = st.number_input("5 ans (%)", value=DEFAULT_COURBE["5A"], **_courbe_kw) / 100
        courbe_taux["7A"] = st.number_input("7 ans (%)", value=DEFAULT_COURBE["7A"], **_courbe_kw) / 100
        courbe_taux["10A"] = st.number_input("10 ans (%)", value=DEFAULT_COURBE["10A"], **_courbe_kw) / 100
    _courbe_tenors = np.array([TENORS_ANS[k] for k in courbe_taux])
    _courbe_valeurs = np.array([courbe_taux[k] for k in courbe_taux])
    courbe_manuelle = (_courbe_tenors, _courbe_valeurs)

with st.sidebar.expander("Modele de risque de taux", expanded=False):
    st.caption("Modele a 3 facteurs de courbe (court/moyen/long), calibrable.")
    _risk_kw = dict(disabled=not fichier_charge)
    vol_court = st.number_input("Volatilite court terme (bp/an)", value=RISK_DEFAULTS["vol_court_bp"], min_value=1.0, max_value=300.0, step=5.0, **_risk_kw)
    vol_moyen = st.number_input("Volatilite moyen terme (bp/an)", value=RISK_DEFAULTS["vol_moyen_bp"], min_value=1.0, max_value=300.0, step=5.0, **_risk_kw)
    vol_long = st.number_input("Volatilite long terme (bp/an)", value=RISK_DEFAULTS["vol_long_bp"], min_value=1.0, max_value=300.0, step=5.0, **_risk_kw)
    rho_cm = st.slider("Correlation court/moyen", 0.0, 1.0, RISK_DEFAULTS["rho_cm"], 0.05, disabled=not fichier_charge)
    rho_ml = st.slider("Correlation moyen/long", 0.0, 1.0, RISK_DEFAULTS["rho_ml"], 0.05, disabled=not fichier_charge)
    rho_cl = st.slider("Correlation court/long", 0.0, 1.0, RISK_DEFAULTS["rho_cl"], 0.05, disabled=not fichier_charge)
    lambda_risk = st.number_input("Aversion au risque (lambda)", value=RISK_DEFAULTS["lambda_risk"], min_value=0.0, max_value=1000.0, step=5.0, **_risk_kw)
    kappa_conv = st.number_input("Prime a la convexite (kappa)", value=RISK_DEFAULTS["kappa_conv"], min_value=0.0, max_value=1.0, step=0.01, format="%0.2f", **_risk_kw)
    kappa_cp = st.number_input("Penalite concentration contrepartie", value=RISK_DEFAULTS["kappa_cp"], min_value=0.0, max_value=1.0, step=0.01, format="%0.2f", **_risk_kw)
    coupon_cible_kda = st.number_input("Coupon cible (base kDA/mois en moyenne)", value=RISK_DEFAULTS["coupon_cible_kda"], min_value=0.0, max_value=500000.0, step=50.0,
                                     help="Contrainte dure : plancher sur le total annuel des coupons percus (= 12 x cette valeur, en kDA), "
                                          "assoupli par l'abattement ci-dessous. Depasser la cible reste toujours autorise. 0 = pas de contrainte.",
                                     **_risk_kw)
    abattement_coupon_pct = st.number_input("Abattement accepte sur le plancher (%)", value=20.0, min_value=0.0, max_value=100.0, step=5.0,
                                     help="Assouplit le plancher annuel, car viser exactement le meme montant chaque mois n'est pas realiste. "
                                          "Ex. 20% avec une cible de 200 kDA/mois -> le plancher annuel tombe a 160 kDA/mois en moyenne (pas de limite haute).",
                                     **_risk_kw)

risk_params = dict(vol_court_bp=vol_court, vol_moyen_bp=vol_moyen, vol_long_bp=vol_long,
                    rho_cm=rho_cm, rho_ml=rho_ml, rho_cl=rho_cl,
                    lambda_risk=lambda_risk, kappa_conv=kappa_conv, kappa_cp=kappa_cp, coupon_cible_kda=coupon_cible_kda,
                    abattement_coupon_pct=abattement_coupon_pct)

# ==================================================================
# SIDEBAR — Tresorerie (solde d'ouverture, marche monetaire, annexes)
# Persiste sur disque (tresorerie_data.json) : rien n'est perdu entre deux sessions.
# ==================================================================
if 'mm_operations' not in st.session_state:
    _solde_disque, _mm_disque, _an_disque = charger_tresorerie()
    st.session_state.mm_operations = _mm_disque
    st.session_state.annexe_operations = _an_disque
    if 'solde_ouverture' not in st.session_state:
        st.session_state['solde_ouverture'] = _solde_disque

with st.sidebar.expander("Tresorerie", expanded=True):
    st.markdown("**Solde d'ouverture**")
    solde_ouverture = st.number_input("Solde d'ouverture (DA)", value=0.0, min_value=0.0, step=1_000_000.0,
                                       format="%0.0f", key="solde_ouverture")

    st.markdown("---")
    st.markdown("**Marche monetaire (MM)**")
    with st.form("form_mm_ajout", clear_on_submit=True):
        mm_date_valeur = st.date_input("Date de valeur", value=DATE_EVAL, key="mm_date_valeur")
        mm_date_echeance = st.date_input("Date d'echeance", value=DATE_EVAL + pd.Timedelta(days=7), key="mm_date_echeance")
        mm_contrepartie = st.text_input("Contrepartie", key="mm_contrepartie")
        mm_taux = st.number_input("Taux (%)", value=3.50, min_value=0.0, max_value=30.0, step=0.05, format="%0.2f", key="mm_taux")
        mm_montant = st.number_input("Montant (DA)", value=0.0, min_value=0.0, step=1_000_000.0, format="%0.0f", key="mm_montant")
        if st.form_submit_button("Ajouter l'operation MM") and mm_montant > 0:
            st.session_state.mm_operations.append({
                "Date valeur": pd.Timestamp(mm_date_valeur), "Date echeance": pd.Timestamp(mm_date_echeance),
                "Contrepartie": mm_contrepartie, "Taux (%)": mm_taux, "Montant (DA)": mm_montant,
            })
            sauvegarder_tresorerie()

    if st.session_state.mm_operations:
        st.caption("Echeancier — tri par date d'echeance, interet simple couru jusqu'a l'echeance")
        df_mm = pd.DataFrame(st.session_state.mm_operations).sort_values("Date echeance").reset_index(drop=True)
        jours = (df_mm["Date echeance"] - df_mm["Date valeur"]).dt.days.clip(lower=0)
        df_mm["Interet (DA)"] = df_mm["Montant (DA)"] * (df_mm["Taux (%)"] / 100) * jours / 365
        df_mm["Capital + interet (DA)"] = df_mm["Montant (DA)"] + df_mm["Interet (DA)"]
        df_mm_aff = df_mm.copy()
        df_mm_aff["Date valeur"] = df_mm_aff["Date valeur"].dt.strftime("%d/%m/%Y")
        df_mm_aff["Date echeance"] = df_mm_aff["Date echeance"].dt.strftime("%d/%m/%Y")
        st.dataframe(df_mm_aff.style.format({"Taux (%)": "{:.2f}", "Montant (DA)": "{:,.0f}",
                                              "Interet (DA)": "{:,.0f}", "Capital + interet (DA)": "{:,.0f}"}),
                     use_container_width=True, height=min(38 * (len(df_mm_aff) + 1) + 3, 260))
        st.caption(f"Capital place : {df_mm['Montant (DA)'].sum():,.0f} DA · "
                   f"Interets projetes : {df_mm['Interet (DA)'].sum():,.0f} DA")
        prochaine = df_mm[df_mm["Date echeance"] >= DATE_EVAL]
        if not prochaine.empty:
            p = prochaine.iloc[0]
            st.caption(f"Prochaine echeance : {p['Date echeance'].strftime('%d/%m/%Y')} — "
                       f"{p['Capital + interet (DA)']:,.0f} DA (dont {p['Interet (DA)']:,.0f} DA d'interet), {p['Contrepartie']}")
        if st.button("Vider le MM", key="btn_vider_mm"):
            st.session_state.mm_operations = []
            sauvegarder_tresorerie()
            st.rerun()

    st.markdown("---")
    st.markdown("**Annexes**")
    with st.form("form_annexe_ajout", clear_on_submit=True):
        an_devise = st.text_input("Devise", value="EUR", key="an_devise").strip().upper()
        an_trader = st.selectbox("Trader", ["BENKADI Amine", "BOUDEFFA Nassima"], key="an_trader")
        an_date = st.date_input("Date de valeur", value=DATE_EVAL, key="an_date")
        an_montant = st.number_input("Montant d'achat (devise)", value=0.0, min_value=0.0, step=10_000.0, format="%0.2f", key="an_montant")
        an_cours = st.number_input("Cours utilise (DZD)", value=0.0, min_value=0.0, step=0.01, format="%0.4f", key="an_cours")
        if st.form_submit_button("Ajouter a l'annexe") and an_montant > 0 and an_cours > 0:
            st.session_state.annexe_operations.append({
                "Sens": "Achat", "Devise": an_devise, "Trader": an_trader,
                "Date valeur": pd.Timestamp(an_date), "Montant": an_montant, "Cours": an_cours,
                "Contrevaleur DZD": an_montant * an_cours,
            })
            sauvegarder_tresorerie()

    if st.session_state.annexe_operations:
        df_an = pd.DataFrame(st.session_state.annexe_operations)
        for col in ("Cours", "Contrevaleur DZD"):
            if col not in df_an.columns:
                df_an[col] = 0.0
            df_an[col] = df_an[col].fillna(0.0)
        for dev in sorted(df_an["Devise"].unique()):
            montant_dev = df_an.loc[df_an["Devise"] == dev, "Montant"].sum()
            st.caption(f"{dev} — Achats : {montant_dev:,.0f}")
        st.caption(f"**Total contrevaleur : {df_an['Contrevaleur DZD'].sum():,.0f} DZD**")
        df_an_aff = df_an.sort_values("Date valeur").copy()
        df_an_aff["Date valeur"] = df_an_aff["Date valeur"].dt.strftime("%d/%m/%Y")
        ligne_total = pd.DataFrame([{"Devise": "TOTAL", "Trader": "", "Date valeur": "", "Sens": "",
                                     "Montant": 0.0, "Cours": 0.0, "Contrevaleur DZD": df_an["Contrevaleur DZD"].sum()}])
        df_an_aff = pd.concat([df_an_aff, ligne_total], ignore_index=True)
        st.dataframe(df_an_aff.style.format({"Montant": "{:,.0f}", "Cours": "{:,.4f}", "Contrevaleur DZD": "{:,.0f}"}),
                     use_container_width=True, height=min(38 * (len(df_an_aff) + 1) + 3, 260))
        if st.button("Vider les annexes", key="btn_vider_annexe"):
            st.session_state.annexe_operations = []
            sauvegarder_tresorerie()
            st.rerun()

    sauvegarder_tresorerie()

st.sidebar.markdown("<hr style='border-color:#E3E6EC; margin:22px 0;'>", unsafe_allow_html=True)

if fichier_charge:
    if st.sidebar.button("Lancer l'optimisation", use_container_width=True, type="primary"):
        st.session_state.data = None
        st.session_state.historique_ventes = None
        fichier = st.session_state['_fichier']
        marche_df, nb_ecartes = None, 0
        courbe = courbe_manuelle  # courbe du sidebar par defaut
        if st.session_state.get('_fichier_marche') is not None:
            try:
                marche_df, nb_ecartes = charger_univers_marche(st.session_state['_fichier_marche'], courbe)
                st.session_state.historique_ventes = charger_historique_ventes(st.session_state['_fichier_marche'])
            except Exception as e:
                st.warning(f"Fichier de titres en circulation non exploitable : {e}")
        # Le fichier courbe Excel surcharge la courbe manuelle si present
        if st.session_state.get('_fichier_courbe') is not None:
            try:
                courbe_fichier = charger_courbe_taux(st.session_state['_fichier_courbe'])
                if courbe_fichier is not None:
                    courbe = courbe_fichier
            except Exception as e:
                st.warning(f"Courbe de taux non exploitable : {e}")
        if marche_df is not None:
            try:
                marche_df, nb_ecartes = charger_univers_marche(st.session_state['_fichier_marche'], courbe)
            except Exception:
                pass
        with st.spinner("Resolution du programme quadratique..."):
            try:
                st.session_state.data = run_optimization(
                    fichier, input_rdt / 100, input_dur, input_vente / 100, risk_params, marche_df, DATE_EVAL,
                    input_seuil_court, input_seuil_long, input_placement / 100, input_trading / 100,
                    input_part_court / 100, input_part_moyen / 100, input_part_long / 100)
                if marche_df is not None:
                    st.session_state.data["nb_titres_marche"] = len(marche_df)
                    st.session_state.data["nb_titres_marche_ecartes"] = nb_ecartes
                    st.session_state.data["courbe_utilisee"] = courbe is not None
            except Exception as e:
                st.error(f"Erreur de resolution : {e}")

    if st.sidebar.button("Reinitialiser", use_container_width=True):
        st.session_state.data = None
        st.session_state.historique_ventes = None
        st.session_state._fichier = None
        st.session_state._fichier_marche = None
        st.session_state._fichier_courbe = None
        # les widgets file_uploader gardent leur propre etat sous leur cle malgre la ligne
        # ci-dessus (qui ne vide que nos variables a nous) : il faut aussi l'effacer, sinon
        # le fichier revient tout seul au rerun suivant.
        for k in ("up_principal", "up_marche", "up_courbe"):
            st.session_state.pop(k, None)
        st.rerun()
else:
    st.sidebar.button("Lancer l'optimisation", use_container_width=True, disabled=True)
    st.sidebar.caption("Chargez un fichier de position pour activer l'optimisation.")

# ==================================================================
# TABLEAU DE BORD (6 tabs)
# ==================================================================
if st.session_state.data is not None:
    d = st.session_state.data
    df = d["df"]
    seuil_c, seuil_l = d["seuil_court"], d["seuil_long"]
    part_P, part_T = d["part_placement"], d["part_trading"]
    maturite_labels = [f'Court (<= {seuil_c} ans)', f'Moyen ({seuil_c}-{seuil_l} ans)', f'Long (> {seuil_l} ans)']

    if d["mode_utilise"] == "maximiser" and not d.get("coupon_actif"):
        st.markdown(badge("Rendement cible inatteignable sous contraintes — meilleur compromis retenu", "warn"), unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Tableau de bord", "Allocation", "Risque", "Execution", "P&L & Coupons", "Contreparties", "Tresorerie"
    ])

    # ======================== 1. TABLEAU DE BORD ========================
    with tab1:
        atteint_rdt = d["rdt_cib"] >= input_rdt / 100 - 1e-6
        rdt_softened = not atteint_rdt and d.get("coupon_actif")
        if atteint_rdt:
            badge_rdt = badge('Objectif de rendement atteint', 'ok')
        elif rdt_softened:
            badge_rdt = badge('Rendement legerement sous la cible (lissage priorise)', 'ok')
        else:
            badge_rdt = badge('Objectif de rendement non atteint', 'warn')
        st.markdown(f'<div style="margin-bottom:18px;">{badge_rdt}</div>', unsafe_allow_html=True)

        # --- 1. Objectifs cibles : est-on sur la trajectoire visee ? ---
        st.markdown("<div class='sidebar-eyebrow' style='margin-bottom:10px;'>Objectifs cibles</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gauge = min(100, d["rdt_cib"] / (input_rdt / 100) * 100)
            st.markdown(kpi_card("Rendement net", f'{d["rdt_cib"]:.2%}', f'Cible : {input_rdt/100:.2%}',
                                 "pos" if (atteint_rdt or rdt_softened) else "neg", gauge, "ok" if (atteint_rdt or rdt_softened) else "warn"), unsafe_allow_html=True)
        with c2:
            risk_bp = np.sqrt(max(d["risque_cib"], 0)) * 10000
            risk_bp_act = np.sqrt(max(d["risque_act"], 0)) * 10000
            delta_risk = risk_bp - risk_bp_act
            st.markdown(kpi_card("Risque de taux (eq. vol.)", f'{risk_bp:.0f} pb', f'{"+" if delta_risk>=0 else ""}{delta_risk:.0f} pb vs actuel',
                                 "neg" if delta_risk > 0 else "pos"), unsafe_allow_html=True)
        with c3:
            gauge_dur = min(100, d["dur_cib"] / input_dur * 100)
            st.markdown(kpi_card("Duration cible", f'{d["dur_cib"]:.2f} ans', f'Limite : {input_dur:.2f} ans',
                                 "neu", gauge_dur, "ok"), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("Gain vs. actuel", f'+{d["gain_pb"]:.0f} pb', f'+{d["gain_da"]/1e6:,.0f} M DA / an', "pos"), unsafe_allow_html=True)

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        # --- 2. P&L & tresorerie : la situation financiere du moment ---
        with st.container(border=True, key="panel_pl_tresorerie_v3"):
            st.markdown("### P&L et tresorerie")
            st.markdown('<div class="panel-sub">Gain projete sur l\'annee et disponibilites au bilan</div>', unsafe_allow_html=True)
            _pl_an_cib = d["rdt_cib"] * d["BUDGET"]
            _pl_cible_da = input_pl_cible * 1e9
            _atteint_pl = _pl_cible_da <= 0 or _pl_an_cib >= _pl_cible_da - 1e-6
            _solde_ouv = st.session_state.get("solde_ouverture", 0.0)
            rdt_brut_port = float(d["w_cib"] @ df["Rendement"].values)
            perte_fiscale = (rdt_brut_port - d["rdt_cib"]) * d["BUDGET"]
            pt1, pt2, pt3, pt4 = st.columns(4)
            pt1.metric("P&L net annuel (cible)", f"{_pl_an_cib/1e9:,.2f} Md DA",
                      f"{'Objectif atteint' if _atteint_pl else 'Objectif : '+format(_pl_cible_da/1e9, ',.1f')+' Md DA'}",
                      delta_color=("off" if _atteint_pl else "inverse"))
            pt2.metric("Solde d'ouverture", f"{_solde_ouv:,.0f} DA")
            pt3.metric("Cout fiscal & commission", f"{perte_fiscale/1e6:,.0f} M DA")
            pt4.metric("Commissions execution", f"{d['commissions_fixes_DA']/1e6:,.1f} M DA", f"{d['nb_operations']} ops", delta_color="off")

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        # --- 3. Repartition : ou est investi le portefeuille cible ---
        with st.container(border=True, key="panel_repartition_v3"):
            st.markdown("### Repartition du portefeuille")
            st.markdown('<div class="panel-sub">Rendement par poche et principale contrepartie</div>', unsafe_allow_html=True)
            _ctp_counts = df["Contrepartie"].value_counts()
            _ctp_top = _ctp_counts.idxmax() if not _ctp_counts.empty else "—"
            _ctp_top_nb = int(_ctp_counts.max()) if not _ctp_counts.empty else 0
            rp1, rp2, rp3 = st.columns(3)
            rp1.metric("Rdt. placement", f"{d['rdt_P']:.2%}")
            rp2.metric("Rdt. trading", f"{d['rdt_T']:.2%}")
            rp3.metric("Contrepartie la plus utilisee", _ctp_top, f"{_ctp_top_nb} operations", delta_color="off")

    # ======================== 2. ALLOCATION ========================
    with tab2:
        st.markdown("## Composition du portefeuille")
        st.markdown('<div style="color:#5B6472; margin-top:-10px; margin-bottom:22px;">Profil de maturites, repartition par type et rendement par tranche</div>', unsafe_allow_html=True)

        # Profil de maturites — graphique
        with st.container(border=True, key="panel_maturites_v3"):
            st.markdown("### Profil de maturites — poche placement")
            st.markdown(f'<div class="panel-sub">Cible de gestion ALM : court (<= {seuil_c} ans) / moyen ({seuil_c}-{seuil_l} ans) / long (> {seuil_l} ans) sur {part_P:.0%} du portefeuille</div>', unsafe_allow_html=True)
            m_c_all = (df["Delai_annees"] <= seuil_c).astype(float).values
            m_l_all = (df["Delai_annees"] > seuil_l).astype(float).values
            m_m_all = 1.0 - m_c_all - m_l_all
            maturite_actu = [float(m_c_all @ df["Poids_actuel"]) * 100,
                            float(m_m_all @ df["Poids_actuel"]) * 100,
                            float(m_l_all @ df["Poids_actuel"]) * 100]
            cible_court_pct = d["part_court"] * part_P * 100
            cible_moyen_pct = d["part_moyen"] * part_P * 100
            cible_long_pct = d["part_long"] * part_P * 100
            maturite_cible_vals = [cible_court_pct, cible_moyen_pct, cible_long_pct]

            fig_mat = go.Figure(data=[
                go.Bar(name='Actuel', x=maturite_labels, y=maturite_actu,
                       marker_color=PALETTE["actuel"], marker_line_width=0, marker_cornerradius=6),
                go.Bar(name='Objectif', x=maturite_labels, y=maturite_cible_vals,
                       marker_color=PALETTE["cible"], marker_line_width=0, marker_cornerradius=6),
            ])
            fig_mat.update_layout(barmode='group', yaxis_title='Poids (%)', xaxis_title=None,
                                  legend_title=None,
                                  margin=dict(t=30, b=10, l=50, r=10))
            st.plotly_chart(style_plot(fig_mat), use_container_width=True, config=PLOTLY_CONFIG)

        # Repartition par type (donuts)
        with st.container(border=True, key="panel_rep_type_v3"):
            st.markdown("### Repartition par type de titre")
            st.markdown('<div class="panel-sub">Composition actuelle vs cible</div>', unsafe_allow_html=True)
            type_colors = {t: PALETTE.get(t, "#9AA3B2") for t in set(list(d["rep_type"].index) + list(d["rep_type_cib"].index))}
            rc1, rc2 = st.columns(2)
            with rc1:
                fig_don_act = px.pie(values=d["rep_type"].values * 100, names=d["rep_type"].index, hole=0.6,
                                      color=d["rep_type"].index, color_discrete_map=type_colors)
                fig_don_act.update_traces(textinfo="label+percent", textfont_size=11, textposition="inside", texttemplate="%{label}<br>%{percent}")
                fig_don_act.update_layout(showlegend=False, annotations=[dict(text="Actuel", x=0.5, y=0.5, font_size=14, font_color="#5B6472", showarrow=False)],
                                          margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_don_act, use_container_width=True, config=PLOTLY_CONFIG)
            with rc2:
                fig_don_cib = px.pie(values=d["rep_type_cib"].values * 100, names=d["rep_type_cib"].index, hole=0.6,
                                      color=d["rep_type_cib"].index, color_discrete_map=type_colors)
                fig_don_cib.update_traces(textinfo="label+percent", textfont_size=11, textposition="inside", texttemplate="%{label}<br>%{percent}")
                fig_don_cib.update_layout(showlegend=False, annotations=[dict(text="Cible", x=0.5, y=0.5, font_size=14, font_color="#5B6472", showarrow=False)],
                                           margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_don_cib, use_container_width=True, config=PLOTLY_CONFIG)

        # Rendement par tranche de maturite
        with st.container(border=True, key="panel_rdt_mat_v2"):
            st.markdown("### Rendement par tranche de maturite")
            st.markdown('<div class="panel-sub">Rendement net moyen pondere, actuel vs cible, par tranche d\'un an</div>', unsafe_allow_html=True)
            largeur_bin = 1.0
            mat_max = float(np.ceil(df["Delai_annees"].max()))
            bornes = np.arange(0, mat_max + largeur_bin, largeur_bin)
            labels_bins = [f"{int(bornes[i])}-{int(bornes[i+1])} ans" for i in range(len(bornes) - 1)]
            bin_idx = np.clip(np.digitize(df["Delai_annees"].values, bornes) - 1, 0, len(labels_bins) - 1)

            def _rdt_bin(poids):
                vals = []
                for b in range(len(labels_bins)):
                    mask = (bin_idx == b)
                    enc = float(poids[mask].sum())
                    vals.append(float((poids[mask] * df["Rendement_net"].values[mask]).sum()) / enc * 100 if enc > 1e-9 else None)
                return vals

            rdt_bin_actu = _rdt_bin(df["Poids_actuel"].values)
            rdt_bin_cib = _rdt_bin(d["w_cib"])
            tranches_ok = [(l, a, c) for l, a, c in zip(labels_bins, rdt_bin_actu, rdt_bin_cib) if a is not None][::-1]
            labels_ok = [l for l, _, _ in tranches_ok]
            actu_ok = [a for _, a, _ in tranches_ok]
            cib_ok = [c for _, _, c in tranches_ok]
            echelle_max = max(max(actu_ok), max(cib_ok)) * 1.15
            fig_rdt_mat = go.Figure()
            fig_rdt_mat.add_trace(go.Scatter(x=labels_ok, y=actu_ok, mode='lines+markers', name='Actuel',
                                            line=dict(color=PALETTE["actuel"], width=2.5),
                                            marker=dict(size=8, color=PALETTE["actuel"])))
            fig_rdt_mat.add_trace(go.Scatter(x=labels_ok, y=cib_ok, mode='lines+markers', name='Cible',
                                            line=dict(color=PALETTE["cible"], width=2.5, dash='dash'),
                                            marker=dict(size=8, color=PALETTE["cible"])))
            fig_rdt_mat.update_layout(xaxis_title="Tranche de maturite", yaxis_title="Rendement net moyen (%)", legend_title=None)
            st.plotly_chart(style_plot(fig_rdt_mat), use_container_width=True, config=PLOTLY_CONFIG)

        # Position detaillee (expander)
        with st.expander("Position detaillee — toutes les lignes", expanded=False):
            pos_aff = df[["ID", "Type", "Echeance", "Coupon", "Rendement", "Rendement_net",
                          "Duration_mod", "Convexite", "Contrepartie", "Poids_actuel", "Poids_cible",
                          "Ecart", "Mouvement_DA", "Nouveau_titre"]].copy()
            pos_aff["Echeance"] = pos_aff["Echeance"].dt.strftime('%d/%m/%Y')
            st.dataframe(
                pos_aff.style.format({"Coupon": "{:.2%}", "Rendement": "{:.2%}", "Rendement_net": "{:.2%}",
                                       "Poids_actuel": "{:.2%}", "Poids_cible": "{:.2%}", "Ecart": "{:.2%}",
                                       "Mouvement_DA": "{:,.0f} DA"}),
                height=450, use_container_width=True)

    # ======================== 3. RISQUE ========================
    with tab3:
        st.markdown("## Analyse de risque")
        st.markdown('<div style="color:#5B6472; margin-top:-10px; margin-bottom:22px;">Exposition au risque par segment de courbe</div>', unsafe_allow_html=True)

        # Exposition par segment
        with st.container(border=True, key="panel_risque_segment_v3"):
            st.markdown("### Exposition au risque par segment de courbe")
            st.markdown('<div class="panel-sub">Exposition en duree (annees x poids) par facteur de risque : actuel vs cible</div>', unsafe_allow_html=True)
            Bexp = d["BucketExp"]
            b_act_disp = Bexp @ d["w_act"]
            b_cib_disp = Bexp @ d["w_cib"]
            fig_risk = go.Figure(data=[
                go.Bar(name='Actuel', x=maturite_labels, y=b_act_disp, marker=dict(color=PALETTE["actuel"], cornerradius=8)),
                go.Bar(name='Cible', x=maturite_labels, y=b_cib_disp, marker=dict(color=PALETTE["cible"], cornerradius=8)),
            ])
            fig_risk.update_layout(barmode='group', yaxis_title="Exposition (annees x poids)", legend_title=None)
            st.plotly_chart(style_plot(fig_risk), use_container_width=True, config=PLOTLY_CONFIG)
            # Detail table
            risk_tbl = pd.DataFrame({
                "Segment": maturite_labels,
                "Actuel": [f"{v:.3f}" for v in b_act_disp],
                "Cible": [f"{v:.3f}" for v in b_cib_disp],
                "Variation": [f"{(c-a):+.3f}" for a, c in zip(b_act_disp, b_cib_disp)],
            })
            st.dataframe(risk_tbl, hide_index=True, use_container_width=True)

    # ======================== 4. EXECUTION ========================
    with tab4:
        st.markdown("## Plan d'execution")
        st.markdown(f'<div style="color:#5B6472; margin-top:-10px; margin-bottom:22px;">Plan d\'execution detaille : quotidien, mensuel et annuel — rythme maximal 1 operation/jour ouvre (5/semaine)</div>', unsafe_allow_html=True)
        st.caption("Colonne 'Taux_limite' : a l'ACHAT, rendement minimum a negocier — le plus eleve entre "
                   "ce qu'exige le rendement cible du portefeuille et le taux de la courbe des taux pour la "
                   "maturite residuelle du titre (on n'achete jamais moins cher que le marche) ; a la VENTE, "
                   "rendement maximum de cession pour ne pas faire baisser le P&L cible (moins-value estimee "
                   "via la duration du titre — case vide si titre proche de l'echeance, sans contrainte utile). "
                   "Colonne 'Taux_courbe_max' (achats uniquement) : plafond indicatif de la zone cible, +12,5% "
                   "(milieu de la fourchette +10 a 15%) au-dessus du taux de la courbe.")

        # Arrondi d'affichage des montants en DA a la dizaine de millions la plus proche (les valeurs
        # exactes restent utilisees pour le budget/QP — seul l'affichage est arrondi).
        def _arrondi_10m(x):
            return round(x / 1e7) * 1e7

        ventes = d["mouvements"][d["mouvements"]["Ecart"] < 0].copy().sort_values("Mouvement_DA")
        achats = d["mouvements"][d["mouvements"]["Ecart"] > 0].copy().sort_values("Mouvement_DA", ascending=False)

        # --- Taux limite d'execution : le rendement a respecter pour ne pas degrader le rendement
        # cible du portefeuille. Le coussin (ecart entre rendement cible actuel et objectif) est
        # reparti au prorata du volume total des operations a executer.
        # - ACHAT : taux plancher = rendement minimum a obtenir (sinon la moyenne ponderee du
        #   portefeuille cible baisse sous l'objectif), releve si besoin au taux de la courbe des
        #   taux pour la maturite residuelle du titre (on n'achete jamais moins cher que le marche
        #   pour cette duree). Le plafond indicatif (Taux_courbe_max) est ce taux de courbe majore
        #   de 10 a 15% (on retient le milieu, 12,5%, comme repere central de la fourchette).
        # - VENTE : taux plafond = rendement maximum de cession (au-dela, la moins-value realisee,
        #   approximee par duration x ecart de taux, entame le coussin de P&L) ; atténué par la
        #   duration du titre (impact prix ~ duration x ecart de taux) — pas de contrainte utile
        #   pour un titre a duration quasi nulle (proche de l'echeance).
        _objectif_rdt = input_rdt / 100
        _coussin_taux = d["rdt_cib"] - _objectif_rdt
        _coussin_DA = _coussin_taux * d["BUDGET"]
        _volume_total_ops = float(pd.concat([ventes["Mouvement_DA"].abs(), achats["Mouvement_DA"].abs()]).sum())
        _ajustement_taux = (_coussin_DA / _volume_total_ops) if _volume_total_ops > 1e-6 else 0.0
        achats["Taux_courbe_min"] = achats["Delai_annees"].apply(lambda a: _rendement_depuis_courbe(a, courbe_manuelle))
        achats["Taux_courbe_max"] = achats["Taux_courbe_min"] * 1.125
        achats["Taux_limite"] = np.maximum(achats["Rendement_net"] - _ajustement_taux, achats["Taux_courbe_min"])
        _duree_min = 0.05
        ventes["Taux_limite"] = ventes["Rendement_net"] + np.where(
            ventes["Duration_mod"] > _duree_min, _ajustement_taux / ventes["Duration_mod"].clip(lower=_duree_min), np.nan)

        nouveaux = achats[achats["Nouveau_titre"] == "Oui"]
        for tbl in (ventes, achats, nouveaux):
            tbl["Echeance"] = tbl["Echeance"].dt.strftime('%d/%m/%Y')

        nb_ventes_total = len(ventes)
        nb_achats_total = len(achats)

        objectif_annuel_DA = input_pl_cible * 1e9
        objectif_mensuel_DA = objectif_annuel_DA / 12
        objectif_journalier_DA = objectif_annuel_DA / 365

        mm_ops_liq = st.session_state.get("mm_operations", [])
        solde_ouv_liq = st.session_state.get("solde_ouverture", 0.0)
        annexe_ops_liq = st.session_state.get("annexe_operations", [])
        fin_annee = pd.Timestamp(year=DATE_EVAL.year, month=12, day=31)

        # --- Budget d'execution : solde d'ouverture, moins MM en cours et annexes, plafonne a 20% ---
        mm_en_cours = [op for op in mm_ops_liq if op["Date echeance"] > DATE_EVAL]
        total_mm_immobilise = sum(op["Montant (DA)"] for op in mm_en_cours)
        total_annexes_dzd = sum(op.get("Contrevaleur DZD", 0.0) for op in annexe_ops_liq)
        tresorerie_disponible_base = max(solde_ouv_liq - total_mm_immobilise - total_annexes_dzd, 0.0)
        budget_max_achats_DA = tresorerie_disponible_base * 0.20

        # --- Ordonnancement au rythme max 1 operation/jour ouvre (un jour ouvre = 5/semaine, donc les deux
        # contraintes sont satisfaites automatiquement en ne planifiant que sur les jours ouvres). Le
        # bilan de chaque semaine (bloc de 5 jours ouvres, aligne avec le regroupement du plan hebdomadaire)
        # vise un equilibre achats/ventes ~50/50 (quota 3/2 qui alterne de cote d'une semaine a l'autre pour
        # rester equilibre sur la duree) ; l'ordre des operations DANS la semaine n'est pas contraint, priorite
        # simplement aux montants les plus importants. Un achat n'est retenu que si le budget d'execution le
        # permet encore ; si un des deux sens manque pour completer la semaine, l'autre comble les jours restants.
        jours_ouvres = pd.bdate_range(start=DATE_EVAL, end=fin_annee, freq=JOUR_OUVRE_CBD)
        jours_ouvres_list = list(jours_ouvres)

        ventes_restantes = []
        for _, row in ventes.iterrows():
            ventes_restantes.append({"Sens": "VENTE", "ID": row["ID"], "Type": row["Type"], "Contrepartie": row["Contrepartie"],
                                    "Echeance": row["Echeance"], "Rendement_net": row["Rendement_net"], "Taux_limite": row["Taux_limite"],
                                    "Montant (DA)": row["Mouvement_DA"], "Agent": AGENTS.get(row["Type"].split()[0], ""),
                                    "_abs": abs(row["Mouvement_DA"])})
        achats_restants = []
        for _, row in achats.iterrows():
            achats_restants.append({"Sens": "ACHAT", "ID": row["ID"], "Type": row["Type"], "Contrepartie": row["Contrepartie"],
                                    "Echeance": row["Echeance"], "Rendement_net": row["Rendement_net"], "Taux_limite": row["Taux_limite"],
                                    "Taux_courbe_max": row["Taux_courbe_max"],
                                    "Montant (DA)": row["Mouvement_DA"], "Agent": AGENTS.get(row["Type"].split()[0], ""),
                                    "_abs": abs(row["Mouvement_DA"])})
        ventes_restantes.sort(key=lambda o: o["_abs"], reverse=True)
        achats_restants.sort(key=lambda o: o["_abs"], reverse=True)

        operations_planifiees = []
        cumulative_achats_planifies = 0.0

        def _pop_prochain_achat_finance():
            for i, op in enumerate(achats_restants):
                if cumulative_achats_planifies + op["Montant (DA)"] <= budget_max_achats_DA:
                    return achats_restants.pop(i)
            return None

        idx_jour = 0
        semaine_num = 0
        while idx_jour < len(jours_ouvres_list) and (ventes_restantes or achats_restants):
            jours_semaine = jours_ouvres_list[idx_jour: idx_jour + 5]
            n_jours = len(jours_semaine)
            quota_achat = (n_jours + 1) // 2 if semaine_num % 2 == 0 else n_jours // 2
            quota_vente = n_jours - quota_achat

            ops_semaine = []
            for _ in range(quota_vente):
                if ventes_restantes:
                    ops_semaine.append(ventes_restantes.pop(0))
            for _ in range(quota_achat):
                op = _pop_prochain_achat_finance()
                if op is not None:
                    ops_semaine.append(op)
            # comble les jours restants avec l'autre sens si l'un des deux n'a pas suffi au quota
            while len(ops_semaine) < n_jours and (ventes_restantes or achats_restants):
                op = ventes_restantes.pop(0) if ventes_restantes else _pop_prochain_achat_finance()
                if op is None:
                    break
                ops_semaine.append(op)

            ops_semaine.sort(key=lambda o: o["_abs"], reverse=True)
            for jour, op in zip(jours_semaine, ops_semaine):
                if op["Sens"] == "ACHAT":
                    cumulative_achats_planifies += op["Montant (DA)"]
                operations_planifiees.append({"Date": jour, **op})

            idx_jour += 5
            semaine_num += 1

        operations_non_planifiees = ventes_restantes + achats_restants
        nb_achats_non_planifies = sum(1 for o in operations_non_planifiees if o["Sens"] == "ACHAT")
        montant_achats_non_planifies = sum(o["Montant (DA)"] for o in operations_non_planifiees if o["Sens"] == "ACHAT")
        nb_ventes_non_planifiees = sum(1 for o in operations_non_planifiees if o["Sens"] == "VENTE")

        # --- PLAN QUOTIDIEN ---
        with st.container(border=True, key="plan_quotidien"):
            st.markdown("### Plan du jour")
            st.markdown(f'<div class="panel-sub">Une seule operation aujourd\'hui ({DATE_EVAL.strftime("%d/%m/%Y")}), priorite au plus gros montant dans le budget disponible</div>', unsafe_allow_html=True)

            op_du_jour = next((o for o in operations_planifiees if o["Date"] == DATE_EVAL), None)
            if op_du_jour is not None:
                pq1, pq2 = st.columns(2)
                pq1.metric(f"{op_du_jour['Sens']} du jour", f"{abs(op_du_jour['Montant (DA)'])/1e6:,.0f} M DA")
                pq2.metric("Objectif journalier (P&L)", f"{objectif_journalier_DA/1e6:,.0f} M DA")
                ligne_jour = {k: v for k, v in op_du_jour.items() if k not in ("_abs", "Date")}
                ligne_jour["Montant (DA)"] = _arrondi_10m(ligne_jour["Montant (DA)"])
                st.dataframe(pd.DataFrame([ligne_jour]).style.format({"Rendement_net": "{:.2%}", "Taux_limite": "{:.2%}",
                                                                       "Taux_courbe_max": "{:.2%}", "Montant (DA)": "{:,.0f} DA"}, na_rep=""),
                            use_container_width=True, height=76)
                _lbl_taux = "plancher a l'achat" if op_du_jour["Sens"] == "ACHAT" else "plafond a la vente"
                _val_taux = ligne_jour.get("Taux_limite")
                if _val_taux is not None and not (isinstance(_val_taux, float) and np.isnan(_val_taux)):
                    st.caption(f"Taux {_lbl_taux} pour tenir le rendement/P&L cible : {_val_taux:.2%}")
                _val_courbe_max = ligne_jour.get("Taux_courbe_max")
                if op_du_jour["Sens"] == "ACHAT" and _val_courbe_max is not None and not (isinstance(_val_courbe_max, float) and np.isnan(_val_courbe_max)):
                    st.caption(f"Zone cible vs courbe des taux (maturite residuelle) : {_val_taux:.2%} a {_val_courbe_max:.2%} (+10 a 15%)")
            else:
                if DATE_EVAL.dayofweek in (4, 5):
                    st.markdown(badge("Jour non ouvre — aucune operation programmee", "ok"), unsafe_allow_html=True)
                else:
                    st.markdown(badge("Aucune operation programmee aujourd'hui", "ok"), unsafe_allow_html=True)
                st.caption(f"Objectif journalier (P&L) : {objectif_journalier_DA/1e6:,.0f} M DA")
            st.caption(f"Budget d'execution disponible pour les achats : {budget_max_achats_DA:,.0f} DA "
                       f"(20% de {tresorerie_disponible_base:,.0f} DA = solde d'ouverture - MM en cours - annexes)")

        # --- PLAN HEBDOMADAIRE (detail d'une semaine choisie : titres, MM, coupons, annexes) ---
        with st.container(border=True, key="plan_hebdomadaire"):
            st.markdown("### Plan hebdomadaire")
            st.markdown('<div class="panel-sub">Choisissez une semaine pour en voir le detail complet</div>', unsafe_allow_html=True)

            # Semaines = blocs de 5 jours ouvres consecutifs a partir d'aujourd'hui (pas des semaines
            # calendaires lundi-dimanche, pour garantir 5 operations par semaine comme demande, sans
            # semaine partielle en tete de liste). La fin de chaque bloc couvre aussi le week-end qui
            # suit, pour que les echeances MM/coupons tombant un samedi/dimanche restent rattachees a
            # la bonne semaine (pas de trou de couverture calendaire).
            jours_ouvres_list = list(jours_ouvres)
            debuts_bucket = jours_ouvres_list[0::5]
            fins_bucket = {}
            for i, debut in enumerate(debuts_bucket):
                fins_bucket[debut] = (debuts_bucket[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(debuts_bucket) else fin_annee
            bornes_par_debut = fins_bucket

            def _bucket_semaine(date):
                idx = 0
                for i, debut in enumerate(debuts_bucket):
                    if date >= debut:
                        idx = i
                    else:
                        break
                return debuts_bucket[idx]

            semaines_detail = {}

            def _semaine_detail(date):
                debut = _bucket_semaine(date)
                return semaines_detail.setdefault(debut, {"titres": [], "mm": [], "coupons": [], "annexes": []})

            for op_s in operations_planifiees:
                _semaine_detail(op_s["Date"])["titres"].append({
                    "Date": op_s["Date"], "Sens": op_s["Sens"], "ID": op_s["ID"], "Type": op_s["Type"],
                    "Contrepartie": op_s["Contrepartie"], "Montant (DA)": op_s["Montant (DA)"],
                    "Taux_limite": op_s["Taux_limite"], "Taux_courbe_max": op_s.get("Taux_courbe_max"),
                    "Agent": op_s["Agent"]})

            for opmm in mm_ops_liq:
                if DATE_EVAL <= opmm["Date valeur"] <= fin_annee:
                    _semaine_detail(opmm["Date valeur"])["mm"].append({
                        "Date": opmm["Date valeur"], "Mouvement": "Sortie (placement)",
                        "Contrepartie": opmm["Contrepartie"], "Montant (DA)": opmm["Montant (DA)"]})
                if DATE_EVAL <= opmm["Date echeance"] <= fin_annee:
                    jours_mm = max((opmm["Date echeance"] - opmm["Date valeur"]).days, 0)
                    montant_entree = opmm["Montant (DA)"] * (1 + (opmm["Taux (%)"] / 100) * jours_mm / 365)
                    _semaine_detail(opmm["Date echeance"])["mm"].append({
                        "Date": opmm["Date echeance"], "Mouvement": "Entree (echeance)",
                        "Contrepartie": opmm["Contrepartie"], "Montant (DA)": montant_entree})

            for opan in annexe_ops_liq:
                if DATE_EVAL <= opan["Date valeur"] <= fin_annee:
                    _semaine_detail(opan["Date valeur"])["annexes"].append({
                        "Date": opan["Date valeur"], "Devise": opan["Devise"], "Trader": opan["Trader"],
                        "Montant": opan["Montant"], "Contrevaleur DZD": opan.get("Contrevaleur DZD", 0.0)})

            titres_detenus_s = df[df["Poids_actuel"] > 0]
            for _, row in titres_detenus_s.iterrows():
                prochaine_cpn = prochaine_date_coupon(row["Echeance"], DATE_EVAL)
                if DATE_EVAL <= prochaine_cpn <= fin_annee:
                    _semaine_detail(prochaine_cpn)["coupons"].append({
                        "Date": prochaine_cpn, "Type": row["Type"], "Contrepartie": row["Contrepartie"],
                        "Montant (DA)": row["VN"] * row["Coupon"]})

            if not semaines_detail:
                st.caption("Aucun flux a venir sur l'horizon.")
            else:
                debuts_tries = sorted(semaines_detail.keys())
                labels_semaine = {deb: f"{deb.strftime('%d/%m/%Y')} au {bornes_par_debut[deb].strftime('%d/%m/%Y')}"
                                  for deb in debuts_tries}
                lundi_choisi = st.selectbox("Semaine", debuts_tries, index=0,
                                            format_func=lambda l: labels_semaine[l], key="semaine_hebdo_choisie")

                detail = semaines_detail[lundi_choisi]
                total_achats = sum(o["Montant (DA)"] for o in detail["titres"] if o["Sens"] == "ACHAT")
                total_ventes = sum(o["Montant (DA)"] for o in detail["titres"] if o["Sens"] == "VENTE")
                total_mm_sortie = sum(o["Montant (DA)"] for o in detail["mm"] if o["Mouvement"].startswith("Sortie"))
                total_mm_entree = sum(o["Montant (DA)"] for o in detail["mm"] if o["Mouvement"].startswith("Entree"))
                total_coupons = sum(o["Montant (DA)"] for o in detail["coupons"])
                total_annexes = sum(o["Contrevaleur DZD"] for o in detail["annexes"])
                flux_net = total_ventes + total_mm_entree + total_coupons - total_achats - total_mm_sortie - total_annexes

                ps1, ps2, ps3, ps4 = st.columns(4)
                ps1.metric("Achats / Ventes", f"{total_achats/1e6:,.0f} / {total_ventes/1e6:,.0f} M DA")
                ps2.metric("MM sorties / entrees", f"{total_mm_sortie/1e6:,.0f} / {total_mm_entree/1e6:,.0f} M DA")
                ps3.metric("Coupons / Annexes", f"{total_coupons/1e6:,.0f} M DA / {total_annexes/1e6:,.0f} M DZD")
                ps4.metric("Flux net de la semaine", f"{flux_net/1e6:,.0f} M DA")

                if detail["titres"]:
                    st.markdown("**Operations titres**")
                    df_t = pd.DataFrame(detail["titres"]).sort_values("Date")
                    df_t["Date"] = df_t["Date"].dt.strftime("%d/%m/%Y")
                    df_t["Montant (DA)"] = df_t["Montant (DA)"].apply(_arrondi_10m)
                    st.dataframe(df_t.style.format({"Montant (DA)": "{:,.0f} DA", "Taux_limite": "{:.2%}", "Taux_courbe_max": "{:.2%}"}, na_rep=""), use_container_width=True,
                                hide_index=True, height=min(38 * (len(df_t) + 1) + 3, 300))

                if detail["mm"]:
                    st.markdown("**Marche monetaire**")
                    df_m = pd.DataFrame(detail["mm"]).sort_values("Date")
                    df_m["Date"] = df_m["Date"].dt.strftime("%d/%m/%Y")
                    st.dataframe(df_m.style.format({"Montant (DA)": "{:,.0f} DA"}), use_container_width=True,
                                hide_index=True, height=min(38 * (len(df_m) + 1) + 3, 220))

                if detail["coupons"]:
                    st.markdown("**Coupons**")
                    df_c = pd.DataFrame(detail["coupons"]).sort_values("Date")
                    df_c["Date"] = df_c["Date"].dt.strftime("%d/%m/%Y")
                    st.dataframe(df_c.style.format({"Montant (DA)": "{:,.0f} DA"}), use_container_width=True,
                                hide_index=True, height=min(38 * (len(df_c) + 1) + 3, 220))

                if detail["annexes"]:
                    st.markdown("**Annexes**")
                    df_a = pd.DataFrame(detail["annexes"]).sort_values("Date")
                    df_a["Date"] = df_a["Date"].dt.strftime("%d/%m/%Y")
                    st.dataframe(df_a.style.format({"Montant": "{:,.0f}", "Contrevaleur DZD": "{:,.0f}"}), use_container_width=True,
                                hide_index=True, height=min(38 * (len(df_a) + 1) + 3, 220))

                if not (detail["titres"] or detail["mm"] or detail["coupons"] or detail["annexes"]):
                    st.caption("Aucun flux pour cette semaine.")

        # --- PLAN MENSUEL ---
        with st.container(border=True, key="plan_mensuel"):
            st.markdown("### Plan mensuel")
            st.markdown(f'<div class="panel-sub">Operations planifiees jusqu\'au {fin_annee.strftime("%d/%m/%Y")}, au rythme de 1 par jour ouvre (5/semaine max)</div>', unsafe_allow_html=True)

            if operations_planifiees:
                df_plan = pd.DataFrame(operations_planifiees)
                df_plan["Mois"] = df_plan["Date"].dt.strftime("%b-%Y")
                nb_mois_couverts = df_plan["Mois"].nunique()
                pm1, pm2, pm3, pm4 = st.columns(4)
                pm1.metric("Operations planifiees", f"{len(df_plan)}")
                pm2.metric("Volume programme", f"{df_plan['Montant (DA)'].abs().sum()/1e9:,.0f} Md DA")
                pm3.metric("Etalement", f"{nb_mois_couverts} mois")
                pm4.metric("Objectif mensuel (P&L)", f"{objectif_mensuel_DA/1e6:,.0f} M DA")

                df_plan_aff = df_plan.copy()
                df_plan_aff["Date"] = df_plan_aff["Date"].dt.strftime("%d/%m/%Y")
                df_plan_aff["Montant (DA)"] = df_plan_aff["Montant (DA)"].apply(_arrondi_10m)
                st.dataframe(df_plan_aff[["Date", "Mois", "Sens", "ID", "Type", "Contrepartie", "Montant (DA)", "Taux_limite", "Taux_courbe_max", "Agent"]]
                             .style.format({"Montant (DA)": "{:,.0f} DA", "Taux_limite": "{:.2%}", "Taux_courbe_max": "{:.2%}"}, na_rep=""),
                            use_container_width=True, height=min(38 * (len(df_plan_aff) + 1) + 3, 500))
            else:
                st.caption("Aucune operation planifiable.")

            if operations_non_planifiees:
                details = []
                if nb_achats_non_planifies:
                    details.append(f"{nb_achats_non_planifies} achat(s) — budget d'execution insuffisant "
                                   f"({montant_achats_non_planifies/1e9:,.0f} Md DA en attente)")
                if nb_ventes_non_planifiees:
                    details.append(f"{nb_ventes_non_planifiees} vente(s) — horizon trop court a ce rythme")
                st.markdown(badge(f"{len(operations_non_planifiees)} operation(s) non planifiable(s) d'ici le "
                                  f"{fin_annee.strftime('%d/%m/%Y')} : {' ; '.join(details)}", "warn"), unsafe_allow_html=True)

        # --- PLAN ANNUEL ---
        with st.container(border=True, key="plan_annuel"):
            st.markdown("### Plan annuel")
            st.markdown(f'<div class="panel-sub">Vue synthetique de l\'annee {DATE_EVAL.year}</div>', unsafe_allow_html=True)

            df_expiring = d["df"][(d["df"]["Echeance"] <= fin_annee) & (d["df"]["Poids_actuel"] > 0)].copy()

            pa1, pa2, pa3, pa4, pa5 = st.columns(5)
            pa1.metric("Titres echeants", f"{len(df_expiring)}", f"VN: {df_expiring['VN'].sum()/1e9:,.0f} Md DA", delta_color="off")
            pa2.metric("Volume achats total", f"{achats['Mouvement_DA'].sum()/1e9:,.0f} Md DA", f"{nb_achats_total} ops", delta_color="off")
            pa3.metric("Volume ventes total", f"{abs(ventes['Mouvement_DA'].sum())/1e9:,.0f} Md DA", f"{nb_ventes_total} ops", delta_color="off")
            pa4.metric("Cout execution", f"{d['commissions_fixes_DA']/1e6:,.0f} M DA", f"{d['nb_operations']} ops", delta_color="off")
            pa5.metric("Objectif annuel (P&L)", f"{objectif_annuel_DA/1e9:,.0f} Md DA")

            # Resume par type
            resume_rows = []
            for sens_label, sens_df in [("ACHAT", achats), ("VENTE", ventes)]:
                for type_titre in sens_df["Type"].str.split().str[0].unique():
                    sub = sens_df[sens_df["Type"].str.split().str[0] == type_titre]
                    resume_rows.append({
                        "Sens": sens_label, "Type": type_titre,
                        "Nb operations": len(sub),
                        "Volume (Md DA)": f"{abs(sub['Mouvement_DA'].sum())/1e9:,.0f}",
                        "Agent": AGENTS.get(type_titre, "")
                    })
            if resume_rows:
                st.dataframe(pd.DataFrame(resume_rows), hide_index=True, use_container_width=True)

            # Echeances dans l'annee
            if not df_expiring.empty:
                with st.expander("Titres echeants dans l'annee", expanded=False):
                    exp_aff = df_expiring[["ID", "Type", "Echeance", "VN", "Rendement_net", "Contrepartie"]].copy()
                    exp_aff["Echeance"] = exp_aff["Echeance"].dt.strftime('%d/%m/%Y')
                    exp_aff["VN"] = exp_aff["VN"].apply(_arrondi_10m)
                    st.dataframe(exp_aff.style.format({"VN": "{:,.0f} DA", "Rendement_net": "{:.2%}"}),
                                use_container_width=True, height=300)

        # Export
        try:
            excel_buf = exporter_resultats_excel(d, DATE_EVAL)
            st.download_button(
                label="Exporter les resultats (Excel)", data=excel_buf,
                file_name=f"optimisation_{DATE_EVAL.strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, type="primary")
        except Exception:
            st.caption("Export Excel indisponible (openpyxl non installe).")

        # --- TITRES EN COURS ---
        with st.container(border=True, key="panel_titres_en_cours"):
            st.markdown("### Titres en cours")
            st.markdown("<div class=\"panel-sub\">Ensemble des titres actuellement en portefeuille</div>", unsafe_allow_html=True)
            encours_df = d["df"][d["df"]["Nouveau_titre"] == "Non"].copy()
            if encours_df.empty:
                st.caption("Aucun titre en cours.")
            else:
                encours_aff = encours_df[["ID", "Type", "Echeance", "Coupon", "Rendement", "Rendement_net",
                                         "Duration_mod", "Contrepartie", "VN", "Poids_actuel", "Poids_cible", "Mouvement_DA"]].copy()
                encours_aff["Echeance"] = encours_aff["Echeance"].dt.strftime("%d/%m/%Y")
                encours_aff["VN"] = encours_aff["VN"].apply(_arrondi_10m)
                encours_aff["Mouvement_DA"] = encours_aff["Mouvement_DA"].apply(_arrondi_10m)
                st.dataframe(
                    encours_aff.style.format({"Coupon": "{:.2%}", "Rendement": "{:.2%}", "Rendement_net": "{:.2%}",
                                               "Poids_actuel": "{:.2%}", "Poids_cible": "{:.2%}",
                                               "Mouvement_DA": "{:,.0f} DA", "VN": "{:,.0f} DA"}),
                    height=500, use_container_width=True)

    # ======================== 5. P&L & COUPONS ========================
    with tab5:
        st.markdown("## P&L de portage & Coupons")
        st.markdown('<div style="color:#5B6472; margin-top:-10px; margin-bottom:22px;">Projection du portage, calendrier des coupons et P&L realise</div>', unsafe_allow_html=True)
        st.caption("Calcule a partir du rendement net et de l'encours (logique de portage/carry) — non disponible pour un vrai P&L mark-to-market.")

        pl_an_cib = d["rdt_cib"] * d["BUDGET"]
        pl_an_act = d["rdt_act"] * d["BUDGET"]
        pl_mois_cib = pl_an_cib / 12.0
        pl_semaine_cib = pl_an_cib / 52.0
        pl_jour_cib = pl_an_cib / 365.0
        pl_cible_da = input_pl_cible * 1e9

        fin_annee = pd.Timestamp(year=DATE_EVAL.year, month=12, day=31)
        jours_restants = (fin_annee - DATE_EVAL).days
        mois_restants = jours_restants / 30.44
        pl_restant_cib = pl_an_cib * jours_restants / 365.0

        atteint_pl = pl_cible_da <= 0 or pl_restant_cib >= pl_cible_da - 1e-6
        gauge_pl = min(100, pl_restant_cib / pl_cible_da * 100) if pl_cible_da > 0 else 100

        # Objectif P&L + Brut/Net dans un seul panneau
        with st.container(border=True, key="panel_pl_objectif_brut_net"):
            st.markdown("### Synthese P&L")
            st.markdown(f'<div class="panel-sub">Objectif annuel et rendement du portefeuille cible — brut et net</div>', unsafe_allow_html=True)
            # Ligne 1 : objectif
            st.markdown(kpi_card(f"P&L atteignable (~{mois_restants:.1f} mois restants)",
                                 f"{pl_restant_cib/1e9:,.2f} Md DA",
                                 f"Objectif : {pl_cible_da/1e9:,.1f} Md DA",
                                 "pos" if atteint_pl else "neg", gauge_pl,
                                 "ok" if atteint_pl else "warn"), unsafe_allow_html=True)
            # Ligne 2 : P&L brut / net / impot / commissions
            rdt_brut_val = float(d["w_cib"] @ df["Rendement"].values)
            pl_brut_annuel = rdt_brut_val * d["BUDGET"]
            pl_net_annuel = d["rdt_cib"] * d["BUDGET"]
            is_OAT_all = df["Type"].str.startswith("OAT").values
            is_exo = (df["Exonere"] == "Oui").values
            impot_annuel = float(d["w_cib"][~is_exo] @ df["Rendement"].values[~is_exo]) * TAUX_IMPOT * d["BUDGET"]
            comm_oat_annuel = float(d["w_cib"][is_OAT_all].sum()) * COMMISSION_OAT_ANNUELLE * d["BUDGET"]
            pn1, pn2, pn3, pn4 = st.columns(4)
            pn1.metric("P&L brut annuel", f"{pl_brut_annuel/1e9:,.2f} Md DA", f"{rdt_brut_val:.2%}")
            pn2.metric("P&L net annuel", f"{pl_net_annuel/1e9:,.2f} Md DA", f"{d['rdt_cib']:.2%}")
            pn3.metric("Impot sur plus-values", f"-{impot_annuel/1e6:,.1f} M DA")
            pn4.metric("Commissions OAT", f"-{comm_oat_annuel/1e6:,.1f} M DA")
            st.caption("Contribution de ce portefeuille uniquement — hors P&L deja realise.")
            if not atteint_pl:
                st.markdown(f'{badge("Objectif hors de portee", "warn")}'
                            f'<div style="margin-top:4px; color:#5B6472; font-size:13px;">Ecart : <b>{(pl_cible_da-pl_restant_cib)/1e9:,.2f} Md DA</b></div>',
                            unsafe_allow_html=True)

        # Portage par horizon
        with st.container(border=True, key="panel_pl_horizons_v2"):
            st.markdown("### Portage par horizon")
            st.markdown('<div class="panel-sub">Rythme de portage du portefeuille cible (run-rate)</div>', unsafe_allow_html=True)
            ph1, ph2, ph3, ph4 = st.columns(4)
            ph1.metric("Journalier", f"{pl_jour_cib/1e6:,.2f} M DA")
            ph2.metric("Hebdomadaire", f"{pl_semaine_cib/1e6:,.1f} M DA")
            ph3.metric("Mensuel", f"{pl_mois_cib/1e6:,.0f} M DA")
            ph4.metric("Annuel", f"{pl_an_cib/1e9:,.2f} Md DA", f"{(pl_an_cib-pl_an_act)/1e6:+,.0f} M DA vs actuel")

        # Heatmap calendrier des coupons
        with st.container(border=True, key="panel_coupons_heatmap"):
            st.markdown("### Calendrier des coupons")
            st.markdown('<div class="panel-sub">Flux de coupons mensuels, actuel vs cible — couleur = intensite relative</div>', unsafe_allow_html=True)
            mois_labels = ["Jan", "Fev", "Mar", "Avr", "Mai", "Juin", "Juil", "Aout", "Sep", "Oct", "Nov", "Dec"]
            flux_act_M = d["flux_coupon_act"] / 1e6
            flux_cib_M = d["flux_coupon_cib"] / 1e6
            z_data = np.array([flux_act_M, flux_cib_M])
            fig_hm = go.Figure(go.Heatmap(
                z=z_data, x=mois_labels, y=["Actuel", "Cible"],
                colorscale=[[0, "#E3F3EF"], [0.5, "#FFFFFF"], [1, "#FBEAEE"]],
                zmid=np.nanmean(z_data),
                text=[[f"{v:,.0f}" for v in row] for row in z_data],
                texttemplate="%{text}", textfont=dict(size=11),
                hovertemplate="%{y} · %{x}<br>%{z:,.0f} M DA<extra></extra>",
            ))
            fig_hm.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(t=10, l=55, r=20, b=48))
            fig_hm.update_xaxes(showgrid=False)
            fig_hm.update_yaxes(showgrid=False)
            fig_hm.update_layout(plot_bgcolor=PLOTLY_BG, paper_bgcolor=PLOTLY_BG, font=PLOT_FONT)
            st.plotly_chart(fig_hm, use_container_width=True, config=PLOTLY_CONFIG)
            ecart_act = float(d["flux_coupon_act"].max() - d["flux_coupon_act"].min())
            ecart_cib = float(d["flux_coupon_cib"].max() - d["flux_coupon_cib"].min())
            st.caption(f"Ecart max-min : {ecart_act/1e6:,.0f} M DA (actuel) → {ecart_cib/1e6:,.0f} M DA (cible)."
                       " Reglable via 'Lissage des coupons' dans les hypotheses.")

        # Cumul P&L 5 ans (courbe exponentielle reelle)
        with st.container(border=True, key="panel_pl_comparaison_v2"):
            st.markdown("### Portage cumule sur 5 ans")
            st.markdown('<div class="panel-sub">Accumulation du P&L avec capitalisation reelle (composes annuels)</div>', unsafe_allow_html=True)
            horizon_ans = 5
            annees = np.linspace(0, horizon_ans, horizon_ans * 12 + 1)
            cumul_act = [d["BUDGET"] * ((1 + d["rdt_act"]) ** t - 1) / 1e9 for t in annees]
            cumul_cib = [d["BUDGET"] * ((1 + d["rdt_cib"]) ** t - 1) / 1e9 for t in annees]
            fig_pl = go.Figure(data=[
                go.Scatter(name='Actuel', x=annees, y=cumul_act, mode='lines',
                           line=dict(color=PALETTE["actuel"], width=3),
                           fill='tozeroy', fillcolor='rgba(154,163,178,0.12)'),
                go.Scatter(name='Cible', x=annees, y=cumul_cib, mode='lines',
                           line=dict(color=PALETTE["cible"], width=3),
                           fill='tozeroy', fillcolor='rgba(14,124,107,0.12)'),
            ])
            fig_pl.update_layout(yaxis_title="P&L cumule (Md DA)", xaxis_title="Annees", legend_title=None)
            st.plotly_chart(style_plot(fig_pl), use_container_width=True, config=PLOTLY_CONFIG)

        # P&L realise - depuis le debut de l'annee
        with st.container(border=True, key="panel_pl_realise_v3"):
            st.markdown(f"### P&L realise depuis le 01/01/{DATE_EVAL.year}")
            hv = st.session_state.historique_ventes
            if hv is None or hv.empty:
                st.markdown("<div class=\"panel-sub\">Chargez le fichier des titres en circulation pour afficher le P&L realise sur les cessions anterieures.</div>", unsafe_allow_html=True)
            else:
                # Filtrer depuis le debut de l'annee
                debut_annee = pd.Timestamp(year=DATE_EVAL.year, month=1, day=1)
                hv_ytd = hv.dropna(subset=["Date_vente"])
                hv_ytd = hv_ytd[hv_ytd["Date_vente"] >= debut_annee].copy()
                if hv_ytd.empty:
                    st.markdown(f"<div class=\"panel-sub\">Aucune cession realisee depuis le 01/01/{DATE_EVAL.year}.</div>", unsafe_allow_html=True)
                else:
                    pv_brute_ytd = hv_ytd["Plus_value"].sum()
                    pv_nette_ytd = hv_ytd["Plus_value_nette"].sum()
                    nb_ventes_ytd = len(hv_ytd)
                    vol_cede_ytd = hv_ytd["Valeur_nominale"].sum()
                    st.markdown(f'<div class="panel-sub">Cessions realisees entre le 01/01/{DATE_EVAL.year} et le {DATE_EVAL.strftime("%d/%m/%Y")} — {nb_ventes_ytd} operation(s)</div>', unsafe_allow_html=True)
                    pv1, pv2, pv3, pv4 = st.columns(4)
                    pv1.metric("P&L brut (plus-values)", f"{pv_brute_ytd/1e9:,.2f} Md DA")
                    pv2.metric("P&L net (apres impot & com)", f"{pv_nette_ytd/1e9:,.2f} Md DA")
                    pv3.metric("Nombre de cessions", f"{nb_ventes_ytd}")
                    pv4.metric("Volume total cede", f"{vol_cede_ytd/1e9:,.2f} Md DA")

                    # P&L par mois en tableau
                    hv_ytd["Mois"] = hv_ytd["Date_vente"].dt.to_period("M")
                    pl_mensuel = hv_ytd.groupby("Mois").agg(
                        Nb_ventes=("Plus_value", "count"),
                        PV_brute_DA=("Plus_value", "sum"),
                        PV_nette_DA=("Plus_value_nette", "sum"),
                        Volume_cede_DA=("Valeur_nominale", "sum")
                    ).reset_index()
                    pl_mensuel["Mois"] = pl_mensuel["Mois"].astype(str)
                    pl_mensuel["PV_brute_DA"] = pl_mensuel["PV_brute_DA"].apply(lambda x: f"{x:,.0f} DA")
                    pl_mensuel["PV_nette_DA"] = pl_mensuel["PV_nette_DA"].apply(lambda x: f"{x:,.0f} DA")
                    pl_mensuel["Volume_cede_DA"] = pl_mensuel["Volume_cede_DA"].apply(lambda x: f"{x:,.0f} DA")
                    st.dataframe(pl_mensuel, hide_index=True, use_container_width=True, height=300)

                    # Detail de chaque cession
                    with st.expander("Detail de chaque cession", expanded=False):
                        hv_aff = hv_ytd.sort_values("Date_vente", ascending=False).copy()
                        hv_aff["Date_vente"] = hv_aff["Date_vente"].dt.strftime('%d/%m/%Y')
                        st.dataframe(hv_aff[["Type", "Contrepartie", "Date_vente", "Exonere", "Tx_rendement",
                                                "Prix_achat", "Prix_vente", "Plus_value", "Plus_value_nette"]]
                                    .style.format({"Tx_rendement": "{:.2%}", "Prix_achat": "{:.3%}", "Prix_vente": "{:.3%}",
                                                    "Plus_value": "{:,.0f} DA", "Plus_value_nette": "{:,.0f} DA"}),
                                    height=350, use_container_width=True)

    # ======================== 6. CONTREPARTIES ========================
    with tab6:
        st.markdown("## Repartition par contrepartie")
        st.markdown('<div style="color:#5B6472; margin-top:-10px; margin-bottom:22px;">Exposition du portefeuille avant et apres optimisation</div>', unsafe_allow_html=True)

        volume_ctp = (d["rep_ctp"] * d["BUDGET"]).rename("Volume_actuel_DA")
        volume_ctp_cib = (d["rep_ctp_cib"] * d["BUDGET"]).rename("Volume_cible_DA")
        nb_operations_ctp = df["Contrepartie"].value_counts().rename("Nb_operations")
        tbl_ctp = pd.concat([
            (d["rep_ctp"] * 100).rename("Poids_actuel_%"),
            (d["rep_ctp_cib"] * 100).rename("Poids_cible_%"),
            volume_ctp, volume_ctp_cib, nb_operations_ctp,
        ], axis=1).fillna(0)
        tbl_ctp = tbl_ctp.sort_values("Volume_actuel_DA", ascending=False).reset_index().rename(columns={"index": "Contrepartie"})
        top_ctp = tbl_ctp.iloc[0]

        with st.container(border=True, key="panel_ctp_top_v2"):
            st.markdown("### Contrepartie principale")
            st.markdown('<div class="panel-sub">Plus forte exposition actuelle</div>', unsafe_allow_html=True)
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Contrepartie", top_ctp["Contrepartie"])
            tc2.metric("Poids actuel", f"{top_ctp['Poids_actuel_%']:.1f}%")
            tc3.metric("Nb operations", f"{int(top_ctp['Nb_operations'])}", delta_color="off")

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        with st.container(border=True, key="panel_ctp_repartition_v2"):
            st.markdown("### Poids actuel vs cible")
            st.markdown('<div class="panel-sub">Comparaison par contrepartie</div>', unsafe_allow_html=True)
            comp_ctp = tbl_ctp[["Contrepartie", "Poids_actuel_%", "Poids_cible_%"]].melt(
                id_vars="Contrepartie", var_name="variable", value_name="value")
            comp_ctp["variable"] = comp_ctp["variable"].map({"Poids_actuel_%": "Actuel", "Poids_cible_%": "Cible"})
            fig_ctp = px.bar(comp_ctp, x="Contrepartie", y="value", color="variable", barmode="group",
                             color_discrete_map={"Actuel": PALETTE["actuel"], "Cible": PALETTE["cible"]})
            fig_ctp.update_traces(marker_line_width=0)
            fig_ctp.for_each_trace(lambda t: t.update(marker=dict(cornerradius=6)))
            fig_ctp.update_layout(yaxis_title="Poids (%)", xaxis_title=None, legend_title=None)
            st.plotly_chart(style_plot(fig_ctp), use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        with st.container(border=True, key="panel_ctp_table_v2"):
            st.markdown("### Detail par contrepartie")
            st.markdown('<div class="panel-sub">Volume et nombre d\'operations, triees par exposition</div>', unsafe_allow_html=True)
            st.dataframe(tbl_ctp.style.format({"Poids_actuel_%": "{:.1f}%", "Poids_cible_%": "{:.1f}%",
                                                "Volume_actuel_DA": "{:,.0f} DA", "Volume_cible_DA": "{:,.0f} DA"}),
                        use_container_width=True, height=400)

    # ======================== 7. TRESORERIE ========================
    with tab7:
        st.markdown("## Tresorerie")
        st.markdown('<div style="color:#5B6472; margin-top:-10px; margin-bottom:22px;">Solde d\'ouverture, marche monetaire et annexes devises — saisis depuis la barre laterale</div>', unsafe_allow_html=True)

        with st.container(border=True, key="panel_tresorerie_solde"):
            st.markdown("### Solde d'ouverture")
            st.metric("Solde d'ouverture (DA)", f"{st.session_state.get('solde_ouverture', 0.0):,.0f} DA")

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        with st.container(border=True, key="panel_tresorerie_coupons"):
            st.markdown("### Alertes coupons")
            st.markdown('<div class="panel-sub">Tombees de coupon a venir sur les titres actuellement en portefeuille (14 prochains jours)</div>', unsafe_allow_html=True)

            HORIZON_ALERTES_COUPONS_JOURS = 14
            titres_detenus = df[df["Poids_actuel"] > 0]
            alertes_coupon = []
            for _, row in titres_detenus.iterrows():
                prochaine = prochaine_date_coupon(row["Echeance"], DATE_EVAL)
                jours_restants = (prochaine - DATE_EVAL).days
                if 0 <= jours_restants <= HORIZON_ALERTES_COUPONS_JOURS:
                    montant_coupon = row["VN"] * row["Coupon"]
                    alertes_coupon.append({"Date": prochaine, "Jours": jours_restants, "Type": row["Type"],
                                           "Contrepartie": row["Contrepartie"], "Montant (DA)": montant_coupon})

            if not alertes_coupon:
                st.markdown(badge(f"Aucune tombee de coupon dans les {HORIZON_ALERTES_COUPONS_JOURS} prochains jours", "ok"), unsafe_allow_html=True)
            else:
                alertes_coupon.sort(key=lambda a: a["Date"])
                st.caption(f"Total attendu sous {HORIZON_ALERTES_COUPONS_JOURS} jours : "
                           f"{sum(a['Montant (DA)'] for a in alertes_coupon):,.0f} DA sur {len(alertes_coupon)} tombee(s)")
                for a in alertes_coupon:
                    if a["Jours"] == 0:
                        quand = "Aujourd'hui"
                    elif a["Jours"] == 1:
                        quand = "Demain"
                    else:
                        quand = f"Dans {a['Jours']} jours ({a['Date'].strftime('%d/%m/%Y')})"
                    kind = "warn" if a["Jours"] <= 2 else "ok"
                    st.markdown(badge(f"{quand} : coupon de {a['Montant (DA)']:,.0f} DA — {a['Type']} ({a['Contrepartie']})", kind),
                                unsafe_allow_html=True)

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        with st.container(border=True, key="panel_tresorerie_mm"):
            st.markdown("### Marche monetaire (MM)")
            st.markdown('<div class="panel-sub">Echeancier complet, tri par date d\'echeance</div>', unsafe_allow_html=True)
            if not st.session_state.mm_operations:
                st.caption("Aucune operation MM saisie. Ajoutez-en depuis la rubriques Tresorerie de la barre laterale.")
            else:
                df_mm_t = pd.DataFrame(st.session_state.mm_operations).sort_values("Date echeance").reset_index(drop=True)
                jours_t = (df_mm_t["Date echeance"] - df_mm_t["Date valeur"]).dt.days.clip(lower=0)
                df_mm_t["Interet (DA)"] = df_mm_t["Montant (DA)"] * (df_mm_t["Taux (%)"] / 100) * jours_t / 365
                df_mm_t["Capital + interet (DA)"] = df_mm_t["Montant (DA)"] + df_mm_t["Interet (DA)"]

                mt1, mt2, mt3 = st.columns(3)
                mt1.metric("Capital place", f"{df_mm_t['Montant (DA)'].sum():,.0f} DA")
                mt2.metric("Interets projetes", f"{df_mm_t['Interet (DA)'].sum():,.0f} DA")
                mt3.metric("Nb operations", f"{len(df_mm_t)}", delta_color="off")

                df_mm_t_aff = df_mm_t.copy()
                df_mm_t_aff["Date valeur"] = df_mm_t_aff["Date valeur"].dt.strftime("%d/%m/%Y")
                df_mm_t_aff["Date echeance"] = df_mm_t_aff["Date echeance"].dt.strftime("%d/%m/%Y")
                st.dataframe(df_mm_t_aff.style.format({"Taux (%)": "{:.2f}", "Montant (DA)": "{:,.0f}",
                                                        "Interet (DA)": "{:,.0f}", "Capital + interet (DA)": "{:,.0f}"}),
                             use_container_width=True, height=min(38 * (len(df_mm_t_aff) + 1) + 3, 400))

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        with st.container(border=True, key="panel_tresorerie_annexes"):
            st.markdown("### Annexes")
            st.markdown('<div class="panel-sub">Achats devises executes par les traders, avec contrevaleur DZD</div>', unsafe_allow_html=True)
            if not st.session_state.annexe_operations:
                st.caption("Aucune annexe saisie. Ajoutez-en depuis la rubriques Tresorerie de la barre laterale.")
            else:
                df_an_t = pd.DataFrame(st.session_state.annexe_operations).sort_values("Date valeur").reset_index(drop=True)
                for col in ("Cours", "Contrevaleur DZD"):
                    if col not in df_an_t.columns:
                        df_an_t[col] = 0.0
                    df_an_t[col] = df_an_t[col].fillna(0.0)

                devises = sorted(df_an_t["Devise"].unique())
                cols_dev = st.columns(min(len(devises), 4) or 1)
                for col, dev in zip(cols_dev, devises):
                    sous = df_an_t[df_an_t["Devise"] == dev]
                    achat = sous["Montant"].sum()
                    with col:
                        st.metric(f"{dev} — Total achats", f"{achat:,.0f}")
                        for trader in ("BENKADI Amine", "BOUDEFFA Nassima"):
                            montant_trader = sous.loc[sous["Trader"] == trader, "Montant"].sum()
                            if montant_trader:
                                st.caption(f"{trader} : {montant_trader:,.0f} {dev}")
                st.metric("Total contrevaleur", f"{df_an_t['Contrevaleur DZD'].sum():,.0f} DZD")

                df_an_t_aff = df_an_t.copy()
                df_an_t_aff["Date valeur"] = df_an_t_aff["Date valeur"].dt.strftime("%d/%m/%Y")
                ligne_total_t = pd.DataFrame([{"Devise": "TOTAL", "Trader": "", "Date valeur": "", "Sens": "",
                                               "Montant": 0.0, "Cours": 0.0, "Contrevaleur DZD": df_an_t["Contrevaleur DZD"].sum()}])
                df_an_t_aff = pd.concat([df_an_t_aff, ligne_total_t], ignore_index=True)
                st.dataframe(df_an_t_aff.style.format({"Montant": "{:,.0f}", "Cours": "{:,.4f}", "Contrevaleur DZD": "{:,.0f}"}),
                             use_container_width=True, height=min(38 * (len(df_an_t_aff) + 1) + 3, 400))

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        with st.container(border=True, key="panel_tresorerie_echeances"):
            st.markdown("### Echeances a venir")
            fin_annee_tresorerie = pd.Timestamp(year=DATE_EVAL.year, month=12, day=31)
            st.markdown(f'<div class="panel-sub">Titres du portefeuille arrivant a echeance d\'ici le {fin_annee_tresorerie.strftime("%d/%m/%Y")} '
                        f'et echeances MM, toutes sources confondues, triees par date</div>', unsafe_allow_html=True)

            lignes_echeances = []
            titres_echeant = df[(df["Echeance"] > DATE_EVAL) & (df["Echeance"] <= fin_annee_tresorerie) & (df["Poids_actuel"] > 0)]
            for _, row in titres_echeant.iterrows():
                lignes_echeances.append({"Date": row["Echeance"], "Source": "Titre", "Detail": f'{row["Type"]} ({row["Contrepartie"]})',
                                         "Montant (DA)": row["VN"]})
            for op in st.session_state.mm_operations:
                jours_op = max((op["Date echeance"] - op["Date valeur"]).days, 0)
                montant_op = op["Montant (DA)"] * (1 + (op["Taux (%)"] / 100) * jours_op / 365)
                lignes_echeances.append({"Date": op["Date echeance"], "Source": "MM", "Detail": op["Contrepartie"],
                                         "Montant (DA)": montant_op})

            if not lignes_echeances:
                st.caption("Aucune echeance a venir.")
            else:
                df_ech = pd.DataFrame(lignes_echeances).sort_values("Date").reset_index(drop=True)
                st.caption(f"Montant total attendu : {df_ech['Montant (DA)'].sum():,.0f} DA sur {len(df_ech)} echeance(s)")
                df_ech_aff = df_ech.copy()
                df_ech_aff["Date"] = df_ech_aff["Date"].dt.strftime("%d/%m/%Y")
                st.dataframe(df_ech_aff.style.format({"Montant (DA)": "{:,.0f} DA"}), use_container_width=True,
                             height=min(38 * (len(df_ech_aff) + 1) + 3, 400))

# ==================================================================
# EMPTY STATE
# ==================================================================
else:
    with st.container(border=True, key="panel_attente_v2"):
        if st.session_state.get('_fichier') is not None:
            st.markdown("### Fichier charge")
            st.markdown('<div style="color:#5B6472;">Ajustez les parametres dans la barre laterale, puis lancez l\'optimisation.</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="td-empty">
                <div class="td-empty-icon">
                    <svg viewBox="0 0 24 24"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                    <line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
                </div>
                <h3>Pret a optimiser votre portefeuille</h3>
                <p>Chargez votre fichier de position Excel ci-dessus pour lancer le moteur d'optimisation QP.
                Les parametres de gestion et d'aversion au risque sont configurables dans la barre laterale.</p>
                <div class="td-empty-features">
                    <div class="td-empty-feat">
                        <div class="td-empty-feat-icon">
                            <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                        </div>
                        <div class="td-empty-feat-label">Risque<br>multi-facteurs</div>
                    </div>
                    <div class="td-empty-feat">
                        <div class="td-empty-feat-icon">
                            <svg viewBox="0 0 24 24"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
                        </div>
                        <div class="td-empty-feat-label">Rendement<br>net fiscal</div>
                    </div>
                    <div class="td-empty-feat">
                        <div class="td-empty-feat-icon">
                            <svg viewBox="0 0 24 24"><path d="M12 20V10"></path><path d="M18 20V4"></path><path d="M6 20v-4"></path></svg>
                        </div>
                        <div class="td-empty-feat-label">Frontiere<br>efficiente</div>
                    </div>
                    <div class="td-empty-feat">
                        <div class="td-empty-feat-icon">
                            <svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
                        </div>
                        <div class="td-empty-feat-label">Export<br>Excel</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
