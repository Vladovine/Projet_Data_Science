"""
Application Streamlit pour l'Analyse Immobilière DVF + Population
Dashboard interactif complet avec prédictions et recommandations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="La Maison de l'Investissement Immo",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# La Maison de l'Investissement Immo\n\nDashboard d'analyse immobilière complet avec données enrichies, prédictions et recommandations d'investissement."
    }
)

# Style CSS personnalisé (Thème sombre conservé)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
    }
    /* Amélioration du contraste pour le thème sombre */
    .stMarkdown {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# Fonction de chargement des données avec cache
@st.cache_data
def load_data():
    """Charge les données avec mise en cache"""
    try:
        df = pd.read_csv('dvf_population_75_92_final.csv', sep=';')
        df['date_mutation'] = pd.to_datetime(df['date_mutation'])
        return df
    except:
        try:
            df = pd.read_csv('dvf_population_75_92_final.csv', sep=';')
            df['date_mutation'] = pd.to_datetime(df['date_mutation'])
            return df
        except Exception as e:
            st.error(f"Erreur de chargement: {e}")
            return None

# Fonction de prédiction des prix
@st.cache_data
def predict_future_prices(data, commune, years_ahead):
    """Prédit les prix futurs pour une commune"""
    commune_data = data[data['nom_commune'] == commune].copy()
    if len(commune_data) < 10:
        return None
    
    # Préparer les données temporelles
    commune_data['year_numeric'] = commune_data['annee']
    yearly_avg = commune_data.groupby('year_numeric')['prix_m2'].mean().reset_index()
    
    if len(yearly_avg) < 2:
        return None
    
    # Modèle de régression linéaire
    X = yearly_avg['year_numeric'].values.reshape(-1, 1)
    y = yearly_avg['prix_m2'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Prédictions
    current_year = yearly_avg['year_numeric'].max()
    future_years = np.array([current_year + i for i in range(1, years_ahead + 1)]).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    return {
        'years': future_years.flatten(),
        'predictions': predictions,
        'slope': model.coef_[0],
        'current_price': yearly_avg['prix_m2'].iloc[-1]
    }

# Fonction de recommandation
def generate_recommendations(row):
    """Génère des recommandations pour un bien"""
    recommendations = []
    score = 0
    
    # Rendement
    if row['rendement_brut_pct_v2'] >= 4:
        recommendations.append("✅ Excellent rendement locatif")
        score += 2
    elif row['rendement_brut_pct_v2'] >= 3:
        recommendations.append("👍 Bon rendement locatif")
        score += 1
    else:
        recommendations.append("⚠️ Rendement faible")
    
    # Cashflow
    if row['est_cashflow_positif'] == 1:
        recommendations.append("✅ Cashflow positif")
        score += 2
    else:
        recommendations.append("❌ Cashflow négatif")
        score -= 1
    
    # Espaces verts
    if row['est_quartier_vert'] == 1:
        recommendations.append("🌳 Quartier vert - Cadre de vie agréable")
        score += 1
    
    # Transport
    if 'score_transport' in row.index and row['score_transport'] >= 7:
        recommendations.append("🚇 Excellente desserte transport")
        score += 2
    elif 'score_transport' in row.index and row['score_transport'] >= 5:
        recommendations.append("🚌 Bonne desserte transport")
        score += 1
    
    # Évolution prix
    if row['croissance_annuelle_pct'] > 2:
        recommendations.append("📈 Forte croissance des prix")
        score += 2
    elif row['croissance_annuelle_pct'] > 0:
        recommendations.append("📊 Croissance modérée des prix")
        score += 1
    else:
        recommendations.append("📉 Prix en baisse")
        score -= 1
    
    # Volatilité
    if row['volatilite_pct'] < 5:
        recommendations.append("✅ Marché stable")
        score += 1
    elif row['volatilite_pct'] > 10:
        recommendations.append("⚠️ Marché volatile")
        score -= 1
    
    # Score final
    if score >= 6:
        verdict = "🎯 FORTEMENT RECOMMANDÉ"
        risk = "Faible"
    elif score >= 3:
        verdict = "👍 RECOMMANDÉ"
        risk = "Modéré"
    elif score >= 0:
        verdict = "⚠️ À ÉTUDIER"
        risk = "Moyen"
    else:
        verdict = "❌ NON RECOMMANDÉ"
        risk = "Élevé"
    
    return {
        'verdict': verdict,
        'score': score,
        'risk': risk,
        'recommendations': recommendations
    }

# Chargement des données
with st.spinner('📂 Chargement des données...'):
    df = load_data()

if df is None:
    st.error("❌ Impossible de charger les données")
    st.stop()

# Logo et Titre principal
col_logo, col_title = st.columns([1, 4])

with col_logo:
    try:
        st.image("logo.svg", width=120)
    except:
        st.markdown("🏠")

with col_title:
    st.markdown('<h1 class="main-header">La Maison de l\'Investissement Immo</h1>', unsafe_allow_html=True)
    st.markdown(f"**{len(df):,} transactions** | Paris (75) & Hauts-de-Seine (92)")

st.markdown("---")

# Sidebar - Filtres globaux
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
    <h2 style='color: white; margin: 0;'>🎛️ Filtres</h2>
</div>
""", unsafe_allow_html=True)

# Filtre départements
depts = sorted(df['code_departement'].unique())
selected_depts = st.sidebar.multiselect("Départements", depts, default=depts)

# Filtre communes avec option "Toutes"
communes = sorted(df[df['code_departement'].isin(selected_depts)]['nom_commune'].unique())
select_all_communes = st.sidebar.checkbox("Sélectionner toutes les communes", value=False)

if select_all_communes:
    selected_communes = communes
else:
    selected_communes = st.sidebar.multiselect(
        "Communes", 
        communes, 
        default=communes[:10] if len(communes) > 10 else communes
    )

# Filtre années avec option "Toutes"
annees = sorted(df['annee'].unique())
select_all_years = st.sidebar.checkbox("Sélectionner toutes les années", value=True)

if select_all_years:
    selected_years = annees
else:
    selected_years = st.sidebar.multiselect("Années", annees, default=annees)

# Filtre type de bien
types = sorted(df['type_bien'].unique())
selected_types = st.sidebar.multiselect("Types de biens", types, default=types)

# Filtre prix au m²
prix_min, prix_max = float(df['prix_m2'].min()), float(df['prix_m2'].quantile(0.95))
prix_range = st.sidebar.slider("Prix au m² (€)", prix_min, prix_max, (prix_min, prix_max), step=100.0)

# Filtre surface
surf_min, surf_max = float(df['surface_reelle_bati'].min()), float(df['surface_reelle_bati'].quantile(0.95))
surf_range = st.sidebar.slider("Surface (m²)", surf_min, surf_max, (surf_min, surf_max), step=5.0)

# Filtres avancés
with st.sidebar.expander("🔍 Filtres Avancés"):
    col1, col2 = st.columns(2)
    
    with col1:
        quartier_vert = st.checkbox("Quartier vert")
        cashflow_positif = st.checkbox("Cashflow positif")
        if 'score_transport' in df.columns:
            proche_transport = st.checkbox("Proche transport")
        else:
            proche_transport = False
    
    with col2:
        quartier_non_vert = st.checkbox("Quartier NON vert")
        cashflow_negatif = st.checkbox("Cashflow négatif")
        if 'score_transport' in df.columns:
            eloigne_transport = st.checkbox("Éloigné transport")
        else:
            eloigne_transport = False
    
    rendement_min = st.slider("Rendement minimum (%)", 0.0, 10.0, 0.0, 0.5)

# Appliquer les filtres
filtered = df[
    (df['code_departement'].isin(selected_depts)) &
    (df['nom_commune'].isin(selected_communes)) &
    (df['annee'].isin(selected_years)) &
    (df['type_bien'].isin(selected_types)) &
    (df['prix_m2'].between(prix_range[0], prix_range[1])) &
    (df['surface_reelle_bati'].between(surf_range[0], surf_range[1])) &
    (df['rendement_brut_pct_v2'] >= rendement_min)
]

# Filtres avancés mutuellement exclusifs
if quartier_vert and not quartier_non_vert:
    filtered = filtered[filtered['est_quartier_vert'] == 1]
elif quartier_non_vert and not quartier_vert:
    filtered = filtered[filtered['est_quartier_vert'] == 0]

if cashflow_positif and not cashflow_negatif:
    filtered = filtered[filtered['est_cashflow_positif'] == 1]
elif cashflow_negatif and not cashflow_positif:
    filtered = filtered[filtered['est_cashflow_positif'] == 0]

if 'est_proche_transport' in filtered.columns:
    if proche_transport and not eloigne_transport:
        filtered = filtered[filtered['est_proche_transport'] == 1]
    elif eloigne_transport and not proche_transport:
        filtered = filtered[filtered['est_proche_transport'] == 0]

# Afficher le nombre de résultats
st.sidebar.markdown("---")
st.sidebar.metric("📊 Transactions", f"{len(filtered):,}")
st.sidebar.metric("📉 % du total", f"{len(filtered)/len(df)*100:.1f}%")

if len(filtered) == 0:
    st.warning("⚠️ Aucune donnée. Ajustez les filtres.")
    st.stop()

# Créer des onglets
tabs = st.tabs([
    "📊 Vue d'ensemble",
    "🗺️ Géographie",
    "🚇 Transports",
    "💰 Analyse Financière",
    "🌳 Espaces Verts",
    "📈 Socio-Économique",
    "🔮 Évolution & Prédictions",
    "💡 Recommandations",
    "🎯 Opportunités"
])

# TAB 1: Vue d'ensemble
with tabs[0]:
    st.header("📊 Vue d'Ensemble du Marché")
    
    # Curseur pour sélectionner une année spécifique
    col_year, col_all = st.columns([3, 1])
    with col_year:
        selected_year_overview = st.select_slider(
            "Sélectionner une année pour l'analyse",
            options=['Toutes'] + sorted(filtered['annee'].unique().tolist()),
            value='Toutes'
        )
    
    # Filtrer par année si sélectionnée
    if selected_year_overview != 'Toutes':
        filtered_year = filtered[filtered['annee'] == selected_year_overview]
    else:
        filtered_year = filtered
    
    # Métriques principales
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("🏠 Logements", f"{len(filtered_year):,}")
    with col2:
        st.metric("💰 Prix Médian", f"{filtered_year['valeur_fonciere'].median():,.0f}€")
    with col3:
        st.metric("📐 Prix/m² Médian", f"{filtered_year['prix_m2'].median():,.0f}€")
    with col4:
        st.metric("📏 Surface Médiane", f"{filtered_year['surface_reelle_bati'].median():.0f}m²")
    with col5:
        st.metric("📈 Rendement Médian", f"{filtered_year['rendement_brut_pct_v2'].median():.2f}%")
    with col6:
        cashflow_pct = (filtered_year['est_cashflow_positif'].sum() / len(filtered_year) * 100)
        st.metric("💵 Cashflow +", f"{cashflow_pct:.1f}%")
    
    st.markdown("---")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution des Prix au m²")
        fig = px.histogram(filtered_year, x='prix_m2', nbins=50, 
                          labels={'prix_m2': 'Prix au m² (€)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Répartition par Type de Bien")
        type_counts = filtered_year['type_bien'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index)
        st.plotly_chart(fig, use_container_width=True)
    
    # Évolution temporelle
    st.subheader("Évolution du Prix Médian au m²")
    evolution = filtered.groupby(filtered['date_mutation'].dt.to_period('M'))['prix_m2'].median().reset_index()
    evolution['date_mutation'] = evolution['date_mutation'].astype(str)
    fig = px.line(evolution, x='date_mutation', y='prix_m2', markers=True,
                 labels={'date_mutation': 'Mois', 'prix_m2': 'Prix médian au m² (€)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques par commune
    st.subheader("Statistiques par Commune")
    if len(selected_communes) > 0:
        commune_stats = filtered_year.groupby('nom_commune').agg({
            'prix_m2': ['mean', 'median'],
            'rendement_brut_pct_v2': 'median',
            'id_mutation': 'count'
        }).round(2)
        commune_stats.columns = ['Prix m² moyen', 'Prix m² médian', 'Rendement médian (%)', 'Transactions']
        commune_stats = commune_stats.sort_values('Transactions', ascending=False)
        st.dataframe(commune_stats, use_container_width=True)
    else:
        st.info("Sélectionnez au moins une commune pour voir les statistiques détaillées.")

# TAB 2: Géographie
with tabs[1]:
    st.header("🗺️ Analyse Géographique")
    
    if len(selected_communes) == 0:
        st.warning("⚠️ Veuillez sélectionner au moins une commune dans les filtres.")
    else:
        # Carte interactive
        st.subheader("Carte des Prix Médians par Commune")
        map_data = filtered.groupby(['nom_commune', 'latitude', 'longitude']).agg({
            'prix_m2': 'median',
            'id_mutation': 'count'
        }).reset_index()
        
        fig = px.scatter_mapbox(
            map_data,
            lat='latitude',
            lon='longitude',
            size='id_mutation',
            color='prix_m2',
            hover_name='nom_commune',
            hover_data={'prix_m2': ':,.0f', 'id_mutation': ':,'},
            color_continuous_scale='Viridis',
            size_max=30,
            zoom=9,
            height=600
        )
        fig.update_layout(mapbox_style='open-street-map')
        st.plotly_chart(fig, use_container_width=True)
        
        # Top communes
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 15 - Prix au m²")
            top_prix = filtered.groupby('nom_commune')['prix_m2'].median().nlargest(15).sort_values()
            fig = px.bar(x=top_prix.values, y=top_prix.index, orientation='h',
                        labels={'x': 'Prix médian au m² (€)', 'y': 'Commune'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top 15 - Volume")
            top_volume = filtered['nom_commune'].value_counts().head(15).sort_values()
            fig = px.bar(x=top_volume.values, y=top_volume.index, orientation='h',
                        labels={'x': 'Nombre de transactions', 'y': 'Commune'})
            st.plotly_chart(fig, use_container_width=True)

# TAB 3: Transports
with tabs[2]:
    st.header("🚇 Proximité aux Transports en Commun")
    
    if 'score_transport' not in filtered.columns:
        st.warning("⚠️ Les données de transport ne sont pas disponibles.")
        st.info("💡 Exécutez le script 'add_transport_data.py' pour ajouter les données de transport.")
    else:
        # Métriques transport
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Score Moyen", f"{filtered['score_transport'].mean():.2f}/10")
        with col2:
            st.metric("Distance Moyenne", f"{filtered['distance_transport_km'].mean():.2f} km")
        with col3:
            st.metric("Lignes Moyennes", f"{filtered['nb_lignes_transport'].mean():.1f}")
        with col4:
            proche_pct = (filtered['est_proche_transport'].sum() / len(filtered) * 100)
            st.metric("% Proche Transport", f"{proche_pct:.1f}%")
        
        st.markdown("---")
        
        # Carte interactive des zones proches des transports
        st.subheader("🗺️ Carte des Zones Proches des Transports")
        transport_map = filtered.groupby(['nom_commune', 'latitude', 'longitude']).agg({
            'score_transport': 'mean',
            'distance_transport_km': 'mean',
            'id_mutation': 'count'
        }).reset_index()
        
        fig = px.scatter_mapbox(
            transport_map,
            lat='latitude',
            lon='longitude',
            size='id_mutation',
            color='score_transport',
            hover_name='nom_commune',
            hover_data={
                'score_transport': ':.2f',
                'distance_transport_km': ':.2f',
                'id_mutation': ':,'
            },
            color_continuous_scale='RdYlGn',
            size_max=30,
            zoom=9,
            height=600,
            labels={'score_transport': 'Score Transport'}
        )
        fig.update_layout(mapbox_style='open-street-map')
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphiques transport
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution Score Transport")
            fig = px.histogram(filtered, x='score_transport', nbins=30,
                              labels={'score_transport': 'Score de proximité transport'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Répartition par Catégorie")
            cat_counts = filtered['categorie_transport'].value_counts()
            fig = px.pie(values=cat_counts.values, names=cat_counts.index)
            st.plotly_chart(fig, use_container_width=True)
        
        # Impact sur les prix
        st.subheader("Impact de la Proximité Transport sur les Prix")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(filtered, x='est_proche_transport', y='prix_m2',
                        labels={'est_proche_transport': 'Proche Transport', 'prix_m2': 'Prix au m² (€)'})
            fig.update_xaxes(ticktext=['Non', 'Oui'], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            sample = filtered.sample(min(2000, len(filtered)))
            fig = px.scatter(sample, x='score_transport', y='prix_m2', 
                           color='type_bien', opacity=0.5,
                           labels={'score_transport': 'Score transport', 'prix_m2': 'Prix au m² (€)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques par catégorie
        st.subheader("Statistiques par Catégorie de Proximité")
        stats_transport = filtered.groupby('categorie_transport').agg({
            'prix_m2': ['mean', 'median'],
            'rendement_brut_pct_v2': 'median',
            'temps_trajet_centre_min': 'mean',
            'id_mutation': 'count'
        }).round(2)
        stats_transport.columns = ['Prix m² moyen', 'Prix m² médian', 'Rendement médian (%)', 
                                   'Temps trajet (min)', 'Transactions']
        st.dataframe(stats_transport, use_container_width=True)

# TAB 4: Analyse Financière
with tabs[3]:
    st.header("💰 Analyse Financière et Rentabilité")
    
    # Métriques financières
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Loyer Moyen/mois", f"{filtered['loyer_mensuel_estime'].median():,.0f}€")
    with col2:
        st.metric("Mensualité Médiane", f"{filtered['mensualite'].median():,.0f}€")
    with col3:
        st.metric("Taux Couverture", f"{filtered['taux_couverture_pct'].median():.1f}%")
    with col4:
        st.metric("Taux Intérêt", f"{filtered['taux_interet_pct'].median():.2f}%")
    
    st.markdown("---")
    
    # Carte interactive des zones les plus rentables
    st.subheader("🗺️ Carte des Zones les Plus Rentables")
    rentability_map = filtered.groupby(['nom_commune', 'latitude', 'longitude']).agg({
        'rendement_brut_pct_v2': 'median',
        'cashflow_mensuel': 'median',
        'id_mutation': 'count'
    }).reset_index()
    
    fig = px.scatter_mapbox(
        rentability_map,
        lat='latitude',
        lon='longitude',
        size='id_mutation',
        color='rendement_brut_pct_v2',
        hover_name='nom_commune',
        hover_data={
            'rendement_brut_pct_v2': ':.2f',
            'cashflow_mensuel': ':,.0f',
            'id_mutation': ':,'
        },
        color_continuous_scale='Greens',
        size_max=30,
        zoom=9,
        height=600,
        labels={'rendement_brut_pct_v2': 'Rendement (%)'}
    )
    fig.update_layout(mapbox_style='open-street-map')
    st.plotly_chart(fig, use_container_width=True)
    
    # Graphiques financiers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution du Rendement Brut")
        fig = px.histogram(filtered, x='rendement_brut_pct_v2', nbins=40,
                          labels={'rendement_brut_pct_v2': 'Rendement brut (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Cashflow Mensuel")
        fig = px.histogram(filtered, x='cashflow_mensuel', nbins=40,
                          labels={'cashflow_mensuel': 'Cashflow mensuel (€)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Rendement par type de bien
    st.subheader("Rendement par Type de Bien")
    fig = px.box(filtered, x='type_bien', y='rendement_brut_pct_v2',
                labels={'type_bien': 'Type de bien', 'rendement_brut_pct_v2': 'Rendement brut (%)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse par profil emprunteur
    st.subheader("Statistiques par Profil Emprunteur")
    profil_stats = filtered.groupby('profil_emprunteur').agg({
        'prix_m2': 'median',
        'rendement_brut_pct_v2': 'median',
        'mensualite': 'median',
        'id_mutation': 'count'
    }).round(2)
    profil_stats.columns = ['Prix m² médian', 'Rendement médian (%)', 'Mensualité médiane', 'Transactions']
    st.dataframe(profil_stats, use_container_width=True)

# TAB 5: Espaces Verts
with tabs[4]:
    st.header("🌳 Impact des Espaces Verts")
    
    # Métriques espaces verts
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pct_vert = (filtered['est_quartier_vert'].sum() / len(filtered) * 100)
        st.metric("% Quartiers Verts", f"{pct_vert:.1f}%")
    with col2:
        st.metric("Score Moyen", f"{filtered['score_espaces_verts_norm'].mean():.2f}")
    with col3:
        st.metric("Nb Espaces Verts", f"{filtered['nb_espaces_verts'].mean():.1f}")
    with col4:
        st.metric("Superficie Moy.", f"{filtered['superficie_espaces_verts_m2'].mean():,.0f}m²")
    
    st.markdown("---")
    
    # Carte interactive des zones avec espaces verts
    st.subheader("🗺️ Carte des Zones avec Espaces Verts")
    green_map = filtered.groupby(['nom_commune', 'latitude', 'longitude']).agg({
        'score_espaces_verts_norm': 'mean',
        'nb_espaces_verts': 'mean',
        'superficie_espaces_verts_m2': 'mean',
        'id_mutation': 'count'
    }).reset_index()
    
    fig = px.scatter_mapbox(
        green_map,
        lat='latitude',
        lon='longitude',
        size='superficie_espaces_verts_m2',
        color='score_espaces_verts_norm',
        hover_name='nom_commune',
        hover_data={
            'score_espaces_verts_norm': ':.2f',
            'nb_espaces_verts': ':.1f',
            'superficie_espaces_verts_m2': ':,.0f',
            'id_mutation': ':,'
        },
        color_continuous_scale='Greens',
        size_max=30,
        zoom=9,
        height=600,
        labels={'score_espaces_verts_norm': 'Score Espaces Verts'}
    )
    fig.update_layout(mapbox_style='open-street-map')
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparaison quartier vert vs non vert
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prix selon Quartier Vert")
        fig = px.box(filtered, x='est_quartier_vert', y='prix_m2',
                    labels={'est_quartier_vert': 'Quartier Vert', 'prix_m2': 'Prix au m² (€)'})
        fig.update_xaxes(ticktext=['Non', 'Oui'], tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Distribution Score Espaces Verts")
        fig = px.histogram(filtered, x='score_espaces_verts_norm', nbins=30,
                          labels={'score_espaces_verts_norm': 'Score espaces verts'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques comparatives
    st.subheader("Comparaison Quartier Vert vs Non Vert")
    stats_ev = filtered.groupby('est_quartier_vert').agg({
        'prix_m2': ['mean', 'median'],
        'rendement_brut_pct_v2': 'median',
        'id_mutation': 'count'
    }).round(2)
    stats_ev.columns = ['Prix m² moyen', 'Prix m² médian', 'Rendement médian (%)', 'Transactions']
    
    # Renommer l'index en fonction des valeurs présentes
    index_mapping = {0: 'Non Vert', 1: 'Quartier Vert'}
    stats_ev.index = [index_mapping.get(idx, str(idx)) for idx in stats_ev.index]
    
    st.dataframe(stats_ev, use_container_width=True)

# TAB 6: Socio-Économique
with tabs[5]:
    st.header("📈 Analyse Socio-Économique")
    
    # Métriques socio-économiques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Population Moyenne", f"{filtered['population'].mean():,.0f}")
    with col2:
        st.metric("Revenu Médian", f"{filtered['revenu_median'].mean():,.0f}€")
    with col3:
        st.metric("Taux Pauvreté Moyen", f"{filtered['taux_pauvrete'].mean():.1f}%")
    
    st.markdown("---")
    
    # Corrélations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prix vs Revenu Médian")
        sample = filtered.sample(min(2000, len(filtered)))
        fig = px.scatter(sample, x='revenu_median', y='prix_m2', opacity=0.5,
                        labels={'revenu_median': 'Revenu médian (€)', 'prix_m2': 'Prix au m² (€)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Prix vs Taux de Pauvreté")
        fig = px.scatter(sample, x='taux_pauvrete', y='prix_m2', opacity=0.5,
                        labels={'taux_pauvrete': 'Taux de pauvreté (%)', 'prix_m2': 'Prix au m² (€)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Évolution et tendances
    st.subheader("Tendances de Prix")
    col1, col2 = st.columns(2)
    
    with col1:
        tendance_counts = filtered['tendance_categorie'].value_counts()
        fig = px.pie(values=tendance_counts.values, names=tendance_counts.index,
                    title="Répartition des Tendances")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Statistiques d'Évolution")
        st.metric("Évolution Prix m² Moyenne", f"{filtered['evolution_prix_m2_euros'].mean():+,.0f}€")
        st.metric("Croissance Annuelle Moyenne", f"{filtered['croissance_annuelle_pct'].mean():+.2f}%")
        st.metric("Volatilité Moyenne", f"{filtered['volatilite_pct'].mean():.2f}%")

# TAB 7: Évolution & Prédictions
with tabs[6]:
    st.header("🔮 Évolution et Prédictions des Prix")
    
    st.info("💡 Prédictions basées sur les tendances historiques des prix par commune")
    
    # Paramètres de prédiction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if len(selected_communes) > 0:
            commune_prediction = st.selectbox(
                "Sélectionner une commune pour la prédiction",
                options=selected_communes,
                key="commune_prediction_select"
            )
        else:
            st.warning("Veuillez sélectionner au moins une commune dans les filtres.")
            commune_prediction = None
    
    with col2:
        years_ahead = st.slider(
            "Nombre d'années à prédire",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum 10 ans pour garantir la fiabilité",
            key="years_ahead_prediction"
        )
    
    if commune_prediction:
        # Calculer les prédictions
        prediction_result = predict_future_prices(df, commune_prediction, years_ahead)
        
        if prediction_result is None:
            st.warning(f"⚠️ Données insuffisantes pour {commune_prediction}")
        else:
            # Afficher les prédictions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prix Actuel", f"{prediction_result['current_price']:,.0f}€/m²")
            with col2:
                future_price = prediction_result['predictions'][-1]
                st.metric(
                    f"Prix Prédit ({years_ahead} ans)",
                    f"{future_price:,.0f}€/m²",
                    delta=f"{((future_price - prediction_result['current_price']) / prediction_result['current_price'] * 100):+.1f}%"
                )
            with col3:
                st.metric("Tendance Annuelle", f"{prediction_result['slope']:+,.0f}€/m²/an")
            
            st.markdown("---")
            
            # Graphique de prédiction
            st.subheader(f"Prédiction des Prix - {commune_prediction}")
            
            # Données historiques
            commune_hist = df[df['nom_commune'] == commune_prediction].copy()
            hist_yearly = commune_hist.groupby('annee')['prix_m2'].mean().reset_index()
            
            # Créer le graphique
            fig = go.Figure()
            
            # Historique
            fig.add_trace(go.Scatter(
                x=hist_yearly['annee'],
                y=hist_yearly['prix_m2'],
                mode='lines+markers',
                name='Historique',
                line=dict(color='blue', width=2)
            ))
            
            # Prédictions
            fig.add_trace(go.Scatter(
                x=prediction_result['years'],
                y=prediction_result['predictions'],
                mode='lines+markers',
                name='Prédiction',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                xaxis_title="Année",
                yaxis_title="Prix au m² (€)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Carte des zones à risque et recommandées
            st.subheader("🗺️ Carte des Zones d'Investissement")
            
            # Calculer le score de risque pour chaque commune
            risk_scores = []
            for commune in selected_communes:
                commune_data = filtered[filtered['nom_commune'] == commune]
                if len(commune_data) > 0:
                    # Facteurs de risque
                    volatilite = commune_data['volatilite_pct'].mean()
                    croissance = commune_data['croissance_annuelle_pct'].mean()
                    rendement = commune_data['rendement_brut_pct_v2'].median()
                    cashflow_pos = commune_data['est_cashflow_positif'].mean()
                    
                    # Score de risque (plus bas = moins risqué)
                    risk_score = (
                        volatilite * 0.3 -
                        croissance * 0.3 -
                        rendement * 0.2 -
                        cashflow_pos * 20 * 0.2
                    )
                    
                    # Catégorisation
                    if risk_score < -5:
                        category = "🟢 Recommandé"
                        color_val = 3
                    elif risk_score < 0:
                        category = "🟡 Modéré"
                        color_val = 2
                    else:
                        category = "🔴 À Risque"
                        color_val = 1
                    
                    risk_scores.append({
                        'nom_commune': commune,
                        'latitude': commune_data['latitude'].iloc[0],
                        'longitude': commune_data['longitude'].iloc[0],
                        'risk_score': risk_score,
                        'category': category,
                        'color_val': color_val,
                        'croissance': croissance,
                        'rendement': rendement,
                        'volatilite': volatilite
                    })
            
            if risk_scores:
                risk_df = pd.DataFrame(risk_scores)
                
                fig = px.scatter_mapbox(
                    risk_df,
                    lat='latitude',
                    lon='longitude',
                    color='category',
                    size=abs(risk_df['risk_score']) + 5,
                    hover_name='nom_commune',
                    hover_data={
                        'croissance': ':.2f',
                        'rendement': ':.2f',
                        'volatilite': ':.2f',
                        'risk_score': ':.2f'
                    },
                    color_discrete_map={
                        "🟢 Recommandé": "green",
                        "🟡 Modéré": "yellow",
                        "🔴 À Risque": "red"
                    },
                    zoom=9,
                    height=600
                )
                fig.update_layout(
                    mapbox_style='open-street-map',
                    legend=dict(
                        title="Catégorie d'Investissement",
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Légende explicative
                st.markdown("""
                **Légende de la Carte:**
                - 🟢 **Recommandé**: Faible volatilité, bonne croissance, bon rendement, cashflow positif
                - 🟡 **Modéré**: Profil équilibré, à étudier selon vos critères
                - 🔴 **À Risque**: Forte volatilité ou faible croissance, investissement plus risqué
                
                *La taille des marqueurs représente l'intensité du score de risque*
                """)
                
                # Tableau récapitulatif
                st.subheader("Récapitulatif par Zone")
                risk_summary = risk_df[['nom_commune', 'category', 'croissance', 'rendement', 'volatilite']].copy()
                risk_summary.columns = ['Commune', 'Catégorie', 'Croissance (%)', 'Rendement (%)', 'Volatilité (%)']
                risk_summary = risk_summary.sort_values('Catégorie')
                st.dataframe(risk_summary.round(2), use_container_width=True)

# TAB 8: Recommandations
with tabs[7]:
    st.header("💡 Recommandations d'Investissement")
    
    st.info("💡 Recommandations personnalisées basées sur l'analyse complète des données")
    
    # Paramètres de recherche
    col1, col2, col3 = st.columns(3)
    with col1:
        budget_reco = st.number_input(
            "Budget maximum (€)",
            min_value=100000,
            max_value=2000000,
            value=500000,
            step=50000,
            key="budget_reco"
        )
    with col2:
        rendement_reco = st.number_input(
            "Rendement minimum (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            key="rendement_reco"
        )
    with col3:
        nb_recommendations = st.slider(
            "Nombre de recommandations",
            min_value=5,
            max_value=50,
            value=20,
            key="nb_recommendations"
        )
    
    # Filtrer les biens selon les critères
    candidates = filtered[
        (filtered['valeur_fonciere'] <= budget_reco) &
        (filtered['rendement_brut_pct_v2'] >= rendement_reco)
    ].copy()
    
    if len(candidates) == 0:
        st.warning("❌ Aucun bien ne correspond à vos critères. Ajustez les paramètres.")
    else:
        # Générer les recommandations
        st.success(f"✅ {len(candidates):,} biens analysés")
        
        # Calculer le score pour chaque bien
        recommendations_list = []
        
        for idx, row in candidates.head(nb_recommendations * 3).iterrows():
            reco = generate_recommendations(row)
            recommendations_list.append({
                'commune': row['nom_commune'],
                'type_bien': row['type_bien'],
                'prix': row['valeur_fonciere'],
                'surface': row['surface_reelle_bati'],
                'prix_m2': row['prix_m2'],
                'rendement': row['rendement_brut_pct_v2'],
                'cashflow': row['cashflow_mensuel'],
                'score': reco['score'],
                'verdict': reco['verdict'],
                'risk': reco['risk'],
                'recommendations': reco['recommendations']
            })
        
        # Trier par score
        recommendations_df = pd.DataFrame(recommendations_list)
        recommendations_df = recommendations_df.sort_values('score', ascending=False).head(nb_recommendations)
        
        # Afficher les meilleures recommandations
        st.subheader(f"🏆 Top {nb_recommendations} Recommandations")
        
        for idx, reco in recommendations_df.iterrows():
            with st.expander(f"{reco['verdict']} - {reco['commune']} - {reco['type_bien']} - Score: {reco['score']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Caractéristiques:**")
                    st.write(f"💰 Prix: {reco['prix']:,.0f}€")
                    st.write(f"📐 Surface: {reco['surface']:.0f}m²")
                    st.write(f"📊 Prix/m²: {reco['prix_m2']:,.0f}€")
                
                with col2:
                    st.markdown("**Performance:**")
                    st.write(f"📈 Rendement: {reco['rendement']:.2f}%")
                    st.write(f"💵 Cashflow: {reco['cashflow']:,.0f}€/mois")
                    st.write(f"⚠️ Risque: {reco['risk']}")
                
                with col3:
                    st.markdown("**Recommandations:**")
                    for rec in reco['recommendations']:
                        st.write(rec)
        
        # Statistiques des recommandations
        st.markdown("---")
        st.subheader("📊 Statistiques des Recommandations")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prix Moyen", f"{recommendations_df['prix'].mean():,.0f}€")
        with col2:
            st.metric("Rendement Moyen", f"{recommendations_df['rendement'].mean():.2f}%")
        with col3:
            st.metric("Cashflow Moyen", f"{recommendations_df['cashflow'].mean():,.0f}€")
        with col4:
            st.metric("Score Moyen", f"{recommendations_df['score'].mean():.1f}")
        
        # Répartition par verdict
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Répartition par Verdict")
            verdict_counts = recommendations_df['verdict'].value_counts()
            fig = px.pie(values=verdict_counts.values, names=verdict_counts.index)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Répartition par Niveau de Risque")
            risk_counts = recommendations_df['risk'].value_counts()
            fig = px.pie(values=risk_counts.values, names=risk_counts.index)
            st.plotly_chart(fig, use_container_width=True)

# TAB 9: Opportunités
with tabs[8]:
    st.header("🎯 Opportunités d'Investissement")
    
    st.info("💡 Recherche des meilleures opportunités selon vos critères")
    
    # Paramètres de recherche
    col1, col2, col3 = st.columns(3)
    with col1:
        budget_max = st.number_input("Budget maximum (€)", 
                                     min_value=100000, 
                                     max_value=2000000, 
                                     value=500000, 
                                     step=50000,
                                     key="budget_max_opp")
    with col2:
        rend_min = st.number_input("Rendement minimum (%)", 
                                   min_value=0.0, 
                                   max_value=10.0, 
                                   value=3.0, 
                                   step=0.5,
                                   key="rend_min_opp")
    with col3:
        surface_min = st.number_input("Surface minimum (m²)", 
                                     min_value=10, 
                                     max_value=200, 
                                     value=30, 
                                     step=5,
                                     key="surface_min_opp")
    
    # Filtrer les opportunités
    opportunities = filtered[
        (filtered['valeur_fonciere'] <= budget_max) &
        (filtered['rendement_brut_pct_v2'] >= rend_min) &
        (filtered['surface_reelle_bati'] >= surface_min) &
        (filtered['est_cashflow_positif'] == 1)
    ].copy()
    
    # Calculer un score d'opportunité
    if len(opportunities) > 0:
        # Score de base
        score_base = (
            opportunities['rendement_brut_pct_v2'] * 0.30 +
            opportunities['taux_couverture_pct'] / 10 * 0.20 +
            opportunities['score_espaces_verts_norm'] * 10 * 0.15 +
            (opportunities['evolution_prix_m2_pct'] > 0).astype(int) * 5 * 0.15 +
            (opportunities['croissance_annuelle_pct'] > 0).astype(int) * 5 * 0.10
        )
        
        # Ajouter le score transport si disponible
        if 'score_transport' in opportunities.columns:
            score_base = score_base * 0.90 + opportunities['score_transport'] * 0.10
        
        opportunities['score_opportunite'] = score_base
        
        st.success(f"✅ {len(opportunities):,} opportunités trouvées !")
        
        # Top 10 opportunités
        st.subheader("🏆 Top 10 Opportunités")
        
        cols_to_show = [
            'nom_commune', 'type_bien', 'valeur_fonciere', 'surface_reelle_bati',
            'prix_m2', 'rendement_brut_pct_v2', 'cashflow_mensuel', 
            'score_espaces_verts_norm', 'score_opportunite'
        ]
        
        if 'score_transport' in opportunities.columns:
            cols_to_show.insert(-1, 'score_transport')
        
        top_10 = opportunities.nlargest(10, 'score_opportunite')[cols_to_show]
        
        top_10_display = top_10.copy()
        col_names = ['Commune', 'Type', 'Prix', 'Surface', 'Prix/m²', 
                    'Rendement%', 'Cashflow', 'Score EV']
        
        if 'score_transport' in opportunities.columns:
            col_names.append('Score Transport')
        
        col_names.append('Score Total')
        top_10_display.columns = col_names
        top_10_display = top_10_display.round(2)
        st.dataframe(top_10_display, use_container_width=True)
        
        # Visualisations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rendement vs Prix")
            sample_opp = opportunities.sample(min(500, len(opportunities)))
            fig = px.scatter(sample_opp, x='valeur_fonciere', y='rendement_brut_pct_v2',
                           color='cashflow_mensuel', size='surface_reelle_bati',
                           hover_data=['nom_commune', 'type_bien'],
                           labels={'valeur_fonciere': 'Prix (€)', 
                                  'rendement_brut_pct_v2': 'Rendement (%)',
                                  'cashflow_mensuel': 'Cashflow'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Distribution par Commune")
            commune_counts = opportunities['nom_commune'].value_counts().head(10)
            fig = px.bar(x=commune_counts.values, y=commune_counts.index, orientation='h',
                        labels={'x': 'Nombre d\'opportunités', 'y': 'Commune'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("❌ Aucune opportunité trouvée. Ajustez vos critères.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h3 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🏠 La Maison de l'Investissement Immo
    </h3>
    <p style='opacity: 0.7;'>Données enrichies: Transactions, Population, Revenus, Espaces Verts, Transports, Analyses Financières, Prédictions & Recommandations</p>
    <p style='opacity: 0.5; font-size: 0.9rem;'>Paris (75) & Hauts-de-Seine (92) | DVF 2020-2022</p>
</div>
""", unsafe_allow_html=True)
