import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os


st.set_page_config(layout="wide", page_title="Air Quality Dashboard", page_icon="🏭")

st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 20px;}
    h1 {color: #2c3e50;}
    h2 {color: #e74c3c;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    
 
    def get_path(filename):
        return os.path(filename)  
    try:
        
        c_day = pd.read_csv('city_day.csv', parse_dates=['Date'])
       
        s_day = pd.read_csv('station_day.csv', parse_dates=['Date'])
        
        s_hour = pd.read_csv('station_hour.zip', parse_dates=['Datetime'])
        
        
        stations = pd.read_csv('stations.csv')
        

        s_day = pd.merge(s_day, stations[['StationId', 'StationName', 'City', 'Status']], on='StationId', how='left')
        s_hour = pd.merge(s_hour, stations[['StationId', 'StationName', 'City']], on='StationId', how='left')
        
        return c_day, s_day, s_hour, stations
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

city_day, station_day, station_hour, stations = load_data()



if city_day is not None:
    
    
    st.sidebar.title("🎛️ Controls")
    city_list = sorted(city_day['City'].dropna().unique())
    selected_city = st.sidebar.selectbox("Select City", city_list, index=city_list.index('Delhi') if 'Delhi' in city_list else 0)
    
    pollutant_cols = ["PM2.5","PM10","NO","NO2","CO","SO2","O3"]
    selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutant_cols, index=0)

    st.title("🏭 India Air Quality Analysis")
    st.markdown("### A Data-Driven Approach to Pollution Hotspots, Trends, and Health Impact")

  
    tab_health, tab_trends, tab_cluster, tab_audit = st.tabs([
        "🚬 Public Health Impact", 
        "📈 Hotspots & Trends", 
        "🔗 Correlations & Clusters", 
        "🛠️ Sensor Audit"
    ])

   
    with tab_health:
        st.header("Public Health & Policy Impact")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🚬 Cigarette Equivalence")
            st.info("Scientific Rule of Thumb: 22 µg/m³ of PM2.5 ≈ 1 Cigarette smoked.")
            
          
            stats = city_day.groupby('City')['PM2.5'].agg(['mean']).reset_index()
            stats['Cigarettes_Per_Day'] = stats['mean'] / 22
            stats = stats.sort_values(by='Cigarettes_Per_Day', ascending=False).head(15)
            
            fig_cig = px.bar(stats, x='Cigarettes_Per_Day', y='City', orientation='h',
                             title="Breathing the air is equivalent to smoking...",
                             color='Cigarettes_Per_Day', color_continuous_scale='Reds')
            fig_cig.add_vline(x=1, line_dash="dash", line_color="green", annotation_text="1 Cigarette Equiv.")
            st.plotly_chart(fig_cig, use_container_width=True)

        with col2:
            st.subheader("⚠️ Risk Matrix: Chronic vs Acute")
            st.markdown("Separating cities with **consistently bad air** (Chronic) from those with **sudden spikes** (Acute).")
            
            risk_df = city_day.groupby('City')['AQI'].agg(['mean', 'std', lambda x: x.quantile(0.95)]).reset_index()
            risk_df.columns = ['City', 'Chronic (Avg AQI)', 'Volatility', 'Acute (95th% AQI)']
            
            fig_risk = px.scatter(risk_df, x='Chronic (Avg AQI)', y='Acute (95th% AQI)',
                                  size='Volatility', hover_name='City', color='Chronic (Avg AQI)',
                                  color_continuous_scale='YlOrRd',
                                  title="Risk Matrix (Size = Volatility)")
            
           
            fig_risk.add_hline(y=300, line_dash="dot", annotation_text="Severe Spikes")
            fig_risk.add_vline(x=150, line_dash="dot", annotation_text="Unhealthy Avg")
            st.plotly_chart(fig_risk, use_container_width=True)

       
        st.subheader("📉 City Progress (2016 vs 2019)")
        df_prog = city_day.copy()
        df_prog['Year'] = df_prog['Date'].dt.year
        df_trend = df_prog[df_prog['Year'].isin([2016, 2019])].groupby(['City', 'Year'])['PM2.5'].mean().unstack()
        df_trend['Change_Pct'] = ((df_trend[2019] - df_trend[2016]) / df_trend[2016]) * 100
        df_trend = df_trend.dropna().sort_values(by='Change_Pct')
        
       
        colors = ['green' if x < 0 else 'red' for x in df_trend['Change_Pct']]
        
        fig_prog = go.Figure()
        fig_prog.add_trace(go.Bar(x=df_trend.index, y=df_trend['Change_Pct'], marker_color=colors))
        fig_prog.update_layout(title="% Change in PM2.5 (Negative is Good)", xaxis_tickangle=-90)
        st.plotly_chart(fig_prog, use_container_width=True)

   
    with tab_trends:
        col_t1, col_t2 = st.columns([2, 1])
        
        with col_t1:
            st.subheader(f"📅 Daily Trends: {selected_city}")
            df_city = city_day[city_day['City'] == selected_city].sort_values("Date")
            df_city[f'{selected_pollutant}_30d'] = df_city[selected_pollutant].rolling(30, min_periods=7).mean()

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df_city['Date'], y=df_city[selected_pollutant], mode='lines', name='Daily', opacity=0.3, line=dict(color='gray')))
            fig_trend.add_trace(go.Scatter(x=df_city['Date'], y=df_city[f'{selected_pollutant}_30d'], mode='lines', name='30-Day Trend', line=dict(width=3, color='red')))
            fig_trend.update_layout(title=f"{selected_pollutant} Trend", hovermode="x unified")
            st.plotly_chart(fig_trend, use_container_width=True)

        with col_t2:
            st.subheader("🔥 Seasonality Heatmap")
            df_heat = df_city.copy()
            df_heat['Year'] = df_heat['Date'].dt.year
            df_heat['Month'] = df_heat['Date'].dt.month_name()
            pivot = df_heat.pivot_table(index='Month', columns='Year', values=selected_pollutant, aggfunc='mean')
            months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            pivot = pivot.reindex(months_order)
            
            fig_hm = px.imshow(pivot, color_continuous_scale='YlOrRd', title=f"Monthly Avg {selected_pollutant}")
            st.plotly_chart(fig_hm, use_container_width=True)

        st.divider()
        
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🎆 Diwali Impact")
            diwali_dates = {2015:'2015-11-11', 2016:'2016-10-30', 2017:'2017-10-19', 2018:'2018-11-07', 2019:'2019-10-27', 2020:'2020-11-14'}
            impact_data = []
            for y, d in diwali_dates.items():
                d_date = pd.to_datetime(d)
                before = df_city[(df_city['Date'] >= d_date - pd.Timedelta(days=15)) & (df_city['Date'] < d_date)][selected_pollutant].mean()
                after = df_city[(df_city['Date'] >= d_date) & (df_city['Date'] < d_date + pd.Timedelta(days=15))][selected_pollutant].mean()
                if pd.notnull(before): impact_data.append({'Year': y, 'Before': before, 'After': after})
            
            if impact_data:
                idf = pd.DataFrame(impact_data)
                fig_div = px.bar(idf, x='Year', y=['Before', 'After'], barmode='group', title="15 Days Before vs After Diwali")
                st.plotly_chart(fig_div, use_container_width=True)
            else:
                st.warning("Insufficient data for Diwali analysis.")

        with c2:
            st.subheader(f"🏆 Top Polluted Cities ({selected_pollutant})")
            rank_year = st.selectbox("Select Year", [2016, 2017, 2018, 2019, 2020])
            df_yr = city_day[city_day['Date'].dt.year == rank_year]
            rank = df_yr.groupby("City")[selected_pollutant].mean().sort_values(ascending=False).head(10).reset_index()
            fig_rk = px.bar(rank, y='City', x=selected_pollutant, orientation='h', color=selected_pollutant, title=f"Top 10 Cities in {rank_year}")
            st.plotly_chart(fig_rk, use_container_width=True)

    with tab_cluster:
        st.header("Advanced Analytics")
        
        c_clust1, c_clust2 = st.columns([1, 2])
        
        with c_clust1:
            st.subheader("🔢 Correlations")
            scope = st.radio("Scope", ["All India", f"Current City ({selected_city})"])
            
            if scope == "All India":
                corr_df = city_day[pollutant_cols].corr()
            else:
                corr_df = city_day[city_day['City'] == selected_city][pollutant_cols].corr()
            
            fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title="Pollutant Correlations")
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with c_clust2:
            st.subheader("🧩 City Clustering (PCA)")
            n_k = st.slider("Number of Clusters", 2, 6, 4)
            
           
            df_pivot = city_day.groupby('City')[pollutant_cols].mean()
            df_pivot = df_pivot.fillna(df_pivot.mean())
            
            
            scaler = StandardScaler()
            X = scaler.fit_transform(df_pivot)
            kmeans = KMeans(n_clusters=n_k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X)
            df_pivot['Cluster'] = labels.astype(str)
            
           
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X)
            df_pivot['PC1'] = pcs[:, 0]
            df_pivot['PC2'] = pcs[:, 1]
            
            fig_pca = px.scatter(df_pivot.reset_index(), x='PC1', y='PC2', color='Cluster', 
                                 text='City', title="Cities Grouped by Pollution Profile",
                                 hover_data=pollutant_cols)
            fig_pca.update_traces(textposition='top center')
            st.plotly_chart(fig_pca, use_container_width=True)
            
            st.markdown("**Cluster Profiles (Avg Values):**")
            st.dataframe(df_pivot.groupby('Cluster')[pollutant_cols].mean().style.background_gradient(cmap='Greens'))

  
    with tab_audit:
        st.header("🛠️ Data Quality & Sensor Audit")
        
        col_a1, col_a2 = st.columns([2, 1])
        
        with col_a1:
            st.subheader(f"📊 Data Availability: {selected_city}")
            df_avail = station_day[station_day['City'] == selected_city].copy()
            if not df_avail.empty:
                df_avail['HasData'] = df_avail['PM2.5'].notnull().astype(int)
                pivot_av = df_avail.pivot_table(index='StationName', columns='Date', values='HasData', aggfunc='max')
                pivot_av_m = pivot_av.resample('M', axis=1).max() # Monthly view to save rendering time
                
                fig_av = px.imshow(pivot_av_m, color_continuous_scale=['purple', 'yellow'], 
                                   title="Yellow = Active, Purple = Inactive (Monthly View)")
                fig_av.update_xaxes(title="Date")
                fig_av.update_yaxes(title="Station")
                st.plotly_chart(fig_av, use_container_width=True)
            else:
                st.warning("No station data found for this city.")

        with col_a2:
            st.subheader("🕵️ Frozen Sensor Detection")
            st.markdown("Identifies sensors stuck on the exact same value for > 12 hours (Hardware Fault).")
            
            if st.button("Run Hardware Scan (Heavy Compute)"):
                with st.spinner("Scanning hourly data..."):
                
                    df_scan = station_hour[station_hour['City'] == selected_city].sort_values(['StationId', 'Datetime'])
                    frozen_list = []
                    
                    if not df_scan.empty:
                        for sid, group in df_scan.groupby('StationId'):
                            group['diff'] = group['PM2.5'].diff()
                            group['block_key'] = (group['diff'] != 0).cumsum()
                            max_streak = group.groupby('block_key').size().max()
                            
                            if max_streak >= 12:
                                frozen_list.append({
                                    'Station': group['StationName'].iloc[0],
                                    'Max Frozen Hours': max_streak
                                })
                        
                        if frozen_list:
                            st.error(f"Found {len(frozen_list)} Faulty Sensors!")
                            st.dataframe(pd.DataFrame(frozen_list).sort_values('Max Frozen Hours', ascending=False))
                        else:
                            st.success("No frozen sensors detected in this city.")
                    else:
                        st.warning("No hourly data available.")

        st.subheader("📋 Strategic Recommendations")
        rec1, rec2, rec3 = st.columns(3)
        rec1.info("**1. Decommission Ghosts**\n\nStations with <20% uptime should be repaired or removed to save server costs.")
        rec2.warning("**2. Auto-Alerts**\n\nImplement triggers when variance = 0 for >6 hours to catch frozen sensors early.")
        rec3.success("**3. Coverage Gaps**\n\nFocus expansion on 2017-2018 missing zones identified in the Heatmap.")

else:

    st.warning("Data loading failed. Please check paths.")


