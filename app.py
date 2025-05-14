import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats # Import stats for zscore

# Load and cache data
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        if 'NIPP PEKERJA' in df.columns:
            df['NIPP PEKERJA'] = df['NIPP PEKERJA'].astype(str)
        numeric_cols = ['BOBOT', 'TARGET TW TERKAIT', 'REALISASI TW TERKAIT']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True)
        if 'SKOR PENCAPAIAN' not in df.columns and 'REALISASI TW TERKAIT' in df.columns and 'TARGET TW TERKAIT' in df.columns:
            df['SKOR PENCAPAIAN'] = np.where(df['TARGET TW TERKAIT'] != 0, (df['REALISASI TW TERKAIT'] / df['TARGET TW TERKAIT']) * 100, 0)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure the file is uploaded correctly.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Generate synthetic historical data for prediction demonstration
@st.cache_data
def generate_synthetic_history(df_current, kpi_name_col='NAMA KPI', value_col='REALISASI TW TERKAIT', n_periods=12):
    df_hist_all = []
    if df_current.empty or kpi_name_col not in df_current.columns or value_col not in df_current.columns:
        return pd.DataFrame()

    for kpi_name in df_current[kpi_name_col].unique():
        current_value = df_current[df_current[kpi_name_col] == kpi_name][value_col].mean()
        if pd.isna(current_value):
            current_value = 50 

        trend = np.linspace(current_value * 0.8, current_value * 1.1, n_periods)
        noise = np.random.normal(0, current_value * 0.05, n_periods)
        historical_values = np.clip(trend + noise, 0, None)
        periods = [f"T-{n_periods-i}" for i in range(n_periods)]
        
        df_hist_kpi = pd.DataFrame({
            kpi_name_col: kpi_name,
            'PERIODE': periods,
            value_col: historical_values
        })
        df_hist_all.append(df_hist_kpi)
    
    if not df_hist_all:
        return pd.DataFrame()
    return pd.concat(df_hist_all, ignore_index=True)

DATA_FILE_PATH = '/home/ubuntu/upload/kpi_cleaned.csv'
df_kpi = load_data(DATA_FILE_PATH)
df_kpi_history = generate_synthetic_history(df_kpi)

st.set_page_config(layout="wide")
st.title("Aplikasi Analisis dan Prediksi Kinerja KPI")

st.sidebar.title("Navigasi Fitur")
app_mode = st.sidebar.selectbox(
    "Pilih Fitur:",
    ["Dasbor Kinerja KPI", "Prediksi Kinerja KPI", "Deteksi Anomali KPI"]
)

if df_kpi.empty:
    st.warning("Data KPI tidak dapat dimuat. Beberapa fitur mungkin tidak berfungsi.")
else:
    if app_mode == "Dasbor Kinerja KPI":
        st.header("Dasbor Kinerja KPI Interaktif")
        st.sidebar.header("Filter Data Dasbor")
        unique_posisi = df_kpi["POSISI PEKERJA"].unique()
        selected_posisi = st.sidebar.multiselect(
            "Pilih Posisi Pekerja:",
            options=unique_posisi,
            default=unique_posisi[:3] if len(unique_posisi) > 3 else unique_posisi
        )
        unique_kpi_nama = df_kpi["NAMA KPI"].unique()
        selected_kpi_nama = st.sidebar.multiselect(
            "Pilih Nama KPI:",
            options=unique_kpi_nama,
            default=unique_kpi_nama[:3] if len(unique_kpi_nama) > 3 else unique_kpi_nama
        )
        unique_perusahaan = df_kpi["PERUSAHAAN"].unique()
        selected_perusahaan = st.sidebar.multiselect(
            "Pilih Perusahaan:",
            options=unique_perusahaan,
            default=unique_perusahaan
        )

        df_filtered = df_kpi[
            df_kpi["POSISI PEKERJA"].isin(selected_posisi) &
            df_kpi["NAMA KPI"].isin(selected_kpi_nama) &
            df_kpi["PERUSAHAAN"].isin(selected_perusahaan)
        ]

        if df_filtered.empty:
            st.warning("Tidak ada data yang cocok dengan filter yang dipilih.")
        else:
            st.subheader("Ringkasan Data KPI Terfilter")
            col1, col2, col3 = st.columns(3)
            total_kpis = df_filtered.shape[0]
            avg_realisasi = df_filtered["REALISASI TW TERKAIT"].mean()
            avg_target = df_filtered["TARGET TW TERKAIT"].mean()
            avg_pencapaian = 0
            if 'SKOR PENCAPAIAN' in df_filtered.columns:
                 avg_pencapaian = df_filtered["SKOR PENCAPAIAN"].mean()

            with col1:
                st.metric(label="Total KPI Terfilter", value=total_kpis)
            with col2:
                st.metric(label="Rata-rata Realisasi", value=f"{avg_realisasi:.2f}")
            with col3:
                st.metric(label="Rata-rata Target", value=f"{avg_target:.2f}")
            if 'SKOR PENCAPAIAN' in df_filtered.columns:
                st.metric(label="Rata-rata Skor Pencapaian (%)", value=f"{avg_pencapaian:.2f}%")

            st.subheader("Data KPI Terfilter")
            st.dataframe(df_filtered)

            st.subheader("Visualisasi Kinerja KPI")
            st.write("Perbandingan Realisasi vs Target untuk KPI Terpilih (Top 10 berdasarkan Target)")
            df_plot_bar = df_filtered.nlargest(10, 'TARGET TW TERKAIT')
            if not df_plot_bar.empty:
                fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
                index = np.arange(len(df_plot_bar["NAMA KPI"]))
                bar_width = 0.35
                rects1 = ax_bar.bar(index - bar_width/2, df_plot_bar["TARGET TW TERKAIT"], bar_width, label='Target')
                rects2 = ax_bar.bar(index + bar_width/2, df_plot_bar["REALISASI TW TERKAIT"], bar_width, label='Realisasi')
                ax_bar.set_xlabel("Nama KPI")
                ax_bar.set_ylabel("Nilai")
                ax_bar.set_title("Realisasi vs Target KPI")
                ax_bar.set_xticks(index)
                ax_bar.set_xticklabels(df_plot_bar["NAMA KPI"], rotation=45, ha="right")
                ax_bar.legend()
                st.pyplot(fig_bar)
            else:
                st.write("Tidak cukup data untuk menampilkan grafik perbandingan.")

            if 'SKOR PENCAPAIAN' in df_filtered.columns:
                st.write("Distribusi Skor Pencapaian KPI")
                fig_hist, ax_hist = plt.subplots()
                sns.histplot(df_filtered["SKOR PENCAPAIAN"], kde=True, ax=ax_hist, bins=15)
                ax_hist.set_title("Distribusi Skor Pencapaian")
                ax_hist.set_xlabel("Skor Pencapaian (%)")
                ax_hist.set_ylabel("Frekuensi")
                st.pyplot(fig_hist)
            else:
                st.write("Kolom 'SKOR PENCAPAIAN' tidak ditemukan untuk membuat histogram.")

    elif app_mode == "Prediksi Kinerja KPI":
        st.header("Prediksi Kinerja KPI")
        if df_kpi_history.empty:
            st.warning("Data historis sintetis tidak dapat dibuat. Fitur prediksi tidak tersedia.")
        else:
            st.sidebar.header("Filter Data Prediksi")
            kpi_options_pred = sorted(df_kpi_history['NAMA KPI'].unique())
            selected_kpi_for_pred = st.sidebar.selectbox("Pilih KPI untuk Prediksi:", kpi_options_pred, key='pred_kpi_select')

            if selected_kpi_for_pred:
                df_hist_selected = df_kpi_history[df_kpi_history['NAMA KPI'] == selected_kpi_for_pred].copy()
                df_hist_selected['PERIODE_NUM'] = df_hist_selected['PERIODE'].apply(lambda x: int(x.split('-')[1]))
                
                X = df_hist_selected[['PERIODE_NUM']]
                y = df_hist_selected['REALISASI TW TERKAIT']

                model = LinearRegression()
                model.fit(X, y)

                future_periods_num = np.array([X['PERIODE_NUM'].max() + i for i in range(1, 4)]).reshape(-1, 1)
                future_predictions = model.predict(future_periods_num)
                future_periods_labels = [f"T+{i}" for i in range(1,4)]

                st.subheader(f"Prediksi untuk KPI: {selected_kpi_for_pred}")
                fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
                ax_pred.plot(df_hist_selected['PERIODE_NUM'], y, label='Data Historis (Sintetis)', marker='o')
                ax_pred.plot(future_periods_num.flatten(), future_predictions, label='Prediksi', marker='x', linestyle='--')
                
                all_period_nums = list(X['PERIODE_NUM'].values) + list(future_periods_num.flatten())
                all_period_labels = list(df_hist_selected['PERIODE'].values) + future_periods_labels
                
                ax_pred.set_xticks(all_period_nums)
                ax_pred.set_xticklabels(all_period_labels, rotation=45, ha="right")
                
                ax_pred.set_xlabel("Periode")
                ax_pred.set_ylabel("Realisasi KPI")
                ax_pred.set_title(f"Tren Historis dan Prediksi untuk {selected_kpi_for_pred}")
                ax_pred.legend()
                st.pyplot(fig_pred)

                st.write("Detail Prediksi:")
                pred_data = {'Periode': future_periods_labels, 'Nilai Prediksi': np.round(future_predictions,2)}
                st.dataframe(pd.DataFrame(pred_data))
            else:
                st.write("Silakan pilih KPI dari sidebar untuk melihat prediksi.")

    elif app_mode == "Deteksi Anomali KPI":
        st.header("Deteksi Anomali KPI")
        st.sidebar.header("Pengaturan Deteksi Anomali")
        
        # Select column for anomaly detection
        anomaly_column_options = ['REALISASI TW TERKAIT', 'SKOR PENCAPAIAN']
        if 'SKOR PENCAPAIAN' not in df_kpi.columns:
            anomaly_column_options = ['REALISASI TW TERKAIT']
            
        selected_anomaly_col = st.sidebar.selectbox(
            "Pilih Kolom untuk Deteksi Anomali:", 
            anomaly_column_options,
            key='anomaly_col_select'
        )
        
        z_threshold = st.sidebar.slider("Threshold Z-score untuk Anomali:", 1.0, 4.0, 3.0, 0.1, key='z_thresh_slider')

        if selected_anomaly_col:
            st.subheader(f"Deteksi Anomali pada Kolom: {selected_anomaly_col}")
            # Make a copy to avoid SettingWithCopyWarning
            df_anomaly_check = df_kpi.copy()
            df_anomaly_check['Z_SCORE'] = np.abs(stats.zscore(df_anomaly_check[selected_anomaly_col]))
            df_anomalies = df_anomaly_check[df_anomaly_check['Z_SCORE'] > z_threshold]

            if df_anomalies.empty:
                st.success(f"Tidak ditemukan anomali signifikan pada kolom '{selected_anomaly_col}' dengan threshold Z-score > {z_threshold}.")
            else:
                st.warning(f"Ditemukan {len(df_anomalies)} anomali pada kolom '{selected_anomaly_col}' dengan threshold Z-score > {z_threshold}:")
                st.dataframe(df_anomalies[['NIPP PEKERJA', 'POSISI PEKERJA', 'NAMA KPI', selected_anomaly_col, 'Z_SCORE']])

                # Visualization of anomalies
                st.write(f"Visualisasi Distribusi {selected_anomaly_col} dengan Anomali Teridentifikasi")
                fig_anomaly, ax_anomaly = plt.subplots(figsize=(10, 6))
                sns.histplot(df_anomaly_check[selected_anomaly_col], ax=ax_anomaly, kde=True, label='Distribusi Normal', color='blue')
                sns.scatterplot(data=df_anomalies, x=selected_anomaly_col, y=[0]*len(df_anomalies), color='red', s=100, label=f'Anomali (Z > {z_threshold})', ax=ax_anomaly)
                ax_anomaly.set_title(f"Distribusi {selected_anomaly_col} dan Anomali")
                ax_anomaly.set_xlabel(selected_anomaly_col)
                ax_anomaly.set_ylabel("Frekuensi / Densitas")
                ax_anomaly.legend()
                st.pyplot(fig_anomaly)
        else:
            st.write("Silakan pilih kolom dari sidebar untuk melakukan deteksi anomali.")

