import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime

# Configuração da Página
st.set_page_config(page_title="Otimizador de Carteira - Markowitz", layout="wide")

st.title("🎯 Otimização de Portfólio: Fronteira Eficiente")
st.markdown("Análise baseada na Teoria Moderna de Portfólio para os ativos selecionados.")

# --- SIDEBAR DE CONFIGURAÇÕES ---
st.sidebar.header("Parâmetros do Modelo")
risk_free_rate = st.sidebar.number_input("Taxa Livre de Risco (Selic) % a.a.", value=10.75) / 100
rf_monthly = (1 + risk_free_rate)**(1/12) - 1
start_date = st.sidebar.date_input("Data de Início da Análise", value=datetime(2024, 1, 1))
num_portfolios = st.sidebar.slider("Simulações de Monte Carlo", 1000, 10000, 5000)

# --- FUNÇÃO DE PROCESSAMENTO DE DADOS ---
@st.cache_data
def load_data():
    # 1. Ativos de Mercado via Yahoo Finance
    tickers = ["IVVB11.SA", "XFIX11.SA"] # XFIX11 como proxy de IFIX
    market_data = yf.download(tickers, start=start_date, interval="1mo")['Adj Close'].pct_change().dropna()
    market_data.columns = ['IVVB11', 'IFIX']

    # 2. Dados Hardcoded dos Fundos (Mensal)
    # 2024 Completo + 2025 Jan-Nov
    ret_tarpon = [-0.0606, 0.0498, -0.0260, 0.0224, -0.0234, 0.0020, 0.0208, 0.0395, -0.0020, 0.0052, -0.0295, -0.0096,
                  0.0449, 0.0257, 0.0747, 0.0747, 0.0035, 0.0159, -0.0695, 0.0357, 0.0049, 0.0228, 0.1063]
    
    ret_btg = [0.0081, 0.0095, 0.0113, 0.0113, 0.0122, 0.0148, 0.0131, 0.0150, 0.0233, 0.0048, 0.0106, 0.0100,
               0.0105, 0.0105, 0.0105, 0.0105, 0.0105, 0.0105, 0.0105, 0.0105, 0.0105, 0.0105, 0.0105]

    # Criar range de datas para os fundos
    dates = pd.date_range(start="2024-01-01", periods=len(ret_tarpon), freq="MS")
    
    funds_df = pd.DataFrame({
        'Tarpon GT 90': ret_tarpon,
        'BTG Cred Corp': ret_btg,
        'Selic Pós': [0.0085] * len(ret_tarpon) # Proxy de liquidez estável
    }, index=dates)

    # Consolidar DataFrames
    combined_df = pd.concat([market_data, funds_df], axis=1).dropna()
    return combined_df

df_returns = load_data()

# --- LÓGICA DE OTIMIZAÇÃO ---
def get_portfolio_stats(weights, returns):
    weights = np.array(weights)
    port_return = np.sum(returns.mean() * weights) * 12
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 12, weights)))
    sharpe = (port_return - risk_free_rate) / port_vol
    return port_return, port_vol, sharpe

def minimize_sharpe(weights, returns):
    return -get_portfolio_stats(weights, returns)[2]

# Constraints: soma dos pesos = 1 e limites entre 0 e 1
num_assets = len(df_returns.columns)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
init_guess = num_assets * [1. / num_assets]

optimized = minimize(minimize_sharpe, init_guess, args=(df_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = optimized.x

# --- DASHBOARD LAYOUT ---
tab1, tab2, tab3 = st.tabs(["📊 Alocação & Risco", "📈 Fronteira Eficiente", "🕒 Backtest Simulado"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Carteira de Máximo Sharpe")
        alloc_df = pd.DataFrame({'Ativo': df_returns.columns, 'Peso': best_weights})
        fig_pie = px.pie(alloc_df, values='Peso', names='Ativo', hole=0.4, color_discrete_sequence=px.colors.qualitative.T10)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Matriz de Correlação")
        corr_matrix = df_returns.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    # Simulação Monte Carlo
    port_returns = []
    port_vols = []
    for _ in range(num_portfolios):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        r, v, _ = get_portfolio_stats(w, df_returns)
        port_returns.append(r)
        port_vols.append(v)

    opt_ret, opt_vol, opt_sharpe = get_portfolio_stats(best_weights, df_returns)

    fig_frontier = go.Figure()
    fig_frontier.add_trace(go.Scatter(x=port_vols, y=port_returns, mode='markers', 
                                      marker=dict(color=(np.array(port_returns)-risk_free_rate)/np.array(port_vols), 
                                      colorscale='Viridis', showscale=True, title="Sharpe"), name="Portfólios"))
    fig_frontier.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode='markers', 
                                      marker=dict(color='gold', size=15, symbol='star'), name="Máximo Sharpe"))
    
    fig_frontier.update_layout(title="Fronteira Eficiente de Markowitz", xaxis_title="Volatilidade Anualizada", yaxis_title="Retorno Anualizado")
    st.plotly_chart(fig_frontier, use_container_width=True)
    
    st.success(f"**Estatísticas do Portfólio Ótimo:** Retorno: {opt_ret:.2%} | Volatilidade: {opt_vol:.2%} | Índice de Sharpe: {opt_sharpe:.2f}")

with tab3:
    st.subheader("Evolução Patrimonial (Base 100)")
    # Backtest simplificado baseado nos retornos históricos
    portfolio_monthly_ret = (df_returns * best_weights).sum(axis=1)
    cumulative_ret = (1 + portfolio_monthly_ret).cumprod() * 100
    
    fig_backtest = px.line(cumulative_ret, title="Simulação de Crescimento de R$ 100,00", labels={'value': 'Patrimônio', 'index': 'Data'})
    st.plotly_chart(fig_backtest, use_container_width=True)

    # Métricas de drawdown
    peak = cumulative_ret.cummax()
    drawdown = (cumulative_ret - peak) / peak
    st.error(f"Drawdown Máximo no período: {drawdown.min():.2%}")
