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
st.markdown("Análise baseada na Teoria Moderna de Portfólio (Markowitz) utilizando dados reais e extraídos.")

# --- SIDEBAR DE CONFIGURAÇÕES ---
st.sidebar.header("Parâmetros do Modelo")
risk_free_rate = st.sidebar.number_input("Taxa Livre de Risco (Selic) % a.a.", value=10.75) / 100
start_date = st.sidebar.date_input("Data de Início da Análise", value=datetime(2024, 1, 1))
num_portfolios = st.sidebar.slider("Simulações de Monte Carlo", 1000, 10000, 5000)

# --- FUNÇÃO DE PROCESSAMENTO DE DADOS ---
@st.cache_data
def load_data(start_dt):
    # 1. Ativos de Mercado (Ajuste para evitar KeyError: 'Adj Close')
    tickers = ["IVVB11.SA", "XFIX11.SA"]
    # Usamos auto_adjust=True para que a coluna 'Close' já seja o valor ajustado
    data = yf.download(tickers, start=start_dt, interval="1mo", auto_adjust=True)
    
    # Selecionamos a coluna 'Close' e limpamos o MultiIndex se necessário
    market_data = data['Close'].pct_change().dropna()
    market_data.columns = ['IFIX', 'IVVB11'] # Ordem alfabética do yfinance

    # 2. Dados dos Fundos (Extraídos dos PDFs)
    # 2024 (Jan-Dez) + 2025 (Jan-Nov)
    ret_tarpon = [-0.0606, 0.0498, -0.0260, 0.0224, -0.0234, 0.0020, 0.0208, 0.0395, -0.0020, 0.0052, -0.0295, -0.0096,
                  0.0449, 0.0257, 0.0747, 0.0747, 0.0035, 0.0159, -0.0695, 0.0357, 0.0049, 0.0228, 0.1063]
    
    ret_btg = [0.0081, 0.0095, 0.0113, 0.0113, 0.0122, 0.0148, 0.0131, 0.0150, 0.0233, 0.0048, 0.0106, 0.0100,
               0.0105, 0.0105, 0.0105, 0.0105, 0.0105, 0.0105, 0.0105, 0.0105, 0.0105, 0.0105, 0.0105]

    dates = pd.date_range(start="2024-01-01", periods=len(ret_tarpon), freq="MS")
    
    funds_df = pd.DataFrame({
        'Tarpon GT 90': ret_tarpon,
        'BTG Cred Corp': ret_btg,
        'Selic Pós': [0.0085] * len(ret_tarpon)
    }, index=dates)

    # Garantir que os índices de data sejam compatíveis para o join
    market_data.index = market_data.index.tz_localize(None).to_period('M').to_timestamp()
    funds_df.index = funds_df.index.to_period('M').to_timestamp()

    combined_df = pd.concat([market_data, funds_df], axis=1).dropna()
    return combined_df

# Carregamento dos dados
try:
    df_returns = load_data(start_date)
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

# --- CÁLCULOS DE MARKOWITZ ---
def get_portfolio_stats(weights, returns):
    weights = np.array(weights)
    port_return = np.sum(returns.mean() * weights) * 12
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 12, weights)))
    # Índice de Sharpe Anualizado
    sharpe = (port_return - risk_free_rate) / port_vol
    return port_return, port_vol, sharpe

def minimize_sharpe(weights, returns):
    return -get_portfolio_stats(weights, returns)[2]

# Configuração da Otimização
num_assets = len(df_returns.columns)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
init_guess = num_assets * [1. / num_assets]

optimized = minimize(minimize_sharpe, init_guess, args=(df_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = optimized.x

# --- INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📊 Alocação Sugerida", "📈 Fronteira Eficiente", "🕒 Histórico"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        alloc_df = pd.DataFrame({'Ativo': df_returns.columns, 'Peso': best_weights})
        alloc_df['Peso %'] = (alloc_df['Peso'] * 100).round(2)
        fig_pie = px.pie(alloc_df, values='Peso', names='Ativo', title="Composição Ótima (Max Sharpe)",
                         color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig_pie)
    
    with col2:
        st.write("### Tabela de Pesos")
        st.dataframe(alloc_df[['Ativo', 'Peso %']], hide_index=True)
        st.info("Esta alocação representa o ponto de tangência na fronteira eficiente.")

with tab2:
    # Simulação para plotar a fronteira
    prets, pvols = [], []
    for _ in range(num_portfolios):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        r, v, _ = get_portfolio_stats(w, df_returns)
        prets.append(r)
        pvols.append(v)
    
    opt_ret, opt_vol, opt_sharpe = get_portfolio_stats(best_weights, df_returns)
    
    fig_frontier = go.Figure()
    fig_frontier.add_trace(go.Scatter(x=pvols, y=prets, mode='markers', name='Portfólios Aleatórios',
                                      marker=dict(color=(np.array(prets)-risk_free_rate)/np.array(pvols), 
                                      colorscale='Viridis', showscale=True)))
    fig_frontier.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode='markers', name='Max Sharpe',
                                      marker=dict(color='red', size=15, symbol='star')))
    fig_frontier.update_layout(title="Fronteira Eficiente", xaxis_title="Risco (Volatilidade)", yaxis_title="Retorno Anualizado")
    st.plotly_chart(fig_frontier)

with tab3:
    # Simulação de Backtest
    portfolio_daily_ret = (df_returns * best_weights).sum(axis=1)
    cumulative_ret = (1 + portfolio_daily_ret).cumprod() * 100
    st.line_chart(cumulative_ret)
    st.write(f"**Retorno Acumulado no Período:** {(cumulative_ret.iloc[-1]-100):.2f}%")
