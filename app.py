import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# Configuração da página
st.set_page_config(page_title="Otimização de Portfólio", layout="wide")

# Função principal do app Streamlit
st.title('Otimização de Portfólio com Teoria Moderna de Portfólio (Markowitz)')

# Sidebar para configurações do usuário
st.sidebar.header('Configurações')
rf_annual = st.sidebar.number_input('Taxa Livre de Risco Anual (%)', min_value=0.0, max_value=20.0, value=10.0) / 100
start_date_input = st.sidebar.date_input('Data de Início da Análise', value=pd.to_datetime('2020-01-01'))

# CORREÇÃO: Converter data do seletor para Timestamp do Pandas
start_date = pd.to_datetime(start_date_input)

# Seção 1: Coleta e Processamento de Dados
# Baixar dados de mercado via yfinance
assets_market = ['IVVB11.SA', 'XFIX11.SA']
data_market = yf.download(assets_market, start=start_date, end=pd.to_datetime('today'))['Adj Close']

# Calcular retornos mensais
returns_market = data_market.pct_change().dropna().resample('M').apply(lambda x: (1 + x).prod() - 1)

# Preparar índice temporal mensal completo
end_date = pd.to_datetime('2025-11-30')
monthly_index = pd.date_range(start=start_date, end=end_date, freq='M')

# Ativo 3: Selic Pós-fixada
selic_returns = pd.Series(0.0085, index=monthly_index, name='Selic')

# Ativo 4: Fundo A (Tarpon GT 90 FIC FIA)
tarpon_2024 = np.array([-6.06, 4.98, -2.60, 2.24, -2.34, 0.20, 2.08, 3.95, -0.20, 0.52, -2.95, -0.96]) / 100
tarpon_2025 = np.array([4.49, 2.57, 7.47, 7.47, 0.35, 1.59, -6.95, 3.57, 0.49, 2.28, 10.63]) / 100
tarpon_real_data = np.concatenate([tarpon_2024, tarpon_2025])
tarpon_real_index = pd.date_range(start='2024-01-31', periods=len(tarpon_real_data), freq='M')
tarpon_real = pd.Series(tarpon_real_data, index=tarpon_real_index)

# Gerar dados sintéticos para Tarpon
tarpon_mean_monthly = 0.0203
tarpon_std_monthly = np.std(tarpon_real)
synth_limit = pd.to_datetime('2024-01-01')
synth_start = start_date if start_date < synth_limit else synth_limit
synth_end = pd.to_datetime('2023-12-31')

if synth_start <= synth_end:
    synth_index = pd.date_range(start=synth_start, end=synth_end, freq='M')
    tarpon_synth = np.random.normal(tarpon_mean_monthly, tarpon_std_monthly, len(synth_index))
    tarpon_synth_series = pd.Series(tarpon_synth, index=synth_index)
    tarpon_returns = pd.concat([tarpon_synth_series, tarpon_real]).reindex(monthly_index)
else:
    tarpon_returns = tarpon_real.reindex(monthly_index)

# Ativo 5: Fundo B (BTG Cred Corp Incentivado)
btg_2024 = np.array([0.81, 0.95, 1.13, 1.13, 1.22, 1.48, 1.31, 1.50, 2.33, 0.48, 1.06, 1.00]) / 100
btg_2025 = np.array([1.05] * 11) / 100
btg_real_data = np.concatenate([btg_2024, btg_2025])
btg_real_index = pd.date_range(start='2024-01-31', periods=len(btg_real_data), freq='M')
btg_real = pd.Series(btg_real_data, index=btg_real_index)

# Gerar dados sintéticos para BTG
btg_mean_monthly = 0.0105
btg_std_monthly = 0.0113 / np.sqrt(12)

if synth_start <= synth_end:
    btg_synth = np.random.normal(btg_mean_monthly, btg_std_monthly, len(synth_index))
    btg_synth_series = pd.Series(btg_synth, index=synth_index)
    btg_returns = pd.concat([btg_synth_series, btg_real]).reindex(monthly_index)
else:
    btg_returns = btg_real.reindex(monthly_index)

# Consolidar todos os retornos
returns = returns_market.reindex(monthly_index)
returns['Selic'] = selic_returns
returns['Tarpon'] = tarpon_returns
returns['BTG'] = btg_returns
returns = returns.dropna()

# Seção 2: Cálculos Estatísticos
mean_returns_annual = returns.mean() * 12
cov_annual = returns.cov() * 12

def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return, port_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (port_return - risk_free_rate) / port_std

# Seção 3: Otimização
num_assets = len(returns.columns)
initial_weights = [1.0 / num_assets] * num_assets
bounds = tuple((0, 1) for _ in range(num_assets))
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

opt_result = minimize(negative_sharpe, initial_weights, args=(mean_returns_annual, cov_annual, rf_annual),
                      method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_result.x

# Exibição dos Pesos
st.subheader('Alocação Ótima de Portfólio (Máximo Sharpe)')
df_weights = pd.DataFrame({'Ativo': returns.columns, 'Peso (%)': optimal_weights * 100})
st.table(df_weights.set_index('Ativo').style.format("{:.2f}%"))

# Seção 4: Visualizações
col1, col2 = st.columns(2)

with col1:
    # 4.1: Fronteira Eficiente
    def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            w = np.random.dirichlet(np.ones(len(mean_returns)))
            ret, std = portfolio_performance(w, mean_returns, cov_matrix)
            results[0, i] = std
            results[1, i] = ret
            results[2, i] = (ret - risk_free_rate) / std
        return results

    random_results = generate_random_portfolios(5000, mean_returns_annual, cov_annual, rf_annual)
    fig_frontier = go.Figure()
    fig_frontier.add_trace(go.Scatter(x=random_results[0, :], y=random_results[1, :], mode='markers',
                                      marker=dict(color=random_results[2, :], colorscale='Viridis', size=5, showscale=True,
                                      colorbar=dict(title='Sharpe'))))
    
    opt_return, opt_std = portfolio_performance(optimal_weights, mean_returns_annual, cov_annual)
    fig_frontier.add_trace(go.Scatter(x=[opt_std], y=[opt_return], mode='markers',
                                      marker=dict(color='red', symbol='star', size=15), name='Máximo Sharpe'))
    
    fig_frontier.update_layout(title='Fronteira Eficiente', xaxis_title='Risco (Volatilidade)', yaxis_title='Retorno')
    st.plotly_chart(fig_frontier, use_container_width=True)

with col2:
    # 4.3: Gráfico de Pizza
    fig_pie = px.pie(df_weights, values='Peso (%)', names='Ativo', title='Alocação Ideal (%)', hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

# 4.2: Heatmap
st.subheader('Correlação entre Ativos')
selected_assets = ['Tarpon', 'BTG', 'IVVB11.SA', 'XFIX11.SA']
corr_matrix = returns[selected_assets].corr()
fig_heatmap = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r')
st.plotly_chart(fig_heatmap, use_container_width=True)
