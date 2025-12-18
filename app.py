import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# Função principal do app Streamlit
st.title('Otimização de Portfólio com Teoria Moderna de Portfólio (Markowitz)')

# Sidebar para configurações do usuário
st.sidebar.header('Configurações')
rf_annual = st.sidebar.number_input('Taxa Livre de Risco Anual (%)', min_value=0.0, max_value=20.0, value=10.0) / 100
start_date = st.sidebar.date_input('Data de Início da Análise', value=pd.to_datetime('2020-01-01'))

# Seção 1: Coleta e Processamento de Dados
# Baixar dados de mercado via yfinance (IVVB11.SA e XFIX11.SA como proxy para IFIX)
assets_market = ['IVVB11.SA', 'XFIX11.SA']  # Usando XFIX11.SA como proxy para IFIX, conforme sugestão
data_market = yf.download(assets_market, start=start_date, end=pd.to_datetime('today'))['Adj Close']
# Calcular retornos mensais: pct_change diário, então aggregar para mensal (retorno composto)
returns_market = data_market.pct_change().dropna().resample('M').apply(lambda x: (1 + x).prod() - 1)

# Preparar índice temporal mensal completo a partir da data de início até novembro/2025 (data atual simulada)
end_date = pd.to_datetime('2025-11-30')  # Até novembro/2025, conforme dados fornecidos
monthly_index = pd.date_range(start=start_date, end=end_date, freq='M')

# Ativo 3: Selic Pós-fixada - Simulação constante com retorno mensal de 0.85% (volatilidade ~0)
selic_returns = pd.Series(0.0085, index=monthly_index, name='Selic')

# Ativo 4: Fundo A (Tarpon GT 90 FIC FIA) - Dados hardcoded reais + sintéticos para histórico
# Dados reais fornecidos (em % - converter para decimal)
tarpon_2024 = np.array([-6.06, 4.98, -2.60, 2.24, -2.34, 0.20, 2.08, 3.95, -0.20, 0.52, -2.95, -0.96]) / 100
tarpon_2025 = np.array([4.49, 2.57, 7.47, 7.47, 0.35, 1.59, -6.95, 3.57, 0.49, 2.28, 10.63]) / 100
tarpon_real_data = np.concatenate([tarpon_2024, tarpon_2025])
tarpon_real_index = pd.date_range(start='2024-01-31', periods=len(tarpon_real_data), freq='M')
tarpon_real = pd.Series(tarpon_real_data, index=tarpon_real_index)

# Gerar dados sintéticos para período anterior a 2024 (baseado em média 2.03% e std dos dados reais)
tarpon_mean_monthly = 0.0203
tarpon_std_monthly = np.std(tarpon_real)  # Usar std dos dados reais para realismo
synth_start = start_date if start_date < pd.to_datetime('2024-01-01') else pd.to_datetime('2024-01-01')
synth_end = pd.to_datetime('2023-12-31')
synth_index = pd.date_range(start=synth_start, end=synth_end, freq='M')
num_synth_months = len(synth_index)
if num_synth_months > 0:
    tarpon_synth = np.random.normal(tarpon_mean_monthly, tarpon_std_monthly, num_synth_months)
    tarpon_synth_series = pd.Series(tarpon_synth, index=synth_index)
    tarpon_returns = pd.concat([tarpon_synth_series, tarpon_real]).reindex(monthly_index)
else:
    tarpon_returns = tarpon_real.reindex(monthly_index)

# Ativo 5: Fundo B (BTG Cred Corp Incentivado) - Dados hardcoded reais + sintéticos
# Dados reais 2024 (em % - converter para decimal)
btg_2024 = np.array([0.81, 0.95, 1.13, 1.13, 1.22, 1.48, 1.31, 1.50, 2.33, 0.48, 1.06, 1.00]) / 100
# Para 2025: Usar média constante de 1.05% mensal (baixa volatilidade)
btg_2025 = np.array([1.05] * 11) / 100  # Jan a Nov/2025
btg_real_data = np.concatenate([btg_2024, btg_2025])
btg_real_index = pd.date_range(start='2024-01-31', periods=len(btg_real_data), freq='M')
btg_real = pd.Series(btg_real_data, index=btg_real_index)

# Gerar dados sintéticos para período anterior (média 1.05%, volatilidade mensal baseada em anual 1.13%)
btg_mean_monthly = 0.0105
btg_std_annual = 0.0113
btg_std_monthly = btg_std_annual / np.sqrt(12)
synth_index = pd.date_range(start=synth_start, end=synth_end, freq='M')
num_synth_months = len(synth_index)
if num_synth_months > 0:
    btg_synth = np.random.normal(btg_mean_monthly, btg_std_monthly, num_synth_months)
    btg_synth_series = pd.Series(btg_synth, index=synth_index)
    btg_returns = pd.concat([btg_synth_series, btg_real]).reindex(monthly_index)
else:
    btg_returns = btg_real.reindex(monthly_index)

# Consolidar todos os retornos em um único DataFrame mensal
returns = returns_market.reindex(monthly_index)
returns['Selic'] = selic_returns
returns['Tarpon'] = tarpon_returns
returns['BTG'] = btg_returns
returns = returns.dropna()  # Remover qualquer NaN (embora sintéticos cubram)

# Seção 2: Cálculos Estatísticos
# Retornos médios e matriz de covariância mensais
mean_returns_monthly = returns.mean()
cov_monthly = returns.cov()

# Anualizar para consistência no Sharpe (Teoria Markowitz usa métricas anualizadas tipicamente)
mean_returns_annual = mean_returns_monthly * 12
cov_annual = cov_monthly * 12

# Funções auxiliares para performance de portfólio
def portfolio_performance(weights, mean_returns, cov_matrix):
    # Calcula retorno e risco (std) anualizados do portfólio
    port_return = np.dot(weights, mean_returns)
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    # Função a minimizar: -Sharpe Ratio (para maximizar Sharpe)
    port_return, port_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (port_return - risk_free_rate) / port_std

# Seção 3: Otimização de Portfólio (Maximizar Sharpe com restrições)
num_assets = len(returns.columns)
initial_weights = [1.0 / num_assets] * num_assets
bounds = tuple((0, 1) for _ in range(num_assets))  # Pesos entre 0 e 1 (sem short/alavancagem)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Soma pesos = 1

# Otimização usando scipy.optimize (SLSQP para problemas com restrições)
opt_result = minimize(negative_sharpe, initial_weights, args=(mean_returns_annual, cov_annual, rf_annual),
                      method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_result.x

# Exibir pesos ótimos
st.subheader('Alocação Ótima de Portfólio (Máximo Sharpe)')
st.write(pd.DataFrame({'Ativo': returns.columns, 'Peso (%)': optimal_weights * 100}).set_index('Ativo'))

# Seção 4: Visualizações com Plotly
# 4.1: Fronteira Eficiente com 5.000 portfólios aleatórios
def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))  # [std, return, sharpe]
    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(len(mean_returns)))  # Weights >0, sum=1
        port_return, port_std = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = port_std
        results[1, i] = port_return
        results[2, i] = (port_return - risk_free_rate) / port_std
    return results

num_portfolios = 5000
random_results = generate_random_portfolios(num_portfolios, mean_returns_annual, cov_annual, rf_annual)

fig_frontier = go.Figure()
fig_frontier.add_trace(go.Scatter(x=random_results[0, :], y=random_results[1, :], mode='markers',
                                  marker=dict(color=random_results[2, :], colorscale='Viridis', size=5, showscale=True,
                                              colorbar=dict(title='Sharpe Ratio'))))
# Destaque o portfólio ótimo com estrela dourada
opt_return, opt_std = portfolio_performance(optimal_weights, mean_returns_annual, cov_annual)
fig_frontier.add_trace(go.Scatter(x=[opt_std], y=[opt_return], mode='markers',
                                  marker=dict(color='gold', symbol='star', size=15), name='Máximo Sharpe'))
fig_frontier.update_layout(title='Fronteira Eficiente de Portfólios', xaxis_title='Risco Anualizado (Std Dev)',
                           yaxis_title='Retorno Anualizado Esperado', showlegend=True)
st.plotly_chart(fig_frontier)

# 4.2: Heatmap de Correlação (apenas Tarpon, BTG, IVVB11, IFIX/XFIX11 - conforme especificado)
corr_matrix = returns.corr()
selected_assets = ['Tarpon', 'BTG', 'IVVB11.SA', 'XFIX11.SA']
fig_heatmap = px.imshow(corr_matrix.loc[selected_assets, selected_assets].values,
                        x=selected_assets, y=selected_assets, color_continuous_scale='RdBu_r',
                        labels=dict(color='Correlação'), text_auto='.2f')
fig_heatmap.update_layout(title='Matriz de Correlação entre Ativos Selecionados')
st.plotly_chart(fig_heatmap)

# 4.3: Gráfico de Pizza para Alocação Ideal
fig_pie = px.pie(values=optimal_weights, names=returns.columns, title='Alocação Ideal de Ativos (%)',
                 hole=0.3)  # Donut style para melhor visual
st.plotly_chart(fig_pie)
