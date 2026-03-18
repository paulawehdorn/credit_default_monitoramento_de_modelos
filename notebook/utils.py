"""
utils.py — Funções auxiliares de visualização e análise para monitoramento de modelos de crédito.

Importado pelo notebook principal (mm_trab_refatorado.ipynb) para manter
o notebook focado na análise, sem código de suporte misturado.

Funções disponíveis:
    - plot_distribuicao_por_grupos()  : distribuição categórica ao longo do tempo (3 grupos)
    - plot_estatisticas_por_tipo()    : média / mediana / CV de variável numérica por tipo
    - calcular_ks()                   : estatística KS via scipy
    - calcular_fpd()                  : taxa de FPD por faixa de score
    - calcular_woe_iv()               : WOE e IV para uma variável
    - plot_woe_analysis()             : visualização da análise WOE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# ------------------------------------------------------------------
# Defaults das janelas temporais e paleta de cores
# (espelham as constantes da célula de config do notebook)
# ------------------------------------------------------------------
_JANELAS_DEFAULT = {
    'modelagem_inicio' : '2022-03-01',
    'observacao_inicio': '2023-03-01',
    'limbo_inicio'     : '2024-03-01',
    'producao_inicio'  : '2024-05-01',
    'vazio_inicio'     : '2025-01-01',
    'spike_inicio'     : '2024-10-01',
}

_CORES_DEFAULT = {
    'modelagem' : 'lightgreen',
    'observacao': 'lightblue',
    'limbo'     : 'dimgray',
    'producao'  : 'lightcoral',
    'vazio'     : 'lightgray',
    'spike'     : 'orange',
}


# ==================================================================
# 1. VISUALIZAÇÕES DE EDA
# ==================================================================

def plot_distribuicao_por_grupos(
        df_contratado, df_adimplente, df_inadimplente, col,
        titulo_legenda=None, janelas=None, cores=None):
    """
    Plota a distribuição de uma variável categórica ao longo do tempo
    para três grupos: Contratados, Adimplentes e Inadimplentes.

    Linhas são interrompidas onde os valores são zero.
    Todos os subplots usam a mesma escala no eixo y.
    """
    j = {**_JANELAS_DEFAULT, **(janelas or {})}
    c = {**_CORES_DEFAULT,   **(cores   or {})}

    grupos = [
        (df_contratado,  'Contratados Total'),
        (df_adimplente,  'Adimplentes'),
        (df_inadimplente,'Inadimplentes'),
    ]

    tabelas = []
    for df_g, _ in grupos:
        t = pd.crosstab(df_g['ano_mes'], df_g[col], normalize='index').replace(0, np.nan)
        tabelas.append(t)

    ylim_max = min(
        max((t.max().max() for t in tabelas if not t.empty), default=0) * 1.1,
        1.0
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Distribuição de {col} ao longo do tempo', fontsize=16)

    for ax, (tabela, titulo), (df_g, _) in zip(axes, grupos, grupos):
        tabela.plot(ax=ax, legend=False, linewidth=2)

        # Usar pd.Timestamp para consistência com o eixo datetime criado por tabela.plot()
        tmax = pd.Timestamp(tabela.index.max())
        ts   = {k: pd.Timestamp(v) for k, v in j.items()}

        ax.axvspan(ts['modelagem_inicio'],  ts['observacao_inicio'], color=c['modelagem'],  alpha=0.3)
        ax.axvspan(ts['observacao_inicio'], ts['limbo_inicio'],       color=c['observacao'], alpha=0.3)
        ax.axvspan(ts['limbo_inicio'],      ts['producao_inicio'],    color=c['limbo'],      alpha=0.4)
        ax.axvspan(ts['producao_inicio'],   tmax,                     color=c['producao'],   alpha=0.3)
        ax.axvspan(ts['spike_inicio'],      ts['vazio_inicio'],       color=c['spike'],      alpha=0.4)
        ax.axvspan(ts['vazio_inicio'],      tmax,                     color=c['vazio'],      alpha=0.4)

        ax.set_title(titulo)
        ax.set_xlabel('Ano-Mês')
        ax.set_ylabel('Proporção %')
        ax.set_ylim(0, ylim_max)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.tick_params(axis='x', rotation=45)

    lines  = axes[0].get_lines()
    labels = [ln.get_label() for ln in lines]
    fig.legend(lines, labels,
               loc='center', bbox_to_anchor=(0.5, -0.05),
               ncol=len(labels), title=titulo_legenda or f'Valores de {col}',
               columnspacing=1.5)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    return fig, axes


def plot_estatisticas_por_tipo(df, col_numerica, titulo=None, janelas=None, cores=None):
    """
    Plota média, mediana e coeficiente de variação de uma variável numérica
    ao longo do tempo para cada tipo (Adimplente / Inadimplente / Nao_Contratado).
    """
    j = {**_JANELAS_DEFAULT, **(janelas or {})}
    c = {**_CORES_DEFAULT,   **(cores   or {})}

    df_stats = (df.groupby(['ano_mes', 'tipo'])[col_numerica]
                  .agg(media='mean', mediana='median', std='std')
                  .reset_index())
    df_stats['cv'] = df_stats['std'] / df_stats['media']

    ordem_tipos = ['Adimplente', 'Inadimplente', 'Nao_Contratado']
    cor_tipo    = {'Nao_Contratado': '#95A5A6', 'Adimplente': '#2ECC71', 'Inadimplente': '#E74C3C'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(titulo or f'Estatísticas de {col_numerica} ao longo do tempo', fontsize=16)

    for ax, (estat, rotulo) in zip(axes, [
            ('media',   'Média'),
            ('mediana', 'Mediana'),
            ('cv',      'Coeficiente de Variação (σ/μ)')]):

        for tipo in ordem_tipos:
            sub = df_stats[df_stats['tipo'] == tipo]
            if not sub.empty:
                ax.plot(sub['ano_mes'], sub[estat],
                        'o-', label=tipo, color=cor_tipo.get(tipo, '#333'),
                        linewidth=2.5, markersize=6)

        tmax = pd.Timestamp(df['ano_mes'].max())
        ts   = {k: pd.Timestamp(v) for k, v in j.items()}

        ax.axvspan(ts['modelagem_inicio'],  ts['observacao_inicio'], color=c['modelagem'],  alpha=0.3)
        ax.axvspan(ts['observacao_inicio'], ts['limbo_inicio'],       color=c['observacao'], alpha=0.3)
        ax.axvspan(ts['limbo_inicio'],      ts['producao_inicio'],    color=c['limbo'],      alpha=0.4)
        ax.axvspan(ts['producao_inicio'],   tmax,                     color=c['producao'],   alpha=0.3)
        ax.axvspan(ts['spike_inicio'],      ts['vazio_inicio'],       color=c['spike'],      alpha=0.4)
        ax.axvspan(ts['vazio_inicio'],      tmax,                     color=c['vazio'],      alpha=0.4)

        if estat == 'cv':
            ax.set_ylabel('Coeficiente de Variação')
            ax.axhline(0.2, color='red', linestyle='--', alpha=0.5, lw=1, label='CV=0.2')
        else:
            ax.set_ylabel(f'{rotulo} de {col_numerica}')

        ax.set_xlabel('Ano-Mês')
        ax.set_title(rotulo, fontsize=13)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    return fig, axes


# ==================================================================
# 2. MÉTRICAS DE MONITORAMENTO
# ==================================================================

def calcular_ks(y_target, y_feature):
    """
    Calcula a estatística KS entre distribuições de bons e maus pagadores.

    Parameters
    ----------
    y_target  : array-like  — 0 = adimplente, 1 = inadimplente
    y_feature : array-like  — score ou probabilidade do modelo

    Returns
    -------
    ks_statistic : float
    p_value      : float
    """
    scores_adimplentes   = y_feature[y_target == 0]
    scores_inadimplentes = y_feature[y_target == 1]
    ks_statistic, p_value = ks_2samp(scores_inadimplentes, scores_adimplentes)
    return ks_statistic, p_value


def calcular_fpd(df, col_contrato='contratado', col_fpd='FPD',
                 col_score='score', score_minimo=4):
    """
    Calcula taxa de FPD (First Payment Default) por faixa de score.

    Filtra apenas contratos efetivados e com score ativo (> score_minimo).

    Returns
    -------
    DataFrame com colunas: score, total_contratos, total_fpd, taxa_fpd
    """
    contratados = df[(df[col_contrato] == 1) & (df[col_score] > score_minimo)].copy()
    contratados['fpd_flag'] = contratados[col_fpd].fillna(0)

    resultado = (contratados.groupby(col_score)
                             .agg(total_contratos=('fpd_flag', 'count'),
                                  total_fpd=('fpd_flag', 'sum'),
                                  taxa_fpd=('fpd_flag', 'mean'))
                             .round(4)
                             .reset_index())
    return resultado


def calcular_woe_iv(df, var, target='target'):
    """
    Calcula WOE (Weight of Evidence) e IV (Information Value) para uma variável.

    Parameters
    ----------
    df     : DataFrame com a variável e target
    var    : nome da variável (categórica ou binned)
    target : nome da coluna target (0 = adimplente, 1 = inadimplente)

    Returns
    -------
    resultado : DataFrame com WOE e IV por categoria
    iv_total  : float — IV total da variável
    """
    tabela = pd.crosstab(df[var], df[target], margins=True, margins_name='Total')
    tabela.columns = ['Adimplentes', 'Inadimplentes', 'Total']

    total_adim  = tabela.loc['Total', 'Adimplentes']
    total_inad  = tabela.loc['Total', 'Inadimplentes']
    total_geral = tabela.loc['Total', 'Total']

    resultado = tabela.drop('Total').copy()
    resultado['%_Adimplentes']   = resultado['Adimplentes']   / total_adim
    resultado['%_Inadimplentes'] = resultado['Inadimplentes'] / total_inad
    resultado['WOE'] = np.log(
        (resultado['%_Inadimplentes'] + 1e-10) /
        (resultado['%_Adimplentes']   + 1e-10)
    )
    resultado['IV_contrib'] = (resultado['%_Inadimplentes'] - resultado['%_Adimplentes']) * resultado['WOE']

    iv_total = resultado['IV_contrib'].sum()
    resultado.loc['Total'] = [total_adim, total_inad, total_geral, 1.0, 1.0, np.nan, iv_total]

    return resultado, iv_total


def plot_woe_analysis(df, var, target='target', titulo=None):
    """
    Visualiza análise WOE para uma variável: distribuição por classe + barras WOE.

    Returns
    -------
    woe_df   : DataFrame completo com WOE/IV
    iv_total : float
    """
    woe_df, iv_total = calcular_woe_iv(df, var, target)
    woe_plot = woe_df.drop('Total').copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Distribuição por classe ---
    ax1 = axes[0]
    x, w = range(len(woe_plot.index)), 0.35
    ax1.bar([i - w/2 for i in x], woe_plot['%_Adimplentes'],   w, label='Adimplentes',   color='#2ECC71', alpha=0.7)
    ax1.bar([i + w/2 for i in x], woe_plot['%_Inadimplentes'], w, label='Inadimplentes', color='#E74C3C', alpha=0.7)
    ax1.set_xlabel(var)
    ax1.set_ylabel('Proporção')
    ax1.set_title('Distribuição por Classe')
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(woe_plot.index, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # --- WOE ---
    ax2 = axes[1]
    cores_bar = ['#E74C3C' if w > 0 else '#2ECC71' for w in woe_plot['WOE']]
    ax2.bar(list(x), woe_plot['WOE'], color=cores_bar, alpha=0.8, edgecolor='black')
    ax2.axhline(0, color='black', lw=0.8)
    for i, (_, row) in enumerate(woe_plot.iterrows()):
        offset = 0.02 if row['WOE'] > 0 else -0.05
        ax2.text(i, row['WOE'] + offset, f'{row["WOE"]:.2f}', ha='center', fontsize=9)

    ax2.set_xlabel(var)
    ax2.set_ylabel('WOE')
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(woe_plot.index, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # Interpretação do IV
    if   iv_total < 0.02: poder, cor_iv = 'Não preditivo', 'red'
    elif iv_total < 0.10: poder, cor_iv = 'Fraco',         'orange'
    elif iv_total < 0.30: poder, cor_iv = 'Médio',         'steelblue'
    else:                 poder, cor_iv = 'Forte',         'green'

    ax2.set_title(f'Perfil WOE — IV Total: {iv_total:.4f} ({poder})')
    ax2.text(0.5, -0.18, f'IV = {iv_total:.4f} — Poder preditivo: {poder}',
             transform=ax2.transAxes, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor=cor_iv, alpha=0.25))

    plt.suptitle(titulo or f'Análise WOE — {var}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return woe_df, iv_total
