# functions/calc_auc.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
from . import calculate
from datetime import date
warnings.filterwarnings('ignore')


def split_and_reshape_data(data):
    results = []
    column_mapping = {
        'crr': ['crr', 'crr1', 'crr2', 'crr_av2', 'crr_av3'],
        'd_as': ['d_as', 'd_as1', 'd_as2', 'd_as_av2', 'd_as_av3'],
        'lvr': ['lvr', 'lvr1', 'lvr2', 'lvr_av2', 'lvr_av3'],
        'dsr': ['dsr', 'dsr1', 'dsr2', 'dsr_av2', 'dsr_av3'],
        'ggr': ['ggr', 'ggr1', 'ggr2', 'ggr_av2', 'ggr_av3'],
        'invr': ['invr', 'invr1', 'invr2', 'invr_av2', 'invr_av3'],
        'd_srv': ['d_srv', 'd_srv1', 'd_srv2', 'd_srv_av2', 'd_srv_av3'],
        'l_gdp': ['l_gdp', 'l_gdp1', 'l_gdp2', 'l_gdp_av2', 'l_gdp_av3'],
        'exd_s': ['exd_s', 'exd_s1', 'exd_s2', 'exd_s_av2', 'exd_s_av3'],
        'bb_gdp': ['bb_gdp', 'bb_gdp1', 'bb_gdp2', 'bb_gdp_av2', 'bb_gdp_av3'],
        'ggd': ['ggd', 'ggd1', 'ggd2', 'ggd_av2', 'ggd_av3'],
        'ca': ['ca', 'ca1', 'ca2', 'ca_av2', 'ca_av3'],
        'gd_gdp': ['gd_gdp', 'gd_gdp1', 'gd_gdp2', 'gd_gdp_av2', 'gd_gdp_av3'],
        'cest': ['cest', 'cest1', 'cest2', 'cest_av2', 'cest_av3'],
        'pest': ['pest', 'pest1', 'pest2', 'pest_av2', 'pest_av3'],
        'rcgr_c': ['rcgr_c1', 'rcgr_c2', 'rcgr_c3'],
        'rcgr_h': ['rcgr_h1', 'rcgr_h2', 'rcgr_h3']
    }
    row_names = ['разрыв', 'прирост за год', 'прирост за 2 года', 'отклонение от среднего за 2 года', 'отклонение от среднего за 3 года']
    rcgr_names = ['прирост за год', 'прирост за 2 года', 'прирост за 3 года']
    dates = sorted(data['Date'].unique())
    for date in dates:
        date_data = data[data['Date'] == date].iloc[0]
        for root, columns in column_mapping.items():
            available_cols = [col for col in columns if col in data.columns]
            
            if root in ['rcgr_c', 'rcgr_h']:
                for i, col in enumerate(available_cols):
                    if i < 3:
                        results.append({
                            'Date': date,
                            'Indicator': root,
                            'Metric': rcgr_names[i],
                            'Value': date_data[col]
                        })
            else:
                for i, col in enumerate(available_cols):
                    if i < 5:
                        results.append({
                            'Date': date,
                            'Indicator': root,
                            'Metric': row_names[i],
                            'Value': date_data[col]
                        })
    result_df = pd.DataFrame(results)
    
    df_date1 = result_df[result_df['Date'] == dates[0]].pivot(index='Indicator', columns='Metric', values='Value')
    df_date2 = result_df[result_df['Date'] == dates[1]].pivot(index='Indicator', columns='Metric', values='Value')

    df_date1 = df_date1.reset_index()
    df_date2 = df_date2.reset_index()

    df_date1.columns.name = None
    df_date2.columns.name = None
    df_date1 = df_date1[['Indicator', 'разрыв', 'прирост за год', 'прирост за 2 года', 'прирост за 3 года', 'отклонение от среднего за 2 года', 'отклонение от среднего за 3 года']]
    df_date2 = df_date2[['Indicator', 'разрыв', 'прирост за год', 'прирост за 2 года', 'прирост за 3 года', 'отклонение от среднего за 2 года', 'отклонение от среднего за 3 года']]

    
    return df_date1, df_date2


def rasch_auc(df):
    dict_n = {
        'Corparate':['dsr', 'dsr1', 'dsr2', 'dsr_av2', 'dsr_av3', 
                 'ggr', 'ggr1', 'ggr2', 'ggr_av2', 'ggr_av3', 'rcgr_c1', 
                 'rcgr_c2', 'rcgr_c3', 'invr', 'invr1', 'invr2', 'invr_av2', 
                 'invr_av3'],
        'Households': ['d_srv', 'd_srv1', 'd_srv2', 'd_srv_av2', 'd_srv_av3', 
                 'l_gdp', 'l_gdp1', 'l_gdp2', 'l_gdp_av2', 'l_gdp_av3', 'rcgr_h1', 
                 'rcgr_h2', 'rcgr_h3'],
        'Goverment': ['exd_s', 'exd_s1', 'exd_s2', 'exd_s_av2', 
                 'exd_s_av3', 'bb_gdp', 'bb_gdp1', 'bb_gdp2', 'bb_gdp_av2', 'bb_gdp_av3', 
                 'ggd', 'ggd1', 'ggd2', 'ggd_av2', 'ggd_av3'],
        'Banks': ['crr', 'crr1', 'crr2', 'crr_av2', 'crr_av3', 'd_as', 'd_as1', 'd_as2', 
                 'd_as_av2', 'd_as_av3', 'lvr', 'lvr1', 'lvr2', 'lvr_av2', 
                 'lvr_av3'],
        'External': ['ca', 'ca1', 'ca2', 'ca_av2', 'ca_av3', 'gd_gdp', 
                 'gd_gdp1', 'gd_gdp2', 'gd_gdp_av2', 'gd_gdp_av3'],
        'Nedv': ['cest', 'cest1', 
                 'cest2', 'cest_av2', 'cest_av3', 'pest', 'pest1', 'pest2', 
                 'pest_av2', 'pest_av3']
    }
    
    max_values = df.max()
    max_values_processed = max_values.where(max_values > 0, 0.0)
    all_columns = max_values_processed.index.tolist()
    
    result_df = pd.DataFrame(index=['Max AUROC in indicator', 'Sum of max AUROC in Mj', 'Number of indicators in Mj', 
                                    'Average sum of max AUROC in Mj', 'Sum of average max in sector', 'Weight indicator in Mj',
                                    'Weight Mj in Sector', 'Weight indicator in Sector'], columns=all_columns)
    
    def extract_root(col):
        import re
        if col.startswith('rcgr_c'):
            return 'rcgr_c1'
        if col.startswith('rcgr_h'):
            return 'rcgr_h1'
        
        roots = ['dsr', 'ggr', 'invr', 'd_srv', 'l_gdp', 
                'exd_s', 'bb_gdp', 'ggd', 'crr', 'd_as', 'lvr', 'ca', 
                'gd_gdp', 'cest', 'pest']
        for root in roots:
            if root in col:
                return root
        return col
    
    df_cols = pd.DataFrame({
        'col': all_columns,
        'val': [max_values_processed[col] for col in all_columns],
        'root': [extract_root(col) for col in all_columns]
    })
    
    root_sums = df_cols.groupby('root')['val'].sum().to_dict()
    root_counts = df_cols.groupby('root')['val'].count().to_dict()
    
    for col in all_columns:
        result_df.loc['Max AUROC in indicator', col] = max_values_processed[col]
        
        root = extract_root(col)
        if col == root:
            result_df.loc['Sum of max AUROC in Mj', col] = root_sums[root]
            result_df.loc['Number of indicators in Mj', col] = root_counts[root]
            result_df.loc['Average sum of max AUROC in Mj', col] = root_sums[root] / root_counts[root] if root_counts[root] != 0.0 else 0.0
        
        result_df.loc['Weight indicator in Mj', col] = max_values_processed[col] / root_sums[root] if root_sums[root] != 0.0 else 0.0
    
    sector_roots = {}
    sector_avg_sums = {}
    
    for sector, cols in dict_n.items():
        roots_in = set()
        for c in cols:
            if c in all_columns:
                roots_in.add(extract_root(c))
        sector_roots[sector] = roots_in
        
        sector_avg_sum = 0
        for root in roots_in:
            sector_avg_sum += root_sums[root] / root_counts[root] if root_counts[root] != 0.0 else 0.0
        sector_avg_sums[sector] = sector_avg_sum
    
    for col in all_columns:
        root = extract_root(col)
        if col == root:
            for sector, roots in sector_roots.items():
                if root in roots:
                    result_df.loc['Sum of average max in sector', col] = sector_avg_sums[sector]
                    avg_root = root_sums[root] / root_counts[root] if root_counts[root] != 0.0 else 0.0
                    result_df.loc['Weight Mj in Sector', col] = avg_root / sector_avg_sums[sector] if sector_avg_sums[sector] != 0.0 else 0.0
    
    for col in all_columns:
        root = extract_root(col)
        if result_df.loc['Weight Mj in Sector', col] is None or pd.isna(result_df.loc['Weight Mj in Sector', col]):
            for sector, roots in sector_roots.items():
                if root in roots:
                    avg_root = root_sums[root] / root_counts[root] if root_counts[root] != 0.0 else 0.0
                    result_df.loc['Weight Mj in Sector', col] = avg_root / sector_avg_sums[sector] if sector_avg_sums[sector] != 0.0 else 0.0
                    break
    
    for col in all_columns:
        weight_indicator_mj = result_df.loc['Weight indicator in Mj', col]
        weight_mj_sector = result_df.loc['Weight Mj in Sector', col]
        if weight_indicator_mj is not None and weight_mj_sector is not None:
            result_df.loc['Weight indicator in Sector', col] = weight_indicator_mj * weight_mj_sector
    return result_df


def indicator_data__threshold(df, df1):
    if df is None or df.empty:
        df = df1.copy()
        start_q = pd.read_excel('Исходные данные.xlsx', sheet_name='prep_q', skiprows=1)
        start_q['Date'] = pd.to_datetime(start_q['Date']).dt.date
        df = pd.merge(df, start_q[['Date', 'bnr']], on='Date', how='left')
    else:
        df = pd.merge(df, df1, how='left', on='Date')
    return df


def find_auc(df, year):
    df = df.loc[(df['Date']>=date(2005, 1, 1)) & (df['Date']<=date(year-1, 10, 1))]
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    variables = [name for name in df.columns if name != 'Date' and name != 'bnr']
    lags = list(range(4, 13))
    n1_vectors = {}
    n2_vectors = {}
    z_vectors = {}
    auc_vectors = {}
    
    # 5. Основной цикл по переменным
    for var in variables:
        n1_vectors[var] = pd.Series(index=lags, dtype=float)
        n2_vectors[var] = pd.Series(index=lags, dtype=float)
        z_vectors[var] = pd.Series(index=lags, dtype=float)
        auc_vectors[var] = pd.Series(index=lags, dtype=float)
        
        # Обрабатываем каждый лаг
        for lag in lags:
            
            # 4. Создаем лаговую переменную
            lag_col = f'{var}_lag{lag:02d}'
            df[lag_col] = df[var].shift(lag)
            
            # Подготавливаем данные для регрессии
            model_data = df[['bnr', lag_col]].copy()
            model_data = model_data.dropna()
            
            if len(model_data) < 20:
                print(f"Недостаточно данных ({len(model_data)} наблюдений)")
                continue
            
            # 4. Строим логистическую регрессию
            X = sm.add_constant(model_data[lag_col])
            y = model_data['bnr']
            
            try:
                # Оцениваем модель
                model = sm.Logit(y, X).fit(disp=0, maxiter=100)
                forecast_name = f'bnrf_{var}_{lag:02d}'
                df[forecast_name] = np.nan
                y_pred = model.predict(X)
                df.loc[y_pred.index, forecast_name] = y_pred
                median_val = y_pred.median()
                pos_idx = y == 1
                neg_idx = y == 0
                n1 = neg_idx.sum()  # количество 0
                n2 = pos_idx.sum()  # количество 1
                
                if n1 >= 2 and n2 >= 2:
                    y_pred_pos = y_pred[pos_idx]
                    y_pred_neg = y_pred[neg_idx]
                    u_stat, p_value = stats.mannwhitneyu(y_pred_pos, y_pred_neg, alternative='two-sided')
                    
                    mu = n1 * n2 / 2
                    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                    z_value = (u_stat - mu) / sigma
                    auc = u_stat / (n1 * n2)
                    n1_vectors[var][lag] = n1
                    n2_vectors[var][lag] = n2
                    z_vectors[var][lag] = z_value
                    auc_vectors[var][lag] = auc
                else:
                    print(f"Недостаточно наблюдений в группах: N1={n1}, N2={n2}")
                    
            except Exception as e:
                print(f"Ошибка: {e}")
    results_df = pd.DataFrame(index=variables, columns=[f"{lag}" for lag in sorted(lags)])
    
    for var in variables:
        for lag in lags:
            col_name = f"{lag}"
            results_df.loc[var, col_name] = auc_vectors[var].get(lag, np.nan)
    sorted_cols = [f"{lag}" for lag in sorted(lags, reverse=True)]
    results_df = results_df[sorted_cols]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', '{:.4f}'.format)

    return results_df.T


def deviation(df_ind, df_th, dict):
    df = df_ind[['Date', 'bnr']].copy()
    exclude_columns = ['Date', 'bnr']
    columns_to_process = [col for col in df_ind.columns if col not in exclude_columns]
    for col in columns_to_process:
        df[f'{col}'] = (df_ind[f'{col}']-df_th[f'{col}'])/dict[f'{col}']
    return df


def calc_prir(df, type = None):
    exclude_columns = ['Date', 'bnr', 'rcgrf1', 'rcgrf2', 'rcgrf3', 'rcgru1', 'rcgru2', 'rcgru3']
    columns_to_process = [col for col in df.columns if col not in exclude_columns]
    for col in columns_to_process:
        df[f'{col}1'] = df[col] - df[col].shift(4)
        df[f'{col}2'] = (df[col]-df[col].shift(8))/2
        df[f'{col}_av2'] = df[f'{col}']-(df[f'{col}'].rolling(window=8, min_periods=8).mean()).shift(1)
        df[f'{col}_av3'] = df[f'{col}']-(df[f'{col}'].rolling(window=12, min_periods=12).mean()).shift(1)
    results_dict = {}
    if type != None:
        for col in df.columns:
            if col != 'Date' and col != 'bnr':
                filtered_data = df.loc[df['Date'] >= date(2009, 1, 1), col]
                result_value = 2 * filtered_data.std(ddof=1)
                results_dict[col] = result_value
    return df, results_dict


def itog(results=None, analysis_mode=None):
    """
    Генерирует барометр уязвимостей на основе переданных результатов
    
    Parameters:
    -----------
    results : dict, optional
        Результаты комплексного анализа из calculate.run_comprehensive_analysis()
        Если None, выполняется повторный анализ
    analysis_mode : str, optional
        Режим работы: 'actual' или 'preliminary'
        Если None, берется из results или определяется автоматически
    """
    # Если результаты не переданы, выполняем анализ (для обратной совместимости)
    if results is None:
        results = calculate.run_comprehensive_analysis()
    
    # Определяем режим работы
    if analysis_mode is None:
        analysis_mode = results.get('analysis_mode', 'actual')
    
    target_date = results.get('target_date', None)
    
    if analysis_mode == 'preliminary':
        print("\n" + "=" * 50)
        print("ПРЕДВАРИТЕЛЬНАЯ ОЦЕНКА")
        print("=" * 50)
        print("Для критических производных показателей применяется сдвиг на один квартал назад")
        print("Сдвинутые производные показатели:")
        print("  - ca (sto_y % ВВП) и threshold_sto")
        print("  - gd_gdp (vvd_mean % ВВП) и threshold_vvd")
        print("  - d_srv (КОД) и threshold_serv")
        print("  - l_gdp (credits % ВВП) и threshold_cred")
        print("  - rcgr_h1, rcgr_h2, rcgr_h3 и соответствующие thresholds")
        print("  - pest (ln(stoim_zil_nedv)) и threshold_nedv")
        print("=" * 50 + "\n")

    last_date = results['start_q'][['Date', 'vvd', 'Rev_ConsBud_exFSZN_YtD_mnBYN']].dropna().iloc[-1, 0]

    names_ind = {
    'domestic_thresholds': ['Date', 'd_serv_a', 'credits % ВВП', 'факт -1 zadolz % CPI', 'факт -2 zadolz % CPI', 'факт -3 zadolz % CPI', 'ln(stoim_zil_nedv)'],
    'banking_thresholds': ['Date', 'crb % ВВП', 'Leverage', 'crb_to_dep'],
    'debt_thresholds': ['Date', 'vvd_mean % ВВП', 'sto_y % ВВП'],
    'corporate_thresholds': ['Date', 'dsr', 'ggr', 'invr', 'cest', 'факт -1 ln_cred*100', 'факт -2 ln_cred*100', 'факт -3 ln_cred*100'],
    'government_thresholds': ['Date', 'ggd', 'ser_exdebt_gdp', 'bb_gdp']}
    names_th = {
    'domestic_thresholds': ['Date', 'threshold_serv', 'threshold_cred', 'threshold -1 zadolz % CPI', 'threshold -2 zadolz % CPI', 'threshold -3 zadolz % CPI', 'threshold_nedv'],
    'banking_thresholds': ['Date', 'threshold_crb', 'threshold_leverage', 'threshold_crb_to_dep'],
    'debt_thresholds': ['Date', 'threshold_vvd', 'threshold_sto'],
    'corporate_thresholds': ['Date', 'threshold_dsr', 'threshold_ggr', 'threshold_invr', 'threshold_cest', 'threshold -1 ln_cred*100', 'threshold -2 ln_cred*100', 'threshold -3 ln_cred*100'],
    'government_thresholds': ['Date', 'threshold_ggd', 'threshold_ser', 'threshold_bb_gdp']
    }
    ind, th = None, None
    for name in names_ind.keys():
        ind = indicator_data__threshold(ind, results[name][names_ind[name]].loc[(results[name]['Date']>=date(2005, 1, 1))&(results[name]['Date']<=last_date)])

    for name in names_th.keys():
        th = indicator_data__threshold(th, results[name][names_th[name]].loc[(results[name]['Date']>=date(2005, 1, 1))&(results[name]['Date']<=last_date)])

    rename_ind = {
    'd_serv_a': 'd_srv',
    'credits % ВВП': 'l_gdp',
    'факт -1 zadolz % CPI': 'rcgr_h1',
    'факт -2 zadolz % CPI': 'rcgr_h2', 
    'факт -3 zadolz % CPI': 'rcgr_h3',
    'ln(stoim_zil_nedv)': 'pest',
    'crb % ВВП': 'crr',
    'Leverage': 'lvr',
    'crb_to_dep': 'd_as',
    'vvd_mean % ВВП': 'gd_gdp',
    'sto_y % ВВП': 'ca',
    'dsr': 'dsr',
    'ggr': 'ggr',
    'invr': 'invr',
    'cest': 'cest',
    'факт -1 ln_cred*100': 'rcgr_c1',
    'факт -2 ln_cred*100': 'rcgr_c2',
    'факт -3 ln_cred*100': 'rcgr_c3',
    'ggd': 'ggd',
    'ser_exdebt_gdp': 'exd_s',
    'bb_gdp': 'bb_gdp'}

    rename_th = {
    'threshold_serv': 'd_srv',
    'threshold_cred': 'l_gdp',
    'threshold -1 zadolz % CPI': 'rcgr_h1',
    'threshold -2 zadolz % CPI': 'rcgr_h2', 
    'threshold -3 zadolz % CPI': 'rcgr_h3',
    'threshold_nedv': 'pest',
    'threshold_crb': 'crr',
    'threshold_leverage': 'lvr',
    'threshold_crb_to_dep': 'd_as',
    'threshold_vvd': 'gd_gdp',
    'threshold_sto': 'ca',
    'threshold_dsr': 'dsr',
    'threshold_ggr': 'ggr',
    'threshold_invr': 'invr',
    'threshold_cest': 'cest',
    'threshold -1 ln_cred*100': 'rcgr_c1',
    'threshold -2 ln_cred*100': 'rcgr_c2',
    'threshold -3 ln_cred*100': 'rcgr_c3',
    'threshold_ggd': 'ggd',
    'threshold_ser': 'exd_s',
    'threshold_bb_gdp': 'bb_gdp'
}
    
    ind = ind.rename(columns=rename_ind)
    th = th.rename(columns=rename_th)

    ind[['ca', 'bb_gdp']] = -ind[['ca', 'bb_gdp']]
    th[['ca', 'bb_gdp']] = -th[['ca', 'bb_gdp']]

    # ============================================================
    # ПРЕДВАРИТЕЛЬНЫЙ РЕЖИМ: Сдвиг ПРОИЗВОДНЫХ критических показателей
    # ПОСЛЕ всех расчетов (перед deviation)
    # ============================================================
    if analysis_mode == 'preliminary':
        print("Применение сдвига для производных критических показателей (после расчетов)...")
        
        # Производные критические столбцы
        critical_derived_columns = [
            'gd_gdp', 'dsr', 'ggr', 'ca', 
            'rcgr_c1', 'rcgr_c2', 'rcgr_c3', 'exd_s'
        ]
        
        # Сохраняем последнюю дату
        last_date_ind = ind['Date'].iloc[-1] if not ind.empty else None
        last_date_th = th['Date'].iloc[-1] if not th.empty else None
        
        # Применяем сдвиг и ЗАПОЛНЯЕМ NaN
        for col in critical_derived_columns:
            if col in ind.columns:
                # Запоминаем значение для последней даты
                last_value = ind.loc[ind['Date'] == last_date_ind, col].iloc[0] if last_date_ind is not None and not ind[ind['Date'] == last_date_ind].empty else None
                # Применяем сдвиг
                ind[col] = ind[col].shift(1)
                # Заполняем NaN (на первой дате) значением с последней даты (заполняем вперед и назад)
                ind[col] = ind[col].bfill().ffill()
                ind[col] = ind[col].fillna(0)
                print(f"  Сдвиг индикатора: {col}")
            if col in th.columns:
                last_value = th.loc[th['Date'] == last_date_th, col].iloc[0] if last_date_th is not None and not th[th['Date'] == last_date_th].empty else None
                th[col] = th[col].shift(1)
                th[col] = th[col].bfill().ffill()
                th[col] = th[col].fillna(0)
                print(f"  Сдвиг порога: {col}")
        
        print("\nСдвиг применен для производных критических столбцов ПОСЛЕ расчетов")
        print("  - ca (sto_y % ВВП) и threshold_sto")
        print("  - gd_gdp (vvd_mean % ВВП) и threshold_vvd")
        print("  - d_srv (КОД) и threshold_serv")
        print("  - l_gdp (credits % ВВП) и threshold_cred")
        print("  - rcgr_h1, rcgr_h2, rcgr_h3 и соответствующие thresholds")
        print("  - pest (ln(stoim_zil_nedv)) и threshold_nedv")
        print("=" * 50 + "\n")
    # ============================================================

    ind, std_ind = calc_prir(ind, 1)
    th, _ = calc_prir(th)
    
    desired_order = ['Date', 'bnr', 'dsr', 'dsr1', 'dsr2', 'dsr_av2', 'dsr_av3', 
                 'ggr', 'ggr1', 'ggr2', 'ggr_av2', 'ggr_av3', 'rcgr_c1', 
                 'rcgr_c2', 'rcgr_c3', 'invr', 'invr1', 'invr2', 'invr_av2', 
                 'invr_av3', 'd_srv', 'd_srv1', 'd_srv2', 'd_srv_av2', 'd_srv_av3', 
                 'l_gdp', 'l_gdp1', 'l_gdp2', 'l_gdp_av2', 'l_gdp_av3', 'rcgr_h1', 
                 'rcgr_h2', 'rcgr_h3', 'exd_s', 'exd_s1', 'exd_s2', 'exd_s_av2', 
                 'exd_s_av3', 'bb_gdp', 'bb_gdp1', 'bb_gdp2', 'bb_gdp_av2', 'bb_gdp_av3', 
                 'ggd', 'ggd1', 'ggd2', 'ggd_av2', 'ggd_av3', 'crr', 'crr1', 
                 'crr2', 'crr_av2', 'crr_av3', 'd_as', 'd_as1', 'd_as2', 
                 'd_as_av2', 'd_as_av3', 'lvr', 'lvr1', 'lvr2', 'lvr_av2', 
                 'lvr_av3', 'ca', 'ca1', 'ca2', 'ca_av2', 'ca_av3', 'gd_gdp', 
                 'gd_gdp1', 'gd_gdp2', 'gd_gdp_av2', 'gd_gdp_av3', 'cest', 'cest1', 
                 'cest2', 'cest_av2', 'cest_av3', 'pest', 'pest1', 'pest2', 
                 'pest_av2', 'pest_av3']

    ind = ind[desired_order]
    th = th[desired_order]
    std_ind = {key: std_ind[key] for key in desired_order if key in std_ind}
    deviation_df = deviation(ind, th, std_ind)
    AUCROC_Weight = find_auc(deviation_df, int(last_date.year))
    AUCROC_Weight2 = AUCROC_Weight-0.5
    AUCROC_Weight2.index =  -AUCROC_Weight2.index.astype(int)
    df_ind_std = pd.DataFrame(std_ind,  index=['annual SD'])
    res = rasch_auc(AUCROC_Weight2)
    deviation_df = deviation_df.loc[deviation_df['Date']>=date(2006,1,1)]
    calcul = deviation_df.copy()
    barometr = calcul[['Date', 'bnr']].copy()
    for col in calcul.columns:
        if col not in ['Date', 'bnr']:
            calcul[col] = calcul[col].where(calcul[col] >= -1, -1) * res.loc['Weight indicator in Sector', col]
    dict_n = {
        'Corparate':['dsr', 'dsr1', 'dsr2', 'dsr_av2', 'dsr_av3', 
                 'ggr', 'ggr1', 'ggr2', 'ggr_av2', 'ggr_av3', 'rcgr_c1', 
                 'rcgr_c2', 'rcgr_c3', 'invr', 'invr1', 'invr2', 'invr_av2', 
                 'invr_av3'],
        'Households': ['d_srv', 'd_srv1', 'd_srv2', 'd_srv_av2', 'd_srv_av3', 
                 'l_gdp', 'l_gdp1', 'l_gdp2', 'l_gdp_av2', 'l_gdp_av3', 'rcgr_h1', 
                 'rcgr_h2', 'rcgr_h3'],
        'Goverment': ['exd_s', 'exd_s1', 'exd_s2', 'exd_s_av2', 
                 'exd_s_av3', 'bb_gdp', 'bb_gdp1', 'bb_gdp2', 'bb_gdp_av2', 'bb_gdp_av3', 
                 'ggd', 'ggd1', 'ggd2', 'ggd_av2', 'ggd_av3'],
        'Banks': ['crr', 'crr1', 'crr2', 'crr_av2', 'crr_av3', 'd_as', 'd_as1', 'd_as2', 
                 'd_as_av2', 'd_as_av3', 'lvr', 'lvr1', 'lvr2', 'lvr_av2', 
                 'lvr_av3'],
        'External': ['ca', 'ca1', 'ca2', 'ca_av2', 'ca_av3', 'gd_gdp', 
                 'gd_gdp1', 'gd_gdp2', 'gd_gdp_av2', 'gd_gdp_av3'],
        'Nedv': ['cest', 'cest1', 
                 'cest2', 'cest_av2', 'cest_av3', 'pest', 'pest1', 'pest2', 
                 'pest_av2', 'pest_av3']
    }

    barometr['Total'],  barometr['Total_pos'],  barometr['Total_neg'] = 0.0, 0.0, 0.0
    for sector, columns in dict_n.items():
        available_cols = [col for col in columns if col in calcul.columns]
        barometr[sector] = calcul[available_cols].sum(axis=1)
        barometr['Total'] += barometr[sector]
    
    for col in barometr.columns:
        if col not in ['Date', 'bnr', 'Total', 'Total_pos', 'Total_neg']:
            barometr[f'{col}_positive'] = barometr[col].clip(lower=0)
            barometr['Total_pos']+=barometr[f'{col}_positive']
            barometr[f'{col}_negative'] = barometr[col].where(barometr[col] < 0, 0.0)
            barometr['Total_neg']+=barometr[f'{col}_negative']
    
    col_pos = ['Date', 'bnr', 'Corparate', 'Corparate_positive', 'Corparate_negative', 'Households', 'Households_positive', 'Households_negative',
               'Goverment', 'Goverment_positive', 'Goverment_negative', 'Banks', 'Banks_positive', 'Banks_negative', 'External', 'External_positive', 'External_negative',
               'Nedv', 'Nedv_positive', 'Nedv_negative', 'Total', 'Total_pos', 'Total_neg']
    barometr = barometr[col_pos]
    barometr['cred_r'] = ind['crr']-th['crr']
    barometr['crit'] = 1.5
    barometr = barometr.loc[barometr['Date']>=date(2009,1,1)]
    last_two_calc = calcul.iloc[-2:]
    last_two_calc = last_two_calc.drop(columns=['bnr'])
    df1, df2 = split_and_reshape_data(last_two_calc)
    df1.index = df1['Indicator']
    df1 = df1.drop(columns=['Indicator'])
    df2.index = df2['Indicator']
    df2 = df2.drop(columns=['Indicator'])
    df1 =df1.T
    df2 = df2.T

    with pd.ExcelWriter("Valnerabilities_Barometer.xlsx", mode="a", if_sheet_exists='overlay') as writer:
            df_ind_std.to_excel(writer, sheet_name='Indicators Data', index=True, header=False, startrow=2, startcol=1)
            ind.to_excel(writer, sheet_name="Indicators Data", index=False, startrow=4)
            th.to_excel(writer, sheet_name='Threshold', index=False, startrow=4)
            deviation_df.to_excel(writer, sheet_name='Deviation_SD', index=False, startrow=4)
            AUCROC_Weight.to_excel(writer, sheet_name='AUCROC Weight', startcol=2, startrow=2)
            AUCROC_Weight2.to_excel(writer, sheet_name='AUCROC Weight', header=False, startcol=2, startrow=13)
            res.to_excel(writer, sheet_name='AUCROC Weight', header=False, startcol=2, startrow=23)
            pd.DataFrame([res.loc['Weight indicator in Sector'].values], columns=res.columns).to_excel(writer, sheet_name='Calculation', index=False, header=False, startcol=2, startrow=2)
            calcul.to_excel(writer, sheet_name='Calculation', index=False, startrow=3)
            barometr.to_excel(writer, sheet_name='Barometr_1', index=False, header=False, startrow=3)
            df1[['l_gdp', 'rcgr_h', 'd_srv']].T.to_excel(writer, sheet_name='H', index=False, header=False, startrow=26, startcol=8)
            df1[['ggr', 'rcgr_c', 'dsr', 'invr']].T.to_excel(writer, sheet_name='C', index=False, header=False, startrow=26, startcol=8)
            df1[['cest', 'pest']].dropna().T.to_excel(writer, sheet_name='RE', index=False, header=False, startrow=26, startcol=7)
            df1[['gd_gdp', 'ca']].dropna().T.to_excel(writer, sheet_name='E', index=False, header=False, startrow=26, startcol=7)
            df1[['crr', 'd_as', 'lvr']].dropna().T.to_excel(writer, sheet_name='B', index=False, header=False, startrow=26, startcol=7)
            df1[['ggd', 'bb_gdp', 'exd_s']].dropna().T.to_excel(writer, sheet_name='G', index=False, header=False, startrow=26, startcol=7)
            df2[['l_gdp', 'rcgr_h', 'd_srv']].T.to_excel(writer, sheet_name='H', index=False, header=False, startrow=26, startcol=1)
            df2[['ggr', 'rcgr_c', 'dsr', 'invr']].T.to_excel(writer, sheet_name='C', index=False, header=False, startrow=26, startcol=1)
            df2[['cest', 'pest']].dropna().T.to_excel(writer, sheet_name='RE', index=False, header=False, startrow=26, startcol=1)
            df2[['gd_gdp', 'ca']].dropna().T.to_excel(writer, sheet_name='E', index=False, header=False, startrow=26, startcol=1)
            df2[['crr', 'd_as', 'lvr']].dropna().T.to_excel(writer, sheet_name='B', index=False, header=False, startrow=26, startcol=1)
            df2[['ggd', 'bb_gdp', 'exd_s']].dropna().T.to_excel(writer, sheet_name='G', index=False, header=False, startrow=26, startcol=1)