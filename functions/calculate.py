# functions/calculate.py
import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import seasonal_decompose
import requests
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


# ============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def determine_mode(start_q, target_date):
    """
    Определяет режим работы: фактический или предварительный
    
    Parameters:
    -----------
    start_q : DataFrame
        Квартальные данные
    target_date : datetime
        Целевая дата (отчетный квартал)
    
    Returns:
    --------
    str: 'actual' или 'preliminary'
    """
    # Список столбцов, которые могут отсутствовать в отчетном квартале
    preliminary_columns = [
        'vvd', 'vvd_crat', 'vvd_dolg', 'vvd % ВВП', 'sto',
        'pogash_kk_n', 'pogash_dk_n', 'IntPymt_NRes_q_thUSD',
        'ExtDebtSvc_Gov_Int_mnUSD', 'ExtDebtSvc_Gov_Pr_mnUSD'
    ]
    
    # Проверяем наличие данных за целевой квартал
    target_data = start_q[start_q['Date'] == target_date]
    
    if target_data.empty:
        return 'preliminary'
    
    # Проверяем, есть ли пропуски в критических столбцах
    for col in preliminary_columns:
        if col in target_data.columns:
            if pd.isna(target_data[col].iloc[0]):
                return 'preliminary'
        else:
            return 'preliminary'
    
    return 'actual'


def get_available_quarter(start_q, target_date, mode):
    """
    Определяет доступный квартал для предварительной оценки
    
    Parameters:
    -----------
    start_q : DataFrame
        Квартальные данные
    target_date : datetime
        Целевая дата (отчетный квартал)
    mode : str
        Режим работы
    
    Returns:
    --------
    datetime: доступная дата для расчетов
    """
    if mode == 'actual':
        return target_date
    
    # Для предварительного режима ищем последний доступный квартал
    # с полными данными по критическим столбцам
    preliminary_columns = [
        'vvd', 'vvd_crat', 'vvd_dolg', 'vvd % ВВП', 'sto',
        'pogash_kk_n', 'pogash_dk_n', 'IntPymt_NRes_q_thUSD',
        'ExtDebtSvc_Gov_Int_mnUSD', 'ExtDebtSvc_Gov_Pr_mnUSD'
    ]
    
    # Получаем все даты
    dates = start_q['Date'].unique()
    dates = sorted(dates)
    
    # Ищем последний квартал с полными данными
    available_date = None
    for date in reversed(dates):
        if date > target_date:
            continue
        data = start_q[start_q['Date'] == date]
        all_present = True
        for col in preliminary_columns:
            if col in data.columns:
                if pd.isna(data[col].iloc[0]):
                    all_present = False
                    break
            else:
                all_present = False
                break
        if all_present:
            available_date = date
            break
    
    # Если не найден квартал с полными данными, используем последний доступный
    if available_date is None:
        available_date = dates[-1] if len(dates) > 0 else target_date
    
    return available_date


def fill_missing_values(series):
    """Заполняет пропущенные значения линейной интерполяцией"""
    result = series.copy()
    n = len(result)
    i = 0
    
    while i < n:
        if pd.notna(result[i]):
            i += 1
            continue
        
        start_idx = i
        missing_count = 0

        while i < n and pd.isna(result[i]):
            missing_count += 1
            i += 1
        
        if i >= n:
            break
        
        prev_idx = start_idx - 1
        if prev_idx >= 0 and pd.notna(result[prev_idx]):
            prev_value = result[prev_idx]
            next_value = result[i]
            
            if missing_count == 1:
                result[start_idx] = (prev_value + next_value) / 2
            else:
                step = (next_value - prev_value) / missing_count
                for j in range(missing_count):
                    result[start_idx + j] = prev_value + step * (j + 1)
    
    return result


def convert_dates_to_date(df, columns):
    """Конвертирует datetime в date для всех указанных колонок"""
    if df is None:
        return None
    if df.empty:
        return df
    for col in columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col]).dt.date
            except Exception:
                pass
    return df


# ============================================================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================================================

def load_data():
    """Загружает исходные данные из Excel файла"""
    print("Загрузка данных...")
    start_q = pd.read_excel('Исходные данные.xlsx', sheet_name='prep_q', skiprows=1)
    start_m = pd.read_excel('Исходные данные.xlsx', sheet_name='prep_m', skiprows=1)
    params = pd.read_excel('Исходные данные.xlsx', sheet_name='param')
    
    print(f"Загружено: {start_q.shape[0]} квартальных записей, {start_m.shape[0]} месячных записей")
    return start_q, start_m, params


def fetch_exchange_rates(start_date, end_date):
    """Загружает исторические данные обменного курса BYR/USD с сайта НБРБ"""
    
    def get_code_usd(begin_date):
        """Определяет код валюты USD для заданной даты"""
        if begin_date < datetime(2021, 7, 9):
            return 145
        else:
            return 431
    
    def fetch_data_from_api(code, begin_date, end_date, rates_df):
        """Получает данные курса валюты из API НБРБ"""
        url = f"https://www.nbrb.by/API/ExRates/Rates/Dynamics/{code}?startDate={begin_date}&endDate={end_date}"
        response = requests.get(url, verify=False)
        data = response.json()
        df = pd.json_normalize(data)
        
        df['Cur_ID'] = df['Cur_ID'].astype('Int64')
        df['Date'] = pd.to_datetime(df['Date'])
        df['Cur_OfficialRate'] = df['Cur_OfficialRate'].astype(float)
        
        rates_df = rates_df.merge(df[['Date', 'Cur_OfficialRate']], on='Date', how='left')
        rates_df['BYR/USD'] = rates_df['Cur_OfficialRate'].where(
            rates_df['Cur_OfficialRate'].notna(), rates_df['BYR/USD']
        )
        rates_df = rates_df.drop(columns=['Cur_OfficialRate'])
        
        return rates_df
    
    # Создание диапазонов дат для запросов
    date_range = pd.date_range(start=start_date, end=end_date, freq='365D')
    additional_dates = [datetime(2003, 1, 1), datetime(2016, 7, 1), datetime(2021, 7, 9)]
    date_range = sorted(set(date_range).union(additional_dates))
    
    dates = pd.DataFrame(date_range, columns=['BeginDate'])
    dates['EndDate'] = dates['BeginDate'].shift(-1).fillna(datetime.now())
    dates['EndDate'] = dates['EndDate'].dt.date
    dates['Code_USD'] = dates['BeginDate'].apply(get_code_usd)
    
    # Создание датафрейма для хранения курсов
    dates_rates = pd.date_range(start=start_date, end=end_date, freq='D')
    rates = pd.DataFrame(dates_rates, columns=['Date'])
    rates['BYR/USD'] = np.nan
    
    # Получение данных для каждого периода
    print("Загрузка данных обменного курса...")
    for index, row in dates.iterrows():
        rates = fetch_data_from_api(row['Code_USD'], row['BeginDate'], row['EndDate'], rates)
        print(f"Загружен период: {row['BeginDate'].strftime('%Y-%m-%d')} - {row['EndDate'].strftime('%Y-%m-%d')}")
    
    # Очистка и преобразование данных
    rates = rates.dropna()
    rates.loc[rates['Date'] < '2016-07-01', 'BYR/USD'] /= 10000
    
    # Агрегация по месяцам
    date_month = pd.date_range(start=start_date, end=end_date, freq='MS')
    rates_m = pd.DataFrame(date_month, columns=['Date'])
    rates_m = pd.merge(rates_m, rates[['Date', 'BYR/USD']], on='Date', how='left')
    
    print(f"Загружено {rates_m.shape[0]} месячных записей курса валют")
    return rates_m


# ============================================================================
# 3. ОБРАБОТКА МЕСЯЧНЫХ ДАННЫХ
# ============================================================================

def process_monthly_data(start_m, rates_m):
    """Обрабатывает месячные данные, добавляя расчетные показатели"""
    print("Обработка месячных данных...")
    
    # Объединение с данными о курсе валют
    start_m = pd.merge(start_m, rates_m[['Date', 'BYR/USD']], on='Date', how='left')
    
    # Расчет процентных долгов
    start_m['GIntDebt_CGov_EoP_pct'] = np.where(
        start_m['Date'] <= pd.Timestamp(2022, 5, 1), 
        np.nan, 
        start_m['GIntDebt_CGov_EoP_mnBYN'] / start_m['IntDebt_CGov_EoP_mnBYN'] * 100
    )
    
    start_m['GExtDebt_CGov_EoP_pct'] = np.where(
        start_m['Date'] <= pd.Timestamp(2022, 5, 1), 
        np.nan, 
        start_m['GExtDebt_CGov_EoP_mnBYN'] / start_m['ExtDebt_CGov_EoP_mnBYN'] * 100
    )
    
    # Заполнение пропущенных значений
    start_m['GExtDebt_CGov_EoP_pct'] = fill_missing_values(start_m['GExtDebt_CGov_EoP_pct'])
    start_m['GIntDebt_CGov_EoP_pct'] = fill_missing_values(start_m['GIntDebt_CGov_EoP_pct'])
    start_m['GIntDebt_CGov_EoP_pct'] = start_m['GIntDebt_CGov_EoP_pct'].ffill()
    
    # Расчет абсолютных значений долгов
    start_m['GExtDebt_CGov_EoP_mnBYN'] = np.where(
        start_m['GExtDebt_CGov_EoP_mnBYN'].isna(),
        start_m['ExtDebt_CGov_EoP_mnBYN'] * start_m['GExtDebt_CGov_EoP_pct'] / 100,
        start_m['GExtDebt_CGov_EoP_mnBYN']
    )
    
    start_m['GIntDebt_CGov_EoP_mnBYN'] = np.where(
        start_m['GIntDebt_CGov_EoP_mnBYN'].isna(),
        start_m['IntDebt_CGov_EoP_mnBYN'] * start_m['GIntDebt_CGov_EoP_pct'] / 100,
        start_m['GIntDebt_CGov_EoP_mnBYN']
    )
    
    # Расчет общих долгов
    start_m['Долг центрального правительства и долг, гарантированный центральным правительством'] = (
        start_m['GExtDebt_CGov_EoP_mnBYN'] + 
        start_m['GIntDebt_CGov_EoP_mnBYN'] + 
        start_m['ExtDebt_CGov_EoP_mnBYN'] + 
        start_m['IntDebt_CGov_EoP_mnBYN']
    )
    
    start_m['Долг ЦП и долг, гарантированный ЦП, млн. BYN'] = (
        start_m['Долг центрального правительства и долг, гарантированный центральным правительством']
        .rolling(window=2).mean()
    )
    
    start_m['Долг ЦП и долг, гарантированный ЦП, млн. USD'] = (
        (start_m['Долг центрального правительства и долг, гарантированный центральным правительством'] / 
         start_m['BYR/USD'])
        .rolling(window=2).mean()
    )
    
    # Расчет внешнего государственного долга
    start_m['ExtGovDebt_mnBYN'] = (
        ((start_m['ExtDebt_CGov_EoP_mnBYN'] + start_m['GExtDebt_CGov_EoP_mnBYN'])
         .rolling(window=2).mean())
        .rolling(window=3).mean()
    )
    
    start_m['ExtGovDebt_mnUSD'] = (
        (((start_m['ExtDebt_CGov_EoP_mnBYN'] + start_m['GExtDebt_CGov_EoP_mnBYN']) / 
          start_m['BYR/USD'])
         .rolling(window=2).mean())
        .rolling(window=3).mean()
    )
    
    return start_m


def calculate_quarterly_indicators(start_m):
    """Вычисляет квартальные показатели на основе месячных данных"""
    print("Расчет квартальных показателей...")
    
    start_m['NER_usdbyn_weight'] = start_m['NER_usdbyn_weight_m'].rolling(window=3).mean()
    start_m['crb_nc_q'] = start_m['crb_nc'].rolling(window=3).mean()
    start_m['crb_fc_q'] = start_m['crb_fc'].rolling(window=3).mean()
    start_m['stoim_zil_nedv_q'] = (start_m['NER_usdbyn_weight_m'] * start_m['zil_nedv_stoim']).rolling(window=3).mean()
    start_m['precent_stavki_nedviz_cred_q'] = start_m['precent_stavki_nedviz_cred'].rolling(window=3).mean()
    start_m['precent_stavki_potreb_cred_q'] = start_m['precent_stavki_potreb_cred'].rolling(window=3).mean()
    start_m['precent_stavki_ul_byn_q'] = start_m['precent_stavki_ul_byn'].rolling(window=3).mean()
    start_m['precent_stavki_ul_skv_q'] = start_m['precent_stavki_ul_skv'].rolling(window=3).mean()
    start_m['TotDebt_CGov_GInc_mnBYN'] = start_m['Долг ЦП и долг, гарантированный ЦП, млн. BYN'].rolling(window=3).mean()
    start_m['TotDebt_CGov_GInc_mnUSD'] = start_m['Долг ЦП и долг, гарантированный ЦП, млн. USD'].rolling(window=3).mean()
    
    return start_m


def process_quarterly_data(start_q, start_m):
    """Обрабатывает квартальные данные, добавляя агрегированные показатели"""
    print("Обработка квартальных данных...")
    
    # Смещение дат для выравнивания
    start_m['Date'] = start_m['Date'] - pd.DateOffset(months=2)
    
    # Выбор нужных колонок для объединения
    merge_columns = [
        'Date', 'NER_usdbyn_weight', 'crb_nc_q', 'crb_fc_q', 'stoim_zil_nedv_q',
        'precent_stavki_nedviz_cred_q', 'precent_stavki_potreb_cred_q', 
        'precent_stavki_ul_byn_q', 'precent_stavki_ul_skv_q',
        'TotDebt_CGov_GInc_mnBYN', 'TotDebt_CGov_GInc_mnUSD', 
        'ExtGovDebt_mnBYN', 'ExtGovDebt_mnUSD'
    ]
    
    start_q = pd.merge(start_q, start_m[merge_columns], on='Date', how='left')
    
    # Расчет номинального ВВП в USD
    start_q['nGDP_mnUSD_q'] = start_q['nGDP_mnUSD_q'].where(
        start_q['nGDP_mnUSD_q'].notna(), 
        start_q['nGDP_q'] / start_q['NER_usdbyn_weight']
    )
    
    # Сезонная корректировка
    mask = start_q['Date'] >= '2011-01-01'
    
    if mask.sum() > 0:
        result = seasonal_decompose(
            start_q.loc[mask, 'crb_nc_q'].dropna(), 
            period=4, 
            model='additive'
        )
        
        start_q['crb_nc_sa'] = start_q['crb_nc_q']
        start_q.loc[mask, 'crb_nc_sa'] = start_q.loc[mask, 'crb_nc_q'] - result.seasonal
        start_q.loc[mask, 'crb_sa'] = start_q.loc[mask, 'crb_nc_sa'] + start_q.loc[mask, 'crb_fc_q']
    
    # Обновление процентных ставок
    start_q['precent_stavki_nedviz_cred_q'] = np.where(
        start_q['precent_stavki_nedviz_cred'].notna(), 
        start_q['precent_stavki_nedviz_cred'], 
        start_q['precent_stavki_nedviz_cred_q']
    )
    
    start_q['precent_stavki_potreb_cred_q'] = np.where(
        start_q['precent_stavki_potreb_cred'].notna(), 
        start_q['precent_stavki_potreb_cred'], 
        start_q['precent_stavki_potreb_cred_q']
    )
    
    return start_q


# ============================================================================
# 4. РАСЧЕТ ДОПОЛНИТЕЛЬНЫХ МЕТРИК
# ============================================================================

def calculate_additional_quarterly_metrics(start_q):
    """Вычисляет дополнительные квартальные метрики"""
    print("Расчет дополнительных квартальных метрик...")
    
    start_q['vvd_mean'] = start_q['vvd'].rolling(window=2).mean()
    start_q['act_bank_q'] = start_q['act_bank'].rolling(window=2).mean()
    start_q['cap_bank_q'] = start_q['cap_bank'].rolling(window=2).mean()
    start_q['Leverage'] = start_q['act_bank_q'] / start_q['cap_bank_q']
    
    return start_q


def filter_data_to_current_quarter(start_q):
    """Фильтрует данные до текущего квартала"""
    current_date = pd.Timestamp.now()
    current_quarter = current_date.quarter
    
    quarter_to_month = {1: 1, 2: 4, 3: 7, 4: 10}
    first_month = quarter_to_month.get(current_quarter, 1)
    
    first_date_of_quarter = pd.Timestamp(
        year=current_date.year, 
        month=first_month, 
        day=1
    )
    
    now_data_date = start_q[['Date', 'vvd', 'Rev_ConsBud_exFSZN_YtD_mnBYN']].dropna().iloc[-1, 0]
    
    start_q = start_q.loc[start_q['Date'] <= now_data_date]
    print(f"Данные отфильтрованы до {now_data_date.strftime('%Y-%m-%d')}")
    
    return start_q, now_data_date


def calculate_yearly_metrics(start_q):
    """Вычисляет годовые метрики"""
    print("Расчет годовых метрик...")
    
    start_q['gdp_nsa_usd_y'] = start_q['nGDP_mnUSD_q'].rolling(window=4).sum()
    start_q['gdp_nsa_byn_y'] = start_q['nGDP_q'].rolling(window=4).sum()
    start_q['sto_y'] = start_q['sto'].rolling(window=4).sum()
    start_q['- sto_y'] = -start_q['sto_y']
    start_q['sto_y % ВВП'] = start_q['sto_y'] / start_q['gdp_nsa_usd_y'] * 100
    start_q['vvd_mean % ВВП'] = start_q['vvd_mean'] / start_q['gdp_nsa_usd_y'] * 100
    
    return start_q


def forecast_vvd(start_q, params, analysis_mode='actual', target_date=None):
    """Прогнозирует значения ВВД на основе параметров"""
    print("Прогнозирование ВВД...")
    
    vvd_row = params[params['Название показателя'] == 
                    "ВВД по всем секторам экономики на конец периода, млн.долл США*"]
    
    if vvd_row.empty:
        print("Предупреждение: Не найдены параметры для прогнозирования ВВД")
        return start_q
    
    consts = vvd_row.iloc[0, 2:].dropna().values.tolist()
    
    # ============================================================
    # ПРЕДВАРИТЕЛЬНЫЙ РЕЖИМ: прогноз строится как для предыдущего квартала
    # ============================================================
    if analysis_mode == 'preliminary' and target_date is not None:
        # Находим предыдущий квартал (где есть данные)
        prev_date = target_date - pd.DateOffset(months=3)
        print(f"  Предварительный режим: прогноз ВВД строится для {prev_date.strftime('%Y-%m-%d')}")
        
        # Находим данные только до предыдущего квартала
        start_q_filtered = start_q[start_q['Date'] <= prev_date].copy()
        
        # Находим последнее доступное значение vvd в отфильтрованных данных
        vvd_series = start_q_filtered['vvd'].dropna()
        if len(vvd_series) >= 4:
            # Берем значение за 4 квартала до последней даты в отфильтрованных данных
            last_vvd = vvd_series.iloc[-4]
        elif len(vvd_series) > 0:
            last_vvd = vvd_series.iloc[-1]
        else:
            last_vvd = 35000
            print(f"  Предварительный режим: данные VVD отсутствуют, используется значение по умолчанию = {last_vvd}")
        
        # Определяем последнюю дату в отфильтрованных данных
        last_date = start_q_filtered['Date'].iloc[-1]
        print(f"  Последняя дата с данными: {last_date.strftime('%Y-%m-%d')}, последнее VVD = {last_vvd}")
        
        # Строим прогноз от последней даты с данными
        new_rows = []
        for i in range(8):
            new_vvd = last_vvd * consts[i]
            new_date = last_date + pd.DateOffset(months=3 * (i + 1))
            
            new_row = {
                'Date': new_date,
                'vvd': new_vvd,
                'vvd % ВВП': np.nan,
                'sto': np.nan,
                'nGDP_mnUSD_q': np.nan,
                'vvd_mean': np.nan,
                'gdp_nsa_usd_y': np.nan,
                'vvd_mean % ВВП': np.nan,
                'annual_diff_vvd': np.nan
            }
            new_rows.append(new_row)
        
        # Добавляем прогнозные строки к ОРИГИНАЛЬНОМУ start_q (не к отфильтрованному)
        if new_rows:
            start_q = pd.concat([start_q, pd.DataFrame(new_rows[1:])], ignore_index=True)
            print(f"  Добавлено {len(new_rows)} прогнозных записей ВВД от {last_date.strftime('%Y-%m-%d')}")
        
        # Теперь заполняем пропущенное значение на отчетный квартал
        # Берем первый прогнозный период (который соответствует отчетному кварталу)
        target_mask = start_q['Date'] == target_date
        if target_mask.any():
            # Находим первое прогнозное значение (оно должно соответствовать отчетному кварталу)
            first_forecast = new_rows[0]['vvd'] if new_rows else last_vvd * consts[0]
            idx = start_q[target_mask].index[0]
            start_q.loc[idx, 'vvd'] = first_forecast
            print(f"  Заполнено пропущенное значение VVD на {target_date.strftime('%Y-%m-%d')} = {first_forecast}")
        
        # Расчет годового прироста ВВД для предварительного режима
        vvd_series_all = start_q['vvd'].dropna()
        if len(vvd_series_all) >= 9:
            last_val = vvd_series_all.iloc[-1]
            prev_val = vvd_series_all.iloc[-9] if len(vvd_series_all) >= 9 else vvd_series_all.iloc[0]
            start_q['annual_diff_vvd'] = (start_q['vvd'] - start_q['vvd'].shift(8)) / 2
            start_q['annual_diff_vvd'] = start_q['annual_diff_vvd'].fillna((last_val - prev_val) / 2)
        else:
            start_q['annual_diff_vvd'] = (start_q['vvd'] - start_q['vvd'].shift(8)) / 2
        
        return start_q
    
    # ============================================================
    # ФАКТИЧЕСКИЙ РЕЖИМ: обычный расчет
    # ============================================================
    else:
        # Берем значение за 4 квартала до последней даты
        vvd_series = start_q['vvd'].dropna()
        if len(vvd_series) >= 4:
            last_vvd = vvd_series.iloc[-4]
        elif len(vvd_series) > 0:
            last_vvd = vvd_series.iloc[-1]
        else:
            last_vvd = 35000
        
        last_date = start_q['Date'].iloc[-1]
        
        new_rows = []
        for i in range(8):
            new_vvd = last_vvd * consts[i]
            new_date = last_date + pd.DateOffset(months=3 * (i + 1))
            
            new_row = {
                'Date': new_date,
                'vvd': new_vvd,
                'vvd % ВВП': np.nan,
                'sto': np.nan,
                'nGDP_mnUSD_q': np.nan,
                'vvd_mean': np.nan,
                'gdp_nsa_usd_y': np.nan,
                'vvd_mean % ВВП': np.nan,
                'annual_diff_vvd': np.nan
            }
            new_rows.append(new_row)
        
        if new_rows:
            start_q = pd.concat([start_q, pd.DataFrame(new_rows)], ignore_index=True)
            print(f"Добавлено {len(new_rows)} прогнозных записей ВВД")
        
        start_q['annual_diff_vvd'] = (start_q['vvd'] - start_q['vvd'].shift(8)) / 2
        
        return start_q


# ============================================================================
# 5. ВНЕШНИЙ ДОЛГ И САЛЬДО ТОРГОВОГО БАЛАНСА
# ============================================================================

def calculate_debt_thresholds(start_q, params, analysis_mode='actual', target_date=None):
    """Рассчитывает пороговые значения для внешнего долга и сальдо торгового баланса"""
    print("Расчет пороговых значений долга...")
    
    # ============================================================
    # ПРЕДВАРИТЕЛЬНЫЙ РЕЖИМ: Разделение данных до dropna()
    # ============================================================
    if analysis_mode == 'preliminary' and target_date is not None:
        prev_date = target_date - pd.DateOffset(months=3)
        print(f"  Предварительный режим: разделение данных для HP-фильтра")
        print(f"  Критические данные (vvd_mean % ВВП, sto_y % ВВП) до: {prev_date.strftime('%Y-%m-%d')}")
        
        # 1. КРИТИЧЕСКАЯ ЧАСТЬ: vvd_mean % ВВП и sto_y % ВВП (только до предыдущего квартала)
        rasch_critical = start_q[['Date', 'vvd_mean % ВВП', 'sto_y % ВВП']].copy()
        rasch_critical = rasch_critical[rasch_critical['Date'] <= prev_date].copy()
        rasch_critical = rasch_critical.dropna()
        rasch_critical = rasch_critical[rasch_critical['Date'] >= pd.Timestamp('2002-01-01')].dropna()
        
        # Проверка наличия данных
        if rasch_critical.empty:
            print("  Предупреждение: Нет критических данных для HP-фильтра")
            rasch_critical = pd.DataFrame(columns=['Date', 'vvd_mean % ВВП', 'sto_y % ВВП'])
        
        # Получаем параметры для прогнозирования
        vvd_params = params.loc[
            params['Название показателя'] == "Валовый внешний долг к ВВП"
        ]

        
        # --- РАСЧЕТ ДЛЯ КРИТИЧЕСКОЙ ЧАСТИ (vvd_mean % ВВП и sto_y % ВВП) ---

        if not rasch_critical.empty:
            # Для sto_y % ВВП - просто устанавливаем пороговое значение
            # (оно не требует HP-фильтра, только константа -2.3)
            rasch_critical['threshold_sto'] = -2.3
            last_date_critical = rasch_critical['Date'].iloc[-1]
            
            # Добавляем прогнозные строки для HP-фильтра (для vvd_mean % ВВП)
            new_rows = []
            for i in range(12):
                if i == 0:
                    if not vvd_params.empty:
                        new_vvd = vvd_params['Начальные значения (HP)'].values[0]
                    else:
                        new_vvd = rasch_critical['vvd_mean % ВВП'].iloc[-1]
                elif i == 1:
                    if not vvd_params.empty:
                        new_vvd = vvd_params['Unnamed: 2'].values[0]
                    else:
                        new_vvd = rasch_critical['vvd_mean % ВВП'].iloc[-1]
                else:
                    if not vvd_params.empty:
                        last_vvd = rasch_critical['vvd_mean % ВВП'].iloc[-2] if len(rasch_critical) >= 2 else rasch_critical['vvd_mean % ВВП'].iloc[-1]
                        new_vvd = last_vvd + vvd_params['Шаги'].values[0]
                    else:
                        new_vvd = rasch_critical['vvd_mean % ВВП'].iloc[-1]
                
                new_date = rasch_critical['Date'].iloc[-1] + pd.DateOffset(months=3)
                new_row = {'Date': new_date, 'vvd_mean % ВВП': new_vvd}
                new_rows.append(new_row)
                
                rasch_critical = pd.concat([rasch_critical, pd.DataFrame(new_rows)], ignore_index=True)
                new_rows = []
            
            # HP-фильтр для критических данных (vvd_mean % ВВП)
            try:
                cycle, trend = hpfilter(rasch_critical['vvd_mean % ВВП'], 1600)
                rasch_critical['threshold_vvd'] = pd.Series(trend)
                rasch_critical['threshold_vvd'] = rasch_critical['threshold_vvd'].where(
                    rasch_critical['Date'] <= last_date_critical, np.nan
                )
                
                # Итерационная корректировка для vvd_mean % ВВП
                if not vvd_params.empty:
                    std_trend = rasch_critical['vvd_mean % ВВП'].where(
                        (rasch_critical['Date'] >= pd.Timestamp(2009, 1, 1)) & 
                        (rasch_critical['Date'] <= last_date_critical)
                    ).dropna().std(ddof=1)
                    
                    if std_trend is not None and not np.isnan(std_trend):
                        last_trend = rasch_critical['threshold_vvd'].where(
                            rasch_critical['Date'] == last_date_critical
                        ).dropna().iloc[0] if not rasch_critical[rasch_critical['Date'] == last_date_critical].empty else 10
                        
                        first_vvd = rasch_critical['vvd_mean % ВВП'].where(
                            rasch_critical['Date'] == last_date_critical + pd.DateOffset(months=3)
                        ).dropna().iloc[0] if not rasch_critical[rasch_critical['Date'] == last_date_critical + pd.DateOffset(months=3)].empty else 10
                        
                        max_iter = 50
                        iter_count = 0
                        while round(first_vvd, 10) != round(last_trend - std_trend, 10) and iter_count < max_iter:
                            iter_count += 1
                            for i in range(12):
                                if i == 0:
                                    new_vvd = last_trend - std_trend
                                    first_vvd = new_vvd
                                elif i == 1:
                                    new_vvd = last_trend + std_trend
                                else:
                                    last_vvd_val = rasch_critical['vvd_mean % ВВП'].where(
                                        rasch_critical['Date'] == last_date_critical + pd.DateOffset(months=3*(i-1))
                                    ).dropna().iloc[0] if not rasch_critical[rasch_critical['Date'] == last_date_critical + pd.DateOffset(months=3*(i-1))].empty else last_trend
                                    new_vvd = last_vvd_val + vvd_params['Шаги'].values[0]
                                
                                date_reset = last_date_critical + pd.DateOffset(months=3*(i+1))
                                if date_reset in rasch_critical['Date'].values:
                                    rasch_critical.loc[rasch_critical['Date'] == date_reset, 'vvd_mean % ВВП'] = new_vvd
                            
                            cycle, trend = hpfilter(rasch_critical['vvd_mean % ВВП'], 1600)
                            rasch_critical['threshold_vvd'] = pd.Series(trend)
                            rasch_critical['threshold_vvd'] = rasch_critical['threshold_vvd'].where(
                                rasch_critical['Date'] <= last_date_critical, np.nan
                            )
                            
                            std_trend = rasch_critical['vvd_mean % ВВП'].where(
                                (rasch_critical['Date'] >= pd.Timestamp(2009, 1, 1)) & 
                                (rasch_critical['Date'] <= last_date_critical)
                            ).dropna().std(ddof=1)
                            
                            last_trend = rasch_critical['threshold_vvd'].where(
                                rasch_critical['Date'] == last_date_critical
                            ).dropna().iloc[0] if not rasch_critical[rasch_critical['Date'] == last_date_critical].empty else last_trend
            except Exception as e:
                print(f"  Предупреждение: ошибка при HP-фильтре для vvd: {e}")
                rasch_critical['threshold_vvd'] = rasch_critical['vvd_mean % ВВП'].rolling(window=8, min_periods=4).mean()
        
        # --- MERGE: возвращаем только критическую часть ---
        # В предварительном режиме возвращаем только данные до prev_date
        result = rasch_critical[['Date', 'vvd_mean % ВВП', 'threshold_vvd', 'sto_y % ВВП', 'threshold_sto']].copy()
        
        return result
    
    # ============================================================
    # ФАКТИЧЕСКИЙ РЕЖИМ: Обычный расчет
    # ============================================================
    else:
        # Подготовка данных
        rasch = start_q[['Date', 'vvd_mean % ВВП', 'sto_y % ВВП']].copy()
        rasch = rasch.dropna()
        rasch = rasch[['Date', 'vvd_mean % ВВП', 'sto_y % ВВП']].where(
            rasch['Date'] >= pd.Timestamp('2002-01-01')
        ).dropna()
        
        rasch['threshold_sto'] = -2.3
        last_date_know = rasch['Date'].iloc[-1]
        
        # Прогнозирование ВВД
        vvd_params = params.loc[
            params['Название показателя'] == "Валовый внешний долг к ВВП"
        ]
        
        new_rows = []
        for i in range(12):
            if i == 0:
                if not vvd_params.empty:
                    new_vvd = vvd_params['Начальные значения (HP)'].values[0]
                else:
                    new_vvd = rasch['vvd_mean % ВВП'].iloc[-1]
            elif i == 1:
                if not vvd_params.empty:
                    new_vvd = vvd_params['Unnamed: 2'].values[0]
                else:
                    new_vvd = rasch['vvd_mean % ВВП'].iloc[-1]
            else:
                if not vvd_params.empty:
                    last_vvd = rasch['vvd_mean % ВВП'].iloc[-2]
                    new_vvd = last_vvd + vvd_params['Шаги'].values[0]
                else:
                    new_vvd = rasch['vvd_mean % ВВП'].iloc[-1]
            
            new_date = rasch['Date'].iloc[-1] + pd.DateOffset(months=3)
            new_row = {'Date': new_date, 'vvd_mean % ВВП': new_vvd}
            new_rows.append(new_row)
            
            rasch = pd.concat([rasch, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows = []
        
        # HP-фильтр для выделения тренда
        try:
            cycle, trend = hpfilter(rasch['vvd_mean % ВВП'], 1600)
            rasch['threshold_vvd'] = pd.Series(trend)
            rasch['threshold_vvd'] = rasch['threshold_vvd'].where(
                rasch['Date'] <= last_date_know, np.nan
            )
        except Exception as e:
            print(f"  Предупреждение: ошибка при HP-фильтре: {e}")
            rasch['threshold_vvd'] = rasch['vvd_mean % ВВП'].rolling(window=8, min_periods=4).mean()
        
        # Расчет стандартного отклонения
        std_trend = rasch['vvd_mean % ВВП'].where(
            (rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
            (rasch['Date'] <= last_date_know)
        ).dropna().std(ddof=1)
        
        if std_trend is not None and not np.isnan(std_trend):
            last_trend = rasch['threshold_vvd'].where(
                rasch['Date'] == last_date_know
            ).dropna().iloc[0] if not rasch[rasch['Date'] == last_date_know].empty else 10
            
            first_vvd = rasch['vvd_mean % ВВП'].where(
                rasch['Date'] == last_date_know + pd.DateOffset(months=3)
            ).dropna().iloc[0] if not rasch[rasch['Date'] == last_date_know + pd.DateOffset(months=3)].empty else 10
            
            rasch = rasch[['Date', 'vvd_mean % ВВП', 'threshold_vvd', 
                           'sto_y % ВВП', 'threshold_sto']]
            
            # Итерационная корректировка
            if not vvd_params.empty:
                max_iter = 50
                iter_count = 0
                while round(first_vvd, 10) != round(last_trend - std_trend, 10) and iter_count < max_iter:
                    iter_count += 1
                    for i in range(12):
                        if i == 0:
                            new_vvd = last_trend - std_trend
                            first_vvd = new_vvd
                        elif i == 1:
                            new_vvd = last_trend + std_trend
                        else:
                            last_vvd_val = rasch['vvd_mean % ВВП'].where(
                                rasch['Date'] == last_date_know + pd.DateOffset(months=3*(i-1))
                            ).dropna().iloc[0] if not rasch[rasch['Date'] == last_date_know + pd.DateOffset(months=3*(i-1))].empty else last_trend
                            new_vvd = last_vvd_val + vvd_params['Шаги'].values[0]
                        
                        date_reset = last_date_know + pd.DateOffset(months=3*(i+1))
                        if date_reset in rasch['Date'].values:
                            rasch.loc[rasch['Date'] == date_reset, 'vvd_mean % ВВП'] = new_vvd
                    
                    try:
                        cycle, trend = hpfilter(rasch['vvd_mean % ВВП'], 1600)
                        rasch['threshold_vvd'] = pd.Series(trend)
                        rasch['threshold_vvd'] = rasch['threshold_vvd'].where(
                            rasch['Date'] <= last_date_know, np.nan
                        )
                    except Exception as e:
                        break
                    
                    std_trend = rasch['vvd_mean % ВВП'].where(
                        (rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
                        (rasch['Date'] <= last_date_know)
                    ).dropna().std(ddof=1)
                    
                    last_trend = rasch['threshold_vvd'].where(
                        rasch['Date'] == last_date_know
                    ).dropna().iloc[0] if not rasch[rasch['Date'] == last_date_know].empty else last_trend
        
        return rasch


# ============================================================================
# 6. БАНКОВСКИЙ СЕКТОР
# ============================================================================

def calculate_banking_indicators(start_q, now_data_date):
    """Рассчитывает банковские индикаторы"""
    print("Расчет банковских индикаторов...")
    
    bank = start_q[['Date', 'crb_sa', 'gdp_nsa_byn_y', 'Leverage', 
                    'dep_byn_ur', 'dep_byn_fl', 'dep_usd_ur', 'dep_usd_fl']].copy()
    bank = bank.loc[bank['Date'] <= now_data_date]
    
    # Расчет показателей
    bank['crb % ВВП'] = bank['crb_sa'] / bank['gdp_nsa_byn_y'] * 100
    bank['dep_ur'] = bank['dep_byn_ur'] + bank['dep_usd_ur']
    bank['dep_fl'] = bank['dep_byn_fl'] + bank['dep_usd_fl']
    bank['dep'] = bank['dep_fl'] + bank['dep_ur']
    bank['dep_all'] = bank['dep'].rolling(window=2).mean()
    bank['crb_to_dep'] = bank['crb_sa'] / bank['dep_all'] * 100
    
    return bank


def calculate_banking_thresholds(bank_data, params, analysis_mode='actual', target_date=None):
    """Рассчитывает пороговые значения для банковских индикаторов"""
    print("Расчет пороговых значений банковских индикаторов...")
    
    # ============================================================
    # ПРЕДВАРИТЕЛЬНЫЙ РЕЖИМ: все данные используются полностью
    # ============================================================
    if analysis_mode == 'preliminary' and target_date is not None:
        print(f"  Предварительный режим: банковский сектор (все показатели НЕ критические)")
        print(f"  Используются все доступные данные")
        
        # Все банковские показатели НЕ критические, используем полные данные
        bank_non_critical = bank_data[['Date', 'crb % ВВП', 'Leverage', 'crb_to_dep']].copy()
        bank_non_critical = bank_non_critical.dropna()
        bank_non_critical = bank_non_critical[bank_non_critical['Date'] >= pd.Timestamp('2002-01-01')].dropna()
        
        # Удаление строк с пропусками
        while not bank_non_critical.empty and bank_non_critical.iloc[-1].isna().any():
            bank_non_critical = bank_non_critical[:-1]
        
        if bank_non_critical.empty:
            print("  Предупреждение: Нет данных для банковского сектора")
            return pd.DataFrame(columns=['Date', 'crb % ВВП', 'threshold_crb', 
                                         'Leverage', 'threshold_leverage', 
                                         'crb_to_dep', 'threshold_crb_to_dep'])
        
        # --- Расчет пороговых значений для НЕКРИТИЧЕСКИХ показателей ---
        
        # 1. Порог для crb_to_dep (среднее за период с 2016)
        mean_th = bank_non_critical[bank_non_critical['Date'] >= pd.Timestamp(2016, 4, 1)]['crb_to_dep'].dropna().mean()
        bank_non_critical['threshold_crb_to_dep'] = mean_th
        
        # 2. Порог для Leverage (среднее за весь период)
        bank_non_critical['threshold_leverage'] = bank_non_critical['Leverage'].dropna().mean()
        
        # 3. Порог для crb % ВВП (тренд через HP-фильтр)
        last_date_bank = bank_non_critical['Date'].iloc[-1]
        
        # Получаем параметры для прогнозирования CRB
        crb_params = params.loc[
            params['Название показателя'] == "Требования банков к экономике к ВВП, %"
        ]
        
        # Прогнозирование для HP-фильтра (только для crb % ВВП)
        # Используем полные данные, включая последний квартал
        if not crb_params.empty:
            new_rows = []
            for i in range(64):
                if i == 0:
                    new_crb = crb_params['Начальные значения (HP)'].values[0]
                elif i == 1:
                    new_crb = crb_params['Unnamed: 2'].values[0]
                else:
                    last_crb = bank_non_critical['crb % ВВП'].iloc[-2] if len(bank_non_critical) >= 2 else bank_non_critical['crb % ВВП'].iloc[-1]
                    new_crb = last_crb + crb_params['Шаги'].values[0]
                
                new_date_bank = bank_non_critical['Date'].iloc[-1] + pd.DateOffset(months=3)
                new_row = {'Date': new_date_bank, 'crb % ВВП': new_crb}
                new_rows.append(new_row)
                
                bank_non_critical = pd.concat([bank_non_critical, pd.DataFrame(new_rows)], ignore_index=True)
                new_rows = []
        
        # HP-фильтр для CRB
        try:
            cycle, trend = hpfilter(bank_non_critical['crb % ВВП'], 1600)
            bank_non_critical['threshold_crb'] = pd.Series(trend)
            # Оставляем тренд только для известных данных (до последней даты)
            bank_non_critical['threshold_crb'] = bank_non_critical['threshold_crb'].where(
                bank_non_critical['Date'] <= last_date_bank, np.nan
            )
        except Exception as e:
            print(f"  Предупреждение: ошибка при HP-фильтре для crb: {e}")
            bank_non_critical['threshold_crb'] = bank_non_critical['crb % ВВП'].rolling(window=8, min_periods=4).mean()
        
        # Замена NaN в пороговых значениях
        bank_non_critical.loc[bank_non_critical['crb_to_dep'].isna(), 'threshold_crb_to_dep'] = np.nan
        bank_non_critical.loc[bank_non_critical['Leverage'].isna(), 'threshold_leverage'] = np.nan
        
        # Возвращаем результат
        result = bank_non_critical[['Date', 'crb % ВВП', 'threshold_crb', 
                                     'Leverage', 'threshold_leverage', 
                                     'crb_to_dep', 'threshold_crb_to_dep']].copy()
        
        return result
    
    # ============================================================
    # ФАКТИЧЕСКИЙ РЕЖИМ: Обычный расчет (без изменений)
    # ============================================================
    else:
        rasch_bank = bank_data[['Date', 'crb % ВВП', 'Leverage', 'crb_to_dep']].copy()
        
        # Удаление строк с пропусками
        while not rasch_bank.empty and rasch_bank.iloc[-1].isna().any():
            rasch_bank = rasch_bank[:-1]
        
        rasch_bank = rasch_bank[rasch_bank['Date'] >= pd.Timestamp(2002, 1, 1)]
        
        if rasch_bank.empty:
            print("  Предупреждение: Нет данных для банковского сектора")
            return pd.DataFrame(columns=['Date', 'crb % ВВП', 'threshold_crb', 
                                         'Leverage', 'threshold_leverage', 
                                         'crb_to_dep', 'threshold_crb_to_dep'])
        
        # Расчет пороговых значений
        mean_th = rasch_bank[rasch_bank['Date'] >= pd.Timestamp(2016, 4, 1)]['crb_to_dep'].dropna().mean()
        rasch_bank['threshold_crb_to_dep'] = mean_th
        rasch_bank['threshold_leverage'] = rasch_bank['Leverage'].dropna().mean()
        
        # Замена NaN в пороговых значениях
        rasch_bank.loc[rasch_bank['crb_to_dep'].isna(), 'threshold_crb_to_dep'] = np.nan
        rasch_bank.loc[rasch_bank['Leverage'].isna(), 'threshold_leverage'] = np.nan
        
        last_date_bank = rasch_bank['Date'].iloc[-1]
        crb_params = params.loc[
            params['Название показателя'] == "Требования банков к экономике к ВВП, %"
        ]
        
        # Прогнозирование
        if not crb_params.empty:
            new_rows = []
            for i in range(64):
                if i == 0:
                    new_crb = crb_params['Начальные значения (HP)'].values[0]
                elif i == 1:
                    new_crb = crb_params['Unnamed: 2'].values[0]
                else:
                    last_crb = rasch_bank['crb % ВВП'].iloc[-2] if len(rasch_bank) >= 2 else rasch_bank['crb % ВВП'].iloc[-1]
                    new_crb = last_crb + crb_params['Шаги'].values[0]
                
                new_date_bank = rasch_bank['Date'].iloc[-1] + pd.DateOffset(months=3)
                new_row = {'Date': new_date_bank, 'crb % ВВП': new_crb}
                new_rows.append(new_row)
                
                rasch_bank = pd.concat([rasch_bank, pd.DataFrame(new_rows)], ignore_index=True)
                new_rows = []
        
        # HP-фильтр для CRB
        try:
            cycle, trend = hpfilter(rasch_bank['crb % ВВП'], 1600)
            rasch_bank['threshold_crb'] = pd.Series(trend)
            rasch_bank['threshold_crb'] = rasch_bank['threshold_crb'].where(
                rasch_bank['Date'] <= last_date_bank, np.nan
            )
        except Exception as e:
            print(f"  Предупреждение: ошибка при HP-фильтре для crb: {e}")
            rasch_bank['threshold_crb'] = rasch_bank['crb % ВВП'].rolling(window=8, min_periods=4).mean()
        
        # Стандартное отклонение и итерационная корректировка
        std_crb = rasch_bank['crb % ВВП'].where(
            (rasch_bank['Date'] >= pd.Timestamp(2009, 1, 1)) & 
            (rasch_bank['Date'] <= last_date_bank)
        ).dropna().std(ddof=1)
        
        if std_crb is not None and not np.isnan(std_crb) and not crb_params.empty:
            last_crb_series = rasch_bank.loc[rasch_bank['Date'] == last_date_bank, 'threshold_crb'].dropna()
            if not last_crb_series.empty:
                last_crb = last_crb_series.iloc[0]
                first_crb_series = rasch_bank.loc[
                    rasch_bank['Date'] == last_date_bank + pd.DateOffset(months=3), 'crb % ВВП'
                ].dropna()
                if not first_crb_series.empty:
                    first_crb = first_crb_series.iloc[0]
                    
                    max_iter = 50
                    iter_count = 0
                    while round(first_crb, 10) != round(last_crb - std_crb, 10) and iter_count < max_iter:
                        iter_count += 1
                        for i in range(64):
                            if i == 0:
                                new_crb = last_crb - std_crb
                                first_crb = new_crb
                            elif i == 1:
                                new_crb = last_crb + std_crb
                            else:
                                last_crb_val_series = rasch_bank.loc[
                                    rasch_bank['Date'] == last_date_bank + pd.DateOffset(months=3 * (i-1)), 
                                    'crb % ВВП'
                                ].dropna()
                                if not last_crb_val_series.empty:
                                    last_crb_val = last_crb_val_series.iloc[0]
                                else:
                                    last_crb_val = last_crb
                                new_crb = last_crb_val + 0.2295
                            
                            date_reset = last_date_bank + pd.DateOffset(months=3*(i+1))
                            if date_reset in rasch_bank['Date'].values:
                                rasch_bank.loc[rasch_bank['Date'] == date_reset, 'crb % ВВП'] = new_crb
                        
                        try:
                            cycle, trend = hpfilter(rasch_bank['crb % ВВП'], 1600)
                            rasch_bank['threshold_crb'] = pd.Series(trend)
                            rasch_bank['threshold_crb'] = rasch_bank['threshold_crb'].where(
                                rasch_bank['Date'] <= last_date_bank, np.nan
                            )
                        except Exception as e:
                            break
                        
                        std_crb = rasch_bank['crb % ВВП'].where(
                            (rasch_bank['Date'] >= pd.Timestamp(2009, 1, 1)) & 
                            (rasch_bank['Date'] <= last_date_bank)
                        ).dropna().std(ddof=1)
                        
                        last_crb_series = rasch_bank.loc[rasch_bank['Date'] == last_date_bank, 'threshold_crb'].dropna()
                        if not last_crb_series.empty:
                            last_crb = last_crb_series.iloc[0]
                        else:
                            break
        
        return rasch_bank


# ============================================================================
# 7. ДОМАШНИЕ ХОЗЯЙСТВА
# ============================================================================

def calculate_duration_metric(df, loan_type):
    """Рассчитывает дюрацию для потребительских или ипотечных кредитов"""
    if loan_type == 'потреб':
        subset = df.loc[df['Date'] >= pd.Timestamp(2019, 10, 1)]
        
        subset['Погашено, млн. руб.'] = (
            subset['vidan_cred_potreb'] - subset['Прирост задолженности (потреб)']
        )
        subset['среднее за год'] = subset['Погашено, млн. руб.'].rolling(window=4).mean()
        
        subset['дюрация'] = subset.apply(
            lambda row: 0 if row['среднее за год'] < 0 
            else row['Задолженность по потреб. кредитам ФЛ (среднее за год), млн.рублей'] / row['среднее за год'], 
            axis=1
        )
        
        mean_duration = subset.loc[
            (subset['Date'] >= pd.Timestamp(2020, 10, 1)) & 
            (subset['Date'] <= pd.Timestamp(2024, 1, 1)), 'дюрация'
        ].mean()
        
        df['shift'] = df['credits_potreb'].shift(2)
        
        df['Погашено, млн. руб.(потреб)'] = df.apply(
            lambda row: row['shift'] / mean_duration if (row['Date'] <= pd.Timestamp(2019, 10, 1)) 
            else row['vidan_cred_potreb'] - row['Прирост задолженности (потреб)'], 
            axis=1
        )
        
        df['Погашено за год'] = df['Погашено, млн. руб.(потреб)'].rolling(window=4).mean()
        df['Дюрация (потреб.)'] = (
            df['Задолженность по потреб. кредитам ФЛ (среднее за год), млн.рублей'] / 
            df['Погашено за год']
        )
    
    else:  # недвиж
        df['shift'] = df['credits_finans'].shift(2)
        
        df['Погашено, млн. руб.(недвиж)'] = df.apply(
            lambda row: row['shift'] / row['W'] if (row['Date'] <= pd.Timestamp(2019, 10, 1)) 
            else row['vidan_credits_nedv'] - row['Прирост задолженности (недвиж)'], 
            axis=1
        )
        
        df['Погашено за год'] = df['Погашено, млн. руб.(недвиж)'].rolling(window=4).mean()
        df['Дюрация (недвиж.)'] = (
            df['Задолженность по кредитам на недвиж. ФЛ (среднее за год), млн.рублей'] / 
            df['Погашено за год']
        )
    
    df = df.drop(columns=['Погашено за год', 'shift'], errors='ignore')
    return df


def calculate_domestic_indicators(start_q, now_data_date):
    """Рассчитывает индикаторы внутреннего долга и недвижимости"""
    print("Расчет индикаторов внутреннего долга...")
    
    # Подготовка данных
    dom = start_q[['Date', 'gdp_nsa_byn_y', 'credits_potreb', 'credits_finans', 
                   'zadolz_fl_byn', 'zadol_fl_usd', 'AvBi_CPI_q_sa', 'stoim_zil_nedv_q',
                   'precent_stavki_nedviz_cred_q', 'precent_stavki_potreb_cred_q', 
                   'vidan_cred_potreb', 'vidan_credits_nedv']].copy()
    
    dom = dom.loc[
        (dom['Date'] >= pd.Timestamp(2004, 10, 1)) & 
        (dom['Date'] <= now_data_date)
    ]
    
    # Расчет базовых показателей
    dom['credits % ВВП'] = (
        (dom['credits_potreb'] + dom['credits_finans']).rolling(window=2).mean()
    ) / dom['gdp_nsa_byn_y'] * 100
    
    dom['stoim_zil_nedv_sr'] = dom['stoim_zil_nedv_q'] / dom['AvBi_CPI_q_sa']
    dom['stoim_zil_nedv_y'] = (
        dom['stoim_zil_nedv_q'] / dom['AvBi_CPI_q_sa']
    ).rolling(window=4).mean()
    
    mean_nedv = dom.loc[
        (dom['Date'] >= pd.Timestamp(2018, 1, 1)) & 
        (dom['Date'] <= pd.Timestamp(2018, 10, 1)), 'stoim_zil_nedv_y'
    ].mean()
    
    dom['stoim_zil_nedv_id'] = dom['stoim_zil_nedv_y'] / mean_nedv
    dom['ln(stoim_zil_nedv)'] = np.log(dom['stoim_zil_nedv_y'] / mean_nedv) * 100
    
    # Расчет показателей задолженности
    dom['Средняя задолженность по потреб. кредитам ФЛ, млн.рублей'] = (
        dom['credits_potreb'].rolling(window=2).mean()
    )
    
    dom['Средняя задолженность по кредитам на недвиж. ФЛ, млн.рублей'] = (
        dom['credits_finans'].rolling(window=2).mean()
    )
    
    dom['Задолженность по потреб. кредитам ФЛ (среднее за год), млн.рублей'] = (
        dom['credits_potreb'].rolling(window=2).mean()
    ).rolling(window=4).mean()
    
    dom['Задолженность по кредитам на недвиж. ФЛ (среднее за год), млн.рублей'] = (
        dom['credits_finans'].rolling(window=2).mean()
    ).rolling(window=4).mean()
    
    dom['Прирост задолженности (потреб)'] = (
        dom['credits_potreb'] - dom['credits_potreb'].shift(1)
    )
    
    dom['Прирост задолженности (недвиж)'] = (
        dom['credits_finans'] - dom['credits_finans'].shift(1)
    )
    
    # Расчет W (специфический показатель)
    new_data_nedv = [np.nan, 100.0]
    for i in range(2, len(dom)):
        W_next = (
            dom.iloc[i-2]['credits_finans'] / dom.iloc[i-1]['credits_finans'] * 
            (new_data_nedv[i-1] - 1)
        ) + (1 - dom.iloc[i-2]['credits_finans'] / dom.iloc[i-1]['credits_finans']) * new_data_nedv[i-1]
        new_data_nedv.append(W_next)
    
    dom['W'] = pd.Series(new_data_nedv, index=dom.index).shift(1)
    
    # Расчет дюрации
    dom = calculate_duration_metric(dom, 'потреб')
    dom = calculate_duration_metric(dom, 'недвиж')
    
    # Удаление вспомогательных колонок
    dom = dom.drop(columns=['W'], errors='ignore')
    
    # Расчет КОД (Коэффициент обслуживания долга)
    dom['КОД (потреб)'] = (
        dom['precent_stavki_potreb_cred_q'] / 4 * 
        dom['Задолженность по потреб. кредитам ФЛ (среднее за год), млн.рублей'] /
        ((1 - (1 + dom['precent_stavki_potreb_cred_q'] / 400) ** (-dom['Дюрация (потреб.)'])) * 
         dom['gdp_nsa_byn_y'] / 4)
    )
    
    dom['КОД (недвиж)'] = (
        dom['precent_stavki_nedviz_cred_q'] / 4 * 
        dom['Задолженность по кредитам на недвиж. ФЛ (среднее за год), млн.рублей'] /
        ((1 - (1 + dom['precent_stavki_nedviz_cred_q'] / 400) ** (-dom['Дюрация (недвиж.)'])) * 
         dom['gdp_nsa_byn_y'] / 4)
    )
    
    dom['КОД'] = dom['КОД (потреб)'] + dom['КОД (недвиж)']
    
    return dom


def calculate_domestic_thresholds(start_q, now_data_date, params, analysis_mode='actual', target_date=None):
    """Рассчитывает пороговые значения для индикаторов внутреннего долга"""
    print("Расчет пороговых значений внутреннего долга...")
    
    # ============================================================
    # ПРЕДВАРИТЕЛЬНЫЙ РЕЖИМ: все данные используются полностью
    # ============================================================
    if analysis_mode == 'preliminary' and target_date is not None:
        print(f"  Предварительный режим: сектор домашних хозяйств (все показатели НЕ критические)")
        print(f"  Используются все доступные данные")
        
        # Все показатели домохозяйств НЕ критические, используем полные данные
        # Подготовка данных для анализа задолженности (полный период)
        dom1 = start_q[['Date', 'gdp_nsa_byn_y', 'credits_potreb', 'credits_finans', 
                   'zadolz_fl_byn', 'zadol_fl_usd', 'AvBi_CPI_q_sa']].copy()
        
        dom1 = dom1.loc[
            (dom1['Date'] >= pd.Timestamp(2002, 1, 1)) & 
            (dom1['Date'] <= now_data_date)
        ]
        
        # Логарифмические преобразования
        dom1['ln(zadolz % CPI)*100'] = np.log(
            ((dom1['zadolz_fl_byn'] + dom1['zadol_fl_usd']).rolling(window=2).mean()) /
            dom1['AvBi_CPI_q_sa']
        ) * 100
        
        # Расчет различных лагов
        dom1['факт -1 zadolz % CPI'] = dom1['ln(zadolz % CPI)*100'] - dom1['ln(zadolz % CPI)*100'].shift(4)
        dom1['факт -2 zadolz % CPI'] = (
            dom1['ln(zadolz % CPI)*100'] - dom1['ln(zadolz % CPI)*100'].shift(8)
        ) / 2
        
        dom1['факт -3 zadolz % CPI'] = (
            dom1['ln(zadolz % CPI)*100'] - dom1['ln(zadolz % CPI)*100'].shift(12)
        ) / 3

        # Расчет индикаторов (полный период)
        dom = calculate_domestic_indicators(start_q, now_data_date)
        
        # Переименовываем колонку КОД в d_serv_a если она существует
        if 'КОД' in dom.columns:
            dom = dom.rename(columns={'КОД': 'd_serv_a'})
        
        # Проверяем наличие колонки d_serv_a
        if 'd_serv_a' not in dom.columns:
            print("  Предупреждение: колонка 'd_serv_a' не найдена в dom, создаем пустую")
            dom['d_serv_a'] = np.nan
        
        # Объединение данных
        dom_rasch = dom1[['Date', 'ln(zadolz % CPI)*100', 'факт -1 zadolz % CPI', 
                         'факт -2 zadolz % CPI', 'факт -3 zadolz % CPI']].copy()
        
        dom_rasch = pd.merge(
            dom_rasch, 
            dom[['Date', 'd_serv_a', 'credits % ВВП', 'ln(stoim_zil_nedv)']], 
            how='left', 
            on='Date'
        )
        
        # Установка порогов
        dom_rasch = dom_rasch.reset_index(drop=True)
        dom_rasch['threshold_serv'] = 0.55 * 0.4 * 0.5 * 100
        
        last_date_dom = dom_rasch['Date'].iloc[-1]
        
        # Получение параметров
        credit_params = params.loc[params['Название показателя'] == "Кредит ФЛ к ВВП"]
        zadol_params = params.loc[
            params['Название показателя'] == "Логарифм (задолженности ФЛ, скорректированный на ИПЦ)*100"
        ]
        nedv_params = params.loc[
            params['Название показателя'] == "Логарифм индекса цен на жилую недвижимость, скорректированных на ИПЦ, *100"
        ]
        
        # Прогнозирование для HP-фильтра
        if not dom_rasch.empty:
            new_rows = []
            for i in range(13):
                if i == 0:
                    new_cred = credit_params['Начальные значения (HP)'].values[0] if not credit_params.empty else dom_rasch['credits % ВВП'].iloc[-1]
                    new_zadol = zadol_params['Начальные значения (HP)'].values[0] if not zadol_params.empty else dom_rasch['ln(zadolz % CPI)*100'].iloc[-1]
                    new_nedv = nedv_params['Начальные значения (HP)'].values[0] if not nedv_params.empty else dom_rasch['ln(stoim_zil_nedv)'].iloc[-1]
                elif i == 1:
                    new_cred = credit_params['Unnamed: 2'].values[0] if not credit_params.empty else dom_rasch['credits % ВВП'].iloc[-1]
                    new_zadol = zadol_params['Unnamed: 2'].values[0] if not zadol_params.empty else dom_rasch['ln(zadolz % CPI)*100'].iloc[-1]
                    new_nedv = nedv_params['Unnamed: 2'].values[0] if not nedv_params.empty else dom_rasch['ln(stoim_zil_nedv)'].iloc[-1]
                else:
                    last_cred = dom_rasch['credits % ВВП'].iloc[-2] if len(dom_rasch) >= 2 else dom_rasch['credits % ВВП'].iloc[-1]
                    last_zadol = dom_rasch['ln(zadolz % CPI)*100'].iloc[-2] if len(dom_rasch) >= 2 else dom_rasch['ln(zadolz % CPI)*100'].iloc[-1]
                    last_nedv = dom_rasch['ln(stoim_zil_nedv)'].iloc[-2] if len(dom_rasch) >= 2 else dom_rasch['ln(stoim_zil_nedv)'].iloc[-1]
                    new_cred = last_cred + (credit_params['Шаги'].values[0] if not credit_params.empty else 0)
                    new_zadol = last_zadol + (zadol_params['Шаги'].values[0] if not zadol_params.empty else 0)
                    new_nedv = last_nedv + (nedv_params['Шаги'].values[0] if not nedv_params.empty else 0)
                
                new_date = dom_rasch['Date'].iloc[-1] + pd.DateOffset(months=3)
                new_row = {
                    'Date': new_date,
                    'credits % ВВП': new_cred,
                    'ln(zadolz % CPI)*100': new_zadol,
                    'ln(stoim_zil_nedv)': new_nedv
                }
                new_rows.append(new_row)
                
                dom_rasch = pd.concat([dom_rasch, pd.DataFrame(new_rows)], ignore_index=True)
                new_rows = []
            
            # HP-фильтр для выделения трендов
            try:
                credits_clean = dom_rasch['credits % ВВП'].dropna()
                zadol_clean = dom_rasch['ln(zadolz % CPI)*100'].dropna()
                nedv_clean = dom_rasch['ln(stoim_zil_nedv)'].dropna()
                
                if len(credits_clean) > 4:
                    cycle1, trend1 = hpfilter(credits_clean, 1600)
                    dom_rasch['threshold_cred'] = pd.Series(trend1, index=credits_clean.index)
                else:
                    dom_rasch['threshold_cred'] = np.nan
                
                if len(zadol_clean) > 4:
                    cycle2, trend2 = hpfilter(zadol_clean, 1600)
                    dom_rasch['HP_zadolz'] = pd.Series(trend2, index=zadol_clean.index)
                else:
                    dom_rasch['HP_zadolz'] = np.nan
                
                if len(nedv_clean) > 4:
                    cycle3, trend3 = hpfilter(nedv_clean, 1600)
                    dom_rasch['threshold_nedv'] = pd.Series(trend3, index=nedv_clean.index)
                else:
                    dom_rasch['threshold_nedv'] = np.nan
                
                dom_rasch['threshold_cred'] = dom_rasch['threshold_cred'].where(
                    dom_rasch['Date'] <= last_date_dom, np.nan
                )
                dom_rasch['HP_zadolz'] = dom_rasch['HP_zadolz'].where(
                    dom_rasch['Date'] <= last_date_dom, np.nan
                )
                dom_rasch['threshold_nedv'] = dom_rasch['threshold_nedv'].where(
                    dom_rasch['Date'] <= last_date_dom, np.nan
                )
            except Exception as e:
                print(f"  Предупреждение: ошибка при HP-фильтре для domestic: {e}")
                dom_rasch['threshold_cred'] = dom_rasch['credits % ВВП'].rolling(window=8, min_periods=4).mean()
                dom_rasch['HP_zadolz'] = dom_rasch['ln(zadolz % CPI)*100'].rolling(window=8, min_periods=4).mean()
                dom_rasch['threshold_nedv'] = dom_rasch['ln(stoim_zil_nedv)'].rolling(window=8, min_periods=4).mean()
        
        # Расчет лагов для пороговых значений
        dom_rasch['threshold -1 zadolz % CPI'] = (
            dom_rasch['HP_zadolz'] - dom_rasch['HP_zadolz'].shift(4)
        )
        
        dom_rasch['threshold -2 zadolz % CPI'] = (
            dom_rasch['HP_zadolz'] - dom_rasch['HP_zadolz'].shift(8)
        ) / 2
        
        dom_rasch['threshold -3 zadolz % CPI'] = (
            dom_rasch['HP_zadolz'] - dom_rasch['HP_zadolz'].shift(12)
        ) / 3
        
        return dom_rasch, dom, dom1
    
    # ============================================================
    # ФАКТИЧЕСКИЙ РЕЖИМ: Обычный расчет (без изменений)
    # ============================================================
    else:
        # Подготовка данных для анализа задолженности
        dom1 = start_q[['Date', 'gdp_nsa_byn_y', 'credits_potreb', 'credits_finans', 
                   'zadolz_fl_byn', 'zadol_fl_usd', 'AvBi_CPI_q_sa']].copy()
        
        dom1 = dom1.loc[
            (dom1['Date'] >= pd.Timestamp(2002, 1, 1)) & 
            (dom1['Date'] <= now_data_date)
        ]
        
        # Логарифмические преобразования
        dom1['ln(zadolz % CPI)*100'] = np.log(
            ((dom1['zadolz_fl_byn'] + dom1['zadol_fl_usd']).rolling(window=2).mean()) /
            dom1['AvBi_CPI_q_sa']
        ) * 100
        
        # Расчет различных лагов
        dom1['факт -1 zadolz % CPI'] = dom1['ln(zadolz % CPI)*100'] - dom1['ln(zadolz % CPI)*100'].shift(4)
        dom1['факт -2 zadolz % CPI'] = (
            dom1['ln(zadolz % CPI)*100'] - dom1['ln(zadolz % CPI)*100'].shift(8)
        ) / 2
        
        dom1['факт -3 zadolz % CPI'] = (
            dom1['ln(zadolz % CPI)*100'] - dom1['ln(zadolz % CPI)*100'].shift(12)
        ) / 3
        
        # Расчет индикаторов
        dom = calculate_domestic_indicators(start_q, now_data_date)
        
        # Переименовываем колонку КОД в d_serv_a если она существует
        if 'КОД' in dom.columns:
            dom = dom.rename(columns={'КОД': 'd_serv_a'})
        
        # Объединение данных
        dom_rasch = dom1[['Date', 'ln(zadolz % CPI)*100', 'факт -1 zadolz % CPI', 
                         'факт -2 zadolz % CPI', 'факт -3 zadolz % CPI']].copy()
        
        dom_rasch = pd.merge(
            dom_rasch, 
            dom[['Date', 'd_serv_a', 'credits % ВВП', 'ln(stoim_zil_nedv)']], 
            how='left', 
            on='Date'
        )
        
        # Установка порогов
        dom_rasch = dom_rasch.reset_index(drop=True)
        dom_rasch['threshold_serv'] = 0.55 * 0.4 * 0.5 * 100
        
        last_date_dom = dom_rasch['Date'].iloc[-1]
        
        # Получение параметров
        credit_params = params.loc[params['Название показателя'] == "Кредит ФЛ к ВВП"]
        zadol_params = params.loc[
            params['Название показателя'] == "Логарифм (задолженности ФЛ, скорректированный на ИПЦ)*100"
        ]
        nedv_params = params.loc[
            params['Название показателя'] == "Логарифм индекса цен на жилую недвижимость, скорректированных на ИПЦ, *100"
        ]
        
        # Прогнозирование
        new_rows = []
        for i in range(13):
            if i == 0:
                new_cred = credit_params['Начальные значения (HP)'].values[0] if not credit_params.empty else dom_rasch['credits % ВВП'].iloc[-1]
                new_zadol = zadol_params['Начальные значения (HP)'].values[0] if not zadol_params.empty else dom_rasch['ln(zadolz % CPI)*100'].iloc[-1]
                new_nedv = nedv_params['Начальные значения (HP)'].values[0] if not nedv_params.empty else dom_rasch['ln(stoim_zil_nedv)'].iloc[-1]
            elif i == 1:
                new_cred = credit_params['Unnamed: 2'].values[0] if not credit_params.empty else dom_rasch['credits % ВВП'].iloc[-1]
                new_zadol = zadol_params['Unnamed: 2'].values[0] if not zadol_params.empty else dom_rasch['ln(zadolz % CPI)*100'].iloc[-1]
                new_nedv = nedv_params['Unnamed: 2'].values[0] if not nedv_params.empty else dom_rasch['ln(stoim_zil_nedv)'].iloc[-1]
            else:
                last_cred = dom_rasch['credits % ВВП'].iloc[-2] if len(dom_rasch) >= 2 else dom_rasch['credits % ВВП'].iloc[-1]
                last_zadol = dom_rasch['ln(zadolz % CPI)*100'].iloc[-2] if len(dom_rasch) >= 2 else dom_rasch['ln(zadolz % CPI)*100'].iloc[-1]
                last_nedv = dom_rasch['ln(stoim_zil_nedv)'].iloc[-2] if len(dom_rasch) >= 2 else dom_rasch['ln(stoim_zil_nedv)'].iloc[-1]
                new_cred = last_cred + (credit_params['Шаги'].values[0] if not credit_params.empty else 0)
                new_zadol = last_zadol + (zadol_params['Шаги'].values[0] if not zadol_params.empty else 0)
                new_nedv = last_nedv + (nedv_params['Шаги'].values[0] if not nedv_params.empty else 0)
            
            new_date = dom_rasch['Date'].iloc[-1] + pd.DateOffset(months=3)
            new_row = {
                'Date': new_date,
                'credits % ВВП': new_cred,
                'ln(zadolz % CPI)*100': new_zadol,
                'ln(stoim_zil_nedv)': new_nedv
            }
            new_rows.append(new_row)
            
            dom_rasch = pd.concat([dom_rasch, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows = []
        
        # HP-фильтр для выделения трендов
        try:
            credits_clean = dom_rasch['credits % ВВП'].dropna()
            zadol_clean = dom_rasch['ln(zadolz % CPI)*100'].dropna()
            nedv_clean = dom_rasch['ln(stoim_zil_nedv)'].dropna()
            
            if len(credits_clean) > 4:
                cycle1, trend1 = hpfilter(credits_clean, 1600)
                dom_rasch['threshold_cred'] = pd.Series(trend1, index=credits_clean.index)
            else:
                dom_rasch['threshold_cred'] = np.nan
            
            if len(zadol_clean) > 4:
                cycle2, trend2 = hpfilter(zadol_clean, 1600)
                dom_rasch['HP_zadolz'] = pd.Series(trend2, index=zadol_clean.index)
            else:
                dom_rasch['HP_zadolz'] = np.nan
            
            if len(nedv_clean) > 4:
                cycle3, trend3 = hpfilter(nedv_clean, 1600)
                dom_rasch['threshold_nedv'] = pd.Series(trend3, index=nedv_clean.index)
            else:
                dom_rasch['threshold_nedv'] = np.nan
            
            dom_rasch['threshold_cred'] = dom_rasch['threshold_cred'].where(
                dom_rasch['Date'] <= last_date_dom, np.nan
            )
            dom_rasch['HP_zadolz'] = dom_rasch['HP_zadolz'].where(
                dom_rasch['Date'] <= last_date_dom, np.nan
            )
            dom_rasch['threshold_nedv'] = dom_rasch['threshold_nedv'].where(
                dom_rasch['Date'] <= last_date_dom, np.nan
            )
        except Exception as e:
            print(f"  Предупреждение: ошибка при HP-фильтре для domestic: {e}")
            dom_rasch['threshold_cred'] = dom_rasch['credits % ВВП'].rolling(window=8, min_periods=4).mean()
            dom_rasch['HP_zadolz'] = dom_rasch['ln(zadolz % CPI)*100'].rolling(window=8, min_periods=4).mean()
            dom_rasch['threshold_nedv'] = dom_rasch['ln(stoim_zil_nedv)'].rolling(window=8, min_periods=4).mean()
        
        # Расчет стандартных отклонений и итерационные корректировки
        std_cred = dom_rasch.loc[
            (dom_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
            (dom_rasch['Date'] <= last_date_dom), 'credits % ВВП'
        ].dropna().std(ddof=1)
        
        if std_cred is not None and not np.isnan(std_cred) and not credit_params.empty:
            last_cred_series = dom_rasch.loc[dom_rasch['Date'] == last_date_dom, 'threshold_cred'].dropna()
            if not last_cred_series.empty:
                last_cred = last_cred_series.iloc[0]
                first_cred_series = dom_rasch.loc[
                    dom_rasch['Date'] == (last_date_dom + pd.DateOffset(months=3)), 'credits % ВВП'
                ].dropna()
                if not first_cred_series.empty:
                    first_cred = first_cred_series.iloc[0]
                    max_iter = 50
                    iter_count = 0
                    while round(first_cred, 10) != round(last_cred - std_cred, 10) and iter_count < max_iter:
                        iter_count += 1
                        for i in range(13):
                            if i == 0:
                                new_cred = last_cred - std_cred
                                first_cred = new_cred
                            elif i == 1:
                                new_cred = last_cred + std_cred
                            else:
                                last_cred_val_series = dom_rasch.loc[
                                    dom_rasch['Date'] == (last_date_dom + pd.DateOffset(months=3 * (i-1))), 
                                    'credits % ВВП'
                                ].dropna()
                                if not last_cred_val_series.empty:
                                    last_cred_val = last_cred_val_series.iloc[0]
                                else:
                                    last_cred_val = last_cred
                                new_cred = last_cred_val + (credit_params['Шаги'].values[0] if not credit_params.empty else 0)
                            
                            date_reset = last_date_dom + pd.DateOffset(months=3*(i+1))
                            if date_reset in dom_rasch['Date'].values:
                                dom_rasch.loc[dom_rasch['Date'] == date_reset, 'credits % ВВП'] = new_cred
                        
                        try:
                            credits_clean = dom_rasch['credits % ВВП'].dropna()
                            if len(credits_clean) > 4:
                                cycle1, trend1 = hpfilter(credits_clean, 1600)
                                dom_rasch['threshold_cred'] = pd.Series(trend1, index=credits_clean.index)
                                dom_rasch['threshold_cred'] = dom_rasch['threshold_cred'].where(
                                    dom_rasch['Date'] <= last_date_dom, np.nan
                                )
                        except Exception as e:
                            break
                        
                        std_cred = dom_rasch.loc[
                            (dom_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
                            (dom_rasch['Date'] <= last_date_dom), 'credits % ВВП'
                        ].dropna().std(ddof=1)
                        
                        last_cred_series = dom_rasch.loc[dom_rasch['Date'] == last_date_dom, 'threshold_cred'].dropna()
                        if not last_cred_series.empty:
                            last_cred = last_cred_series.iloc[0]
                        else:
                            break
        
        # Стандартное отклонение для zadol (задолженность)
        std_zadol = dom_rasch.loc[
            (dom_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
            (dom_rasch['Date'] <= last_date_dom), 'ln(zadolz % CPI)*100'
        ].dropna().std(ddof=1)
        
        if std_zadol is not None and not np.isnan(std_zadol) and not zadol_params.empty:
            last_zadol_series = dom_rasch.loc[dom_rasch['Date'] == last_date_dom, 'HP_zadolz'].dropna()
            if not last_zadol_series.empty:
                last_zadol = last_zadol_series.iloc[0]
                first_zadol_series = dom_rasch.loc[
                    dom_rasch['Date'] == (last_date_dom + pd.DateOffset(months=3)), 'ln(zadolz % CPI)*100'
                ].dropna()
                if not first_zadol_series.empty:
                    first_zadol = first_zadol_series.iloc[0]
                    max_iter = 50
                    iter_count = 0
                    while round(first_zadol, 10) != round(last_zadol - std_zadol, 10) and iter_count < max_iter:
                        iter_count += 1
                        for i in range(13):
                            if i == 0:
                                new_zadol = last_zadol - std_zadol
                                first_zadol = new_zadol
                            elif i == 1:
                                new_zadol = last_zadol + std_zadol
                            else:
                                last_zadol_val_series = dom_rasch.loc[
                                    dom_rasch['Date'] == (last_date_dom + pd.DateOffset(months=3 * (i-1))), 
                                    'ln(zadolz % CPI)*100'
                                ].dropna()
                                if not last_zadol_val_series.empty:
                                    last_zadol_val = last_zadol_val_series.iloc[0]
                                else:
                                    last_zadol_val = last_zadol
                                new_zadol = last_zadol_val + (zadol_params['Шаги'].values[0] if not zadol_params.empty else 0)
                            
                            date_reset = last_date_dom + pd.DateOffset(months=3*(i+1))
                            if date_reset in dom_rasch['Date'].values:
                                dom_rasch.loc[dom_rasch['Date'] == date_reset, 'ln(zadolz % CPI)*100'] = new_zadol
                        
                        try:
                            zadol_clean = dom_rasch['ln(zadolz % CPI)*100'].dropna()
                            if len(zadol_clean) > 4:
                                cycle2, trend2 = hpfilter(zadol_clean, 1600)
                                dom_rasch['HP_zadolz'] = pd.Series(trend2, index=zadol_clean.index)
                                dom_rasch['HP_zadolz'] = dom_rasch['HP_zadolz'].where(
                                    dom_rasch['Date'] <= last_date_dom, np.nan
                                )
                        except Exception as e:
                            break
                        
                        std_zadol = dom_rasch.loc[
                            (dom_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
                            (dom_rasch['Date'] <= last_date_dom), 'ln(zadolz % CPI)*100'
                        ].dropna().std(ddof=1)
                        
                        last_zadol_series = dom_rasch.loc[dom_rasch['Date'] == last_date_dom, 'HP_zadolz'].dropna()
                        if not last_zadol_series.empty:
                            last_zadol = last_zadol_series.iloc[0]
                        else:
                            break
        
        # Стандартное отклонение для nedv (недвижимость)
        std_nedv = dom_rasch.loc[
            (dom_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
            (dom_rasch['Date'] <= last_date_dom), 'ln(stoim_zil_nedv)'
        ].dropna().std(ddof=1)
        
        if std_nedv is not None and not np.isnan(std_nedv) and not nedv_params.empty:
            last_nedv_series = dom_rasch.loc[dom_rasch['Date'] == last_date_dom, 'threshold_nedv'].dropna()
            if not last_nedv_series.empty:
                last_nedv = last_nedv_series.iloc[0]
                first_nedv_series = dom_rasch.loc[
                    dom_rasch['Date'] == (last_date_dom + pd.DateOffset(months=3)), 'ln(stoim_zil_nedv)'
                ].dropna()
                if not first_nedv_series.empty:
                    first_nedv = first_nedv_series.iloc[0]
                    max_iter = 50
                    iter_count = 0
                    while round(first_nedv, 10) != round(last_nedv - std_nedv, 10) and iter_count < max_iter:
                        iter_count += 1
                        for i in range(13):
                            if i == 0:
                                new_nedv = last_nedv - std_nedv
                                first_nedv = new_nedv
                            elif i == 1:
                                new_nedv = last_nedv + std_nedv
                            else:
                                last_nedv_val_series = dom_rasch.loc[
                                    dom_rasch['Date'] == (last_date_dom + pd.DateOffset(months=3 * (i-1))), 
                                    'ln(stoim_zil_nedv)'
                                ].dropna()
                                if not last_nedv_val_series.empty:
                                    last_nedv_val = last_nedv_val_series.iloc[0]
                                else:
                                    last_nedv_val = last_nedv
                                new_nedv = last_nedv_val + (nedv_params['Шаги'].values[0] if not nedv_params.empty else 0)
                            
                            date_reset = last_date_dom + pd.DateOffset(months=3*(i+1))
                            if date_reset in dom_rasch['Date'].values:
                                dom_rasch.loc[dom_rasch['Date'] == date_reset, 'ln(stoim_zil_nedv)'] = new_nedv
                        
                        try:
                            nedv_clean = dom_rasch['ln(stoim_zil_nedv)'].dropna()
                            if len(nedv_clean) > 4:
                                cycle3, trend3 = hpfilter(nedv_clean, 1600)
                                dom_rasch['threshold_nedv'] = pd.Series(trend3, index=nedv_clean.index)
                                dom_rasch['threshold_nedv'] = dom_rasch['threshold_nedv'].where(
                                    dom_rasch['Date'] <= last_date_dom, np.nan
                                )
                        except Exception as e:
                            break
                        
                        std_nedv = dom_rasch.loc[
                            (dom_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
                            (dom_rasch['Date'] <= last_date_dom), 'ln(stoim_zil_nedv)'
                        ].dropna().std(ddof=1)
                        
                        last_nedv_series = dom_rasch.loc[dom_rasch['Date'] == last_date_dom, 'threshold_nedv'].dropna()
                        if not last_nedv_series.empty:
                            last_nedv = last_nedv_series.iloc[0]
                        else:
                            break
        
        # Расчет лагов для пороговых значений
        dom_rasch['threshold -1 zadolz % CPI'] = (
            dom_rasch['HP_zadolz'] - dom_rasch['HP_zadolz'].shift(4)
        )
        
        dom_rasch['threshold -2 zadolz % CPI'] = (
            dom_rasch['HP_zadolz'] - dom_rasch['HP_zadolz'].shift(8)
        ) / 2
        
        dom_rasch['threshold -3 zadolz % CPI'] = (
            dom_rasch['HP_zadolz'] - dom_rasch['HP_zadolz'].shift(12)
        ) / 3
        
        return dom_rasch, dom, dom1


# ============================================================================
# 8. ЮРИДИЧЕСКИЕ ЛИЦА
# ============================================================================

def prepare_corporate_data(start_q, rates, now_data_date):
    """Подготавливает данные для анализа юридических лиц"""
    print("Подготовка данных для анализа юридических лиц...")
    
    # Объединение с данными о курсе валют
    start_q = pd.merge(start_q, rates, on='Date', how='left')
    start_q['BYR/USD'] = start_q['BYR/USD'].shift(-1)
    
    # Выбор нужных колонок
    ur_licha = start_q[['Date', 'price_com_nedv', 'zap_got_prod', 'nGDP_q', 'GDP_q', 
                       'zadolz_ul_crat_byn', 'vidan_cred_ul_crat_byn', 'zadolz_ul_dolg_byn',
                       'vidan_cred_ul_dolg_byn', 'zadolz_ul_crat_usd', 'vidan_cred_ul_crat_usd', 
                       'zadolz_ul_dolg_usd', 'vidan_cred_ul_dolg_usd', 'NER_usdbyn_weight',
                       'vvd_crat', 'vvd_dolg', 'BYR/USD', 'precent_stavki_ul_byn_q', 
                       'gdp_nsa_byn_y', 'precent_stavki_ul_skv_q', 'pogash_kk_n', 
                       'pogash_dk_n', 'IntPymt_NRes_q_thUSD', 'a', 'b']].copy()
    
    # Фильтрация по датам
    ur_licha = ur_licha.loc[
        (ur_licha['Date'] >= pd.Timestamp(2001, 1, 1)) & 
        (ur_licha['Date'] <= now_data_date)
    ]
    
    return ur_licha


def calculate_real_estate_indicators(ur_licha):
    """Рассчитывает индикаторы коммерческой недвижимости"""
    print("Расчет индикаторов коммерческой недвижимости...")
    
    # Сезонная корректировка цен на коммерческую недвижимость
    res = seasonal_decompose(ur_licha['price_com_nedv'].dropna(), period=4, model='additive')
    ur_licha['price_com_nedv_sa'] = ur_licha['price_com_nedv'] - res.seasonal
    
    # Расчет дефлятора ВВП
    ur_licha['defgdp'] = ur_licha['nGDP_q'] / ur_licha['GDP_q']
    
    # Сезонная корректировка дефлятора
    defgdp_seasonal = seasonal_decompose(ur_licha['defgdp'].dropna(), period=4).seasonal
    ur_licha['defgdp_sa'] = ur_licha['defgdp'] - defgdp_seasonal
    
    # Расчет реальных цен на недвижимость
    ur_licha['среднее за квартал (price)'] = ur_licha['price_com_nedv_sa'] / ur_licha['defgdp_sa']
    ur_licha['среднее за год (price)'] = ur_licha['среднее за квартал (price)'].rolling(window=4).mean()
    
    # Базовый период 2018
    mean_2018 = ur_licha.loc[
        (ur_licha['Date'] >= pd.Timestamp(2018, 1, 1)) & 
        (ur_licha['Date'] <= pd.Timestamp(2018, 10, 1)), 
        'среднее за год (price)'
    ].mean()
    
    ur_licha['Цены на коммерческую недвижимость, руб за м.кв. (скорректированные на дефлятор, 2018=1)'] = (
        ur_licha['среднее за год (price)'] / mean_2018
    )
    
    ur_licha['cest'] = np.log(
        ur_licha['Цены на коммерческую недвижимость, руб за м.кв. (скорректированные на дефлятор, 2018=1)']
    ) * 100
    
    # Индикатор инвестиций
    ur_licha['invr'] = ur_licha['zap_got_prod'].rolling(window=2).mean()
    
    return ur_licha


def calculate_credit_indicators(ur_licha):
    """Рассчитывает кредитные индикаторы для юридических лиц"""
    print("Расчет кредитных индикаторов для ЮЛ...")
    
    # ВВД по кредитам от нерезидентов
    ur_licha['ВВД по кредитам займам прочих секторов (среднее за период)'] = (
        (ur_licha['vvd_crat'].rolling(window=2).mean() + 
         ur_licha['vvd_dolg'].rolling(window=2).mean()) * ur_licha['NER_usdbyn_weight']
    )
    
    # Общая задолженность ЮЛ перед банками
    ur_licha['Задолж. ЮЛ перед банками, среднее за кв.'] = (
        ur_licha['zadolz_ul_crat_byn'] + ur_licha['zadolz_ul_dolg_byn'] + 
        ur_licha['zadolz_ul_crat_usd'] + ur_licha['zadolz_ul_dolg_usd']
    ).rolling(window=2).mean()
    
    # Альтернативный расчет задолженности
    ur_licha['a'] = (ur_licha['a'] + ur_licha['b']).rolling(window=2).mean()
    
    ur_licha['Задолж. ЮЛ перед банками, среднее за кв.'] = np.where(
        ur_licha['Задолж. ЮЛ перед банками, среднее за кв.'].notna(),
        ur_licha['Задолж. ЮЛ перед банками, среднее за кв.'],
        ur_licha['a']
    )
    
    # Итого кредиты от банков и нерезидентов
    ur_licha['Итого банки и нерезиденты, (2018, скорр. на дефлятор)'] = (
        ur_licha['Задолж. ЮЛ перед банками, среднее за кв.'] + 
        ur_licha['ВВД по кредитам займам прочих секторов (среднее за период)']
    ) / ur_licha['defgdp_sa']

    # Логарифмические преобразования
    ur_licha['ln_cred*100'] = np.log(ur_licha['Итого банки и нерезиденты, (2018, скорр. на дефлятор)']) * 100
    
    # Отношение долга к ВВП
    ur_licha['ggr'] = (
        ur_licha['Задолж. ЮЛ перед банками, среднее за кв.'] + 
        ur_licha['ВВД по кредитам займам прочих секторов (среднее за период)']
    ) / ur_licha['nGDP_q'].rolling(window=4).sum() * 100
    
    return ur_licha


def adjust_currency_values(ur_licha):
    """Корректирует значения в иностранной валюте"""
    print("Корректировка значений в иностранной валюте...")
    
    # Конвертация валютных значений
    ur_licha['zadolz_ul_crat_usd2'] = ur_licha['zadolz_ul_crat_usd'] / ur_licha['BYR/USD']
    ur_licha['vidan_cred_ul_crat_usd2'] = ur_licha['vidan_cred_ul_crat_usd'] / ur_licha['NER_usdbyn_weight']
    ur_licha['zadolz_ul_dolg_usd2'] = ur_licha['zadolz_ul_dolg_usd'] / ur_licha['BYR/USD']
    ur_licha['vidan_cred_ul_dolg_usd2'] = ur_licha['vidan_cred_ul_dolg_usd'] / ur_licha['NER_usdbyn_weight']
    
    return ur_licha


def calculate_duration_metrics_corporate(ur_licha):
    """Рассчитывает метрики дюрации для различных видов кредитов юридических лиц"""
    print("Расчет метрик дюрации для ЮЛ...")
    
    # Функция для расчета дюрации
    def calculate_loan_duration(df, zadol_col, vidan_col, loan_type, currency_type, term_type):
        
        df[f'Прирост задолж. {loan_type}, {currency_type}, {term_type}.'] = (
            df[zadol_col] - df[zadol_col].shift(1)
        )
        
        df[f'Погашено, {loan_type}, {currency_type}, {term_type}.'] = (
            df[vidan_col] - df[f'Прирост задолж. {loan_type}, {currency_type}, {term_type}.']
        )
        
        df[f'Средняя задолж., {loan_type}, {currency_type}, {term_type}.'] = (
            df[zadol_col].rolling(window=2).mean()
        )
        
        df['ср_п'] = df[f'Погашено, {loan_type}, {currency_type}, {term_type}.'].rolling(window=4).mean()
        df['ср_з'] = df[f'Средняя задолж., {loan_type}, {currency_type}, {term_type}.'].rolling(window=4).mean()
        
        df[f'Средняя задолж., {loan_type}, {currency_type}, {term_type}. (среднее за год)'] = (
            df[f'Средняя задолж., {loan_type}, {currency_type}, {term_type}.'].rolling(window=4).mean()
        )
        
        if term_type == 'К':  # Краткосрочные
            df[f'Дюрация приведенная ({loan_type}, {currency_type}, {term_type}, банки), кварталов'] = df.apply(
                lambda row: 0 if row['ср_п'] < 0 else row['ср_з'] / (row['ср_п'] / 4), 
                axis=1
            )
        else:  # Долгосрочные
            df[f'Дюрация приведенная ({loan_type}, {currency_type}, {term_type}, банки), кварталов'] = df.apply(
                lambda row: 0 if row['ср_п'] < 0 else row['ср_з'] / row['ср_п'], 
                axis=1
            )
        
        return df
    
    # Расчет дюрации для разных типов кредитов
    ur_licha = calculate_loan_duration(ur_licha, 'zadolz_ul_crat_byn', 'vidan_cred_ul_crat_byn', 'ЮЛ', 'НВ', 'К')
    ur_licha = calculate_loan_duration(ur_licha, 'zadolz_ul_dolg_byn', 'vidan_cred_ul_dolg_byn', 'ЮЛ', 'НВ', 'Д')
    ur_licha = calculate_loan_duration(ur_licha, 'zadolz_ul_crat_usd2', 'vidan_cred_ul_crat_usd2', 'ЮЛ', 'ИВ', 'К')
    ur_licha = calculate_loan_duration(ur_licha, 'zadolz_ul_dolg_usd2', 'vidan_cred_ul_dolg_usd2', 'ЮЛ', 'ИВ', 'Д')
    
    return ur_licha


def adjust_historical_durations(ur_licha):
    """Корректирует исторические значения дюраций"""
    print("Корректировка исторических значений дюраций...")
    
    # Корректировки для 2007 года
    ur_licha['ср_п'] = ur_licha['Погашено, ЮЛ, НВ, Д.'].rolling(window=8).mean()
    ur_licha['ср_з'] = ur_licha['Средняя задолж., ЮЛ, НВ, Д.'].rolling(window=8).mean()
    
    ur_licha.loc[
        ur_licha['Date'] == pd.Timestamp(2007, 1, 1), 
        'Дюрация приведенная (ЮЛ, НВ, Д, банки), кварталов'
    ] = (
        ur_licha.loc[ur_licha['Date'] == pd.Timestamp(2007, 1, 1), 'ср_з'] / 
        ur_licha.loc[ur_licha['Date'] == pd.Timestamp(2007, 1, 1), 'ср_п']
    )
    
    # Корректировки для 2005 года
    date_mask = (ur_licha['Date'] < pd.Timestamp(2005, 10, 1)) & (ur_licha['Date'] >= pd.Timestamp(2005, 1, 1))
    
    ur_licha.loc[date_mask, 'Дюрация приведенная (ЮЛ, ИВ, Д, банки), кварталов'] = 8.0
    ur_licha.loc[date_mask, 'Дюрация приведенная (ЮЛ, ИВ, К, банки), кварталов'] = 4.0
    ur_licha.loc[date_mask, 'Дюрация приведенная (ЮЛ, НВ, Д, банки), кварталов'] = 5.0
    ur_licha.loc[date_mask, 'Дюрация приведенная (ЮЛ, НВ, К, банки), кварталов'] = 4.0
    
    return ur_licha


def calculate_debt_service_ratio_corporate(ur_licha):
    """Рассчитывает коэффициенты обслуживания долга (КОД) для юридических лиц"""
    print("Расчет коэффициентов обслуживания долга для ЮЛ...")
    
    # КОД для национальной валюты, краткосрочные
    ur_licha['КОД (ЮЛ, К, НВ)'] = (
        ur_licha['precent_stavki_ul_byn_q'] / 4 * 
        ur_licha['Средняя задолж., ЮЛ, НВ, К.'].rolling(window=4).mean() /
        ((1 - (1 + ur_licha['precent_stavki_ul_byn_q'] / 400) ** 
          (-ur_licha['Дюрация приведенная (ЮЛ, НВ, К, банки), кварталов'])) * 
         (ur_licha['gdp_nsa_byn_y']) / 4)
    )
    
    # КОД для национальной валюты, долгосрочные
    ur_licha['КОД (ЮЛ, Д, НВ)'] = (
        ur_licha['precent_stavki_ul_byn_q'] / 4 * 
        ur_licha['Средняя задолж., ЮЛ, НВ, Д.'].rolling(window=4).mean() /
        ((1 - (1 + ur_licha['precent_stavki_ul_byn_q'] / 400) ** 
          (-ur_licha['Дюрация приведенная (ЮЛ, НВ, Д, банки), кварталов'])) * 
         (ur_licha['gdp_nsa_byn_y']) / 4)
    )
    
    # Рублевая доходность для иностранной валюты
    ur_licha['Рублевая доходность кредитов банков юрлицам в ИВ, %'] = (
        ur_licha['precent_stavki_ul_skv_q'] * ur_licha['NER_usdbyn_weight'] / 
        ur_licha['NER_usdbyn_weight'].shift(1)
    )
    
    # КОД для иностранной валюты, краткосрочные
    ur_licha['КОД (ЮЛ, К, ИВ)'] = (
        ur_licha['Рублевая доходность кредитов банков юрлицам в ИВ, %'] / 4 * 
        (ur_licha['zadolz_ul_crat_usd'].rolling(window=2).mean()).rolling(window=4).mean() /
        ((1 - (1 + ur_licha['Рублевая доходность кредитов банков юрлицам в ИВ, %'] / 400) ** 
          (-ur_licha['Дюрация приведенная (ЮЛ, ИВ, К, банки), кварталов'])) * 
         ur_licha['gdp_nsa_byn_y'] / 4)
    )
    
    # КОД для иностранной валюты, долгосрочные
    ur_licha['КОД (ЮЛ, Д, ИВ)'] = (
        ur_licha['Рублевая доходность кредитов банков юрлицам в ИВ, %'] / 4 * 
        (ur_licha['zadolz_ul_dolg_usd'].rolling(window=2).mean()).rolling(window=4).mean() /
        ((1 - (1 + ur_licha['Рублевая доходность кредитов банков юрлицам в ИВ, %'] / 400) ** 
          (-ur_licha['Дюрация приведенная (ЮЛ, ИВ, Д, банки), кварталов'])) * 
         ur_licha['gdp_nsa_byn_y'] / 4)
    )
    
    return ur_licha


def calculate_non_resident_indicators(ur_licha):
    """Рассчитывает индикаторы для кредитов от нерезидентов"""
    print("Расчет индикаторов для кредитов от нерезидентов...")
    
    # Годовые выплаты нерезидентам
    ur_licha['IntPymt_NRes_q_thUSD_yy'] = ur_licha['IntPymt_NRes_q_thUSD'].rolling(window=4).sum()
    
    # Средняя задолженность перед нерезидентами
    ur_licha['Задолженность (ЮЛ, Н, К, ИВ, ср за год)'] = (
        ur_licha['vvd_crat'].rolling(window=2).mean()
    ).rolling(window=4).mean()
    
    ur_licha['Задолженность (ЮЛ, Н, Д, ИВ, ср за год)'] = (
        ur_licha['vvd_dolg'].rolling(window=2).mean()
    ).rolling(window=4).mean()
    
    # Процентная ставка по кредитам нерезидентов
    ur_licha['Процентная ставка по кредитам банков нерезидентов в ИВ, %'] = (
        ur_licha['IntPymt_NRes_q_thUSD_yy'] / 
        (ur_licha['Задолженность (ЮЛ, Н, Д, ИВ, ср за год)'] + 
         ur_licha['Задолженность (ЮЛ, Н, К, ИВ, ср за год)']) * 100
    )
    
    ur_licha['Процентная ставка по кредитам банков нерезидентов в ИВ, %'] = (
        ur_licha['Процентная ставка по кредитам банков нерезидентов в ИВ, %'].bfill()
    )
    
    # Рублевая доходность
    ur_licha['Рублевая доходность кредитов нерезидентов юрлицам в ИВ, %'] = (
        ur_licha['Процентная ставка по кредитам банков нерезидентов в ИВ, %'] * 
        ur_licha['NER_usdbyn_weight'] / ur_licha['NER_usdbyn_weight'].shift(1)
    )
    
    # Задолженность в рублевом эквиваленте
    ur_licha['Задолженность (ЮЛ, Н, К, конец периода)'] = (
        ur_licha['vvd_crat'] * ur_licha['BYR/USD']
    )
    
    ur_licha['Задолж. (ЮЛ, Н, К, ИВ, кредиты, ср за год, эквив)'] = (
        ur_licha['Задолженность (ЮЛ, Н, К, конец периода)'].rolling(window=2).mean()
    ).rolling(window=4).mean()
    
    ur_licha['Задолженность (ЮЛ, Н, Д, конец периода)'] = (
        ur_licha['vvd_dolg'] * ur_licha['BYR/USD']
    )
    
    ur_licha['Задолж. (ЮЛ, Н, Д, ИВ, кредиты, ср за год, эквив)'] = (
        ur_licha['Задолженность (ЮЛ, Н, Д, конец периода)'].rolling(window=2).mean()
    ).rolling(window=4).mean()
    
    # Дюрация для кредитов нерезидентов
    ur_licha['Средняя задолженность по краткосрочным кредитам ЮЛ в ИВ, млн долл США'] = (
        ur_licha['vvd_crat'].rolling(window=2).mean()
    )
    
    ur_licha['ср_п'] = ur_licha['pogash_kk_n'].rolling(window=4).mean()
    ur_licha['ср_з'] = ur_licha['Средняя задолженность по краткосрочным кредитам ЮЛ в ИВ, млн долл США'].rolling(window=4).mean()
    
    
    ur_licha['Прирост задолженности (ЮЛ, Н, К)'] = ur_licha['vvd_crat'] - ur_licha['vvd_crat'].shift(1)
    ur_licha['Выдано (ЮЛ, Н, К)'] = ur_licha['Прирост задолженности (ЮЛ, Н, К)'] + ur_licha['pogash_kk_n']
    
    ur_licha['Средняя задолженность по краткосрочным кредитам ЮЛ в ИВ, млн долл США (среднее за год)'] = (
        ur_licha['Средняя задолженность по краткосрочным кредитам ЮЛ в ИВ, млн долл США'].rolling(window=4).mean()
    )
    
    ur_licha['Дюрация приведенная (ЮЛ, ИВ, К, Н), кварталов'] = ur_licha.apply(
        lambda row: 0 if row['ср_п'] < 0 else row['ср_з'] / (row['ср_п'] / 4), 
        axis=1
    )
    
    # Корректировки для 2005 года
    ur_licha.loc[
        (ur_licha['Date'] >= '2005-01-01') & (ur_licha['Date'] <= '2005-07-01'), 
        'Дюрация приведенная (ЮЛ, ИВ, К, Н), кварталов'
    ] = 6.0
    
    ur_licha['Приведенное погашение задолженности (ЮЛ, Н, К)'] = (
        ur_licha['pogash_kk_n'] / ur_licha['Дюрация приведенная (ЮЛ, ИВ, К, Н), кварталов']
    )
    
    # Долгосрочные кредиты
    ur_licha['Средняя задолженность по долгосрочным кредитам ЮЛ в ИВ, млн долл США'] = (
        ur_licha['vvd_dolg'].rolling(window=2).mean()
    )
    
    ur_licha['ср_п'] = ur_licha['pogash_dk_n'].rolling(window=4).mean()
    ur_licha['ср_з'] = ur_licha['Средняя задолженность по долгосрочным кредитам ЮЛ в ИВ, млн долл США'].rolling(window=4).mean()
    
    
    ur_licha['Прирост задолженности (ЮЛ, Н, Д)'] = ur_licha['vvd_dolg'] - ur_licha['vvd_dolg'].shift(1)
    ur_licha['Выдано (ЮЛ, Н, Д)'] = ur_licha['Прирост задолженности (ЮЛ, Н, Д)'] + ur_licha['pogash_dk_n']
    
    ur_licha['Средняя задолженность по долгосрочным кредитам ЮЛ в ИВ, млн долл США (среднее за год)'] = (
        ur_licha['Средняя задолженность по долгосрочным кредитам ЮЛ в ИВ, млн долл США'].rolling(window=4).mean()
    )
    
    ur_licha['Дюрация приведенная (ЮЛ, ИВ, Д, Н), кварталов'] = ur_licha.apply(
        lambda row: 0 if row['ср_п'] < 0 else row['ср_з'] / row['ср_п'], 
        axis=1
    )
    
    ur_licha.loc[
        (ur_licha['Date'] >= '2005-01-01') & (ur_licha['Date'] <= '2005-07-01'), 
        'Дюрация приведенная (ЮЛ, ИВ, Д, Н), кварталов'
    ] = 8.0
    
    # КОД для кредитов нерезидентов
    ur_licha['КОД (ЮЛ, К, ИВ, Н)'] = (
        ur_licha['Рублевая доходность кредитов нерезидентов юрлицам в ИВ, %'] / 4 * 
        ur_licha['Задолж. (ЮЛ, Н, К, ИВ, кредиты, ср за год, эквив)'] /
        ((1 - (1 + ur_licha['Рублевая доходность кредитов нерезидентов юрлицам в ИВ, %'] / 400) ** 
          (-ur_licha['Дюрация приведенная (ЮЛ, ИВ, К, Н), кварталов'])) * 
         ur_licha['gdp_nsa_byn_y'] / 4)
    )
    
    ur_licha['КОД (ЮЛ, Д, ИВ, Н)'] = (
        ur_licha['Рублевая доходность кредитов нерезидентов юрлицам в ИВ, %'] / 4 * 
        ur_licha['Задолж. (ЮЛ, Н, Д, ИВ, кредиты, ср за год, эквив)'] /
        ((1 - (1 + ur_licha['Рублевая доходность кредитов нерезидентов юрлицам в ИВ, %'] / 400) ** 
          (-ur_licha['Дюрация приведенная (ЮЛ, ИВ, Д, Н), кварталов'])) * 
         ur_licha['gdp_nsa_byn_y'] / 4)
    )
    
    # Общий коэффициент обслуживания долга
    ur_licha['dsr'] = (
        ur_licha['КОД (ЮЛ, Д, ИВ)'] + ur_licha['КОД (ЮЛ, Д, ИВ, Н)'] + 
        ur_licha['КОД (ЮЛ, К, ИВ)'] + ur_licha['КОД (ЮЛ, К, ИВ, Н)'] + 
        ur_licha['КОД (ЮЛ, К, НВ)'] + ur_licha['КОД (ЮЛ, Д, НВ)']
    )
    
    # Удаление временных колонок
    ur_licha = ur_licha.drop(columns=['ср_п', 'ср_з'], errors='ignore')
    
    return ur_licha


def prepare_corporate_threshold_analysis(ur_licha, now_data_date):
    """Подготавливает данные для анализа пороговых значений юридических лиц"""
    print("Подготовка данных для анализа пороговых значений ЮЛ...")
    
    ur_licha_rasch = ur_licha[['Date', 'dsr', 'cest', 'invr', 'ln_cred*100', 'ggr']].copy()
    ur_licha_rasch = ur_licha_rasch.loc[ur_licha_rasch['Date'] <= now_data_date]
    
    # Установка пороговых значений
    ur_licha_rasch['threshold_dsr'] = 0.45 * 0.4 * 100
    ur_licha_rasch['threshold_invr'] = ur_licha_rasch['invr'].mean()
    
    # Расчет лагов
    ur_licha_rasch['факт -1 ln_cred*100'] = ur_licha_rasch['ln_cred*100'] - ur_licha_rasch['ln_cred*100'].shift(4)
    ur_licha_rasch['факт -2 ln_cred*100'] = (
        ur_licha_rasch['ln_cred*100'] - ur_licha_rasch['ln_cred*100'].shift(8)
    ) / 2
    
    ur_licha_rasch['факт -3 ln_cred*100'] = (
        ur_licha_rasch['ln_cred*100'] - ur_licha_rasch['ln_cred*100'].shift(12)
    ) / 3
    
    return ur_licha_rasch


def forecast_corporate_indicators(ur_licha_rasch, params, analysis_mode='actual', target_date=None):
    """Прогнозирование показателей для юридических лиц"""
    print("Прогнозирование показателей для юридических лиц...")
    
    # ============================================================
    # ПРЕДВАРИТЕЛЬНЫЙ РЕЖИМ: 
    # - Критические (ggr, ln_cred*100, dsr): прогноз от prev_date
    # - Некритические (cest): прогноз как обычно
    # ============================================================
    if analysis_mode == 'preliminary' and target_date is not None:
        prev_date = target_date - pd.DateOffset(months=3)
        print(f"  Предварительный режим: прогноз КРИТИЧЕСКИХ показателей ЮЛ строится для {prev_date.strftime('%Y-%m-%d')}")
        print(f"  Критические показатели: ggr, ln_cred*100, dsr")
        print(f"  Некритические показатели: cest (прогноз как обычно)")
        
        # Сохраняем последнюю дату в оригинальных данных
        last_date_original = ur_licha_rasch['Date'].iloc[-1]
        
        # Обрезаем данные до предыдущего квартала для КРИТИЧЕСКИХ показателей
        ur_licha_rasch_critical = ur_licha_rasch[ur_licha_rasch['Date'] <= prev_date].copy()
        
        if ur_licha_rasch_critical.empty:
            print("  Предупреждение: нет данных для прогнозирования критических показателей ЮЛ")
            return ur_licha_rasch, ur_licha_rasch['Date'].iloc[-1] if not ur_licha_rasch.empty else prev_date
        
        last_date_critical = ur_licha_rasch_critical['Date'].iloc[-1]
        
        # Получение параметров с проверкой на существование
        ggr_param = params[params['Название показателя'] == "All Debt to GDP ratio (банки и нерезиденты), %"]
        cest_param = params[params['Название показателя'] == "Логарифм цен на коммерческую недвижимость, скорректированных на дефлятор, *100"]
        cred_param = params[params['Название показателя'] == "Логарифм кредита ЮЛ (банки+нерезиденты), скорректированного на дефлятор ВВП, *100"]
        
        # Проверка наличия параметров для критических показателей
        missing_critical_params = []
        if ggr_param.empty:
            missing_critical_params.append("All Debt to GDP ratio (банки и нерезиденты), %")
        if cred_param.empty:
            missing_critical_params.append("Логарифм кредита ЮЛ (банки+нерезиденты), скорректированного на дефлятор ВВП, *100")
        
        if missing_critical_params:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: Отсутствуют параметры для критических показателей:")
            for p in missing_critical_params:
                print(f"    - {p}")
            print("  Будут использованы расчетные значения вместо прогнозных.")
            
            # Прогноз для критических показателей с использованием последних значений
            new_rows_critical = []
            for i in range(20):
                new_date = last_date_critical + pd.DateOffset(months=3 * (i + 1))
                
                last_ggr = ur_licha_rasch_critical['ggr'].dropna().iloc[-1] if not ur_licha_rasch_critical['ggr'].dropna().empty else 0
                last_cred = ur_licha_rasch_critical['ln_cred*100'].dropna().iloc[-1] if not ur_licha_rasch_critical['ln_cred*100'].dropna().empty else 0
                
                new_row = {
                    'Date': new_date,
                    'ggr': last_ggr,
                    'ln_cred*100': last_cred
                }
                new_rows_critical.append(new_row)
            
            # Прогноз для НЕКРИТИЧЕСКИХ показателей (cest) - как обычно
            # Используем все данные для cest (не обрезаем)
            ur_licha_rasch_cest = ur_licha_rasch[['Date', 'cest']].copy()
            ur_licha_rasch_cest = ur_licha_rasch_cest.dropna()
            
            if not ur_licha_rasch_cest.empty:
                last_date_cest = ur_licha_rasch_cest['Date'].iloc[-1]
                last_cest = ur_licha_rasch_cest['cest'].iloc[-1]
                
                new_rows_cest = []
                for i in range(13):
                    new_date = last_date_cest + pd.DateOffset(months=3 * (i + 1))
                    if not cest_param.empty and i >= 2:
                        # Используем параметры для прогноза
                        if i == 0:
                            new_cest = cest_param['Начальные значения (HP)'].values[0]
                        elif i == 1:
                            new_cest = cest_param['Unnamed: 2'].values[0]
                        else:
                            new_cest = new_rows_cest[-1]['cest'] + cest_param['Шаги'].values[0]
                    else:
                        # Используем последнее значение
                        new_cest = last_cest + (i + 1) * 0.1  # небольшой шаг
                    
                    new_rows_cest.append({'Date': new_date, 'cest': new_cest})
                
                # Объединяем прогнозы критических и некритических показателей
                # Создаем DataFrame с объединенными данными
                critical_df = pd.DataFrame(new_rows_critical)
                cest_df = pd.DataFrame(new_rows_cest)
                
                # Объединяем по дате
                merged_df = pd.merge(critical_df, cest_df, on='Date', how='outer')
                
                # Добавляем к оригинальному DataFrame
                if not merged_df.empty:
                    ur_licha_rasch = pd.concat([ur_licha_rasch, merged_df], ignore_index=True)
                    print(f"  Добавлено {len(merged_df)} прогнозных записей для ЮЛ")
            else:
                # Если нет данных для cest, добавляем только критические
                if new_rows_critical:
                    ur_licha_rasch = pd.concat([ur_licha_rasch, pd.DataFrame(new_rows_critical)], ignore_index=True)
                    print(f"  Добавлено {len(new_rows_critical)} прогнозных записей для критических показателей ЮЛ")
            
            return ur_licha_rasch, last_date_critical
        
        # ============================================================
        # ПРОГНОЗ ДЛЯ КРИТИЧЕСКИХ ПОКАЗАТЕЛЕЙ (ggr, ln_cred*100)
        # ============================================================
        new_rows_critical = []
        for i in range(20):
            if i == 0:
                new_ggr = ggr_param['Начальные значения (HP)'].values[0]
                new_cred = cred_param['Начальные значения (HP)'].values[0]
            elif i == 1:
                new_ggr = ggr_param['Unnamed: 2'].values[0]
                new_cred = cred_param['Unnamed: 2'].values[0]
            else:
                new_ggr = ur_licha_rasch_critical['ggr'].iloc[-2] + ggr_param['Шаги'].values[0]
                
                if i < 16:
                    new_cred = ur_licha_rasch_critical['ln_cred*100'].iloc[-2] + cred_param['Шаги'].values[0]
                else:
                    new_cred = None
            
            new_date = last_date_critical + pd.DateOffset(months=3 * (i + 1))
            new_row = {
                'Date': new_date,
                'ggr': new_ggr,
                'ln_cred*100': new_cred
            }
            clean_row = {k: v for k, v in new_row.items() if v is not None}
            new_rows_critical.append(clean_row)
        
        # ============================================================
        # ПРОГНОЗ ДЛЯ НЕКРИТИЧЕСКИХ ПОКАЗАТЕЛЕЙ (cest) - как обычно
        # ============================================================
        # Используем все данные для cest (не обрезаем)
        ur_licha_rasch_cest = ur_licha_rasch[['Date', 'cest']].copy()
        ur_licha_rasch_cest = ur_licha_rasch_cest.dropna()
        new_rows_cest = []
        if not ur_licha_rasch_cest.empty:
            last_date_cest = ur_licha_rasch_cest['Date'].iloc[-1]
            last_cest = ur_licha_rasch_cest['cest'].iloc[-1]
            
            for i in range(13):
                new_date = last_date_cest + pd.DateOffset(months=3 * (i + 1))
                if not cest_param.empty and len(cest_param) > 0:
                    if i == 0:
                        new_cest = cest_param['Начальные значения (HP)'].values[0]
                    elif i == 1:
                        new_cest = cest_param['Unnamed: 2'].values[0]
                    else:
                        # Используем шаг из параметров
                        step = cest_param['Шаги'].values[0] if not cest_param.empty else 0.1
                        # Берем предыдущее прогнозное значение
                        prev_cest = new_rows_cest[-1]['cest'] if new_rows_cest else last_cest
                        new_cest = prev_cest + step
                else:
                    # Если нет параметров, используем последнее значение с небольшим шагом
                    new_cest = last_cest + (i + 1) * 0.1
                
                new_rows_cest.append({'Date': new_date, 'cest': new_cest})
        
        # ============================================================
        # ОБЪЕДИНЕНИЕ ПРОГНОЗОВ
        # ============================================================
        critical_df = pd.DataFrame(new_rows_critical)
        cest_df = pd.DataFrame(new_rows_cest) if new_rows_cest else pd.DataFrame()

        # Обновляем ur_licha_rasch данными из первой строки critical_df для target_date
        if not critical_df.empty:
            first_critical_row = critical_df.iloc[0]
            mask = ur_licha_rasch['Date'] == target_date
            if mask.any():
                for col in ['ggr', 'ln_cred*100']:
                    if col in first_critical_row and pd.notna(first_critical_row[col]):
                        ur_licha_rasch.loc[mask, col] = first_critical_row[col]
                print(f"  Обновлены критические показатели для даты {target_date.strftime('%Y-%m-%d')}")
            else:
                # Если даты нет, добавляем новую строку
                new_row = {'Date': target_date}
                for col in ['ggr', 'ln_cred*100']:
                    if col in first_critical_row:
                        new_row[col] = first_critical_row[col]
                ur_licha_rasch = pd.concat([ur_licha_rasch, pd.DataFrame([new_row])], ignore_index=True)
                print(f"  Добавлена новая строка для даты {target_date.strftime('%Y-%m-%d')}")

        # Объединяем по дате
        if not critical_df.empty and not cest_df.empty:
            merged_df = pd.merge(critical_df, cest_df, on='Date', how='outer')
        elif not critical_df.empty:
            merged_df = critical_df
        elif not cest_df.empty:
            merged_df = cest_df
        else:
            merged_df = pd.DataFrame()

        # Добавляем прогнозные записи (пропускаем первую, т.к. она уже добавлена/обновлена)
        if not merged_df.empty and len(merged_df) > 1:
            ur_licha_rasch = pd.concat([ur_licha_rasch, merged_df.iloc[1:]], ignore_index=True)
            print(f"  Добавлено {len(merged_df) - 1} дополнительных прогнозных записей для ЮЛ")
        elif not merged_df.empty:
            print(f"  Прогнозные записи уже добавлены/обновлены")

        return ur_licha_rasch, last_date_critical
    
    # ============================================================
    # ФАКТИЧЕСКИЙ РЕЖИМ: обычный расчет (без изменений)
    # ============================================================
    else:
        last_date_ur = ur_licha_rasch['Date'].iloc[-1]
        
        # Получение параметров с проверкой на существование
        ggr_param = params[params['Название показателя'] == "All Debt to GDP ratio (банки и нерезиденты), %"]
        cest_param = params[params['Название показателя'] == "Логарифм цен на коммерческую недвижимость, скорректированных на дефлятор, *100"]
        cred_param = params[params['Название показателя'] == "Логарифм кредита ЮЛ (банки+нерезиденты), скорректированного на дефлятор ВВП, *100"]
        
        # Проверка наличия параметров
        missing_params = []
        if ggr_param.empty:
            missing_params.append("All Debt to GDP ratio (банки и нерезиденты), %")
        if cest_param.empty:
            missing_params.append("Логарифм цен на коммерческую недвижимость, скорректированных на дефлятор, *100")
        if cred_param.empty:
            missing_params.append("Логарифм кредита ЮЛ (банки+нерезиденты), скорректированного на дефлятор ВВП, *100")
        
        if missing_params:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Отсутствуют параметры в файле 'Исходные данные.xlsx' на листе 'param':")
            for p in missing_params:
                print(f"  - {p}")
            print("Будут использованы расчетные значения вместо прогнозных.")
            
            # Если параметры отсутствуют, используем расчетные значения
            new_rows = []
            for i in range(20):
                new_date = ur_licha_rasch['Date'].iloc[-1] + pd.DateOffset(months=3)
                
                # Используем последние известные значения
                last_ggr = ur_licha_rasch['ggr'].dropna().iloc[-1] if not ur_licha_rasch['ggr'].dropna().empty else 0
                last_cest = ur_licha_rasch['cest'].dropna().iloc[-1] if not ur_licha_rasch['cest'].dropna().empty else 0
                last_cred = ur_licha_rasch['ln_cred*100'].dropna().iloc[-1] if not ur_licha_rasch['ln_cred*100'].dropna().empty else 0
                
                new_row = {
                    'Date': new_date,
                    'ggr': last_ggr,
                    'ln_cred*100': last_cred,
                    'cest': last_cest
                }
                new_rows.append(new_row)
                
                ur_licha_rasch = pd.concat([ur_licha_rasch, pd.DataFrame(new_rows)], ignore_index=True)
                new_rows = []
            
            return ur_licha_rasch, last_date_ur
        
        # Прогнозирование с параметрами
        new_rows = []
        for i in range(20):
            if i == 0:
                new_ggr = ggr_param['Начальные значения (HP)'].values[0]
                new_cest = cest_param['Начальные значения (HP)'].values[0]
                new_cred = cred_param['Начальные значения (HP)'].values[0]
            elif i == 1:
                new_ggr = ggr_param['Unnamed: 2'].values[0]
                new_cest = cest_param['Unnamed: 2'].values[0]
                new_cred = cred_param['Unnamed: 2'].values[0]
            else:
                new_ggr = ur_licha_rasch['ggr'].iloc[-2] + ggr_param['Шаги'].values[0]
                
                if i < 16:
                    new_cred = ur_licha_rasch['ln_cred*100'].iloc[-2] + cred_param['Шаги'].values[0]
                    if i < 13:
                        new_cest = ur_licha_rasch['cest'].iloc[-2] + cest_param['Шаги'].values[0]
                    else:
                        new_cest = None
                else:
                    new_cred = None
            
            new_date = ur_licha_rasch['Date'].iloc[-1] + pd.DateOffset(months=3)
            new_row = {
                'Date': new_date,
                'ggr': new_ggr,
                'ln_cred*100': new_cred,
                'cest': new_cest
            }
            new_rows.append(new_row)
            
            ur_licha_rasch = pd.concat([ur_licha_rasch, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows = []
        
        return ur_licha_rasch, last_date_ur

def calculate_hp_filters_corporate(ur_licha_rasch, last_date_ur, analysis_mode='actual', target_date=None):
    """Применяет HP-фильтр для выделения трендов"""
    print("Применение HP-фильтра...")
    
    # ============================================================
    # ПРЕДВАРИТЕЛЬНЫЙ РЕЖИМ: разделение для HP-фильтра
    # ============================================================
    if analysis_mode == 'preliminary' and target_date is not None:
        prev_date = target_date - pd.DateOffset(months=3)
        print(f"  Предварительный режим: HP-фильтр для критических показателей до {prev_date.strftime('%Y-%m-%d')}")
        print(f"  Критические: ggr, ln_cred*100")
        print(f"  Некритические: cest (полные данные)")
        
        # Создаем пустые столбцы по умолчанию
        ur_licha_rasch['threshold_cest'] = np.nan
        ur_licha_rasch['threshold_ggr'] = np.nan
        ur_licha_rasch['HP_ln_cred'] = np.nan
        
        # ============================================================
        # 1. НЕКРИТИЧЕСКИЙ ПОКАЗАТЕЛЬ (cest) - используем все данные
        # ============================================================
        cest_data = ur_licha_rasch['cest'].dropna()
        if len(cest_data) > 4:
            try:
                cycle1, trend1 = hpfilter(cest_data, 1600)
                ur_licha_rasch['threshold_cest'] = pd.Series(trend1, index=cest_data.index)
                # Ограничиваем только известными данными
                ur_licha_rasch['threshold_cest'] = ur_licha_rasch['threshold_cest'].where(
                    ur_licha_rasch['Date'] <= target_date, np.nan
                )
                print(f"  HP-фильтр для cest: использованы все данные ({len(cest_data)} записей)")
            except Exception as e:
                print(f"  Предупреждение: HP-фильтр для cest не удался: {e}")
        
        # ============================================================
        # 2. КРИТИЧЕСКИЕ ПОКАЗАТЕЛИ (ggr, ln_cred*100) - только до prev_date
        # ============================================================
        # Обрезаем данные до prev_date для критических показателей
        ur_licha_rasch_critical = ur_licha_rasch[ur_licha_rasch['Date'] <= prev_date].copy()
        
        # ggr
        ggr_data = ur_licha_rasch_critical['ggr'].dropna()
        if len(ggr_data) > 4:
            try:
                cycle2, trend2 = hpfilter(ggr_data, 1600)
                # Создаем Series с индексом из оригинального DataFrame
                ur_licha_rasch['threshold_ggr'] = pd.Series(index=ur_licha_rasch.index, dtype=float)
                # Заполняем значения для дат, где есть данные
                for idx in ggr_data.index:
                    if idx in ur_licha_rasch.index:
                        ur_licha_rasch.loc[idx, 'threshold_ggr'] = trend2.loc[idx] if idx in trend2.index else np.nan
                # Ограничиваем только известными данными
                ur_licha_rasch['threshold_ggr'] = ur_licha_rasch['threshold_ggr'].where(
                    ur_licha_rasch['Date'] <= prev_date, np.nan
                )
                print(f"  HP-фильтр для ggr: использованы данные до {prev_date.strftime('%Y-%m-%d')} ({len(ggr_data)} записей)")
            except Exception as e:
                print(f"  Предупреждение: HP-фильтр для ggr не удался: {e}")
        
        # ln_cred*100
        cred_data = ur_licha_rasch_critical['ln_cred*100'].dropna()
        if len(cred_data) > 4:
            try:
                cycle3, trend3 = hpfilter(cred_data, 1600)
                # Создаем Series с индексом из оригинального DataFrame
                ur_licha_rasch['HP_ln_cred'] = pd.Series(index=ur_licha_rasch.index, dtype=float)
                # Заполняем значения для дат, где есть данные
                for idx in cred_data.index:
                    if idx in ur_licha_rasch.index:
                        ur_licha_rasch.loc[idx, 'HP_ln_cred'] = trend3.loc[idx] if idx in trend3.index else np.nan
                # Ограничиваем только известными данными
                ur_licha_rasch['HP_ln_cred'] = ur_licha_rasch['HP_ln_cred'].where(
                    ur_licha_rasch['Date'] <= prev_date, np.nan
                )
                print(f"  HP-фильтр для ln_cred: использованы данные до {prev_date.strftime('%Y-%m-%d')} ({len(cred_data)} записей)")
            except Exception as e:
                print(f"  Предупреждение: HP-фильтр для ln_cred не удался: {e}")
        return ur_licha_rasch
    
    # ============================================================
    # ФАКТИЧЕСКИЙ РЕЖИМ: обычный расчет
    # ============================================================
    else:
        # Проверка наличия данных для HP-фильтра
        cest_data = ur_licha_rasch['cest'].dropna()
        ggr_data = ur_licha_rasch['ggr'].dropna()
        cred_data = ur_licha_rasch['ln_cred*100'].dropna()
        
        # Создаем пустые столбцы по умолчанию
        ur_licha_rasch['threshold_cest'] = np.nan
        ur_licha_rasch['threshold_ggr'] = np.nan
        ur_licha_rasch['HP_ln_cred'] = np.nan
        
        # HP-фильтр для cest (если есть данные)
        if len(cest_data) > 4:
            try:
                cycle1, trend1 = hpfilter(cest_data, 1600)
                ur_licha_rasch['threshold_cest'] = pd.Series(trend1, index=cest_data.index)
                ur_licha_rasch['threshold_cest'] = ur_licha_rasch['threshold_cest'].where(
                    ur_licha_rasch['Date'] <= last_date_ur, np.nan
                )
            except Exception as e:
                print(f"Предупреждение: HP-фильтр для cest не удался: {e}")
        
        # HP-фильтр для ggr (если есть данные)
        if len(ggr_data) > 4:
            try:
                cycle2, trend2 = hpfilter(ggr_data, 1600)
                ur_licha_rasch['threshold_ggr'] = pd.Series(trend2, index=ggr_data.index)
                ur_licha_rasch['threshold_ggr'] = ur_licha_rasch['threshold_ggr'].where(
                    ur_licha_rasch['Date'] <= last_date_ur, np.nan
                )
            except Exception as e:
                print(f"Предупреждение: HP-фильтр для ggr не удался: {e}")
        
        # HP-фильтр для кредитов (если есть данные)
        if len(cred_data) > 4:
            try:
                cycle3, trend3 = hpfilter(cred_data, 1600)
                ur_licha_rasch['HP_ln_cred'] = pd.Series(trend3, index=cred_data.index)
                ur_licha_rasch['HP_ln_cred'] = ur_licha_rasch['HP_ln_cred'].where(
                    ur_licha_rasch['Date'] <= last_date_ur, np.nan
                )
            except Exception as e:
                print(f"Предупреждение: HP-фильтр для ln_cred не удался: {e}")
        
        return ur_licha_rasch


def calculate_standard_deviations_corporate(ur_licha_rasch, last_date_ur):
    """Рассчитывает стандартные отклонения для различных показателей"""
    
    # Функция для безопасного расчета
    def safe_std(data, default=0.1):
        if len(data) > 1:
            return data.std(ddof=1)
        return default
    
    def safe_last(data, default=0.0):
        if len(data) > 0:
            return data.iloc[-1]
        return default
    
    def safe_first(data, default=0.0):
        if len(data) > 0:
            return data.iloc[0]
        return default
    
    # Стандартное отклонение для ggr
    ggr_data = ur_licha_rasch.loc[
        (ur_licha_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
        (ur_licha_rasch['Date'] <= last_date_ur), 'ggr'
    ].dropna()
    std_ggr = safe_std(ggr_data)
    
    # Последнее и первое значения для ggr
    threshold_ggr_data = ur_licha_rasch.loc[
        ur_licha_rasch['Date'] <= last_date_ur, 'threshold_ggr'
    ].dropna()
    last_ggr = safe_last(threshold_ggr_data, safe_last(ggr_data, 10.0))
    first_ggr = safe_first(ggr_data, 10.0)
    
    # Стандартное отклонение для cest
    cest_data = ur_licha_rasch.loc[
        (ur_licha_rasch['Date'] >= pd.Timestamp(2011, 1, 1)) & 
        (ur_licha_rasch['Date'] <= last_date_ur), 'cest'
    ].dropna()
    std_cest = safe_std(cest_data)
    
    # Последнее и первое значения для cest
    threshold_cest_data = ur_licha_rasch.loc[
        ur_licha_rasch['Date'] <= last_date_ur, 'threshold_cest'
    ].dropna()
    last_cest = safe_last(threshold_cest_data, safe_last(cest_data, 0.0))
    first_cest = safe_first(cest_data, 0.0)
    
    # Стандартное отклонение для кредитов
    cred_data = ur_licha_rasch.loc[
        (ur_licha_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
        (ur_licha_rasch['Date'] <= last_date_ur), 'ln_cred*100'
    ].dropna()
    std_cred = safe_std(cred_data)
    
    # Последнее и первое значения для кредитов
    hp_cred_data = ur_licha_rasch.loc[
        ur_licha_rasch['Date'] <= last_date_ur, 'HP_ln_cred'
    ].dropna()
    last_cred = safe_last(hp_cred_data, safe_last(cred_data, 0.0))
    first_cred = safe_first(cred_data, 0.0)
    
    return {
        'std_ggr': std_ggr, 'last_ggr': last_ggr, 'first_ggr': first_ggr,
        'std_cest': std_cest, 'last_cest': last_cest, 'first_cest': first_cest,
        'std_cred': std_cred, 'last_cred': last_cred, 'first_cred': first_cred
    }


def iterative_adjustment_ggr(ur_licha_rasch, params, last_date_ur, std_ggr, last_ggr, first_ggr):
    """Итерационная корректировка показателя ggr"""
    print("Итерационная корректировка ggr...")
    
    # Проверка наличия параметров
    ggr_param = params[params['Название показателя'] == "All Debt to GDP ratio (банки и нерезиденты), %"]
    if ggr_param.empty:
        print("Предупреждение: параметры для ggr не найдены, пропуск итерационной корректировки")
        return ur_licha_rasch
    
    try:
        step = ggr_param['Шаги'].values[0]
    except:
        step = 0.1
    
    max_iterations = 50
    iteration = 0
    
    while round(first_ggr, 10) != round(last_ggr - std_ggr, 10) and iteration < max_iterations:
        iteration += 1
        for i in range(20):
            if i == 0:
                new_ggr = last_ggr - std_ggr
                first_ggr = new_ggr
            elif i == 1:
                new_ggr = last_ggr + std_ggr
            else:
                last_ggr_val = ur_licha_rasch.loc[
                    ur_licha_rasch['Date'] == (last_date_ur + pd.DateOffset(months=3*(i-1))), 
                    'ggr'
                ].iloc[0] if len(ur_licha_rasch.loc[
                    ur_licha_rasch['Date'] == (last_date_ur + pd.DateOffset(months=3*(i-1))), 
                    'ggr'
                ]) > 0 else last_ggr
                new_ggr = last_ggr_val + step
            
            date_reset = last_date_ur + pd.DateOffset(months=3*(i+1))
            if date_reset in ur_licha_rasch['Date'].values:
                ur_licha_rasch.loc[ur_licha_rasch['Date'] == date_reset, 'ggr'] = new_ggr
        
        # Пересчет HP-фильтра
        ggr_data = ur_licha_rasch['ggr'].dropna()
        if len(ggr_data) > 4:
            try:
                cycle2, trend2 = hpfilter(ggr_data, 1600)
                ur_licha_rasch['threshold_ggr'] = pd.Series(trend2, index=ggr_data.index)
                ur_licha_rasch['threshold_ggr'] = ur_licha_rasch['threshold_ggr'].where(
                    ur_licha_rasch['Date'] <= last_date_ur, np.nan
                )
                
                # Пересчет стандартных отклонений
                std_ggr = ur_licha_rasch.loc[
                    (ur_licha_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
                    (ur_licha_rasch['Date'] <= last_date_ur), 'ggr'
                ].std(ddof=1)
                
                last_ggr = ur_licha_rasch.loc[
                    ur_licha_rasch['Date'] == last_date_ur, 'threshold_ggr'
                ].iloc[0] if len(ur_licha_rasch.loc[
                    ur_licha_rasch['Date'] == last_date_ur, 'threshold_ggr'
                ]) > 0 else last_ggr
            except Exception as e:
                print(f"Предупреждение: ошибка при пересчете HP-фильтра для ggr: {e}")
                break
        else:
            break
    
    return ur_licha_rasch


def iterative_adjustment_cred_corporate(ur_licha_rasch, params, last_date_ur, std_cred, last_cred, first_cred):
    """Итерационная корректировка показателя кредитов"""
    print("Итерационная корректировка кредитов...")
    
    # Проверка наличия параметров
    cred_param = params[params['Название показателя'] == "Логарифм кредита ЮЛ (банки+нерезиденты), скорректированного на дефлятор ВВП, *100"]
    if cred_param.empty:
        print("Предупреждение: параметры для кредитов не найдены, пропуск итерационной корректировки")
        return ur_licha_rasch
    
    try:
        step = cred_param['Шаги'].values[0]
    except:
        step = 0.1
    
    max_iterations = 50
    iteration = 0
    
    while round(first_cred, 10) != round(last_cred - std_cred, 10) and iteration < max_iterations:
        iteration += 1
        for i in range(16):
            if i == 0:
                new_cred = last_cred - std_cred
                first_cred = new_cred
            elif i == 1:
                new_cred = last_cred + std_cred
            else:
                last_cred_val = ur_licha_rasch.loc[
                    ur_licha_rasch['Date'] == (last_date_ur + pd.DateOffset(months=3*(i-1))), 
                    'ln_cred*100'
                ].iloc[0] if len(ur_licha_rasch.loc[
                    ur_licha_rasch['Date'] == (last_date_ur + pd.DateOffset(months=3*(i-1))), 
                    'ln_cred*100'
                ]) > 0 else last_cred
                new_cred = last_cred_val + step
            
            date_reset = last_date_ur + pd.DateOffset(months=3*(i+1))
            if date_reset in ur_licha_rasch['Date'].values:
                ur_licha_rasch.loc[ur_licha_rasch['Date'] == date_reset, 'ln_cred*100'] = new_cred
        
        # Пересчет HP-фильтра
        cred_data = ur_licha_rasch['ln_cred*100'].dropna()
        if len(cred_data) > 4:
            try:
                cycle3, trend3 = hpfilter(cred_data, 1600)
                ur_licha_rasch['HP_ln_cred'] = pd.Series(trend3, index=cred_data.index)
                ur_licha_rasch['HP_ln_cred'] = ur_licha_rasch['HP_ln_cred'].where(
                    ur_licha_rasch['Date'] <= last_date_ur, np.nan
                )
                
                # Пересчет стандартных отклонений
                std_cred = ur_licha_rasch.loc[
                    (ur_licha_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
                    (ur_licha_rasch['Date'] <= last_date_ur), 'ln_cred*100'
                ].std(ddof=1)
                
                last_cred = ur_licha_rasch.loc[
                    ur_licha_rasch['Date'] == last_date_ur, 'HP_ln_cred'
                ].iloc[0] if len(ur_licha_rasch.loc[
                    ur_licha_rasch['Date'] == last_date_ur, 'HP_ln_cred'
                ]) > 0 else last_cred
            except Exception as e:
                print(f"Предупреждение: ошибка при пересчете HP-фильтра для кредитов: {e}")
                break
        else:
            break
    
    return ur_licha_rasch


def iterative_adjustment_cest(ur_licha_rasch, params, last_date_ur, std_cest, last_cest, first_cest):
    """Итерационная корректировка показателя недвижимости"""
    print("Итерационная корректировка недвижимости...")
    
    # Проверка наличия параметров
    cest_param = params[params['Название показателя'] == "Логарифм цен на коммерческую недвижимость, скорректированных на дефлятор, *100"]
    if cest_param.empty:
        print("Предупреждение: параметры для недвижимости не найдены, пропуск итерационной корректировки")
        return ur_licha_rasch
    
    try:
        step = cest_param['Шаги'].values[0]
    except:
        step = 0.1
    
    max_iterations = 50
    iteration = 0
    
    while round(first_cest, 10) != round(last_cest - std_cest, 10) and iteration < max_iterations:
        iteration += 1
        for i in range(13):
            if i == 0:
                new_cest = last_cest - std_cest
                first_cest = new_cest
            elif i == 1:
                new_cest = last_cest + std_cest
            else:
                last_cest_val = ur_licha_rasch.loc[
                    ur_licha_rasch['Date'] == (last_date_ur + pd.DateOffset(months=3*(i-1))), 
                    'cest'
                ].iloc[0] if len(ur_licha_rasch.loc[
                    ur_licha_rasch['Date'] == (last_date_ur + pd.DateOffset(months=3*(i-1))), 
                    'cest'
                ]) > 0 else last_cest
                new_cest = last_cest_val + step
            
            date_reset = last_date_ur + pd.DateOffset(months=3*(i+1))
            if date_reset in ur_licha_rasch['Date'].values:
                ur_licha_rasch.loc[ur_licha_rasch['Date'] == date_reset, 'cest'] = new_cest
        
        # Пересчет HP-фильтра
        cest_data = ur_licha_rasch['cest'].dropna()
        if len(cest_data) > 4:
            try:
                cycle1, trend1 = hpfilter(cest_data, 1600)
                ur_licha_rasch['threshold_cest'] = pd.Series(trend1, index=cest_data.index)
                ur_licha_rasch['threshold_cest'] = ur_licha_rasch['threshold_cest'].where(
                    ur_licha_rasch['Date'] <= last_date_ur, np.nan
                )
                
                # Пересчет стандартных отклонений
                std_cest = ur_licha_rasch.loc[
                    (ur_licha_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
                    (ur_licha_rasch['Date'] <= last_date_ur), 'cest'
                ].std(ddof=1)
                
                last_cest = ur_licha_rasch.loc[
                    ur_licha_rasch['Date'] == last_date_ur, 'threshold_cest'
                ].iloc[0] if len(ur_licha_rasch.loc[
                    ur_licha_rasch['Date'] == last_date_ur, 'threshold_cest'
                ]) > 0 else last_cest
            except Exception as e:
                print(f"Предупреждение: ошибка при пересчете HP-фильтра для cest: {e}")
                break
        else:
            break
    
    return ur_licha_rasch


def calculate_lagged_thresholds_corporate(ur_licha_rasch):
    """Рассчитывает лагированные пороговые значения"""
    print("Расчет лагированных пороговых значений...")
    
    ur_licha_rasch['threshold -1 ln_cred*100'] = (
        ur_licha_rasch['HP_ln_cred'] - ur_licha_rasch['HP_ln_cred'].shift(4)
    )
    
    ur_licha_rasch['threshold -2 ln_cred*100'] = (
        ur_licha_rasch['HP_ln_cred'] - ur_licha_rasch['HP_ln_cred'].shift(8)
    ) / 2
    
    ur_licha_rasch['threshold -3 ln_cred*100'] = (
        ur_licha_rasch['HP_ln_cred'] - ur_licha_rasch['HP_ln_cred'].shift(12)
    ) / 3
    
    return ur_licha_rasch


def run_corporate_analysis(start_q, rates, params, now_data_date, analysis_mode='actual', target_date=None):
    """Основная функция для анализа юридических лиц"""
    print("=" * 50)
    print("АНАЛИЗ ЮРИДИЧЕСКИХ ЛИЦ")
    print("=" * 50)
    
    try:
        # 1. Подготовка данных
        ur_licha = prepare_corporate_data(start_q, rates, now_data_date)
        
        # Проверка на пустые данные
        if ur_licha is None or ur_licha.empty:
            print("Ошибка: нет данных для анализа юридических лиц")
            return pd.DataFrame(), pd.DataFrame()
        
        # 2. Расчет индикаторов недвижимости
        ur_licha = calculate_real_estate_indicators(ur_licha)
        
        # 3. Расчет кредитных индикаторов
        ur_licha = calculate_credit_indicators(ur_licha)
        
        # 4. Корректировка валютных значений
        ur_licha = adjust_currency_values(ur_licha)
        
        # 5. Расчет метрик дюрации
        ur_licha = calculate_duration_metrics_corporate(ur_licha)
        
        # 6. Корректировка исторических значений
        ur_licha = adjust_historical_durations(ur_licha)
        
        # 7. Расчет коэффициентов обслуживания долга
        ur_licha = calculate_debt_service_ratio_corporate(ur_licha)
        
        # 8. Расчет индикаторов нерезидентов
        ur_licha = calculate_non_resident_indicators(ur_licha)
        
        # 9. Подготовка данных для анализа пороговых значений
        ur_licha_rasch = prepare_corporate_threshold_analysis(ur_licha, now_data_date)
        
        # Проверка на пустые данные
        if ur_licha_rasch is None or ur_licha_rasch.empty:
            print("Ошибка: нет данных для анализа пороговых значений юридических лиц")
            return ur_licha, pd.DataFrame()
        
        # 10. Прогнозирование показателей
        ur_licha_rasch, last_date_ur = forecast_corporate_indicators(ur_licha_rasch, params, analysis_mode, target_date)
        
        # 11. Применение HP-фильтра
        ur_licha_rasch = calculate_hp_filters_corporate(ur_licha_rasch, last_date_ur, analysis_mode, target_date)
        # 12. Расчет стандартных отклонений
        stats = calculate_standard_deviations_corporate(ur_licha_rasch, last_date_ur)
        
        # 13. Итерационная корректировка показателей (только если есть данные)
        ggr_data = ur_licha_rasch['ggr'].dropna()
        threshold_ggr_data = ur_licha_rasch['threshold_ggr'].dropna()
        if not ggr_data.empty and not threshold_ggr_data.empty and len(ggr_data) > 1:
            try:
                ur_licha_rasch = iterative_adjustment_ggr(
                    ur_licha_rasch, params, last_date_ur, 
                    stats['std_ggr'], stats['last_ggr'], stats['first_ggr']
                )
            except Exception as e:
                print(f"Предупреждение: итерационная корректировка ggr не удалась: {e}")
        
        cred_data = ur_licha_rasch['ln_cred*100'].dropna()
        hp_cred_data = ur_licha_rasch['HP_ln_cred'].dropna()
        if not cred_data.empty and not hp_cred_data.empty and len(cred_data) > 1:
            try:
                ur_licha_rasch = iterative_adjustment_cred_corporate(
                    ur_licha_rasch, params, last_date_ur,
                    stats['std_cred'], stats['last_cred'], stats['first_cred']
                )
            except Exception as e:
                print(f"Предупреждение: итерационная корректировка кредитов не удалась: {e}")
        
        cest_data = ur_licha_rasch['cest'].dropna()
        threshold_cest_data = ur_licha_rasch['threshold_cest'].dropna()
        if not cest_data.empty and not threshold_cest_data.empty and len(cest_data) > 1:
            try:
                ur_licha_rasch = iterative_adjustment_cest(
                    ur_licha_rasch, params, target_date,
                    stats['std_cest'], stats['last_cest'], stats['first_cest']
                )
            except Exception as e:
                print(f"Предупреждение: итерационная корректировка cest не удалась: {e}")
        
        # 14. Расчет лагированных пороговых значений
        ur_licha_rasch = calculate_lagged_thresholds_corporate(ur_licha_rasch)

        print("\n" + "=" * 50)
        print("АНАЛИЗ ЮРИДИЧЕСКИХ ЛИЦ ЗАВЕРШЕН")
        print("=" * 50)
        
        print(f"\nРезультаты анализа:")
        print(f"- Основные данные: {ur_licha.shape[0] if ur_licha is not None else 0} записей")
        print(f"- Данные для анализа порогов: {ur_licha_rasch.shape[0] if ur_licha_rasch is not None else 0} записей")
        if ur_licha_rasch is not None and not ur_licha_rasch.empty:
            print(f"- Последняя дата: {ur_licha_rasch['Date'].iloc[-1].strftime('%Y-%m-%d')}")
        
        return ur_licha, ur_licha_rasch
        
    except Exception as e:
        print(f"Ошибка при выполнении анализа юридических лиц: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()


# ============================================================================
# 9. ГОСУДАРСТВЕННЫЙ СЕКТОР
# ============================================================================

def calculate_government_indicators(start_q, now_data_date, params):
    """Рассчитывает индикаторы государственного сектора"""
    print("Расчет индикаторов государственного сектора...")
    
    gos = start_q[['Date', 'ExtDebtSvc_Gov_Pr_mnUSD', 'ExtDebtSvc_Gov_Int_mnUSD', 'gdp_nsa_usd_y', 
                   'Rev_ConsBud_exFSZN_YtD_mnBYN', 'Rev_FSZN_YtD_mnBYN', 'Exp_ConsBud_exFSZN_YtD_mnBYN', 
                   'Exp_FSZN_YtD_mnBYN', 'gdp_nsa_byn_y', 'TotDebt_CGov_GInc_mnBYN',
                   'TotDebt_CGov_GInc_mnUSD', 'ExtGovDebt_mnBYN', 'ExtGovDebt_mnUSD', 
                   'GDP_EQ_q_sa', 'GDP_q_sa']].copy()
    
    # Расчет обслуживания внешнего долга
    gos['ExtDebtSvc_Gov_all_mnUSD'] = gos['ExtDebtSvc_Gov_Pr_mnUSD'] + gos['ExtDebtSvc_Gov_Int_mnUSD']
    gos['ExtDebtSvc_Gov_avg_mnUSD'] = np.where(
        gos['Date'].dt.month == 1, 
        gos['ExtDebtSvc_Gov_all_mnUSD'], 
        gos['ExtDebtSvc_Gov_all_mnUSD'] - gos['ExtDebtSvc_Gov_all_mnUSD'].shift(1)
    )
    gos['ExtDebtSvc_Gov_allavg_mnUSD'] = gos['ExtDebtSvc_Gov_avg_mnUSD'].rolling(window=4).sum()
    gos['ser_exdebt_gdp'] = gos['ExtDebtSvc_Gov_allavg_mnUSD'] / gos['gdp_nsa_usd_y'] * 100
    
    # Расчет доходов и расходов консолидированного бюджета
    gos['Rev_ConsBud_exFSZN_mnBYN'] = np.where(
        gos['Date'] <= pd.Timestamp(2008, 10, 1),
        gos['Rev_ConsBud_exFSZN_YtD_mnBYN'],
        np.where(gos['Date'].dt.month == 1, 
                gos['Rev_ConsBud_exFSZN_YtD_mnBYN'], 
                gos['Rev_ConsBud_exFSZN_YtD_mnBYN'] - gos['Rev_ConsBud_exFSZN_YtD_mnBYN'].shift(1))
    )
    
    gos['Rev_FSZN_mnBYN'] = np.where(
        gos['Date'] <= pd.Timestamp(2008, 10, 1),
        gos['Rev_FSZN_YtD_mnBYN'],
        np.where(gos['Date'].dt.month == 1, 
                gos['Rev_FSZN_YtD_mnBYN'], 
                gos['Rev_FSZN_YtD_mnBYN'] - gos['Rev_FSZN_YtD_mnBYN'].shift(1))
    )
    
    gos['Exp_ConsBud_exFSZN_mnBYN'] = np.where(
        gos['Date'] <= pd.Timestamp(2008, 10, 1),
        gos['Exp_ConsBud_exFSZN_YtD_mnBYN'],
        np.where(gos['Date'].dt.month == 1, 
                gos['Exp_ConsBud_exFSZN_YtD_mnBYN'], 
                gos['Exp_ConsBud_exFSZN_YtD_mnBYN'] - gos['Exp_ConsBud_exFSZN_YtD_mnBYN'].shift(1))
    )
    
    gos['Exp_FSZN_mnBYN'] = np.where(
        gos['Date'] <= pd.Timestamp(2008, 10, 1),
        gos['Exp_FSZN_YtD_mnBYN'],
        np.where(gos['Date'].dt.month == 1, 
                gos['Exp_FSZN_YtD_mnBYN'], 
                gos['Exp_FSZN_YtD_mnBYN'] - gos['Exp_FSZN_YtD_mnBYN'].shift(1))
    )
    
    gos['Rev_FSZN_mnBYN'] = gos['Rev_FSZN_mnBYN'].fillna(0)
    gos['Exp_FSZN_mnBYN'] = gos['Exp_FSZN_mnBYN'].fillna(0)
    
    gos['Rev_ConsBud_incFSZN_mnBYN'] = gos['Rev_ConsBud_exFSZN_mnBYN'] + gos['Rev_FSZN_mnBYN']
    gos['Exp_ConsBud_incFSZN_mnBYN'] = gos['Exp_ConsBud_exFSZN_mnBYN'] + gos['Exp_FSZN_mnBYN']
    
    # Ручная корректировка для 2002-2003
    gos.loc[(gos['Date'] >= pd.Timestamp(2002, 1, 1)) & (gos['Date'] <= pd.Timestamp(2003, 10, 1)), 
            'Rev_ConsBud_incFSZN_mnBYN'] = [158.2, 204.4, 217.7, 285.6, 239.8, 299.8, 323.2, 357.2]
    gos.loc[(gos['Date'] >= pd.Timestamp(2002, 1, 1)) & (gos['Date'] <= pd.Timestamp(2003, 10, 1)), 
            'Exp_ConsBud_incFSZN_mnBYN'] = [161.1, 215.5, 215.9, 283.3, 234.8, 307.7, 324.5, 409.2]
    
    # Расчет баланса бюджета и долговых показателей
    gos['Bal_ConsBud_mnBYN'] = gos['Rev_ConsBud_incFSZN_mnBYN'] - gos['Exp_ConsBud_incFSZN_mnBYN']
    gos['bb_gdp'] = gos['Bal_ConsBud_mnBYN'].rolling(window=4).sum() / gos['gdp_nsa_byn_y'] * 100
    
    gos['TotDebt_CGov_GInc_mnBYN % ВВП'] = gos['TotDebt_CGov_GInc_mnBYN'] / gos['gdp_nsa_byn_y'] * 100
    gos['TotDebt_CGov_GInc_mnUSD % ВВП'] = gos['TotDebt_CGov_GInc_mnUSD'] / gos['gdp_nsa_usd_y'] * 100
    gos['ggd'] = 0.35 * gos['TotDebt_CGov_GInc_mnBYN % ВВП'] + 0.65 * gos['TotDebt_CGov_GInc_mnUSD % ВВП']
    
    gos['Доля внешнего долга (млн. BYN), %'] = gos['ExtGovDebt_mnBYN'] / gos['TotDebt_CGov_GInc_mnBYN']
    gos['Доля внешнего долга (млн. USD), %'] = gos['ExtGovDebt_mnUSD'] / gos['TotDebt_CGov_GInc_mnUSD']
    
    gos['Rev_ConsBud_incFSZN_mnBYN_y % ВВП'] = (
        gos['Rev_ConsBud_incFSZN_mnBYN'].rolling(window=4).sum()
    ) / gos['gdp_nsa_byn_y'] * 100
    
    gos['Sh_ExtD_TotD'] = (
        gos['ExtGovDebt_mnBYN'] / gos['TotDebt_CGov_GInc_mnBYN'] + 
        gos['ExtGovDebt_mnUSD'] / gos['TotDebt_CGov_GInc_mnUSD']
    ) / 2
    
    # Расчет пороговых значений доходов бюджета
    gos['db'] = np.nan
    gos.loc[(gos['Date'] >= pd.Timestamp(2004, 1, 1)) & (gos['Date'] <= pd.Timestamp(2005, 1, 1)), 'db'] = (
        gos.loc[(gos['Date'] >= pd.Timestamp(2002, 10, 1)) & (gos['Date'] <= pd.Timestamp(2003, 10, 1)), 
                'Rev_ConsBud_incFSZN_mnBYN_y % ВВП'].mean()
    )
    
    gos.loc[(gos['Date'] >= pd.Timestamp(2005, 7, 1)) & (gos['Date'] <= pd.Timestamp(2011, 4, 1)), 'db'] = (
        gos.loc[(gos['Date'] >= pd.Timestamp(2005, 1, 1)) & (gos['Date'] <= pd.Timestamp(2010, 7, 1)), 
                'Rev_ConsBud_incFSZN_mnBYN_y % ВВП'].mean()
    )
    
    gos.loc[gos['Date'] == pd.Timestamp(2005, 4, 1), 'db'] = (
        gos.loc[gos['Date'] == pd.Timestamp(2005, 1, 1), 'db'].iloc[0] + 
        gos.loc[gos['Date'] == pd.Timestamp(2005, 7, 1), 'db'].iloc[0]
    ) / 2
    
    gos.loc[(gos['Date'] >= pd.Timestamp(2011, 7, 1)) & (gos['Date'] <= pd.Timestamp(now_data_date)), 'db'] = (
        gos.loc[(gos['Date'] >= pd.Timestamp(2010, 7, 1)) & (gos['Date'] <= pd.Timestamp(2024, 7, 1)), 
                'Rev_ConsBud_incFSZN_mnBYN_y % ВВП'].mean()
    )
    
    return gos


def prepare_government_threshold_analysis(gos, now_data_date):
    """Подготавливает данные для анализа пороговых значений госсектора"""
    
    gos_rasch = gos[['Date', 'ggd', 'ser_exdebt_gdp', 'db', 'Sh_ExtD_TotD', 'GDP_EQ_q_sa', 'bb_gdp']].copy()
    gos_rasch['Th'] = 14
    gos_rasch['y'] = (gos_rasch['GDP_EQ_q_sa'] / gos_rasch['GDP_EQ_q_sa'].shift(1))**4 - 1
    gos_rasch = gos_rasch.loc[gos_rasch['Date'] <= pd.Timestamp(now_data_date)]
    
    return gos_rasch


def forecast_government_indicators(gos_rasch, params):
    """Прогнозирование показателей государственного сектора"""
    
    last_date_gos = gos_rasch['Date'].iloc[-1]
    
    ggd_params = params.loc[
        params['Название показателя'] == "Долг сектора государственного управления к ВВП, % (внешний + внутренний)"
    ]
    
    rev_params = params.loc[
        params['Название показателя'] == "Доля внешнего госдолга в валовом госдолге"
    ]
    
    new_rows = []
    for i in range(17):
        if i == 0:
            new_ggd = ggd_params['Начальные значения (HP)'].values[0]
            new_rev = rev_params['Начальные значения (HP)'].values[0]
        elif i == 1:
            new_ggd = ggd_params['Unnamed: 2'].values[0]
            new_rev = rev_params['Unnamed: 2'].values[0]
        else:
            new_ggd = gos_rasch['ggd'].iloc[-2] + ggd_params['Шаги'].values[0]
            new_rev = gos_rasch['Sh_ExtD_TotD'].iloc[-2] + rev_params['Шаги'].values[0]
        
        new_date = gos_rasch['Date'].iloc[-1] + pd.DateOffset(months=3)
        new_row = {
            'Date': new_date,
            'ggd': new_ggd,
            'Sh_ExtD_TotD': new_rev
        }
        new_rows.append(new_row)
        
        gos_rasch = pd.concat([gos_rasch, pd.DataFrame(new_rows)], ignore_index=True)
        new_rows = []
    
    return gos_rasch, last_date_gos


def calculate_hp_filters_government(gos_rasch, last_date_gos):
    """Применяет HP-фильтр для показателей госсектора"""
    
    cycle1, trend1 = hpfilter(gos_rasch['ggd'].dropna(), 1600)
    cycle2, trend2 = hpfilter(gos_rasch['ser_exdebt_gdp'].dropna(), 1600)
    
    gos_rasch['threshold_ggd'] = pd.Series(trend1, index=gos_rasch.index)
    gos_rasch['hp_sh'] = pd.Series(trend2, index=gos_rasch.index)
    
    gos_rasch['threshold_ggd'] = gos_rasch['threshold_ggd'].where(
        gos_rasch['Date'] <= last_date_gos, np.nan
    )
    gos_rasch['hp_sh'] = gos_rasch['hp_sh'].where(
        gos_rasch['Date'] <= last_date_gos, np.nan
    )
    
    return gos_rasch


def calculate_government_standard_deviations(gos_rasch, last_date_gos):
    """Рассчитывает стандартные отклонения для показателей госсектора"""
    
    std_ggd = gos_rasch.loc[
        (gos_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
        (gos_rasch['Date'] <= last_date_gos), 'ggd'
    ].std(ddof=1)
    
    last_ggd = gos_rasch.loc[
        gos_rasch['Date'] == last_date_gos, 'threshold_ggd'
    ].iloc[0]
    
    first_ggd = gos_rasch.loc[
        gos_rasch['Date'] == (last_date_gos + pd.DateOffset(months=3)), 'ggd'
    ].iloc[0]
    
    std_rev = gos_rasch.loc[
        (gos_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
        (gos_rasch['Date'] <= last_date_gos), 'Sh_ExtD_TotD'
    ].std(ddof=1)
    
    last_rev = gos_rasch.loc[
        gos_rasch['Date'] == last_date_gos, 'hp_sh'
    ].iloc[0]
    
    first_rev = gos_rasch.loc[
        gos_rasch['Date'] == (last_date_gos + pd.DateOffset(months=3)), 'Sh_ExtD_TotD'
    ].iloc[0]
    
    return {
        'std_ggd': std_ggd, 'last_ggd': last_ggd, 'first_ggd': first_ggd,
        'std_rev': std_rev, 'last_rev': last_rev, 'first_rev': first_rev
    }


def iterative_adjustment_ggd(gos_rasch, params, last_date_gos, std_ggd, last_ggd, first_ggd):
    """Итерационная корректировка показателя государственного долга"""
    print("Итерационная корректировка государственного долга...")
    
    ggd_params = params.loc[
        params['Название показателя'] == "Долг сектора государственного управления к ВВП, % (внешний + внутренний)"
    ]
    
    while round(first_ggd, 10) != round(last_ggd - std_ggd, 10):
        for i in range(17):
            if i == 0:
                new_ggd = last_ggd - std_ggd
                first_ggd = new_ggd
            elif i == 1:
                new_ggd = last_ggd + std_ggd
            else:
                last_ggd_val = gos_rasch.loc[
                    gos_rasch['Date'] == (last_date_gos + pd.DateOffset(months=3*(i-1))), 
                    'ggd'
                ].iloc[0]
                new_ggd = last_ggd_val + ggd_params['Шаги'].values[0]
            
            date_reset = last_date_gos + pd.DateOffset(months=3*(i+1))
            gos_rasch.loc[gos_rasch['Date'] == date_reset, 'ggd'] = new_ggd
        
        cycle1, trend1 = hpfilter(gos_rasch['ggd'].dropna(), 1600)
        gos_rasch['threshold_ggd'] = pd.Series(trend1, index=gos_rasch.index)
        gos_rasch['threshold_ggd'] = gos_rasch['threshold_ggd'].where(
            gos_rasch['Date'] <= last_date_gos, np.nan
        )
        
        std_ggd = gos_rasch.loc[
            (gos_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
            (gos_rasch['Date'] <= last_date_gos), 'ggd'
        ].std(ddof=1)
        
        last_ggd = gos_rasch.loc[
            gos_rasch['Date'] == last_date_gos, 'threshold_ggd'
        ].iloc[0]
        
        first_ggd = gos_rasch.loc[
            gos_rasch['Date'] == (last_date_gos + pd.DateOffset(months=3)), 'ggd'
        ].iloc[0]
    
    return gos_rasch


def iterative_adjustment_rev(gos_rasch, params, last_date_gos, std_rev, last_rev, first_rev):
    """Итерационная корректировка доли внешнего долга"""
    print("Итерационная корректировка доли внешнего долга...")
    
    rev_params = params.loc[
        params['Название показателя'] == "Доля внешнего госдолга в валовом госдолге"
    ]
    
    while round(first_rev, 10) != round(last_rev - std_rev, 10):
        for i in range(17):
            if i == 0:
                new_rev = last_rev - std_rev
                first_rev = new_rev
            elif i == 1:
                new_rev = last_rev + std_rev
            else:
                last_rev_val = gos_rasch.loc[
                    gos_rasch['Date'] == (last_date_gos + pd.DateOffset(months=3*(i-1))), 
                    'Sh_ExtD_TotD'
                ].iloc[0]
                new_rev = last_rev_val + rev_params['Шаги'].values[0]
            
            date_reset = last_date_gos + pd.DateOffset(months=3*(i+1))
            gos_rasch.loc[gos_rasch['Date'] == date_reset, 'Sh_ExtD_TotD'] = new_rev
        
        cycle2, trend2 = hpfilter(gos_rasch['Sh_ExtD_TotD'].dropna(), 1600)
        gos_rasch['hp_sh'] = pd.Series(trend2, index=gos_rasch.index)
        gos_rasch['hp_sh'] = gos_rasch['hp_sh'].where(
            gos_rasch['Date'] <= last_date_gos, np.nan
        )
        
        std_rev = gos_rasch.loc[
            (gos_rasch['Date'] >= pd.Timestamp(2009, 1, 1)) & 
            (gos_rasch['Date'] <= last_date_gos), 'Sh_ExtD_TotD'
        ].std(ddof=1)
        
        last_rev = gos_rasch.loc[
            gos_rasch['Date'] == last_date_gos, 'hp_sh'
        ].iloc[0]
        
        first_rev = gos_rasch.loc[
            gos_rasch['Date'] == (last_date_gos + pd.DateOffset(months=3)), 'Sh_ExtD_TotD'
        ].iloc[0]
    
    return gos_rasch


def calculate_government_additional_metrics(gos_rasch):
    """Рассчитывает дополнительные метрики для госсектора"""
    
    gos_rasch['d'] = gos_rasch['threshold_ggd'] / 100
    gos_rasch['threshold_bb_gdp'] = -100 * gos_rasch['d'] * gos_rasch['y'] / (1 + gos_rasch['y'])
    gos_rasch['threshold_ser'] = (gos_rasch['Th'] / 100) * gos_rasch['db'] * gos_rasch['hp_sh']
    
    return gos_rasch


def run_government_analysis(start_q, params, now_data_date, analysis_mode='actual', target_date=None):
    """Основная функция для анализа государственного сектора"""
    print("=" * 50)
    print("АНАЛИЗ ГОСУДАРСТВЕННОГО СЕКТОРА")
    print("=" * 50)
    
    try:
        # 1. Расчет индикаторов госсектора
        gos = calculate_government_indicators(start_q, now_data_date, params)
        # 2. Подготовка данных для анализа пороговых значений
        gos = gos.loc[gos['Date'] <= now_data_date]
        gos_rasch = prepare_government_threshold_analysis(gos, now_data_date)
        # 3. Прогнозирование показателей
        gos_rasch, last_date_gos = forecast_government_indicators(gos_rasch, params)
        # 4. Применение HP-фильтра
        gos_rasch = calculate_hp_filters_government(gos_rasch, last_date_gos)
        # 5. Расчет стандартных отклонений
        stats = calculate_government_standard_deviations(gos_rasch, last_date_gos)
        
        # 6. Итерационная корректировка показателей
        gos_rasch = iterative_adjustment_ggd(
            gos_rasch, params, last_date_gos,
            stats['std_ggd'], stats['last_ggd'], stats['first_ggd']
        )
        gos_rasch = iterative_adjustment_rev(
            gos_rasch, params, last_date_gos,
            stats['std_rev'], stats['last_rev'], stats['first_rev']
        )
        # 7. Расчет дополнительных метрик
        gos_rasch = calculate_government_additional_metrics(gos_rasch)
        print("\n" + "=" * 50)
        print("АНАЛИЗ ГОСУДАРСТВЕННОГО СЕКТОРА ЗАВЕРШЕН")
        print("=" * 50)
        
        print(f"\nРезультаты анализа:")
        print(f"- Основные данные: {gos.shape[0]} записей")
        print(f"- Данные для анализа порогов: {gos_rasch.shape[0]} записей")
        print(f"- Последняя дата: {gos_rasch['Date'].iloc[-1].strftime('%Y-%m-%d')}")
        
        return gos, gos_rasch
        
    except Exception as e:
        print(f"Ошибка при выполнении анализа государственного сектора: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================================
# 10. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

def save_to_excel(bank, rasch_bank, start_q, start_copy, rasch, dom, dom_rasch, dom1, 
                  ur_licha, ur_licha_rasch, gos, gos_rasch, now_data_date, analysis_mode='actual'):
    """
    Сохраняет результаты в Excel файлы
    
    Parameters:
    -----------
    analysis_mode : str
        Режим работы: 'actual' или 'preliminary'
    """
    print("Сохранение результатов в Excel...")
    
    # Создаем папку если её нет
    import os
    output_dir = "Выходные данные"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана папка: {output_dir}")
    
    # Конвертация дат с проверкой на None
    bank = convert_dates_to_date(bank, ['Date']) if bank is not None else None
    rasch_bank = convert_dates_to_date(rasch_bank, ['Date']) if rasch_bank is not None else None
    start_copy = convert_dates_to_date(start_copy, ['Date']) if start_copy is not None else None
    start_q = convert_dates_to_date(start_q, ['Date']) if start_q is not None else None
    rasch = convert_dates_to_date(rasch, ['Date']) if rasch is not None else None
    dom = convert_dates_to_date(dom, ['Date']) if dom is not None else None
    dom_rasch = convert_dates_to_date(dom_rasch, ['Date']) if dom_rasch is not None else None
    dom1 = convert_dates_to_date(dom1, ['Date']) if dom1 is not None else None
    ur_licha = convert_dates_to_date(ur_licha, ['Date']) if ur_licha is not None else None
    ur_licha_rasch = convert_dates_to_date(ur_licha_rasch, ['Date']) if ur_licha_rasch is not None else None
    gos = convert_dates_to_date(gos, ['Date']) if gos is not None else None
    gos_rasch = convert_dates_to_date(gos_rasch, ['Date']) if gos_rasch is not None else None
    
    # Проверка на наличие данных перед сохранением
    if ur_licha is None:
        print("Предупреждение: Нет данных для сохранения в 'Юридические лица.xlsx'")
        ur_licha = pd.DataFrame()
    if ur_licha_rasch is None:
        print("Предупреждение: Нет данных для сохранения пороговых значений юридических лиц")
        ur_licha_rasch = pd.DataFrame()
    
    try:
        # Банковский сектор
        if bank is not None and rasch_bank is not None:
            file_path = os.path.join(output_dir, "Банковский сектор.xlsx")
            # Проверяем существует ли файл
            if os.path.exists(file_path):
                mode = 'a'
                if_sheet_exists = 'overlay'
            else:
                mode = 'w'
                if_sheet_exists = None
            
            with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
                if 'crb % ВВП' in bank.columns and 'crb_sa' in bank.columns:
                    bank[['Date', 'crb % ВВП', 'crb_sa']].to_excel(writer, sheet_name="Требования", index=False, startrow=2)
                if start_copy is not None and 'Leverage' in start_copy.columns:
                    start_copy[['Date', 'Leverage', 'act_bank_q', 'cap_bank_q', 'act_bank', 'cap_bank']].to_excel(
                        writer, sheet_name="Леверидж", index=False, startrow=2)
                if 'dep_all' in bank.columns:
                    bank[['Date', 'dep_all', 'dep', 'dep_ur', 'dep_fl', 'dep_byn_ur', 'dep_byn_fl', 'dep_usd_ur', 'dep_usd_fl']].to_excel(
                        writer, sheet_name="Депозиты", index=False, startrow=3)
                rasch_bank[['Date', 'crb % ВВП', 'threshold_crb', 'Leverage', 'threshold_leverage', 'crb_to_dep', 'threshold_crb_to_dep']].to_excel(
                    writer, sheet_name="расчет", index=False, startrow=1)
            print(f"  Сохранен: {file_path}")
        
        # Домашние хозяйства
        if dom is not None and dom_rasch is not None and dom1 is not None:
            file_path = os.path.join(output_dir, "Домашние хозяйства.xlsx")
            if os.path.exists(file_path):
                mode = 'a'
                if_sheet_exists = 'overlay'
            else:
                mode = 'w'
                if_sheet_exists = None
            
            with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
                if 'credits % ВВП' in dom.columns:
                    dom[['Date', 'credits % ВВП', 'credits_potreb', 'credits_finans']].to_excel(
                        writer, sheet_name="Потреб+недвиж", index=False, startrow=2)
                
                duration_cols = ['Date', 'credits_potreb', 'vidan_cred_potreb', 'Средняя задолженность по потреб. кредитам ФЛ, млн.рублей', 
                     'Прирост задолженности (потреб)', 'Погашено, млн. руб.(потреб)', 'Дюрация (потреб.)', 
                     'credits_finans', 'vidan_credits_nedv', 'Средняя задолженность по кредитам на недвиж. ФЛ, млн.рублей', 
                     'Прирост задолженности (недвиж)', 'Погашено, млн. руб.(недвиж)', 'Дюрация (недвиж.)']
                available_duration_cols = [col for col in duration_cols if col in dom.columns]
                if available_duration_cols:
                    dom[available_duration_cols].to_excel(
                        writer, sheet_name="Дюрация", index=False, startrow=1)
                
                kod_cols = ['Date', 'Задолженность по потреб. кредитам ФЛ (среднее за год), млн.рублей', 
                     'Задолженность по кредитам на недвиж. ФЛ (среднее за год), млн.рублей', 
                     'Дюрация (потреб.)', 'Дюрация (недвиж.)', 'precent_stavki_potreb_cred_q',
                     'precent_stavki_nedviz_cred_q', 'КОД', 'КОД (потреб)', 'КОД (недвиж)']
                if 'КОД' not in dom.columns:
                    dom['КОД'] = np.nan
                available_kod_cols = [col for col in kod_cols if col in dom.columns]
                if available_kod_cols:
                    dom[available_kod_cols].to_excel(
                        writer, sheet_name="КОД", index=False, startrow=2)
                
                if 'zadolz_fl_byn' in dom1.columns and 'zadol_fl_usd' in dom1.columns and 'ln(zadolz % CPI)*100' in dom1.columns:
                    dom1[['Date', 'zadolz_fl_byn', 'zadol_fl_usd', 'ln(zadolz % CPI)*100']].to_excel(
                        writer, sheet_name="Q_физлица_2005_23", index=False, startrow=2)
                
                if 'stoim_zil_nedv_q' in dom.columns:
                    dom[['Date', 'stoim_zil_nedv_q', 'stoim_zil_nedv_sr', 'stoim_zil_nedv_y', 'stoim_zil_nedv_id']].to_excel(
                        writer, sheet_name='Жил_недвиж (кв)', index=False, startrow=2)
                
                rasch_cols = ['Date', 'd_serv_a', 'threshold_serv', 'credits % ВВП', 'threshold_cred', 'ln(zadolz % CPI)*100', 
                           'факт -1 zadolz % CPI', 'факт -2 zadolz % CPI', 'факт -3 zadolz % CPI', 'HP_zadolz',
                           'threshold -1 zadolz % CPI', 'threshold -2 zadolz % CPI', 'threshold -3 zadolz % CPI', 
                           'ln(stoim_zil_nedv)', 'threshold_nedv']
                available_rasch_cols = [col for col in rasch_cols if col in dom_rasch.columns]
                if available_rasch_cols:
                    dom_rasch[available_rasch_cols].to_excel(
                        writer, sheet_name="расчет", index=False, startrow=1)
            print(f"  Сохранен: {file_path}")
        
        # Юридические лица
        file_path = os.path.join(output_dir, "Юридические лица.xlsx")
        if os.path.exists(file_path):
            mode = 'a'
            if_sheet_exists = 'overlay'
        else:
            mode = 'w'
            if_sheet_exists = None
        
        with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
            cols_duration_b = ['Date', 'BYR/USD', 'NER_usdbyn_weight',
                     'zadolz_ul_crat_byn', 'vidan_cred_ul_crat_byn', 'Средняя задолж., ЮЛ, ИВ, К.', 
                     'Прирост задолж. ЮЛ, НВ, К.', 'Погашено, ЮЛ, НВ, К.', 
                     'Дюрация приведенная (ЮЛ, НВ, К, банки), кварталов',
                     'zadolz_ul_dolg_byn', 'vidan_cred_ul_dolg_byn', 'Средняя задолж., ЮЛ, НВ, Д.', 
                     'Прирост задолж. ЮЛ, НВ, Д.', 'Погашено, ЮЛ, НВ, Д.', 
                     'Дюрация приведенная (ЮЛ, НВ, Д, банки), кварталов',
                     'zadolz_ul_crat_usd2', 'vidan_cred_ul_crat_usd2', 'Средняя задолж., ЮЛ, ИВ, К.', 
                     'Прирост задолж. ЮЛ, ИВ, К.', 'Погашено, ЮЛ, ИВ, К.', 
                     'Дюрация приведенная (ЮЛ, ИВ, К, банки), кварталов',
                     'zadolz_ul_dolg_usd2', 'vidan_cred_ul_dolg_usd2', 'Средняя задолж., ЮЛ, ИВ, Д.', 
                     'Прирост задолж. ЮЛ, ИВ, Д.', 'Погашено, ЮЛ, ИВ, Д.', 
                     'Дюрация приведенная (ЮЛ, ИВ, Д, банки), кварталов']
            available_cols = [col for col in cols_duration_b if col in ur_licha.columns]
            if available_cols:
                ur_licha[available_cols].to_excel(
                    writer, sheet_name='Дюрация Б', index=False, startrow=2)
            
            cols_kod_b = ['Date', 'Средняя задолж., ЮЛ, НВ, К. (среднее за год)', 
                     'Средняя задолж., ЮЛ, НВ, Д. (среднее за год)', 
                     'Дюрация приведенная (ЮЛ, НВ, К, банки), кварталов', 
                     'Дюрация приведенная (ЮЛ, НВ, Д, банки), кварталов',
                     'precent_stavki_ul_byn_q', 'КОД (ЮЛ, К, НВ)', 'КОД (ЮЛ, Д, НВ)', 
                     'Средняя задолж., ЮЛ, ИВ, К. (среднее за год)',  
                     'Средняя задолж., ЮЛ, ИВ, Д. (среднее за год)', 
                     'Дюрация приведенная (ЮЛ, ИВ, К, банки), кварталов', 
                     'Дюрация приведенная (ЮЛ, ИВ, Д, банки), кварталов', 
                     'Рублевая доходность кредитов банков юрлицам в ИВ, %', 
                     'КОД (ЮЛ, К, ИВ)', 'КОД (ЮЛ, Д, ИВ)']
            available_cols = [col for col in cols_kod_b if col in ur_licha.columns]
            if available_cols:
                ur_licha[available_cols].to_excel(
                    writer, sheet_name='КОД Б', index=False, startrow=2)
            
            cols_duration_n = ['Date', 'BYR/USD', 'NER_usdbyn_weight',
                     'vvd_crat', 'Выдано (ЮЛ, Н, К)', 'Средняя задолженность по краткосрочным кредитам ЮЛ в ИВ, млн долл США', 
                     'Прирост задолженности (ЮЛ, Н, К)', 'pogash_kk_n', 'Приведенное погашение задолженности (ЮЛ, Н, К)', 
                     'Дюрация приведенная (ЮЛ, ИВ, К, Н), кварталов',
                     'vvd_dolg', 'Выдано (ЮЛ, Н, Д)', 'Средняя задолженность по долгосрочным кредитам ЮЛ в ИВ, млн долл США', 
                     'Прирост задолженности (ЮЛ, Н, Д)', 'pogash_dk_n', 
                     'Дюрация приведенная (ЮЛ, ИВ, Д, Н), кварталов']
            available_cols = [col for col in cols_duration_n if col in ur_licha.columns]
            if available_cols:
                ur_licha[available_cols].to_excel(
                    writer, sheet_name='Дюрация Н', startrow=2, index=False)
            
            cols_kod_n = ['Date', 'Задолж. (ЮЛ, Н, К, ИВ, кредиты, ср за год, эквив)', 
                     'Задолж. (ЮЛ, Н, Д, ИВ, кредиты, ср за год, эквив)', 
                     'Дюрация приведенная (ЮЛ, ИВ, К, Н), кварталов', 
                     'Дюрация приведенная (ЮЛ, ИВ, Д, Н), кварталов',  
                     'Рублевая доходность кредитов нерезидентов юрлицам в ИВ, %', 
                     'КОД (ЮЛ, К, ИВ, Н)', 'КОД (ЮЛ, Д, ИВ, Н)',
                     'Задолженность (ЮЛ, Н, К, ИВ, ср за год)', 'Задолженность (ЮЛ, Н, Д, ИВ, ср за год)', 
                     'IntPymt_NRes_q_thUSD', 'IntPymt_NRes_q_thUSD_yy', 
                     'Процентная ставка по кредитам банков нерезидентов в ИВ, %']
            available_cols = [col for col in cols_kod_n if col in ur_licha.columns]
            if available_cols:
                ur_licha[available_cols].to_excel(
                    writer, sheet_name='КОД Н', startrow=2, index=False)
            
            cols_rasch = ['Date', 'dsr', 'threshold_dsr', 'ggr', 'threshold_ggr', 'invr', 'threshold_invr', 
                           'cest', 'threshold_cest', 'ln_cred*100', 'факт -1 ln_cred*100', 'факт -2 ln_cred*100', 
                           'факт -3 ln_cred*100', 'HP_ln_cred', 'threshold -1 ln_cred*100', 
                           'threshold -2 ln_cred*100', 'threshold -3 ln_cred*100']
            available_cols = [col for col in cols_rasch if col in ur_licha_rasch.columns]
            if available_cols:
                ur_licha_rasch[available_cols].to_excel(
                    writer, sheet_name='расчет', index=False, startrow=1)
        print(f"  Сохранен: {file_path}")
        
        # Государственный сектор
        if gos is not None and gos_rasch is not None:
            file_path = os.path.join(output_dir, "Госсектор.xlsx")
            if os.path.exists(file_path):
                mode = 'a'
                if_sheet_exists = 'overlay'
            else:
                mode = 'w'
                if_sheet_exists = None
            
            with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
                gos_cols = ['Date', 'Rev_ConsBud_incFSZN_mnBYN', 'Exp_ConsBud_incFSZN_mnBYN', 'Bal_ConsBud_mnBYN', 
                     'Rev_ConsBud_exFSZN_mnBYN', 'Rev_FSZN_mnBYN', 'Exp_ConsBud_exFSZN_mnBYN', 'Exp_FSZN_mnBYN', 
                     'Rev_ConsBud_exFSZN_YtD_mnBYN', 'Rev_FSZN_YtD_mnBYN', 'Exp_ConsBud_exFSZN_YtD_mnBYN', 
                     'Exp_FSZN_YtD_mnBYN']
                available_gos_cols = [col for col in gos_cols if col in gos.columns]
                if available_gos_cols:
                    gos[available_gos_cols].to_excel(
                        writer, sheet_name='Доходы_расходы КБ', index=False, startrow=3)
                
                extdebt_cols = ['Date', 'ExtDebtSvc_Gov_all_mnUSD', 'ExtDebtSvc_Gov_Int_mnUSD', 'ExtDebtSvc_Gov_Pr_mnUSD', 
                     'ExtDebtSvc_Gov_avg_mnUSD', 'ExtDebtSvc_Gov_allavg_mnUSD']
                available_extdebt_cols = [col for col in extdebt_cols if col in gos.columns]
                if available_extdebt_cols:
                    gos[available_extdebt_cols].to_excel(
                        writer, sheet_name='Обслуживание ГВД', index=False, startrow=3)
                
                debt_cols = ['Date', 'TotDebt_CGov_GInc_mnBYN', 'TotDebt_CGov_GInc_mnBYN % ВВП', 'TotDebt_CGov_GInc_mnUSD', 
                     'TotDebt_CGov_GInc_mnUSD % ВВП', 'ggd', 'Доля внешнего долга (млн. BYN), %', 
                     'Доля внешнего долга (млн. USD), %']
                available_debt_cols = [col for col in debt_cols if col in gos.columns]
                if available_debt_cols:
                    gos[available_debt_cols].to_excel(
                        writer, sheet_name='Госдолг (кв)', index=False, startrow=3)
                
                gos_rasch_cols = ['Date', 'ggd', 'threshold_ggd', 'ser_exdebt_gdp', 'db', 'Th', 'Sh_ExtD_TotD', 'hp_sh', 
                           'threshold_ser', 'bb_gdp', 'GDP_EQ_q_sa', 'y', 'd', 'threshold_bb_gdp']
                available_gos_rasch_cols = [col for col in gos_rasch_cols if col in gos_rasch.columns]
                if available_gos_rasch_cols:
                    gos_rasch[available_gos_rasch_cols].to_excel(
                        writer, sheet_name='расчет', index=False, startrow=1)
            print(f"  Сохранен: {file_path}")
        
        # Вся экономика
        if start_q is not None and rasch is not None:
            file_path = os.path.join(output_dir, "Вся экономика.xlsx")
            if os.path.exists(file_path):
                mode = 'a'
                if_sheet_exists = 'overlay'
            else:
                mode = 'w'
                if_sheet_exists = None
            
            with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
                start_q_cols = ['Date', 'vvd', 'vvd % ВВП', 'vvd_mean', 'vvd_mean % ВВП', 'annual_diff_vvd', 
                         'sto', 'sto_y', '- sto_y', 'sto_y % ВВП']
                available_start_q_cols = [col for col in start_q_cols if col in start_q.columns]
                if available_start_q_cols:
                    start_q[available_start_q_cols].to_excel(
                        writer, sheet_name='СТО и ВД', index=False, startrow=2)
                
                rasch_cols = ['Date', 'vvd_mean % ВВП', 'threshold_vvd', 'sto_y % ВВП', 'threshold_sto']
                available_rasch_cols = [col for col in rasch_cols if col in rasch.columns]
                if available_rasch_cols:
                    rasch[available_rasch_cols].to_excel(
                        writer, sheet_name='расчет', index=False, startrow=1)
            print(f"  Сохранен: {file_path}")
        
        print("\n✅ Все результаты успешно сохранены в папку 'Выходные данные'")
        
    except Exception as e:
        print(f"❌ Ошибка при сохранении в Excel: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 11. ОСНОВНАЯ ФУНКЦИЯ - КОМПЛЕКСНЫЙ АНАЛИЗ
# ============================================================================

def run_comprehensive_analysis(mode='auto', target_date=None):
    """
    Запускает комплексный анализ всех секторов экономики
    
    Parameters:
    -----------
    mode : str
        Режим работы: 'auto' - автоматическое определение,
        'actual' - фактический режим,
        'preliminary' - предварительный режим
    target_date : datetime, optional
        Целевая дата для отчета. Если None, используется текущая дата.
    
    Returns:
    --------
    dict: результаты анализа
    """
    print("=" * 70)
    print("КОМПЛЕКСНЫЙ АНАЛИЗ ЭКОНОМИЧЕСКИХ ИНДИКАТОРОВ")
    print("=" * 70)
    
    # Инициализируем переменную analysis_mode до всех блоков
    analysis_mode = mode
    now_data_date = None
    
    try:
        # Шаг 1: Загрузка данных
        start_q, start_m, params = load_data()
        
        # Шаг 2: Загрузка данных обменного курса
        start_date = datetime(1999, 1, 1).date()
        end_date = datetime.now().date()
        rates_m = fetch_exchange_rates(start_date, end_date)
        
        # Шаг 3: Обработка месячных данных
        start_m = process_monthly_data(start_m, rates_m)
        
        # Шаг 4: Расчет квартальных показателей
        start_m = calculate_quarterly_indicators(start_m)
        
        # Шаг 5: Обработка квартальных данных
        start_q = process_quarterly_data(start_q, start_m)
        
        # Шаг 6: Расчет дополнительных метрик
        start_q = calculate_additional_quarterly_metrics(start_q)
        
        # Шаг 7: ОПРЕДЕЛЕНИЕ РЕЖИМА РАБОТЫ
        # Определяем целевую дату для отчета
        if target_date is None:
            # Если дата не указана, используем последнюю доступную дату
            target_date = start_q['Date'].max()
        else:
            target_date = pd.Timestamp(target_date)
        
        # ОПРЕДЕЛЕНИЕ РЕЖИМА РАБОТЫ
        if mode == 'auto':
            analysis_mode = determine_mode(start_q, target_date)
        else:
            analysis_mode = mode
        
        print(f"\nРежим работы: {analysis_mode.upper()}")
        print(f"Целевая дата отчета: {target_date.strftime('%Y-%m-%d')}")
        
        # ============================================================
        # ФИЛЬТРАЦИЯ ДАННЫХ: ТОЛЬКО ДЛЯ ФАКТИЧЕСКОГО РЕЖИМА
        # ============================================================
        if analysis_mode == 'actual':
            # В фактическом режиме фильтруем данные до текущего квартала
            start_q, now_data_date = filter_data_to_current_quarter(start_q)
            print(f"Фактический режим: данные отфильтрованы до {now_data_date.strftime('%Y-%m-%d')}")
        else:
            # В предварительном режиме НЕ фильтруем данные
            # Используем все доступные данные
            now_data_date = start_q['Date'].max()
            print(f"Предварительный режим: данные НЕ фильтруются, используются все доступные данные до {now_data_date.strftime('%Y-%m-%d')}")
            
            # Для предварительного режима определяем дату предыдущего квартала
            prev_date = target_date - pd.DateOffset(months=3)
            print(f"  Критические показатели будут рассчитаны до: {prev_date.strftime('%Y-%m-%d')}")
        # ============================================================
        
        if analysis_mode == 'preliminary':
            print("\nПРЕДВАРИТЕЛЬНЫЙ РЕЖИМ:")
            print("Критические показатели будут рассчитаны до предыдущего квартала")
            print("Сдвиг производных критических показателей будет применен при формировании барометра")
            print()

        # Шаг 8: Расчет годовых метрик
        start_q = calculate_yearly_metrics(start_q)
        
        # Шаг 9: Прогнозирование ВВД
        start_q = forecast_vvd(start_q, params, analysis_mode, target_date)
        # Шаг 10: Расчет пороговых значений долга (с передачей режима)
        rasch = calculate_debt_thresholds(start_q, params, analysis_mode, target_date)
        
        # Шаг 11: Расчет банковских индикаторов
        bank = calculate_banking_indicators(start_q, now_data_date)
        rasch_bank = calculate_banking_thresholds(bank, params, analysis_mode, target_date)
        
        # Шаг 12: Расчет индикаторов домашних хозяйств (с передачей режима)
        dom_rasch, dom, dom1 = calculate_domestic_thresholds(
            start_q, now_data_date, params, analysis_mode, target_date
        )
        
        # Создаем dom1_full для обратной совместимости
        dom1 = pd.merge(start_q[['Date', 'gdp_nsa_byn_y', 'credits_potreb', 'credits_finans', 
                       'zadolz_fl_byn', 'zadol_fl_usd', 'AvBi_CPI_q_sa']], dom1[['Date', 'ln(zadolz % CPI)*100']], on='Date', how='left')
        dom1 = dom1.loc[(dom1['Date'] >= pd.Timestamp(2002, 1, 1)) & (dom1['Date'] <= now_data_date)]
        
        # Шаг 13: Анализ юридических лиц (с передачей режима)
        ur_licha, ur_licha_rasch = run_corporate_analysis(
            start_q, rates_m, params, now_data_date, analysis_mode, target_date
        )
        # Шаг 14: Анализ государственного сектора (с передачей режима)
        gos, gos_rasch = run_government_analysis(
            start_q, params, now_data_date, analysis_mode, target_date
        )
        
        # Шаг 15: Подготовка данных для сохранения
        start_copy = start_q.loc[(start_q['Date'] >= pd.Timestamp(2005, 1, 1)) & 
                                 (start_q['Date'] <= now_data_date)].copy()
        
        # Шаг 16: Сохранение результатов
        save_to_excel(bank, rasch_bank, start_q, start_copy, rasch, dom, dom_rasch, dom1,
                      ur_licha, ur_licha_rasch, gos, gos_rasch, now_data_date, analysis_mode)
        
        print("\n" + "=" * 70)
        print("КОМПЛЕКСНЫЙ АНАЛИЗ УСПЕШНО ЗАВЕРШЕН")
        print("=" * 70)
        
        return {
            'start_q': start_q,
            'start_m': start_m,
            'params': params,
            'debt_thresholds': rasch,
            'banking_data': bank,
            'banking_thresholds': rasch_bank,
            'domestic_thresholds': dom_rasch,
            'domestic_data': dom,
            'corporate_data': ur_licha,
            'corporate_thresholds': ur_licha_rasch,
            'government_data': gos,
            'government_thresholds': gos_rasch,
            'analysis_mode': analysis_mode,
            'target_date': target_date
        }
        
    except Exception as e:
        print(f"Ошибка при выполнении комплексного анализа: {e}")
        import traceback
        traceback.print_exc()
        return None