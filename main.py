# main.py
import pandas as pd
import numpy as np
from functions import calc_auc
from datetime import datetime
import sys
import os


def clear_screen():
    """Очищает экран терминала"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Выводит заголовок программы"""
    print("=" * 70)
    print("       АНАЛИЗ УЯЗВИМОСТЕЙ ЭКОНОМИКИ (БАРОМЕТР)")
    print("=" * 70)
    print()


def print_menu():
    """Выводит меню выбора режима"""
    print("Выберите режим работы:")
    print("  1. Автоматический режим")
    print("     - Система сама определит наличие данных")
    print("     - Если данных за отчетный квартал нет - переключится на предварительный режим")
    print()
    print("  2. Фактический режим (ACTUAL)")
    print("     - Используются фактические данные за отчетный квартал")
    print("     - Требует наличия всех данных")
    print()
    print("  3. Предварительный режим (PRELIMINARY)")
    print("     - Для критических производных показателей применяется сдвиг на один квартал назад")
    print("     - Позволяет строить отчет при неполных данных")
    print()
    print("  4. Выход")
    print()
    print("-" * 70)


def get_user_choice():
    """Получает выбор пользователя"""
    while True:
        try:
            choice = input("Ваш выбор (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print("Ошибка: введите число от 1 до 4")
        except KeyboardInterrupt:
            print("\n\nВыход из программы...")
            sys.exit(0)


def get_target_date():
    """Получает целевую дату от пользователя"""
    print("\nВведите целевую дату для отчета (опционально)")
    print("Формат: ГГГГ-ММ-ДД (например, 2026-06-01)")
    print("Нажмите Enter для использования текущей даты")
    
    while True:
        date_input = input("Дата: ").strip()
        if date_input == "":
            return None
        try:
            target_date = datetime.strptime(date_input, '%Y-%m-%d')
            return target_date
        except ValueError:
            print("Ошибка: неверный формат даты. Используйте ГГГГ-ММ-ДД")


def parse_arguments():
    """Парсит аргументы командной строки для автоматического режима"""
    mode = 'menu'
    target_date = None
    
    for i, arg in enumerate(sys.argv):
        if arg == '--mode' and i + 1 < len(sys.argv):
            mode_arg = sys.argv[i + 1].lower()
            if mode_arg in ['auto', 'actual', 'preliminary']:
                mode = mode_arg
        elif arg == '--date' and i + 1 < len(sys.argv):
            try:
                target_date = datetime.strptime(sys.argv[i + 1], '%Y-%m-%d')
            except ValueError:
                print(f"Неверный формат даты: {sys.argv[i + 1]}. Используйте YYYY-MM-DD")
        elif arg in ['-h', '--help']:
            print("Использование: python main.py [--mode auto|actual|preliminary] [--date YYYY-MM-DD]")
            print()
            print("  --mode     : Режим работы (auto, actual, preliminary)")
            print("  --date     : Целевая дата для отчета (формат: YYYY-MM-DD)")
            print("  -h, --help : Показать эту справку")
            print()
            print("Если режим не указан, будет показано интерактивное меню.")
            sys.exit(0)
    
    return mode, target_date


def run_analysis(mode, target_date):
    """Запускает анализ в выбранном режиме"""
    try:
        print("Запуск анализа...")
        print("-" * 70)
        
        from functions import calculate
        
        # Выполняем комплексный анализ
        results = calculate.run_comprehensive_analysis(mode=mode, target_date=target_date)
        
        if results and results.get('analysis_mode'):
            analysis_mode = results['analysis_mode']
            
            print("\n" + "=" * 70)
            print(f"АНАЛИЗ ЗАВЕРШЕН В РЕЖИМЕ: {analysis_mode.upper()}")
            print("=" * 70)
            
            if analysis_mode == 'preliminary':
                print("\nПРИМЕЧАНИЕ: Использован предварительный режим")
                print("Для критических производных показателей применен сдвиг на один квартал назад")
                print("Сдвинутые производные показатели:")
                print("  - ca (sto_y % ВВП) и threshold_sto")
                print("  - gd_gdp (vvd_mean % ВВП) и threshold_vvd")
                print("  - d_srv (КОД) и threshold_serv")
                print("  - l_gdp (credits % ВВП) и threshold_cred")
                print("  - rcgr_h1, rcgr_h2, rcgr_h3 и соответствующие thresholds")
                print("  - pest (ln(stoim_zil_nedv)) и threshold_nedv")
            
            print("\nГенерация барометра уязвимостей...")
            # ПЕРЕДАЕМ РЕЗУЛЬТАТЫ И РЕЖИМ В itog()
            calc_auc.itog(results=results, analysis_mode=analysis_mode)
            
            print("\n" + "=" * 70)
            print("РЕЗУЛЬТАТЫ СОХРАНЕНЫ В ФАЙЛ: Valnerabilities_Barometer.xlsx")
            print("=" * 70)
            
        else:
            print("\nОшибка: не удалось выполнить анализ")
            
    except Exception as e:
        print(f"\nОШИБКА при выполнении анализа: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Главная функция программы"""
    clear_screen()
    print_header()
    
    mode_arg, target_date_arg = parse_arguments()
    
    if mode_arg != 'menu':
        mode = mode_arg
        target_date = target_date_arg
        print(f"Режим работы: {mode.upper()}")
        if target_date:
            print(f"Целевая дата: {target_date.strftime('%Y-%m-%d')}")
        print()
        run_analysis(mode, target_date)
        return
    
    while True:
        print_menu()
        choice = get_user_choice()
        
        if choice == 4:
            print("\nВыход из программы...")
            break
        
        mode_map = {1: 'auto', 2: 'actual', 3: 'preliminary'}
        mode = mode_map[choice]
        target_date = get_target_date()
        
        clear_screen()
        print_header()
        print(f"Выбран режим: {mode.upper()}")
        if target_date:
            print(f"Целевая дата: {target_date.strftime('%Y-%m-%d')}")
        print()
        
        run_analysis(mode, target_date)
        
        print("\n" + "=" * 70)
        input("Нажмите Enter для продолжения...")
        clear_screen()
        print_header()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
        sys.exit(0)