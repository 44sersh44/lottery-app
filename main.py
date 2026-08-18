import flet as ft
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.profile_manager import ProfileManager
from services.data_service import DataService
from services.prediction_service import PredictionService
from services.stats_service import StatsService


def main(page: ft.Page):
    page.title = "Lottery Predictor"
    page.theme_mode = ft.ThemeMode.DARK
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 20

    # --- Сервисы ---
    profile_manager = ProfileManager("profiles")
    data_service = DataService(profile_manager)
    prediction_service = PredictionService(data_service)
    stats_service = StatsService(data_service)

    # --- Интерфейс ---
    status_text = ft.Text("📂 Выберите профиль", size=16, color=ft.Colors.GREY_400)
    loading_indicator = ft.ProgressRing(visible=False)
    train_status = ft.Text("", size=14, color=ft.Colors.GREY_500)

    # --- Функция обновления статуса ---
    def update_status(msg):
        status_text.value = msg
        page.update()

    # --- Профили: по умолчанию "7-49" ---
    profile_names = profile_manager.get_profile_names()
    if not profile_names:
        profile_names = ["Нет профилей"]

    default_profile = "7-49"
    if default_profile in profile_names:
        profile_manager.select_profile(default_profile)
        initial_value = default_profile
        status_text.value = f"✅ Активный профиль: {default_profile}"
    else:
        if profile_names and profile_names[0] != "Нет профилей":
            profile_manager.select_profile(profile_names[0])
            initial_value = profile_names[0]
            status_text.value = f"✅ Активный профиль: {initial_value}"
        else:
            initial_value = None

    profile_dropdown = ft.Dropdown(
        options=[ft.dropdown.Option(name) for name in profile_names],
        value=initial_value,
        width=300,
    )

    def on_profile_change(e):
        name = e.control.value
        if name and name != "Нет профилей":
            profile_manager.select_profile(name)
            status_text.value = f"✅ Активный профиль: {name}"
            page.update()

    profile_dropdown.on_change = on_profile_change

    # --- Заглушка для загрузки CSV ---
    def load_csv():
        status_text.value = "⚠️ Загрузка CSV доступна только на ПК"
        page.update()

    # --- Результаты ---
    result_column = ft.Column(spacing=10, scroll=ft.ScrollMode.AUTO)
    models_ready = [False]

    # ---------- Функции ----------
    def make_prediction():
        if not models_ready[0]:
            status_text.value = "⚠️ Сначала обучите модели (пункт 11)"
            page.update()
            return
        result_column.controls.clear()
        loading_indicator.visible = True
        page.update()

        def do_predict():
            try:
                result = prediction_service.make_prediction(5, "1", None)
                rows = result.get("rows", [])
                page.run_coroutine(update_results(rows))
            except Exception as e:
                page.run_coroutine(show_error(str(e)))
            finally:
                loading_indicator.visible = False
                page.update()

        threading.Thread(target=do_predict, daemon=True).start()

    async def update_results(rows):
        result_column.controls.clear()
        if rows:
            for i, row in enumerate(rows[:5]):
                pred_text = " ".join(str(x) for x in row[1:]) if len(row) > 1 else str(row)
                result_column.controls.append(
                    ft.Container(
                        content=ft.Text(f"{row[0]}. {pred_text}", size=16),
                        padding=8,
                        bgcolor=ft.Colors.GREY_900 if i % 2 == 0 else ft.Colors.GREY_850,
                        border_radius=8
                    )
                )
        else:
            result_column.controls.append(ft.Text("Нет данных", color=ft.Colors.GREY_500))
        await page.update_async()

    async def show_error(msg):
        status_text.value = f"❌ {msg}"
        await page.update_async()

    def show_last_blocks():
        result_column.controls.clear()
        try:
            blocks = stats_service.show_last_blocks(10)
            if not blocks:
                result_column.controls.append(ft.Text("Нет данных о тиражах", color=ft.Colors.GREY_500))
                page.update()
                return
            for item in blocks:
                block_str = f"поле1: {item['block']}"
                if item.get('block2') is not None:
                    block_str += f"  поле2: {item['block2']}"
                draw_info = f"Тираж {item['draw_id']}" if item.get('draw_id') else ""
                date_time = f"{item['date']} {item['time']}".strip()
                card = ft.Container(
                    content=ft.Column([
                        ft.Text(f"№{item['index']}   {block_str}", size=15),
                        ft.Text(f"{draw_info}   {date_time}", size=12, color=ft.Colors.GREY_400),
                    ]),
                    padding=8,
                    bgcolor=ft.Colors.GREY_900,
                    border_radius=8,
                    margin=ft.margin.symmetric(vertical=4),
                )
                result_column.controls.append(card)
        except Exception as e:
            result_column.controls.append(ft.Text(f"❌ Ошибка: {e}", color=ft.Colors.RED))
        page.update()

    def show_rating():
        result_column.controls.clear()
        try:
            detailed = prediction_service.evaluate_models_detailed(0.2)
            if not detailed:
                result_column.controls.append(ft.Text("Не удалось получить рейтинг", color=ft.Colors.GREY_500))
                page.update()
                return
            sorted_items = sorted(detailed.items(), key=lambda x: x[1]['avg_sq'], reverse=True)
            for i, (name, stats) in enumerate(sorted_items[:10], 1):
                card = ft.Container(
                    content=ft.Text(f"{i}. {name} — рейтинг: {stats['avg_sq']:.2f} (среднее: {stats['avg']:.2f})"),
                    padding=8,
                    bgcolor=ft.Colors.GREY_900 if i % 2 == 0 else ft.Colors.GREY_850,
                    border_radius=8,
                )
                result_column.controls.append(card)
        except Exception as e:
            result_column.controls.append(ft.Text(f"❌ Ошибка: {e}", color=ft.Colors.RED))
        page.update()

    def train_models():
        try:
            loading_indicator.visible = True
            train_status.value = "⏳ Обучение моделей... (подождите)"
            page.update()
            prediction_service.train_all(force_retrain=False, parallel=True)
            models_ready[0] = True
            loading_indicator.visible = False
            train_status.value = "✅ Модели готовы"
            page.update()
        except Exception as e:
            loading_indicator.visible = False
            train_status.value = f"❌ Ошибка обучения: {e}"
            page.update()

    # --- Меню (аккордеон) ---
    profile_menu = ft.ExpansionTile(
        title=ft.Text("📁 Управление профилями", weight=ft.FontWeight.BOLD),
        controls=[
            ft.ListTile(title=ft.Text("1. Создать профиль"), on_click=lambda e: update_status("🔧 Создание профиля (в разработке)")),
            ft.ListTile(title=ft.Text("2. Выбрать профиль"), on_click=lambda e: update_status("🔧 Выбор профиля (в разработке)")),
            ft.ListTile(title=ft.Text("3. Загрузить CSV"), on_click=lambda e: load_csv()),
            ft.ListTile(title=ft.Text("4. Удалить профиль"), on_click=lambda e: update_status("🔧 Удаление профиля (в разработке)")),
        ]
    )

    predict_menu = ft.ExpansionTile(
        title=ft.Text("🔮 Прогнозы", weight=ft.FontWeight.BOLD),
        controls=[
            ft.ListTile(title=ft.Text("5. Сделать прогноз (ансамбль)"), on_click=lambda e: make_prediction()),
            ft.ListTile(title=ft.Text("6. Прогноз по позициям"), on_click=lambda e: update_status("🔧 Прогноз по позициям (в разработке)")),
            ft.ListTile(title=ft.Text("7. Сравнить прогноз с реальным"), on_click=lambda e: update_status("🔧 Сравнение (в разработке)")),
            ft.ListTile(title=ft.Text("21. 🎯 Фильтрованный прогноз"), on_click=lambda e: update_status("🔧 Фильтрованный прогноз (в разработке)")),
        ]
    )

    stats_menu = ft.ExpansionTile(
        title=ft.Text("📊 Статистика", weight=ft.FontWeight.BOLD),
        controls=[
            ft.ListTile(title=ft.Text("8. Частотный анализ"), on_click=lambda e: update_status("🔧 Частотный анализ (в разработке)")),
            ft.ListTile(title=ft.Text("9. Последние 10 блоков"), on_click=lambda e: show_last_blocks()),
            ft.ListTile(title=ft.Text("10. Очистить историю"), on_click=lambda e: update_status("🔧 Очистка истории (в разработке)")),
        ]
    )

    model_menu = ft.ExpansionTile(
        title=ft.Text("🧠 Модели", weight=ft.FontWeight.BOLD),
        controls=[
            ft.ListTile(title=ft.Text("11. Обучить все модели"), on_click=lambda e: train_models()),
            ft.ListTile(title=ft.Text("12. Рейтинг методов"), on_click=lambda e: show_rating()),
            ft.ListTile(title=ft.Text("13. Бэктест"), on_click=lambda e: update_status("🔧 Бэктест (в разработке)")),
            ft.ListTile(title=ft.Text("14. Прогноз от каждой модели"), on_click=lambda e: update_status("🔧 Прогноз от каждой модели (в разработке)")),
            ft.ListTile(title=ft.Text("15. Настройка методов"), on_click=lambda e: update_status("🔧 Настройка методов (в разработке)")),
        ]
    )

    analysis_menu = ft.ExpansionTile(
        title=ft.Text("📐 Глубокий анализ", weight=ft.FontWeight.BOLD),
        controls=[
            ft.ListTile(title=ft.Text("20. Статистический анализ"), on_click=lambda e: update_status("🔧 Статистический анализ (в разработке)")),
            ft.ListTile(title=ft.Text("23. Байесовский анализ"), on_click=lambda e: update_status("🔧 Байесовский анализ (в разработке)")),
            ft.ListTile(title=ft.Text("24. Монте-Карло симуляции"), on_click=lambda e: update_status("🔧 Монте-Карло (в разработке)")),
        ]
    )

    opt_menu = ft.ExpansionTile(
        title=ft.Text("⚙️ Оптимизация", weight=ft.FontWeight.BOLD),
        controls=[
            ft.ListTile(title=ft.Text("17. Walk-Forward оптимизация"), on_click=lambda e: update_status("🔧 Walk-Forward (в разработке)")),
            ft.ListTile(title=ft.Text("25. Прогноз со сдвигом"), on_click=lambda e: update_status("🔧 Прогноз со сдвигом (в разработке)")),
            ft.ListTile(title=ft.Text("29. Локальная оптимизация"), on_click=lambda e: update_status("🔧 Локальная оптимизация (в разработке)")),
            ft.ListTile(title=ft.Text("30. Создать ансамбль из лучших"), on_click=lambda e: update_status("🔧 Создание ансамбля (в разработке)")),
        ]
    )

    extra_menu = ft.ExpansionTile(
        title=ft.Text("🔄 Дополнительно", weight=ft.FontWeight.BOLD),
        controls=[
            ft.ListTile(title=ft.Text("16. Показать графики"), on_click=lambda e: update_status("🔧 Графики (в разработке)")),
            ft.ListTile(title=ft.Text("32. Сравнить прогнозы моделей"), on_click=lambda e: update_status("🔧 Сравнение прогнозов (в разработке)")),
            ft.ListTile(title=ft.Text("33. Анализ консенсуса чисел"), on_click=lambda e: update_status("🔧 Консенсус (в разработке)")),
        ]
    )

    menu_column = ft.Column(
        controls=[
            profile_menu,
            predict_menu,
            stats_menu,
            model_menu,
            analysis_menu,
            opt_menu,
            extra_menu,
        ],
        spacing=0,
    )

    # --- Сборка страницы ---
    page.add(
        ft.Text("🎯 Лотерейный Предиктор", size=28, weight=ft.FontWeight.BOLD),
        ft.Divider(height=10),
        ft.Text("Профиль:", size=16, weight=ft.FontWeight.BOLD),
        profile_dropdown,
        status_text,
        ft.Row([loading_indicator, train_status], alignment=ft.MainAxisAlignment.START),
        ft.Divider(height=15),
        ft.Text("📋 Меню:", size=18, weight=ft.FontWeight.BOLD),
        menu_column,
        ft.Divider(height=15),
        ft.Text("📊 Результаты:", size=16, weight=ft.FontWeight.BOLD),
        result_column,
    )

    # --- Фоновое обучение моделей (если профиль уже выбран) ---
    if profile_dropdown.value and profile_dropdown.value != "Нет профилей":
        threading.Thread(target=train_models, daemon=True).start()


# Запуск в отдельном окне (для ПК)
ft.app(target=main, view=ft.AppView.FLET_APP)