
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from datetime import datetime
import os

class EpidemicEngine:
    @staticmethod
    def sirs_ode(y, t, beta, gamma, sigma, N):
        S, I, R = y
        dS = -beta * S * I / N + sigma * R
        dI = beta * S * I / N - gamma * I
        dR = gamma * I - sigma * R
        return [dS, dI, dR]

    @classmethod
    def solve(cls, days, N, beta, gamma, sigma, I0, R0=0):
        S0 = N - I0 - R0
        y0 = [S0, I0, R0]
        t = np.arange(0, days, 1)
        sol = odeint(cls.sirs_ode, y0, t, args=(beta, gamma, sigma, N))
        S, I, R = sol.T
        return (S, I, R), t

class Calibrator:
    @staticmethod
    def loss_function(params, actual_data, N):
        beta, gamma, sigma, I0 = params
        if any(p <= 0 for p in params):
            return 1e15
        days = len(actual_data)
        (S, I, R), _ = EpidemicEngine.solve(days, N, beta, gamma, sigma, I0)
        weights = np.linspace(1, 3, days)
        return np.mean(weights * (I - actual_data) ** 2)

    @classmethod
    def fit(cls, actual_data, N, bounds=None, smooth=True, window_length=7, polyorder=2):
        if smooth:
            actual_data = savgol_filter(actual_data, window_length, polyorder, mode='mirror')
            actual_data = np.maximum(actual_data, 0)
        init_guess = [0.4, 0.1, 0.01, max(1, actual_data[0])]
        if bounds is None:
            bounds = [
                (0.01, 2.0),
                (0.01, 0.5),
                (0.0, 0.1),
                (0.1, N / 100)
            ]
        result = minimize(
            lambda p: cls.loss_function(p, actual_data, N),
            init_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'disp': False}
        )
        if not result.success:
            print("Оптимизация не сошлась:", result.message)
            return init_guess, 1e15
        return result.x, result.fun

def preprocess_data(df, country=None, start_date=None, end_date=None, N_days_active=10, auto_population=True):
    if 'dateRep' in df.columns:
        df = df.rename(columns={'dateRep': 'date'})
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        for col in df.columns:
            if 'date' in col.lower():
                df = df.rename(columns={col: 'date'})
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                break
        else:
            raise ValueError("Не найдена колонка с датой")

    if 'countriesAndTerritories' in df.columns:
        df = df.rename(columns={'countriesAndTerritories': 'country'})
    elif 'location' in df.columns:
        df = df.rename(columns={'location': 'country'})
    elif 'country' not in df.columns:
        df['country'] = 'Global'

    if country:
        mask = df['country'].str.contains(country, case=False, na=False)
        if not mask.any():
            mask = df['country'].str.lower() == country.lower()
        df = df[mask].copy()
        if df.empty:
            raise ValueError(f'Страна "{country}" не найдена в данных')

    df = df.sort_values('date').reset_index(drop=True)

    if 'cases' in df.columns:
        new_cases = df['cases'].fillna(0)
    elif 'new_cases' in df.columns:
        new_cases = df['new_cases'].fillna(0)
    else:
        if 'confirmed' in df.columns:
            new_cases = df['confirmed'].diff().fillna(0)
        elif 'total_cases' in df.columns:
            new_cases = df['total_cases'].diff().fillna(0)
        else:
            raise ValueError("Не найдены данные о случаях")
    df['new_cases'] = new_cases

    if 'deaths' in df.columns:
        new_deaths = df['deaths'].fillna(0)
    elif 'new_deaths' in df.columns:
        new_deaths = df['new_deaths'].fillna(0)
    else:
        if 'total_deaths' in df.columns:
            new_deaths = df['total_deaths'].diff().fillna(0)
        else:
            new_deaths = 0
    df['new_deaths'] = new_deaths

    df['cum_cases'] = df['new_cases'].cumsum()
    df['cum_deaths'] = df['new_deaths'].cumsum()

    if 'active' in df.columns and not df['active'].isnull().all():
        pass
    else:
        if 'recovered' in df.columns:
            df['active'] = df['cum_cases'] - df['cum_deaths'] - df['recovered']
        else:
            df['active'] = df['new_cases'].rolling(window=N_days_active, min_periods=1).sum()

    df['active'] = df['active'].clip(lower=0)

    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]

    df = df.dropna(subset=['active']).reset_index(drop=True)

    population = None
    if auto_population:
        if 'population' in df.columns:
            pop_series = df['population'].dropna()
            if len(pop_series) > 0:
                population = int(pop_series.iloc[-1])
    return (df, population) if auto_population else df

def plot_calibration(active_data, I_model, dates, params, loss):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(dates, active_data, 'o', label='Реальные активные случаи', markersize=3, alpha=0.6)
    ax.plot(dates, I_model, 'r-', label='Модель SIRS', linewidth=2)
    ax.set_xlabel('Дата')
    ax.set_ylabel('Количество')
    beta, gamma, sigma, I0 = params
    ax.set_title(f'Калибровка SIRS: β={beta:.3f}, γ={gamma:.3f}, σ={sigma:.5f}, I₀={I0:.1f}, MSE={loss:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_forecast(S, I, R, Reff, dates, measures_desc, historical_dates=None, historical_I=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.plot(dates, S, 'b-', label='Восприимчивые (прогноз)')
    ax1.set_title('Динамика восприимчивых S(t)')
    ax1.grid(True)
    ax1.legend()
    ax2.plot(dates, I, 'r-', label='Заражённые (прогноз)', linewidth=2)
    if historical_dates is not None and historical_I is not None:
        ax2.plot(historical_dates, historical_I, 'o', color='gray', markersize=3, alpha=0.6, label='Реальные данные')
    ax2.set_title('Динамика заражённых I(t)')
    ax2.grid(True)
    ax2.legend()
    ax3.plot(dates, R, 'g-', label='Выздоровевшие (прогноз)')
    ax3.set_title('Динамика выздоровевших R(t)')
    ax3.grid(True)
    ax3.legend()
    ax4.plot(dates, Reff, 'm-', label='R_eff(t) (прогноз)')
    ax4.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Порог 1')
    ax4.set_title('Эффективное репродуктивное число')
    ax4.grid(True)
    ax4.legend()
    for ax in axes.flat:
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    plt.suptitle(f'Прогноз при мерах: {measures_desc}')
    return fig

class SirCalibratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SIRS Калибратор и анализ контрмер")
        self.root.geometry("900x700")

        self.df = None
        self.processed_df = None
        self.active_data = None
        self.dates = None
        self.N = tk.IntVar(value=1_000_000)
        self.country_var = tk.StringVar()
        self.start_date_var = tk.StringVar()
        self.end_date_var = tk.StringVar()
        self.forecast_days = tk.IntVar(value=100)
        self.lstm_forecast_days = tk.IntVar(value=50)
        self.optimal_params = None
        self.optimal_loss = None

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        self.tab_load = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_load, text="Загрузка/калибровка")
        self.create_load_tab()

        self.tab_scenario = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_scenario, text="Сценарии")
        self.create_scenario_tab()

        self.tab_lstm = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_lstm, text="Прогноз LSTM")
        self.create_lstm_tab()

        self.status = tk.Label(root, text="Готов к работе", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def create_load_tab(self):
        frame_top = ttk.LabelFrame(self.tab_load, text="1. Загрузка данных")
        frame_top.pack(fill='x', padx=10, pady=5)

        tk.Button(frame_top, text="Выбрать CSV-файл", command=self.load_file).grid(row=0, column=0, padx=5, pady=5)
        self.file_label = tk.Label(frame_top, text="Файл не выбран")
        self.file_label.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(frame_top, text="Страна:").grid(row=1, column=0, sticky='e', padx=5)
        tk.Entry(frame_top, textvariable=self.country_var).grid(row=1, column=1, sticky='w', padx=5)
        tk.Label(frame_top, text="Начальная дата (ГГГГ-ММ-ДД):").grid(row=2, column=0, sticky='e', padx=5)
        tk.Entry(frame_top, textvariable=self.start_date_var).grid(row=2, column=1, sticky='w', padx=5)
        tk.Label(frame_top, text="Конечная дата:").grid(row=3, column=0, sticky='e', padx=5)
        tk.Entry(frame_top, textvariable=self.end_date_var).grid(row=3, column=1, sticky='w', padx=5)
        tk.Label(frame_top, text="Общая популяция N :").grid(row=4, column=0, sticky='e', padx=5)
        tk.Entry(frame_top, textvariable=self.N).grid(row=4, column=1, sticky='w', padx=5)
        tk.Button(frame_top, text="Применить фильтры и предобработать", command=self.preprocess).grid(row=5, column=0, columnspan=2, pady=10)
        tk.Button(self.tab_load, text="2. Калибровать SIRS", command=self.calibrate, bg="lightblue").pack(pady=10)
        self.calib_frame = ttk.Frame(self.tab_load)
        self.calib_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.calib_canvas = None

    def load_file(self):
        filename = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv')])
        if filename:
            try:
                self.df = pd.read_csv(filename)
                self.file_label.config(text=os.path.basename(filename))
                self.status.config(text=f"Загружен {len(self.df)} строк")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{e}")

    def preprocess(self):
        if self.df is None:
            messagebox.showerror("Ошибка", "Сначала выберите CSV-файл!")
            return
        try:
            country = self.country_var.get() if self.country_var.get() else None
            start = self.start_date_var.get() if self.start_date_var.get() else None
            end = self.end_date_var.get() if self.end_date_var.get() else None
            self.processed_df, detected_pop = preprocess_data(self.df, country, start, end, auto_population=True)
            self.active_data = self.processed_df['active'].values
            self.dates = self.processed_df['date'].values

            if detected_pop is not None and detected_pop > 0:
                self.N.set(detected_pop)
                self.status.config(text=f"Предобработано: {len(self.processed_df)} дней. Автоматически установлена популяция {detected_pop}")
            else:
                self.status.config(text=f"Предобработано: {len(self.processed_df)} дней. Популяция не найдена, используйте ручной ввод.")

            messagebox.showinfo("Успех", f"Данные обработаны. Активных случаев: {len(self.active_data)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка предобработки:\n{e}")

    def calibrate(self):
        if self.active_data is None or len(self.active_data) == 0:
            messagebox.showerror("Ошибка", "Нет данных для калибровки. Сначала выполните предобработку.")
            return
        N = self.N.get()
        if N <= 0:
            messagebox.showerror("Ошибка", "Популяция N должна быть положительным числом.")
            return
        if np.max(self.active_data) > 0.1 * N:
            messagebox.showwarning("Предупреждение", f"Максимальное число активных случаев ({np.max(self.active_data):.0f}) превышает 10% от N ({N}). Возможно, N задана слишком маленькой. Проверьте данные.")

        self.status.config(text="Калибровка SIRS... (может занять несколько секунд)")
        self.root.update()
        try:
            opt_params, loss = Calibrator.fit(self.active_data, N, smooth=True)
            self.optimal_params = opt_params
            self.optimal_loss = loss
            beta, gamma, sigma, I0 = opt_params
            days = len(self.active_data)
            (S, I, R), t = EpidemicEngine.solve(days, N, beta, gamma, sigma, I0)
            self.last_S = S[-1]
            self.last_I = I[-1]
            self.last_R = R[-1]
            self.last_date = self.dates[-1]
            fig = plot_calibration(self.active_data, I, self.dates, (beta, gamma, sigma, I0), loss)
            if self.calib_canvas:
                self.calib_canvas.get_tk_widget().destroy()
            self.calib_canvas = FigureCanvasTkAgg(fig, self.calib_frame)
            self.calib_canvas.draw()
            self.calib_canvas.get_tk_widget().pack(fill='both', expand=True)
            self.status.config(text=f"Калибровка завершена: β={beta:.3f}, γ={gamma:.3f}, σ={sigma:.5f}, I₀={I0:.1f}, MSE={loss:.2f}")
            messagebox.showinfo("Успех", "Калибровка выполнена. Перейдите на вкладку 'Сценарии'.")
        except Exception as e:
            self.status.config(text="Ошибка калибровки")
            messagebox.showerror("Ошибка", str(e))

    def create_scenario_tab(self):
        self.measures_frame = ttk.LabelFrame(self.tab_scenario, text="Контрмеры (эффективность от 0 до 0.9)")
        self.measures_frame.pack(fill='x', padx=10, pady=5)
        self.measure_vars = {}
        measure_names = ["Маски", "Дистанцирование", "Карантин", "Вакцинация"]
        for i, name in enumerate(measure_names):
            tk.Label(self.measures_frame, text=name).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            var = tk.DoubleVar(value=0.0)
            scale = tk.Scale(self.measures_frame, from_=0.0, to=0.9, resolution=0.05, orient='horizontal', variable=var, length=300)
            scale.grid(row=i, column=1, padx=5, pady=2)
            self.measure_vars[name] = var
        tk.Label(self.tab_scenario, text="Длительность прогноза (дни):").pack(pady=5)
        tk.Entry(self.tab_scenario, textvariable=self.forecast_days).pack()
        tk.Button(self.tab_scenario, text="Построить прогноз SIRS", command=self.run_forecast, bg="lightgreen").pack(pady=10)
        self.forecast_frame = ttk.Frame(self.tab_scenario)
        self.forecast_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.forecast_canvas = None

    def run_forecast(self):
        if self.optimal_params is None:
            messagebox.showerror("Ошибка", "Сначала выполните калибровку!")
            return
        beta0, gamma, sigma, I0 = self.optimal_params
        N = self.N.get()
        days = self.forecast_days.get()
        measures = [v.get() for v in self.measure_vars.values()]
        beta_eff = beta0 * np.prod([1 - k for k in measures])
        (S, I, R), t = EpidemicEngine.solve(days, N, beta_eff, gamma, sigma, I0, R0=0)
        if self.dates is not None and len(self.dates) > 0:
            last_date = self.dates[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        else:
            forecast_dates = np.arange(days)
        Reff = beta_eff * S / (gamma * N)
        desc = ", ".join([f"{name}: {v.get()*100:.0f}%" for name, v in self.measure_vars.items()])
        hist_dates = self.dates if self.dates is not None else None
        hist_I = self.active_data if self.active_data is not None else None
        fig = plot_forecast(S, I, R, Reff, forecast_dates, desc, historical_dates=hist_dates, historical_I=hist_I)
        if self.forecast_canvas:
            self.forecast_canvas.get_tk_widget().destroy()
        self.forecast_canvas = FigureCanvasTkAgg(fig, self.forecast_frame)
        self.forecast_canvas.draw()
        self.forecast_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.status.config(text="Прогноз построен")

    def create_lstm_tab(self):
        frame = ttk.Frame(self.tab_lstm)
        frame.pack(fill='x', padx=10, pady=5)
        tk.Label(frame, text="Прогноз LSTM и SIRS на будущие дни", font=('Arial', 14)).pack(pady=5)
        tk.Label(frame, text="Длительность прогноза (дни):").pack(pady=5)
        tk.Entry(frame, textvariable=self.lstm_forecast_days, width=10).pack(pady=2)
        tk.Button(frame, text="Построить прогноз", command=self.run_lstm_forecast, bg="lightblue").pack(pady=10)
        self.lstm_frame = ttk.Frame(self.tab_lstm)
        self.lstm_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.lstm_canvas = None

    def run_lstm_forecast(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            from sklearn.preprocessing import MinMaxScaler
        except ImportError:
            messagebox.showerror("Ошибка", "TensorFlow не установлен. Установите: pip install tensorflow")
            return

        if self.active_data is None or len(self.active_data) == 0:
            messagebox.showerror("Ошибка", "Нет данных. Сначала загрузите и предобработайте данные на первой вкладке.")
            return

        data = self.active_data
        N = self.N.get()
        forecast_horizon = self.lstm_forecast_days.get()
        if forecast_horizon < 1:
            messagebox.showerror("Ошибка", "Горизонт прогноза должен быть положительным числом.")
            return

        self.status.config(text="Обучение LSTM на всех данных...")
        self.root.update()

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

        window = 14
        if len(scaled_data) <= window:
            messagebox.showerror("Ошибка", "Слишком мало данных для построения LSTM.")
            return

        X, y = [], []
        for i in range(len(scaled_data) - window):
            X.append(scaled_data[i:i+window])
            y.append(scaled_data[i+window])
        X = np.array(X).reshape((len(X), window, 1))
        y = np.array(y)

        model = Sequential([
            LSTM(50, activation='relu', input_shape=(window, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=20, verbose=0, validation_split=0.1)

        last_sequence = scaled_data[-window:].reshape(1, window, 1)
        lstm_forecast = []
        for _ in range(forecast_horizon):
            next_val_scaled = model.predict(last_sequence, verbose=0)[0, 0]
            lstm_forecast.append(next_val_scaled)
            new_seq = np.append(last_sequence[0, 1:, 0], next_val_scaled).reshape(1, window, 1)
            last_sequence = new_seq
        lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()

        if self.optimal_params is None:
            self.status.config(text="Калибровка SIRS на всех данных...")
            self.root.update()
            opt_params, loss = Calibrator.fit(data, N, smooth=True)
            self.optimal_params = opt_params
            self.optimal_loss = loss
        beta0, gamma, sigma, I0 = self.optimal_params

        (S_sirs, I_sirs, R_sirs), _ = EpidemicEngine.solve(forecast_horizon, N, beta0, gamma, sigma, I0, R0=0)

        if self.dates is not None and len(self.dates) > 0:
            last_date = self.dates[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
        else:
            forecast_dates = np.arange(forecast_horizon)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(forecast_dates, lstm_forecast, 's-', label='LSTM прогноз', markersize=4, color='blue')
        ax.plot(forecast_dates, I_sirs, '^-', label='SIRS прогноз (без контрмер)', markersize=4, color='red')
        ax.set_xlabel('Дата')
        ax.set_ylabel('Активные случаи')
        
        ax.legend()
        ax.grid(True)

        if self.lstm_canvas:
            self.lstm_canvas.get_tk_widget().destroy()
        self.lstm_canvas = FigureCanvasTkAgg(fig, self.lstm_frame)
        self.lstm_canvas.draw()
        self.lstm_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.status.config(text="Прогноз построен")

    def export_results(self):
        if self.optimal_params is None:
            messagebox.showerror("Ошибка", "Нет результатов для экспорта.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if filename:
            with open(filename, 'w') as f:
                f.write("Параметры калибровки SIRS:\n")
                f.write(f"β = {self.optimal_params[0]:.5f}\n")
                f.write(f"γ = {self.optimal_params[1]:.5f}\n")
                f.write(f"σ = {self.optimal_params[2]:.5f}\n")
                f.write(f"I₀ = {self.optimal_params[3]:.1f}\n")
                f.write(f"MSE = {self.optimal_loss:.2f}\n")
                f.write(f"Дата: {datetime.now()}\n")
            self.status.config(text=f"Результаты сохранены в {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SirCalibratorApp(root)
    tk.Button(root, text="Экспорт результатов", command=app.export_results).pack(side=tk.BOTTOM, pady=5)
    root.mainloop()
