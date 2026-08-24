"""EADLD 精简原生桌面界面。"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from .backend import CASE_LABELS, CASE_PRESETS, DEFAULTS, DEMO_DEFAULTS, RunManager


BLUE = "#1668E3"
GREEN = "#13A36F"
RED = "#D8495B"
INK = "#0D1C2E"
MUTED = "#718096"
LINE = "#E3E9F0"
SOFT = "#F3F6FA"
WHITE = "#FFFFFF"


class ImagePanel(tk.Frame):
    def __init__(self, master, title: str):
        super().__init__(
            master,
            bg=WHITE,
            highlightthickness=1,
            highlightbackground=LINE,
        )
        tk.Label(
            self,
            text=title,
            bg=WHITE,
            fg=INK,
            font=("Microsoft YaHei UI", 12, "bold"),
            anchor="w",
        ).pack(fill="x", padx=18, pady=13)
        tk.Frame(self, bg=LINE, height=1).pack(fill="x")
        self.stage = tk.Label(self, bg="#F8FAFC")
        self.stage.pack(fill="both", expand=True, padx=1, pady=1)
        self._photo = None
        self._key = None

    def show_path(self, path: Path | None) -> None:
        if path is None or not path.exists():
            return
        width, height = self.stage.winfo_width(), self.stage.winfo_height()
        key = (path, path.stat().st_mtime_ns, width, height)
        if key == self._key or width < 20 or height < 20:
            return
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((width - 18, height - 18), Image.Resampling.LANCZOS)
            self._photo = ImageTk.PhotoImage(image)
        self.stage.configure(image=self._photo)
        self._key = key


class EADLDDesktop:
    def __init__(self, root: tk.Tk, demo: bool = False):
        self.root = root
        self.preset = DEMO_DEFAULTS if demo else DEFAULTS
        self.manager = RunManager()
        self.variables: dict[str, tk.StringVar] = {}
        self.demo_case = tk.StringVar(value=CASE_LABELS[self.preset["demo_case"]])
        self.primary_wavelength = tk.IntVar(value=1)
        self.last_metric_size = -1
        self._spec_traces_ready = False
        self._configure_window()
        self._build()
        self.apply_preset()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.poll()

    def _configure_window(self) -> None:
        self.root.title("EADLD")
        self.root.geometry("1480x900")
        self.root.minsize(1220, 760)
        self.root.configure(bg=SOFT)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "EADLD.Horizontal.TProgressbar",
            troughcolor="#DCE4ED",
            background=BLUE,
            bordercolor="#DCE4ED",
            lightcolor=BLUE,
            darkcolor=BLUE,
            thickness=4,
        )

    def _build(self) -> None:
        shell = tk.Frame(self.root, bg=SOFT)
        shell.pack(fill="both", expand=True)
        shell.grid_columnconfigure(1, weight=1)
        shell.grid_rowconfigure(0, weight=1)
        self._build_specs(shell)
        self._build_results(shell)

    def _build_specs(self, parent) -> None:
        side = tk.Frame(
            parent,
            width=365,
            bg=WHITE,
            highlightthickness=1,
            highlightbackground=LINE,
        )
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_propagate(False)
        body = tk.Frame(side, bg=WHITE)
        body.pack(fill="both", expand=True, padx=22, pady=20)
        tk.Label(
            body,
            text="系统规格",
            bg=WHITE,
            fg=INK,
            font=("Microsoft YaHei UI", 18, "bold"),
        ).pack(anchor="w", pady=(0, 14))

        tk.Label(
            body,
            text="演示方案",
            bg=WHITE,
            fg="#526176",
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="w")
        self.case_selector = ttk.Combobox(
            body,
            textvariable=self.demo_case,
            values=list(CASE_LABELS.values()),
            state="readonly",
            font=("Microsoft YaHei UI", 9),
        )
        self.case_selector.pack(fill="x", ipady=3, pady=(3, 8))
        self.case_selector.bind("<<ComboboxSelected>>", self.apply_case_selection)

        specs = tk.Frame(body, bg=WHITE)
        specs.pack(fill="x")
        specs.grid_columnconfigure((0, 1), weight=1, uniform="spec")
        self._grid_entry(specs, "target_efl", "焦距 / mm", 0, 0)
        self._grid_entry(specs, "f_number", "F 数", 0, 1)
        self._grid_entry(specs, "entrance_pupil", "入瞳直径 / mm", 1, 0, True)
        self._grid_entry(specs, "half_field", "半视场 / °", 1, 1)
        self._grid_entry(specs, "n_fields", "视场数", 2, 0)
        self._grid_entry(specs, "distortion_percent", "畸变上限 / %", 2, 1)
        self._grid_entry(specs, "zone_count", "最优环带数", 3, 0, True)

        tk.Label(
            body,
            text="波长",
            bg=WHITE,
            fg=INK,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(anchor="w", pady=(14, 6))
        wavelength_table = tk.Frame(body, bg=WHITE)
        wavelength_table.pack(fill="x")
        wavelength_table.grid_columnconfigure(1, weight=3)
        wavelength_table.grid_columnconfigure(2, weight=2)
        for column, text in enumerate(("#", "波长 / nm", "权重", "主")):
            tk.Label(
                wavelength_table,
                text=text,
                bg=WHITE,
                fg=MUTED,
                font=("Microsoft YaHei UI", 8),
            ).grid(row=0, column=column, sticky="ew", pady=(0, 3))
        for index in range(3):
            tk.Label(
                wavelength_table,
                text=str(index + 1),
                bg=WHITE,
                fg="#526176",
                font=("Segoe UI", 9),
            ).grid(row=index + 1, column=0, padx=(0, 7))
            self._table_entry(wavelength_table, f"wave{index}", index + 1, 1)
            self._table_entry(wavelength_table, f"weight{index}", index + 1, 2)
            tk.Radiobutton(
                wavelength_table,
                variable=self.primary_wavelength,
                value=index,
                bg=WHITE,
                activebackground=WHITE,
                selectcolor=WHITE,
                highlightthickness=0,
            ).grid(row=index + 1, column=3, padx=(7, 0))

        tk.Label(
            body,
            text="求解参数",
            bg=WHITE,
            fg=INK,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(anchor="w", pady=(14, 5))
        solver = tk.Frame(body, bg=WHITE)
        solver.pack(fill="x")
        solver.grid_columnconfigure((0, 1), weight=1, uniform="solver")
        self._grid_entry(solver, "steps", "迭代步数", 0, 0)
        self._grid_entry(solver, "visual_every", "光路刷新", 0, 1)
        self._grid_entry(solver, "n_r", "径向采样", 1, 0)
        self._grid_entry(solver, "n_theta", "角向采样", 1, 1)
        self._grid_entry(solver, "lm_parameter", "LM 阻尼", 2, 0)

        self.start_button = self._button(
            body, "开始优化", self.start_run, BLUE, WHITE, 2
        )
        self.start_button.pack(fill="x", pady=(14, 0))
        self.stop_button = self._button(
            body, "停止", self.stop_run, "#FFF1F3", RED
        )
        self.stop_button.pack(fill="x", pady=(7, 0))
        self.stop_button.configure(state="disabled")
        self.progress = ttk.Progressbar(
            body,
            style="EADLD.Horizontal.TProgressbar",
            maximum=100,
        )
        self.progress.pack(fill="x", pady=(16, 7))
        self.status_label = tk.Label(
            body,
            text="",
            bg=WHITE,
            fg=MUTED,
            font=("Microsoft YaHei UI", 8),
        )
        self.status_label.pack(anchor="w")
        self.error_label = tk.Label(
            body,
            text="",
            bg=WHITE,
            fg=RED,
            justify="left",
            wraplength=235,
            font=("Microsoft YaHei UI", 8),
        )
        self.error_label.pack(fill="x", pady=(8, 0))

    def _build_results(self, parent) -> None:
        content = tk.Frame(parent, bg=SOFT)
        content.grid(row=0, column=1, sticky="nsew", padx=18, pady=18)
        content.grid_columnconfigure(0, weight=1, uniform="result")
        content.grid_columnconfigure(1, weight=1, uniform="result")
        content.grid_rowconfigure(0, weight=2)
        content.grid_rowconfigure(1, weight=3)

        self._build_loss(content).grid(
            row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6)
        )
        self.layout_panel = ImagePanel(content, "光路图")
        self.layout_panel.grid(
            row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6)
        )
        self.spot_panel = ImagePanel(content, "点列图")
        self.spot_panel.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(0, 6),
            pady=(6, 0),
        )
        self.mtf_panel = ImagePanel(content, "MTF")
        self.mtf_panel.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=(6, 0),
            pady=(6, 0),
        )

    def _build_loss(self, parent) -> tk.Frame:
        panel = tk.Frame(
            parent,
            bg=WHITE,
            highlightthickness=1,
            highlightbackground=LINE,
        )
        header = tk.Frame(panel, bg=WHITE)
        header.pack(fill="x", padx=18, pady=13)
        tk.Label(
            header,
            text="损失函数",
            bg=WHITE,
            fg=INK,
            font=("Microsoft YaHei UI", 12, "bold"),
        ).pack(side="left")
        self.loss_value = tk.Label(
            header,
            text="—",
            bg=WHITE,
            fg=BLUE,
            font=("Segoe UI", 11, "bold"),
        )
        self.loss_value.pack(side="right")
        tk.Frame(panel, bg=LINE, height=1).pack(fill="x")
        self.figure = Figure(figsize=(4.4, 2.3), dpi=100, facecolor=WHITE)
        self.axes = self.figure.add_subplot(111)
        self.chart = FigureCanvasTkAgg(self.figure, panel)
        self.chart.get_tk_widget().configure(bg=WHITE, highlightthickness=0)
        self.chart.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=6)
        self._draw_chart([])
        return panel

    def _grid_entry(
        self, parent, name: str, label: str, row: int, column: int, readonly=False
    ) -> None:
        cell = tk.Frame(parent, bg=WHITE)
        cell.grid(row=row, column=column, sticky="ew", padx=(0, 6) if column == 0 else (6, 0), pady=3)
        tk.Label(
            cell,
            text=label,
            bg=WHITE,
            fg="#526176",
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="w")
        variable = self.variables.setdefault(name, tk.StringVar())
        entry = tk.Entry(
            cell,
            textvariable=variable,
            bg="#F8FAFC" if not readonly else "#EEF2F6",
            readonlybackground="#EEF2F6",
            fg=INK,
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground="#DCE4ED",
            highlightcolor=BLUE,
            font=("Segoe UI", 9),
        )
        entry.pack(fill="x", ipady=4, pady=(3, 0))
        if readonly:
            entry.configure(state="readonly")

    def _table_entry(self, parent, name: str, row: int, column: int) -> None:
        variable = self.variables.setdefault(name, tk.StringVar())
        tk.Entry(
            parent,
            textvariable=variable,
            bg="#F8FAFC",
            fg=INK,
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground="#DCE4ED",
            highlightcolor=BLUE,
            justify="center",
            font=("Segoe UI", 9),
        ).grid(row=row, column=column, sticky="ew", padx=3, pady=2, ipady=3)

    @staticmethod
    def _button(parent, text, command, bg, fg, height=1):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Microsoft YaHei UI", 9, "bold"),
            height=height,
        )

    def apply_preset(self) -> None:
        for name in (
            "target_efl",
            "f_number",
            "half_field",
            "n_fields",
            "distortion_percent",
            "steps",
            "visual_every",
            "n_r",
            "n_theta",
            "lm_parameter",
            "zone_count",
        ):
            self.variables[name].set(str(self.preset[name]))
        for index, value in enumerate(self.preset["wavelengths"]):
            self.variables[f"wave{index}"].set(str(value))
        for index, value in enumerate(self.preset["wavelength_weights"]):
            self.variables[f"weight{index}"].set(str(value))
        self.primary_wavelength.set(self.preset["primary_wavelength"])
        if not self._spec_traces_ready:
            self.variables["target_efl"].trace_add(
                "write", self._update_entrance_pupil
            )
            self.variables["f_number"].trace_add(
                "write", self._update_entrance_pupil
            )
            self._spec_traces_ready = True
        self._update_entrance_pupil()

    def apply_case_selection(self, _event=None) -> None:
        selected = self.demo_case.get()
        key = next(key for key, label in CASE_LABELS.items() if label == selected)
        self.preset = CASE_PRESETS[key]
        self.apply_preset()

    def _update_entrance_pupil(self, *_args) -> None:
        try:
            diameter = float(self.variables["target_efl"].get()) / float(
                self.variables["f_number"].get()
            )
            value = f"{diameter:.3f}"
        except (ValueError, ZeroDivisionError):
            value = "—"
        self.variables["entrance_pupil"].set(value)

    def read_inputs(self) -> dict:
        return self.preset | {
            "demo_case": next(
                key
                for key, label in CASE_LABELS.items()
                if label == self.demo_case.get()
            ),
            "target_efl": self.variables["target_efl"].get(),
            "f_number": self.variables["f_number"].get(),
            "half_field": self.variables["half_field"].get(),
            "n_fields": self.variables["n_fields"].get(),
            "distortion_percent": self.variables["distortion_percent"].get(),
            "steps": self.variables["steps"].get(),
            "visual_every": self.variables["visual_every"].get(),
            "n_r": self.variables["n_r"].get(),
            "n_theta": self.variables["n_theta"].get(),
            "lm_parameter": self.variables["lm_parameter"].get(),
            "wavelengths": [
                self.variables[f"wave{index}"].get() for index in range(3)
            ],
            "wavelength_weights": [
                self.variables[f"weight{index}"].get() for index in range(3)
            ],
            "primary_wavelength": self.primary_wavelength.get(),
        }

    def start_run(self) -> None:
        self.error_label.configure(text="")
        try:
            self.manager.start(self.read_inputs())
        except (ValueError, RuntimeError, OSError) as exc:
            self.error_label.configure(text=str(exc))

    def stop_run(self) -> None:
        self.manager.stop()

    def poll(self) -> None:
        try:
            self.render(self.manager.snapshot())
        except (OSError, RuntimeError) as exc:
            self.error_label.configure(text=str(exc))
        self.root.after(1000, self.poll)

    def render(self, status: dict) -> None:
        running = status["state"] in {"starting", "running"}
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        maximum = status["max_steps"] or 1
        self.progress.configure(value=min(100, status["step"] / maximum * 100))
        state = {
            "idle": "",
            "starting": "准备中",
            "running": "优化中",
            "completed": "完成",
            "failed": "失败",
            "stopped": "已停止",
        }.get(status["state"], status["state"])
        suffix = (
            f"  {status['step']} / {status['max_steps']}" if status["max_steps"] else ""
        )
        self.status_label.configure(
            text=state + suffix,
            fg=GREEN if status["state"] == "completed" else RED if status["state"] == "failed" else MUTED,
        )
        series = status["metrics"].get("loss", [])
        self.loss_value.configure(
            text="—" if not series else self._compact(series[-1]["value"])
        )
        if len(series) != self.last_metric_size:
            self._draw_chart(series)
            self.last_metric_size = len(series)
        self.layout_panel.show_path(status["artifacts"].get("layout"))
        self.spot_panel.show_path(status["artifacts"].get("spot_diagrams"))
        self.mtf_panel.show_path(status["artifacts"].get("mtf"))

    def _draw_chart(self, series: list[dict]) -> None:
        self.axes.clear()
        self.axes.set_facecolor(WHITE)
        self.axes.grid(True, color="#E8EDF3", linewidth=0.7)
        self.axes.tick_params(colors="#7E8DA0", labelsize=7, length=0)
        for spine in self.axes.spines.values():
            spine.set_visible(False)
        if series:
            steps = [point["step"] for point in series]
            values = [max(point["value"], 1e-20) for point in series]
            self.axes.semilogy(
                steps,
                values,
                color=BLUE,
                linewidth=2.4,
            )
            floor = min(values) * 0.82
            self.axes.fill_between(steps, values, floor, color=BLUE, alpha=0.08)
            self.axes.scatter(
                steps[-1:], values[-1:], s=28, color=BLUE, edgecolor=WHITE, zorder=3
            )
        else:
            self.axes.set_xticks([])
            self.axes.set_yticks([])
        self.figure.tight_layout(pad=1.0)
        self.chart.draw_idle()

    @staticmethod
    def _compact(value: float) -> str:
        absolute = abs(value)
        return f"{value:.2e}" if (0 < absolute < 1e-3) or absolute >= 1e5 else f"{value:.5f}"

    def on_close(self) -> None:
        if self.manager.snapshot()["state"] in {"starting", "running"}:
            if not messagebox.askyesno("EADLD", "停止优化并退出？"):
                return
        self.manager.stop()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="EADLD desktop")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--autorun", action="store_true")
    args = parser.parse_args()
    root = tk.Tk()
    desktop = EADLDDesktop(root, demo=args.demo)
    if args.autorun:
        root.after(800, desktop.start_run)
    root.mainloop()


if __name__ == "__main__":
    main()
