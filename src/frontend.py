"""
Interface do sistema (Tkinter)
------------------------------------------------------------------
Desenvolve a interface com o usuário: entradas de dados, botões e
exibição dos resultados do Sistema de Otimização Restrita.

Liga-se a:
  - integracao.executar_otimizacao(...)  -> roda o método escolhido
  - graficos.exibir_graficos(...)        -> curvas de nível + convergência (2D)
  - graficos.exibir_grafico_3d(...)      -> superfície 3D da função objetivo

Execução:
    python frontend.py
"""

import matplotlib
matplotlib.use("TkAgg")  # garante backend compativel com Tkinter

import matplotlib.pyplot as plt
plt.ion()  # modo interativo: plt.show() nao bloqueia/congela a janela principal

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

from integracao import executar_otimizacao
from graficos import exibir_graficos, exibir_grafico_3d


# ----------------------------------------------------------------------
# Helpers puros (sem GUI) - faceis de testar isoladamente
# ----------------------------------------------------------------------
def parse_ponto_inicial(texto):
    """Converte 'x0, y0' (ou 'x0 y0') em [float, float]."""
    bruto = texto.replace(",", " ").replace(";", " ").split()
    if len(bruto) != 2:
        raise ValueError(
            "Informe exatamente dois valores para o ponto inicial. Ex.: 0.5, 0.5"
        )
    try:
        return [float(bruto[0]), float(bruto[1])]
    except ValueError:
        raise ValueError("O ponto inicial deve conter apenas numeros. Ex.: 0.5, 0.5")


def parse_float(texto, nome, minimo=None):
    """Converte um campo de texto em float, com mensagem de erro amigavel."""
    try:
        valor = float(texto.replace(",", "."))
    except ValueError:
        raise ValueError(f"O campo '{nome}' deve ser um numero.")
    if minimo is not None and valor <= minimo:
        raise ValueError(f"O campo '{nome}' deve ser maior que {minimo}.")
    return valor


def parse_inteiro(texto, nome, minimo=1):
    try:
        valor = int(float(texto))
    except ValueError:
        raise ValueError(f"O campo '{nome}' deve ser um numero inteiro.")
    if valor < minimo:
        raise ValueError(f"O campo '{nome}' deve ser >= {minimo}.")
    return valor


def formatar_resultado(resultado):
    """Monta o texto exibido na area de resultados a partir do dict retornado."""
    x_final = resultado["x_final"]
    linhas = [
        f"Metodo      : {resultado['nome']}",
        f"Status      : {resultado['status']}",
        f"Iteracoes   : {resultado['iteracoes']}",
        "",
        f"x*          : ({x_final[0]:.6f}, {x_final[1]:.6f})",
        f"f(x*)       : {resultado['f_final']:.6f}",
    ]
    return "\n".join(linhas)


# ----------------------------------------------------------------------
# Aplicacao Tkinter
# ----------------------------------------------------------------------
class AppOtimizacao(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Otimizacao Restrita")
        self.geometry("640x680")
        self.minsize(560, 600)

        # Guarda o ultimo resultado para os botoes de grafico
        self.ultimo_resultado = None
        self.ultima_funcao_alvo = None

        self._montar_widgets()
        self._atualizar_estado_campos()

    # ------------------------------------------------------------------
    # Construcao da interface
    # ------------------------------------------------------------------
    def _montar_widgets(self):
        pad = {"padx": 8, "pady": 4}

        titulo = ttk.Label(
            self, text="Otimizacao Restrita", font=("Segoe UI", 16, "bold")
        )
        titulo.pack(pady=(12, 0))
        ttk.Label(
            self, text="Metodo de Barreiras / Metodo do Gradiente"
        ).pack(pady=(0, 8))

        # ---- Bloco de entradas ----
        frm = ttk.LabelFrame(self, text="Entradas de dados")
        frm.pack(fill="x", padx=12, pady=6)
        frm.columnconfigure(1, weight=1)

        # Metodo
        ttk.Label(frm, text="Metodo:").grid(row=0, column=0, sticky="w", **pad)
        self.metodo_var = tk.StringVar(value="barreira")
        box_metodo = ttk.Frame(frm)
        box_metodo.grid(row=0, column=1, sticky="w", **pad)
        ttk.Radiobutton(
            box_metodo, text="Barreiras", value="barreira",
            variable=self.metodo_var, command=self._atualizar_estado_campos
        ).pack(side="left")
        ttk.Radiobutton(
            box_metodo, text="Gradiente", value="gradiente",
            variable=self.metodo_var, command=self._atualizar_estado_campos
        ).pack(side="left", padx=(12, 0))

        # Funcao objetivo
        ttk.Label(frm, text="Funcao objetivo f(x, y):").grid(
            row=1, column=0, sticky="w", **pad
        )
        self.entry_funcao = ttk.Entry(frm)
        self.entry_funcao.insert(0, "x**2 + y**2")
        self.entry_funcao.grid(row=1, column=1, sticky="ew", **pad)

        # Restricoes
        ttk.Label(frm, text="Restricoes g(x, y) < 0\n(uma por linha):").grid(
            row=2, column=0, sticky="nw", **pad
        )
        self.text_restricoes = tk.Text(frm, height=4, width=30)
        self.text_restricoes.insert("1.0", "x + y - 1")
        self.text_restricoes.grid(row=2, column=1, sticky="ew", **pad)

        # Ponto inicial
        ttk.Label(frm, text="Ponto inicial (x0, y0):").grid(
            row=3, column=0, sticky="w", **pad
        )
        self.entry_ponto = ttk.Entry(frm)
        self.entry_ponto.insert(0, "0.0, 0.0")
        self.entry_ponto.grid(row=3, column=1, sticky="ew", **pad)

        # ---- Bloco de parametros ----
        frm_par = ttk.LabelFrame(self, text="Parametros")
        frm_par.pack(fill="x", padx=12, pady=6)
        for c in range(4):
            frm_par.columnconfigure(c, weight=1)

        ttk.Label(frm_par, text="mu:").grid(row=0, column=0, sticky="w", **pad)
        self.entry_mu = ttk.Entry(frm_par, width=10)
        self.entry_mu.insert(0, "10")
        self.entry_mu.grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(frm_par, text="passo:").grid(row=0, column=2, sticky="w", **pad)
        self.entry_passo = ttk.Entry(frm_par, width=10)
        self.entry_passo.insert(0, "1")
        self.entry_passo.grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(frm_par, text="tolerancia:").grid(row=1, column=0, sticky="w", **pad)
        self.entry_tol = ttk.Entry(frm_par, width=10)
        self.entry_tol.insert(0, "1e-6")
        self.entry_tol.grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(frm_par, text="max. iter.:").grid(row=1, column=2, sticky="w", **pad)
        self.entry_maxiter = ttk.Entry(frm_par, width=10)
        self.entry_maxiter.insert(0, "100")
        self.entry_maxiter.grid(row=1, column=3, sticky="w", **pad)

        # ---- Botoes ----
        frm_btn = ttk.Frame(self)
        frm_btn.pack(fill="x", padx=12, pady=8)

        self.btn_otimizar = ttk.Button(
            frm_btn, text="Otimizar", command=self.on_otimizar
        )
        self.btn_otimizar.pack(side="left")

        self.btn_2d = ttk.Button(
            frm_btn, text="Graficos 2D", command=self.on_grafico_2d, state="disabled"
        )
        self.btn_2d.pack(side="left", padx=(8, 0))

        self.btn_3d = ttk.Button(
            frm_btn, text="Grafico 3D", command=self.on_grafico_3d, state="disabled"
        )
        self.btn_3d.pack(side="left", padx=(8, 0))

        ttk.Button(
            frm_btn, text="Limpar", command=self.on_limpar
        ).pack(side="right")

        # ---- Resultados ----
        frm_res = ttk.LabelFrame(self, text="Resultados")
        frm_res.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.text_resultado = scrolledtext.ScrolledText(
            frm_res, height=10, font=("Consolas", 11), state="disabled"
        )
        self.text_resultado.pack(fill="both", expand=True, padx=6, pady=6)

    # ------------------------------------------------------------------
    # Habilita/desabilita campos conforme o metodo escolhido
    # ------------------------------------------------------------------
    def _atualizar_estado_campos(self):
        eh_barreira = self.metodo_var.get() == "barreira"
        self.text_restricoes.config(state="normal" if eh_barreira else "disabled")
        self.entry_mu.config(state="normal" if eh_barreira else "disabled")
        self.entry_passo.config(state="disabled" if eh_barreira else "normal")

    # ------------------------------------------------------------------
    # Escreve na area de resultados
    # ------------------------------------------------------------------
    def _mostrar_texto(self, texto):
        self.text_resultado.config(state="normal")
        self.text_resultado.delete("1.0", "end")
        self.text_resultado.insert("1.0", texto)
        self.text_resultado.config(state="disabled")

    # ------------------------------------------------------------------
    # Callbacks dos botoes
    # ------------------------------------------------------------------
    def on_otimizar(self):
        try:
            metodo = self.metodo_var.get()
            texto_funcao = self.entry_funcao.get().strip()
            if not texto_funcao:
                raise ValueError("Informe a funcao objetivo.")

            ponto_inicial = parse_ponto_inicial(self.entry_ponto.get())
            tolerancia = parse_float(self.entry_tol.get(), "tolerancia", minimo=0)
            max_iter = parse_inteiro(self.entry_maxiter.get(), "max. iter.", minimo=1)

            if metodo == "barreira":
                texto_restricoes = self.text_restricoes.get("1.0", "end").strip()
                if not texto_restricoes:
                    raise ValueError(
                        "O Metodo de Barreiras precisa de pelo menos uma restricao."
                    )
                mu = parse_float(self.entry_mu.get(), "mu", minimo=0)
                passo = 1.0
            else:
                texto_restricoes = ""
                mu = 10.0
                passo = parse_float(self.entry_passo.get(), "passo", minimo=0)

            resultado, funcao_alvo = executar_otimizacao(
                texto_funcao=texto_funcao,
                texto_restricoes=texto_restricoes,
                ponto_inicial=ponto_inicial,
                metodo=metodo,
                mu=mu,
                passo=passo,
                tolerancia=tolerancia,
                max_iter=max_iter,
            )

            self.ultimo_resultado = resultado
            self.ultima_funcao_alvo = funcao_alvo

            self._mostrar_texto(formatar_resultado(resultado))
            self.btn_2d.config(state="normal")
            self.btn_3d.config(state="normal")

        except Exception as erro:
            messagebox.showerror("Erro", str(erro))

    def on_grafico_2d(self):
        if self.ultimo_resultado is None:
            return
        try:
            exibir_graficos([self.ultimo_resultado], self.ultima_funcao_alvo)
        except Exception as erro:
            messagebox.showerror("Erro ao gerar grafico 2D", str(erro))

    def on_grafico_3d(self):
        if self.ultimo_resultado is None:
            return
        try:
            exibir_grafico_3d([self.ultimo_resultado], self.ultima_funcao_alvo)
        except Exception as erro:
            messagebox.showerror("Erro ao gerar grafico 3D", str(erro))

    def on_limpar(self):
        self._mostrar_texto("")
        self.ultimo_resultado = None
        self.ultima_funcao_alvo = None
        self.btn_2d.config(state="disabled")
        self.btn_3d.config(state="disabled")


def main():
    app = AppOtimizacao()
    app.mainloop()


if __name__ == "__main__":
    main()
