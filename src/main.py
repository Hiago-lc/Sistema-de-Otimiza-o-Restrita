
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
x, y = sp.symbols('x y')
#Luiz Fernando 
def metodo_barreira(funcao, restricoes, variaveis, ponto_inicial,
                     mu=10, tolerancia=1e-6, max_iter=100):
    x = sp.Matrix(variaveis)
    barreira = 0
    for g in restricoes:
        barreira += -sp.log(-g)
    phi = funcao + (1 / mu) * barreira
    gradiente = sp.Matrix([sp.diff(phi, v) for v in variaveis])
    hessiana = sp.hessian(phi, variaveis)
    ponto = sp.Matrix(ponto_inicial)
    for k in range(max_iter):
        substituicao = {}
        for i in range(len(variaveis)):
            substituicao[variaveis[i]] = ponto[i]
        grad_val = gradiente.subs(substituicao).evalf()
        hess_val = hessiana.subs(substituicao).evalf()
        try:
            direcao = -hess_val.inv() * grad_val
        except:
            print("Hessiana singular")
            return ponto
        novo_ponto = ponto + direcao
        erro = float(direcao.norm())
        print(f"Iteração {k + 1}")
        print("Ponto atual:", ponto)
        print("Novo ponto :", novo_ponto)
        print("Erro:", erro)
        print("------------------------")
        ponto = novo_ponto
        if erro < tolerancia:
            break
    return ponto
#Gustavo Barcelos
def exibir_graficos(lista_resultados, funcao_alvo):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # 1. Descobrir os limites do gráfico juntando as coordenadas de todos os métodos
    x1_todos = []
    x2_todos = []
    for res in lista_resultados:
        x1_todos.extend(res["hist_x"][:, 0])
        x2_todos.extend(res["hist_x"][:, 1])
    # ZOOM AUTOMÁTICO DE 15% (Substitui a margem cravada de 1.0)
    margem_x1 = max((max(x1_todos) - min(x1_todos)) * 0.15, 0.2)
    margem_x2 = max((max(x2_todos) - min(x2_todos)) * 0.15, 0.2)
    x1_min, x1_max = min(x1_todos) - margem_x1, max(x1_todos) + margem_x1
    x2_min, x2_max = min(x2_todos) - margem_x2, max(x2_todos) + margem_x2
    X1_grid, X2_grid = np.meshgrid(
        np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100)
    )
    Z_grid = funcao_alvo([X1_grid, X2_grid])
    # Desenha o relevo de fundo
    ax1.contour(X1_grid, X2_grid, Z_grid, levels=40, cmap="viridis")
    # Cores para cada método
    cores = ["red", "blue", "green", "orange", "purple"]
    # 2. Desenha a linha de cada método no gráfico
    for i, res in enumerate(lista_resultados):
        cor = cores[i % len(cores)]
        nome = res["nome"]
        hist_x = res["hist_x"]
        hist_f = res["hist_f"]
        # Curva de Deslocamento (x1 vs x2)
        ax1.plot(hist_x[:, 0], hist_x[:, 1], marker="o", color=cor, linestyle="-", label=nome, markersize=4)
        # Marcação do ponto inicial e final
        if i == 0:
            ax1.plot(hist_x[0, 0], hist_x[0, 1], marker="s", color="black", markersize=8, label="Início")
        ax1.plot(hist_x[-1, 0], hist_x[-1, 1], marker="*", color=cor, markersize=10)
        # Curva de Convergência (Iteração vs f(x))
        ax2.plot(range(len(hist_f)), hist_f, marker="o", color=cor, linestyle="-", label=nome, markersize=4)
    ax1.set_title("Curvas de Nível e Deslocamento")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax2.set_title("Convergência da Função Objetivo")
    ax2.set_xlabel("Iterações")
    ax2.set_ylabel("f(x, y)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
#Guilherme Versiani Araújo
def exibir_grafico_3d(lista_resultados, funcao_alvo):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Reutiliza o mesmo zoom automático do gráfico 2D 
    x1_todos = []
    x2_todos = []
    for res in lista_resultados:
        x1_todos.extend(res["hist_x"][:, 0])
        x2_todos.extend(res["hist_x"][:, 1])
    margem_x1 = max((max(x1_todos) - min(x1_todos)) * 0.15, 0.2)
    margem_x2 = max((max(x2_todos) - min(x2_todos)) * 0.15, 0.2)
    x1_min, x1_max = min(x1_todos) - margem_x1, max(x1_todos) + margem_x1
    x2_min, x2_max = min(x2_todos) - margem_x2, max(x2_todos) + margem_x2
    X1_grid, X2_grid = np.meshgrid(
        np.linspace(x1_min, x1_max, 80),
        np.linspace(x2_min, x2_max, 80)
    )
    Z_grid = funcao_alvo([X1_grid, X2_grid])
    # Superfície 3D semi-transparente com barra de cores
    surf = ax.plot_surface(X1_grid, X2_grid, Z_grid,
                           cmap='viridis', alpha=0.6, edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='f(x₁, x₂)')
    # Projeção das curvas de nível na base do gráfico (tipo "sombra")
    z_base = float(np.min(Z_grid)) - (float(np.max(Z_grid)) - float(np.min(Z_grid))) * 0.1
    ax.contourf(X1_grid, X2_grid, Z_grid,
                levels=15, cmap='viridis',
                offset=z_base, alpha=0.3, zdir='z')
    # Cores para cada método (igual ao gráfico 2D)
    cores = ["red", "blue", "green", "orange", "purple"]
    for i, res in enumerate(lista_resultados):
        cor = cores[i % len(cores)]
        nome = res["nome"]
        hist_x = res["hist_x"]
        hist_f = res["hist_f"]
        # Linha do trajeto percorrido sobre a superfície
        ax.plot(hist_x[:, 0], hist_x[:, 1], hist_f,
                color=cor, linestyle="-", linewidth=2, label=nome, zorder=5)
        # Pontos individuais ao longo do trajeto
        ax.scatter(hist_x[:, 0], hist_x[:, 1], hist_f,
                   color=cor, s=20, zorder=5)
        # Ponto inicial (apenas para o primeiro método, igual ao 2D)
        if i == 0:
            ax.scatter(hist_x[0, 0], hist_x[0, 1], hist_f[0],
                       color="black", marker="s", s=100,
                       label="Início", zorder=6)
        # Ponto final (ótimo encontrado) — estrela destacada
        ax.scatter(hist_x[-1, 0], hist_x[-1, 1], hist_f[-1],
                   color=cor, marker="*", s=250, zorder=6)
    ax.set_title("Superfície 3D da Função Objetivo")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("f(x₁, x₂)")
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()
