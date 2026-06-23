import sympy as sp
import numpy as np

from criterios import criterio_parada_funcao_objetivo


x, y = sp.symbols('x y')


# --- Luiz Fernando ---
def metodo_barreira(funcao, restricoes, variaveis, ponto_inicial,
                     mu=10, tolerancia=1e-6, max_iter=100):
    x_sym = sp.Matrix(variaveis)
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

        sub_atual = {variaveis[i]: ponto[i] for i in range(len(variaveis))}
        sub_novo = {variaveis[i]: novo_ponto[i] for i in range(len(variaveis))}

        f_atual = float(phi.subs(sub_atual).evalf())
        f_novo = float(phi.subs(sub_novo).evalf())

        print(f"Iteração {k + 1}")
        print("Ponto atual:", ponto)
        print("Novo ponto :", novo_ponto)
        print("Erro:", erro)
        print("------------------------")

        ponto = novo_ponto

        if criterio_parada_funcao_objetivo(f_novo, f_atual, tolerancia):
            print("Parada por convergência da função objetivo (Barreira).")
            break

        if erro < tolerancia:
            break

    return ponto


# --- Givaldo Cesar ---
def metodo_gradiente(funcao, funcao_expressao, var_x, var_y, ponto_inicial, passo=1, tipo_otimizacao="minimo"):
    max_iteracoes = 10000
    tolerancia = 0.001

    diff_x = sp.diff(funcao_expressao, var_x)
    diff_y = sp.diff(funcao_expressao, var_y)

    gradiente = sp.lambdify((var_x, var_y), [diff_x, diff_y], modules=['numpy'])

    iteracao = 1
    x_atual = np.array(ponto_inicial, dtype=float)
    f_atual = funcao(x_atual)

    f_anterior = None

    while iteracao <= max_iteracoes:
        try:
            vetor_gradiente = np.array(gradiente(*x_atual), dtype=float)
        except:
            vetor_gradiente = np.array(gradiente(x_atual), dtype=float)

        norma_gradiente = np.linalg.norm(vetor_gradiente)

        if norma_gradiente < tolerancia:
            break

        if tipo_otimizacao == 'maximo':
            direcao = vetor_gradiente
        else:
            direcao = -vetor_gradiente

        passo_k = passo
        fator_reducao = 0.5
        c = 0.0001

        produto_interno = np.dot(vetor_gradiente, direcao)

        while passo_k > 1e-8:
            f_atual = funcao(x_atual)
            x_possivel = x_atual + (passo_k * direcao)
            f_possivel = funcao(x_possivel)

            if tipo_otimizacao == 'maximo':
                if f_possivel >= f_atual + (c * passo_k * produto_interno):
                    break
            else:
                if f_possivel <= f_atual + (c * passo_k * produto_interno):
                    break

            passo_k = passo_k * fator_reducao

        x_atual = x_atual + (passo_k * direcao)
        f_novo = funcao(x_atual)

        if criterio_parada_funcao_objetivo(f_novo, f_atual, tolerancia):
            print(f"Parada por convergência da função (Iteração {iteracao})")
            f_atual = f_novo
            break

        f_atual = f_novo
        iteracao += 1

    return x_atual, f_atual