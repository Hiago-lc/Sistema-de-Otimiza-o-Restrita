import sympy as sp
import numpy as np

from criterios import criterio_parada_funcao_objetivo


x, y = sp.symbols("x y")


def converter_funcao_texto(texto_funcao):
    funcao_expr = sp.sympify(texto_funcao)
    return funcao_expr


def converter_restricoes_texto(texto_restricoes):
    restricoes = []

    for linha in texto_restricoes.splitlines():
        linha = linha.strip()

        if linha != "":
            restricoes.append(sp.sympify(linha))

    return restricoes


def criar_funcao_numerica(funcao_expr):
    funcao_lambdify = sp.lambdify((x, y), funcao_expr, modules=["numpy"])

    def funcao_numpy(v):
        x1, x2 = v
        return funcao_lambdify(x1, x2)

    return funcao_numpy


def verificar_ponto_viavel(restricoes, ponto_inicial):
    substituicao = {
        x: ponto_inicial[0],
        y: ponto_inicial[1]
    }

    for g in restricoes:
        valor = float(g.subs(substituicao).evalf())

        if valor >= 0:
            return False

    return True


def executar_barreira_com_historico(funcao_expr, restricoes, ponto_inicial,
                                    mu=10, tolerancia=1e-6, max_iter=100):
    variaveis = [x, y]

    barreira = 0

    for g in restricoes:
        barreira += -sp.log(-g)

    phi = funcao_expr + (1 / mu) * barreira

    gradiente = sp.Matrix([sp.diff(phi, v) for v in variaveis])
    hessiana = sp.hessian(phi, variaveis)

    ponto = sp.Matrix(ponto_inicial)

    funcao_original_num = sp.lambdify((x, y), funcao_expr, modules=["numpy"])

    hist_x = []
    hist_f = []

    hist_x.append(np.array([float(ponto[0]), float(ponto[1])]))
    hist_f.append(float(funcao_original_num(float(ponto[0]), float(ponto[1]))))

    status = "Máximo de iterações atingido"

    for k in range(max_iter):
        substituicao = {
            variaveis[i]: ponto[i]
            for i in range(len(variaveis))
        }

        grad_val = gradiente.subs(substituicao).evalf()
        hess_val = hessiana.subs(substituicao).evalf()

        try:
            direcao = -hess_val.inv() * grad_val
        except:
            status = "Hessiana singular"
            break

        novo_ponto = ponto + direcao
        erro = float(direcao.norm())

        sub_atual = {
            variaveis[i]: ponto[i]
            for i in range(len(variaveis))
        }

        sub_novo = {
            variaveis[i]: novo_ponto[i]
            for i in range(len(variaveis))
        }

        f_atual_phi = float(phi.subs(sub_atual).evalf())
        f_novo_phi = float(phi.subs(sub_novo).evalf())

        ponto = novo_ponto

        x1 = float(ponto[0])
        x2 = float(ponto[1])

        hist_x.append(np.array([x1, x2]))
        hist_f.append(float(funcao_original_num(x1, x2)))

        if criterio_parada_funcao_objetivo(f_novo_phi, f_atual_phi, tolerancia):
            status = "Convergiu pela função objetivo"
            break

        if erro < tolerancia:
            status = "Convergiu pela norma da direção"
            break

    resultado = {
        "nome": "Método de Barreiras",
        "hist_x": np.array(hist_x),
        "hist_f": np.array(hist_f),
        "x_final": np.array(hist_x[-1]),
        "f_final": hist_f[-1],
        "iteracoes": len(hist_x) - 1,
        "status": status
    }

    return resultado


def executar_gradiente_com_historico(funcao_expr, ponto_inicial,
                                     passo=1, tolerancia=0.001,
                                     max_iter=10000, tipo_otimizacao="minimo"):
    funcao_num = sp.lambdify((x, y), funcao_expr, modules=["numpy"])

    diff_x = sp.diff(funcao_expr, x)
    diff_y = sp.diff(funcao_expr, y)

    gradiente = sp.lambdify((x, y), [diff_x, diff_y], modules=["numpy"])

    ponto = np.array(ponto_inicial, dtype=float)

    hist_x = [ponto.copy()]
    hist_f = [float(funcao_num(ponto[0], ponto[1]))]

    status = "Máximo de iterações atingido"

    for k in range(max_iter):
        vetor_gradiente = np.array(gradiente(ponto[0], ponto[1]), dtype=float)
        norma_gradiente = np.linalg.norm(vetor_gradiente)

        if norma_gradiente < tolerancia:
            status = "Convergiu pela norma do gradiente"
            break

        if tipo_otimizacao == "maximo":
            direcao = vetor_gradiente
        else:
            direcao = -vetor_gradiente

        passo_k = passo
        fator_reducao = 0.5
        c = 0.0001

        produto_interno = np.dot(vetor_gradiente, direcao)

        f_atual = float(funcao_num(ponto[0], ponto[1]))

        while passo_k > 1e-8:
            ponto_teste = ponto + passo_k * direcao
            f_teste = float(funcao_num(ponto_teste[0], ponto_teste[1]))

            if tipo_otimizacao == "maximo":
                if f_teste >= f_atual + c * passo_k * produto_interno:
                    break
            else:
                if f_teste <= f_atual + c * passo_k * produto_interno:
                    break

            passo_k = passo_k * fator_reducao

        novo_ponto = ponto + passo_k * direcao
        f_novo = float(funcao_num(novo_ponto[0], novo_ponto[1]))

        ponto = novo_ponto

        hist_x.append(ponto.copy())
        hist_f.append(f_novo)

        if criterio_parada_funcao_objetivo(f_novo, f_atual, tolerancia):
            status = "Convergiu pela função objetivo"
            break

    resultado = {
        "nome": "Método do Gradiente",
        "hist_x": np.array(hist_x),
        "hist_f": np.array(hist_f),
        "x_final": np.array(hist_x[-1]),
        "f_final": hist_f[-1],
        "iteracoes": len(hist_x) - 1,
        "status": status
    }

    return resultado


def executar_otimizacao(texto_funcao, texto_restricoes, ponto_inicial,
                        metodo="barreira", mu=10, passo=1,
                        tolerancia=1e-6, max_iter=100):
    funcao_expr = converter_funcao_texto(texto_funcao)
    restricoes = converter_restricoes_texto(texto_restricoes)
    funcao_alvo = criar_funcao_numerica(funcao_expr)

    if metodo == "barreira":
        if len(restricoes) == 0:
            raise ValueError("O Método de Barreiras precisa de pelo menos uma restrição.")

        if not verificar_ponto_viavel(restricoes, ponto_inicial):
            raise ValueError("O ponto inicial não pertence à região viável. Use um ponto que satisfaça g(x,y) < 0.")

        resultado = executar_barreira_com_historico(
            funcao_expr=funcao_expr,
            restricoes=restricoes,
            ponto_inicial=ponto_inicial,
            mu=mu,
            tolerancia=tolerancia,
            max_iter=max_iter
        )

    elif metodo == "gradiente":
        resultado = executar_gradiente_com_historico(
            funcao_expr=funcao_expr,
            ponto_inicial=ponto_inicial,
            passo=passo,
            tolerancia=tolerancia,
            max_iter=max_iter,
            tipo_otimizacao="minimo"
        )

    else:
        raise ValueError("Método inválido.")

    return resultado, funcao_alvo