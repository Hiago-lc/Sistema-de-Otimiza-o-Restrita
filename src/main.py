import sympy as sp

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


