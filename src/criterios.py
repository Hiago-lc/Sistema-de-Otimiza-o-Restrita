# --- Lucas Cintra ---

def criterio_parada_funcao_objetivo(f_atual, f_anterior, tolerancia=1e-6):
    """
    Verifica se a diferença absoluta entre o valor da função objetivo
    na iteração atual e na anterior é menor que a tolerância definida.
    """
    if f_anterior is None:
        return False

    variacao = abs(f_atual - f_anterior)
    return variacao < tolerancia