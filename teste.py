from metodosimplex import SimplexPrimal

'''
  Min  <c,x>
    suj. a: Ax  = b
            x >= 0

  A: Matriz dos coeficientes das restrições
  b: lado direito das restrições, os valores devem ser não negativos
  c: vetor de coeficientes da função objetivo
'''

print("problema com solução ótima")

A = [[4, 3.5, 2.5, 1, 0, 0],
    [0.4, 0.5, 0.1, 0, 1, 0],
    [12, 12, 6, 0, 0, 1]]
b = [1500, 70, 3000]
c = [-10, -7.5, -5]

prob = SimplexPrimal(A, b, c)
prob.resolver()

print("----------------------------")
print("problema ilimitado")

A = [[1, -2, 1, 0],
    [-1, 1, 0, 1]]
b = [4, 3]
c = [-1, -3]

prob = SimplexPrimal(A, b, c)
prob.resolver()
