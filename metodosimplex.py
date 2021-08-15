# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(precision=3)

class SimplexPrimal():
    '''
SimplexPrimal
=============
Implementação do método simplex de duas fases para resolver o problema de 
programação linear na forma padrão
::
    min  <c,x>
    s.a. Ax  = b
          x >= 0

Parâmetros
----------
A : m x n array
    Matriz das restrições lineares, com as variáveis de folga já inclusas.
b : m array
    Lado direito das restrições lineares, que deve necessariamente ser
    não-negativo.
c : p array
    Vetor de coeficientes da função objetivo. Se p <= n, o algoritmo completa c com zeros.
    '''
    def __init__(self,A,b,c, output='screen,minimal'):
        self.A = np.array(A)
        self.b = np.array(b)
        self.c = np.array(c)
        self.output = output
        self.check()

    def check(self):
        zeros = np.zeros_like(self.b)
        posicoes, = np.where(self.b < zeros)
        if len(posicoes) > 0:
            raise ValueError("O vetor b deve ser maior que zero.")


    def resolver(self):
        self.preparar_output('abrir')
        self._output('','definirproblema')
        (m,n) = self.A.shape
        self.base = m*[False]
        for j in range(m):
            ej = basecanonica(j,m)
            ej = ej.reshape((m,1))
            vetortruefalse = np.all(self.A==ej,axis=0)
            posicao, = np.where(vetortruefalse)
            if len(posicao) > 0:
                coluna = posicao[0]
                self.base[j] = coluna
        if False in self.base:
            msg = "Problema não tem base, vamos para a fase 1."
            self._output(msg,'[m]resolver')
            try:
                self.fase1()
                self.fase2()
                self._output("Solução x =",'[m]solucao')
                self._output(self.solucao(),'[m]solucao')
            except Exception as e:
                self._output(str(e),'[m]error')
                self.preparar_output('fechar')
        else:
            msg = "Problema tem base, vamos começar a pivotar."
            self._output(msg,'[m]resolver')
            try:
                self.jatembase()
                self._output("Solução x =",'[m]solucao')
                self._output(self.solucao(),'[m]solucao')
            except Exception as e:
                self._output(str(e),'[m]error')
                self.preparar_output('fechar')
        try:
            self.preparar_output('fechar')
        except:
            pass

    def jatembase(self):
        (m,n) = self.A.shape
        self.tableau = np.zeros((m+1,n+1))
        self.tableau[0:m,0:n] = self.A
        self.tableau[0:m,n] = self.b
        self.tableau[m,0:len(self.c)] = -self.c
        self._output('Tableau','[m]tableau')
        self._output(self.tableau,'[m]jatembase')
        self.run()

    def fase1(self):
        (m,n) = self.A.shape
        a = self.base.count(False)
        self.artificiais = []
        self.tableau = np.zeros((m+1,n+a+1))
        self.tableau[0:m,0:n] = self.A
        self.tableau[0:m,n+a] = self.b
        for j in range(m):
            if self.base[j] == False:
                self.tableau[0:m,n+j] = basecanonica(j,m)
                self.artificiais.append(n+j)
                self.base[j] = n+j
        self._output('Iniciando a fase 1.','[m]fase1')
        self.tableau[m,n:n+a] = -np.ones(a)
        self._output(self.tableau,'[m]custorelarti')
        self._output("Pivotando nas variáveis artificiais.",'[m]inipivotart')
        for j in self.artificiais:
            i = self.base.index(j)
            self.tableau[m,:] = self.tableau[m,:] + self.tableau[i,:]
            self._output(self.tableau,'pivotart')
        self._output(self.tableau,'[m]fimpivotart')
        self.run()
        self.deletar = []
        for k in self.base:
            if k in self.artificiais:
                i = self.base.index(k)
                if self.tableau[i,-1] != 0:
                    raise SemSolucoesViaveis()
                else:
                    self.retirar_artificial_da_base(k)
        if len(self.deletar)>0:
            self.tableau = np.delete(self.tableau, self.deletar,axis=0)

    def fase2(self):
        self._output('Iniciando a fase 2.','[m]fase2')
        (M,N) = self.tableau.shape
        m = M - 1
        self.tableau = np.delete(self.tableau, self.artificiais,axis=1)
        self.tableau[m,0:len(self.c)] = -self.c
        self._output('Vetor de custo relativo adicionado.','[m]fase2vetor')
        self._output(self.tableau,'[m]fase2vetor')
        for j in self.base:
            i = self.tableau[0:m,j].argmax()
            self.tableau[m,:] = self.tableau[m,:] - self.tableau[m,j]*self.tableau[i,:]
            self._output(self.tableau,'zerarcustrel')
        self._output('Custo relativo dos elementos da base zerados.','zerarcustrel')
        self._output(self.tableau,'[m]custorelativozerado')
        self.run()

    def run(self):
        (M,N) = self.tableau.shape
        maximo = self.tableau[M-1,0:N-1].max()
        while  maximo > 0:
            quemEntra = self.tableau[M-1,0:N-1].argmax()
            quemSai = self.quem_sai_da_base(quemEntra)
            self.base[quemSai] = quemEntra
            msg = "Pivotando na linha {} e coluna {}."
            self._output(msg.format(quemSai+1, quemEntra+1),'[m]entraesai')
            self.pivotar(quemSai,quemEntra)
            self._output(self.tableau,'[m]run')
            maximo = self.tableau[M-1,0:N-1].max()

    def quem_sai_da_base(self,k):
        (M,N) = self.tableau.shape
        j = -1
        I = [i for i in range(M-1) if self.tableau[i,k] > 0]
        if len(I) == 0:
            raise ProblemaIlimitado()
        razoes = np.array([self.tableau[i,j]/self.tableau[i,k] for i in I])
        razaominima = razoes.min()
        I = [I[i] for i in range(razoes.size) if razoes[i] == razaominima]
        while len(I) > 1:
            j = j + 1 
            razoes = np.array([self.tableau[i,j]/self.tableau[i,k] for i in I])
            razaominima = razoes.min()
            I = [I[i] for i in range(razoes.size) if razoes[i] == razaominima]
            if j == N-1:
                raise LinhasLD(I)
        return I[0]

    def pivotar(self,i,j):
        (M,N) = self.tableau.shape
        self.tableau[i,:] = self.tableau[i,:]/self.tableau[i,j]
        self._output(self.tableau,'linhaipelopivo')
        for k in range(M):
            if k != i:
                self.tableau[k,:] = self.tableau[k,:] - self.tableau[k,j]*self.tableau[i,:]
                self._output(self.tableau, 'pivotando')

    def retirar_artificial_da_base(self,coluna):
        linha = self.base.index(coluna)
        (m,n) = self.A.shape
        naobasicas = [x for x in range(n) if x not in self.base]
        colunas = [j for j in naobasicas if self.tableau[linha,j] > 0]
        if len(colunas) > 0:
            self.base[linha] = colunas[0]
            msg = "Retirando a variável artifical {} da base."
            self._output(msg.format(coluna+1),'[m]retbase')
            self.pivotar(linha,colunas[0])
            self._output(self.tableau,'[m]retbase')
        else:
            msg = "Não é possível remover a variável artifical {} da base, "
            msg += "teremos que deletar a linha correspondente."
            self._output(msg.format(coluna+1),'[m]retbase')
            self.base.remove(coluna)
            self.deletar.append(linha)

    def solucao(self):
        (M,N) = self.tableau.shape
        (m,n) = self.A.shape
        x = np.zeros(n)
        for i in range(M-1):
            x[self.base[i]] = self.tableau[i,-1]
        return x

    def _output(self,thing,source):
        if source=='definirproblema':
            if 'screen' in self.output:
                pass

            if 'latex' in self.output:
                texto = '\\begin{eqnarray*} \\min & '+ bmatrix(self.c)  +'x \\\\'
                texto = texto + '\\text{s. a.} &' + bmatrix(self.A) + 'x = '
                bempe = bmatrix(self.b.reshape((len(self.b),1)))
                texto = texto + bempe + '\end{eqnarray*}\n'
                self.latex.write(texto)
            if 'file' in self.output:
                self.file.write('A = \n'+str(self.A)+'\n')
                self.file.write('b = \n'+str(self.b)+'\n')
                self.file.write('c = \n'+str(self.c)+'\n')
        if 'minimal' in self.output:
            if 'screen' in self.output and '[m]' in source:
                print(thing)
            if 'latex' in self.output  and '[m]' in source:
                if type(thing) == str:
                    self.latex.write('\\par ' + thing + '\n')
                if type(thing) == np.ndarray:
                    matriz = bmatrix(thing)
                    self.latex.write('\\['+ matriz +'\\]\n')
            if 'file' in self.output  and '[m]' in source:
                self.file.write(str(thing)+'\n')
        else:
            if 'screen' in self.output:
                print(thing)
            if 'latex' in self.output:
                if type(thing) == str:
                    self.latex.write('\\par ' + thing + '\n')
                if type(thing) == np.ndarray:
                    matriz = bmatrix(thing)
                    self.latex.write('\\['+ matriz +'\\]\n')
            if 'file' in self.output:
                self.file.write(str(thing)+'\n')

    def preparar_output(self,mode):
        if mode=='abrir':
            if 'latex' in self.output:
                preamble = """\\documentclass[a4paper,12pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[brazil]{babel}
\\usepackage{amsmath,amssymb,amsfonts}
\\begin{document}
"""

                start = self.output.index('latex=') + len('latex=')
                end = self.output.index('.tex') + len('.tex')
                filename = self.output[start:end]
                self.latex = open(filename,'w',encoding='utf-8')
                self.latex.write(preamble)
            if 'file' in self.output:
                start = self.output.index('file=') + len('file=')
                end = self.output.index('.txt') + len('.txt')
                filename = self.output[start:end]
                self.file = open(filename,'w',encoding='utf-8')
        if mode=='fechar':
            if 'latex' in self.output:
                self.latex.write('\\end{document}')
                self.latex.close()
            if 'file' in self.output:
                self.file.close()



# Fim da Classe Simplex

def basecanonica(j,n):
    ej = np.zeros(n)
    ej[j] = 1
    return ej

def bmatrix(a):
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)


class LinhasLD(Exception):
    def __init__(self, linhas):
        self.linhas = linhas
        self.word = 'linhas'
        self.texto = ''
        self.txt()
    def txt(self):
        K = len(self.linhas)
        lnhs =  ''
        for i in range(K):
            if i < K-1:
                lnhs =  lnhs + '{}, '
            else:
                lnhs = lnhs + 'e {} '
        self.texto = 'As '+self.word+' ' + lnhs + 'sao linearmente dependentes.'
        self.texto = self.texto.format(*self.linhas)
    def __str__(self):
        return self.texto


class ColunasLD(LinhasLD):
    def __init__(self,colunas):
        self.linhas = colunas
        self.word = 'colunas'
        self.texto = ''
        self.txt()


class SemSolucoesViaveis(Exception):
    def __str__(self):
        return "O conjunto viável é vazio."


class ProblemaIlimitado(Exception):
    def __str__(self):
        return "Solução Ilimitada: O -gradiente aponta em uma direção ilimitada do conjunto."
