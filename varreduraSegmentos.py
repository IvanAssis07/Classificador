from pytrees import AVLTree
import sys

def segColineares(ponto1, ponto2, ponto3):
    if ((ponto2.x) <= max(ponto1.x, ponto3.x) and (ponto2.x) >= min(ponto1.x, ponto3.x) and
            (ponto2.y) <= max (ponto1.y, ponto3.y) and (ponto2.y) >= min(ponto1.y, ponto3.y)):
        return True
    return False

# 0 -> colinear, 1 -> horário, 2 -> anti-horário
def direcaoSeg(ponto1, ponto2, ponto3):
    produtoVetorial = ((ponto2.x - ponto1.x) * (ponto3.y - ponto1.y) - (ponto3.x - ponto1.x) * (ponto2.y - ponto1.y))

    if (produtoVetorial == 0):
        return 0
    elif (produtoVetorial > 0):
        return 1
    else:
        return 2

def intersecaoSeg(seg1,seg2):
    # Condição para impedir a checagem de interseção de segmentos do mesmo polígono
    if seg1.poligonoID == seg2.poligonoID:
        return False

    ponto1 = seg1.pontoEsquerdo
    ponto2 = seg1.pontoDireito
    ponto3 = seg2.pontoEsquerdo
    ponto4 = seg2.pontoDireito

    d1 = direcaoSeg(ponto3, ponto4, ponto1)
    d2 = direcaoSeg(ponto3, ponto4, ponto2)
    d3 = direcaoSeg(ponto1, ponto2, ponto3)
    d4 = direcaoSeg(ponto1, ponto2, ponto4)

    if ((d1 != d2 and d3 != d4)):
        return True

    # Caso onde um segmento tem um de seus pontos finais no outro
    if (d1 == 0 and segColineares(ponto3, ponto1, ponto4)):
        return True

    if (d2 == 0 and segColineares(ponto3, ponto2, ponto4)):
        return True

    if (d3 == 0 and segColineares(ponto1, ponto3, ponto2)):
        return True

    if (d4 == 0 and segColineares(ponto1, ponto4, ponto2)):
        return True

    return False
    
# class Ponto:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __repr__(self): 
#         return "(% s, % s)" % (self.x, self.y)
# Primeiramente definimos a classe Ponto
class Ponto:
  def __init__(self,x,y):
    self.x = x
    self.y = y

  def __sub__(self,b):
    x = self.x - b.x
    y = self.y - b.y
    return Ponto(x,y)

  def __repr__(self): 
    return "(% s, % s)" % (self.x, self.y)

  # Aqui sobrecarregamos o operador de < para funcionar para pontos
  def __lt__(self, other):
    if self.y < other.y:
      return True

    if self.y == other.y and self.x < other.x:
      return True
      
    return False

class Segmento:
    def __init__(self, pontoEsquerdo, pontoDireito, poligonoID):
        self.pontoEsquerdo = pontoEsquerdo
        self.pontoDireito = pontoDireito
        a = (pontoDireito.y - pontoEsquerdo.y)/(pontoDireito.x - pontoEsquerdo.x)
        b = (pontoEsquerdo.y - a * pontoEsquerdo.x)
        # a -> coeficiente angular, b -> coeficiente linear
        self.chave = (a, b)
        # 0 representa um polígono, 1 representa o outro
        self.poligonoID = poligonoID 

    def __repr__(self): 
        return "(% s, % s, % s)" % (self.pontoEsquerdo, self.pontoDireito,self.chave)

def construtorSeg(pontos, poligonoID):
  seg = []
  for i in range(len(pontos)):
    if i+1 < len(pontos):  
        if pontos[i].x ==  pontos[i+1].x:
            pontos[i].x += 0.0001
            seg.append(Segmento(pontos[i], pontos[i+1], poligonoID))
        else:
            seg.append(Segmento(pontos[i], pontos[i+1], poligonoID))
    else:
        if pontos[i].x ==  pontos[0].x:
            pontos[i].x += 0.0001
            seg.append(Segmento(pontos[i], pontos[0], poligonoID))
        else:
            seg.append(Segmento(pontos[i], pontos[0], poligonoID))
  return seg

class Evento:
    def __init__(self, x, y, pontoInicial, index):
        self.x = x 
        self.y = y 
        self.pontoInicial = pontoInicial
        self.index = index

    def __lt__(self, other):
        if (self.x < other.x):
            return True

        if (self.x == other.x):
            if self.pontoInicial == True and other.pontoInicial == False:
                return True
            if self.pontoInicial == False and other.pontoInicial == True:
                return False
            else:
                return self.y < other.y

        if (self.x > other.x): 
            return False
        else:
            return self.y > other.y

    def __repr__(self): 
        return "(% s, % s)" % (self.x, self.y)

class Nodo:
    def __init__(self, x, segmento):
        self.x = x
        self.segmento = segmento

    def __lt__(self, other):
        return ((self.segmento.chave[0] * self.x) + self.segmento.chave[1]) < ((other.segmento.chave[0] * other.x) + other.segmento.chave[1])
    
    def __le__(self, other):
        return ((self.segmento.chave[0] * self.x) + self.segmento.chave[1]) <= ((other.segmento.chave[0] * other.x) + other.segmento.chave[1])

    def __repr__(self):
        return "(% s, % s)" % (self.segmento.chave[0], self.segmento.chave[1])
    
    def __eq__(self,other):
        if self.segmento.chave[0] == other.segmento.chave[0] and self.segmento.chave[1] == other.segmento.chave[1]:
            return True

    def __gt__(self, other):
       return ((self.segmento.chave[0] * self.x) + self.segmento.chave[1]) > ((other.segmento.chave[0] * other.x) + other.segmento.chave[1])

    def __ge__(self, other):
        return ((self.segmento.chave[0] * self.x) + self.segmento.chave[1]) >= ((other.segmento.chave[0] * other.x) + other.segmento.chave[1])

    def __getitem__(self,i):
        return self.segmento

def acima(arvore, nodo): 
    nodo = arvore.search(nodo)
    if (nodo == None):
        return None
    nodoPai = nodo.parent

    if(nodo.right != None):
        return nodo.right
    else:
        if(nodo.parent != None and nodo.parent.left == nodo):
            return nodo.parent

        while(nodoPai != None):
            if (nodoPai.val > nodo.val):
                return nodoPai
            else:
                nodoPai = nodoPai.parent
        return None

def abaixo(arvore, nodo):
    nodo = arvore.search(nodo)
    if (nodo == None):
        return None
    nodoPai = nodo.parent

    if(nodo == None): 
        return nodo

    if(nodo.left != None):
        return nodo.left
    else:
        if(nodo.parent != None and nodo.parent.right == nodo):
            return nodo.parent

        while(nodoPai != None):
            if (nodoPai.val < nodo.val):
                return nodoPai
            else:
                nodoPai = nodoPai.parent
        return None


def varreduraSeg(seg1, seg2):
    segmentos = seg1 + seg2
    eventos = []
    
    for n in range(len(segmentos)):
        eventos.append(Evento(segmentos[n].pontoEsquerdo.x, segmentos[n].pontoEsquerdo.y, True, n))
        eventos.append(Evento(segmentos[n].pontoDireito.x, segmentos[n].pontoDireito.y, False, n))
    
    eventos.sort()
    arvore = AVLTree()

    for p in eventos:
        if p.pontoInicial == True:
            arvore.insert(Nodo(p.x, segmentos[p.index]))
            nodoAcima = acima(arvore, Nodo(p.x, segmentos[p.index]))
            nodoAbaixo = abaixo(arvore, Nodo(p.x, segmentos[p.index]))

            if((nodoAcima != None and intersecaoSeg(segmentos[p.index], nodoAcima.val.segmento))
                or nodoAbaixo != None and intersecaoSeg(segmentos[p.index], nodoAbaixo.val.segmento)):
                return True

        if p.pontoInicial == False:
            nodoAcima = acima(arvore, Nodo(p.x, segmentos[p.index]))
            nodoAbaixo = abaixo(arvore, Nodo(p.x, segmentos[p.index]))

            if(nodoAcima != None and nodoAbaixo != None and intersecaoSeg(nodoAcima.val.segmento, nodoAbaixo.val.segmento)):
                return True

            arvore.delete(Nodo(p.x, segmentos[p.index]))

    return False

def polDentroPol(ponto, segmentos):
    intersecoes = 0

    for n in range(len(segmentos)):
        if (intersecaoSeg(Segmento(ponto, Ponto(sys.maxsize, ponto.y), 1), segmentos[n])):
            intersecoes += 1

    # Se tiver um número par de interseções está fora, ímpar está dentro
    if (intersecoes % 2 == 0):
        return False
    else:
        return True

def envDentroEnv(seg1, seg2):
    intersecoes = 0

    if (seg1[0].pontoEsquerdo > seg2[0].pontoEsquerdo):
        # Traçar uma reta de um ponto ao infinito e contar interseções
        for n in range(len(seg2)):
            if (intersecaoSeg(Segmento(seg1[0].pontoEsquerdo, Ponto(sys.maxsize, seg1[0].pontoEsquerdo.y ), 1), seg2[n])):
                intersecoes += 1
    else:
        # Traçar uma reta de um ponto ao infinito e contar interseções
        for n in range(len(seg1)):
            if (intersecaoSeg(Segmento(seg2[0].pontoEsquerdo, Ponto(sys.maxsize, seg2[0].pontoEsquerdo.y ), 1), seg1[n])):
                intersecoes += 1

    # Se tiver um número par de interseções está fora, ímpar está dentro
    if (intersecoes % 2 == 0):
        return False
    else:
        return True
        
# ### teste 1 - True
# A = Ponto(1, 1)
# B = Ponto(4, 4)
# C = Ponto(3, 2)

# D = Ponto(3, 2.5)
# E = Ponto(6, 3)
# F = Ponto(6.5, 0.5)

# segPoly1 = construtorSeg((A, B, C), 0)
# segPoly2 = construtorSeg((D, E, F), 1)
# print(varreduraSeg(segPoly1, segPoly2))
# print(polDentroPol())

# ### teste 2 - False
# A = Ponto(1, 1)
# B = Ponto(4, 4)
# C = Ponto(3, 2)

# D = Ponto(3.5, 2.5)
# E = Ponto(6, 3)
# F = Ponto(6.5, 0.5)

# segPoly1 = construtorSeg((A, B, C), 0)
# segPoly2 = construtorSeg((D, E, F), 1)
# print(varreduraSeg(segPoly1, segPoly2))

# ### teste 3 - False
# A = Ponto(1, 7)
# B = Ponto(3, 1)
# C = Ponto(10, 1)

# D = Ponto(6, 8)
# E = Ponto(7, 7)
# F = Ponto(9, 9)

# segPoly1 = construtorSeg((A, B, C), 0)
# segPoly2 = construtorSeg((D, E, F), 1)
# print(varreduraSeg(segPoly1, segPoly2))

# ### teste 4 - True
# A = Ponto(9, 7)
# B = Ponto(3, 1)
# C = Ponto(5, 8)

# D = Ponto(4, 4)
# E = Ponto(10, 3)
# F = Ponto(9, 4)

# segPoly1 = construtorSeg((A, B, C), 0)
# segPoly2 = construtorSeg((D, E, F), 1)
# print(varreduraSeg(segPoly1, segPoly2))

# ### teste 5 - True
# A = Ponto(2, 9)
# B = Ponto(10, 11)
# C = Ponto(10, 4)

# D = Ponto(4, 2)
# E = Ponto(9, 6)
# # F = Ponto(10,3)

# segPoly1 = construtorSeg((A, B, C), 0)
# segPoly2 = construtorSeg((D, E), 1)
# print(varreduraSeg(segPoly1, segPoly2))

# ### teste 6 - True
# A = Ponto(4, 6)
# B = Ponto(8, 5)
# C = Ponto(1, 4)

# D = Ponto(5, 3)
# E = Ponto(9, 5)
# F = Ponto(9, 7)

# segPoly1 = construtorSeg((A, B, C), 0)
# segPoly2 = construtorSeg((D, E, F), 1)
# print(varreduraSeg(segPoly1, segPoly2))

# ### teste 7 - True
# A = Ponto(3, 10)
# B = Ponto(6, 10)
# C = Ponto(8, 5)

# D = Ponto(8, 2)
# E = Ponto(10, 5)
# F = Ponto(3, 6)

# segPoly1 = construtorSeg((A, B, C), 0)
# segPoly2 = construtorSeg((D, E, F), 1)
# print(varreduraSeg(segPoly1, segPoly2))

# ### teste 8 - True
# A = Ponto(4, 7)
# B = Ponto(7, 1)
# C = Ponto(5, 4)

# D = Ponto(10, 7)
# F = Ponto(2, 7)

# segPoly1 = construtorSeg((A, B, C), 0)
# segPoly2 = construtorSeg((D, F), 1)
# print(varreduraSeg(segPoly1, segPoly2))

