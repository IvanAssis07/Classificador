from matplotlib import test
from pytrees import AVLTree

def onSegment(point1, point2, point3):
    if ((point2.x) <= max(point1.x, point3.x) and (point2.x) >= min(point1.x, point3.x) and
            (point2.y) <= max (point1.y, point3.y) and (point2.y) >= min(point1.y, point3.y)):
        return True
    return False

# 0 -> colinear, 1 -> clockwise, 2 -> anti-clockwise
def direction(point1, point2, point3):
    crossProduct = ((point2.x - point1.x) * (point3.y - point1.y) - (point3.x - point1.x) * (point2.y - point1.y))

    if (crossProduct == 0):
        return 0
    elif (crossProduct > 0):
        return 1
    else:
        return 2

def segmentsIntercept(seg1,seg2):

    if seg1.polygon == seg2.polygon:
        return False

    point1 = seg1.leftPoint
    point2 = seg1.rightPoint
    point3 = seg2.leftPoint
    point4 = seg2.rightPoint

    d1 = direction(point3, point4, point1)
    d2 = direction(point3, point4, point2)
    d3 = direction(point1, point2, point3)
    d4 = direction(point1, point2, point4)

    if ((d1 != d2 and d3 != d4)):
        return True

    # Cases when one segment has one endpoint in another one
    if (d1 == 0 and onSegment(point3, point1, point4)):
        return True

    if (d2 == 0 and onSegment(point3, point2, point4)):
        return True

    if (d3 == 0 and onSegment(point1, point3, point2)):
        return True

    if (d4 == 0 and onSegment(point1, point4, point2)):
        return True

    return False
    
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self): 
        return "(% s, % s)" % (self.x, self.y)

class Segment:
    def __init__(self, left, right, polygon):
        self.leftPoint = left
        self.rightPoint = right
        a = (right.y - left.y)/(right.x - left.x)
        b = (left.y - a * left.x)
        self.key = (a, b)
        # 0 for one polygon, 1 for the other
        self.polygon = polygon 

    def __repr__(self): 
        return "(% s, % s, % s)" % (self.leftPoint, self.rightPoint,self.key)

def buildSegments(env, polygon):
  seg = []
  for i in range(len(env)):
    if i+i < len(env):  
        seg.append(Segment(env[i],env[i+1],polygon))
    else:
        seg.append(Segment(env[i],env[0],polygon))
  return seg

class Event:
    def __init__(self, x, y, startPoint, index):
        self.x = x 
        self.y = y 
        self.startPoint = startPoint
        self.index = index

    def __lt__(self, other):
        if (self.x < other.x):
            return True
        if (self.x == other.x):
            if self.startPoint == True and other.startPoint == False:
                return True
            if self.startPoint == False and other.startPoint == True:
                return False
            else:
                return self.y < other.y
            # if(self.y == other.y):
            #     return False
            # else:
            #     return self.y < other.y
        if (self.x > other.x): 
            return False
        else:
            return self.y > other.y

    def __repr__(self): 
        return "(% s, % s)" % (self.x, self.y)

class Node1:
    def __init__(self, x, segment):
        self.x = x
        self.segment = segment

    def __lt__(self, other):
        return ((self.segment.key[0] * self.x) + self.segment.key[1]) < ((other.segment.key[0] * other.x) + other.segment.key[1])
    
    def __le__(self, other):
        return ((self.segment.key[0] * self.x) + self.segment.key[1]) <= ((other.segment.key[0] * other.x) + other.segment.key[1])

    def __repr__(self):
        return "(% s, % s)" % (self.segment.key[0], self.segment.key[1])
    
    def __eq__(self,other):
        if self.segment.key[0] == other.segment.key[0] and self.segment.key[1] == other.segment.key[1]:
            return True

    def __gt__(self, other):
       return ((self.segment.key[0] * self.x) + self.segment.key[1]) > ((other.segment.key[0] * other.x) + other.segment.key[1])

    def __ge__(self, other):
        return ((self.segment.key[0] * self.x) + self.segment.key[1]) >= ((other.segment.key[0] * other.x) + other.segment.key[1])

    def __getitem__(self,i):
        return self.segment

# def above(tree, node):   
#     node = tree.search(node)
#     if(node == None): 
#         return node

#     if(node.right != None):
#         return node.right
#     else:
#         if(node.parent != None and node.parent.left == node):
#             return node.parent
#         else:
#             return None

def above(tree, node): 
    node = tree.search(node)
    if (node == None):
        return None
    parentNode = node.parent

    if(node.right != None):
        return node.right
    else:
        if(node.parent != None and node.parent.left == node):
            return node.parent

        while(parentNode != None):
            if (parentNode.val > node.val):
                return parentNode
            else:
                parentNode = parentNode.parent
        return None

            
        # primeiro pai que é maior que o nó que fornecemos
# def below(tree, node):
#     node = tree.search(node)
#     if(node == None): 
#         return node

#     if(node.left != None):
#         return node.left
#     else:
#         if(node.parent != None and node.parent.right == node):
#             return node.parent
#         else:
#             return None
def below(tree, node):
    node = tree.search(node)
    if (node == None):
        return None
    parentNode = node.parent

    if(node == None): 
        return node

    if(node.left != None):
        return node.left
    else:
        if(node.parent != None and node.parent.right == node):
            return node.parent
        else:
            while(parentNode != None):
                if (parentNode.val < node.val):
                    return parentNode
                else:
                    parentNode = parentNode.parent
            return None


def sweepSegments(seg1,seg2):
    Segments = seg1+seg2
    events = []
    for n in range(len(Segments)):
        events.append(Event(Segments[n].leftPoint.x, Segments[n].leftPoint.y, True, n))
        events.append(Event(Segments[n].rightPoint.x, Segments[n].rightPoint.y, False, n))
    
    events.sort()
    # print(events)
    tree = AVLTree()

    for p in events:
        # tree.visulize()
        if p.startPoint == True:
            tree.insert(Node1(p.x, Segments[p.index]))
            # tree.visulize()
            nodeAbove = above(tree, Node1(p.x, Segments[p.index]))
            nodeBelow = below(tree, Node1(p.x, Segments[p.index]))
            if((nodeAbove != None and segmentsIntercept(Segments[p.index], nodeAbove.val.segment))
                or nodeBelow != None and segmentsIntercept(Segments[p.index], nodeBelow.val.segment)):
                # count += 1
                return True
        if p.startPoint == False:
            nodeAbove = above(tree, Node1(p.x, Segments[p.index]))
            nodeBelow = below(tree, Node1(p.x, Segments[p.index]))
            if(nodeAbove != None and nodeBelow != None and segmentsIntercept(nodeAbove.val.segment, nodeBelow.val.segment)):
                return True
            tree.delete(Node1(p.x, Segments[p.index]))
            # tree.visulize()
    return False

# A = Point(2, 9)
# B = Point(10, 11)
# C = Point(10, 4)

# segPoly1 = buildSegments((A, B, C), 0)

# D = Point(4, 2)
# E = Point(9, 6)
# # F = Point(9, 4)

# segPoly2 = buildSegments((D, E), 1)

A = Point(2, 10)
B = Point(9, 7)
C = Point(10, 4)

segPoly1 = buildSegments((A, B, C), 0)

D = Point(5, 9)
E = Point(7, 7)
# F = Point(9, 4)

segPoly2 = buildSegments((D, E), 1)

#print(sweepSegments(segPoly1, segPoly2))

#test1 = [Segment(Point(4, 6), Point(8, 5),1), Segment(Point(1, 4), Point(8, 5),1), Segment(Point(5, 3), Point(9, 5),1), Segment(Point(5, 3), Point(9, 7),1)]
# test1 = []
print(sweepSegments(segPoly1,segPoly2))

# a = []
# for n in range(a):
#         Segm.append(Segment(Point(a[n]), Point(a[n]));
# a = [Segment(Point(1,1), Point(2,2)), Segment(Point(1,3), Point(2,4)),Segment(Point(1,2), Point(2, 3)),Segment(Point(0, -1), Point (1, 0)),Segment(Point(0,-1/2), Point(1/2,0))]
# sweepSegments(a[0])

# tree = AVLTree()
# tree.insert(50)
# tree.insert(17)
# tree.insert(12)
# tree.insert(23)
# tree.insert(14)
# tree.insert(9)
# tree.insert(72)
# tree.insert(54)
# tree.insert(76)
# tree.insert(67)
# tree.insert(56)
# tree.visulize()
# print(below(tree, 23))

# node = tree.search(Node1(1,a[1]))
# print(node)
# test = below(tree, Node1(1,a[3]))
# print(test)
# test = below(tree, Node1(1,a[4]))
# print(test)
# test = below(tree, Node1(1,a[0]))
# print(test)
# print(test.val.segment.leftPoint)
# print(test.val.segment.rightPoint)

# tree.delete(Node(1,a[4]))
# tree.visulize()
# print(tree.search())

# sweepSegments(a)



