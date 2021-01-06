"""
Defines two dictionaries for converting 
between text and integer sequences.
"""

char_map_str = """

' 0
<SPACE> 1
h 2 
æ 3 
d 4 
q 5 
l 6 
e 7 
ɪ 8 
ˌ 9 
b 10 
i 11 
ˈ 12 
f 13 
ɔ 14 
r 15 
ə 16 
p 17 
ɛ 18 
v 19 
t 20  
n 21
z 22
a 23
ʊ 24
k 25
s 26
j 27
m 28
o 29 
ð 30 
u 31 
w 32 
ɑ 33 
ŋ 34 
g 35 
θ 36 
ʧ 37 
c 38 
* 39 
ʃ 40 
ʤ 41 
ʒ 42 
y 43 
' 44 
x 45 
 

"""

char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)+1] = ch
index_map[2] = ' '



# char_map_str = """
# ' 0
# <SPACE> 1
# ʌ 2
# ɑ: 3	 
# æ 4	
# e 5	
# ə 6	
# ɜ:ʳ 7
# ɪ 8	 
# i: 9
# ɒ 10
# ɔ: 11
# ʊ 12	 
# u: 13
# aɪ 14	 
# aʊ 15	 
# eɪ 16	 
# oʊ 17
# ɔɪ 18	 
# eəʳ 19
# ɪəʳ 20
# ʊəʳ 21
# b 22	 
# d 23	 
# f 24	 
# g 25
# h 26	 
# j 27	 
# k 28	 
# l 29	 
# m 30	 
# n 31	 
# ŋ 32
# p 33
# r 34
# s 35	 
# ʃ 36	 
# t 37
# tʃ 38	 
# θ 39
# ð 40
# v 41	 
# w 42	 
# z 43	 
# ʒ 44	 
# dʒ 45
# ˌ 46

# """

