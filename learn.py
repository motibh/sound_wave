import re
import datetime
d= {'a':2,'aa':22,'aaa':222,'aaaa':2222,'ab':2333333333}
l=[kee1 for kee1 in d if d[kee1]>666]
print(l)

text = 'a random string.'
pat = re.compile('[Abc]')
res = pat.search(text)
print(res)

today = datetime.date.today()
print(today)
birt = datetime.date(1979,5,29)
print(birt)
long = today- birt
print(long)

list1 = [1,2,3,4,5]
list2 = ['one','two','three']
zipped = list(zip(list1,list2))
print(list(zip(*zipped)))

for (l1,l2) in zipped:
    print(l1)
    print(l2)

