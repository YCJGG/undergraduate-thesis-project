full = open('all.list','r')
#partial = open('partial_feature.list','r')
test = open('test.list','a')
train = open('train.list','a')
full = list(full)
#partial = list(partial)
k = 0
flag = 0
for i in range(len(full)):
    f = full[i].strip().split()
    #p = partial[i].strip().split()
    label = int(f[1])
    
    if flag >= 10:
        k+=1
        flag = 0
    if label == k and flag < 10:
        test.write(f[0]+' '+f[1]+'\n')
        flag+=1
    else:
        train.write(f[0]+' '+f[1]+'\n')

