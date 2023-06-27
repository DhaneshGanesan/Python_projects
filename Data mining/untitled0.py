def compute_and_send_code(exam):
    code=[0] * 10 
    # Dont change anything above this line
    x=range(1,3)
    for i in x:
        while(i<2):
            if(exam[i-1]+exam[i]+exam[i+1]>=2):
                code[0]=1
                i=i+1 
            else:
                 code[0]=0
                 i=i+1
            break 
    x=range(4,6)
    for i in x:
        while(i<5):
            if(exam[i]+exam[i-1]+exam[i+1]>=2):
                code[1]=1
                i=i+1 
            else:
                code[1]=0
                i=i+1
            break
    x=range(7,9)
    for i in x:
        while(i<8):
            if(exam[i]+exam[i-1]+exam[i+1]>=2):
                code[2]=1
                i=i+1 
            else:
                code[2]=0
                i=i+1
            break
    x=range(10,12)
    for i in x:
        while(i<11):
            if(exam[i]+exam[i-1]+exam[i+1]>=2):
                code[3]=1
                i=i+1 
            else:
                code[3]=0
                i=i+1
            break
    x=range(13,15)
    for i in x:
        while(i<14):
            if(exam[i]+exam[i-1]+exam[i+1]>=2):
                code[4]=1
                i=i+1 
            else:
                code[4]=0
                i=i+1
            break

break
    range(7,9)
    for i in x: 
        while(i<8):
        if(exam[i]+exam[i-1]+exam[i+1]>=2):
    code[2]=1
    i=i+1 
    else:
    code[2]=0
    i=i+1
    break
     x=(10,12)
     for i in x:
         while(i&lt;11):
             if(exam[i]+exam[i-1]+exam[i+1]>=2):
    code[3]=1
    i=i+1
    break 