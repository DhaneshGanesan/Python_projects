# Compute the honking protocol for the exam cheaters

def compute_and_send_code(exam):
    code = [0] * 10
    # Dont change anything above this line
    # ==========================

    # TODO Add your solution here.
    for i in range(1,3):
        while(i<2):
            if(exam[i-1]+exam[i]+exam[i+1]>=2):
                i+=1
                code[0] = 1
            else:
                i+=1
                code[0] = 0
            break 
    for i in range(4,6):
        while(i<5):
            if(exam[i-1]+exam[i]+exam[i+1]>=2):
                i+=1
                code[1] = 1
            else:
                i+=1
                code[1] = 0
            break
    for i in range(7,9):
        while(i<8):
            if(exam[i-1]+exam[i]+exam[i+1]>=2):
                i+=1
                code[2] = 1
            else:
                i+=1
                code[2] = 0
            break
    for i in range(10,12):
        while(i<11):
            if(exam[i-1]+exam[i]+exam[i+1]>=2):
                i+=1
                code[3] = 1
            else:
                i+=1
                code[3] = 0
            break
    for i in range(13,15):
        while(i<14):
            if(exam[i-1]+exam[i]+exam[i+1]>=2):
                i+=1
                code[4] = 1
            else:
                i+=1
                code[4] = 0
            break
    # ==========================
    # Dont change anything below this line
    return code


def enter_solution_based_on_code(code):
    answer = [0] * 20
    # Dont change anything above this line
    # ==========================

    # TODO Add your solution here.

    # ==========================
    # Dont change anything below this line
    return answer



