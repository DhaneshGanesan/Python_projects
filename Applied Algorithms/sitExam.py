# Simulates taking the exam; it computes the worst possible grade when
# using the protocol implemented in examCheatingCode

import examCheatingCode


def sit_exams(n_questions=20, n_honks=10):
    exam = [0] * n_questions
    total_mark = 0
    while True:
        # encode exam solutions in 10 bit
        code = examCheatingCode.compute_and_send_code(exam)
        if len(code) != n_honks or any(map(lambda b: b != 0 and b != 1, code)):
            raise Exception("Illegal code!")
        # compute answers to the exam from the 10 bit
        answers = examCheatingCode.enter_solution_based_on_code(code)
        if len(answers) != n_questions or any(map(lambda b: b != 0 and b != 1, answers)):
            raise Exception("Illegal decoded exam answers!")
        mark = 0
        for i in range(n_questions):
            if answers[i] == exam[i]:
                mark += 1
        total_mark += mark
        if exam == [1] * n_questions:
            return total_mark / 2 ** 20
        else:
            increment(exam)


def increment(bits):
    n = len(bits)
    for i in reversed(range(n)):
        if bits[i] == 0:
            bits[i] = 1
            return
        else:
            bits[i] = 0


my_avg_mark = sit_exams()
print(my_avg_mark)
